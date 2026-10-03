"""
Datos sintéticos para probar el pipeline sin conexión.
Simula ~230 fondos a partir de un modelo de factores: gestores indexados casi idénticos,
fondos activos, sectoriales, renta fija, mixtos, monetarios, oro, alternativos con NAV
suavizado, fondos en USD, fondos con poco historial y algún apalancado.
NO son fondos reales: sirven solo para comprobar que todo funciona.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from datos import INFO_VACIA, DatosFondo

SEMILLA = 7
DIAS = 252


@lru_cache(maxsize=1)
def _mundo():
    rng = np.random.default_rng(SEMILLA)
    fechas = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=int(DIAS * 5.2))
    n = len(fechas)

    def f(mu, vol):
        return rng.normal(mu / DIAS, vol / np.sqrt(DIAS), n)

    mundo = f(0.07, 0.15)
    fx_usd = f(0.0, 0.07)                         # EURUSD (USD por EUR), log-variación
    fac = {
        "RV Desarrollados": mundo,
        "RV Emergentes": 0.8 * mundo + f(0.01, 0.12),
        "RF Gobierno EUR": -0.05 * mundo + f(0.02, 0.05),
        "Oro": 0.1 * mundo + f(0.05, 0.14),
    }
    fac["RF Corporativa EUR"] = 0.7 * fac["RF Gobierno EUR"] + 0.08 * mundo + f(0.008, 0.02)
    extra = {
        "eeuu": f(0.01, 0.05), "europa": f(-0.005, 0.05), "japon": f(0.0, 0.08),
        "tech": f(0.03, 0.12), "small": f(0.0, 0.07), "hy": f(0.02, 0.03),
        "corto": f(0.025, 0.012), "alt": f(0.03, 0.04),
    }
    return rng, fechas, fac, extra, fx_usd


@lru_cache(maxsize=1)
def _fondos():
    rng, fechas, F, E, fx_usd = _mundo()
    n = len(fechas)
    W, EM, GOV, CORP, ORO = (F[k] for k in ("RV Desarrollados", "RV Emergentes", "RF Gobierno EUR",
                                             "RF Corporativa EUR", "Oro"))

    def ruido(te, alfa=0.0):
        return rng.normal(alfa / DIAS, te / np.sqrt(DIAS), n)

    fondos = []   # (nombre, retornos diarios EUR, moneda, inicio_indice, suavizado)

    def add(nombre, r, moneda="EUR", inicio=0, suav=0.0):
        fondos.append((nombre, r, moneda, inicio, suav))

    for i in range(12):
        add(f"Indexado MSCI World {i + 1}", W + ruido(0.004, -0.002))
    for i in range(40):
        add(f"RV Global Activo {i + 1}", 0.95 * W + ruido(0.04, rng.normal(0, 0.015)))
    for i in range(25):
        add(f"RV EE. UU. {i + 1}", 1.05 * W + E["eeuu"] + ruido(0.03, rng.normal(0, 0.01)))
    for i in range(20):
        add(f"RV Europa {i + 1}", 0.9 * W + E["europa"] + ruido(0.035, rng.normal(0, 0.01)))
    for i in range(8):
        add(f"RV Japón {i + 1}", 0.8 * W + E["japon"] + ruido(0.04))
    for i in range(15):
        add(f"RV Emergentes {i + 1}", EM + ruido(0.04, rng.normal(0, 0.01)))
    for i in range(12):
        add(f"RV Tecnología {i + 1}", 1.2 * W + E["tech"] + ruido(0.05, rng.normal(0.01, 0.01)))
    for i in range(6):
        add(f"RV Small Caps {i + 1}", 1.1 * W + E["small"] + ruido(0.04))
    for i in range(12):
        add(f"RF Gobierno EUR {i + 1}", GOV + ruido(0.006, -0.002))
    for i in range(12):
        add(f"RF Corporativa EUR {i + 1}", CORP + ruido(0.008, rng.normal(0, 0.003)))
    for i in range(8):
        add(f"RF High Yield {i + 1}", 0.5 * CORP + 0.25 * W + E["hy"] + ruido(0.015))
    for i in range(10):
        add(f"RF Corto Plazo {i + 1}", 0.3 * CORP + E["corto"] + ruido(0.004))
    for i in range(15):
        pct = rng.uniform(0.25, 0.7)
        add(f"Mixto {int(pct * 100)}% RV {i + 1}", pct * W + (1 - pct) * CORP + ruido(0.015))
    for i in range(6):
        add(f"Monetario EUR {i + 1}", np.full(n, 0.02 / DIAS) + ruido(0.002))
    for i in range(5):
        add(f"Oro y metales {i + 1}", 0.9 * ORO + 0.2 * W + ruido(0.05))
    for i in range(6):
        add(f"Retorno Absoluto {i + 1}", 0.1 * W + E["alt"] + ruido(0.02), suav=0.6 if i < 3 else 0.0)
    for i in range(4):
        add(f"RV EE. UU. clase USD {i + 1}", 1.05 * W + E["eeuu"] + ruido(0.03) - fx_usd, moneda="USD")
    for i in range(10):
        add(f"RV Temática reciente {i + 1}", 1.1 * W + ruido(0.08), inicio=int(n * 0.65))
    for i in range(2):
        add(f"Apalancado 3x {i + 1}", 3 * W + ruido(0.03))

    salida = {}
    for k, (nombre, r, moneda, inicio, suav) in enumerate(fondos):
        r = r.copy()
        if suav > 0:                               # NAV suavizado (activo poco líquido)
            for t in range(1, n):
                r[t] = (1 - suav) * r[t] + suav * r[t - 1]
        if moneda == "USD":                         # convierte a precio en USD
            r = r + fx_usd
        precio = 100 * np.exp(np.cumsum(r))
        s = pd.Series(precio, index=fechas).iloc[inicio:]
        isin = f"XX{k:010d}"
        salida[isin] = (nombre, s, moneda)
    return salida


def universo_demo() -> pd.DataFrame:
    fondos = _fondos()
    return pd.DataFrame({"isin": list(fondos), "nombre": [v[0] for v in fondos.values()],
                         "ticker": None})


class FuenteDemo:
    nombre = "demo"

    def descargar(self, isin, ticker, inicio, fin) -> DatosFondo:
        fondos = _fondos()
        if isin not in fondos:
            return DatosFondo(isin, pd.Series(dtype=float), dict(INFO_VACIA), "no existe (demo)")
        nombre, s, moneda = fondos[isin]
        info = dict(INFO_VACIA, nombre=nombre, moneda=moneda, id_fuente="demo")
        return DatosFondo(isin, s, info)

    def serie_ticker(self, ticker, inicio, fin) -> pd.Series:
        _, fechas, F, _, fx_usd = _mundo()
        if ticker == "EURUSD=X":
            return pd.Series(1.10 * np.exp(np.cumsum(fx_usd)), index=fechas)
        import tomllib
        from pathlib import Path
        with open(Path(__file__).parent / "config.toml", "rb") as f:
            factores = tomllib.load(f)["estilo"]["factores"]
        inverso = {v: k for k, v in factores.items()}
        nombre = inverso.get(ticker)
        if nombre not in F:
            return pd.Series(dtype=float)
        return pd.Series(100 * np.exp(np.cumsum(F[nombre])), index=fechas)
