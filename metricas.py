"""
Métricas de rentabilidad/riesgo y análisis de estilo.

Referencias:
  - Lo (2002), "The Statistics of Sharpe Ratios": ajuste por autocorrelación.
  - Getmansky, Lo y Makarov (2004): rentabilidades "suavizadas" en activos poco líquidos.
  - Bailey y López de Prado (2012), "The Sharpe Ratio Efficient Frontier": Probabilistic
    Sharpe Ratio (PSR), que tiene en cuenta longitud de la muestra, asimetría y curtosis.
  - Sharpe (1992), "Asset Allocation: Management Style and Performance Measurement":
    análisis de estilo por rentabilidades.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

PERIODOS = {"W-FRI": 52, "W": 52, "ME": 12, "M": 12, "D": 252, "B": 252}


def periodos_por_anio(frecuencia: str) -> int:
    return PERIODOS.get(frecuencia, 52)


def a_rendimientos(precios: pd.DataFrame, frecuencia: str = "W-FRI") -> pd.DataFrame:
    """Precios diarios -> rendimientos simples a la frecuencia indicada.
    No rellena huecos largos: si un fondo no tiene dato en un periodo, queda NaN."""
    p = precios.resample(frecuencia).last()
    r = p.pct_change(fill_method=None).iloc[1:]
    # descarta el último periodo si aún no ha terminado (semana en curso)
    if len(r) and r.index[-1] > precios.index.max():
        r = r.iloc[:-1]
    return r


def tipo_periodico(rf_anual: float, periodos: int) -> float:
    return (1 + rf_anual) ** (1 / periodos) - 1


def _rf_en(rf_periodo, index: pd.Index):
    """rf_periodo puede ser un escalar (comportamiento histórico) o una serie ya en
    unidades del periodo: la alinea al índice pedido sin dejar huecos."""
    if np.isscalar(rf_periodo) or rf_periodo is None:
        return rf_periodo
    return rf_periodo.reindex(index).ffill().bfill()


def construir_rf_periodo(estr_diario: pd.Series | None, index_objetivo: pd.DatetimeIndex,
                         periodos: int, tipo_libre_riesgo_fijo: float) -> pd.Series:
    """Serie de tipo libre de riesgo en unidades del periodo, alineada a index_objetivo.

    Si no hay €STR (fuente = "fijo" o falló la descarga), serie constante con
    tipo_libre_riesgo_fijo. Si lo hay, compone el tipo diario (ACT/365) dentro de
    cada periodo del índice objetivo; los periodos sin dato de €STR (antes de su
    histórico, huecos) se rellenan con el valor fijo convertido a periódico, para
    no dejar nunca un NaN."""
    fijo_periodico = tipo_periodico(tipo_libre_riesgo_fijo, periodos)
    if estr_diario is None or estr_diario.empty:
        return pd.Series(fijo_periodico, index=index_objetivo)

    diario = estr_diario.sort_index()
    factor_diario = (1 + diario) ** (1 / 365)
    log_factor = np.log(factor_diario)
    log_acum = log_factor.cumsum()
    # para cada fecha del índice objetivo, compón desde el cierre del periodo anterior
    bordes = index_objetivo.insert(0, index_objetivo[0] - (index_objetivo[1] - index_objetivo[0])
                                   if len(index_objetivo) > 1 else index_objetivo[0])
    log_acum_en_bordes = log_acum.reindex(bordes, method="ffill")
    compuesto = np.exp(log_acum_en_bordes.diff().iloc[1:]) - 1
    compuesto.index = index_objetivo
    compuesto = compuesto.reindex(index_objetivo)
    return compuesto.fillna(fijo_periodico)


def factor_lo(rho: float, q: int) -> float:
    """Ratio varianza real / varianza 'ingenua' al agregar q periodos con autocorrelación
    AR(1) de parámetro rho (Lo, 2002). >1 si rho > 0."""
    k = np.arange(1, q)
    return float(1 + 2 * np.sum((q - k) / q * rho ** k))


def max_drawdown(r: pd.Series) -> float:
    riqueza = (1 + r.fillna(0)).cumprod()
    return float((riqueza / riqueza.cummax() - 1).min())


def psr(sr: float, n: int, asim: float, curt: float, sr_ref: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio: P(SR verdadero > sr_ref). sr en unidades del periodo."""
    if n < 3 or not np.isfinite(sr):
        return np.nan
    denom = 1 - asim * sr + (curt - 1) / 4 * sr ** 2
    if denom <= 0:
        return np.nan
    return float(stats.norm.cdf((sr - sr_ref) * np.sqrt(n - 1) / np.sqrt(denom)))


def metricas_fondo(r: pd.Series, rf_periodo: float | pd.Series, periodos: int, umbral_autocorr: float) -> dict:
    r = r.dropna()
    n = len(r)
    if n < 10:
        return {"n_obs": n}
    rf = _rf_en(rf_periodo, r.index)
    exceso = r - rf
    media, sd = exceso.mean(), r.std(ddof=1)
    rent_anual = float((1 + r).prod() ** (periodos / n) - 1)
    vol = float(sd * np.sqrt(periodos))
    rho = float(r.autocorr(lag=1)) if n > 20 else 0.0
    rho = 0.0 if not np.isfinite(rho) else rho

    # Ajuste por autocorrelación solo si es positiva y relevante (suavizado de NAV)
    ajuste = factor_lo(rho, periodos) if rho > umbral_autocorr else 1.0
    vol_aj = vol * np.sqrt(ajuste)
    sr_periodo = media / sd if sd > 0 else np.nan
    sr_periodo_aj = sr_periodo / np.sqrt(ajuste) if sd > 0 else np.nan
    sharpe = sr_periodo * np.sqrt(periodos)
    sharpe_aj = sr_periodo_aj * np.sqrt(periodos)

    abajo = exceso[exceso < 0]
    dd = np.sqrt((abajo ** 2).sum() / n) if n else np.nan
    sortino = media / dd * np.sqrt(periodos) if dd and dd > 0 else np.nan

    asim = float(stats.skew(r, bias=False))
    curt = float(stats.kurtosis(r, fisher=False, bias=False))
    var95 = float(np.quantile(r, 0.05))
    cvar95 = float(r[r <= var95].mean())

    return {
        "n_obs": n,
        "rent_anual": rent_anual,
        "vol_anual": vol,
        "vol_ajustada": vol_aj,
        "autocorr_1": rho,
        "sharpe": sharpe,
        "sharpe_ajustado": sharpe_aj,
        "sortino": sortino,
        "psr": psr(sr_periodo_aj, n, asim, curt),
        "max_drawdown": max_drawdown(r),
        "asimetria": asim,
        "curtosis": curt,
        "cvar95_periodo": cvar95,
    }


def tabla_metricas(rend: pd.DataFrame, rf_periodo: float | pd.Series, periodos: int, umbral_autocorr: float) -> pd.DataFrame:
    filas = {c: metricas_fondo(rend[c], rf_periodo, periodos, umbral_autocorr) for c in rend.columns}
    return pd.DataFrame.from_dict(filas, orient="index")


# ----------------------------------------------------------------------------
# Análisis de estilo (Sharpe, 1992)
# ----------------------------------------------------------------------------
def analisis_estilo(r_fondo: pd.Series, r_factores: pd.DataFrame, rf_periodo: float | pd.Series) -> dict:
    """Regresión restringida: r_fondo ≈ Σ w_i r_factor_i + (1 - Σw) rf,
    con w_i >= 0 y Σ w_i <= 1 (el resto se interpreta como liquidez).
    Devuelve pesos de estilo y R²."""
    datos = pd.concat([r_fondo.rename("_y"), r_factores], axis=1).dropna()
    if len(datos) < 26 or r_factores.shape[1] == 0:
        return {}
    rf = _rf_en(rf_periodo, datos.index)
    rf_vals = rf.values if not np.isscalar(rf) else rf
    y = datos["_y"].values - rf_vals
    X = datos[r_factores.columns].values - (rf_vals if np.isscalar(rf_vals) else rf_vals.reshape(-1, 1))
    k = X.shape[1]

    def sse(w):
        e = y - X @ w
        return e @ e

    res = minimize(
        sse, np.full(k, 1 / (k + 1)), method="SLSQP",
        bounds=[(0, 1)] * k,
        constraints=[{"type": "ineq", "fun": lambda w: 1 - w.sum()}],
        options={"ftol": 1e-12, "maxiter": 500},
    )
    w = np.clip(res.x, 0, 1)
    resid = y - X @ w
    sst = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / sst if sst > 0 else np.nan
    out = {f"estilo_{c}": float(v) for c, v in zip(r_factores.columns, w)}
    out["estilo_Liquidez"] = float(max(0.0, 1 - w.sum()))
    out["estilo_R2"] = float(r2)
    return out


def tabla_estilo(rend: pd.DataFrame, rend_factores: pd.DataFrame, rf_periodo: float) -> pd.DataFrame:
    if rend_factores.empty:
        return pd.DataFrame(index=rend.columns)
    filas = {c: analisis_estilo(rend[c], rend_factores, rf_periodo) for c in rend.columns}
    return pd.DataFrame.from_dict(filas, orient="index")


def clasificar(fila: pd.Series, grupos: dict, vol_minima: float) -> str:
    """Tipo de activo a partir del estilo (preferente) o de la composición declarada."""
    if np.isfinite(fila.get("vol_anual", np.nan)) and fila["vol_anual"] < vol_minima:
        return "Monetario / liquidez"
    pesos = {}
    for factor, grupo in grupos.items():
        v = fila.get(f"estilo_{factor}", np.nan)
        if np.isfinite(v):
            pesos[grupo] = pesos.get(grupo, 0) + v
    r2 = fila.get("estilo_R2", np.nan)
    if pesos and np.isfinite(r2) and r2 >= 0.5:
        total = sum(pesos.values()) + fila.get("estilo_Liquidez", 0)
        rv = pesos.get("RV", 0) / total if total else 0
        rf = (pesos.get("RF", 0) + fila.get("estilo_Liquidez", 0)) / total if total else 0
        otros = {g: v / total for g, v in pesos.items() if g not in ("RV", "RF")}
        if rv >= 0.75:
            return "Renta variable"
        if rf >= 0.75:
            return "Renta fija"
        for g, v in otros.items():
            if v >= 0.6:
                return g
        return "Mixto"
    # Sin buen ajuste de estilo: usar composición declarada, si existe
    comp = {k: fila.get(k, np.nan) for k in ("pct_rv", "pct_rf", "pct_liquidez", "pct_otros")}
    vals = {k: v for k, v in comp.items() if np.isfinite(v)}
    if vals:
        total = sum(vals.values()) or 1
        escala = 100 if total > 1.5 else 1  # algunas fuentes dan %, otras fracción
        rv = vals.get("pct_rv", 0) / escala
        rf = (vals.get("pct_rf", 0) + vals.get("pct_liquidez", 0)) / escala
        if rv >= 0.75:
            return "Renta variable"
        if rf >= 0.75:
            return "Renta fija"
        if vals.get("pct_otros", 0) / escala >= 0.5:
            return "Alternativo / otros"
        return "Mixto"
    if np.isfinite(r2):
        return "Alternativo / otros"   # no lo explican bien los índices tradicionales
    return "Sin clasificar"
