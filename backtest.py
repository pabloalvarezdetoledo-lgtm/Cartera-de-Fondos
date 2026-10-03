"""
Backtest walk-forward: en cada fecha de rebalanceo se repite TODO el proceso
(selección + pesos) usando solo datos disponibles hasta ese día, y se mantiene la
cartera hasta el siguiente rebalanceo dejando que los pesos deriven.

Limitación inevitable: el universo son los fondos que existen hoy (sesgo de
supervivencia), así que el backtest es algo optimista en términos absolutos.
Sirve sobre todo para COMPARAR métodos entre sí.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import datos as D
import metricas as M
from optimizacion import calcular_pesos
from seleccion import seleccionar


def fechas_rebalanceo(indice: pd.DatetimeIndex, inicio: int, frecuencia: str) -> list[pd.Timestamp]:
    idx = indice[inicio:]
    if len(idx) == 0:
        return []
    fin_periodo = pd.Series(idx, index=idx).resample(frecuencia).last().dropna()
    # primera fecha: el primer cierre de periodo con ventana completa
    fechas = [f for f in fin_periodo.values if f >= indice[inicio - 1]]
    fechas = [pd.Timestamp(f) for f in fechas]
    if fechas and fechas[-1] < indice[-1]:
        fechas.append(indice[-1])   # último tramo hasta el final de los datos
    return fechas


def ejecutar(rend: pd.DataFrame, info: pd.DataFrame, rend_factores: pd.DataFrame,
             cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    mcfg, bcfg, ccfg = cfg["metricas"], cfg["backtest"], cfg["cartera"]
    periodos = M.periodos_por_anio(mcfg["frecuencia"])
    estr = D.descargar_estr(cfg) if mcfg.get("rf_fuente", "fijo") == "estr" else None
    rf_p = M.construir_rf_periodo(estr, rend.index, periodos, mcfg["tipo_libre_riesgo"])
    W = int(mcfg["ventana_anios"] * periodos)
    if len(rend) <= W + periodos // 4:
        print("  Histórico insuficiente para backtest fuera de muestra "
              f"(hay {len(rend)} periodos, la ventana usa {W}). Aumenta 'anios_historico'.")
        return pd.DataFrame(), pd.DataFrame()

    rend_estres = None
    if ccfg.get("usar_corr_estres", False):
        ini = pd.Timestamp(ccfg.get("estres_inicio", "2022-01-01"))
        fin = pd.Timestamp(ccfg.get("estres_fin", "2022-12-31"))
        rend_estres = rend.loc[(rend.index >= ini) & (rend.index <= fin)]

    fechas = fechas_rebalanceo(rend.index, W, bcfg["rebalanceo"])
    metodos = list(bcfg["metodos"]) + ["universo_1N"]
    retornos = {m: [] for m in metodos}
    rotacion = {m: [] for m in metodos}
    pesos_prev = {m: pd.Series(dtype=float) for m in metodos}
    sel_previa = None

    for j, t in enumerate(fechas[:-1]):
        t_sig = fechas[j + 1]
        ventana = rend.loc[:t].iloc[-W:]
        tabla, sel = seleccionar(ventana, info, rend_factores, cfg, verbose=False,
                                 rf_periodo=rf_p, seleccion_previa=sel_previa)
        elegibles = tabla.index[~tabla["estado"].str.startswith("excluido: historial")]
        tramo = rend.loc[(rend.index > t) & (rend.index <= t_sig)]
        print(f"  Rebalanceo {t.date()}: {len(sel)} fondos seleccionados")

        credito_info = None
        if ccfg.get("limitar_credito", False) and sel:
            col_est = "estilo_RF Corporativa EUR"
            credito_info = pd.DataFrame(index=sel)
            credito_info["subcategoria_universo"] = (info["subcategoria_universo"].reindex(sel)
                                                      if "subcategoria_universo" in info.columns else "")
            credito_info[col_est] = tabla[col_est].reindex(sel) if col_est in tabla.columns else 0.0

        for m in metodos:
            if m == "universo_1N":
                w = pd.Series(1 / len(elegibles), index=elegibles)
            elif len(sel) == 0:
                w = pd.Series(dtype=float)
            else:
                re = rend_estres[sel] if rend_estres is not None else None
                w = calcular_pesos(m, ventana[sel], rf_p, cfg["cartera"], credito_info, re)
            rotacion[m].append(_rotacion(pesos_prev[m], w))
            r_tramo, w_final = _mantener(w, tramo)
            retornos[m].append(r_tramo)
            pesos_prev[m] = w_final
        sel_previa = sel

    serie = pd.DataFrame({m: pd.concat(retornos[m]) for m in metodos})
    resumen = pd.DataFrame({m: _estadisticas(serie[m], rf_p, periodos) for m in metodos}).T
    resumen["rotacion_media"] = [np.mean(rotacion[m][1:]) if len(rotacion[m]) > 1 else np.nan
                                 for m in metodos]
    resumen["rebalanceos"] = len(fechas) - 1
    if "hrp" in serie.columns:
        jk = {m: _jobson_korkie(serie[m], serie["hrp"], rf_p) for m in metodos}
        resumen["jk_p_valor"] = [jk[m][1] if m != "hrp" else np.nan for m in metodos]
    return serie, resumen


def _mantener(w: pd.Series, tramo: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Buy-and-hold durante el tramo. Un dato faltante cuenta como rentabilidad 0."""
    if w.empty:
        return pd.Series(0.0, index=tramo.index), w
    r = tramo[w.index].fillna(0.0)
    w_act = w.values.copy()
    salida = []
    for _, fila in r.iterrows():
        rp = float(w_act @ fila.values)
        salida.append(rp)
        w_act = w_act * (1 + fila.values) / (1 + rp)
    return pd.Series(salida, index=tramo.index), pd.Series(w_act, index=w.index)


def _rotacion(anterior: pd.Series, nueva: pd.Series) -> float:
    idx = anterior.index.union(nueva.index)
    return float((anterior.reindex(idx, fill_value=0) - nueva.reindex(idx, fill_value=0)).abs().sum() / 2)


def _estadisticas(r: pd.Series, rf_periodo: float | pd.Series, periodos: int) -> dict:
    r = r.dropna()
    T = len(r)
    rf = rf_periodo if np.isscalar(rf_periodo) else rf_periodo.reindex(r.index)
    exceso = r - rf
    sd = r.std(ddof=1)
    vol = sd * np.sqrt(periodos)
    sr_periodo = exceso.mean() / sd if sd else np.nan
    sharpe = sr_periodo * np.sqrt(periodos)
    # Error estándar del Sharpe (Lo, 2002, caso iid), anualizado
    se_sharpe = (np.sqrt((1 + 0.5 * sr_periodo ** 2) / T) * np.sqrt(periodos)
                if T > 1 and np.isfinite(sr_periodo) else np.nan)
    return {
        "rent_anual": (1 + r).prod() ** (periodos / T) - 1,
        "vol_anual": vol,
        "sharpe": sharpe,
        "se_sharpe": se_sharpe,
        "ic95_inferior": sharpe - 1.96 * se_sharpe if np.isfinite(se_sharpe) else np.nan,
        "ic95_superior": sharpe + 1.96 * se_sharpe if np.isfinite(se_sharpe) else np.nan,
        "max_drawdown": M.max_drawdown(r),
        "semanas": T,
    }


def _jobson_korkie(r1: pd.Series, r2: pd.Series, rf_periodo: float | pd.Series) -> tuple[float, float]:
    """Test de diferencia de Sharpe (Jobson y Korkie, 1981; corrección de varianza de
    Memmel, 2003). Devuelve (z, p-valor) en unidades periódicas; r1 y r2 deben ser
    rentabilidades del mismo periodo (se emparejan por fecha)."""
    df = pd.concat([r1.rename("r1"), r2.rename("r2")], axis=1).dropna()
    T = len(df)
    if T < 10:
        return np.nan, np.nan
    rf = rf_periodo if np.isscalar(rf_periodo) else rf_periodo.reindex(df.index)
    sd1, sd2 = df["r1"].std(ddof=1), df["r2"].std(ddof=1)
    if not sd1 or not sd2:
        return np.nan, np.nan
    sr1 = (df["r1"] - rf).mean() / sd1
    sr2 = (df["r2"] - rf).mean() / sd2
    rho = df["r1"].corr(df["r2"])
    var = (2 - 2 * rho + 0.5 * sr1 ** 2 + 0.5 * sr2 ** 2 - rho ** 2 * sr1 * sr2) / T
    if not np.isfinite(var) or var <= 0:
        return np.nan, np.nan
    z = (sr1 - sr2) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def registrar_prueba(cfg: dict, resumen: pd.DataFrame, ruta: Path) -> None:
    """Añade una fila a salida/registro_pruebas.csv con fecha, hash del config,
    parámetros clave y Sharpe por método (histórico para un futuro deflated Sharpe ratio)."""
    ccfg, mcfg, scfg, fcfg = cfg["cartera"], cfg["metricas"], cfg["seleccion"], cfg["filtros"]
    config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
    fila = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "hash_config": config_hash,
        "rf_fuente": mcfg.get("rf_fuente", "fijo"),
        "limitar_credito": ccfg.get("limitar_credito", False),
        "usar_corr_estres": ccfg.get("usar_corr_estres", False),
        "suelo": ccfg.get("suelo", "fijo"),
        "ranking_multiventana": scfg.get("ranking_multiventana", False),
        "usar_histeresis": scfg.get("usar_histeresis", False),
        "excluir_anomalos": fcfg.get("excluir_anomalos", True),
        "metodo": ccfg.get("metodo"),
        "numero_fondos": scfg.get("numero_fondos"),
    }
    if resumen is not None and not resumen.empty:
        for m in resumen.index:
            v = resumen.loc[m, "sharpe"]
            fila[f"sharpe_{m}"] = round(float(v), 4) if np.isfinite(v) else ""
    nuevo = not ruta.exists()
    with open(ruta, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(fila.keys()))
        if nuevo:
            w.writeheader()
        w.writerow(fila)
