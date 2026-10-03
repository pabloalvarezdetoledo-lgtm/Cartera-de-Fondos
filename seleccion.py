"""
Selección cuantitativa del conjunto de fondos:
  1. historial suficiente
  2. filtros de riesgo (volatilidad, drawdown, PSR)
  3. eliminación de "duplicados" (fondos casi idénticos por correlación)
  4. reducción a N clusters y elección del mejor fondo de cada uno
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

import metricas as M
from optimizacion import distancia_correlacion

COLUMNA_RANKING = {"psr": "psr", "sharpe": "sharpe_ajustado", "sortino": "sortino"}


def _psr_vectorizado(sr: pd.Series, n: pd.Series, asim: pd.Series, curt: pd.Series,
                     sr_ref: float) -> pd.Series:
    """Igual que metricas.psr pero vectorizado sobre un índice de fondos."""
    denom = 1 - asim * sr + (curt - 1) / 4 * sr ** 2
    valido = (n >= 3) & np.isfinite(sr) & np.isfinite(denom) & (denom > 0)
    denom_seguro = denom.where(denom > 0, np.nan)
    z = (sr - sr_ref) * np.sqrt((n - 1).clip(lower=0)) / np.sqrt(denom_seguro)
    out = pd.Series(stats.norm.cdf(z), index=sr.index)
    return out.where(valido)


def _ranking_multiventana(rend: pd.DataFrame, candidatos: list, rf_periodo, periodos: int,
                          umbral_autocorr: float, ventanas_anios: list, psr_referencia: float) -> pd.Series:
    """Media del PSR en varias sub-ventanas (p. ej. 1, 2 y 3 años), frente a un Sharpe
    anual de referencia (psr_referencia) en vez de 0: frente a 0 casi todos los fondos
    buenos salen con PSR ≈ 100 % y el ranking no discrimina."""
    sr_ref_periodo = psr_referencia / np.sqrt(periodos)
    columnas = []
    for anios_v in ventanas_anios:
        W_v = max(1, int(anios_v * periodos))
        sub = rend[candidatos].iloc[-W_v:]
        met_v = M.tabla_metricas(sub, rf_periodo, periodos, umbral_autocorr)
        sr_periodo_aj = met_v["sharpe_ajustado"] / np.sqrt(periodos)
        columnas.append(_psr_vectorizado(sr_periodo_aj, met_v["n_obs"], met_v["asimetria"],
                                         met_v["curtosis"], sr_ref_periodo))
    return pd.concat(columnas, axis=1).mean(axis=1)


def seleccionar(rend: pd.DataFrame, info: pd.DataFrame, rend_factores: pd.DataFrame,
                cfg: dict, verbose: bool = True, rf_periodo: float | pd.Series | None = None,
                seleccion_previa: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    mcfg, fcfg, scfg = cfg["metricas"], cfg["filtros"], cfg["seleccion"]
    periodos = M.periodos_por_anio(mcfg["frecuencia"])
    rf_p = rf_periodo if rf_periodo is not None else M.tipo_periodico(mcfg["tipo_libre_riesgo"], periodos)
    col_rank = COLUMNA_RANKING[scfg["criterio_ranking"]]

    tabla = pd.DataFrame(index=rend.columns)
    tabla["estado"] = "candidato"

    # 1. Historial --------------------------------------------------------------
    cobertura = rend.notna().mean()
    primer_dato = rend.apply(lambda s: s.first_valid_index())
    limite_inicio = rend.index[min(len(rend) - 1, int(0.05 * len(rend)))]
    sin_historial = (cobertura < mcfg["cobertura_minima"]) | (primer_dato > limite_inicio)
    tabla.loc[sin_historial, "estado"] = "excluido: historial insuficiente"
    tabla["cobertura"] = cobertura

    validos = tabla.index[tabla["estado"] == "candidato"]
    if len(validos) == 0:
        return tabla, []

    # 2. Métricas, estilo y filtros ---------------------------------------------
    met = M.tabla_metricas(rend[validos], rf_p, periodos, mcfg["autocorrelacion_alerta"])
    est = M.tabla_estilo(rend[validos], rend_factores.reindex(rend.index), rf_p)
    tabla = tabla.join(met).join(est)
    comp = info.reindex(tabla.index)[["pct_rv", "pct_rf", "pct_liquidez", "pct_otros"]]
    tabla = tabla.join(comp)
    grupos = cfg.get("estilo", {}).get("grupos", {})
    tabla["tipo"] = [
        M.clasificar(tabla.loc[i], grupos, fcfg["volatilidad_minima"]) if i in validos else None
        for i in tabla.index
    ]

    def excluir(mascara, motivo):
        m = mascara & (tabla["estado"] == "candidato")
        tabla.loc[m, "estado"] = f"excluido: {motivo}"

    if fcfg.get("excluir_anomalos", True):
        r_abs_max = rend[validos].abs().max()
        anomalos = (tabla["vol_anual"] > 2.0) | (r_abs_max.reindex(tabla.index) > 0.5)
        anomalos = anomalos & (tabla["estado"] == "candidato")
        if anomalos.any() and verbose:
            print(f"    {int(anomalos.sum())} fondos excluidos por datos anómalos: "
                  f"{', '.join(anomalos.index[anomalos])}")
        excluir(anomalos, "datos anómalos")

    excluir(tabla["vol_anual"] < fcfg["volatilidad_minima"], "monetario / volatilidad muy baja")
    excluir(tabla["vol_anual"] > fcfg["volatilidad_maxima"], "volatilidad excesiva")
    excluir(tabla["max_drawdown"] < -fcfg["drawdown_maximo"], "caída máxima excesiva")
    if fcfg.get("psr_minimo", 0) > 0:
        excluir(tabla["psr"] < fcfg["psr_minimo"], "PSR bajo")

    if scfg.get("ranking_multiventana", False) and scfg["criterio_ranking"] == "psr":
        candidatos_previos = list(tabla.index[tabla["estado"] == "candidato"])
        tabla.loc[candidatos_previos, "psr_multiventana"] = _ranking_multiventana(
            rend, candidatos_previos, rf_p, periodos, mcfg["autocorrelacion_alerta"],
            scfg.get("ventanas_psr", [1, 2, 3]), scfg.get("psr_referencia", 0.5))
        col_rank = "psr_multiventana"

    excluir(tabla[col_rank].isna(), "métricas no calculables")

    candidatos = list(tabla.index[tabla["estado"] == "candidato"])
    if verbose:
        print(f"    {len(candidatos)} fondos superan historial y filtros de riesgo")
    if len(candidatos) <= 1:
        tabla.loc[candidatos, "estado"] = "seleccionado"
        return tabla, candidatos

    c = rend[candidatos].corr(min_periods=20).fillna(0).values.copy()
    np.fill_diagonal(c, 1.0)
    corr = pd.DataFrame(c, index=candidatos, columns=candidatos)

    # 3. Duplicados: enlace completo => todos los pares del grupo tienen ρ >= umbral ----
    umbral = np.sqrt(0.5 * (1 - scfg["correlacion_duplicados"]))
    d = distancia_correlacion(corr)
    grupos_dup = fcluster(linkage(squareform(d.values, checks=False), "complete"), umbral, "distance")
    tabla.loc[candidatos, "grupo_duplicados"] = grupos_dup
    representantes = []
    for g in np.unique(grupos_dup):
        miembros = [c for c, gg in zip(candidatos, grupos_dup) if gg == g]
        mejor = tabla.loc[miembros, col_rank].idxmax()
        representantes.append(mejor)
        for m in miembros:
            if m != mejor:
                tabla.loc[m, "estado"] = f"duplicado de {mejor}"
    if verbose:
        print(f"    {len(representantes)} tras eliminar duplicados (ρ >= {scfg['correlacion_duplicados']})")

    # 4. N clusters, el mejor de cada uno ------------------------------------------
    n_obj = scfg["numero_fondos"]
    if len(representantes) <= n_obj:
        seleccion = representantes
        tabla.loc[representantes, "cluster"] = range(1, len(representantes) + 1)
    else:
        c2 = corr.loc[representantes, representantes]
        enlace = linkage(squareform(distancia_correlacion(c2).values, checks=False), "ward")
        clusters = fcluster(enlace, n_obj, "maxclust")
        tabla.loc[representantes, "cluster"] = clusters
        usar_hist = scfg.get("usar_histeresis", False) and seleccion_previa
        margen = scfg.get("margen_histeresis", 0.05)
        seleccion = []
        for k in np.unique(clusters):
            miembros = [r for r, kk in zip(representantes, clusters) if kk == k]
            mejor = tabla.loc[miembros, col_rank].idxmax()
            if usar_hist:
                previos = [m for m in miembros if m in seleccion_previa]
                if previos:
                    titular = tabla.loc[previos, col_rank].idxmax()
                    if tabla.loc[mejor, col_rank] - tabla.loc[titular, col_rank] <= margen:
                        mejor = titular
            seleccion.append(mejor)
            for m in miembros:
                if m != mejor:
                    tabla.loc[m, "estado"] = f"no elegido (cluster {k}, gana {mejor})"

    # 5. Diversidad de tipos de activo -------------------------------------------
    if scfg.get("forzar_diversidad_tipos", False):
        seleccion = _forzar_diversidad(tabla, representantes, seleccion, col_rank)

    tabla.loc[seleccion, "estado"] = "seleccionado"
    return tabla, seleccion


def _forzar_diversidad(tabla, representantes, seleccion, col_rank):
    ignorar = {"Sin clasificar", "Monetario / liquidez", None}
    tipos_disp = set(tabla.loc[representantes, "tipo"]) - ignorar
    for tipo in sorted(tipos_disp, key=str):
        if tipo in set(tabla.loc[seleccion, "tipo"]):
            continue
        entrante = tabla.loc[[r for r in representantes if tabla.loc[r, "tipo"] == tipo], col_rank].idxmax()
        # sale el peor fondo de un tipo que esté representado 2+ veces
        cuenta = tabla.loc[seleccion, "tipo"].value_counts()
        repetidos = [s for s in seleccion if cuenta.get(tabla.loc[s, "tipo"], 0) >= 2]
        if not repetidos:
            seleccion.append(entrante)
        else:
            saliente = tabla.loc[repetidos, col_rank].idxmin()
            seleccion.remove(saliente)
            tabla.loc[saliente, "estado"] = f"sustituido por {entrante} (diversidad de tipos)"
            seleccion.append(entrante)
    return seleccion
