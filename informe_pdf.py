"""
Informe PDF generado dinámicamente a partir de los datos de cada ejecución
(cartera, tabla, correlaciones, backtest...), con una plantilla Jinja2
(plantillas/informe.tex.j2) compilada con pdflatex si está instalado.

A diferencia de un .tex escrito a mano, ningún número ni nombre de fondo está fijo
en la plantilla: todo se calcula aquí a partir de los mismos objetos que ya usan
informe.py (gráficos, Excel) y documentos.py (estado_cartera.md, metodologia.md),
así que el informe nunca puede quedar desincronizado de la cartera real.
"""
from __future__ import annotations

import datetime as dt
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from optimizacion import es_bloque_credito

MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
        "septiembre", "octubre", "noviembre", "diciembre"]

NOMBRES_METODO = {
    "hrp": "HRP", "inversa_vol": "Inversa de la volatilidad", "min_var": "Mínima varianza",
    "max_sharpe": "Máximo Sharpe (encogido)", "equiponderada": "Equiponderada (seleccionados)",
    "universo_1N": "1/N (todo el universo filtrado)",
}

PNGS = ["pesos.png", "dendrograma.png", "riesgo_rentabilidad.png", "correlaciones.png",
       "frontera_eficiente.png", "backtest.png"]


# ----------------------------------------------------------------------------
# Utilidades de formato y escapado LaTeX
# ----------------------------------------------------------------------------
def _tex(texto) -> str:
    """Escapa texto arbitrario (nombres de fondo, gestoras...) para LaTeX."""
    if texto is None or (isinstance(texto, float) and not np.isfinite(texto)):
        return ""
    s = str(texto)
    partes = s.split('"')
    s = partes[0]
    for i, p in enumerate(partes[1:]):
        s += ("``" if i % 2 == 0 else "''") + p
    for viejo, nuevo in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                         ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                         ("}", r"\}"), ("~", r"\textasciitilde{}"),
                         ("^", r"\textasciicircum{}")]:
        s = s.replace(viejo, nuevo)
    return s


def _num(x, decimales: int = 1) -> str:
    if x is None or not np.isfinite(x):
        return "n/d"
    return f"{x:.{decimales}f}".replace(".", "{,}").replace("-", "$-$")


def _pct(x, decimales: int = 1) -> str:
    if x is None or not np.isfinite(x):
        return "n/d"
    return _num(x * 100, decimales)


def _fecha_es(fecha: dt.date) -> str:
    return f"{fecha.day} de {MESES[fecha.month - 1]} de {fecha.year}"


# ----------------------------------------------------------------------------
# Construcción del contexto de la plantilla
# ----------------------------------------------------------------------------
def _fondos_contexto(cartera: pd.DataFrame, tabla: pd.DataFrame, corr: pd.DataFrame,
                     cfg: dict) -> list[dict]:
    mcfg, ccfg = cfg["metricas"], cfg["cartera"]
    umbral_autocorr = mcfg["autocorrelacion_alerta"]
    pmax, pmin, suelo = ccfg["peso_maximo"], ccfg["peso_minimo"], ccfg.get("suelo", "fijo")
    estilo_cols = [c for c in tabla.columns if c.startswith("estilo_") and c not in
                  ("estilo_R2", "estilo_Liquidez")]

    fondos = []
    for i, (isin, fila) in enumerate(cartera.iterrows(), start=1):
        peso = float(fila["peso"])
        contrib = float(fila.get("contrib_riesgo", np.nan))
        factores = []
        for c in estilo_cols:
            v = tabla.loc[isin, c] if isin in tabla.index and c in tabla.columns else np.nan
            if np.isfinite(v) and v > 0.01:
                factores.append({"nombre": c.replace("estilo_", ""), "pct": _pct(float(v))})
        factores_texto = "; ".join(f"{f['nombre']}: {f['pct']}\\,\\%" for f in factores)
        inv_min = fila.get("inversion_minima")
        inv_min = float(inv_min) if pd.notna(inv_min) else None
        corr_nombre, corr_valor = None, None
        if isin in corr.index and len(corr.columns) > 1:
            fila_corr = corr.loc[isin].drop(isin, errors="ignore").abs()
            if len(fila_corr):
                otro = fila_corr.idxmax()
                corr_nombre = cartera.loc[otro, "nombre"] if otro in cartera.index else otro
                corr_valor = float(corr.loc[isin, otro])
        autocorr = float(fila.get("autocorr_1", np.nan))
        fondos.append({
            "codigo": f"F{i}", "isin": isin, "nombre": _tex(fila.get("nombre", isin)),
            "tipo": _tex(fila.get("tipo", "")), "peso": peso, "peso_pct": _pct(peso),
            "contrib_pct": _pct(contrib) if np.isfinite(contrib) else "n/d",
            "concentracion": _num(contrib / peso, 2) if peso and np.isfinite(contrib) else "n/d",
            "rent_pct": _pct(float(fila.get("rent_anual", np.nan))),
            "vol_aj_pct": _pct(float(fila.get("vol_ajustada", np.nan))),
            "vol_anual_pct": _pct(float(fila.get("vol_anual", np.nan))),
            "ajuste_lo": np.isfinite(autocorr) and autocorr > umbral_autocorr,
            "autocorr": _num(autocorr, 3),
            "sharpe_aj": _num(float(fila.get("sharpe_ajustado", np.nan)), 2),
            "psr_pct": _pct(float(fila.get("psr", np.nan))),
            "sortino": _num(float(fila.get("sortino", np.nan)), 2),
            "dd_pct": _pct(float(fila.get("max_drawdown", np.nan))),
            "ter_pct": _pct(float(fila.get("ter", np.nan))) if "ter" in fila and np.isfinite(fila.get("ter", np.nan)) else "n/d",
            "r2": _num(float(tabla.loc[isin, "estilo_R2"]), 2) if isin in tabla.index and "estilo_R2" in tabla.columns and np.isfinite(tabla.loc[isin, "estilo_R2"]) else "n/d",
            "factores": factores, "factores_texto": factores_texto,
            "corr_max_nombre": _tex(corr_nombre) if corr_nombre is not None else None,
            "corr_max_valor": _num(corr_valor, 2) if corr_valor is not None else None,
            "en_techo": abs(peso - pmax) < 1e-6,
            "en_suelo": suelo == "fijo" and abs(peso - pmin) < 1e-6,
            "gestora": _tex(fila.get("gestora", "")),
            "subcategoria": _tex(fila.get("subcategoria_universo", "")),
            "inversion_minima_texto": f"{inv_min:,.0f}".replace(",", ".") if inv_min is not None else None,
            "_contrib_num": contrib, "_peso_num": peso,
            "_concentracion_num": contrib / peso if peso else np.nan,
        })
    return fondos


def _superlativos(fondos: list[dict], corr: pd.DataFrame) -> dict:
    if not fondos:
        return {}
    by_contrib = sorted(fondos, key=lambda f: f["_contrib_num"], reverse=True)
    by_conc = sorted(fondos, key=lambda f: f["_concentracion_num"], reverse=True)
    corr_media = {}
    for f in fondos:
        isin = f["isin"]
        if isin in corr.index:
            serie = corr.loc[isin].drop(isin, errors="ignore")
            if len(serie):
                corr_media[isin] = float(serie.mean())
    out = {
        "max_contrib": by_contrib[0], "max_concentracion": by_conc[0],
        "min_concentracion": by_conc[-1],
    }
    if corr_media:
        isin_max = max(corr_media, key=corr_media.get)
        isin_min = min(corr_media, key=corr_media.get)
        out["mas_conectado"] = next(f for f in fondos if f["isin"] == isin_max)
        out["mas_conectado_valor"] = _num(corr_media[isin_max], 2)
        out["mas_aislado"] = next(f for f in fondos if f["isin"] == isin_min)
        out["mas_aislado_valor"] = _num(corr_media[isin_min], 2)
    return out


def _bloque_credito(cartera: pd.DataFrame, tabla: pd.DataFrame, cfg: dict) -> dict:
    ccfg = cfg["cartera"]
    if not ccfg.get("limitar_credito", False):
        return {"activo": False}
    col = "estilo_RF Corporativa EUR"
    subcat = cartera.get("subcategoria_universo", pd.Series(index=cartera.index, dtype=object))
    estilo = tabla[col].reindex(cartera.index) if col in tabla.columns else pd.Series(0.0, index=cartera.index)
    es_credito = es_bloque_credito(subcat, estilo, ccfg.get("credito_palabras", []),
                                   ccfg.get("credito_umbral_estilo", 0.30))
    miembros = cartera.index[es_credito.reindex(cartera.index).fillna(False)]
    peso = float(cartera.loc[miembros, "peso"].sum()) if len(miembros) else 0.0
    return {
        "activo": True, "peso_pct": _pct(peso),
        "techo_pct": _pct(ccfg.get("peso_maximo_credito", 0.50)),
        "vinculante": peso >= ccfg.get("peso_maximo_credito", 0.50) - 1e-9,
        "fondos": [_tex(cartera.loc[i, "nombre"]) for i in miembros],
    }


def _riesgos_cola(cartera: pd.DataFrame, cfg: dict) -> list[dict]:
    palabras = cfg.get("informe", {}).get("palabras_riesgo_cola", [])
    advertencias = {
        "cat bond": "riesgo de catástrofe natural o siniestro asegurador: un historial "
                    "tranquilo no implica ausencia de riesgo de cola, solo que no se ha "
                    "materializado un evento en la ventana analizada.",
        "insurance": "riesgo actuarial o de reaseguro, no de mercado: puede comportarse de "
                    "forma muy distinta a su historial reciente ante un evento asegurador.",
        "merger": "riesgo de ruptura de operaciones corporativas (fusiones/adquisiciones "
                  "pendientes), que se materializa de forma súbita y poco predecible a "
                  "partir de la volatilidad histórica de precios.",
        "arbitrage": "riesgo de ejecución y de ruptura de las posiciones que arbitra, "
                    "típicamente invisible en una ventana sin eventos de estrés.",
        "event": "riesgo ligado a eventos corporativos puntuales, con historial de "
                "rentabilidad potencialmente poco representativo de episodios de estrés.",
        "frontier": "riesgo país y de liquidez de mercados frontera, generalmente mayor "
                   "que el que sugiere la volatilidad de precios observada.",
        "style premia": "riesgo de reversión o saturación (crowding) de factores "
                        "sistemáticos, difícil de anticipar a partir del historial de "
                        "rentabilidad por sí solo.",
    }
    grupos = []
    for palabra in palabras:
        p = palabra.lower()
        miembros = []
        for isin, fila in cartera.iterrows():
            texto = f"{fila.get('nombre', '')} {fila.get('subcategoria_universo', '')}".lower()
            if p in texto:
                miembros.append(_tex(fila.get("nombre", isin)))
        if miembros:
            grupos.append({
                "palabra": palabra, "fondos": miembros,
                "advertencia": advertencias.get(p, "riesgo idiosincrático no bien capturado "
                                                    "por la volatilidad histórica de precios."),
            })
    return grupos


def _backtest_contexto(resumen_bt: pd.DataFrame) -> dict:
    if resumen_bt is None or resumen_bt.empty:
        return {"activo": False}
    filas = []
    for m, fila in resumen_bt.iterrows():
        filas.append({
            "metodo": NOMBRES_METODO.get(m, m), "es_hrp": m == "hrp",
            "rent_pct": _pct(float(fila["rent_anual"])), "vol_pct": _pct(float(fila["vol_anual"])),
            "sharpe": _num(float(fila["sharpe"]), 2),
            "ic95": f"[{_num(float(fila['ic95_inferior']), 2)}, {_num(float(fila['ic95_superior']), 2)}]"
                   if np.isfinite(fila.get("ic95_inferior", np.nan)) else "n/d",
            "jk_p": _num(float(fila["jk_p_valor"]), 2) if np.isfinite(fila.get("jk_p_valor", np.nan)) else "--",
            "dd_pct": _pct(float(fila["max_drawdown"])), "rotacion_pct": _pct(float(fila["rotacion_media"])),
        })
    hrp = resumen_bt.loc["hrp"] if "hrp" in resumen_bt.index else resumen_bt.iloc[0]
    jk_vals = resumen_bt["jk_p_valor"].dropna()
    return {
        "activo": True, "filas": filas,
        "sharpe_hrp": _num(float(hrp["sharpe"]), 2),
        "ic95_hrp": f"[{_num(float(hrp['ic95_inferior']), 2)}, {_num(float(hrp['ic95_superior']), 2)}]",
        "se_sharpe_hrp": _num(float(hrp["se_sharpe"]), 2),
        "dd_hrp_pct": _pct(float(hrp["max_drawdown"])),
        "rotacion_hrp_pct": _pct(float(hrp["rotacion_media"])),
        "n_rebalanceos": int(hrp["rebalanceos"]), "n_semanas": int(hrp["semanas"]),
        "jk_min_p": _num(float(jk_vals.min()), 2) if len(jk_vals) else "n/d",
        "jk_significativo": bool((jk_vals < 0.05).any()) if len(jk_vals) else False,
        "ic95_cruza_cero": hrp["ic95_inferior"] < 0 < hrp["ic95_superior"],
    }


def _correlaciones_contexto(cartera: pd.DataFrame, corr: pd.DataFrame) -> dict:
    codigos = {isin: f"F{i}" for i, isin in enumerate(cartera.index, start=1)}
    cods = [codigos[i] for i in cartera.index if i in corr.index]
    filas = []
    for isin in cartera.index:
        if isin not in corr.index:
            continue
        valores = [_num(float(corr.loc[isin, j]), 2) if j in corr.columns else "n/d"
                  for j in cartera.index]
        filas.append({"codigo": codigos[isin], "valores": valores})
    return {"codigos": cods, "filas": filas}


def construir_contexto(cfg: dict, cartera: pd.DataFrame, tabla: pd.DataFrame, info: pd.DataFrame,
                       ventana: pd.DataFrame, periodos: int, stats: dict, exposicion: pd.Series,
                       exposicion_estilo: pd.Series, corr: pd.DataFrame, resumen_bt: pd.DataFrame,
                       intensidad: float, universo: pd.DataFrame, precios: pd.DataFrame,
                       sel: list) -> dict:
    mcfg, fcfg, scfg, ccfg = cfg["metricas"], cfg["filtros"], cfg["seleccion"], cfg["cartera"]
    hoy = dt.date.today()
    fondos = _fondos_contexto(cartera, tabla, corr, cfg)
    exp_tipo = [{"nombre": _tex(k), "pct": _pct(float(v))} for k, v in exposicion.items()]
    exp_estilo = [{"nombre": _tex(k.replace("estilo_", "")), "pct": _pct(float(v))}
                 for k, v in exposicion_estilo.items()]

    return {
        "fecha_informe": _fecha_es(hoy), "fuente": cfg["datos"]["fuente"],
        "ventana_inicio": str(ventana.index[0].date()), "ventana_fin": str(ventana.index[-1].date()),
        "n_semanas": len(ventana), "ventana_anios": mcfg["ventana_anios"], "periodos": periodos,
        "universo_n": len(universo), "con_datos_n": precios.shape[1],
        "preseleccionados_n": len(sel), "cartera_n": len(cartera),
        "metodo": ccfg["metodo"].upper(), "vol_esperada_pct": _pct(stats["vol_esperada"]),
        "rent_hist_pct": _pct(stats["rent_hist"]), "vol_hist_pct": _pct(stats["vol_hist"]),
        "sharpe_hist": _num(stats["sharpe_hist"], 2), "dd_hist_pct": _pct(stats["dd_hist"]),
        "intensidad_pct": _pct(intensidad),
        "rf_fuente": mcfg.get("rf_fuente", "fijo"), "tipo_libre_riesgo_pct": _pct(mcfg["tipo_libre_riesgo"]),
        "autocorrelacion_alerta": _num(mcfg["autocorrelacion_alerta"], 2),
        "criterio_ranking": scfg["criterio_ranking"],
        "ranking_multiventana": scfg.get("ranking_multiventana", False),
        "ventanas_psr": scfg.get("ventanas_psr", []), "psr_referencia": _num(scfg.get("psr_referencia", 0.5), 1),
        "usar_histeresis": scfg.get("usar_histeresis", False),
        "margen_histeresis_pct": _pct(scfg.get("margen_histeresis", 0.05)),
        "numero_fondos_cluster": scfg["numero_fondos"],
        "correlacion_duplicados": _num(scfg["correlacion_duplicados"], 2),
        "cobertura_minima_pct": _pct(mcfg["cobertura_minima"]),
        "volatilidad_minima_pct": _pct(fcfg["volatilidad_minima"]), "volatilidad_maxima_pct": _pct(fcfg["volatilidad_maxima"]),
        "drawdown_maximo_pct": _pct(fcfg["drawdown_maximo"]),
        "excluir_anomalos": fcfg.get("excluir_anomalos", True),
        "peso_maximo_pct": _pct(ccfg["peso_maximo"]), "peso_minimo_pct": _pct(ccfg["peso_minimo"]),
        "suelo": ccfg.get("suelo", "fijo"), "minimo_fondos": ccfg.get("minimo_fondos", 8),
        "enlace_hrp": ccfg.get("enlace_hrp", "ward"), "decimales_peso": ccfg["decimales_peso"],
        "rebalanceo": cfg["backtest"]["rebalanceo"],
        "fondos": fondos, "superlativos": _superlativos(fondos, corr),
        "exposicion_tipo": exp_tipo, "exposicion_estilo": exp_estilo,
        "credito_umbral_estilo_pct": _pct(ccfg.get("credito_umbral_estilo", 0.30), 0),
        "estres_inicio": ccfg.get("estres_inicio", ""), "estres_fin": ccfg.get("estres_fin", ""),
        "usar_corr_estres": ccfg.get("usar_corr_estres", False),
        "credito": _bloque_credito(cartera, tabla, cfg),
        "cola": _riesgos_cola(cartera, cfg),
        "backtest": _backtest_contexto(resumen_bt),
        "correlaciones": _correlaciones_contexto(cartera, corr),
    }


# ----------------------------------------------------------------------------
# Plantilla + compilación
# ----------------------------------------------------------------------------
def _entorno(raiz: Path):
    import jinja2
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(raiz / "plantillas")),
        block_start_string=r"\BLOCK{", block_end_string="}",
        variable_start_string=r"\VAR{", variable_end_string="}",
        comment_start_string=r"\#{", comment_end_string="}",
        trim_blocks=True, lstrip_blocks=True,
    )
    return env


def generar(cfg: dict, raiz: Path, salida: Path, cartera: pd.DataFrame, tabla: pd.DataFrame,
           info: pd.DataFrame, ventana: pd.DataFrame, periodos: int, stats: dict,
           exposicion: pd.Series, exposicion_estilo: pd.Series, corr: pd.DataFrame,
           resumen_bt: pd.DataFrame, intensidad: float, universo: pd.DataFrame,
           precios: pd.DataFrame, sel: list) -> Path | None:
    carpeta = salida / "informe"
    carpeta.mkdir(parents=True, exist_ok=True)
    for png in PNGS:
        origen = salida / png
        if origen.exists():
            shutil.copy(origen, carpeta / png)

    contexto = construir_contexto(cfg, cartera, tabla, info, ventana, periodos, stats,
                                  exposicion, exposicion_estilo, corr, resumen_bt, intensidad,
                                  universo, precios, sel)
    contexto["imagenes"] = {p.replace(".png", ""): p for p in PNGS if (carpeta / p).exists()}

    env = _entorno(raiz)
    plantilla = env.get_template("informe.tex.j2")
    render = plantilla.render(**contexto)

    hoy = dt.date.today()
    ruta_tex = carpeta / f"Cartera_Fondos_CFA_{hoy:%Y-%m-%d}.tex"
    ruta_tex.write_text(render, encoding="utf-8")

    historico = carpeta / "historico"
    historico.mkdir(exist_ok=True)

    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        print("   [aviso] pdflatex no está instalado: se ha dejado el .tex sin compilar "
              f"({ruta_tex}). Instala una distribución TeX (p. ej. MiKTeX) y vuelve a "
              "ejecutar, o compílalo tú mismo.")
        shutil.copy(ruta_tex, historico / ruta_tex.name)
        return ruta_tex

    log = carpeta / "pdflatex.log"
    ok = True
    with open(log, "w", encoding="utf-8") as lf:
        for _ in range(2):   # dos pasadas para el índice
            r = subprocess.run(
                [pdflatex, "-interaction=nonstopmode", "-halt-on-error", ruta_tex.name],
                cwd=str(carpeta), stdout=lf, stderr=subprocess.STDOUT,
            )
            if r.returncode != 0:
                ok = False
                break

    ruta_pdf = ruta_tex.with_suffix(".pdf")
    if ok and ruta_pdf.exists():
        shutil.copy(ruta_pdf, historico / ruta_pdf.name)
        return ruta_pdf
    print(f"   [aviso] La compilación con pdflatex falló; revisa {log}. Se ha dejado el "
          f".tex sin compilar ({ruta_tex}).")
    shutil.copy(ruta_tex, historico / ruta_tex.name)
    return ruta_tex
