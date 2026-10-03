"""
Pipeline completo: universo -> datos -> métricas -> selección -> cartera HRP -> backtest -> informe.

Uso:
    python main.py                      # usa config.toml
    python main.py --fuente demo        # prueba con datos sintéticos
    python main.py --sin-backtest       # más rápido
"""
from __future__ import annotations

import argparse
import os
import sys
import tomllib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

# La consola de Windows suele usar cp1252, que no admite caracteres como "→";
# sin esto, cualquier print con texto no-latino-1 revienta con UnicodeEncodeError.
if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import backtest as B  # noqa: E402
import datos as D  # noqa: E402
import documentos as DOC  # noqa: E402
import informe as I  # noqa: E402
import informe_pdf as PDF  # noqa: E402
import metricas as M  # noqa: E402
from optimizacion import (calcular_pesos, contribucion_riesgo, enlazar,  # noqa: E402
                          ledoit_wolf_correlacion, redondear_pesos)
from seleccion import seleccionar  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)


def main():
    ap = argparse.ArgumentParser(description="Selección cuantitativa de fondos y cartera HRP")
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--fuente", help="yahoo | morningstar | csv | demo (sobrescribe config)")
    ap.add_argument("--universo", help="CSV de ISIN (sobrescribe config)")
    ap.add_argument("--sin-backtest", action="store_true")
    args = ap.parse_args()

    os.chdir(AQUI)
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    if args.fuente:
        cfg["datos"]["fuente"] = args.fuente
    if args.universo:
        cfg["datos"]["archivo_universo"] = args.universo
    if args.sin_backtest:
        cfg["backtest"]["activar"] = False
    if cfg["datos"]["fuente"] == "demo":
        cfg["datos"]["carpeta_cache"] = "cache_demo"
        cfg["salida"]["carpeta"] = "salida_demo"

    salida = Path(cfg["salida"]["carpeta"])
    salida.mkdir(exist_ok=True)
    mcfg, ccfg = cfg["metricas"], cfg["cartera"]
    periodos = M.periodos_por_anio(mcfg["frecuencia"])

    # 1. Universo y datos ----------------------------------------------------------
    if cfg["datos"]["fuente"] == "demo":
        from demo import universo_demo
        universo = universo_demo()
    else:
        universo = D.cargar_universo(cfg["datos"]["archivo_universo"])
    print(f"\n1) Descargando {len(universo)} fondos (fuente: {cfg['datos']['fuente']})")
    precios, info = D.descargar_universo(universo, cfg)
    if precios.empty:
        sys.exit("No se ha podido descargar ningún fondo. Revisa el universo y la conexión.")
    print(f"   Descargados con éxito: {precios.shape[1]} de {len(universo)}")

    print("\n2) Índices de estilo")
    factores = D.descargar_factores_estilo(cfg)
    print(f"   {factores.shape[1]} índices: {', '.join(factores.columns) or 'ninguno'}")

    rend = M.a_rendimientos(precios, mcfg["frecuencia"])
    rend_fact = M.a_rendimientos(factores, mcfg["frecuencia"]) if not factores.empty else pd.DataFrame(index=rend.index)

    estr = D.descargar_estr(cfg) if mcfg.get("rf_fuente", "fijo") == "estr" else None
    rf_p = M.construir_rf_periodo(estr, rend.index, periodos, mcfg["tipo_libre_riesgo"])

    rend_estres = None
    if ccfg.get("usar_corr_estres", False):
        ini = pd.Timestamp(ccfg.get("estres_inicio", "2022-01-01"))
        fin = pd.Timestamp(ccfg.get("estres_fin", "2022-12-31"))
        rend_estres = rend.loc[(rend.index >= ini) & (rend.index <= fin)]

    # 2. Selección sobre la ventana más reciente --------------------------------------
    W = int(mcfg["ventana_anios"] * periodos)
    ventana = rend.iloc[-W:]
    rf_p_ventana = rf_p if np.isscalar(rf_p) else rf_p.reindex(ventana.index)
    print(f"\n3) Selección ({ventana.index[0].date()} → {ventana.index[-1].date()}, {len(ventana)} semanas)")
    tabla, sel = seleccionar(ventana, info, rend_fact, cfg, rf_periodo=rf_p_ventana)
    if not sel:
        sys.exit("Ningún fondo supera los filtros. Revisa config.toml (ventana, cobertura, filtros).")
    print(f"    {len(sel)} fondos seleccionados")

    # 3. Pesos --------------------------------------------------------------------
    metodo = ccfg["metodo"]
    print(f"\n4) Construcción de cartera: {metodo.upper()}")
    credito_info = None
    if ccfg.get("limitar_credito", False):
        col_est = "estilo_RF Corporativa EUR"
        credito_info = pd.DataFrame(index=sel)
        credito_info["subcategoria_universo"] = (info["subcategoria_universo"].reindex(sel)
                                                  if "subcategoria_universo" in info.columns else "")
        credito_info[col_est] = tabla[col_est].reindex(sel) if col_est in tabla.columns else 0.0
    re_sel = rend_estres[sel] if rend_estres is not None else None
    w = calcular_pesos(metodo, ventana[sel], rf_p_ventana, ccfg, credito_info, re_sel)
    w = w[w > 0]
    w_red = redondear_pesos(w, cfg["cartera"]["decimales_peso"])
    re_w = rend_estres[w.index] if rend_estres is not None else None
    cov, corr, intensidad = ledoit_wolf_correlacion(ventana[w.index], re_w,
                                                     ccfg.get("usar_corr_estres", False))

    nombres = info["nombre"].reindex(tabla.index)
    tabla.insert(0, "nombre", nombres)
    cols = ["nombre", "tipo", "rent_anual", "vol_anual", "vol_ajustada", "sharpe_ajustado", "psr",
            "sortino", "max_drawdown", "autocorr_1", "estilo_R2", "cluster"]
    cartera = tabla.loc[w.index, [c for c in cols if c in tabla.columns]].copy()
    cartera.insert(2, "peso", w)
    cartera.insert(3, "peso_%_redondeado", w_red)
    cartera.insert(4, "contrib_riesgo", contribucion_riesgo(w, cov))
    for col in ("gestora", "subcategoria_universo", "inversion_minima"):
        if col in info.columns:
            cartera[col] = info[col].reindex(cartera.index)
    cartera["ter"] = info["ter"].reindex(cartera.index)
    cartera["moneda_original"] = info["moneda"].reindex(cartera.index)
    cartera["id_fuente"] = info["id_fuente"].reindex(cartera.index)
    cartera = cartera.sort_values("peso", ascending=False)
    cartera.index.name = "isin"

    r_cartera = (ventana[w.index].fillna(0) @ w)
    stats = {
        "vol_esperada": float(np.sqrt(w @ cov.values @ w) * np.sqrt(periodos)),
        "rent_hist": float((1 + r_cartera).prod() ** (periodos / len(r_cartera)) - 1),
        "vol_hist": float(r_cartera.std() * np.sqrt(periodos)),
        "sharpe_hist": float((r_cartera - rf_p_ventana).mean() / r_cartera.std() * np.sqrt(periodos)),
        "dd_hist": M.max_drawdown(r_cartera),
    }
    exposicion = cartera.groupby("tipo")["peso"].sum().sort_values(ascending=False)
    estilo_cols = [c for c in tabla.columns if c.startswith("estilo_") and c != "estilo_R2"]
    exposicion_estilo = (tabla.loc[w.index, estilo_cols].fillna(0).T @ w).rename("peso_cartera")

    print()
    print(I.texto_resumen(cartera, stats, exposicion))
    print()
    print(cartera[["nombre", "tipo", "peso_%_redondeado"]].to_string())

    # 4. Backtest -------------------------------------------------------------------
    serie_bt, resumen_bt = pd.DataFrame(), pd.DataFrame()
    if cfg["backtest"]["activar"]:
        print("\n5) Backtest walk-forward")
        serie_bt, resumen_bt = B.ejecutar(rend, info, rend_fact, cfg)
        if not resumen_bt.empty:
            print(resumen_bt.round(3).to_string())
        B.registrar_prueba(cfg, resumen_bt, salida / "registro_pruebas.csv")

    # 5. Informe --------------------------------------------------------------------
    print("\n6) Informe")
    imgs = [
        I.grafico_pesos(cartera, salida / "pesos.png"),
        I.grafico_dendrograma(enlazar(corr, cfg["cartera"].get("enlace_hrp", "ward")),
                              list(nombres.reindex(w.index).fillna(pd.Series(w.index, index=w.index))),
                              salida / "dendrograma.png"),
        I.grafico_riesgo_rentabilidad(tabla, list(w.index), salida / "riesgo_rentabilidad.png"),
        I.grafico_correlaciones(corr, nombres.reindex(w.index), salida / "correlaciones.png"),
        I.grafico_frontera_eficiente(ventana[w.index], cov, w, periodos,
                                     float(rf_p_ventana if np.isscalar(rf_p_ventana) else rf_p_ventana.mean()),
                                     nombres.reindex(w.index), salida / "frontera_eficiente.png"),
    ]
    if not serie_bt.empty:
        imgs.append(I.grafico_backtest(serie_bt, salida / "backtest.png"))

    resumen = pd.DataFrame({
        "valor": [len(cartera), metodo, stats["vol_esperada"], stats["rent_hist"], stats["vol_hist"],
                  stats["sharpe_hist"], stats["dd_hist"], intensidad,
                  f"{ventana.index[0].date()} → {ventana.index[-1].date()}",
                  len(universo), precios.shape[1], int((tabla["estado"] == "seleccionado").sum())],
    }, index=["fondos en cartera", "método", "volatilidad esperada", "rentabilidad histórica (ventana)",
              "volatilidad histórica (ventana)", "Sharpe histórico (ventana)", "caída máxima (ventana)",
              "shrinkage Ledoit-Wolf", "ventana de estimación", "fondos en el universo",
              "fondos con datos", "fondos seleccionados"])
    resumen.index.name = "concepto"
    hojas = {
        "Cartera": cartera,
        "Resumen": resumen,
        "Exposición": pd.concat([exposicion.rename("peso").to_frame().assign(nivel="tipo de activo"),
                                 exposicion_estilo.rename("peso").to_frame().assign(nivel="estilo (regresión)")]),
        "Universo": tabla.sort_values(["estado", "psr"], ascending=[True, False]),
        "Correlaciones": corr.rename(index=nombres, columns=nombres).round(2),
        "Configuración": I.configuracion_plana(cfg),
    }
    hojas["Exposición"].index.name = "grupo"
    hojas["Universo"].index.name = "isin"
    if not resumen_bt.empty:
        resumen_bt.index.name = "método"
        hojas["Backtest"] = resumen_bt
    info_out = info.copy()
    info_out.index.name = "isin"
    hojas["Datos descarga"] = info_out

    # Primero los CSV (no dependen de nada), después el Excel
    csv = salida / "cartera_para_ironia.csv"
    cartera[["nombre", "peso_%_redondeado"]].to_csv(csv, sep=";", decimal=",", encoding="utf-8-sig")
    carpeta_csv = salida / "csv"
    carpeta_csv.mkdir(exist_ok=True)
    for nombre_hoja, df in hojas.items():
        df.to_csv(carpeta_csv / f"{nombre_hoja}.csv", sep=";", decimal=",", encoding="utf-8-sig")
    print(f"   {csv}\n   tablas en {carpeta_csv}/\n   gráficos en {salida}/")
    try:
        excel = I.exportar_excel(salida / "cartera_inicial.xlsx", hojas, imgs)
        print(f"   {excel}")
    except ImportError as e:
        print(f"   [aviso] No se ha generado el Excel: falta una librería ({e.name}). "
              f"Instálala con:  pip install openpyxl pillow   y vuelve a ejecutar "
              f"(los datos están en caché, tardará poco).")

    # 6. Documentos para el Proyecto de Claude ---------------------------------------
    print("\n7) Documentos del proyecto")
    DOC.generar(cfg, AQUI, salida, cartera, tabla, info, ventana, periodos, resumen_bt)

    # 7. Informe PDF (plantilla Jinja2 + pdflatex) -----------------------------------
    print("\n8) Informe PDF")
    ruta_informe = PDF.generar(cfg, AQUI, salida, cartera, tabla, info, ventana, periodos,
                               stats, exposicion, exposicion_estilo, corr, resumen_bt,
                               intensidad, universo, precios, sel)
    if ruta_informe:
        print(f"   {ruta_informe}")


if __name__ == "__main__":
    main()