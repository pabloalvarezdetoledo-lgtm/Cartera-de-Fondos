"""
Documentos de seguimiento para subir a un Proyecto de Claude.

Se generan en {carpeta_salida}/proyecto/:
  - estado_cartera.md: se SOBRESCRIBE en cada ejecución (foto actual: real vs. sombra, alertas).
  - metodologia.md: se SOBRESCRIBE en cada ejecución (resumen del proceso + parámetros vigentes).
  - diario_decisiones.md: se crea una sola vez y NUNCA se sobrescribe; cada ejecución solo
    añade una línea al final del registro automático.

La "cartera real" vive en cartera_real.csv, en la raíz del proyecto (no dentro de salida/),
porque es un fichero que mantiene la persona a mano y debe sobrevivir a que salida/ (o
salida_demo/) se regenere en cada ejecución.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

import metricas as M
from informe import configuracion_plana

NOMBRE_FICHERO_REAL = "cartera_real.csv"
FACTOR_CREDITO = "RF Corporativa EUR"


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def _pct(x) -> str:
    if x is None or not np.isfinite(x):
        return "n/d"
    return f"{x:.1%}"


def _num(x, decimales: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "n/d"
    return f"{x:.{decimales}f}"


def _nombre(isin: str, cartera: pd.DataFrame, info: pd.DataFrame) -> str:
    if isin in cartera.index:
        return str(cartera.loc[isin, "nombre"])
    if info is not None and isin in info.index:
        n = info.loc[isin, "nombre"]
        if isinstance(n, str) and n:
            return n
    return isin


def _tipo(isin: str, tabla: pd.DataFrame) -> str:
    if tabla is not None and isin in tabla.index:
        t = tabla.loc[isin, "tipo"]
        if isinstance(t, str) and t:
            return t
    return "Sin clasificar"


def _md_tabla(encabezados: list[str], filas: list[list]) -> str:
    out = ["| " + " | ".join(encabezados) + " |",
           "|" + "|".join(["---"] * len(encabezados)) + "|"]
    for fila in filas:
        out.append("| " + " | ".join(str(c) for c in fila) + " |")
    return "\n".join(out)


# ----------------------------------------------------------------------------
# Cartera real
# ----------------------------------------------------------------------------
def cargar_cartera_real(ruta: Path, sombra: pd.Series) -> tuple[pd.Series, bool]:
    """Lee cartera_real.csv (columnas isin;peso, peso en %). Si no existe, la crea
    igual que la cartera sombra y lo señala con el booleano devuelto."""
    if ruta.exists():
        df = pd.read_csv(ruta, sep=";", dtype=str, comment="#", skip_blank_lines=True)
        df.columns = [c.strip().lower() for c in df.columns]
        pesos = df["peso"].str.replace(",", ".", regex=False).astype(float)
        real = pd.Series(pesos.values, index=df["isin"].str.strip().values, dtype=float) / 100
        real = real.groupby(real.index).sum()
        return real, False

    real = sombra.copy()
    pd.DataFrame({"isin": real.index, "peso": (real * 100).round(2)}).to_csv(
        ruta, sep=";", index=False, encoding="utf-8-sig")
    return real, True


# ----------------------------------------------------------------------------
# Cálculos
# ----------------------------------------------------------------------------
def _sharpe(r: pd.Series, rf_anual: float, periodos: int) -> float:
    sd = r.std(ddof=1)
    if not sd or not np.isfinite(sd):
        return float("nan")
    rf_p = M.tipo_periodico(rf_anual, periodos)
    return float((r - rf_p).mean() / sd * np.sqrt(periodos))


def _metricas_cartera(pesos: pd.Series, ventana: pd.DataFrame, cfg: dict, periodos: int) -> dict:
    mcfg = cfg["metricas"]
    faltan = [i for i in pesos.index if i not in ventana.columns]
    r = (ventana.reindex(columns=pesos.index).fillna(0.0) @ pesos) if len(pesos) else pd.Series(dtype=float)
    r = r.dropna()
    tipos_rf = {"0": 0.0, "rf": mcfg["tipo_libre_riesgo"], "estres": mcfg.get("tipo_estres", 0.04)}
    if len(r) < 2:
        out = {"rent_anual": np.nan, "vol_anual": np.nan, "max_drawdown": np.nan}
        out.update({f"sharpe_{k}": np.nan for k in tipos_rf})
    else:
        out = {
            "rent_anual": float((1 + r).prod() ** (periodos / len(r)) - 1),
            "vol_anual": float(r.std(ddof=1) * np.sqrt(periodos)),
            "max_drawdown": M.max_drawdown(r),
        }
        out.update({f"sharpe_{k}": _sharpe(r, rf, periodos) for k, rf in tipos_rf.items()})
    out["isins_sin_datos"] = faltan
    return out


def _exposicion_tipo(pesos: pd.Series, tabla: pd.DataFrame) -> pd.Series:
    if not len(pesos):
        return pd.Series(dtype=float)
    tipos = pd.Index([_tipo(i, tabla) for i in pesos.index])
    return pesos.groupby(tipos).sum().sort_values(ascending=False)


def _peso_credito(pesos: pd.Series, tabla: pd.DataFrame, factor: str = FACTOR_CREDITO) -> float:
    col = f"estilo_{factor}"
    if tabla is None or col not in tabla.columns or not len(pesos):
        return float("nan")
    valores = tabla[col].reindex(pesos.index).fillna(0.0)
    return float((pesos * valores).sum())


def _comparar(real: pd.Series, sombra: pd.Series, banda: float) -> pd.DataFrame:
    idx = real.index.union(sombra.index)
    df = pd.DataFrame(index=idx)
    df["peso_real"] = real.reindex(idx).fillna(0.0)
    df["peso_sombra"] = sombra.reindex(idx).fillna(0.0)
    df["diferencia"] = df["peso_real"] - df["peso_sombra"]
    df["estado"] = np.where(df["diferencia"].abs() > banda, "FUERA DE BANDA", "ok")
    return df.reindex(df["diferencia"].abs().sort_values(ascending=False).index)


def _alertas(real: pd.Series, sombra: pd.Series, ccfg: dict) -> list[str]:
    alertas = []
    for i in real.index:
        if i not in sombra.index:
            alertas.append(f"{i}: está en la cartera real pero ya no forma parte de la cartera sombra.")
    techo, suelo = ccfg["peso_maximo"], ccfg["peso_minimo"]
    for etiqueta, serie in (("sombra", sombra), ("real", real)):
        for i, p in serie.items():
            if p >= techo - 1e-9:
                alertas.append(f"{i} ({etiqueta}): en el techo de peso ({_pct(p)} ≥ {_pct(techo)}).")
            elif 0 < p <= suelo + 1e-9:
                alertas.append(f"{i} ({etiqueta}): en el suelo de peso ({_pct(p)} ≤ {_pct(suelo)}).")
    return alertas


# ----------------------------------------------------------------------------
# estado_cartera.md
# ----------------------------------------------------------------------------
def _escribir_estado(ruta: Path, cfg: dict, cartera: pd.DataFrame, tabla: pd.DataFrame,
                     info: pd.DataFrame, ventana: pd.DataFrame, real: pd.Series, sombra: pd.Series,
                     creada: bool, periodos: int, resumen_bt: pd.DataFrame) -> int:
    mcfg, ccfg = cfg["metricas"], cfg["cartera"]
    banda = ccfg.get("banda_rebalanceo", 0.03)
    comparacion = _comparar(real, sombra, banda)
    n_fuera = int((comparacion["estado"] == "FUERA DE BANDA").sum())

    met_real = _metricas_cartera(real, ventana, cfg, periodos)
    met_sombra = _metricas_cartera(sombra, ventana, cfg, periodos)
    exp_real = _exposicion_tipo(real, tabla)
    exp_sombra = _exposicion_tipo(sombra, tabla)
    credito_real = _peso_credito(real, tabla)
    credito_sombra = _peso_credito(sombra, tabla)
    alertas = _alertas(real, sombra, ccfg)

    L = []
    L.append(f"# Estado de la cartera — {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    L.append("")
    L.append(f"Ventana de estimación: {ventana.index[0].date()} → {ventana.index[-1].date()} "
             f"({len(ventana)} periodos, frecuencia `{mcfg['frecuencia']}`). "
             f"Fuente de datos: `{cfg['datos']['fuente']}`.")
    L.append("")
    if creada:
        L.append(f"> **Aviso**: no existía `{NOMBRE_FICHERO_REAL}`; se ha creado igual que la "
                 f"cartera sombra. Edítalo a mano para reflejar tu cartera real.")
        L.append("")

    L.append("## Cartera sombra (modelo)")
    filas = [[isin, cartera.loc[isin, "nombre"], cartera.loc[isin, "tipo"], _pct(cartera.loc[isin, "peso"]),
              _pct(cartera.loc[isin, "contrib_riesgo"])]
             for isin in cartera.sort_values("peso", ascending=False).index]
    L.append(_md_tabla(["ISIN", "Nombre", "Tipo", "Peso", "Contrib. riesgo"], filas))
    L.append("")

    L.append("## Cartera real")
    filas = [[isin, _nombre(isin, cartera, info), _tipo(isin, tabla), _pct(real[isin])]
             for isin in real.sort_values(ascending=False).index]
    L.append(_md_tabla(["ISIN", "Nombre", "Tipo", "Peso"], filas))
    L.append("")

    L.append(f"## Real frente a sombra (banda de rebalanceo: {_pct(banda)})")
    filas = [[f.Index, _nombre(f.Index, cartera, info), _pct(f.peso_real), _pct(f.peso_sombra),
              _pct(f.diferencia), f.estado]
             for f in comparacion.itertuples(name="F")]
    L.append(_md_tabla(["ISIN", "Nombre", "Peso real", "Peso sombra", "Diferencia", "Estado"], filas))
    L.append("")

    L.append("## Métricas en la ventana de estimación")
    cab = ["Cartera", "Rentabilidad anual", "Volatilidad anual", "Sharpe (rf=0%)",
          f"Sharpe (rf={_pct(mcfg['tipo_libre_riesgo'])})",
          f"Sharpe (estrés={_pct(mcfg.get('tipo_estres', 0.04))})", "Caída máxima"]
    filas = [
        ["Real", _pct(met_real["rent_anual"]), _pct(met_real["vol_anual"]), _num(met_real["sharpe_0"]),
         _num(met_real["sharpe_rf"]), _num(met_real["sharpe_estres"]), _pct(met_real["max_drawdown"])],
        ["Sombra", _pct(met_sombra["rent_anual"]), _pct(met_sombra["vol_anual"]), _num(met_sombra["sharpe_0"]),
         _num(met_sombra["sharpe_rf"]), _num(met_sombra["sharpe_estres"]), _pct(met_sombra["max_drawdown"])],
    ]
    L.append(_md_tabla(cab, filas))
    sin_datos = sorted(set(met_real["isins_sin_datos"]) | set(met_sombra["isins_sin_datos"]))
    if sin_datos:
        L.append("")
        L.append(f"> Sin precio en la ventana (se trata como rentabilidad 0 en las métricas de arriba): "
                 f"{', '.join(sin_datos)}.")
    L.append("")

    L.append("## Exposición por tipo de activo")
    idx = exp_sombra.index.union(exp_real.index)
    orden = (exp_sombra.reindex(idx, fill_value=0) + exp_real.reindex(idx, fill_value=0)).sort_values(ascending=False).index
    filas = [[t, _pct(exp_sombra.get(t, 0.0)), _pct(exp_real.get(t, 0.0))] for t in orden]
    L.append(_md_tabla(["Tipo", "Sombra", "Real"], filas))
    L.append("")
    L.append(f"Peso del bloque de crédito (`{FACTOR_CREDITO}`, análisis de estilo): "
             f"sombra {_pct(credito_sombra)}, real {_pct(credito_real)}.")
    L.append("")

    L.append("## Backtest")
    if resumen_bt is not None and not resumen_bt.empty:
        r = resumen_bt.round(3)
        cab = [r.index.name or "método"] + list(r.columns)
        filas = [[idx] + [str(v) for v in fila] for idx, fila in r.iterrows()]
        L.append(_md_tabla(cab, filas))
    else:
        L.append("No se ha ejecutado en esta corrida (desactivado con `--sin-backtest` o histórico insuficiente).")
    L.append("")

    L.append("## Alertas")
    if alertas:
        L.extend(f"- {a}" for a in alertas)
    else:
        L.append("Sin alertas.")

    ruta.write_text("\n".join(L) + "\n", encoding="utf-8")
    return n_fuera


# ----------------------------------------------------------------------------
# metodologia.md
# ----------------------------------------------------------------------------
def _escribir_metodologia(ruta: Path, cfg: dict) -> None:
    mcfg, fcfg, scfg, ccfg = cfg["metricas"], cfg["filtros"], cfg["seleccion"], cfg["cartera"]
    psr_txt = f", PSR < {fcfg['psr_minimo']}" if fcfg.get("psr_minimo", 0) > 0 else ""
    diversidad_txt = ("Se fuerza que haya al menos un fondo de cada tipo de activo presente en el "
                      "universo de candidatos." if scfg.get("forzar_diversidad_tipos")
                      else "No se fuerza diversidad de tipos de activo entre los seleccionados.")

    rf_txt = (f"Tipo libre de riesgo: €STR diario del BCE compuesto a la frecuencia del pipeline "
              f"(`rf_fuente = \"estr\"`; si falla la descarga se usa el {_pct(mcfg['tipo_libre_riesgo'])} "
              f"fijo)." if mcfg.get("rf_fuente") == "estr" else
              f"Tipo libre de riesgo fijo: {_pct(mcfg['tipo_libre_riesgo'])} anual (`rf_fuente = \"fijo\"`).")

    suelo_txt = (f"un suelo de pesos por contribución marginal al Sharpe (`suelo = \"marginal\"`: sin "
                f"peso_minimo fijo, se elimina iterativamente el fondo con "
                f"Sharpe_i − ρ(i,cartera)·Sharpe_cartera negativo, sin bajar de "
                f"{ccfg.get('minimo_fondos', 8)} fondos)" if ccfg.get("suelo") == "marginal"
                else f"un suelo de {_pct(ccfg['peso_minimo'])} por fondo (`suelo = \"fijo\"`)")

    credito_txt = (f"Techo al bloque de renta fija de crédito: {_pct(ccfg.get('peso_maximo_credito', 0.5))} "
                  f"del total (subcategoría o estilo RF Corporativa EUR ≥ "
                  f"{_pct(ccfg.get('credito_umbral_estilo', 0.3))}), con el exceso redistribuido "
                  f"proporcionalmente." if ccfg.get("limitar_credito") else None)
    estres_txt = (f"Correlación de estrés: la correlación usada es max(normal, {ccfg.get('estres_inicio', '')} "
                 f"→ {ccfg.get('estres_fin', '')}) elemento a elemento." if ccfg.get("usar_corr_estres") else None)
    multiventana_txt = (f"Ranking por PSR multi-ventana: media del PSR a "
                       f"{', '.join(str(a) for a in scfg.get('ventanas_psr', []))} años, frente a un Sharpe "
                       f"anual de referencia de {scfg.get('psr_referencia', 0.5)} (no 0)."
                       if scfg.get("ranking_multiventana") else None)
    histeresis_txt = (f"Histéresis: un fondo ya en cartera mantiene su cluster salvo que el mejor candidato "
                      f"le supere en {scfg.get('margen_histeresis', 0.05)} de puntuación (solo en backtest)."
                      if scfg.get("usar_histeresis") else None)
    anomalos_txt = ("Se excluyen series con volatilidad anual > 200 % o alguna rentabilidad semanal "
                    "con |r| > 50 % (`excluir_anomalos = true`)." if fcfg.get("excluir_anomalos") else None)
    extra = [t for t in (credito_txt, estres_txt, multiventana_txt, histeresis_txt, anomalos_txt) if t]

    L = [
        "# Metodología",
        "",
        f"Última actualización: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Proceso",
        "",
        f"1. **Universo y datos**: se descargan precios diarios de los fondos del universo "
        f"(fuente configurada: `{cfg['datos']['fuente']}`) y de los índices de estilo, convertidos "
        f"a la moneda base (`{cfg['datos']['moneda_base']}`).",
        f"2. **Rentabilidades**: los precios se agregan a frecuencia `{mcfg['frecuencia']}` "
        f"({M.periodos_por_anio(mcfg['frecuencia'])} periodos/año) para evitar el desfase horario "
        f"entre mercados que distorsiona las correlaciones con datos diarios.",
        f"3. **Selección**, sobre la ventana de los últimos {mcfg['ventana_anios']} años:",
        f"   - Se excluyen fondos con cobertura de datos por debajo de {_pct(mcfg['cobertura_minima'])} "
        f"o con historial insuficiente.",
        f"   - Filtros de riesgo: volatilidad anual fuera de [{_pct(fcfg['volatilidad_minima'])}, "
        f"{_pct(fcfg['volatilidad_maxima'])}], caída máxima superior a {_pct(fcfg['drawdown_maximo'])}{psr_txt}.",
        f"   - Se eliminan \"duplicados\": fondos con correlación ≥ {scfg['correlacion_duplicados']} "
        f"entre sí (enlace completo, para que todos los pares del grupo cumplan el umbral).",
        f"   - Clustering jerárquico (enlace `{ccfg.get('enlace_hrp', 'ward')}`) en "
        f"{scfg['numero_fondos']} grupos; de cada grupo se elige el mejor fondo según "
        f"`{scfg['criterio_ranking']}`.",
        f"   - {diversidad_txt}",
        f"   - {rf_txt}",
        f"4. **Construcción de la cartera**: método `{ccfg['metodo']}` (HRP: pesos jerárquicos sobre "
        f"la matriz de correlación, con shrinkage de Ledoit-Wolf en la covarianza), con {suelo_txt} "
        f"y un techo de {_pct(ccfg['peso_maximo'])} por fondo, redondeado a "
        f"{ccfg['decimales_peso']} decimal(es) en %.",
        f"5. **Backtest walk-forward** (si `backtest.activar = true`): en cada fecha de rebalanceo "
        f"(`{cfg['backtest']['rebalanceo']}`) se repite selección + pesos solo con datos disponibles "
        f"hasta esa fecha, se mantiene la cartera hasta el siguiente rebalanceo y se compara contra "
        f"los métodos {cfg['backtest']['metodos']} y 1/N sobre todo el universo elegible. Se reporta "
        f"el error estándar del Sharpe (Lo, 2002) con IC 95 % y un test de Jobson-Korkie/Memmel de "
        f"cada método frente a HRP; cada ejecución añade una fila a `registro_pruebas.csv`.",
        *(["6. **Otros controles activos sobre la cartera**:"] + [f"   - {t}" for t in extra] if extra else []),
        "",
        "## Parámetros vigentes (config.toml)",
        "",
    ]
    params = configuracion_plana(cfg).reset_index()
    L.append(_md_tabla(["Parámetro", "Valor"], params.values.tolist()))
    ruta.write_text("\n".join(L) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------
# diario_decisiones.md
# ----------------------------------------------------------------------------
_PLANTILLA_DIARIO = """# Diario de decisiones

Registro manual de decisiones sobre la cartera (rebalanceos manuales, cambios de parámetros,
exclusiones ad hoc...). Añade una fila por decisión importante. Este fichero no se sobrescribe.

| Fecha | Decisión | Motivo | Qué la desharía | Resultado posterior |
|---|---|---|---|---|
| | | | | |

## Registro automático de ejecuciones
"""


def _actualizar_diario(ruta: Path, n_fuera_banda: int) -> None:
    if not ruta.exists():
        ruta.write_text(_PLANTILLA_DIARIO, encoding="utf-8")
    linea = f"- {dt.date.today().isoformat()} – ejecución del pipeline: {n_fuera_banda} fondos fuera de banda\n"
    with open(ruta, "a", encoding="utf-8") as f:
        f.write(linea)


# ----------------------------------------------------------------------------
# Entrada
# ----------------------------------------------------------------------------
def generar(cfg: dict, raiz: Path, salida: Path, cartera: pd.DataFrame, tabla: pd.DataFrame,
           info: pd.DataFrame, ventana: pd.DataFrame, periodos: int,
           resumen_bt: pd.DataFrame) -> Path:
    carpeta = salida / "proyecto"
    carpeta.mkdir(parents=True, exist_ok=True)
    sombra = cartera["peso"]

    ruta_real = raiz / NOMBRE_FICHERO_REAL
    real, creada = cargar_cartera_real(ruta_real, sombra)

    n_fuera = _escribir_estado(carpeta / "estado_cartera.md", cfg, cartera, tabla, info, ventana,
                               real, sombra, creada, periodos, resumen_bt)
    _escribir_metodologia(carpeta / "metodologia.md", cfg)
    _actualizar_diario(carpeta / "diario_decisiones.md", n_fuera)

    print(f"   {carpeta / 'estado_cartera.md'}")
    print(f"   {carpeta / 'metodologia.md'}")
    print(f"   {carpeta / 'diario_decisiones.md'}")
    if creada:
        print(f"   [aviso] No existía {ruta_real}: se ha creado igual que la cartera sombra.")
    return carpeta
