"""
Estimación de covarianzas y métodos de construcción de cartera.

  - Ledoit y Wolf (2004): shrinkage de la matriz (aquí sobre la matriz de correlaciones,
    reescalada después con las volatilidades muestrales).
  - López de Prado (2016), "Building Diversified Portfolios that Outperform Out of Sample":
    Hierarchical Risk Parity (HRP).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform


# ----------------------------------------------------------------------------
# Covarianza
# ----------------------------------------------------------------------------
def ledoit_wolf_correlacion(rend: pd.DataFrame, rend_estres: pd.DataFrame | None = None,
                            usar_corr_estres: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Shrinkage de Ledoit-Wolf de la correlación hacia la identidad.
    Devuelve (covarianza, correlación, intensidad de shrinkage).

    Si usar_corr_estres=True y se pasa rend_estres (rentabilidades de los mismos
    fondos en un periodo de estrés, p. ej. 2022), la correlación final es
    corr = max(corr_normal, corr_estres) elemento a elemento: la correlación no
    puede ser más baja que en un episodio de estrés conocido. Un fondo sin datos en
    ese periodo conserva su correlación normal. Las volatilidades (y por tanto la
    covarianza) siguen viniendo de la ventana normal."""
    r = rend.copy()
    sd = r.std(ddof=1)
    z = ((r - r.mean()) / sd).fillna(0.0)   # huecos = valor medio (impacto mínimo)
    X = z.values
    n, p = X.shape
    S = X.T @ X / n
    mu = np.trace(S) / p
    delta2 = np.sum((S - mu * np.eye(p)) ** 2) / p
    beta2 = sum(np.sum((np.outer(x, x) - S) ** 2) for x in X) / (n ** 2 * p)
    beta2 = min(beta2, delta2)
    intensidad = beta2 / delta2 if delta2 > 0 else 1.0
    C = intensidad * mu * np.eye(p) + (1 - intensidad) * S
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)                   # re-normaliza a correlación
    corr = pd.DataFrame(C, index=rend.columns, columns=rend.columns)

    if usar_corr_estres and rend_estres is not None and not rend_estres.empty:
        cols = [c for c in rend.columns if c in rend_estres.columns]
        if cols:
            corr_estres = rend_estres[cols].corr(min_periods=20)
            corr_estres = corr_estres.reindex(index=rend.columns, columns=rend.columns)
            valores = np.array(corr_estres.values, copy=True)
            np.fill_diagonal(valores, 1.0)
            corr_estres = pd.DataFrame(valores, index=rend.columns, columns=rend.columns)
            corr_estres = corr_estres.fillna(corr)   # sin datos de estrés -> correlación normal
            corr = np.maximum(corr, corr_estres)

    cov = corr * np.outer(sd.values, sd.values)
    return cov, corr, float(intensidad)


def distancia_correlacion(corr: pd.DataFrame) -> pd.DataFrame:
    """d = sqrt(0.5 (1 - ρ)) ∈ [0, 1]. Proporcional a la distancia euclídea entre
    series estandarizadas, así que es válida también con enlace de Ward."""
    d = np.sqrt(np.clip(0.5 * (1 - corr.values), 0, 1))
    np.fill_diagonal(d, 0)
    return pd.DataFrame(d, index=corr.index, columns=corr.columns)


def enlazar(corr: pd.DataFrame, metodo: str = "ward"):
    d = distancia_correlacion(corr)
    return linkage(squareform(d.values, checks=False), method=metodo)


# ----------------------------------------------------------------------------
# Métodos de pesos
# ----------------------------------------------------------------------------
def pesos_hrp(cov: pd.DataFrame, corr: pd.DataFrame, metodo_enlace: str = "ward") -> pd.Series:
    if len(cov) == 1:
        return pd.Series(1.0, index=cov.index)
    enlace = enlazar(corr, metodo_enlace)
    orden = list(cov.index[leaves_list(enlace)])
    w = pd.Series(1.0, index=orden)
    grupos = [orden]
    while grupos:
        nuevos = []
        for g in grupos:
            if len(g) <= 1:
                continue
            mitad = len(g) // 2
            izq, der = g[:mitad], g[mitad:]
            v_izq, v_der = _var_cluster(cov, izq), _var_cluster(cov, der)
            alfa = 1 - v_izq / (v_izq + v_der)
            w[izq] *= alfa
            w[der] *= 1 - alfa
            nuevos += [izq, der]
        grupos = nuevos
    return w.reindex(cov.index)


def _var_cluster(cov: pd.DataFrame, items: list) -> float:
    c = cov.loc[items, items].values
    ivp = 1 / np.diag(c)
    ivp /= ivp.sum()
    return float(ivp @ c @ ivp)


def pesos_inversa_vol(cov: pd.DataFrame) -> pd.Series:
    iv = 1 / np.sqrt(np.diag(cov.values))
    return pd.Series(iv / iv.sum(), index=cov.index)


def pesos_equiponderada(cov: pd.DataFrame) -> pd.Series:
    return pd.Series(1 / len(cov), index=cov.index)


def pesos_min_var(cov: pd.DataFrame, peso_maximo: float) -> pd.Series:
    n = len(cov)
    S = cov.values
    res = minimize(lambda w: w @ S @ w, np.full(n, 1 / n), method="SLSQP",
                   bounds=[(0, peso_maximo)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"ftol": 1e-12, "maxiter": 1000})
    return pd.Series(np.clip(res.x, 0, None), index=cov.index)


def pesos_max_sharpe(cov: pd.DataFrame, exceso_medio: pd.Series, peso_maximo: float,
                     shrink_media: float = 0.5) -> pd.Series:
    """Máximo Sharpe con medias 'encogidas' hacia la media común (las medias históricas
    son el input más ruidoso de todos). Si ninguna media es positiva -> mínima varianza."""
    mu = (1 - shrink_media) * exceso_medio + shrink_media * exceso_medio.mean()
    mu = mu.reindex(cov.index).fillna(0).values
    if np.all(mu <= 0):
        return pesos_min_var(cov, peso_maximo)
    n = len(cov)
    S = cov.values

    def neg_sharpe(w):
        v = w @ S @ w
        return -(w @ mu) / np.sqrt(v) if v > 0 else 0.0

    res = minimize(neg_sharpe, np.full(n, 1 / n), method="SLSQP",
                   bounds=[(0, peso_maximo)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"ftol": 1e-12, "maxiter": 1000})
    return pd.Series(np.clip(res.x, 0, None), index=cov.index)


def frontera_eficiente(cov: pd.DataFrame, mu: pd.Series, n_puntos: int = 60) -> pd.DataFrame:
    """Frontera eficiente long-only (mínima varianza para cada nivel de rentabilidad objetivo)
    sobre un conjunto de activos dados su covarianza y su rentabilidad esperada, ambas en la
    misma escala temporal (p. ej. ya anualizadas). Devuelve columnas 'vol' y 'ret'."""
    n = len(cov)
    S = cov.values
    m = mu.reindex(cov.index).values

    def vol_min(objetivo: float) -> float:
        res = minimize(lambda w: w @ S @ w, np.full(n, 1 / n), method="SLSQP",
                       bounds=[(0, 1)] * n,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1},
                                    {"type": "eq", "fun": lambda w: w @ m - objetivo}],
                       options={"ftol": 1e-12, "maxiter": 1000})
        return float(np.sqrt(max(res.fun, 0))) if res.success else np.nan

    ret_gmv = float(pesos_min_var(cov, 1.0).values @ m)
    ret_max = float(m.max())
    objetivos = np.linspace(ret_gmv, ret_max, n_puntos)
    vols = [vol_min(t) for t in objetivos]
    return pd.DataFrame({"vol": vols, "ret": objetivos}).dropna()


def es_bloque_credito(subcategoria: pd.Series, estilo_credito: pd.Series,
                      credito_palabras: list[str], umbral_estilo: float = 0.30) -> pd.Series:
    """True para los fondos que pertenecen al bloque de renta fija de crédito: su
    subcategoria_universo contiene alguna de credito_palabras, o su peso de estilo en
    "RF Corporativa EUR" es >= umbral_estilo."""
    idx = subcategoria.index.union(estilo_credito.index)
    sub = subcategoria.reindex(idx).fillna("").astype(str).str.lower()
    patron = "|".join(p.lower() for p in credito_palabras)
    por_palabra = sub.str.contains(patron, regex=True) if patron else pd.Series(False, index=idx)
    por_estilo = estilo_credito.reindex(idx).fillna(0.0) >= umbral_estilo
    return (por_palabra | por_estilo).reindex(idx)


def limitar_bloque_credito(w: pd.Series, es_credito: pd.Series, peso_maximo_credito: float) -> pd.Series:
    """Si el peso conjunto del bloque de crédito supera peso_maximo_credito, lo reescala
    a ese máximo y reparte el exceso proporcionalmente entre el resto de fondos."""
    es_credito = es_credito.reindex(w.index).fillna(False)
    peso_credito = float(w[es_credito].sum())
    if peso_credito <= peso_maximo_credito or peso_credito <= 0:
        return w
    resto = w[~es_credito]
    peso_resto = float(resto.sum())
    w = w.copy()
    w[es_credito] *= peso_maximo_credito / peso_credito
    exceso = peso_credito - peso_maximo_credito
    if peso_resto > 0:
        w[~es_credito] *= (peso_resto + exceso) / peso_resto
    return w / w.sum()


def _sharpe_periodo(r: pd.Series, rf_periodo) -> float:
    rf = rf_periodo if np.isscalar(rf_periodo) else rf_periodo.reindex(r.index)
    exceso = (r - rf).dropna()
    sd = exceso.std(ddof=1)
    return float(exceso.mean() / sd) if sd and np.isfinite(sd) and sd > 0 else np.nan


def _pesos_base(metodo: str, rend: pd.DataFrame, rf_periodo, cfg_cartera: dict,
                credito_info: pd.DataFrame | None, rend_estres: pd.DataFrame | None,
                peso_minimo: float) -> pd.Series:
    usar_estres = cfg_cartera.get("usar_corr_estres", False)
    cov, corr, _ = ledoit_wolf_correlacion(rend, rend_estres, usar_estres)
    pmax = cfg_cartera["peso_maximo"]
    rf_medio = rf_periodo if np.isscalar(rf_periodo) else rf_periodo.reindex(rend.index)
    if metodo == "hrp":
        w = pesos_hrp(cov, corr, cfg_cartera.get("enlace_hrp", "ward"))
    elif metodo == "inversa_vol":
        w = pesos_inversa_vol(cov)
    elif metodo == "equiponderada":
        w = pesos_equiponderada(cov)
    elif metodo == "min_var":
        w = pesos_min_var(cov, max(pmax, 1 / len(cov)))
    elif metodo == "max_sharpe":
        w = pesos_max_sharpe(cov, (rend.sub(rf_medio, axis=0)).mean(), max(pmax, 1 / len(cov)))
    else:
        raise ValueError(f"Método desconocido: {metodo}")
    if metodo != "equiponderada":
        w = aplicar_limites(w, pmax, peso_minimo)
    if metodo != "equiponderada" and cfg_cartera.get("limitar_credito", False) \
            and credito_info is not None and len(credito_info):
        es_credito = es_bloque_credito(
            credito_info["subcategoria_universo"].reindex(w.index),
            credito_info.get("estilo_RF Corporativa EUR", pd.Series(dtype=float)).reindex(w.index),
            cfg_cartera.get("credito_palabras", []),
            cfg_cartera.get("credito_umbral_estilo", 0.30),
        )
        w = limitar_bloque_credito(w, es_credito, cfg_cartera.get("peso_maximo_credito", 0.50))
    return w


def aplicar_suelo_marginal(metodo: str, rend: pd.DataFrame, rf_periodo, cfg_cartera: dict,
                           credito_info: pd.DataFrame | None, rend_estres: pd.DataFrame | None,
                           minimo_fondos: int) -> pd.Series:
    """Elimina iterativamente el fondo con menor contribución marginal al Sharpe de la
    cartera (Sharpe_i − ρ(i, cartera)·Sharpe_cartera), mientras sea negativa y no se
    baje de minimo_fondos, recalculando los pesos (con peso_minimo = 0) cada vez."""
    fondos = list(rend.columns)
    w = _pesos_base(metodo, rend[fondos], rf_periodo, cfg_cartera, credito_info, rend_estres, 0.0)
    while len(fondos) > minimo_fondos:
        r_cartera = rend[fondos].fillna(0.0) @ w.reindex(fondos).fillna(0.0)
        sharpe_cartera = _sharpe_periodo(r_cartera, rf_periodo)
        peor, peor_valor = None, 0.0
        for i in fondos:
            sharpe_i = _sharpe_periodo(rend[i], rf_periodo)
            rho_i = rend[i].corr(r_cartera)
            if not np.isfinite(sharpe_i) or not np.isfinite(rho_i):
                continue
            score = sharpe_i - rho_i * sharpe_cartera
            if score < peor_valor:
                peor_valor, peor = score, i
        if peor is None:
            break
        fondos.remove(peor)
        w = _pesos_base(metodo, rend[fondos], rf_periodo, cfg_cartera, credito_info, rend_estres, 0.0)
    return w


def calcular_pesos(metodo: str, rend: pd.DataFrame, rf_periodo: float | pd.Series, cfg_cartera: dict,
                   credito_info: pd.DataFrame | None = None,
                   rend_estres: pd.DataFrame | None = None) -> pd.Series:
    if metodo != "equiponderada" and cfg_cartera.get("suelo") == "marginal":
        return aplicar_suelo_marginal(metodo, rend, rf_periodo, cfg_cartera, credito_info,
                                      rend_estres, cfg_cartera.get("minimo_fondos", 8))
    peso_minimo = cfg_cartera.get("peso_minimo", 0.0)
    return _pesos_base(metodo, rend, rf_periodo, cfg_cartera, credito_info, rend_estres, peso_minimo)


# ----------------------------------------------------------------------------
# Restricciones y redondeo
# ----------------------------------------------------------------------------
def aplicar_limites(w: pd.Series, peso_maximo: float, peso_minimo: float) -> pd.Series:
    """Aplica un suelo y un techo a TODOS los fondos seleccionados manteniendo, dentro de
    los límites, las proporciones del método:  w_i = clip(c · w_i, mínimo, máximo)  con c
    tal que Σ w_i = 1 (se resuelve por bisección; la suma es monótona en c).

    El suelo evita que un método de paridad de riesgo deje fuera a los fondos más volátiles
    (típicamente toda la renta variable) tras haberlos seleccionado por diversificación."""
    w = w.clip(lower=0).astype(float)
    w = w / w.sum()
    n = len(w)
    if n * peso_maximo < 1:
        warnings.warn(f"Con {n} fondos el peso máximo {peso_maximo:.0%} es imposible; "
                      f"se sube a {1 / n:.1%}.")
        peso_maximo = 1 / n
    if n * peso_minimo > 1:
        warnings.warn(f"Con {n} fondos el peso mínimo {peso_minimo:.0%} es imposible; "
                      f"se baja a {1 / n:.1%}.")
        peso_minimo = 1 / n

    def suma(c):
        return np.clip(c * w.values, peso_minimo, peso_maximo).sum()

    bajo, alto = 0.0, 1.0
    while suma(alto) < 1 and alto < 1e12:
        alto *= 2
    for _ in range(200):
        medio = (bajo + alto) / 2
        if suma(medio) < 1:
            bajo = medio
        else:
            alto = medio
    res = pd.Series(np.clip(alto * w.values, peso_minimo, peso_maximo), index=w.index)
    return res / res.sum()


def redondear_pesos(w: pd.Series, decimales: int = 1) -> pd.Series:
    """Redondeo en % por el método del mayor resto, para que sume exactamente 100."""
    unidad = 10 ** decimales
    bruto = w / w.sum() * 100 * unidad
    base = np.floor(bruto)
    faltan = int(round(100 * unidad - base.sum()))
    orden = (bruto - base).sort_values(ascending=False).index[:faltan]
    base[orden] += 1
    return base / unidad


def contribucion_riesgo(w: pd.Series, cov: pd.DataFrame) -> pd.Series:
    w = w.reindex(cov.index).fillna(0)
    marginal = cov.values @ w.values
    rc = w.values * marginal
    return pd.Series(rc / rc.sum(), index=cov.index)
