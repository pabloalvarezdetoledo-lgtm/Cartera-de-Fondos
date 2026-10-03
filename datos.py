"""
Descarga y caché de datos de fondos.

Cada fuente devuelve, por fondo:
  - una serie diaria de precios de rentabilidad total (dividendos reinvertidos si la
    fuente lo permite) en la moneda del fondo
  - un diccionario de información: nombre, moneda, categoría, composición, TER…

Todo se convierte después a la moneda base (EUR por defecto).
"""
from __future__ import annotations

import pickle
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

INFO_VACIA = {
    "nombre": None,
    "moneda": None,
    "categoria": None,
    "pct_rv": np.nan,
    "pct_rf": np.nan,
    "pct_liquidez": np.nan,
    "pct_otros": np.nan,
    "ter": np.nan,
    "id_fuente": None,
}


@dataclass
class DatosFondo:
    isin: str
    precios: pd.Series            # diaria, moneda del fondo
    info: dict = field(default_factory=dict)
    error: str | None = None


# ----------------------------------------------------------------------------
# Universo
# ----------------------------------------------------------------------------
def cargar_universo(ruta: str | Path) -> pd.DataFrame:
    """Lee el CSV del universo. Columnas: isin (obligatoria), nombre, ticker (opcionales)."""
    df = pd.read_csv(ruta, dtype=str, comment="#", skip_blank_lines=True)
    df.columns = [c.strip().lower() for c in df.columns]
    if "isin" not in df.columns:
        raise ValueError("El universo necesita una columna 'isin'.")
    for c in ("nombre", "ticker"):
        if c not in df.columns:
            df[c] = None
    df["isin"] = df["isin"].str.strip().str.upper()
    df = df[df["isin"].str.len() == 12].drop_duplicates("isin").reset_index(drop=True)
    return df   # columnas extra (categoría, gestora, ogc…) se conservan como información


# columnas opcionales del universo que se copian a la información del fondo
EXTRA_UNIVERSO = {"gestora": "gestora", "categoria": "categoria_universo",
                  "subcategoria": "subcategoria_universo", "ogc": "ogc_universo",
                  "inversion_minima": "inversion_minima", "divisa": "divisa_universo"}


def _texto(x) -> str | None:
    """Celda de texto o None (un NaN sería 'verdadero' en un `or`)."""
    return x.strip() if isinstance(x, str) and x.strip() else None


# ----------------------------------------------------------------------------
# Caché
# ----------------------------------------------------------------------------
class Cache:
    def __init__(self, carpeta: str | Path, dias_validez: int):
        self.carpeta = Path(carpeta)
        self.carpeta.mkdir(parents=True, exist_ok=True)
        self.validez = timedelta(days=dias_validez)

    def _ruta(self, clave: str) -> Path:
        return self.carpeta / f"{clave.replace('/', '_').replace('=', '_')}.pkl"

    def leer(self, clave: str):
        ruta = self._ruta(clave)
        if not ruta.exists():
            return None
        with open(ruta, "rb") as f:
            obj = pickle.load(f)
        if datetime.now() - obj["fecha"] > self.validez:
            return None
        return obj["datos"]

    def guardar(self, clave: str, datos) -> None:
        with open(self._ruta(clave), "wb") as f:
            pickle.dump({"fecha": datetime.now(), "datos": datos}, f)


# ----------------------------------------------------------------------------
# Fuente Yahoo Finance (yfinance)
# ----------------------------------------------------------------------------
class FuenteYahoo:
    """Busca el fondo por ISIN en Yahoo (los fondos europeos suelen tener tickers tipo
    0P0000XXXX.F) y descarga precios ajustados. Composición vía funds_data."""

    nombre = "yahoo"

    def __init__(self, detalle: bool = False):
        import yfinance as yf  # noqa: F401  (se importa aquí para no exigirlo si no se usa)
        self.yf = yf
        self.detalle = detalle   # True = también ficha y composición (2 peticiones más por fondo)

    def resolver_ticker(self, isin: str) -> str | None:
        try:
            res = self.yf.Search(isin, max_results=10, news_count=0, lists_count=0,
                                 raise_errors=True).quotes
        except Exception as e:
            raise ConnectionError(f"búsqueda en Yahoo fallida: {e}") from e
        preferidos = [q for q in res if q.get("quoteType") in ("MUTUALFUND", "ETF")]
        candidatos = preferidos or res
        return candidatos[0]["symbol"] if candidatos else None

    def descargar(self, isin: str, ticker: str | None, inicio: datetime, fin: datetime) -> DatosFondo:
        ticker = ticker or self.resolver_ticker(isin)
        if not ticker:
            return DatosFondo(isin, pd.Series(dtype=float), dict(INFO_VACIA), "ISIN no encontrado en Yahoo")
        t = self.yf.Ticker(ticker)
        hist = t.history(start=inicio, end=fin, auto_adjust=True, actions=False)
        if hist is None or hist.empty:
            return DatosFondo(isin, pd.Series(dtype=float), dict(INFO_VACIA), f"sin histórico ({ticker})")
        precios = hist["Close"].copy()
        precios.index = pd.to_datetime(precios.index).tz_localize(None).normalize()

        info = dict(INFO_VACIA)
        info["id_fuente"] = ticker
        try:  # la divisa viene con el histórico, sin petición extra
            info["moneda"] = (t.history_metadata or {}).get("currency")
        except Exception:
            pass
        if not self.detalle:
            if info["moneda"] in ("GBp", "GBX"):
                precios = precios / 100.0
                info["moneda"] = "GBP"
            return DatosFondo(isin, precios, info)
        try:
            meta = t.info or {}
            info["nombre"] = meta.get("longName") or meta.get("shortName")
            info["moneda"] = meta.get("currency") or info["moneda"]
            info["categoria"] = meta.get("category")
            ter = meta.get("annualReportExpenseRatio") or meta.get("netExpenseRatio")
            info["ter"] = float(ter) if ter is not None else np.nan
        except Exception:
            pass
        try:
            ac = t.funds_data.asset_classes or {}
            info["pct_rv"] = _num(ac.get("stockPosition"))
            info["pct_rf"] = _num(ac.get("bondPosition"))
            info["pct_liquidez"] = _num(ac.get("cashPosition"))
            otros = [ac.get(k) for k in ("preferredPosition", "convertiblePosition", "otherPosition")]
            info["pct_otros"] = np.nansum([_num(x) for x in otros])
        except Exception:
            pass

        # Yahoo cotiza algunos fondos británicos en peniques
        if info["moneda"] in ("GBp", "GBX"):
            precios = precios / 100.0
            info["moneda"] = "GBP"
        return DatosFondo(isin, precios, info)

    def serie_ticker(self, ticker: str, inicio: datetime, fin: datetime) -> pd.Series:
        hist = self.yf.Ticker(ticker).history(start=inicio, end=fin, auto_adjust=True, actions=False)
        if hist is None or hist.empty:
            return pd.Series(dtype=float)
        s = hist["Close"].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        return s


# ----------------------------------------------------------------------------
# Fuente Morningstar (mstarpy, no oficial)
# ----------------------------------------------------------------------------
class FuenteMorningstar:
    """mstarpy >= 11 abre un Chrome sin interfaz (Selenium) para obtener las cookies de
    Morningstar. Necesita Google Chrome instalado. Se reutiliza una sola sesión."""

    nombre = "morningstar"

    def __init__(self, idioma: str = "es"):
        import mstarpy
        from mstarpy.search import MorningstarSession
        self.mstarpy = mstarpy
        self.idioma = idioma
        print("Iniciando sesión en Morningstar (abre un Chrome invisible, tarda unos segundos)…")
        self.sesion = MorningstarSession()

    def descargar(self, isin: str, ticker: str | None, inicio: datetime, fin: datetime) -> DatosFondo:
        try:
            f = self.mstarpy.Funds(isin, language=self.idioma, session=self.sesion)
        except Exception as e:
            return DatosFondo(isin, pd.Series(dtype=float), dict(INFO_VACIA), f"no encontrado: {e}")

        try:
            serie = f.nav(start_date=inicio, end_date=fin, frequency="daily")
        except Exception as e:
            return DatosFondo(isin, pd.Series(dtype=float), dict(INFO_VACIA), f"error NAV: {e}")
        precios = _serie_desde_registros(serie)
        if precios.empty:
            return DatosFondo(isin, precios, dict(INFO_VACIA), "sin histórico")

        info = dict(INFO_VACIA)
        info["nombre"] = getattr(f, "name", None)
        info["id_fuente"] = getattr(f, "code", None)
        try:
            dp = f.dataPoint(["name", "baseCurrency", "morningstarCategory", "ongoingCharge"])
            campos = dp[0] if isinstance(dp, list) and dp else {}
            campos = campos.get("fields", campos)
            info["moneda"] = _valor(campos.get("baseCurrency"))
            info["categoria"] = _valor(campos.get("morningstarCategory"))
            ter = _num(_valor(campos.get("ongoingCharge")))
            info["ter"] = ter / 100 if ter == ter and ter > 1 else ter
        except Exception:
            pass
        try:
            info.update(_parsear_asignacion_morningstar(f.allocationMap()))
        except Exception:
            pass
        return DatosFondo(isin, precios, info)


def _valor(x):
    if isinstance(x, dict):
        return x.get("value", x.get("name"))
    return x


def _serie_desde_registros(registros) -> pd.Series:
    """Convierte la lista de dicts de mstarpy en una serie. Prioriza 'totalReturn'
    (con dividendos) sobre 'nav'. Tolera cambios de formato del proveedor."""
    if not registros:
        return pd.Series(dtype=float)
    df = pd.DataFrame(registros)
    col_fecha = next((c for c in df.columns if c.lower() in ("date", "fecha")), None)
    if col_fecha is None:
        return pd.Series(dtype=float)
    col_valor = next((c for c in ("totalReturn", "nav", "value", "price") if c in df.columns), None)
    if col_valor is None:
        return pd.Series(dtype=float)
    s = pd.Series(pd.to_numeric(df[col_valor], errors="coerce").values,
                  index=pd.to_datetime(df[col_fecha]).dt.normalize())
    return s.dropna().sort_index()


def _parsear_asignacion_morningstar(d: dict) -> dict:
    """Busca de forma defensiva los porcentajes de RV / RF / liquidez / otros."""
    out = {}
    mapa = d.get("allocationMap", d) if isinstance(d, dict) else {}
    buscar = {
        "pct_rv": ("AssetAllocEquity", "AssetAllocNonUSEquity", "AssetAllocUSEquity"),
        "pct_rf": ("AssetAllocBond",),
        "pct_liquidez": ("AssetAllocCash",),
        "pct_otros": ("AssetAllocOther", "AssetAllocNotClassified"),
    }
    for destino, claves in buscar.items():
        total, encontrado = 0.0, False
        for k in claves:
            if k in mapa:
                v = mapa[k]
                v = v.get("netAllocation", v.get("longAllocation")) if isinstance(v, dict) else v
                v = _num(v)
                if v == v:
                    total += v
                    encontrado = True
        # evita contar dos veces Equity total + Equity US/NonUS
        if destino == "pct_rv" and "AssetAllocEquity" in mapa:
            v = mapa["AssetAllocEquity"]
            total = _num(v.get("netAllocation") if isinstance(v, dict) else v)
        if encontrado:
            out[destino] = total
    return out


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


# ----------------------------------------------------------------------------
# Fuente CSV (precios propios)
# ----------------------------------------------------------------------------
class FuenteCSV:
    """Lee un CSV ancho: primera columna fecha, una columna por ISIN con precios de
    rentabilidad total YA EN EUR."""

    nombre = "csv"

    def __init__(self, ruta: str | Path, moneda_base: str = "EUR"):
        df = pd.read_csv(ruta, index_col=0, parse_dates=True)
        df.columns = [c.strip().upper() for c in df.columns]
        self.df = df.sort_index()
        self.moneda = moneda_base

    def descargar(self, isin, ticker, inicio, fin) -> DatosFondo:
        if isin not in self.df.columns:
            return DatosFondo(isin, pd.Series(dtype=float), dict(INFO_VACIA), "no está en el CSV")
        s = self.df[isin].dropna()
        info = dict(INFO_VACIA, moneda=self.moneda)
        return DatosFondo(isin, s[(s.index >= inicio) & (s.index <= fin)], info)


# ----------------------------------------------------------------------------
# €STR (tipo libre de riesgo diario, BCE)
# ----------------------------------------------------------------------------
URL_ESTR = "https://data-api.ecb.europa.eu/service/data/EST/B.EU000A2X2A25.WT?format=csvdata"


def descargar_estr(cfg: dict) -> pd.Series | None:
    """€STR diario (tipo anual, en fracción) publicado por el BCE. Se cachea igual que
    el resto de descargas; si falla cualquier paso, avisa y devuelve None (el llamador
    debe usar entonces tipo_libre_riesgo fijo)."""
    dcfg = cfg["datos"]
    cache = Cache(dcfg["carpeta_cache"], dcfg["dias_validez_cache"])
    s = cache.leer("estr_diario")
    if s is not None:
        return s
    try:
        import io

        import requests
        texto = requests.get(URL_ESTR, timeout=30).text
        df = pd.read_csv(io.StringIO(texto))
        df.columns = [c.strip().upper() for c in df.columns]
        col_fecha = next(c for c in df.columns if c in ("TIME_PERIOD", "FECHA"))
        col_valor = next(c for c in df.columns if c in ("OBS_VALUE", "VALOR"))
        s = pd.Series(pd.to_numeric(df[col_valor], errors="coerce").values,
                      index=pd.to_datetime(df[col_fecha])).dropna().sort_index()
        s = s / 100.0   # el BCE publica el tipo en %
        if s.empty:
            raise ValueError("CSV del €STR sin datos")
    except Exception as e:
        warnings.warn(f"No se pudo descargar el €STR del BCE ({e}); se usa tipo_libre_riesgo fijo.")
        return None
    cache.guardar("estr_diario", s)
    return s


# ----------------------------------------------------------------------------
# Tipo de cambio
# ----------------------------------------------------------------------------
class ConversorDivisa:
    def __init__(self, fuente_yahoo: FuenteYahoo | None, cache: Cache, moneda_base: str = "EUR"):
        self.yahoo = fuente_yahoo
        self.cache = cache
        self.base = moneda_base
        self._memo: dict[str, pd.Series] = {}

    def a_base(self, precios: pd.Series, moneda: str | None, inicio, fin) -> pd.Series | None:
        """Devuelve precios en moneda base, o None si no se puede convertir."""
        if not moneda or moneda.upper() == self.base:
            return precios
        moneda = moneda.upper()
        fx = self._fx(moneda, inicio, fin)
        if fx is None or fx.empty:
            return None
        # ticker Yahoo EURUSD=X = USD por 1 EUR  ->  precio_EUR = precio_USD / fx
        fx = fx.reindex(precios.index).ffill().bfill()
        return (precios / fx).dropna()

    def _fx(self, moneda, inicio, fin):
        if moneda in self._memo:
            return self._memo[moneda]
        clave = f"fx_{self.base}{moneda}"
        s = self.cache.leer(clave)
        if s is None and self.yahoo is not None:
            try:
                s = self.yahoo.serie_ticker(f"{self.base}{moneda}=X", inicio, fin)
                if not s.empty:
                    self.cache.guardar(clave, s)
            except Exception:
                s = None
        self._memo[moneda] = s
        return s


# ----------------------------------------------------------------------------
# Orquestación
# ----------------------------------------------------------------------------
def crear_fuente(cfg: dict):
    fuente = cfg["datos"]["fuente"].lower()
    if fuente == "yahoo":
        return FuenteYahoo(detalle=cfg["datos"].get("descargar_composicion", False))
    if fuente == "morningstar":
        return FuenteMorningstar()
    if fuente == "csv":
        return FuenteCSV(cfg["datos"]["archivo_precios_csv"], cfg["datos"]["moneda_base"])
    if fuente == "demo":
        from demo import FuenteDemo
        return FuenteDemo()
    raise ValueError(f"Fuente desconocida: {fuente}")


def descargar_universo(universo: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descarga todos los fondos. Devuelve:
      precios: DataFrame diario (columnas = ISIN) en moneda base
      info:    DataFrame con la información y el estado de cada fondo
    """
    dcfg = cfg["datos"]
    fin = datetime.now()
    inicio = fin - timedelta(days=int(365.25 * dcfg["anios_historico"]) + 10)
    cache = Cache(dcfg["carpeta_cache"], dcfg["dias_validez_cache"])
    cache_fallos = Cache(dcfg["carpeta_cache"], dcfg["dias_validez_cache_fallos"])
    fuente = crear_fuente(cfg)

    yahoo_fx = fuente if isinstance(fuente, FuenteYahoo) else None
    if yahoo_fx is None and dcfg["fuente"] not in ("demo", "csv"):
        try:
            yahoo_fx = FuenteYahoo()
        except ImportError:
            warnings.warn("yfinance no instalado: no se podrán convertir fondos en otras divisas.")
    if dcfg["fuente"] == "demo":
        yahoo_fx = fuente  # la fuente demo también simula tipos de cambio
    fx = ConversorDivisa(yahoo_fx, cache, dcfg["moneda_base"])

    series, filas = {}, []
    n = len(universo)
    fallos_conexion = 0
    for i, fila in universo.iterrows():
        isin = fila["isin"]
        clave = f"{fuente.nombre}_{isin}"
        clave_fallo = f"{clave}_fallo"
        datos = cache.leer(clave)
        if datos is None:
            datos = cache_fallos.leer(clave_fallo)
        if datos is None:
            try:
                datos = fuente.descargar(isin, _texto(fila.get("ticker")), inicio, fin)
            except Exception as e:  # nunca parar el pipeline por un fondo
                datos = DatosFondo(isin, pd.Series(dtype=float), dict(INFO_VACIA), f"error: {e}")
                fallos_conexion += isinstance(e, ConnectionError) or "onnect" in str(e)
                if fallos_conexion >= 5 and not series:
                    raise SystemExit("Cinco errores de conexión seguidos sin ninguna descarga correcta: "
                                     "revisa tu conexión a internet o prueba otra fuente.")
            if datos.error is None:
                cache.guardar(clave, datos)
            else:
                cache_fallos.guardar(clave_fallo, datos)
            if fuente.nombre not in ("demo", "csv"):
                time.sleep(dcfg["pausa_entre_descargas"])

        info = {**INFO_VACIA, **(datos.info or {})}
        info["isin"] = isin
        info["nombre"] = info["nombre"] or _texto(fila.get("nombre")) or isin
        for col, destino in EXTRA_UNIVERSO.items():
            if col in fila.index:
                info[destino] = fila.get(col)
        if not info.get("moneda") and _texto(fila.get("divisa")):
            info["moneda"] = fila.get("divisa")
        if not np.isfinite(_num(info.get("ter"))) and np.isfinite(_num(fila.get("ogc"))):
            info["ter"] = _num(fila.get("ogc")) / 100
        if not info.get("categoria") and _texto(fila.get("subcategoria")):
            info["categoria"] = fila.get("subcategoria")
        estado = datos.error
        if estado is None:
            convertido = fx.a_base(datos.precios, info["moneda"], inicio, fin)
            if convertido is None:
                estado = f"sin tipo de cambio {info['moneda']}->{dcfg['moneda_base']}"
            else:
                series[isin] = convertido[~convertido.index.duplicated()]
        info["estado_descarga"] = estado or "ok"
        filas.append(info)
        print(f"  [{i + 1:>4}/{n}] {isin}  {info['estado_descarga'][:60]}")

    precios = pd.DataFrame(series).sort_index()
    info = pd.DataFrame(filas).set_index("isin")
    return precios, info


def descargar_factores_estilo(cfg: dict) -> pd.DataFrame:
    """Series de los índices de estilo (ETFs en EUR)."""
    factores = cfg.get("estilo", {}).get("factores", {})
    if not factores:
        return pd.DataFrame()
    dcfg = cfg["datos"]
    fin = datetime.now()
    inicio = fin - timedelta(days=int(365.25 * dcfg["anios_historico"]) + 10)
    cache = Cache(dcfg["carpeta_cache"], dcfg["dias_validez_cache"])
    if dcfg["fuente"] == "demo":
        from demo import FuenteDemo
        fuente = FuenteDemo()
    else:
        try:
            fuente = FuenteYahoo()
        except ImportError:
            warnings.warn("yfinance no instalado: sin análisis de estilo.")
            return pd.DataFrame()
    out = {}
    for nombre, ticker in factores.items():
        clave = f"factor_{ticker}"
        s = cache.leer(clave)
        if s is None:
            try:
                s = fuente.serie_ticker(ticker, inicio, fin)
                if not s.empty:
                    cache.guardar(clave, s)
            except Exception as e:
                warnings.warn(f"No se pudo descargar el factor {nombre} ({ticker}): {e}")
                continue
        if s is not None and not s.empty:
            out[nombre] = s
    return pd.DataFrame(out).sort_index()
