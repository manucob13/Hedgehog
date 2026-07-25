"""
r1000_tickers.py
=================
Utilidad reutilizable para descargar el listado de holdings del ETF
iShares Russell 1000 (IWB) y combinarlos con el universo adicional ya
existente en utils/tickers.py (~210 tickers: Acciones + Índices + ETFs)
para formar el universo completo usado por los distintos screeners del
proyecto.

Uso típico en cualquier página Streamlit:

    from utils.r1000_tickers import get_full_universe, refresh_full_universe

    df_universe, meta = get_full_universe()      # usa caché (6h)
    df_universe, meta = refresh_full_universe()  # fuerza descarga nueva

    df_universe -> DataFrame con columna 'Ticker'
    meta -> dict con conteos y estado de la descarga (para mostrar en UI)

NOTA sobre la fuente de datos:
    La URL clásica "ishares.com/.../?fileType=csv&fileName=IWB_holdings"
    dejó de servir el CSV directamente (ahora devuelve la página HTML
    del producto). El fichero real de holdings se descarga hoy como un
    Excel binario desde el endpoint interno que usa el botón "Data
    Download" de la propia página de iShares. Este módulo usa ese
    endpoint como fuente principal, con el CSV clásico como respaldo
    por si iShares lo reactiva en el futuro.
"""

import io
import requests
import pandas as pd
import streamlit as st

from utils.tickers import create_tickers_universe

# --- Endpoint principal (real) ---
# Excel binario servido por blackrock.com, identificado por el
# portfolioId del fondo (239707 = IWB, iShares Russell 1000 ETF).
IWB_XLSX_URL = (
    "https://www.blackrock.com/varnish-api/blk-one01-product-data/"
    "product-data/api/v1/get-fund-document"
    "?appType=PRODUCT_PAGE&appSubType=ISHARES&targetSite=us-ishares"
    "&locale=en_US&portfolioId=239707&component=fundDownload&userType=individual"
)

# --- Endpoint de respaldo (por si iShares reactiva el CSV clásico) ---
IWB_CSV_URL_FALLBACK = (
    "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/"
    "?fileType=csv&fileName=IWB_holdings&dataType=fund"
)

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "*/*",
}


def _clean_ticker(raw):
    """Normaliza un ticker crudo del fichero de iShares a formato yfinance-friendly."""
    if raw is None:
        return None
    t = str(raw).strip().upper()
    if not t or t in ("-", "NAN", "N/A"):
        return None
    t = t.replace(" ", "")
    return t


def _download_bytes(url, timeout=30):
    """Descarga el contenido crudo (bytes) de una URL. None si falla."""
    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None


def parse_iwb_tickers_from_excel(content_bytes):
    """
    Parsea el fichero Excel (binario) de holdings de IWB y devuelve una
    lista de tickers únicos.

    El fichero trae varias filas de metadata (nombre del fondo, fecha,
    etc.) antes de la fila de cabecera real
    ("Ticker, Name, Sector, Asset Class, ..."). Buscamos dinámicamente
    esa fila en vez de asumir un número fijo de filas a saltar, para
    que no se rompa si iShares cambia el formato ligeramente.
    """
    if not content_bytes:
        return []
    try:
        df_raw = pd.read_excel(io.BytesIO(content_bytes), header=None, engine="openpyxl")
    except Exception:
        return []

    header_idx = None
    for i in range(len(df_raw)):
        val = df_raw.iloc[i, 0]
        if isinstance(val, str) and val.strip().lower() == "ticker":
            header_idx = i
            break
    if header_idx is None:
        return []

    try:
        df = pd.read_excel(io.BytesIO(content_bytes), header=header_idx, engine="openpyxl")
    except Exception:
        return []

    if "Ticker" not in df.columns:
        return []

    df = df[df["Ticker"].notna()]
    tickers = [_clean_ticker(t) for t in df["Ticker"].tolist()]
    tickers = sorted({t for t in tickers if t})
    return tickers


def parse_iwb_tickers_from_csv(csv_text):
    """Parsea el CSV clásico de iShares (usado solo como respaldo)."""
    if not csv_text:
        return []

    lines = csv_text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        first_cell = line.split(",")[0].strip().strip('"')
        if first_cell.lower() == "ticker":
            header_idx = i
            break
    if header_idx is None:
        return []

    data_str = "\n".join(lines[header_idx:])
    try:
        df = pd.read_csv(io.StringIO(data_str), thousands=",")
    except Exception:
        return []

    if "Ticker" not in df.columns:
        return []

    df = df[df["Ticker"].notna()]
    tickers = [_clean_ticker(t) for t in df["Ticker"].tolist()]
    tickers = sorted({t for t in tickers if t})
    return tickers


def download_r1000_tickers():
    """
    Descarga + parsea los holdings de IWB (Russell 1000).
    Intenta primero el endpoint real (Excel binario); si falla, cae al
    CSV clásico como respaldo. Devuelve (tickers_list, ok_bool).
    """
    # 1) Endpoint principal: Excel binario
    content = _download_bytes(IWB_XLSX_URL)
    if content:
        tickers = parse_iwb_tickers_from_excel(content)
        if tickers:
            return tickers, True

    # 2) Respaldo: CSV clásico (por si iShares lo reactiva)
    content = _download_bytes(IWB_CSV_URL_FALLBACK)
    if content:
        try:
            csv_text = content.decode("utf-8", errors="ignore")
        except Exception:
            csv_text = None
        tickers = parse_iwb_tickers_from_csv(csv_text) if csv_text else []
        if tickers:
            return tickers, True

    return [], False


def build_full_universe():
    """
    Combina:
      - Tickers del Russell 1000 (IWB holdings)
      - Universo adicional ya existente en utils/tickers.py

    Devuelve (df_universe, meta):
      - df_universe: DataFrame con columna 'Ticker' (únicos, ordenados)
      - meta: dict con conteos y estado de la descarga, para mostrar en UI
    """
    r1000_tickers, r1000_ok = download_r1000_tickers()

    df_extra = create_tickers_universe()
    extra_tickers = (
        df_extra["Ticker"].astype(str).tolist()
        if isinstance(df_extra, pd.DataFrame)
        else list(df_extra)
    )
    extra_tickers = [_clean_ticker(t) for t in extra_tickers]
    extra_tickers = [t for t in extra_tickers if t]

    all_tickers = sorted(set(r1000_tickers) | set(extra_tickers))

    meta = {
        "r1000_ok": r1000_ok,
        "r1000_count": len(r1000_tickers),
        "extra_count": len(extra_tickers),
        "total_count": len(all_tickers),
    }

    df_universe = pd.DataFrame({"Ticker": all_tickers})
    return df_universe, meta


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def _cached_full_universe():
    """Cache de Streamlit (6h) para no golpear el endpoint de iShares en cada rerun."""
    return build_full_universe()


def get_full_universe():
    """
    Punto de entrada estándar para las páginas: devuelve (df_universe, meta)
    usando caché de 6h. Si la descarga del Russell 1000 falla, cae de
    vuelta al universo adicional de tickers.py para que la página nunca
    se quede sin tickers.
    """
    df_universe, meta = _cached_full_universe()
    if df_universe.empty or not meta.get("r1000_ok"):
        df_extra = create_tickers_universe()
        extra_tickers = (
            df_extra["Ticker"].astype(str).tolist()
            if isinstance(df_extra, pd.DataFrame)
            else list(df_extra)
        )
        extra_tickers = sorted(
            {_clean_ticker(t) for t in extra_tickers if _clean_ticker(t)}
        )
        df_universe = pd.DataFrame({"Ticker": extra_tickers})
        meta = {
            "r1000_ok": False,
            "r1000_count": 0,
            "extra_count": len(extra_tickers),
            "total_count": len(extra_tickers),
        }
    return df_universe, meta


def refresh_full_universe():
    """Fuerza una descarga nueva (limpia caché) y devuelve (df_universe, meta)."""
    _cached_full_universe.clear()
    return get_full_universe()
