"""
r1000_tickers.py
=================
Utilidad reutilizable para descargar el listado de holdings del ETF
iShares Russell 1000 (IWB) directamente desde iShares, extraer los
tickers, y combinarlos con el universo curado de utils/tickers.py
(~210 tickers: Acciones + Índices + ETFs) para formar el universo
completo usado por los distintos screeners del proyecto.

Uso típico en cualquier página Streamlit:

    from utils.r1000_tickers import get_full_universe, refresh_full_universe

    df_universe, meta = get_full_universe()      # usa caché (6h)
    df_universe, meta = refresh_full_universe()  # fuerza descarga nueva

    df_universe -> DataFrame con columna 'Ticker'
    meta -> dict con conteos y estado de la descarga (para mostrar en UI)
"""

import io
import requests
import pandas as pd
import streamlit as st

from utils.tickers import create_tickers_universe

IWB_URL = (
    "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/"
    "?fileType=csv&fileName=IWB_holdings&dataType=fund"
)

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def _clean_ticker(raw):
    """Normaliza un ticker crudo del CSV de iShares a formato yfinance-friendly."""
    if raw is None:
        return None
    t = str(raw).strip().upper()
    if not t or t in ("-", "NAN", "N/A"):
        return None
    t = t.replace(" ", "")
    return t


def download_iwb_holdings_csv(timeout=30):
    """
    Descarga el CSV público de holdings de IWB (iShares Russell 1000 ETF).
    Devuelve el contenido crudo en texto, o None si falla.
    """
    try:
        resp = requests.get(IWB_URL, headers=REQUEST_HEADERS, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except Exception:
        return None


def parse_iwb_tickers(csv_text):
    """
    Parsea el CSV crudo de iShares y devuelve una lista de tickers únicos.

    El CSV de iShares trae varias filas de metadata (nombre del fondo,
    fecha, etc.) antes de la fila de cabecera real
    ("Ticker,Name,Sector,Asset Class,...") y termina con un bloque de
    texto legal. Buscamos dinámicamente la fila de cabecera en vez de
    asumir un número fijo de filas a saltar, para que no se rompa si
    iShares cambia el formato ligeramente.
    """
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
    """Descarga + parsea el CSV de iShares. Devuelve (tickers_list, ok_bool)."""
    csv_text = download_iwb_holdings_csv()
    if csv_text is None:
        return [], False
    tickers = parse_iwb_tickers(csv_text)
    if not tickers:
        return [], False
    return tickers, True


def build_full_universe():
    """
    Combina:
      - Tickers del Russell 1000 (IWB holdings, descargados de iShares)
      - Universo curado ya existente en utils/tickers.py

    Devuelve (df_universe, meta):
      - df_universe: DataFrame con columna 'Ticker' (únicos, ordenados)
      - meta: dict con conteos y estado de la descarga, para mostrar en UI
    """
    r1000_tickers, r1000_ok = download_r1000_tickers()

    df_curated = create_tickers_universe()
    curated_tickers = (
        df_curated["Ticker"].astype(str).tolist()
        if isinstance(df_curated, pd.DataFrame)
        else list(df_curated)
    )
    curated_tickers = [_clean_ticker(t) for t in curated_tickers]
    curated_tickers = [t for t in curated_tickers if t]

    all_tickers = sorted(set(r1000_tickers) | set(curated_tickers))

    meta = {
        "r1000_ok": r1000_ok,
        "r1000_count": len(r1000_tickers),
        "curated_count": len(curated_tickers),
        "total_count": len(all_tickers),
    }

    df_universe = pd.DataFrame({"Ticker": all_tickers})
    return df_universe, meta


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def _cached_full_universe():
    """Cache de Streamlit (6h) para no golpear el CSV de iShares en cada rerun."""
    return build_full_universe()


def get_full_universe():
    """
    Punto de entrada estándar para las páginas: devuelve (df_universe, meta)
    usando caché de 6h. Si la descarga de iShares falla, cae de vuelta al
    universo curado de tickers.py para que la página nunca se quede sin
    tickers.
    """
    df_universe, meta = _cached_full_universe()
    if df_universe.empty or not meta.get("r1000_ok"):
        df_curated = create_tickers_universe()
        curated_tickers = (
            df_curated["Ticker"].astype(str).tolist()
            if isinstance(df_curated, pd.DataFrame)
            else list(df_curated)
        )
        curated_tickers = sorted(
            {_clean_ticker(t) for t in curated_tickers if _clean_ticker(t)}
        )
        df_universe = pd.DataFrame({"Ticker": curated_tickers})
        meta = {
            "r1000_ok": False,
            "r1000_count": 0,
            "curated_count": len(curated_tickers),
            "total_count": len(curated_tickers),
        }
    return df_universe, meta


def refresh_full_universe():
    """Fuerza una descarga nueva (limpia caché) y devuelve (df_universe, meta)."""
    _cached_full_universe.clear()
    return get_full_universe()
