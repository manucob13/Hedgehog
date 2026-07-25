"""
r1000_tickers.py
=================
Utilidad reutilizable para descargar el listado de componentes del
Russell 1000 Index y combinarlos con el universo adicional ya
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
    Los endpoints de BlackRock/iShares (Excel binario y CSV clásico)
    dejaron de ser fiables desde servidores cloud: BlackRock bloquea o
    cambia de formato las peticiones automatizadas, devolviendo datos
    corruptos o vacíos. En su lugar, este módulo usa la tabla
    "Components" de la página de Wikipedia del Russell 1000 Index
    (en.wikipedia.org/wiki/Russell_1000_Index), que se actualiza
    periódicamente y es estable para scraping con pandas.read_html.
"""

import re
import requests
import pandas as pd
import streamlit as st

from utils.tickers import create_tickers_universe

# --- Fuente principal: tabla "Components" de Wikipedia ---
WIKI_R1000_URL = "https://en.wikipedia.org/wiki/Russell_1000_Index"

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "*/*",
}

# El Russell 1000 real tiene ~1,000-1,050 componentes; se usa para
# validar que el parseo no esté vacío o corrupto.
EXPECTED_MIN = 800
EXPECTED_MAX = 1300


def _clean_ticker(raw):
    """Normaliza un ticker crudo a formato yfinance-friendly."""
    if raw is None:
        return None
    t = str(raw).strip().upper()
    if not t or t in ("-", "NAN", "N/A", ""):
        return None
    # Wikipedia a veces incluye notas al pie tipo "AAPL[a]" -> nos
    # quedamos solo con la parte del ticker antes de cualquier corchete.
    t = re.sub(r"\[.*?\]", "", t).strip()
    t = t.replace(" ", "")
    # BRK.B / BF.B -> formato yfinance usa guion: BRK-B / BF-B
    t = t.replace(".", "-")
    if not t:
        return None
    return t


def _download_html(url, timeout=30):
    """Descarga el HTML crudo de una URL. None si falla."""
    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except Exception:
        return None


def parse_r1000_tickers_from_wikipedia(html_text):
    """
    Parsea la tabla "Components" de la página de Wikipedia del Russell
    1000 Index. Busca dinámicamente, entre todas las tablas de la
    página, la que contenga una columna 'Symbol' (o 'Ticker'), en vez
    de asumir un índice de tabla fijo, para tolerar cambios menores en
    el layout de la página.
    """
    if not html_text:
        return []
    try:
        tables = pd.read_html(html_text)
    except Exception:
        return []

    symbol_col_candidates = ("symbol", "ticker")

    for df in tables:
        cols_lower = [str(c).strip().lower() for c in df.columns]
        match_col = None
        for cand in symbol_col_candidates:
            if cand in cols_lower:
                match_col = df.columns[cols_lower.index(cand)]
                break
        if match_col is None:
            continue

        tickers = [_clean_ticker(t) for t in df[match_col].tolist()]
        tickers = sorted({t for t in tickers if t})
        if len(tickers) >= EXPECTED_MIN:
            return tickers

    return []


def download_r1000_tickers():
    """
    Descarga + parsea los componentes del Russell 1000 desde Wikipedia.
    Devuelve (tickers_list, ok_bool). ok=False si la descarga falla o
    el conteo resultante no es razonable.
    """
    html_text = _download_html(WIKI_R1000_URL)
    if not html_text:
        return [], False

    tickers = parse_r1000_tickers_from_wikipedia(html_text)
    if tickers and EXPECTED_MIN <= len(tickers) <= EXPECTED_MAX:
        return tickers, True

    return [], False


def build_full_universe():
    """
    Combina:
      - Tickers del Russell 1000 (Wikipedia)
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

    all_tickers = sorted(set(r1000_tickers) | set(extra_tickers)) if r1000_ok else sorted(set(extra_tickers))

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
    """Cache de Streamlit (6h) para no golpear Wikipedia en cada rerun."""
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
