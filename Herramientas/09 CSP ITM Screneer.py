"""
Cash Secured Put Screener
============================
Objetivo: vender puts OTM/ATM con prima >= umbral mínimo (% del precio de
hoy), eligiendo el strike MÁS ALEJADO del precio (más seguro, menor
probabilidad de asignación) que todavía cumpla ese mínimo.

Como el strike está siempre <= precio actual (OTM/ATM), el intrínseco del
put es 0 y la prima entera es "extrínseco" — por eso el objetivo se pide
directamente como % del precio del subyacente hoy.

FILTROS (activables/desactivables desde la UI salvo los marcados [fijo]):
- Precio del subyacente: rango configurable
- Precio > SMA30 [toggle, OFF por defecto]
- Precio > SMA200 [toggle, OFF por defecto]
- PCR < 1.0 (sesgo alcista/neutral) [toggle]
- OI mínimo del strike [toggle, OFF por defecto — el OI de Alpaca no
  siempre es fiable para nombres poco líquidos]
- Spread bid-ask <= X% de la prima [toggle, OFF por defecto — revisa
  Spread_% en la tabla en vez de confiar ciegamente en el filtro]
- Prima mínima objetivo (% del precio), o banda mín-máx opcional [toggle;
  modo diagnóstico ignora ambos]
- Bid > 0 y Ask > 0 (opción realmente cotizada) [fijo]
- Ticker con opciones semanales listadas [fijo]
- Vencimiento = EXACTAMENTE la fecha objetivo, sin tolerancia [fijo]
- Filtros adicionales POST-escaneo (sobre resultados ya obtenidos, sin
  red): distancia OTM mínima, |delta| máximo, OI mínimo, volumen mínimo.

SELECCIÓN DE STRIKE (dentro de cada ticker): entre los strikes OTM/ATM que
cumplen todos los filtros, se elige el MÁS ALEJADO del precio (mayor
OTM_%) — el más seguro que aún así cumple la prima mínima pedida.
RANKING (entre tickers): por OTM_% descendente (más seguro primero).
PRECIO DE OPCIÓN: midprice (bid+ask)/2 siempre.
PRECIO DEL SUBYACENTE: en vivo vía Alpaca (latest trade), con fallback al
último cierre histórico (yfinance) si Alpaca no devuelve dato.
CAPITAL REQUERIDO: strike x 100 (lo que hay que reservar para asegurar en
efectivo 1 contrato, de ahí "cash secured").
DATOS: opciones (vencimientos, cadena, precio en vivo) vía Alpaca —
utils/utils_alpaca.py (el mismo módulo que usa el screener de covered
calls; get_option_chain ya devuelve puts, no ha hecho falta tocarlo).
Precio diario/SMA30/SMA200/RV10 vía yfinance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, date
from statistics import NormalDist
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import warnings
import socket
import logging
import plotly.graph_objects as go
import plotly.express as px

from utils.utils import check_password
from utils.tickers import create_tickers_universe
from utils.utils_alpaca import (
    get_option_expirations,
    has_weekly_options,
    get_option_chain,
    get_live_price,
)

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("csp_screener")

SOCKET_TIMEOUT = 15
socket.setdefaulttimeout(SOCKET_TIMEOUT)

MAX_WORKERS = 3
SPREAD_MAX_PCT_OF_PREMIUM = 50.0

_N = NormalDist()
RISK_FREE_RATE = 0.045

REASON_ORDER = [
    "ok", "no_daily_data", "price_out_of_range", "sma_unavailable",
    "below_sma30", "below_sma200", "no_expirations", "no_weekly_options",
    "no_expiration_exact_dte", "no_puts_chain", "pcr_bearish",
    "no_put_candidate", "error",
]

REASON_LABELS = {
    "ok": "✅ Pasó todos los filtros",
    "no_daily_data": "Sin datos diarios (Fase 1)",
    "price_out_of_range": "Precio fuera de rango (Fase 1)",
    "sma_unavailable": "No hay suficiente histórico para SMA30/SMA200 (Fase 1)",
    "below_sma30": "Precio <= SMA30 (Fase 1)",
    "below_sma200": "Precio <= SMA200 (Fase 1)",
    "no_expirations": "Sin vencimientos de opciones listados en Alpaca (Fase 2)",
    "no_weekly_options": "Sin cadencia de opciones semanales (Fase 2) [fijo]",
    "no_expiration_exact_dte": "Sin vencimiento EXACTO en la fecha objetivo (Fase 2) [fijo]",
    "no_puts_chain": "Cadena de puts vacía/no disponible en Alpaca (Fase 2)",
    "pcr_bearish": "PCR >= 1.0 — sesgo bajista (Fase 2)",
    "no_put_candidate": "Sin strike OTM que cumpla prima/OI/bid/ask/spread (Fase 2)",
    "error": "Excepción no controlada",
}

_debug_lock = Lock()
_debug_samples = {}
_price_samples = []
_MAX_DEBUG_SAMPLES = 5

def _record_debug(reason, msg):
    with _debug_lock:
        bucket = _debug_samples.setdefault(reason, [])
        if len(bucket) < _MAX_DEBUG_SAMPLES:
            bucket.append(str(msg)[:200])

def _record_price(ticker, price):
    with _debug_lock:
        if len(_price_samples) < 5000:
            _price_samples.append((ticker, price))

def _reset_debug():
    with _debug_lock:
        _debug_samples.clear()
        _price_samples.clear()

def _clean_ticker(raw):
    if raw is None:
        return None
    t = str(raw).strip().replace("&amp;", "&").upper()
    if not t or t in ("-", "NAN", "N/A", ""):
        return None
    return t.replace(" ", "")

# ======================================================================
# 0. UNIVERSO
# ======================================================================

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_full_universe():
    df = create_tickers_universe(include_russell1000=True)
    if df is None or df.empty:
        return pd.DataFrame({"Ticker": []}), {}
    df = df.copy()
    df["Ticker"] = df["Ticker"].apply(_clean_ticker)
    df = df.dropna(subset=["Ticker"]).drop_duplicates("Ticker").reset_index(drop=True)
    n_r = int((df["Type"] == "Russell1000").sum()) if "Type" in df.columns else 0
    meta = {
        "r1000_ok": n_r > 0,
        "r1000_count": n_r,
        "extra_count": len(df) - n_r,
        "total_count": len(df),
    }
    return df[["Ticker"]].sort_values("Ticker").reset_index(drop=True), meta

def refresh_universe():
    get_full_universe.clear()
    return get_full_universe()

# ======================================================================
# 1. DATOS DIARIOS + SMA30/SMA200 (Fase 1 — paralela)
# ======================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def get_daily_data(ticker):
    """Descarga precio diario AJUSTADO vía yf.Ticker().history() — igual
    razón que en el screener de covered calls: yf.download() devuelve
    columnas mal aplanadas en esta instalación. Ventana de 300 días para
    tener margen de sobra para SMA200 (200 sesiones de trading)."""
    try:
        end = datetime.now() + timedelta(days=1)
        start = end - timedelta(days=300)
        data = yf.Ticker(ticker).history(start=start, end=end, interval="1d", auto_adjust=True)
        if data is None or data.empty or len(data) < 205:
            _record_debug("no_daily_data", f"{ticker}: descarga vacía o insuficiente ({0 if data is None else len(data)} filas)")
            return None
        data = data.dropna(subset=["Close"])
        if data.empty or len(data) < 205:
            _record_debug("no_daily_data", f"{ticker}: tras limpiar NaN quedan {len(data)} filas")
            return None
        data.index = pd.to_datetime(data.index)
        return data
    except Exception as e:
        _record_debug("no_daily_data", f"{ticker}: {e}")
        return None

def get_trend_info(close):
    """SMA30 y SMA200 y distancia del precio a cada una. None si no hay
    histórico suficiente para SMA200 (el requisito más exigente)."""
    try:
        if len(close) < 205:
            return None
        sma30 = close.rolling(30).mean()
        sma200 = close.rolling(200).mean()
        price = float(close.iloc[-1])
        sma30_now = float(sma30.iloc[-1])
        sma200_now = float(sma200.iloc[-1])
        return {
            "sma30": round(sma30_now, 2),
            "dist_sma30_pct": round((price - sma30_now) / sma30_now * 100, 2),
            "above_sma30": price > sma30_now,
            "sma200": round(sma200_now, 2),
            "dist_sma200_pct": round((price - sma200_now) / sma200_now * 100, 2),
            "above_sma200": price > sma200_now,
        }
    except Exception:
        return None

def get_rv10(close):
    try:
        if len(close) < 12:
            return None
        rets = np.log(close / close.shift(1)).dropna()
        return round(float(rets.iloc[-10:].std() * np.sqrt(252) * 100), 2)
    except Exception:
        return None

# ======================================================================
# 2. BLACK-SCHOLES DELTA (put)
# ======================================================================

def bs_put_delta(S, K, T_years, sigma, r=RISK_FREE_RATE):
    """Delta de un put europeo (Black-Scholes): N(d1) - 1, siempre entre
    -1 y 0. La magnitud (|delta|) es una aproximación habitual de la
    probabilidad de acabar ITM (es decir, de asignación)."""
    try:
        if T_years <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return None
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_years) / (sigma * np.sqrt(T_years))
        return round(float(_N.cdf(d1) - 1), 3)
    except Exception:
        return None

# ======================================================================
# 3. PUT/CALL RATIO (sobre una cadena ya descargada, sin red)
# ======================================================================

def compute_pcr(calls, puts, current_price, range_pct=15):
    try:
        lo = current_price * (1 - range_pct / 100)
        hi = current_price * (1 + range_pct / 100)
        c_filt = calls[(calls["strike"] >= lo) & (calls["strike"] <= hi)]
        p_filt = puts[(puts["strike"] >= lo) & (puts["strike"] <= hi)]
        c_oi = float(c_filt["openInterest"].fillna(0).sum())
        p_oi = float(p_filt["openInterest"].fillna(0).sum())
        if c_oi == 0:
            return None
        return round(p_oi / c_oi, 3)
    except Exception:
        return None

# ======================================================================
# 4. CANDIDATO: put OTM/ATM más seguro que cumple la prima mínima
# ======================================================================

def find_safest_put_candidate(puts_df, current_price, dte_calendar,
                               premium_min_pct, premium_max_pct,
                               apply_oi_filter, min_oi,
                               apply_spread_filter, spread_max_pct,
                               diagnostic_mode=False):
    """Entre los puts OTM/ATM (strike <= precio, así que intrínseco=0 y
    prima=extrínseco entero), filtra por prima mínima (% del precio) y
    elige el strike MÁS ALEJADO del precio (mayor OTM_%) — el más seguro
    que aún así cumple la prima mínima pedida. Mismo patrón que
    find_deep_itm_candidate() del screener de covered calls, pero
    optimizando por seguridad (distancia OTM) en vez de por downside
    protection."""
    try:
        otm = puts_df[puts_df["strike"] <= current_price].copy()
        if otm.empty:
            return None

        otm["bid"] = pd.to_numeric(otm["bid"], errors="coerce").fillna(0)
        otm["ask"] = pd.to_numeric(otm["ask"], errors="coerce").fillna(0)
        otm["mid"] = (otm["bid"] + otm["ask"]) / 2
        otm = otm[(otm["bid"] > 0) & (otm["ask"] > 0) & (otm["mid"] > 0)]
        if otm.empty:
            return None

        otm["oi"] = pd.to_numeric(
            otm.get("openInterest", pd.Series(0, index=otm.index)), errors="coerce"
        ).fillna(0)
        if apply_oi_filter:
            otm = otm[otm["oi"] >= min_oi]
            if otm.empty:
                return None

        # Intrínseco de un put OTM/ATM (strike <= precio) es siempre 0 —
        # la prima entera es "extrínseco" / tiempo.
        otm["premium_pct"] = otm["mid"] / current_price * 100
        otm["otm_pct"] = (current_price - otm["strike"]) / current_price * 100
        otm["spread_dollar"] = otm["ask"] - otm["bid"]
        otm["spread_pct"] = otm["spread_dollar"] / otm["mid"] * 100

        if apply_spread_filter:
            otm = otm[otm["spread_dollar"] <= otm["mid"] * (spread_max_pct / 100)]
            if otm.empty:
                return None

        if diagnostic_mode:
            candidates = otm
        elif premium_max_pct is not None:
            candidates = otm[
                (otm["premium_pct"] >= premium_min_pct) & (otm["premium_pct"] <= premium_max_pct)
            ]
        else:
            candidates = otm[otm["premium_pct"] >= premium_min_pct]
        if candidates.empty:
            return None

        # El más seguro = el más alejado del precio (mayor OTM_%).
        best = candidates.sort_values("otm_pct", ascending=False).iloc[0]

        T_years = dte_calendar / 365.0
        iv = float(best.get("impliedVolatility") or 0)
        delta = bs_put_delta(current_price, float(best["strike"]), T_years, iv) if iv > 0 else None

        best_premium_pct = float(best["premium_pct"])
        meets_range = best_premium_pct >= premium_min_pct
        if premium_max_pct is not None:
            meets_range = meets_range and best_premium_pct <= premium_max_pct

        return {
            "strike": float(best["strike"]),
            "mid": round(float(best["mid"]), 2),
            "bid": round(float(best["bid"]), 2),
            "ask": round(float(best["ask"]), 2),
            "premium_pct": round(best_premium_pct, 3),
            "otm_pct": round(float(best["otm_pct"]), 2),
            "spread_pct": round(float(best["spread_pct"]), 2),
            "oi": int(best["oi"]),
            "volume": int(pd.to_numeric(best.get("volume", 0), errors="coerce") or 0),
            "iv_pct": round(iv * 100, 2) if iv > 0 else None,
            "delta": delta,
            "in_target_band": bool(meets_range),
        }
    except Exception:
        return None

# ======================================================================
# 5a. FASE 1 — precio diario + SMA30/SMA200 (paralela)
# ======================================================================

def phase1_price_filter(ticker, params):
    try:
        data = get_daily_data(ticker)
        if data is None:
            return None, "no_daily_data"

        close = data["Close"]
        current_price = float(close.iloc[-1])
        _record_price(ticker, current_price)

        if not (params["min_price"] <= current_price <= params["max_price"]):
            _record_debug("price_out_of_range", f"{ticker}: precio calculado = {current_price}")
            return None, "price_out_of_range"

        trend = get_trend_info(close)
        if trend is None:
            return None, "sma_unavailable"

        if params["use_sma30_filter"] and not trend["above_sma30"]:
            return None, "below_sma30"
        if params["use_sma200_filter"] and not trend["above_sma200"]:
            return None, "below_sma200"

        rv = get_rv10(close)

        return {
            "Ticker": ticker,
            "current_price": round(current_price, 2),
            "sma30": trend["sma30"],
            "dist_sma30_pct": trend["dist_sma30_pct"],
            "sma200": trend["sma200"],
            "dist_sma200_pct": trend["dist_sma200_pct"],
            "rv": rv,
        }, "ok"
    except Exception as e:
        _record_debug("error", f"{ticker}: {e}")
        return None, "error"

# ======================================================================
# 5b. FASE 2 — opciones (SECUENCIAL, sin threads)
# ======================================================================

def phase2_options_filter(survivor, params):
    """Datos de opciones (vencimientos + cadena de puts) vía Alpaca.
    Precio en vivo del subyacente vía Alpaca también."""
    ticker = survivor["Ticker"]
    fallback_price = survivor["current_price"]
    target_date = params["target_expiration_date"]
    try:
        expirations = get_option_expirations(ticker)
        if not expirations:
            return None, "no_expirations"

        if not has_weekly_options(expirations):
            return None, "no_weekly_options"

        if target_date not in expirations:
            return None, "no_expiration_exact_dte"

        exp_str = target_date.isoformat()
        dte = (target_date - date.today()).days

        current_price = get_live_price(ticker, fallback_price)

        calls, puts = get_option_chain(ticker, target_date)
        if puts is None or puts.empty:
            return None, "no_puts_chain"

        pcr = compute_pcr(calls, puts, current_price) if calls is not None and not calls.empty else None
        if params["use_pcr_filter"] and pcr is not None and pcr >= 1.0:
            return None, "pcr_bearish"

        candidate = find_safest_put_candidate(
            puts, current_price, dte,
            params["premium_min"], params["premium_max"],
            params["use_oi_filter"], params["min_oi"],
            params["use_spread_filter"], params["spread_max_pct"],
            diagnostic_mode=params["diagnostic_mode"],
        )
        if candidate is None:
            return None, "no_put_candidate"

        iv_rv_ratio = (
            round(candidate["iv_pct"] / survivor["rv"], 3)
            if (candidate["iv_pct"] and survivor["rv"] and survivor["rv"] > 0) else None
        )

        result = {
            "Ticker": ticker,
            "Precio": round(current_price, 2),
            "Vencimiento": exp_str,
            "DTE": dte,
            "Strike": candidate["strike"],
            "OTM_%": candidate["otm_pct"],
            "Extrínseco_%": candidate["premium_pct"],
            "Prima_Mid": candidate["mid"],
            "Bid": candidate["bid"],
            "Ask": candidate["ask"],
            "Capital_Requerido": round(candidate["strike"] * 100, 2),
            "Retorno_Capital_%": round(candidate["mid"] / candidate["strike"] * 100, 3) if candidate["strike"] > 0 else None,
            "Delta": candidate["delta"],
            "IV_%": candidate["iv_pct"],
            "RV_%": survivor["rv"],
            "IV_RV": iv_rv_ratio,
            "OI": candidate["oi"],
            "Volumen": candidate["volume"],
            "Spread_%": candidate["spread_pct"],
            "SMA30": survivor["sma30"],
            "Dist_SMA30_%": survivor["dist_sma30_pct"],
            "SMA200": survivor["sma200"],
            "Dist_SMA200_%": survivor["dist_sma200_pct"],
            "PCR": pcr,
        }
        return result, "ok"
    except Exception as e:
        _record_debug("error", f"{ticker}: {e}")
        return None, "error"

# ======================================================================
# 6. ORQUESTADOR: Fase 1 (paralela) → Fase 2 (secuencial)
# ======================================================================

def run_screener(tickers, params, progress_bar, status_text):
    _reset_debug()
    funnel = {r: 0 for r in REASON_ORDER}
    total = len(tickers)

    status_text.text(f"🔍 Fase 1/2 — precio y tendencia: 0/{total}")
    survivors = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(phase1_price_filter, t, params): t for t in tickers}
        done = 0
        for future in as_completed(futures):
            done += 1
            progress_bar.progress(done / total * 0.5)
            status_text.text(f"🔍 Fase 1/2 — precio y tendencia: {done}/{total} · supervivientes: {len(survivors)}")
            r, reason = None, "error"
            try:
                r, reason = future.result(timeout=SOCKET_TIMEOUT + 10)
            except Exception as e:
                _record_debug("error", f"timeout/fallo esperando resultado: {e}")
            funnel[reason] = funnel.get(reason, 0) + 1
            if r is not None:
                survivors.append(r)

    results = []
    n_survivors = len(survivors)
    for i, s in enumerate(survivors, start=1):
        progress_bar.progress(0.5 + (i / max(n_survivors, 1)) * 0.5)
        status_text.text(f"🔍 Fase 2/2 — opciones: {i}/{n_survivors} ({s['Ticker']}) · encontrados: {len(results)}")
        r, reason = phase2_options_filter(s, params)
        funnel[reason] = funnel.get(reason, 0) + 1
        if r is not None:
            results.append(r)

    status_text.text(f"✅ Completado: {len(results)} candidatos")

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("OTM_%", ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))

    with _debug_lock:
        debug_snapshot = {k: list(v) for k, v in _debug_samples.items()}
        price_snapshot = list(_price_samples)

    return df, funnel, debug_snapshot, price_snapshot

def reset_everything():
    get_daily_data.clear()
    get_option_expirations.clear()
    get_market_trend.clear()
    for key in ("results", "funnel", "debug_snapshot", "price_snapshot",
                "scan_ts", "scanned_total"):
        st.session_state.pop(key, None)

# ======================================================================
# 7. GRÁFICO DE PRECIO
# ======================================================================

def plot_price(ticker):
    try:
        data = get_daily_data(ticker)
        if data is None:
            return None
        sma30 = data["Close"].rolling(30).mean()
        sma200 = data["Close"].rolling(200).mean()
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data.index, open=data["Open"], high=data["High"],
            low=data["Low"], close=data["Close"], name=ticker,
            increasing_line_color="#6daa45", decreasing_line_color="#dd6974",
        ))
        fig.add_trace(go.Scatter(x=data.index, y=sma30, mode="lines", name="SMA30",
                                  line=dict(color="#e8af34", width=2)))
        fig.add_trace(go.Scatter(x=data.index, y=sma200, mode="lines", name="SMA200",
                                  line=dict(color="#5b9bd5", width=2)))
        fig.update_layout(
            title=f"{ticker} — Precio + SMA30 + SMA200",
            template="plotly_dark", height=420,
            xaxis_rangeslider_visible=False, hovermode="x unified",
        )
        return fig
    except Exception:
        return None

# ======================================================================
# 7b. FILTRO DE MERCADO: Indicador Trend SP500-NASDAQ
# ======================================================================
# Portado del notebook "Amplitud de mercado" (AM v1.10, celda 4). Misma
# lógica exacta, pero con yf.Ticker().history() en vez de yf.download()
# (aquí yf.download() da columnas mal aplanadas, igual razón que el resto
# del pipeline). Gate previo a cualquier escaneo: si el mercado no está
# alcista, el escáner queda bloqueado por defecto (con opción de anular).

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_market_trend():
    """SPX y NASDAQ por encima de la MENOR de sus medias de 100/200
    sesiones, con esa media subiendo, cuentan como alcistas (+1) cada uno;
    si no, bajistas (-1). Suma de ambos: +2 Alcista, -2 Bajista, 0 mixto
    (se rellena con el último valor no-mixto vía ffill, igual que el
    notebook original — en la práctica el indicador es casi siempre
    binario). Devuelve None si falla la descarga de cualquiera de los tres
    tickers."""
    tickers = {"^GSPC": "SPX", "^IXIC": "NAS", "VTI": "VTI"}
    end = datetime.now() + timedelta(days=1)
    start = datetime(2015, 1, 1)
    closes = {}
    try:
        for tk, name in tickers.items():
            h = yf.Ticker(tk).history(start=start, end=end, interval="1d", auto_adjust=False)
            if h is None or h.empty:
                logger.warning(f"[market_trend] {tk}: descarga vacía")
                return None
            closes[name] = h["Close"]

        df = pd.concat(closes, axis=1).dropna(how="any")
        df.index = pd.to_datetime(df.index).tz_localize(None)
        if len(df) < 205:
            return None

        df["SPX_MA100"] = df["SPX"].rolling(100).mean()
        df["SPX_MA200"] = df["SPX"].rolling(200).mean()
        df["NAS_MA100"] = df["NAS"].rolling(100).mean()
        df["NAS_MA200"] = df["NAS"].rolling(200).mean()
        df["VTI_MA100"] = df["VTI"].rolling(100).mean()
        df["VTI_MA200"] = df["VTI"].rolling(200).mean()

        df["SPX_Med"] = np.minimum(df["SPX_MA100"], df["SPX_MA200"])
        df["NAS_Med"] = np.minimum(df["NAS_MA100"], df["NAS_MA200"])

        df["SPX_Tendencia"] = np.where(
            (df["SPX"] > df["SPX_Med"]) & (df["SPX_Med"] > df["SPX_Med"].shift(1)), 1, -1)
        df["NAS_Tendencia"] = np.where(
            (df["NAS"] > df["NAS_Med"]) & (df["NAS_Med"] > df["NAS_Med"].shift(1)), 1, -1)

        df["Tendencia"] = df["SPX_Tendencia"] + df["NAS_Tendencia"]
        df["Tendencia"] = df["Tendencia"].replace(0, np.nan).ffill()
        df["Tendencia"] = df["Tendencia"].apply(
            lambda x: 2 if x == 2 else (0 if x == 0 else (-2 if x == -2 else np.nan)))
        df = df.dropna(subset=["Tendencia"])
        if df.empty:
            return None

        start_year = datetime(datetime.now().year, 1, 1)
        df_plot = df[df.index >= start_year]
        if len(df_plot) < 20:
            df_plot = df.tail(150)

        last_trend = df["Tendencia"].iloc[-1]
        trend_label = "Alcista" if last_trend == 2 else ("Bajista" if last_trend == -2 else "Neutral")

        cambios = df[df["Tendencia"] != df["Tendencia"].shift(1)]
        trend_since = cambios.index[-1] if not cambios.empty else df.index[0]

        return {
            "trend_label": trend_label,
            "trend_value": float(last_trend),
            "trend_since": trend_since,
            "df_plot": df_plot,
        }
    except Exception as e:
        logger.warning(f"[market_trend] error: {e}")
        return None

def plot_market_trend(mt):
    """VTI coloreado verde/rojo/ámbar según Tendencia, con MA100/MA200 de
    referencia — versión plotly del gráfico matplotlib del notebook
    original, para encajar con el resto de la app."""
    df_plot = mt["df_plot"].copy()
    color_map = {2: "#6daa45", -2: "#dd6974", 0: "#e8af34"}

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["VTI_MA100"], mode="lines",
                              name="MA 100", line=dict(color="#999999", dash="dash", width=1)))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["VTI_MA200"], mode="lines",
                              name="MA 200", line=dict(color="#5b9bd5", dash="dash", width=1)))

    df_plot["block"] = (df_plot["Tendencia"] != df_plot["Tendencia"].shift()).cumsum()
    first_block = True
    for _, seg in df_plot.groupby("block"):
        color = color_map.get(seg["Tendencia"].iloc[0], "#999999")
        fig.add_trace(go.Scatter(
            x=seg.index, y=seg["VTI"], mode="lines",
            line=dict(color=color, width=2),
            name="Tendencia", showlegend=first_block,
            legendgroup="tendencia",
        ))
        first_block = False

    fig.update_layout(
        title="VTI — Indicador Trend SP500-NASDAQ",
        template="plotly_dark", height=420,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
    )
    return fig

def render_market_filter():
    """Apartado inicial 'Filtro de Mercado': calcula y grafica el
    indicador, y guarda en session_state si el mercado está alcista (o si
    el usuario ha decidido anular el bloqueo) para que el botón de
    escaneo lo respete más abajo."""
    st.markdown("### 📈 Filtro de Mercado")
    st.caption(
        "Indicador Trend SP500-NASDAQ: alcista cuando el S&P 500 y el Nasdaq "
        "Composite cotizan por encima de la menor de sus medias de 100/200 "
        "sesiones, y esa media está subiendo. Si el mercado no está alcista, "
        "el escáner queda bloqueado por defecto."
    )

    mt = get_market_trend()
    if mt is None:
        st.warning(
            "⚠️ No se pudo calcular el indicador de mercado (fallo de datos de "
            "SPX/NASDAQ/VTI). El escáner queda desbloqueado por no poder "
            "confirmar el estado del mercado — revísalo manualmente."
        )
        st.session_state["market_bullish"] = True
        st.session_state["market_override"] = False
        st.divider()
        return

    trend_label = mt["trend_label"]
    since_txt = mt["trend_since"].strftime("%d %b %Y")

    if trend_label == "Alcista":
        st.success(f"🟢 **Mercado ALCISTA** desde {since_txt}")
        st.session_state["market_bullish"] = True
    elif trend_label == "Bajista":
        st.error(
            f"🔴 **Mercado BAJISTA** desde {since_txt} — el indicador no "
            f"recomienda abrir posiciones nuevas ahora mismo."
        )
        st.session_state["market_bullish"] = False
    else:
        st.warning(f"🟡 **Mercado NEUTRAL** desde {since_txt}")
        st.session_state["market_bullish"] = False

    st.plotly_chart(plot_market_trend(mt), use_container_width=True)

    if not st.session_state["market_bullish"]:
        st.session_state["market_override"] = st.checkbox(
            "⚠️ Anular el filtro de mercado y escanear igualmente (bajo tu responsabilidad)",
            value=st.session_state.get("market_override", False),
        )
    else:
        st.session_state["market_override"] = False

    st.divider()

# ======================================================================
# 8. INTERFAZ STREAMLIT
# ======================================================================

def main():
    st.set_page_config(page_title="Cash Secured Put Screener", page_icon="🛡️", layout="wide")

    if not check_password():
        st.stop()

    st.title("🛡️ Cash Secured Put Screener")
    st.markdown(
        "**Objetivo: prima mínima semanal (% del precio de hoy) · "
        "Vencimiento EXACTO (sin tolerancia) · "
        "Ranking por strike más seguro (mayor OTM%)**"
    )
    st.caption(
        "⚙️ Puts OTM/ATM vía Alpaca · filtros opcionales de SMA30/SMA200 · "
        "OI y spread mínimos opcionales (OFF por defecto) · precio diario/"
        "SMA vía yfinance"
    )

    col_title, col_reset = st.columns([5, 1])
    with col_reset:
        if st.button("🧹 Resetear Todo", use_container_width=True,
                      help="Vacía el caché de precios y vencimientos de Alpaca, y borra "
                           "los resultados del último escaneo."):
            reset_everything()
            st.success("Caché y resultados borrados — listo para un escaneo limpio.")
            st.rerun()

    st.divider()

    render_market_filter()

    # ── Universo ───────────────────────────────────────────────────────
    st.markdown("### 📂 Universo de Tickers")
    col_info, col_btn = st.columns([4, 1])
    with col_btn:
        if st.button("🔄 Actualizar Universo", type="primary", use_container_width=True):
            with st.spinner("Reconstruyendo universo..."):
                df_universe, meta = refresh_universe()
                st.session_state["df_universe"] = df_universe
                st.session_state["meta_universe"] = meta

    if "df_universe" not in st.session_state:
        with st.spinner("Cargando universo..."):
            df_universe, meta = get_full_universe()
            st.session_state["df_universe"] = df_universe
            st.session_state["meta_universe"] = meta

    df_universe = st.session_state["df_universe"]
    meta = st.session_state["meta_universe"]
    tickers_all = df_universe["Ticker"].tolist()

    with col_info:
        if meta.get("r1000_ok"):
            st.success(
                f"✅ **{meta['total_count']:,} tickers** "
                f"(Russell 1000: {meta['r1000_count']:,} + Adicionales: {meta['extra_count']:,})"
            )
        else:
            st.warning(f"⚠️ Solo universo adicional: {meta['total_count']:,} tickers")

    st.divider()

    # ── Filtros ────────────────────────────────────────────────────────
    st.markdown("### ⚙️ Parámetros")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**💰 Prima objetivo (% del precio de hoy)**")
        use_premium_band = st.checkbox(
            "Usar banda (mín–máx) en vez de solo mínimo", value=False,
            help="Desactivado (por defecto): umbral mínimo, sin techo — pasan todos "
                 "los strikes con prima >= el valor que pongas. Activado: solo pasan "
                 "los strikes cuya prima caiga DENTRO de la banda.",
        )
        if use_premium_band:
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                premium_min = st.number_input("Prima mínima (%)", min_value=0.10, max_value=5.00, value=0.95, step=0.05)
            with pcol2:
                premium_max = st.number_input("Prima máxima (%)", min_value=0.10, max_value=10.00, value=1.10, step=0.05)
            if premium_max < premium_min:
                st.error(f"⚠️ La prima máxima ({premium_max}%) es menor que la mínima ({premium_min}%) — se intercambian.")
                premium_min, premium_max = premium_max, premium_min
            st.caption(f"✅ Banda activa: **{premium_min}% — {premium_max}%**")
        else:
            premium_min = st.number_input(
                "Prima mínima (% del precio, semanal)", min_value=0.10, max_value=5.00,
                value=1.00, step=0.05,
                help="Umbral mínimo: pasan todos los strikes con prima/precio >= este "
                     "valor. Como los puts son OTM/ATM (intrínseco=0), la prima entera "
                     "es 'extrínseco'.",
            )
            premium_max = None
            st.caption(f"✅ Umbral activo: **≥ {premium_min}%** (sin techo)")

        st.markdown("**💲 Precio del subyacente**")
        pcol3, pcol4 = st.columns(2)
        with pcol3:
            min_price = st.number_input("Precio mínimo ($)", min_value=1, max_value=10000, value=15, step=5)
        with pcol4:
            max_price = st.number_input("Precio máximo ($)", min_value=1, max_value=10000, value=50, step=5)
        if min_price > max_price:
            st.error(f"⚠️ El precio mínimo (${min_price:,}) es mayor que el máximo (${max_price:,}) — se intercambian.")
            min_price, max_price = max_price, min_price
        st.caption(f"✅ Rango activo: **${min_price:,} — ${max_price:,}**")

    with c2:
        st.markdown("**💧 Liquidez mínima**")
        use_oi_filter = st.checkbox(
            "Exigir OI mínimo del strike", value=False,
            help="Desactivado por defecto: el open interest que reporta Alpaca viene "
                 "con frecuencia en 0 o muy bajo para nombres fuera de los índices más "
                 "líquidos, incluso con bid/ask real cotizado.",
        )
        min_oi = st.number_input("OI mínimo del strike", min_value=10, max_value=10000, value=100, step=50,
                                  disabled=not use_oi_filter)
        if not use_oi_filter:
            st.caption("ℹ️ OI mínimo NO se está aplicando — solo informativo en la tabla.")

        st.markdown("**〰️ Spread bid-ask**")
        use_spread_filter = st.checkbox(
            "Exigir spread ≤ X% de la prima", value=False,
            help="Desactivado por defecto — igual razón que en el screener de covered "
                 "calls: puede excluir strikes válidos con spreads anchos pero cotización "
                 "real. Revisa Spread_% en la tabla.",
        )
        spread_max_pct = st.number_input("Spread máximo (% de la prima)", min_value=10, max_value=300,
                                          value=50, step=10, disabled=not use_spread_filter)
        if not use_spread_filter:
            st.caption("ℹ️ Spread máximo NO se está aplicando — revisa Spread_% antes de operar.")

        st.markdown("**🎯 Vencimiento objetivo (EXACTO)**")
        target_date = st.date_input(
            "Fecha de vencimiento objetivo", value=date.today() + timedelta(days=7),
            min_value=date.today() + timedelta(days=1), max_value=date.today() + timedelta(days=180),
            format="DD/MM/YYYY",
            help="El ticker debe tener un vencimiento que caiga EXACTAMENTE en este día "
                 "— sin tolerancia ni búsqueda del más cercano.",
        )
        dte_target = (target_date - date.today()).days
        st.caption(f"📅 Objetivo: **{target_date.strftime('%d %b %Y')}** (DTE={dte_target} días, exacto).")

    with c3:
        st.markdown("**🎚️ Filtros activables**")
        use_sma30_filter = st.checkbox("Precio > SMA30", value=False,
                                        help="Exige que el precio esté por encima de la media móvil de 30 sesiones.")
        use_sma200_filter = st.checkbox("Precio > SMA200", value=False,
                                         help="Exige que el precio esté por encima de la media móvil de 200 sesiones "
                                              "(tendencia de fondo alcista).")
        use_pcr_filter = st.checkbox("PCR < 1.0 (sesgo alcista/neutral)", value=True)
        diagnostic_mode = st.checkbox(
            "🔬 Modo diagnóstico (ignora el umbral de prima)", value=False,
            help="Devuelve el mejor candidato OTM aunque su prima no alcance el mínimo "
                 "configurado, para ver los valores reales del mercado.",
        )
        st.caption(
            "Siempre activos (no configurables): bid>0 y ask>0, opciones semanales "
            "listadas, y vencimiento EXACTO en la fecha objetivo (sin tolerancia)."
        )

    params = {
        "premium_min": premium_min,
        "premium_max": premium_max,
        "min_price": min_price,
        "max_price": max_price,
        "use_oi_filter": use_oi_filter,
        "min_oi": min_oi,
        "use_spread_filter": use_spread_filter,
        "spread_max_pct": spread_max_pct,
        "target_expiration_date": target_date,
        "use_sma30_filter": use_sma30_filter,
        "use_sma200_filter": use_sma200_filter,
        "use_pcr_filter": use_pcr_filter,
        "diagnostic_mode": diagnostic_mode,
    }

    st.divider()

    # ── Escaneo ────────────────────────────────────────────────────────
    st.markdown("### 🚀 Ejecutar Escaneo")

    market_ok = st.session_state.get("market_bullish", True) or st.session_state.get("market_override", False)
    if not market_ok:
        st.error("🔴 Escaneo bloqueado: el Filtro de Mercado no está alcista. Anúlalo arriba si quieres escanear igualmente.")

    with st.expander("🧪 Prueba rápida con universo reducido (opcional)"):
        test_tickers_raw = st.text_input(
            "Tickers de prueba (separados por coma o espacio)", value="",
            placeholder="AAPL, MSFT, NVDA, KO",
        )
    test_tickers = [
        _clean_ticker(t) for t in test_tickers_raw.replace(",", " ").split()
    ] if test_tickers_raw.strip() else []
    test_tickers = [t for t in test_tickers if t]

    scan_universe = test_tickers if test_tickers else tickers_all

    scan_btn = st.button("🎯 INICIAR ESCANEO", type="primary", use_container_width=True,
                          disabled=(len(scan_universe) == 0) or not market_ok)

    if test_tickers:
        st.caption(f"🧪 Modo prueba activo: se escanearán solo **{len(scan_universe)}** tickers.")
    else:
        st.caption(f"ℹ️ Se escanearán **{len(scan_universe):,}** tickers · Fase 1 en paralelo → Fase 2 secuencial.")

    if scan_btn:
        scan_tickers = scan_universe
        if not scan_tickers:
            st.error("⚠️ El universo de tickers está vacío — pulsa 'Actualizar Universo'.")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()
        df_results, funnel, debug_snapshot, price_snapshot = run_screener(
            scan_tickers, params, progress_bar, status_text
        )
        progress_bar.empty()

        st.session_state["results"] = df_results
        st.session_state["funnel"] = funnel
        st.session_state["debug_snapshot"] = debug_snapshot
        st.session_state["price_snapshot"] = price_snapshot
        st.session_state["scan_ts"] = datetime.now()
        st.session_state["scanned_total"] = len(scan_tickers)

        if not df_results.empty:
            st.success(f"✅ **{len(df_results)} candidatos** encontrados sobre {len(scan_tickers):,} tickers analizados")
        else:
            st.warning("⚠️ Ningún ticker cumplió todos los filtros. Mira el embudo de diagnóstico más abajo.")

    st.divider()

    # ── Embudo de diagnóstico ─────────────────────────────────────────
    if "funnel" in st.session_state:
        with st.expander("🔎 Embudo de diagnóstico (cuántos tickers se descartaron en cada paso)"):
            funnel = st.session_state["funnel"]
            total_scanned = st.session_state["scanned_total"]
            rows = []
            for code in REASON_ORDER:
                n = funnel.get(code, 0)
                if n == 0 and code != "ok":
                    continue
                pct = round(n / total_scanned * 100, 1) if total_scanned else 0
                rows.append({"Motivo": REASON_LABELS[code], "Tickers": n, "% del universo": pct})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            debug_snapshot = st.session_state.get("debug_snapshot", {})
            if debug_snapshot:
                with st.expander("🐛 Ver mensajes de error reales (muestras)"):
                    for code in REASON_ORDER:
                        msgs = debug_snapshot.get(code)
                        if not msgs:
                            continue
                        st.markdown(f"**{REASON_LABELS[code]}**")
                        for m in msgs:
                            st.code(m, language=None)
        st.divider()

    render_results()

# ======================================================================
# 9. RESULTADOS
# ======================================================================

def render_results():
    st.markdown("### 📊 Resultados")

    if "results" not in st.session_state or st.session_state["results"].empty:
        st.info("👆 Configura los parámetros y pulsa **INICIAR ESCANEO**.")
        return

    df_all = st.session_state["results"]
    ts = st.session_state["scan_ts"]

    st.markdown("#### 🔧 Filtros adicionales sobre resultados")
    st.caption("Se aplican sobre los candidatos ya encontrados — no relanzan la descarga de datos.")
    colrf1, colrf2, colrf3, colrf4 = st.columns(4)
    with colrf1:
        use_min_otm_filter = st.checkbox("Distancia OTM mínima (%)", value=False)
        min_otm_value = st.number_input("Valor mínimo (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.5,
                                         disabled=not use_min_otm_filter, key="min_otm_value")
    with colrf2:
        use_max_delta_filter = st.checkbox("|Delta| máximo", value=False)
        max_delta_value = st.number_input("Valor máximo", min_value=0.0, max_value=1.0, value=0.30, step=0.01,
                                           disabled=not use_max_delta_filter, key="max_delta_value",
                                           help="Filas sin delta calculable se excluyen al activar este filtro.")
    with colrf3:
        use_min_oi_post_filter = st.checkbox("OI mínimo", value=False)
        min_oi_post_value = st.number_input("Valor mínimo", min_value=0, max_value=100000, value=100, step=50,
                                             disabled=not use_min_oi_post_filter, key="min_oi_post_value")
    with colrf4:
        use_min_vol_post_filter = st.checkbox("Volumen mínimo", value=False)
        min_vol_post_value = st.number_input("Valor mínimo", min_value=0, max_value=100000, value=10, step=10,
                                              disabled=not use_min_vol_post_filter, key="min_vol_post_value")

    df = df_all.copy()
    if use_min_otm_filter:
        df = df[df["OTM_%"] >= min_otm_value]
    if use_max_delta_filter:
        df = df[df["Delta"].notna() & (df["Delta"].abs() <= max_delta_value)]
    if use_min_oi_post_filter:
        df = df[df["OI"] >= min_oi_post_value]
    if use_min_vol_post_filter:
        df = df[df["Volumen"] >= min_vol_post_value]

    if use_min_otm_filter or use_max_delta_filter or use_min_oi_post_filter or use_min_vol_post_filter:
        st.caption(f"ℹ️ Mostrando **{len(df)}** de {len(df_all)} candidatos tras aplicar estos filtros.")

    st.divider()

    if df.empty:
        st.warning("⚠️ Ningún candidato del último escaneo cumple estos filtros adicionales. Ajusta los valores de arriba.")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📊 Candidatos totales", len(df))
    k2.metric("🛡️ OTM máx.", f"{df['OTM_%'].max():.2f}%")
    k3.metric("💰 Prima media", f"{df['Extrínseco_%'].mean():.3f}%")
    k4.metric("🕐 Escaneo", ts.strftime("%H:%M:%S"))

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📋 Ranking Completo", "🔍 Detalle", "📈 Gráficos"])

    with tab1:
        cols_show = [
            "Rank", "Ticker", "Precio", "Strike", "OTM_%", "Extrínseco_%",
            "Prima_Mid", "Bid", "Ask", "Capital_Requerido", "Retorno_Capital_%",
            "Delta", "DTE", "Vencimiento", "IV_%", "RV_%", "IV_RV",
            "OI", "Volumen", "Spread_%", "SMA30", "Dist_SMA30_%",
            "SMA200", "Dist_SMA200_%", "PCR",
        ]
        cols_show = [c for c in cols_show if c in df.columns]

        def color_otm(val):
            if val >= 15:
                return "background-color:#1e3a1e; color:#6daa45"
            if val >= 10:
                return "background-color:#3a3a1e; color:#e8af34"
            return ""

        def color_iv_rv(val):
            try:
                if val is None or pd.isna(val):
                    return ""
                if val >= 2.5:
                    return "background-color:#3a1e1e; color:#dd6974"
                if val >= 1.8:
                    return "background-color:#3a3a1e; color:#e8af34"
            except Exception:
                pass
            return ""

        styler = df[cols_show].style.map(color_otm, subset=["OTM_%"])
        if "IV_RV" in cols_show:
            styler = styler.map(color_iv_rv, subset=["IV_RV"])

        st.dataframe(styler, use_container_width=True, height=550)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar CSV", csv, f"csp_screener_{ts.strftime('%Y%m%d_%H%M')}.csv", "text/csv")

    with tab2:
        selected = st.selectbox("Selecciona un ticker para ver el detalle", options=df["Ticker"].tolist())
        if selected:
            row = df[df["Ticker"] == selected].iloc[0]
            st.markdown(f"## {selected} — Cash Secured Put")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("### 📌 Trade Setup")
                st.markdown(f"""
| Concepto | Valor |
|---|---|
| 💲 Precio subyacente | **${row['Precio']}** |
| 🎯 Strike | **${row['Strike']}** |
| 📅 Vencimiento | **{row['Vencimiento']}** (DTE: {row['DTE']}d) |
| 💵 Prima Mid | **${row['Prima_Mid']}** |
| 📊 Bid / Ask | ${row['Bid']} / ${row['Ask']} |
| 🛡️ Distancia OTM | **{row['OTM_%']}%** |
| 💰 Prima (% del precio) | {row['Extrínseco_%']}% |
| 📐 Delta | {row['Delta']} |
""")
            with col_b:
                st.markdown("### 📊 Contexto")
                st.markdown(f"""
| Concepto | Valor |
|---|---|
| 💵 Capital requerido (1 contrato) | **${row['Capital_Requerido']:,.2f}** |
| 🔄 Retorno sobre capital | {row['Retorno_Capital_%']}% |
| 📈 IV / RV | {row['IV_%']}% / {row['RV_%']}% (ratio: {row['IV_RV']}) |
| 💧 OI / Volumen | {row['OI']:,} / {row['Volumen']:,} |
| 〰️ Spread bid-ask | {row['Spread_%']}% |
| 📐 SMA30 | {row['SMA30']} ({row['Dist_SMA30_%']}%) |
| 📐 SMA200 | {row['SMA200']} ({row['Dist_SMA200_%']}%) |
| 🗳️ PCR | {row['PCR']} |
""")
            st.markdown("---")
            st.markdown("### 💡 Interpretación")
            otm = row["OTM_%"]
            if otm >= 15:
                st.success(f"🟢 **Margen de seguridad alto**: el precio puede caer un {otm:.1f}% antes de que el put entre ITM.")
            elif otm >= 8:
                st.warning(f"🟡 **Margen moderado**: {otm:.1f}% de distancia hasta el strike.")
            else:
                st.error(f"🔴 **Margen bajo**: solo {otm:.1f}% de distancia hasta el strike.")
            st.info(
                f"Vendiendo este put cobras **${row['Prima_Mid']}** por acción "
                f"(${row['Prima_Mid']*100:.2f} por contrato), reservando "
                f"**${row['Capital_Requerido']:,.2f}** en efectivo. Si el subyacente "
                f"cierra por encima del strike **${row['Strike']}** al vencimiento, "
                f"te quedas la prima íntegra sin que te asignen las acciones."
            )

    with tab3:
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            ticker_chart = st.selectbox("Ver gráfico de precio", options=df["Ticker"].tolist(), key="chart_select")
            if ticker_chart:
                fig = plot_price(ticker_chart)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No se pudo cargar el gráfico.")
        with col_g2:
            fig_scatter = px.scatter(
                df, x="Extrínseco_%", y="OTM_%", text="Ticker", color="OTM_%",
                color_continuous_scale=["#dd6974", "#e8af34", "#6daa45"],
                title="Distancia OTM vs Prima%", template="plotly_dark", height=420,
                labels={"Extrínseco_%": "Prima (%)", "OTM_%": "Distancia OTM (%)"},
            )
            fig_scatter.update_traces(textposition="top center", marker_size=10)
            st.plotly_chart(fig_scatter, use_container_width=True)

        fig_bar = px.bar(
            df.head(20), x="Ticker", y="OTM_%", color="OTM_%",
            color_continuous_scale=["#dd6974", "#e8af34", "#6daa45"],
            title="Top 20 — Distancia OTM % (mayor = más seguro)",
            template="plotly_dark", height=400,
            labels={"OTM_%": "Distancia OTM (%)"},
        )
        st.plotly_chart(fig_bar, use_container_width=True)


if __name__ == "__main__":
    main()
