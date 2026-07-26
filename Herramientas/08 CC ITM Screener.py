"""
Deep ITM Covered Call Screener
================================
Objetivo: encontrar covered calls deep ITM con extrínseco 0.85%-1.00% semanal.

FILTROS DUROS (todos activables/desactivables desde la UI):
- Precio del subyacente: rango configurable
- Close > SMA30 (tendencia alcista)             [toggle]
- PCR < 1.0 (sesgo alcista)                       [toggle]
- OI > 100 (liquidez mínima)
- Sin earnings en los próximos 7 días             [toggle]
- Extrínseco entre 0.85% y 1.00% del subyacente   [toggle: modo diagnóstico lo ignora]
- Bid > 0 (opción negociable)
- Vencimiento = próximo viernes (DTE ≤ 7)

RANKING: mayor downside protection % primero (más deep ITM = más protección)
PRECIO DE OPCIÓN: midprice (bid+ask)/2 siempre

ARQUITECTURA (v3) — CAMBIO IMPORTANTE
--------------------------------------
Versiones anteriores mezclaban, dentro del mismo ThreadPoolExecutor, llamadas
a yf.download() (descarga masiva) CON llamadas a stock.options /
stock.option_chain() / stock.calendar (API por objeto Ticker) — dos rutas de
yfinance distintas golpeando la red concurrentemente desde varios hilos. Eso
producía DataFrames con la forma correcta pero Close = NaN para tickers tan
líquidos como AAPL, un patrón de corrupción silenciosa, no un simple 429.

Ahora el pipeline está separado en DOS FASES, como en el screener de
tendencia que sí funciona:

  FASE 1 (paralela, con lock) — SOLO yf.download() de precio diario.
      Aquí se filtra por rango de precio y SMA30.
  FASE 2 (secuencial, sin threads) — SOLO llamadas a Ticker/options,
      una detrás de otra, únicamente sobre los supervivientes de la fase 1.
      Aquí se filtra por earnings, PCR y se busca el candidato ITM.

Esto reduce drásticamente cuántos tickers necesitan la fase 2 (más lenta),
y evita mezclar las dos rutas de red concurrentemente.
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

warnings.filterwarnings('ignore')

# Logs visibles en Streamlit Cloud: menú "Manage app" (abajo a la derecha
# de la app desplegada) → pestaña "Logs". Ahí se ve esto en tiempo real,
# server-side, independientemente de lo que pinte la UI.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("cc_itm_screener")
logger.info(f"yfinance version = {getattr(yf, '__version__', 'desconocida')}")

# Sin esto, una llamada a Yahoo que se quede colgada (sin dar error ni
# timeout) bloquea indefinidamente el socket subyacente. Como la Fase 1
# serializa las descargas con un Lock, UN solo ticker colgado deja a TODOS
# los hilos esperando para siempre — eso es lo que produce la barra de
# progreso congelada. Este timeout global convierte cualquier cuelgue de
# red en una excepción capturable en como mucho SOCKET_TIMEOUT segundos.
SOCKET_TIMEOUT = 15
socket.setdefaulttimeout(SOCKET_TIMEOUT)

# Lock global para las descargas de precio en la Fase 1 — mismo patrón que
# el screener de tendencia que ya funciona en producción.
_yfinance_lock = Lock()

_N = NormalDist()
RISK_FREE_RATE = 0.045

# ── Diagnóstico: motivos de descarte y muestras de error reales ────────
REASON_ORDER = [
    "ok",
    "no_daily_data",
    "price_out_of_range",
    "sma30_unavailable",
    "below_sma30",
    "earnings_this_week",
    "no_expirations",
    "no_friday_expiration",
    "no_calls_chain",
    "pcr_bearish",
    "no_itm_candidate",
    "error",
]

REASON_LABELS = {
    "ok":                    "✅ Pasó todos los filtros",
    "no_daily_data":         "Sin datos diarios (Fase 1 — yf.download)",
    "price_out_of_range":    "Precio fuera de rango (Fase 1)",
    "sma30_unavailable":     "No hay suficiente histórico para SMA30 (Fase 1)",
    "below_sma30":           "Precio ≤ SMA30 — no alcista (Fase 1)",
    "earnings_this_week":    "Earnings en los próximos 7 días (Fase 2)",
    "no_expirations":        "Sin vencimientos de opciones listados (Fase 2)",
    "no_friday_expiration":  "Sin vencimiento viernes con DTE≤7 (Fase 2)",
    "no_calls_chain":        "Cadena de calls vacía/no disponible (Fase 2)",
    "pcr_bearish":           "PCR ≥ 1.0 — sesgo bajista (Fase 2)",
    "no_itm_candidate":      "Sin strike ITM que cumpla extrínseco/OI/bid (Fase 2)",
    "error":                 "Excepción no controlada",
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


def _with_timeout(fn, args=(), kwargs=None, timeout=SOCKET_TIMEOUT + 5):
    """
    DESACTIVADO (v3.1): crear un ThreadPoolExecutor nuevo en cada llamada
    a Yahoo (una por ticker en Fase 1, hasta 3 más por superviviente en
    Fase 2) genera ráfagas de cientos de hilos en un escaneo normal — en
    Streamlit Community Cloud (CPU compartida) eso dispara el throttling
    de la plataforma ("Your app has been throttled"), que a su vez puede
    degradar/romper la ejecución de formas que parecen bugs de datos pero
    no lo son. socket.setdefaulttimeout() ya acota los cuelgues de red sin
    coste de hilos adicionales, así que simplemente ejecutamos fn directo.
    """
    kwargs = kwargs or {}
    return fn(*args, **kwargs)


# ======================================================================
# 0. UNIVERSO
# ======================================================================

def _clean_ticker(raw):
    if raw is None:
        return None
    t = str(raw).strip().upper().replace("&amp;", "&")
    if not t or t in ("-", "NAN", "N/A", ""):
        return None
    return t.replace(" ", "")


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
# 1. DATOS DIARIOS (Fase 1 — mismo patrón que el screener de tendencia)
# ======================================================================

def get_daily_data(ticker, use_lock=True):
    """Descarga precio diario. Mismo patrón exacto que download_weekly_data()
    del screener de tendencia (que sí funciona): un solo yf.download(),
    protegido por lock cuando se llama desde threads, sin mezclarlo con
    ninguna otra API de yfinance."""
    try:
        end   = datetime.now() + timedelta(days=1)
        start = end - timedelta(days=120)
        logger.info(f"[{ticker}] descarga: start={start.date()} end={end.date()} use_lock={use_lock}")

        def _do_download():
            return yf.download(
                ticker, start=start, end=end,
                interval="1d", auto_adjust=False,
                multi_level_index=False, progress=False,
            )

        if use_lock:
            with _yfinance_lock:
                data = _with_timeout(_do_download)
        else:
            data = _with_timeout(_do_download)

        if data is None or data.empty or len(data) < 35:
            logger.warning(f"[{ticker}] descarga vacía o insuficiente: "
                            f"{'None' if data is None else len(data)} filas")
            _record_debug("no_daily_data", f"{ticker}: descarga vacía o insuficiente histórico")
            return None

        n_before = len(data)
        tail_preview = data["Close"].tail(3).to_dict()
        logger.info(f"[{ticker}] filas={n_before} columnas={list(data.columns)} "
                    f"dtypes_close={data['Close'].dtype} tail3={tail_preview}")
        data = data.dropna(subset=["Close"])
        n_after = len(data)
        if data.empty or len(data) < 35:
            logger.warning(f"[{ticker}] tras dropna quedan {n_after}/{n_before} filas — descartado")
            _record_debug(
                "no_daily_data",
                f"{ticker}: filas antes={n_before} después de limpiar NaN={n_after} · "
                f"últimas 3 Close (crudas)={tail_preview}"
            )
            return None

        logger.info(f"[{ticker}] OK — último close válido = {data['Close'].iloc[-1]}")
        data.index = pd.to_datetime(data.index)
        return data
    except Exception as e:
        logger.error(f"[{ticker}] EXCEPCIÓN: {type(e).__name__}: {e}")
        _record_debug("no_daily_data", f"{ticker}: {e}")
        return None


# ======================================================================
# 2. SMA30
# ======================================================================

def get_sma30(close):
    """Devuelve (sma30_valor, dist_pct, slope_positivo) o (None, None, None)."""
    try:
        if len(close) < 35:
            return None, None, None
        sma      = close.rolling(30).mean()
        sma_now  = float(sma.iloc[-1])
        sma_prev = float(sma.iloc[-6])
        price    = float(close.iloc[-1])
        dist_pct = round((price - sma_now) / sma_now * 100, 2)
        slope    = sma_now > sma_prev
        return round(sma_now, 2), dist_pct, slope
    except Exception:
        return None, None, None


# ======================================================================
# 3. VOLATILIDAD REALIZADA
# ======================================================================

def get_rv10(close):
    try:
        if len(close) < 12:
            return None
        rets = np.log(close / close.shift(1)).dropna()
        return round(float(rets.iloc[-10:].std() * np.sqrt(252) * 100), 2)
    except Exception:
        return None


# ======================================================================
# 4. BLACK-SCHOLES DELTA
# ======================================================================

def bs_delta(S, K, T_years, sigma, r=RISK_FREE_RATE):
    try:
        if T_years <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return None
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_years) / (sigma * np.sqrt(T_years))
        return round(float(_N.cdf(d1)), 3)
    except Exception:
        return None


# ======================================================================
# 5. PRÓXIMO VIERNES (DTE ≤ 7)
# ======================================================================

def next_friday():
    today = date.today()
    days_ahead = (4 - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return today + timedelta(days=days_ahead)


def select_friday_expiration(expirations):
    target = next_friday()
    today  = date.today()
    for exp_str in expirations:
        exp = datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp - today).days
        if exp == target and 1 <= dte <= 7:
            return exp_str, dte
    for exp_str in expirations:
        exp = datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp - today).days
        if exp.weekday() == 4 and 1 <= dte <= 7:
            return exp_str, dte
    return None, None


# ======================================================================
# 6. EARNINGS PRÓXIMOS 7 DÍAS
# ======================================================================

def has_earnings_this_week(stock):
    try:
        cal = _with_timeout(lambda: stock.calendar)
        if cal is None:
            return False
        ed = cal.get("Earnings Date")
        if ed is None:
            return False
        if isinstance(ed, list):
            ed = ed[0]
        ed = pd.to_datetime(ed).date()
        return date.today() <= ed <= (date.today() + timedelta(days=7))
    except Exception:
        return False


# ======================================================================
# 7. PUT/CALL RATIO (sobre una cadena ya descargada, sin red)
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
# 8. CANDIDATO DEEP ITM
# ======================================================================

def find_deep_itm_candidate(calls_df, current_price, dte_calendar,
                             extrinsic_min_pct, extrinsic_max_pct,
                             min_oi, diagnostic_mode=False):
    try:
        itm = calls_df[calls_df["strike"] < current_price].copy()
        if itm.empty:
            return None

        itm["bid"] = pd.to_numeric(itm["bid"], errors="coerce").fillna(0)
        itm["ask"] = pd.to_numeric(itm["ask"], errors="coerce").fillna(0)
        itm["mid"] = (itm["bid"] + itm["ask"]) / 2

        itm = itm[(itm["bid"] > 0) & (itm["mid"] > 0)]
        if itm.empty:
            return None

        itm["oi"] = pd.to_numeric(
            itm.get("openInterest", pd.Series(0, index=itm.index)), errors="coerce"
        ).fillna(0)
        itm = itm[itm["oi"] >= min_oi]
        if itm.empty:
            return None

        itm["intrinsic"]     = current_price - itm["strike"]
        itm["extrinsic"]     = itm["mid"] - itm["intrinsic"]
        itm["extrinsic_pct"] = itm["extrinsic"] / current_price * 100
        itm["spread_pct"]    = (itm["ask"] - itm["bid"]) / itm["mid"] * 100
        itm["downside_prot"] = itm["intrinsic"] / current_price * 100

        itm = itm[itm["extrinsic"] > 0]
        if itm.empty:
            return None

        if diagnostic_mode:
            candidates = itm
        else:
            candidates = itm[
                (itm["extrinsic_pct"] >= extrinsic_min_pct) &
                (itm["extrinsic_pct"] <= extrinsic_max_pct)
            ]
        if candidates.empty:
            return None

        best = candidates.sort_values("downside_prot", ascending=False).iloc[0]

        T_years = dte_calendar / 365.0
        iv      = float(best.get("impliedVolatility") or 0)
        delta   = bs_delta(current_price, float(best["strike"]), T_years, iv) if iv > 0 else None

        return {
            "strike":         float(best["strike"]),
            "mid":            round(float(best["mid"]), 2),
            "bid":            round(float(best["bid"]), 2),
            "ask":            round(float(best["ask"]), 2),
            "intrinsic":      round(float(best["intrinsic"]), 2),
            "extrinsic":      round(float(best["extrinsic"]), 2),
            "extrinsic_pct":  round(float(best["extrinsic_pct"]), 3),
            "downside_prot":  round(float(best["downside_prot"]), 2),
            "spread_pct":     round(float(best["spread_pct"]), 2),
            "oi":             int(best["oi"]),
            "volume":         int(pd.to_numeric(best.get("volume", 0), errors="coerce") or 0),
            "iv_pct":         round(iv * 100, 2) if iv > 0 else None,
            "delta":          delta,
            "in_target_band": bool(extrinsic_min_pct <= float(best["extrinsic_pct"]) <= extrinsic_max_pct),
        }
    except Exception:
        return None


# ======================================================================
# 9a. FASE 1 — precio diario + SMA30 (paralela)
# ======================================================================

def phase1_price_filter(ticker, params):
    """Solo yf.download(). Devuelve (registro_o_None, reason)."""
    try:
        data = get_daily_data(ticker, use_lock=True)
        if data is None:
            return None, "no_daily_data"

        close         = data["Close"]
        current_price = float(close.iloc[-1])
        _record_price(ticker, current_price)

        if not (params["min_price"] <= current_price <= params["max_price"]):
            _record_debug("price_out_of_range", f"{ticker}: precio calculado = {current_price}")
            return None, "price_out_of_range"

        sma30, dist_sma_pct, slope_up = get_sma30(close)
        if sma30 is None:
            return None, "sma30_unavailable"

        if params["use_sma_filter"] and current_price <= sma30:
            return None, "below_sma30"

        rv = get_rv10(close)

        return {
            "Ticker": ticker,
            "current_price": round(current_price, 2),
            "sma30": sma30,
            "dist_sma_pct": dist_sma_pct,
            "slope_up": slope_up,
            "rv": rv,
        }, "ok"
    except Exception as e:
        _record_debug("error", f"{ticker}: {e}")
        return None, "error"


# ======================================================================
# 9b. FASE 2 — opciones (SECUENCIAL, sin threads)
# ======================================================================

def phase2_options_filter(survivor, params):
    """Recibe el registro de la Fase 1 y añade las métricas de opciones.
    Se llama en un bucle for normal, nunca dentro de un ThreadPoolExecutor,
    para no mezclar esta API de yfinance con las descargas paralelas."""
    ticker        = survivor["Ticker"]
    current_price = survivor["current_price"]
    try:
        stock = yf.Ticker(ticker)

        if params["use_earnings_filter"] and has_earnings_this_week(stock):
            return None, "earnings_this_week"

        expirations = _with_timeout(lambda: stock.options)
        if not expirations:
            return None, "no_expirations"

        exp_str, dte = select_friday_expiration(expirations)
        if exp_str is None:
            return None, "no_friday_expiration"

        chain = _with_timeout(lambda: stock.option_chain(exp_str))
        calls = chain.calls
        puts  = chain.puts
        if calls is None or calls.empty:
            return None, "no_calls_chain"

        pcr = compute_pcr(calls, puts, current_price)
        if params["use_pcr_filter"] and pcr is not None and pcr >= 1.0:
            return None, "pcr_bearish"

        candidate = find_deep_itm_candidate(
            calls, current_price, dte,
            params["extrinsic_min"], params["extrinsic_max"],
            params["min_oi"], diagnostic_mode=params["diagnostic_mode"],
        )
        if candidate is None:
            return None, "no_itm_candidate"

        iv_rv_ratio = (
            round(candidate["iv_pct"] / survivor["rv"], 3)
            if (candidate["iv_pct"] and survivor["rv"] and survivor["rv"] > 0) else None
        )
        annualized = round(candidate["extrinsic_pct"] * (365 / dte), 1) if dte > 0 else None
        breakeven  = round(current_price - candidate["mid"], 2)

        result = {
            "Ticker": ticker,
            "Precio": current_price,
            "Vencimiento": exp_str,
            "DTE": dte,
            "Strike": candidate["strike"],
            "Downside_Prot_%": candidate["downside_prot"],
            "Extrínseco_%": candidate["extrinsic_pct"],
            "En_Banda": candidate["in_target_band"],
            "Prima_Mid": candidate["mid"],
            "Bid": candidate["bid"],
            "Ask": candidate["ask"],
            "Intrínseco": candidate["intrinsic"],
            "Extrínseco_$": candidate["extrinsic"],
            "Breakeven": breakeven,
            "Delta": candidate["delta"],
            "IV_%": candidate["iv_pct"],
            "RV_%": survivor["rv"],
            "IV_RV": iv_rv_ratio,
            "Ret_Anualizado_%": annualized,
            "OI": candidate["oi"],
            "Volumen": candidate["volume"],
            "Spread_%": candidate["spread_pct"],
            "SMA30": survivor["sma30"],
            "Dist_SMA30_%": survivor["dist_sma_pct"],
            "SMA30_Sube": survivor["slope_up"],
            "PCR": pcr,
        }
        return result, "ok"
    except Exception as e:
        _record_debug("error", f"{ticker}: {e}")
        return None, "error"


# ======================================================================
# 10. ORQUESTADOR: Fase 1 (paralela) → Fase 2 (secuencial)
# ======================================================================

def run_screener(tickers, params, progress_bar, status_text):
    _reset_debug()
    funnel = {r: 0 for r in REASON_ORDER}
    total  = len(tickers)

    # ── FASE 1: precio + SMA30, en paralelo (solo yf.download, con lock) ──
    status_text.text(f"🔍 Fase 1/2 — precio y tendencia: 0/{total}")
    survivors = []
    with ThreadPoolExecutor(max_workers=params.get("max_workers", 6)) as executor:
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

    # ── FASE 2: opciones, SECUENCIAL, solo sobre supervivientes ────────
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
        df = df.sort_values("Downside_Prot_%", ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))

    with _debug_lock:
        debug_snapshot = {k: list(v) for k, v in _debug_samples.items()}
        price_snapshot = list(_price_samples)

    return df, funnel, debug_snapshot, price_snapshot


# ======================================================================
# 11. GRÁFICO DE PRECIO
# ======================================================================

def plot_price(ticker):
    try:
        data = get_daily_data(ticker, use_lock=False)
        if data is None:
            return None
        sma = data["Close"].rolling(30).mean()
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data["Open"], high=data["High"],
            low=data["Low"], close=data["Close"],
            name=ticker,
            increasing_line_color="#6daa45",
            decreasing_line_color="#dd6974",
        ))
        fig.add_trace(go.Scatter(
            x=data.index, y=sma,
            mode="lines", name="SMA30",
            line=dict(color="#e8af34", width=2),
        ))
        fig.update_layout(
            title=f"{ticker} — Precio + SMA30",
            template="plotly_dark",
            height=420,
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
        )
        return fig
    except Exception:
        return None


# ======================================================================
# 12. INTERFAZ STREAMLIT
# ======================================================================

def main():
    st.set_page_config(page_title="Deep ITM CC Screener", page_icon="🎯", layout="wide")

    if not check_password():
        st.stop()

    st.title("🎯 Deep ITM Covered Call Screener")
    st.markdown(
        "**Objetivo: extrínseco 0.85%-1.00% semanal · "
        "Vencimiento próximo viernes · "
        "Ranking por mayor downside protection**"
    )
    st.caption("⚙️ v3 — pipeline en dos fases (precio en paralelo → opciones secuencial)")

    # ── Test de conectividad directo, sin ninguna capa nuestra ─────────
    with st.expander("🧪 Test de conectividad directo (1 ticker, sin lock/threads/wrapper)"):
        st.caption(
            f"yfinance instalado: **{getattr(yf, '__version__', 'desconocida')}** · "
            "Esto llama a yf.download() puro, tal cual, para ver el dato crudo. "
            "Los logs de esta prueba también salen en Streamlit Cloud → Manage app → Logs."
        )
        test_ticker = st.text_input("Ticker a probar", value="AAPL", key="raw_test_ticker")
        if st.button("▶️ Probar ahora", key="raw_test_btn"):
            t = _clean_ticker(test_ticker) or "AAPL"

            st.markdown("**Método 1: `yf.download()`**")
            try:
                raw1 = yf.download(t, period="1mo", interval="1d", progress=False)
                logger.info(f"[TEST] yf.download({t}) shape={raw1.shape if raw1 is not None else None}")
                if raw1 is None or raw1.empty:
                    st.error("Devolvió None o vacío.")
                else:
                    st.write(f"Shape: {raw1.shape} · Columnas: {list(raw1.columns)} · dtype Close: {raw1['Close'].dtype}")
                    st.dataframe(raw1.tail(5))
                    n_nan = raw1["Close"].isna().sum()
                    st.write(f"Valores NaN en Close: {n_nan} / {len(raw1)}")
            except Exception as e:
                logger.error(f"[TEST] yf.download({t}) EXCEPCIÓN: {type(e).__name__}: {e}")
                st.error(f"Excepción: {type(e).__name__}: {e}")

            st.markdown("**Método 2: `yf.Ticker().history()`** (ruta distinta dentro de yfinance)")
            try:
                raw2 = yf.Ticker(t).history(period="1mo", interval="1d")
                logger.info(f"[TEST] Ticker({t}).history() shape={raw2.shape if raw2 is not None else None}")
                if raw2 is None or raw2.empty:
                    st.error("Devolvió None o vacío.")
                else:
                    st.write(f"Shape: {raw2.shape} · Columnas: {list(raw2.columns)}")
                    st.dataframe(raw2.tail(5))
                    n_nan2 = raw2["Close"].isna().sum()
                    st.write(f"Valores NaN en Close: {n_nan2} / {len(raw2)}")
            except Exception as e:
                logger.error(f"[TEST] Ticker({t}).history() EXCEPCIÓN: {type(e).__name__}: {e}")
                st.error(f"Excepción: {type(e).__name__}: {e}")

            st.markdown("**Método 3: `yf.Ticker().fast_info`** (endpoint de solo precio actual, distinto de los dos anteriores)")
            try:
                fi = yf.Ticker(t).fast_info
                last_price = fi.get("lastPrice") if hasattr(fi, "get") else getattr(fi, "last_price", None)
                logger.info(f"[TEST] fast_info({t}) lastPrice={last_price}")
                st.write(f"last_price: {last_price}")
            except Exception as e:
                logger.error(f"[TEST] fast_info({t}) EXCEPCIÓN: {type(e).__name__}: {e}")
                st.error(f"Excepción: {type(e).__name__}: {e}")

    st.divider()

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
    meta        = st.session_state["meta_universe"]
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
        st.markdown("**💰 Extrínseco objetivo**")
        extrinsic_min, extrinsic_max = st.slider(
            "Rango extrínseco (% del precio)",
            min_value=0.50, max_value=2.00, value=(0.85, 1.00), step=0.05,
            help="Solo aplica si el modo diagnóstico está desactivado"
        )

        st.markdown("**💲 Precio del subyacente**")
        min_price, max_price = st.slider(
            "Rango de precio ($)", min_value=5, max_value=1000, value=(20, 500), step=5,
        )

    with c2:
        st.markdown("**💧 Liquidez mínima**")
        min_oi = st.number_input("OI mínimo del strike", min_value=10, max_value=10000, value=100, step=50)

        st.markdown("**🧵 Concurrencia (solo Fase 1)**")
        max_workers = st.slider(
            "Requests en paralelo",
            min_value=1, max_value=10, value=3,
            help="Solo afecta a la descarga de precios (Fase 1). La Fase 2 "
                 "(opciones) siempre corre secuencial, sin threads. Valores "
                 "altos pueden disparar el throttling de CPU de Streamlit "
                 "Cloud en la capa gratuita — si te throttlean, baja esto a 1-2."
        )

        st.markdown("**📅 Próximo viernes**")
        today    = date.today()
        friday   = next_friday()
        dte_days = (friday - today).days
        st.info(f"📅 Próximo viernes: **{friday.strftime('%d %b %Y')}** (DTE: {dte_days} días)")

    with c3:
        st.markdown("**🎚️ Filtros activables**")
        use_sma_filter = st.checkbox("Close > SMA30 (tendencia alcista)", value=True)
        use_pcr_filter = st.checkbox("PCR < 1.0 (sesgo alcista)", value=True)
        use_earnings_filter = st.checkbox("Excluir earnings próximos 7 días", value=True)
        diagnostic_mode = st.checkbox(
            "🔬 Modo diagnóstico (ignora banda de extrínseco)", value=False,
            help="Devuelve el mejor candidato ITM aunque su extrínseco no esté "
                 "en el rango objetivo, para ver los valores reales del mercado."
        )

        st.markdown("**🧪 Subconjunto de depuración**")
        use_custom_subset = st.checkbox(
            "Usar solo un subconjunto de tickers", value=True,
            help="Para iterar rápido en vez de escanear los 1000+ tickers cada vez."
        )
        custom_tickers_raw = st.text_input(
            "Tickers (separados por coma)",
            value="AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,JPM,V,MA,UNH,HD,PG,JNJ,XOM,BAC,KO,PEP,DIS,NFLX",
            disabled=not use_custom_subset,
        )

    params = {
        "extrinsic_min":       extrinsic_min,
        "extrinsic_max":       extrinsic_max,
        "min_price":           min_price,
        "max_price":           max_price,
        "min_oi":              min_oi,
        "use_sma_filter":      use_sma_filter,
        "use_pcr_filter":      use_pcr_filter,
        "use_earnings_filter": use_earnings_filter,
        "diagnostic_mode":     diagnostic_mode,
        "max_workers":         max_workers,
    }

    st.divider()

    # ── Escaneo ────────────────────────────────────────────────────────
    st.markdown("### 🚀 Ejecutar Escaneo")

    custom_subset_preview = [
        _clean_ticker(t) for t in custom_tickers_raw.split(",")
    ] if use_custom_subset else []
    custom_subset_preview = [t for t in custom_subset_preview if t]

    scan_btn = st.button(
        "🎯 INICIAR ESCANEO",
        type="primary",
        use_container_width=True,
        disabled=(len(custom_subset_preview) == 0) if use_custom_subset else (len(tickers_all) == 0),
    )

    est_n = len(custom_subset_preview) if use_custom_subset else len(tickers_all)
    st.caption(
        f"ℹ️ Se escanearán **{est_n:,}** tickers · Fase 1 en paralelo (rápida) "
        f"→ Fase 2 secuencial solo sobre los que sobrevivan a precio/SMA30 "
        f"(más lenta, ~1-2s por ticker)."
    )

    if scan_btn:
        scan_tickers = custom_subset_preview if use_custom_subset else tickers_all
        if not scan_tickers:
            st.error("⚠️ El subconjunto de tickers está vacío — revisa el campo de texto.")
            return

        progress_bar = st.progress(0)
        status_text  = st.empty()
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
            st.warning(
                "⚠️ Ningún ticker cumplió todos los filtros. Mira el embudo de "
                "diagnóstico más abajo para ver en qué fase/paso se están cayendo."
            )

    st.divider()

    # ── Embudo de diagnóstico ─────────────────────────────────────────
    if "funnel" in st.session_state:
        st.markdown("### 🔎 Embudo de diagnóstico")
        st.caption("Cuántos tickers se descartaron en cada paso del último escaneo.")
        funnel = st.session_state["funnel"]
        total_scanned = st.session_state["scanned_total"]
        rows = []
        for code in REASON_ORDER:
            n = funnel.get(code, 0)
            if n == 0 and code != "ok":
                continue
            pct = round(n / total_scanned * 100, 1) if total_scanned else 0
            rows.append({"Motivo": REASON_LABELS[code], "Tickers": n, "% del universo": pct})
        df_funnel = pd.DataFrame(rows)
        st.dataframe(df_funnel, use_container_width=True, hide_index=True)

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

        price_snapshot = st.session_state.get("price_snapshot", [])
        if price_snapshot:
            with st.expander("💲 Ver distribución real de precios obtenidos (Fase 1)"):
                df_prices = pd.DataFrame(price_snapshot, columns=["Ticker", "Precio"])
                pmin, pmax = df_prices["Precio"].min(), df_prices["Precio"].max()
                pmed = df_prices["Precio"].median()
                st.write(
                    f"**{len(df_prices)}** precios obtenidos · "
                    f"mín: **${pmin:.2f}** · mediana: **${pmed:.2f}** · máx: **${pmax:.2f}**"
                )
                colp1, colp2 = st.columns(2)
                with colp1:
                    st.markdown("Más bajos")
                    st.dataframe(df_prices.sort_values("Precio").head(15), hide_index=True, use_container_width=True)
                with colp2:
                    st.markdown("Más altos")
                    st.dataframe(df_prices.sort_values("Precio", ascending=False).head(15), hide_index=True, use_container_width=True)
        st.divider()

    # ── Resultados ─────────────────────────────────────────────────────
    st.markdown("### 📊 Resultados")

    if "results" not in st.session_state or st.session_state["results"].empty:
        st.info("👆 Configura los parámetros y pulsa **INICIAR ESCANEO**.")
        return

    df = st.session_state["results"]
    ts = st.session_state["scan_ts"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📊 Candidatos totales", len(df))
    k2.metric("🛡️ Downside prot. máx.", f"{df['Downside_Prot_%'].max():.2f}%")
    k3.metric("💰 Extrínseco medio", f"{df['Extrínseco_%'].mean():.3f}%")
    k4.metric("🕐 Escaneo", ts.strftime("%H:%M:%S"))

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📋 Ranking Completo", "🔍 Detalle", "📈 Gráficos"])

    with tab1:
        cols_show = [
            "Rank", "Ticker", "Precio", "Strike", "Downside_Prot_%",
            "Extrínseco_%", "En_Banda", "Prima_Mid", "Bid", "Ask",
            "Breakeven", "Delta", "DTE", "Vencimiento",
            "IV_%", "RV_%", "IV_RV", "Ret_Anualizado_%",
            "OI", "Volumen", "Spread_%",
            "SMA30", "Dist_SMA30_%", "SMA30_Sube",
            "PCR",
        ]
        cols_show = [c for c in cols_show if c in df.columns]

        def color_downside(val):
            if val >= 15:
                return "background-color:#1e3a1e; color:#6daa45"
            if val >= 10:
                return "background-color:#3a3a1e; color:#e8af34"
            return ""

        st.dataframe(
            df[cols_show].style.map(color_downside, subset=["Downside_Prot_%"]),
            use_container_width=True, height=550,
        )

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar CSV", csv,
            f"deep_itm_cc_{ts.strftime('%Y%m%d_%H%M')}.csv", "text/csv",
        )

    with tab2:
        selected = st.selectbox("Selecciona un ticker para ver el detalle", options=df["Ticker"].tolist())
        if selected:
            row = df[df["Ticker"] == selected].iloc[0]
            st.markdown(f"## {selected} — Deep ITM Covered Call")

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("### 📌 Trade Setup")
                st.markdown(f"""
| Concepto | Valor |
|---|---|
| 💲 Precio subyacente | **${row['Precio']}** |
| 🎯 Strike (deep ITM) | **${row['Strike']}** |
| 📅 Vencimiento | **{row['Vencimiento']}** (DTE: {row['DTE']}d) |
| 💵 Prima Mid | **${row['Prima_Mid']}** |
| 📊 Bid / Ask | ${row['Bid']} / ${row['Ask']} |
| 🔺 Intrínseco | ${row['Intrínseco']} |
| 🔹 Extrínseco | ${row['Extrínseco_$']} ({row['Extrínseco_%']}%) |
| 🎯 En banda objetivo | {"Sí" if row.get('En_Banda') else "No (modo diagnóstico)"} |
| 🛡️ Downside protection | **{row['Downside_Prot_%']}%** |
| ⚖️ Breakeven | ${row['Breakeven']} |
| 📐 Delta | {row['Delta']} |
""")

            with col_b:
                st.markdown("### 📊 Contexto")
                st.markdown(f"""
| Concepto | Valor |
|---|---|
| 📈 IV / RV | {row['IV_%']}% / {row['RV_%']}% (ratio: {row['IV_RV']}) |
| 🔄 Retorno anualizado | {row['Ret_Anualizado_%']}% |
| 💧 OI / Volumen | {row['OI']:,} / {row['Volumen']:,} |
| 〰️ Spread bid-ask | {row['Spread_%']}% |
| 📐 SMA30 | {row['SMA30']} ({row['Dist_SMA30_%']}% sobre SMA) |
| ↗️ SMA30 subiendo | {row['SMA30_Sube']} |
| 🗳️ PCR | {row['PCR']} |
""")

            st.markdown("---")
            st.markdown("### 💡 Interpretación")

            prot = row["Downside_Prot_%"]
            ext  = row["Extrínseco_%"]

            if prot >= 15:
                st.success(f"🟢 **Protección excelente**: el precio puede caer un {prot:.1f}% antes de que entres en pérdida.")
            elif prot >= 10:
                st.warning(f"🟡 **Protección moderada**: el precio puede caer un {prot:.1f}% antes de pérdida.")
            else:
                st.error(f"🔴 **Protección baja**: solo {prot:.1f}% de margen bajista.")

            st.info(
                f"Con un extrínseco del **{ext}%** sobre un precio de **${row['Precio']}**, "
                f"cobras **${row['Extrínseco_$']}** por acción de prima pura (valor tiempo). "
                f"Si el subyacente cierra por encima del strike **${row['Strike']}** el viernes, "
                f"te quedas esa prima íntegra."
            )

    with tab3:
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            ticker_chart = st.selectbox(
                "Ver gráfico de precio", options=df["Ticker"].tolist(), key="chart_select",
            )
            if ticker_chart:
                fig = plot_price(ticker_chart)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No se pudo cargar el gráfico.")

        with col_g2:
            fig_scatter = px.scatter(
                df, x="Extrínseco_%", y="Downside_Prot_%", text="Ticker",
                color="Downside_Prot_%",
                color_continuous_scale=["#dd6974", "#e8af34", "#6daa45"],
                title="Downside Protection vs Extrínseco%",
                template="plotly_dark", height=420,
                labels={"Extrínseco_%": "Extrínseco (%)", "Downside_Prot_%": "Downside Protection (%)"},
            )
            fig_scatter.update_traces(textposition="top center", marker_size=10)
            st.plotly_chart(fig_scatter, use_container_width=True)

        fig_bar = px.bar(
            df.head(20), x="Ticker", y="Downside_Prot_%",
            color="Downside_Prot_%",
            color_continuous_scale=["#dd6974", "#e8af34", "#6daa45"],
            title="Top 20 — Downside Protection % (mayor = más deep ITM)",
            template="plotly_dark", height=400,
            labels={"Downside_Prot_%": "Downside Protection (%)"},
        )
        st.plotly_chart(fig_bar, use_container_width=True)


if __name__ == "__main__":
    main()
