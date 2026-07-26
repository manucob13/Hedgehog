"""
Deep ITM Covered Call Screener
================================
Objetivo: encontrar covered calls deep ITM con extrínseco 0.85%-1.00% semanal.

FILTROS DUROS (todos activables/desactivables desde la UI salvo los marcados
[fijo], que siempre están activos porque protegen la validez del dato, no
son una preferencia de trading):
- Precio del subyacente: rango configurable
- Close > SMA30 (tendencia alcista)             [toggle]
- PCR < 1.0 (sesgo alcista)                       [toggle]
- OI > 100 (liquidez mínima)
- Sin earnings en los próximos 7 días             [toggle]
- Sin riesgo de dividendo (ex-div antes de vto.
  con dividendo >= extrínseco capturado)          [fijo]
- Spread bid-ask no superior al 50% del extrínseco
  capturado (si no, la prima "no es real")        [fijo]
- Extrínseco entre 0.85% y 1.00% del subyacente   [toggle: modo diagnóstico lo ignora]
- Bid > 0 y Ask > 0 (opción realmente cotizada)   [fijo]
- Vencimiento = próximo viernes (DTE ≤ 7)

RANKING: mayor downside protection % primero (más deep ITM = más protección)
PRECIO DE OPCIÓN: midprice (bid+ask)/2 siempre
PRECIO DEL SUBYACENTE PARA INTRÍNSECO/EXTRÍNSECO: precio en vivo (fast_info),
con fallback automático y transparente al último cierre histórico si no hay
precio en vivo disponible (fin de semana, fallo puntual de red, etc.)

ARQUITECTURA (v3.2) — CAMBIOS DE ESTA REVISIÓN
------------------------------------------------
1. PRECIO EN VIVO PARA EL CÁLCULO DE INTRÍNSECO/EXTRÍNSECO.
   Antes, current_price salía siempre del último Close histórico (día
   anterior o cierre ya viejo). Con una banda de extrínseco de solo
   ±0.15%, correr esto a media sesión del viernes con un precio de ayer
   podía desalinear completamente el cálculo. Ahora, en Fase 2, se pide
   stock.fast_info justo antes de construir el candidato — esto devuelve
   el precio en vivo si el mercado está abierto, y el último precio
   conocido (= último cierre) si está cerrado. Mismo código sirve para
   "viernes a media sesión" y "fin de semana", sin ramas especiales.
   El Close histórico se sigue usando (ahora ajustado, ver punto 4) solo
   para SMA30/RV10, donde no hace falta esa precisión al segundo.
2. FASE 1 REALMENTE PARALELA.
   El _yfinance_lock envolvía toda la descarga dentro de cada tarea del
   ThreadPoolExecutor, así que aunque hubiera N workers configurados solo
   una descarga corría a la vez — el paralelismo no existía en la
   práctica. Se ha quitado el lock: el propio ThreadPoolExecutor(max_workers=N)
   ya acota la concurrencia real a N descargas simultáneas, que es
   justamente lo que se buscaba. La concurrencia deja de ser un control
   expuesto en la UI (ver punto 2b) y pasa a ser un valor fijo interno
   conservador, para no generar throttling en Streamlit Community Cloud.
2b. Se retira de la UI el slider "Requests en paralelo": es un detalle de
   implementación, no una decisión de trading, y no debería exigir que el
   usuario entienda internals de threading para usar el screener. Sigue
   funcionando exactamente igual por debajo, con MAX_WORKERS fijado a un
   valor seguro.
3. FILTRO DE SPREAD RELATIVO AL EXTRÍNSECO.
   Un OI alto (acumulado histórico) no garantiza que el spread bid-ask
   actual sea razonable, sobre todo en deep ITM. Si el spread se come una
   parte grande del extrínseco que se supone que estás cobrando, la prima
   "real" capturable es mucho menor que la que muestra el mid price. Se
   añade un filtro fijo: el spread en dólares no puede superar el 50% del
   extrínseco en dólares del candidato.
4. AUTO-ADJUST EN LA SERIE DIARIA.
   get_daily_data() usaba auto_adjust=False. Si un ticker tuvo un split en
   los últimos 120 días, el Close crudo tiene un salto de escala que
   distorsiona la SMA30 (falsos "por debajo/encima de SMA30"). Ahora se
   pide la serie ajustada (auto_adjust=True) para el cálculo de
   SMA30/RV10/rango de precio en Fase 1.
5. FILTRO DE RIESGO DE DIVIDENDO.
   No basta con evitar earnings: en covered calls deep ITM, la asignación
   anticipada más probable ocurre justo antes de una fecha ex-dividendo
   cuando el extrínseco que le queda a la opción es menor que el
   dividendo a cobrar (a quien tiene la call comprada le compensa
   ejercer antes para cobrar el dividendo). Se añade una consulta de
   calendario de dividendos (ex-date) y del último dividendo pagado (como
   estimación del próximo importe); si la ex-date cae antes o el mismo
   día del vencimiento y el dividendo estimado es >= extrínseco del
   candidato, se descarta con el motivo "dividend_risk".
6. VALIDACIÓN DE ASK.
   Antes solo se exigía bid > 0. Si ask viene en 0/NaN (dato faltante, no
   spread real), el mid quedaba artificialmente bajo (mid = bid/2),
   pudiendo colar candidatos con extrínseco distorsionado. Ahora se exige
   también ask > 0.

Historial de diagnóstico previo (se mantiene por referencia, sigue siendo
la razón de fondo por la que el pipeline usa Ticker().history() y no
yf.download(), y por la que existe el socket.setdefaulttimeout()):
- yf.download() devolvía columnas mal aplanadas en esta instalación
  (AttributeError sobre 'Close' incluso con multi_level_index=False) —
  confirmado con el test de conectividad en vivo. Ticker().history() sí
  funciona limpio. → Todo el pipeline usa Ticker().history().
- La última fila de cada descarga puede ser la sesión de HOY sin cerrar
  (Close = NaN) — se filtra con dropna(subset=["Close"]).
- socket.setdefaulttimeout() como red de seguridad barata contra cuelgues
  de red sin usar un ThreadPoolExecutor nuevo por llamada.
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
# timeout) bloquea indefinidamente el socket subyacente. Este timeout
# global convierte cualquier cuelgue de red en una excepción capturable en
# como mucho SOCKET_TIMEOUT segundos.
SOCKET_TIMEOUT = 15
socket.setdefaulttimeout(SOCKET_TIMEOUT)

# Concurrencia de la Fase 1 (descarga de precios). Antes era un slider en
# la UI; se ha vuelto un valor fijo interno (ver punto 2/2b del docstring):
# es un detalle de implementación, no una decisión de trading, y un valor
# más alto en Streamlit Community Cloud (CPU compartida) puede disparar
# throttling de la plataforma. 4 es un punto conservador que da
# paralelismo real ahora que se ha quitado el lock que lo anulaba.
MAX_WORKERS = 4

# Spread bid-ask máximo permitido, como % del extrínseco en dólares del
# candidato. Si el spread se come más de esto, el extrínseco "de mid
# price" no es realmente capturable. Ver punto 3 del docstring.
SPREAD_MAX_PCT_OF_EXTRINSIC = 50.0

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
    "dividend_risk",
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
    "no_itm_candidate":      "Sin strike ITM que cumpla extrínseco/OI/bid/ask/spread (Fase 2)",
    "dividend_risk":         "Riesgo de asignación por dividendo antes del vencimiento (Fase 2)",
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


def _with_timeout(fn, args=(), kwargs=None):
    """Ejecuta fn directo. El acotado de cuelgues de red ya lo da
    socket.setdefaulttimeout() a nivel global; no hace falta un
    ThreadPoolExecutor nuevo por llamada (eso disparaba throttling de CPU
    en Streamlit Community Cloud, ver histórico de diagnóstico)."""
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
# 1. DATOS DIARIOS (Fase 1 — paralela de verdad, ver punto 2 del docstring)
# ======================================================================

def get_daily_data(ticker):
    """Descarga precio diario AJUSTADO (auto_adjust=True) vía
    yf.Ticker(ticker).history(): en esta instalación yf.download() devuelve
    columnas mal aplanadas incluso con multi_level_index=False (confirmado
    con el test de conectividad: AttributeError sobre 'Close'), mientras
    que Ticker().history() funciona limpio.

    auto_adjust=True (punto 4 del docstring): sin ajustar, un split en los
    últimos 120 días metía un salto de escala en el Close crudo que
    distorsionaba la SMA30. Esta serie ajustada es solo para SMA30/RV10 y
    para el filtro de rango de precio de Fase 1 — el precio que realmente
    se usa para intrínseco/extrínseco en Fase 2 es el precio en vivo
    (ver get_live_price)."""
    try:
        end   = datetime.now() + timedelta(days=1)
        start = end - timedelta(days=120)
        logger.info(f"[{ticker}] descarga: start={start.date()} end={end.date()}")

        def _do_download():
            return yf.Ticker(ticker).history(
                start=start, end=end, interval="1d", auto_adjust=True
            )

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
                f"últimas 3 Close (ajustadas)={tail_preview}"
            )
            return None

        logger.info(f"[{ticker}] OK — último close ajustado válido = {data['Close'].iloc[-1]}")
        data.index = pd.to_datetime(data.index)
        return data
    except Exception as e:
        logger.error(f"[{ticker}] EXCEPCIÓN: {type(e).__name__}: {e}")
        _record_debug("no_daily_data", f"{ticker}: {e}")
        return None


# ======================================================================
# 1b. PRECIO EN VIVO (punto 1 del docstring)
# ======================================================================

def get_live_price(stock, fallback_price):
    """Precio en vivo del subyacente vía fast_info, con fallback
    transparente al último cierre histórico si no está disponible.

    fast_info devuelve el último precio conocido tanto si el mercado está
    abierto (precio en vivo real) como cerrado (= último cierre) — por eso
    el mismo código sirve igual un viernes a media sesión que un fin de
    semana, sin necesidad de detectar en qué caso estamos."""
    try:
        fi = _with_timeout(lambda: stock.fast_info)
        lp = None
        for key in ("last_price", "lastPrice"):
            try:
                val = fi[key] if hasattr(fi, "__getitem__") else getattr(fi, key, None)
            except Exception:
                val = None
            if val is not None:
                lp = val
                break
        if lp is not None and float(lp) > 0:
            return float(lp)
    except Exception as e:
        logger.warning(f"fast_info falló, uso fallback histórico: {e}")
    return fallback_price


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
# 6. EARNINGS PRÓXIMOS 7 DÍAS Y RIESGO DE DIVIDENDO (punto 5 del docstring)
# ======================================================================

def get_earnings_and_dividend_info(stock):
    """Una sola función que reúne fecha de earnings y datos de dividendo,
    reutilizando stock.calendar para ambos y evitando llamadas de red
    duplicadas.

    Devuelve dict:
      - earnings_date: date o None
      - ex_div_date: date o None (próxima fecha ex-dividendo conocida)
      - div_amount: float o None (estimación = último dividendo pagado)
    """
    info = {"earnings_date": None, "ex_div_date": None, "div_amount": None}
    try:
        cal = _with_timeout(lambda: stock.calendar)
        if cal:
            ed = cal.get("Earnings Date")
            if isinstance(ed, list):
                ed = ed[0] if ed else None
            if ed is not None:
                try:
                    info["earnings_date"] = pd.to_datetime(ed).date()
                except Exception:
                    pass

            exd = cal.get("Ex-Dividend Date")
            if isinstance(exd, list):
                exd = exd[0] if exd else None
            if exd is not None:
                try:
                    info["ex_div_date"] = pd.to_datetime(exd).date()
                except Exception:
                    pass
    except Exception:
        pass

    try:
        divs = _with_timeout(lambda: stock.dividends)
        if divs is not None and not divs.empty:
            info["div_amount"] = float(divs.iloc[-1])
    except Exception:
        pass

    return info


def has_earnings_this_week(earnings_date):
    if earnings_date is None:
        return False
    return date.today() <= earnings_date <= (date.today() + timedelta(days=7))


def has_dividend_risk(ex_div_date, div_amount, exp_date_obj, extrinsic_dollar):
    """True si hay una ex-date de dividendo antes o el mismo día del
    vencimiento y el dividendo estimado es >= el extrínseco del candidato
    (riesgo real de asignación anticipada). Si no hay datos suficientes de
    dividendo, no se bloquea — más vale un falso negativo aquí que
    descartar candidatos válidos por falta de dato."""
    if ex_div_date is None or div_amount is None or div_amount <= 0:
        return False
    today = date.today()
    if today <= ex_div_date <= exp_date_obj:
        return div_amount >= extrinsic_dollar
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

        # Punto 6: exigir también ask > 0. Un ask en 0/NaN no es un spread
        # real de $0, es un dato faltante — sin esto el mid quedaba
        # artificialmente bajo (mid = bid/2) y distorsionaba el extrínseco.
        itm = itm[(itm["bid"] > 0) & (itm["ask"] > 0) & (itm["mid"] > 0)]
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
        itm["spread_dollar"] = itm["ask"] - itm["bid"]
        itm["spread_pct"]    = itm["spread_dollar"] / itm["mid"] * 100
        itm["downside_prot"] = itm["intrinsic"] / current_price * 100

        itm = itm[itm["extrinsic"] > 0]
        if itm.empty:
            return None

        # Punto 3: el spread no puede comerse más de la mitad del
        # extrínseco que se supone que se está cobrando — si no, el mid
        # price no representa una prima realmente capturable.
        itm = itm[
            itm["spread_dollar"] <= itm["extrinsic"] * (SPREAD_MAX_PCT_OF_EXTRINSIC / 100)
        ]
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
    """Solo datos diarios. Devuelve (registro_o_None, reason)."""
    try:
        data = get_daily_data(ticker)
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
    ticker           = survivor["Ticker"]
    fallback_price   = survivor["current_price"]
    try:
        stock = yf.Ticker(ticker)

        cal_info = get_earnings_and_dividend_info(stock)

        if params["use_earnings_filter"] and has_earnings_this_week(cal_info["earnings_date"]):
            return None, "earnings_this_week"

        expirations = _with_timeout(lambda: stock.options)
        if not expirations:
            return None, "no_expirations"

        exp_str, dte = select_friday_expiration(expirations)
        if exp_str is None:
            return None, "no_friday_expiration"
        exp_date_obj = datetime.strptime(exp_str, "%Y-%m-%d").date()

        # Punto 1: precio en vivo para todo lo que dependa de precisión de
        # precio (PCR, intrínseco/extrínseco), con fallback transparente
        # al cierre histórico ya calculado en Fase 1.
        current_price = get_live_price(stock, fallback_price)

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

        # Punto 5: riesgo de asignación anticipada por dividendo.
        if has_dividend_risk(cal_info["ex_div_date"], cal_info["div_amount"],
                              exp_date_obj, candidate["extrinsic"]):
            return None, "dividend_risk"

        iv_rv_ratio = (
            round(candidate["iv_pct"] / survivor["rv"], 3)
            if (candidate["iv_pct"] and survivor["rv"] and survivor["rv"] > 0) else None
        )
        annualized = round(candidate["extrinsic_pct"] * (365 / dte), 1) if dte > 0 else None
        breakeven  = round(current_price - candidate["mid"], 2)

        result = {
            "Ticker": ticker,
            "Precio": round(current_price, 2),
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

    # ── FASE 1: precio + SMA30, en paralelo de verdad (ver punto 2) ────
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
        data = get_daily_data(ticker)
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
    st.caption("⚙️ v3.2 — precio en vivo, Fase 1 paralela real, filtro de spread y de riesgo de dividendo")
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
        extrinsic_target = st.number_input(
            "Extrínseco objetivo (% del precio, semanal)",
            min_value=0.10, max_value=5.00, value=1.00, step=0.05,
            help="Se admite ±0.15% alrededor de este valor como banda válida."
        )
        extrinsic_min = round(extrinsic_target - 0.15, 3)
        extrinsic_max = round(extrinsic_target + 0.15, 3)

        st.markdown("**💲 Precio del subyacente**")
        min_price, max_price = st.slider(
            "Rango de precio ($)", min_value=5, max_value=1000, value=(20, 500), step=5,
        )

    with c2:
        st.markdown("**💧 Liquidez mínima**")
        min_oi = st.number_input("OI mínimo del strike", min_value=10, max_value=10000, value=100, step=50)

        st.markdown("**📅 Próximo viernes**")
        today    = date.today()
        friday   = next_friday()
        dte_days = (friday - today).days
        st.info(f"📅 Próximo viernes: **{friday.strftime('%d %b %Y')}** (DTE: {dte_days} días)")

        st.caption(
            f"El precio del subyacente para intrínseco/extrínseco se toma en vivo "
            f"en el momento del escaneo, con fallback automático al último cierre "
            f"si no hay precio en vivo disponible (fin de semana, etc.)."
        )

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
        st.caption(
            "Siempre activos (no configurables): bid>0 y ask>0, spread ≤ 50% del "
            "extrínseco, y exclusión por riesgo de asignación por dividendo."
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
    }

    st.divider()

    # ── Escaneo ────────────────────────────────────────────────────────
    st.markdown("### 🚀 Ejecutar Escaneo")

    scan_btn = st.button(
        "🎯 INICIAR ESCANEO",
        type="primary",
        use_container_width=True,
        disabled=len(tickers_all) == 0,
    )

    st.caption(
        f"ℹ️ Se escanearán **{len(tickers_all):,}** tickers · Fase 1 en paralelo (rápida) "
        f"→ Fase 2 secuencial solo sobre los que sobrevivan a precio/SMA30 "
        f"(más lenta, ~1-2s por ticker)."
    )

    if scan_btn:
        scan_tickers = tickers_all
        if not scan_tickers:
            st.error("⚠️ El universo de tickers está vacío — pulsa 'Actualizar Universo'.")
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
