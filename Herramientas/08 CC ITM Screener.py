"""
Deep ITM Covered Call Screener
================================
Objetivo: encontrar covered calls deep ITM con extrínseco >= umbral mínimo
semanal, con el strike más profundo posible que cumpla ese mínimo.

FILTROS (todos activables/desactivables desde la UI salvo los marcados
[fijo], que siempre están activos porque protegen la validez del dato):
- Precio del subyacente: rango configurable
- Tendencia alcista, 4 niveles: Ninguno / Básico (Close>SMA30) / Medio
  (+ SMA30 con pendiente alcista) / Fuerte (+ SMA10>SMA30) [selector]
- PCR < 1.0 (sesgo alcista) [toggle]
- OI mínimo del strike (liquidez) [toggle, OFF por defecto — el OI que
  reporta Alpaca no siempre es fiable para nombres poco líquidos]
- Sin earnings entre la fecha de ENTRADA y el vencimiento [toggle]
- Sin riesgo de dividendo (ex-div entre entrada y vencimiento con
  dividendo >= extrínseco capturado) [toggle, ON por defecto]
- Spread bid-ask <= X% del extrínseco [toggle, OFF por defecto — en deep
  ITM puede descartar los strikes más profundos, que son justo los que
  interesan; revisa Spread_% en la tabla en vez de confiar en el filtro]
- Extrínseco: umbral mínimo, o banda mín–máx opcional [toggle; modo
  diagnóstico ignora ambos]
- Bid > 0 y Ask > 0 (opción realmente cotizada) [fijo]
- Ticker con opciones semanales listadas (cadencia real de ~7 días) [fijo]
- Vencimiento = EXACTAMENTE la fecha objetivo, sin tolerancia [fijo]
- Filtros adicionales POST-escaneo (sobre resultados ya obtenidos, sin
  red): protección intrínseca mínima, delta mínimo, OI mínimo, volumen
  mínimo — acotan la tabla/gráficos sin relanzar el escaneo.

SELECCIÓN DE STRIKE (dentro de cada ticker): entre los strikes ITM que
cumplen todos los filtros, se elige el de mayor downside protection
(mid/precio) — en la práctica, el strike más profundo cuyo extrínseco
siga >= el mínimo configurado.
RANKING (entre tickers): por Prot_Intrinseca_% (precio-strike / precio)
descendente.
PRECIO DE OPCIÓN: midprice (bid+ask)/2 siempre.
PRECIO DEL SUBYACENTE: en vivo vía Alpaca (latest trade), con fallback al
último cierre histórico (yfinance) si Alpaca no devuelve dato.
DATOS: opciones (vencimientos, cadena, precio en vivo) vía Alpaca —
utils/utils_alpaca.py. Precio diario/SMA30/RV10 y earnings/dividendos vía
yfinance. yf.Ticker().history() en vez de yf.download() porque esta
instalación devuelve columnas mal aplanadas con yf.download().
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
    get_last_chain_diag,
)

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

# Concurrencia de la Fase 1 (descarga de precios). Es un valor fijo
# interno, no un slider en la UI: es un
# detalle de implementación, no una decisión de trading. Bajado de 4 a 3
# tras el aviso de throttling de CPU: con el lock
# quitado, la Fase 1 ahora sí satura CPU/red de verdad con concurrencia
# real, así que 3 workers simultáneos es un punto más prudente para
# Streamlit Community Cloud (CPU compartida) sin renunciar al paralelismo.
MAX_WORKERS = 3

# Spread bid-ask máximo permitido, como % del extrínseco en dólares del
# candidato. Si el spread se come más de esto, el extrínseco "de mid
# price" no es realmente capturable.
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
    "weak_trend",
    "earnings_in_period",
    "no_expirations",
    "no_weekly_options",
    "no_expiration_exact_dte",
    "no_calls_chain",
    "pcr_bearish",
    "no_itm_candidate",
    "dividend_risk",
    "error",
]

REASON_LABELS = {
    "ok": "✅ Pasó todos los filtros",
    "no_daily_data": "Sin datos diarios (Fase 1 — yf.download)",
    "price_out_of_range": "Precio fuera de rango (Fase 1)",
    "sma30_unavailable": "No hay suficiente histórico para SMA30 (Fase 1)",
    "below_sma30": "Precio ≤ SMA30 — no alcista (Fase 1)",
    "weak_trend": "Tendencia insuficiente: SMA30 sin pendiente alcista o SMA10 no confirma (Fase 1)",
    "earnings_in_period": "Earnings entre la fecha de entrada y el vencimiento (Fase 2)",
    "no_expirations": "Sin vencimientos de opciones listados en Alpaca (Fase 2)",
    "no_weekly_options": "Sin cadencia de opciones semanales (Fase 2) [fijo]",
    "no_expiration_exact_dte": "Sin vencimiento EXACTO en la fecha objetivo (Fase 2) [fijo]",
    "no_calls_chain": "Cadena de calls vacía/no disponible en Alpaca (Fase 2)",
    "pcr_bearish": "PCR ≥ 1.0 — sesgo bajista (Fase 2)",
    "no_itm_candidate": "Sin strike ITM que cumpla extrínseco/OI/bid/ask/spread (Fase 2)",
    "dividend_risk": "Riesgo de asignación por dividendo antes del vencimiento (Fase 2)",
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
    t = str(raw).strip().replace("&amp;", "&").upper()
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
# 1. DATOS DIARIOS (Fase 1 — paralela de verdad)
# ======================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def get_daily_data(ticker):
    """Descarga precio diario AJUSTADO (auto_adjust=True) vía
    yf.Ticker(ticker).history(): en esta instalación yf.download() devuelve
    columnas mal aplanadas incluso con multi_level_index=False (confirmado
    con el test de conectividad: AttributeError sobre 'Close'), mientras
    que Ticker().history() funciona limpio.

    auto_adjust=True: sin ajustar, un split en los
    últimos 120 días metía un salto de escala en el Close crudo que
    distorsionaba la SMA30. Esta serie ajustada es solo para SMA30/RV10 y
    para el filtro de rango de precio de Fase 1 — el precio que realmente
    se usa para intrínseco/extrínseco en Fase 2 es el precio en vivo
    (ver get_live_price).

    CACHEADO 30 min (fix de throttling): esta es,
    con diferencia, la llamada más repetida del pipeline — una por cada
    ticker del universo, en cada escaneo. Durante una sesión normal de
    ajuste de parámetros (extrínseco, OI, tendencia...) el usuario relanza
    el escaneo varias veces sobre el mismo universo en pocos minutos; sin
    caché, cada relanzamiento repite miles de descargas idénticas. 30 min
    es corto para no servir datos desfasados en pleno día de mercado, pero
    cubre de sobra una sesión de prueba de parámetros. El botón "Resetear
    Todo" vacía este caché manualmente cuando el
    usuario quiere datos frescos ya."""
    try:
        end = datetime.now() + timedelta(days=1)
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
# 2. TENDENCIA: SMA30 + pendiente + SMA10
# ======================================================================

def get_trend_info(close):
    """Devuelve un dict con todo lo necesario para los 3 niveles de
    exigencia de tendencia, o None si no hay histórico suficiente.

    - sma30 / dist_sma30_pct: igual que antes (nivel Básico)
    - sma30_slope_up: la SMA30 de hoy es mayor que la de hace 5 sesiones
      (nivel Medio — exige que la media de fondo esté subiendo, no solo
      que el precio esté por encima de una media plana o bajando)
    - sma10 / sma10_above_sma30: confirmación de corto plazo (nivel
      Fuerte — evita quedarse en un "alcista de libro" ya agotado, donde
      el precio sigue sobre la SMA30 pero el momentum reciente ya giró)
    """
    try:
        if len(close) < 35:
            return None
        sma30 = close.rolling(30).mean()
        sma10 = close.rolling(10).mean()
        sma30_now = float(sma30.iloc[-1])
        sma30_prev = float(sma30.iloc[-6])
        sma10_now = float(sma10.iloc[-1])
        price = float(close.iloc[-1])
        return {
            "sma30": round(sma30_now, 2),
            "dist_sma30_pct": round((price - sma30_now) / sma30_now * 100, 2),
            "sma30_slope_up": sma30_now > sma30_prev,
            "sma10": round(sma10_now, 2),
            "sma10_above_sma30": sma10_now > sma30_now,
        }
    except Exception:
        return None

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
# 5. EARNINGS Y RIESGO DE DIVIDENDO
# ======================================================================

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_earnings_and_dividend_info(ticker):
    """Una sola función que reúne fecha de earnings y datos de dividendo,
    reutilizando stock.calendar para ambos y evitando llamadas de red
    duplicadas. Recibe el ticker (str) en vez de un objeto Ticker para que
    st.cache_data pueda usarlo como clave de caché — un objeto yf.Ticker()
    nuevo en cada llamada tendría una identidad distinta cada vez y el
    caché nunca acertaría. Cacheado 6h: earnings/dividendos no cambian
    intradía, así que reconsultarlos en cada escaneo de prueba durante una
    sesión de ajuste de parámetros es red/CPU desperdiciada — una causa
    directa del throttling de Streamlit Community Cloud. El botón
    "Resetear Todo" vacía este caché manualmente.

    Devuelve dict:
    - earnings_date: date o None
    - ex_div_date: date o None (próxima fecha ex-dividendo conocida)
    - div_amount: float o None (estimación = último dividendo pagado)
    """
    info = {"earnings_date": None, "ex_div_date": None, "div_amount": None}
    try:
        stock = yf.Ticker(ticker)
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

        divs = _with_timeout(lambda: stock.dividends)
        if divs is not None and not divs.empty:
            info["div_amount"] = float(divs.iloc[-1])
    except Exception as e:
        _record_debug("error", f"{ticker}: calendario/dividendos: {e}")

    return info

def has_earnings_in_period(earnings_date, entry_date, target_date):
    """True si hay earnings conocidos entre la fecha de ENTRADA elegida y
    la fecha de vencimiento objetivo, ambas inclusive.

    Sustituye a has_earnings_before_expiration(), que usaba "hoy" como
    inicio de la ventana en vez de la fecha de entrada real que el
    usuario piensa usar para abrir la posición — si vas a entrar dentro
    de unos días (no hoy mismo), un earnings entre hoy y esa fecha de
    entrada es irrelevante para el trade, y uno después del vencimiento
    tampoco importa. La única ventana que importa es "entrada →
    vencimiento"."""
    if earnings_date is None:
        return False
    return entry_date <= earnings_date <= target_date

def has_dividend_risk(ex_div_date, div_amount, entry_date, exp_date_obj, extrinsic_dollar):
    """True si hay una ex-date de dividendo entre la fecha de ENTRADA y el
    vencimiento (ambas inclusive) y el dividendo estimado es >= el
    extrínseco del candidato (riesgo real de asignación anticipada). Si
    no hay datos suficientes de dividendo, no se bloquea — más vale un
    falso negativo aquí que descartar candidatos válidos por falta de
    dato. Antes usaba "hoy" como inicio de
    la ventana en vez de la fecha de entrada elegida por el usuario."""
    if ex_div_date is None or div_amount is None or div_amount <= 0:
        return False
    if entry_date <= ex_div_date <= exp_date_obj:
        return div_amount >= extrinsic_dollar
    return False

# ======================================================================
# 6. PUT/CALL RATIO (sobre una cadena ya descargada, sin red)
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
# 7. CANDIDATO DEEP ITM
# ======================================================================

def find_deep_itm_candidate(calls_df, current_price, dte_calendar,
                             extrinsic_min_pct, extrinsic_max_pct,
                             min_oi, diagnostic_mode=False, apply_oi_filter=True,
                             apply_spread_filter=True, spread_max_pct=SPREAD_MAX_PCT_OF_EXTRINSIC):
    """extrinsic_min_pct actúa como UMBRAL MÍNIMO por defecto: pasan todos
    los strikes con extrinsic_pct >= extrinsic_min_pct, sin techo superior.
    Si extrinsic_max_pct no es None (banda opcional), el filtro pasa a ser
    una BANDA:
    extrinsic_min_pct <= extrinsic_pct <= extrinsic_max_pct. En modo
    diagnóstico, se ignoran ambos límites y se devuelve el mejor
    candidato ITM real del mercado (por downside protection), para poder
    ver los valores reales aunque no cumplan lo configurado.

    apply_oi_filter: si False, se omite
    por completo el filtro de OI mínimo — el open interest que reporta
    Alpaca viene con frecuencia en 0 o muy bajo para nombres fuera de los
    índices más líquidos, incluso con bid/ask real cotizado, así que
    exigirlo por defecto puede vaciar el escáner sin que el motivo sea
    evidente. El valor de OI se sigue devolviendo en el candidato (para
    mostrarlo en la tabla), solo deja de usarse como filtro duro.

    apply_spread_filter / spread_max_pct:
    si apply_spread_filter es False, se omite el filtro de spread≤X% del
    extrínseco. Diagnóstico real (caso SMCI): en deep ITM el spread en
    dólares no baja tan rápido como el extrínseco al profundizar más, así
    que este filtro puede excluir en bloque justo los strikes MÁS
    profundos (los de mayor downside protection) aunque su extrínseco sí
    cumpla el mínimo configurado — el algoritmo termina eligiendo un
    strike más superficial de lo que el usuario buscaba, sin aviso. El
    spread se sigue calculando y devolviendo (columna Spread_%), solo deja
    de ser un filtro duro cuando está desactivado."""
    try:
        itm = calls_df[calls_df["strike"] < current_price].copy()
        if itm.empty:
            return None

        itm["bid"] = pd.to_numeric(itm["bid"], errors="coerce").fillna(0)
        itm["ask"] = pd.to_numeric(itm["ask"], errors="coerce").fillna(0)
        itm["mid"] = (itm["bid"] + itm["ask"]) / 2

        # Exigir también ask > 0: un ask en 0/NaN no es un spread
        # real de $0, es un dato faltante — sin esto el mid quedaba
        # artificialmente bajo (mid = bid/2) y distorsionaba el extrínseco.
        itm = itm[(itm["bid"] > 0) & (itm["ask"] > 0) & (itm["mid"] > 0)]
        if itm.empty:
            return None

        itm["oi"] = pd.to_numeric(
            itm.get("openInterest", pd.Series(0, index=itm.index)), errors="coerce"
        ).fillna(0)
        # Filtro de OI ahora opcional (ver docstring de la función).
        if apply_oi_filter:
            itm = itm[itm["oi"] >= min_oi]
            if itm.empty:
                return None

        itm["intrinsic"] = current_price - itm["strike"]
        itm["extrinsic"] = itm["mid"] - itm["intrinsic"]
        itm["extrinsic_pct"] = itm["extrinsic"] / current_price * 100
        itm["spread_dollar"] = itm["ask"] - itm["bid"]
        itm["spread_pct"] = itm["spread_dollar"] / itm["mid"] * 100
        # downside_prot: protección real = prima TOTAL (intrínseco+
        # extrínseco) / precio, no solo el intrínseco — se usa para elegir
        # el strike más profundo dentro de cada ticker (ver más abajo).
        # downside_prot_intrinsic (Prot_Intrinseca_%) es la que se expone
        # en la tabla de resultados y se usa para el ranking entre tickers.
        itm["downside_prot"] = itm["mid"] / current_price * 100
        itm["downside_prot_intrinsic"] = itm["intrinsic"] / current_price * 100

        itm = itm[itm["extrinsic"] > 0]
        if itm.empty:
            return None

        # (Opcional): el spread no puede
        # comerse más de spread_max_pct% del extrínseco que se supone que
        # se está cobrando — si no, el mid price no representa una prima
        # realmente capturable. Ver docstring de la función para el caso
        # real (SMCI) donde este filtro excluía los strikes más profundos.
        if apply_spread_filter:
            itm = itm[
                itm["spread_dollar"] <= itm["extrinsic"] * (spread_max_pct / 100)
            ]
            if itm.empty:
                return None

        if diagnostic_mode:
            candidates = itm
        elif extrinsic_max_pct is not None:
            # Banda opcional — extrínseco entre mínimo y máximo.
            candidates = itm[
                (itm["extrinsic_pct"] >= extrinsic_min_pct)
                & (itm["extrinsic_pct"] <= extrinsic_max_pct)
            ]
        else:
            # Umbral mínimo, sin techo superior.
            candidates = itm[itm["extrinsic_pct"] >= extrinsic_min_pct]
        if candidates.empty:
            return None

        best = candidates.sort_values("downside_prot", ascending=False).iloc[0]

        T_years = dte_calendar / 365.0
        iv = float(best.get("impliedVolatility") or 0)
        delta = bs_delta(current_price, float(best["strike"]), T_years, iv) if iv > 0 else None

        best_extrinsic_pct = float(best["extrinsic_pct"])
        meets_range = best_extrinsic_pct >= extrinsic_min_pct
        if extrinsic_max_pct is not None:
            meets_range = meets_range and best_extrinsic_pct <= extrinsic_max_pct

        return {
            "strike": float(best["strike"]),
            "mid": round(float(best["mid"]), 2),
            "bid": round(float(best["bid"]), 2),
            "ask": round(float(best["ask"]), 2),
            "intrinsic": round(float(best["intrinsic"]), 2),
            "extrinsic": round(float(best["extrinsic"]), 2),
            "extrinsic_pct": round(best_extrinsic_pct, 3),
            "downside_prot": round(float(best["downside_prot"]), 2),
            "downside_prot_intrinsic": round(float(best["downside_prot_intrinsic"]), 2),
            "spread_pct": round(float(best["spread_pct"]), 2),
            "oi": int(best["oi"]),
            "volume": int(pd.to_numeric(best.get("volume", 0), errors="coerce") or 0),
            "iv_pct": round(iv * 100, 2) if iv > 0 else None,
            "delta": delta,
            # "in_target_band": cumple el mínimo (y el máximo, si hay banda
            # activa) configurados. Se usa en Consulta Individual; ya no se
            # expone como columna en la tabla del escaneo masivo.
            "in_target_band": bool(meets_range),
        }
    except Exception:
        return None


def find_deep_itm_debug_funnel(calls_df, current_price):
    """Réplica, paso a paso, de los filtros que Consulta Individual SÍ
    aplica siempre (bid>0/ask>0/mid>0, extrínseco>0) para diagnosticar en
    qué escalón se queda en cero un vencimiento que devuelve "sin
    candidato" aunque la cadena sí traiga bid/ask cotizado. Ni el OI
    mínimo ni el spread≤X% del extrínseco se aplican aquí — se mantienen
    solo como datos
    informativos (mediana de OI, spread mediano), nunca como filtro,
    porque en Consulta Individual el objetivo es revisar el ticker sin
    que un dato de Alpaca poco fiable oculte el resultado. Solo se usa en
    quick_lookup — no afecta al pipeline de escaneo masivo. Devuelve un
    dict con el recuento de strikes ITM que sobreviven a cada paso."""
    out = {"itm_total": 0, "with_bid_ask": 0, "extrinsic_positive": 0,
           "oi_median": None, "spread_pct_median": None}
    try:
        itm = calls_df[calls_df["strike"] < current_price].copy()
        out["itm_total"] = len(itm)
        if itm.empty:
            return out

        itm["bid"] = pd.to_numeric(itm["bid"], errors="coerce").fillna(0)
        itm["ask"] = pd.to_numeric(itm["ask"], errors="coerce").fillna(0)
        itm["mid"] = (itm["bid"] + itm["ask"]) / 2
        itm = itm[(itm["bid"] > 0) & (itm["ask"] > 0) & (itm["mid"] > 0)]
        out["with_bid_ask"] = len(itm)
        if itm.empty:
            return out

        itm["oi"] = pd.to_numeric(
            itm.get("openInterest", pd.Series(0, index=itm.index)), errors="coerce"
        ).fillna(0)
        out["oi_median"] = float(itm["oi"].median())  # informativo, no filtra

        itm["intrinsic"] = current_price - itm["strike"]
        itm["extrinsic"] = itm["mid"] - itm["intrinsic"]
        itm = itm[itm["extrinsic"] > 0]
        out["extrinsic_positive"] = len(itm)
        if itm.empty:
            return out

        # Informativo únicamente — no filtra aquí.
        itm["spread_dollar"] = itm["ask"] - itm["bid"]
        itm["spread_pct_of_extrinsic"] = itm["spread_dollar"] / itm["extrinsic"] * 100
        out["spread_pct_median"] = float(itm["spread_pct_of_extrinsic"].median())
        return out
    except Exception as e:
        out["error"] = str(e)
        return out


def _explain_deep_itm_funnel(funnel):
    """Traduce find_deep_itm_debug_funnel() a un mensaje humano que señala
    el primer escalón que se queda en cero."""
    if funnel.get("itm_total", 0) == 0:
        return "No hay ningún strike con strike < precio actual en la cadena descargada (raro — revisa el precio en vivo)."
    if funnel.get("with_bid_ask", 0) == 0:
        return f"De {funnel['itm_total']} strikes ITM, ninguno tiene bid/ask > 0 cotizado."
    oi_med = funnel.get("oi_median")
    oi_txt = f"{oi_med:.0f}" if oi_med is not None else "N/D"
    if funnel.get("extrinsic_positive", 0) == 0:
        return (
            f"De {funnel['with_bid_ask']} strikes ITM con bid/ask válido (mediana de OI "
            f"informativo: {oi_txt}, sin filtrar por él aquí), NINGUNO tiene extrínseco "
            f"positivo (el midprice cotizado está por debajo del valor intrínseco) — "
            f"cotización probablemente poco fiable en este momento/feed."
        )
    spread_med = funnel.get("spread_pct_median")
    spread_txt = f"{spread_med:.0f}%" if spread_med is not None else "N/D"
    return (
        f"Debería haber candidato: {funnel['extrinsic_positive']} strikes con "
        f"extrínseco positivo (mediana de OI informativo: {oi_txt}, mediana de "
        f"spread/extrínseco informativo: {spread_txt} — ninguno de los dos filtra "
        f"aquí). Si aun así no aparece, revisa el extrínseco mínimo configurado."
    )


# ======================================================================
# 8a. FASE 1 — precio diario + SMA30 (paralela)
# ======================================================================

def phase1_price_filter(ticker, params):
    """Solo datos diarios. Devuelve (registro_o_None, reason)."""
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
            return None, "sma30_unavailable"

        level = params["trend_strength"]  # "none" | "basic" | "medium" | "strong"
        if level != "none":
            if current_price <= trend["sma30"]:
                return None, "below_sma30"
            if level in ("medium", "strong") and not trend["sma30_slope_up"]:
                return None, "weak_trend"
            if level == "strong" and not trend["sma10_above_sma30"]:
                return None, "weak_trend"

        rv = get_rv10(close)

        return {
            "Ticker": ticker,
            "current_price": round(current_price, 2),
            "sma30": trend["sma30"],
            "dist_sma_pct": trend["dist_sma30_pct"],
            "slope_up": trend["sma30_slope_up"],
            "sma10": trend["sma10"],
            "sma10_above_sma30": trend["sma10_above_sma30"],
            "rv": rv,
        }, "ok"
    except Exception as e:
        _record_debug("error", f"{ticker}: {e}")
        return None, "error"

# ======================================================================
# 8b. FASE 2 — opciones (SECUENCIAL, sin threads)
# ======================================================================

def phase2_options_filter(survivor, params):
    """Recibe el registro de la Fase 1 y añade las métricas de opciones.
    Se llama en un bucle for normal, nunca dentro de un ThreadPoolExecutor.

    Datos de opciones (vencimientos + cadena) vía Alpaca (utils_alpaca),
    Precio en vivo del subyacente, earnings y
    dividendos se siguen obteniendo vía yfinance, sin cambios."""
    ticker = survivor["Ticker"]
    fallback_price = survivor["current_price"]
    target_date = params["target_expiration_date"]  # date, vencimiento EXACTO exigido
    entry_date = params["entry_date"]  # date, inicio del periodo para earnings/dividendos
    try:
        # Si ambos filtros (earnings y dividendo) están desactivados, no
        # hace falta ni esta llamada.
        if params["use_earnings_filter"] or params["use_dividend_filter"]:
            cal_info = get_earnings_and_dividend_info(ticker)
        else:
            cal_info = {"earnings_date": None, "ex_div_date": None, "div_amount": None}

        if params["use_earnings_filter"] and has_earnings_in_period(cal_info["earnings_date"], entry_date, target_date):
            return None, "earnings_in_period"

        expirations = get_option_expirations(ticker)
        if not expirations:
            return None, "no_expirations"

        # Filtro FIJO — el ticker debe tener cadencia real de opciones
        # semanales, no solo mensuales.
        if not has_weekly_options(expirations):
            return None, "no_weekly_options"

        # Sin tolerancia — el vencimiento elegido en la UI debe existir
        # EXACTAMENTE para este ticker, o se descarta.
        if target_date not in expirations:
            return None, "no_expiration_exact_dte"

        exp_date_obj = target_date
        exp_str = target_date.isoformat()
        dte = (exp_date_obj - date.today()).days

        # Precio en vivo vía Alpaca (latest trade), con
        # fallback transparente al cierre histórico de Fase 1 (yfinance)
        # si Alpaca no devuelve dato.
        current_price = get_live_price(ticker, fallback_price)

        calls, puts = get_option_chain(ticker, exp_date_obj)
        if calls is None or calls.empty:
            return None, "no_calls_chain"

        pcr = compute_pcr(calls, puts, current_price)
        if params["use_pcr_filter"] and pcr is not None and pcr >= 1.0:
            return None, "pcr_bearish"

        candidate = find_deep_itm_candidate(
            calls, current_price, dte,
            params["extrinsic_min"], params["extrinsic_max"],
            params["min_oi"], diagnostic_mode=params["diagnostic_mode"],
            apply_oi_filter=params["use_oi_filter"],
            apply_spread_filter=params["use_spread_filter"],
            spread_max_pct=params["spread_max_pct"],
        )
        if candidate is None:
            return None, "no_itm_candidate"

        # Riesgo de asignación anticipada por dividendo.
        if params["use_dividend_filter"] and has_dividend_risk(
            cal_info["ex_div_date"], cal_info["div_amount"],
            entry_date, exp_date_obj, candidate["extrinsic"]
        ):
            return None, "dividend_risk"

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
            "Prot_Intrinseca_%": candidate["downside_prot_intrinsic"],
            "Extrínseco_%": candidate["extrinsic_pct"],
            "Prima_Mid": candidate["mid"],
            "Bid": candidate["bid"],
            "Ask": candidate["ask"],
            "Intrínseco": candidate["intrinsic"],
            "Extrínseco_$": candidate["extrinsic"],
            "Delta": candidate["delta"],
            "IV_%": candidate["iv_pct"],
            "RV_%": survivor["rv"],
            "IV_RV": iv_rv_ratio,
            "OI": candidate["oi"],
            "Volumen": candidate["volume"],
            "Spread_%": candidate["spread_pct"],
            "SMA30": survivor["sma30"],
            "Dist_SMA30_%": survivor["dist_sma_pct"],
            "SMA30_Sube": survivor["slope_up"],
            "SMA10": survivor.get("sma10"),
            "SMA10>SMA30": survivor.get("sma10_above_sma30"),
            "PCR": pcr,
        }
        return result, "ok"
    except Exception as e:
        _record_debug("error", f"{ticker}: {e}")
        return None, "error"

# ======================================================================
# 9. ORQUESTADOR: Fase 1 (paralela) → Fase 2 (secuencial)
# ======================================================================

def run_screener(tickers, params, progress_bar, status_text):
    _reset_debug()
    funnel = {r: 0 for r in REASON_ORDER}
    total = len(tickers)

    # ── FASE 1: precio + SMA30, en paralelo de verdad ────────────────
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
        df = df.sort_values("Prot_Intrinseca_%", ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))

    with _debug_lock:
        debug_snapshot = {k: list(v) for k, v in _debug_samples.items()}
        price_snapshot = list(_price_samples)

    return df, funnel, debug_snapshot, price_snapshot

# ======================================================================
# 9b. RESET COMPLETO
# ======================================================================

def reset_everything():
    """Vacía el caché de datos (precio diario, earnings/dividendos y
    vencimientos de Alpaca) y borra los resultados de la sesión, para
    forzar un escaneo 100% desde cero en el próximo click de 'INICIAR
    ESCANEO'. No toca el universo de tickers (tiene su propio botón) ni
    los valores de los widgets de filtros — el usuario decide esos aparte."""
    get_daily_data.clear()
    get_earnings_and_dividend_info.clear()
    get_option_expirations.clear()
    for key in ("results", "funnel", "debug_snapshot", "price_snapshot",
                "scan_ts", "scanned_total"):
        st.session_state.pop(key, None)

# ======================================================================
# 9c. CONSULTA INDIVIDUAL: ticker + DTE → datos directos
# ======================================================================

def quick_lookup(ticker, entry_date, target_date, params):
    """Consulta puntual de UN solo ticker para un periodo entrada→
    vencimiento EXACTO (sin tolerancia). A
    diferencia del escaneo masivo, esto es solo informativo para
    tendencia/PCR/earnings/dividendo/extrínseco mínimo — se muestran
    todos como datos, para que el usuario decida, en vez de descartar el
    ticker sin explicación. Lo que SÍ se exige siempre, porque es filtro
    fijo: opciones semanales, vencimiento exacto, y dentro de
    find_deep_itm_candidate (bid>0, ask>0, spread≤50% del extrínseco) —
    para que el resultado sea comparable con lo que devolvería el
    escáner completo.

    Devuelve un dict; si algo falla en el camino, la clave "error" trae
    un mensaje explicando en qué paso se detuvo, y los datos ya
    obtenidos hasta ese punto se incluyen igualmente."""
    out = {"ticker": ticker, "error": None}

    data = get_daily_data(ticker)
    if data is None:
        out["error"] = "Sin datos diarios suficientes para este ticker."
        return out

    close = data["Close"]
    fallback_price = float(close.iloc[-1])
    out["trend"] = get_trend_info(close)
    out["rv"] = get_rv10(close)

    out["cal_info"] = get_earnings_and_dividend_info(ticker)
    out["current_price"] = get_live_price(ticker, fallback_price)
    out["earnings_soon"] = has_earnings_in_period(out["cal_info"]["earnings_date"], entry_date, target_date)

    expirations = get_option_expirations(ticker)
    if not expirations:
        out["error"] = "Este ticker no tiene vencimientos de opciones listados en Alpaca."
        return out

    out["has_weekly"] = has_weekly_options(expirations)
    if not out["has_weekly"]:
        out["error"] = "Este ticker no tiene cadencia de opciones semanales (filtro fijo)."
        return out

    if target_date not in expirations:
        out["error"] = (
            f"No hay vencimiento EXACTO el {target_date.strftime('%d %b %Y')} para este "
            f"ticker (sin tolerancia). Vencimientos disponibles cercanos: "
            f"{', '.join(e.isoformat() for e in expirations if abs((e - target_date).days) <= 14) or 'ninguno cerca'}."
        )
        out["expirations_available"] = [e.isoformat() for e in expirations]
        return out

    exp_date_obj = target_date
    exp_str = target_date.isoformat()
    dte = (exp_date_obj - date.today()).days
    out["exp_str"] = exp_str
    out["dte"] = dte

    calls, puts = get_option_chain(ticker, exp_date_obj)
    out["chain_diag"] = get_last_chain_diag()
    if calls is None or calls.empty:
        d = out["chain_diag"]
        out["error"] = (
            f"Cadena de calls vacía para el vencimiento {exp_str} en Alpaca "
            f"(feed: {d.get('feed', '?')}) — contratos listados: "
            f"{d.get('contracts_total', 0)}, snapshots devueltos: "
            f"{d.get('snapshots_total', 0)}, con bid/ask cotizado: "
            f"{d.get('quotes_with_bid_ask', 0)}."
        )
        return out

    out["pcr"] = compute_pcr(calls, puts, out["current_price"])

    # Consulta Individual NUNCA filtra por
    # OI mínimo — apply_oi_filter=False siempre aquí, sea cual sea el
    # toggle del escaneo masivo. El OI se sigue mostrando en la tabla como
    # dato informativo, pero no descarta el ticker.
    # Tampoco filtra por spread — mismo criterio, para
    # que el usuario vea el mejor strike ITM real y decida con Spread_%
    # a la vista, en vez de que el filtro lo oculte sin explicación.
    candidate = find_deep_itm_candidate(
        calls, out["current_price"], dte,
        params["extrinsic_min"], params["extrinsic_max"],
        params["min_oi"], diagnostic_mode=True, apply_oi_filter=False,
        apply_spread_filter=False,
    )
    out["candidate"] = candidate
    if candidate is None:
        funnel = find_deep_itm_debug_funnel(calls, out["current_price"])
        out["itm_funnel"] = funnel
        out["error"] = (
            "No hay ningún strike ITM que pase los filtros fijos (bid>0, ask>0, "
            "spread≤50% del extrínseco) para este vencimiento — el OI mínimo NO "
            "se aplica aquí. "
            + _explain_deep_itm_funnel(funnel)
        )
        return out

    out["meets_extrinsic_min"] = candidate["in_target_band"]
    out["dividend_risk"] = has_dividend_risk(
        out["cal_info"]["ex_div_date"], out["cal_info"]["div_amount"],
        entry_date, exp_date_obj, candidate["extrinsic"],
    )
    return out

# ======================================================================
# 9d. CALCULADORA: combo buy-write (precio - prima call) para un strike
# ======================================================================

def build_buywrite_combo_calculator(ticker, target_date, strike, extrinsic_min_pct, extrinsic_max_pct):
    """Calculadora de combo buy-write (comprar la acción + vender la call,
    a débito neto) para un STRIKE que el usuario elige a mano.

    Fórmula: prima_call = intrínseco + extrínseco = (precio - strike) +
    extrínseco → combo (débito) = precio - prima_call = strike - extrínseco.
    Como el extrínseco baja el combo, para EXIGIR un extrínseco mínimo el
    combo debe ser COMO MUCHO:
        combo_max = strike - (extrinsic_min_pct/100) * precio
    Si además se fija un extrínseco máximo (banda), el combo no debe bajar
    de combo_min = strike - (extrinsic_max_pct/100) * precio.

    Si el vencimiento y el strike existen en la cadena real de Alpaca,
    también devuelve el combo REAL (precio - mid real) para comparar."""
    out = {"ticker": ticker, "error": None, "current_price": None}

    data = get_daily_data(ticker)
    if data is None:
        out["error"] = "Sin datos diarios suficientes para este ticker."
        return out
    fallback_price = float(data["Close"].iloc[-1])
    current_price = get_live_price(ticker, fallback_price)
    out["current_price"] = current_price

    intrinsic = current_price - strike
    out["intrinsic"] = intrinsic
    out["premium_min"] = intrinsic + (extrinsic_min_pct / 100) * current_price
    out["combo_max"] = strike - (extrinsic_min_pct / 100) * current_price
    if extrinsic_max_pct is not None:
        out["premium_max"] = intrinsic + (extrinsic_max_pct / 100) * current_price
        out["combo_min"] = strike - (extrinsic_max_pct / 100) * current_price
    else:
        out["premium_max"] = None
        out["combo_min"] = None

    # Comparación opcional con datos reales de mercado, si el vencimiento
    # y el strike existen tal cual en la cadena de Alpaca.
    out["real_bid"] = out["real_ask"] = out["real_mid"] = out["real_combo"] = None
    out["cumple_real"] = None
    expirations = get_option_expirations(ticker)
    if expirations and target_date in expirations:
        calls, _ = get_option_chain(ticker, target_date)
        if calls is not None and not calls.empty:
            match = calls[calls["strike"] == strike]
            if not match.empty:
                row = match.iloc[0]
                bid = float(row["bid"]) if pd.notna(row["bid"]) else 0.0
                ask = float(row["ask"]) if pd.notna(row["ask"]) else 0.0
                if bid > 0 and ask > 0:
                    mid = (bid + ask) / 2
                    out["real_bid"], out["real_ask"], out["real_mid"] = bid, ask, mid
                    out["real_combo"] = current_price - mid
                    out["cumple_real"] = out["real_combo"] <= out["combo_max"]
    return out

# ======================================================================
# 10. GRÁFICO DE PRECIO
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
# 11. INTERFAZ STREAMLIT
# ======================================================================

def main():
    st.set_page_config(page_title="Deep ITM CC Screener", page_icon="🎯", layout="wide")

    if not check_password():
        st.stop()

    st.title("🎯 Deep ITM Covered Call Screener")
    st.markdown(
        "**Objetivo: extrínseco mínimo semanal · "
        "Vencimiento EXACTO (sin tolerancia) · "
        "Ranking por mayor protección intrínseca**"
    )
    st.caption(
        "⚙️ OI y spread mínimos opcionales (OFF por defecto) · earnings/dividendos "
        "ligados al periodo entrada→vencimiento · filtros adicionales sobre "
        "resultados (protección/delta/OI/volumen) · calculadora de extrínseco por "
        "strike · datos vía Alpaca"
    )

    col_title, col_reset = st.columns([5, 1])
    with col_reset:
        if st.button("🧹 Resetear Todo", use_container_width=True,
                      help="Vacía el caché de precios, earnings/dividendos y vencimientos "
                           "de Alpaca, y borra los resultados del último escaneo. El "
                           "universo de tickers y los valores de los filtros no se tocan."):
            reset_everything()
            st.success("Caché y resultados borrados — listo para un escaneo limpio.")
            st.rerun()

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
        st.markdown("**💰 Extrínseco**")
        use_extrinsic_band = st.checkbox(
            "Usar banda de extrínseco (mín–máx) en vez de solo mínimo",
            value=False,
            help="Desactivado (por defecto): se comporta como umbral mínimo, "
                 "sin techo — pasan todos los strikes con extrínseco >= el "
                 "valor que pongas. Activado: pasan solo los strikes cuyo "
                 "extrínseco caiga DENTRO de la banda mín–máx, por ejemplo "
                 "0.85% a 1.00%.",
        )
        if use_extrinsic_band:
            ecol1, ecol2 = st.columns(2)
            with ecol1:
                extrinsic_min = st.number_input(
                    "Extrínseco mínimo (%)",
                    min_value=0.10, max_value=5.00, value=0.95, step=0.05,
                )
            with ecol2:
                extrinsic_max = st.number_input(
                    "Extrínseco máximo (%)",
                    min_value=0.10, max_value=10.00, value=1.10, step=0.05,
                )
            if extrinsic_max < extrinsic_min:
                st.error(
                    f"⚠️ El extrínseco máximo ({extrinsic_max}%) es menor que "
                    f"el mínimo ({extrinsic_min}%) — se intercambian automáticamente."
                )
                extrinsic_min, extrinsic_max = extrinsic_max, extrinsic_min
            st.caption(f"✅ Banda activa: **{extrinsic_min}% — {extrinsic_max}%**")
        else:
            extrinsic_min = st.number_input(
                "Extrínseco mínimo (% del precio, semanal)",
                min_value=0.10, max_value=5.00, value=0.95, step=0.05,
                help="Umbral mínimo: pasan todos los strikes con extrínseco >= "
                     "este valor, sin techo superior. Un extrínseco más alto "
                     "que el mínimo es mejor, no se descarta."
            )
            extrinsic_max = None
            st.caption(f"✅ Umbral activo: **≥ {extrinsic_min}%** (sin techo)")

        st.markdown("**💲 Precio del subyacente**")
        st.caption("Solo se escanean tickers cuyo precio actual caiga dentro de este rango.")
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            min_price = st.number_input(
                "Precio mínimo ($)", min_value=1, max_value=10000, value=15, step=5,
                help="Tickers por debajo de este precio no se escanean.",
            )
        with pcol2:
            max_price = st.number_input(
                "Precio máximo ($)", min_value=1, max_value=10000, value=50, step=5,
                help="Tickers por encima de este precio no se escanean.",
            )
        if min_price > max_price:
            st.error(
                f"⚠️ El precio mínimo (${min_price:,}) es mayor que el máximo "
                f"(${max_price:,}) — se intercambian automáticamente."
            )
            min_price, max_price = max_price, min_price
        st.caption(f"✅ Rango activo: **${min_price:,} — ${max_price:,}**")

    with c2:
        st.markdown("**💧 Liquidez mínima**")
        use_oi_filter = st.checkbox(
            "Exigir OI mínimo del strike", value=False,
            help="Desactivado por defecto: el open interest que reporta "
                 "Alpaca viene con frecuencia en 0 o muy bajo para nombres fuera "
                 "de los índices más líquidos, incluso con bid/ask real cotizado "
                 "— exigirlo por defecto puede vaciar el escáner sin que el "
                 "motivo sea evidente (ver embudo de diagnóstico, motivo "
                 "'no_itm_candidate'). Actívalo si confías en el dato de OI para "
                 "tu universo, o si quieres priorizar solo nombres con liquidez "
                 "histórica alta reportada."
        )
        min_oi = st.number_input(
            "OI mínimo del strike", min_value=10, max_value=10000, value=100, step=50,
            disabled=not use_oi_filter,
            help="Solo se aplica si el checkbox de arriba está activado."
        )
        if not use_oi_filter:
            st.caption("ℹ️ OI mínimo NO se está aplicando — el open interest solo se muestra como dato informativo.")

        st.markdown("**〰️ Spread bid-ask**")
        use_spread_filter = st.checkbox(
            "Exigir spread ≤ X% del extrínseco", value=False,
            help="Desactivado por defecto: en deep ITM el spread bid-ask "
                 "en dólares no baja tan rápido como el extrínseco al ir más "
                 "profundo, así que este filtro puede excluir en bloque justo los "
                 "strikes MÁS profundos (los de mayor downside protection) aunque "
                 "cumplan tu extrínseco mínimo — caso real detectado con SMCI, "
                 "donde el strike óptimo (spread grande pero extrínseco ≥1%) se "
                 "descartaba y el escáner elegía uno más superficial sin avisar. "
                 "El spread se sigue mostrando en Spread_% como dato informativo — "
                 "revísalo ahí antes de operar. Actívalo si prefieres que el "
                 "escáner descarte automáticamente primas poco capturables."
        )
        spread_max_pct = st.number_input(
            "Spread máximo (% del extrínseco)", min_value=10, max_value=300, value=50, step=10,
            disabled=not use_spread_filter,
            help="Solo se aplica si el checkbox de arriba está activado."
        )
        if not use_spread_filter:
            st.caption("ℹ️ Spread máximo NO se está aplicando — revisa Spread_% en la tabla antes de operar.")

        st.markdown("**📆 Periodo de la posición**")
        entry_date = st.date_input(
            "Fecha de entrada",
            value=date.today(),
            min_value=date.today(),
            max_value=date.today() + timedelta(days=179),
            format="DD/MM/YYYY",
            help="Día en el que abrirías la posición. Earnings y dividendos se "
                 "comprueban SOLO entre esta fecha y el vencimiento — un earnings "
                 "antes de la entrada o después del vencimiento no afecta al trade.",
        )
        target_date = st.date_input(
            "Fecha de vencimiento objetivo",
            value=max(entry_date + timedelta(days=7), date.today() + timedelta(days=1)),
            min_value=entry_date + timedelta(days=1),
            max_value=entry_date + timedelta(days=180),
            format="DD/MM/YYYY",
            help="Elige la fecha con el calendario. El ticker debe tener un "
                 "vencimiento que caiga EXACTAMENTE en este día — no hay "
                 "tolerancia ni búsqueda del más cercano. Si el ticker no "
                 "cotiza opciones justo ese día, se descarta. Debe ser "
                 "posterior a la fecha de entrada de arriba.",
        )
        if target_date <= entry_date:
            st.error(
                f"⚠️ El vencimiento ({target_date.strftime('%d %b %Y')}) debe ser "
                f"posterior a la entrada ({entry_date.strftime('%d %b %Y')}) — "
                f"ajusta una de las dos fechas."
            )
        dte_target = (target_date - date.today()).days
        st.caption(
            f"📅 Entrada: **{entry_date.strftime('%d %b %Y')}** → Vencimiento: "
            f"**{target_date.strftime('%d %b %Y')}** (DTE={dte_target} días desde "
            f"hoy, exacto, sin tolerancia). Earnings/dividendos se comprueban en "
            f"ese periodo de entrada→vencimiento, no antes ni después. El precio "
            f"del subyacente para intrínseco/extrínseco se toma en vivo en el "
            f"momento del escaneo, con fallback automático al último cierre si "
            f"no hay precio en vivo disponible. Strike, bid, ask y mid "
            f"corresponden siempre al DTE del vencimiento."
        )

    with c3:
        st.markdown("**🎚️ Filtros activables**")

        trend_options = {
            "Ninguno": "none",
            "Básico: Close > SMA30": "basic",
            "Medio: + SMA30 con pendiente alcista": "medium",
            "Fuerte: + SMA10 > SMA30 (cruce alcista corto plazo)": "strong",
        }
        trend_label = st.selectbox(
            "📈 Tendencia alcista",
            options=list(trend_options.keys()),
            index=0,  # "Ninguno" por defecto
            help="Básico = comportamiento anterior (un solo cruce, da falsos "
                 "positivos cerca de la media). Medio añade que la propia SMA30 "
                 "esté subiendo, no solo que el precio esté por encima. Fuerte "
                 "exige además que la SMA10 (corto plazo) confirme, para evitar "
                 "tendencias de fondo ya agotadas."
        )
        trend_strength = trend_options[trend_label]

        use_pcr_filter = st.checkbox("PCR < 1.0 (sesgo alcista)", value=True)
        use_earnings_filter = st.checkbox(
            "Excluir earnings entre entrada y vencimiento", value=True,
            help="Descarta el ticker si tiene earnings conocidos entre la 'Fecha "
                 "de entrada' y la 'Fecha de vencimiento objetivo' elegidas en "
                 "'Periodo de la posición' (no una ventana fija de días) — un "
                 "earnings fuera de ese periodo no afecta a esta posición y no "
                 "la descarta."
        )
        use_dividend_filter = st.checkbox(
            "Excluir riesgo de asignación por dividendo", value=True,
            help="Descarta el candidato si hay una fecha ex-dividendo entre la "
                 "'Fecha de entrada' y el vencimiento (ambas inclusive) y el "
                 "dividendo estimado es mayor o igual que el extrínseco capturado "
                 "(a quien tiene la call comprada le compensaría ejercer antes "
                 "para cobrar el dividendo)."
        )
        diagnostic_mode = st.checkbox(
            "🔬 Modo diagnóstico (ignora el umbral de extrínseco)", value=False,
            help="Devuelve el mejor candidato ITM aunque su extrínseco no alcance "
                 "el mínimo configurado, para ver los valores reales del mercado."
        )
        st.caption(
            "Siempre activos (no configurables): bid>0 y ask>0, opciones "
            "semanales listadas, y vencimiento EXACTO en la fecha objetivo "
            "(sin tolerancia). OI y spread mínimos son opcionales — ver "
            "'Liquidez mínima' arriba."
        )

    params = {
        "extrinsic_min": extrinsic_min,
        "extrinsic_max": extrinsic_max,
        "min_price": min_price,
        "max_price": max_price,
        "min_oi": min_oi,
        "use_oi_filter": use_oi_filter,
        "use_spread_filter": use_spread_filter,
        "spread_max_pct": spread_max_pct,
        "target_expiration_date": target_date,
        "entry_date": entry_date,
        "trend_strength": trend_strength,
        "use_pcr_filter": use_pcr_filter,
        "use_earnings_filter": use_earnings_filter,
        "use_dividend_filter": use_dividend_filter,
        "diagnostic_mode": diagnostic_mode,
    }

    st.divider()

    # ── Consulta individual ────────────────────────────────────────────
    st.markdown("### 🔎 Consulta Individual (ticker + periodo)")
    st.caption(
        "Mete un ticker, una fecha de entrada y una fecha de vencimiento EXACTA "
        "y te devuelve los datos directamente — sin pasar por los filtros de "
        "tendencia, PCR, earnings, dividendo NI OI mínimo del escaneo masivo "
        "(esos aquí son solo informativos, no descartan el ticker). Sí se "
        "exigen siempre, por ser filtros fijos: opciones semanales y "
        "vencimiento exacto."
    )

    lq1, lq2, lq3, lq4 = st.columns([2, 1.5, 1.5, 1])
    with lq1:
        lookup_ticker_raw = st.text_input("Ticker", value="", placeholder="AAPL")
    with lq2:
        lookup_entry_date = st.date_input(
            "Fecha de entrada",
            value=date.today(),
            min_value=date.today(),
            max_value=date.today() + timedelta(days=179),
            format="DD/MM/YYYY",
            key="lookup_entry_date",
        )
    with lq3:
        lookup_target_date = st.date_input(
            "Fecha de vencimiento (exacta)",
            value=max(lookup_entry_date + timedelta(days=7), date.today() + timedelta(days=1)),
            min_value=lookup_entry_date + timedelta(days=1),
            max_value=lookup_entry_date + timedelta(days=180),
            format="DD/MM/YYYY",
            key="lookup_date",
        )
    with lq4:
        st.markdown("&nbsp;")
        lookup_btn = st.button("🔎 Consultar", use_container_width=True)

    if lookup_target_date <= lookup_entry_date:
        st.error("⚠️ El vencimiento debe ser posterior a la fecha de entrada.")

    if lookup_btn:
        lookup_ticker = _clean_ticker(lookup_ticker_raw)
        if not lookup_ticker:
            st.error("⚠️ Escribe un ticker válido.")
        else:
            with st.spinner(f"Consultando {lookup_ticker}..."):
                st.session_state["lookup_result"] = quick_lookup(
                    lookup_ticker, lookup_entry_date, lookup_target_date, params
                )

    lr = st.session_state.get("lookup_result")
    if lr:
        st.markdown(f"#### 📄 {lr['ticker']}")

        if lr.get("current_price") is not None:
            trend = lr.get("trend") or {}
            cal_info = lr.get("cal_info") or {}
            ed = cal_info.get("earnings_date")
            ed_txt = ed.strftime("%d %b %Y") if ed else "sin dato"
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("💲 Precio", f"${lr['current_price']:.2f}")
            m2.metric("📐 SMA30", f"{trend.get('sma30', 'N/D')}")
            m3.metric("📈 RV10 anualizada", f"{lr.get('rv', 'N/D')}%")
            m4.metric(
                "📅 Earnings en el periodo",
                f"⚠️ Sí ({ed_txt})" if lr.get("earnings_soon") else f"No ({ed_txt})",
                help="La fecha entre paréntesis es la que devuelve yfinance para este "
                     "ticker ('sin dato' si yfinance no la reporta). 'Sí' significa que "
                     "cae entre la fecha de entrada y la de vencimiento elegidas arriba."
            )

        if lr.get("error"):
            st.warning(f"⚠️ {lr['error']}")

        candidate = lr.get("candidate")
        if candidate:
            st.markdown(f"**Vencimiento usado:** {lr['exp_str']} (DTE: {lr['dte']}d)")
            colr1, colr2 = st.columns(2)
            with colr1:
                st.markdown(f"""
| Concepto | Valor |
|---|---|
| 🎯 Strike (deep ITM) | **${candidate['strike']}** |
| 💵 Prima Mid | ${candidate['mid']} |
| 📊 Bid / Ask | ${candidate['bid']} / ${candidate['ask']} |
| 🔺 Intrínseco | ${candidate['intrinsic']} |
| 🔹 Extrínseco | ${candidate['extrinsic']} ({candidate['extrinsic_pct']}%) |
| ✅ Cumple extrínseco mínimo | {"Sí" if lr.get('meets_extrinsic_min') else "No"} |
""")
            with colr2:
                st.markdown(f"""
| Concepto | Valor |
|---|---|
| 🛡️ Downside protection (prima total) | **{candidate['downside_prot']}%** |
| 🧱 · solo intrínseco | {candidate['downside_prot_intrinsic']}% |
| 📐 Delta | {candidate.get('delta', 'N/D')} |
| 📈 IV | {candidate.get('iv_pct', 'N/D')}% |
| 〰️ Spread | {candidate['spread_pct']}% |
| 💧 OI / Volumen | {candidate['oi']:,} / {candidate['volume']:,} |
| 🗳️ PCR | {lr.get('pcr', 'N/D')} |
| 💵 Riesgo dividendo | {"⚠️ Sí" if lr.get('dividend_risk') else "No"} |
""")

        st.divider()

    # ── Calculadora de combo buy-write ──────────────────────────────────
    st.markdown("### 🧮 Calculadora de Combo Buy-Write")
    st.caption(
        "Combo = comprar la acción + vender la call, a débito neto. "
        "Fórmula: combo = strike − extrínseco = strike − (extrínseco% × "
        "precio). Metes ticker + strike + tu extrínseco objetivo (mín, y "
        "máx opcional) y te da el precio de combo MÁXIMO que deberías "
        "pagar para cumplir el mínimo (y el mínimo, si fijas un techo). "
        "El precio del subyacente se pide en vivo cada vez que pulsas "
        "Calcular; si el strike existe en la cadena real de Alpaca para "
        "esa fecha, también te da el combo REAL para comparar."
    )

    cc1, cc2 = st.columns([2, 1])
    with cc1:
        calc_ticker_raw = st.text_input("Ticker", value="", placeholder="AAPL", key="calc_ticker_input")
    with cc2:
        calc_strike = st.number_input(
            "Strike", min_value=0.01, max_value=100000.0, value=25.0, step=0.5,
            key="calc_strike_input",
        )

    cc3, cc4 = st.columns(2)
    with cc3:
        calc_entry_date = st.date_input(
            "Fecha de entrada", value=date.today(),
            min_value=date.today(), max_value=date.today() + timedelta(days=179),
            format="DD/MM/YYYY", key="calc_entry_date",
        )
    with cc4:
        calc_target_date = st.date_input(
            "Fecha de vencimiento (exacta)",
            value=max(calc_entry_date + timedelta(days=7), date.today() + timedelta(days=1)),
            min_value=calc_entry_date + timedelta(days=1),
            max_value=calc_entry_date + timedelta(days=180),
            format="DD/MM/YYYY", key="calc_target_date_input",
        )

    cc5, cc6, cc7 = st.columns([1.5, 1.5, 1])
    with cc5:
        calc_extrinsic_min = st.number_input(
            "Extrínseco mínimo objetivo (%)", min_value=0.10, max_value=5.00,
            value=0.95, step=0.05, key="calc_extrinsic_min",
        )
    with cc6:
        calc_extrinsic_max = st.number_input(
            "Extrínseco máximo objetivo (%) — 0 = sin techo", min_value=0.0,
            max_value=10.0, value=0.0, step=0.05, key="calc_extrinsic_max",
        )
    with cc7:
        st.markdown("&nbsp;")
        calc_btn = st.button("🔄 Calcular", use_container_width=True, key="calc_btn")

    if calc_btn:
        calc_ticker = _clean_ticker(calc_ticker_raw)
        if not calc_ticker:
            st.error("⚠️ Escribe un ticker válido.")
        else:
            with st.spinner(f"Calculando {calc_ticker}..."):
                st.session_state["calc_result"] = build_buywrite_combo_calculator(
                    calc_ticker, calc_target_date, calc_strike, calc_extrinsic_min,
                    calc_extrinsic_max if calc_extrinsic_max > 0 else None,
                )

    cr = st.session_state.get("calc_result")
    if cr:
        st.markdown(f"#### 📄 {cr['ticker']} — Strike ${calc_strike}")
        if cr.get("error"):
            st.warning(f"⚠️ {cr['error']}")
        if cr.get("current_price") is not None:
            m1, m2, m3 = st.columns(3)
            m1.metric("💲 Precio en vivo", f"${cr['current_price']:.2f}")
            m2.metric("🔺 Intrínseco", f"${cr['intrinsic']:.2f}")
            m3.metric("📦 Combo MÁXIMO objetivo", f"${cr['combo_max']:.2f}")

            rows = [
                {"Concepto": "Prima call objetivo mínima", "Valor": f"${cr['premium_min']:.2f}"},
                {"Concepto": "Combo (débito) MÁXIMO a pagar", "Valor": f"${cr['combo_max']:.2f}"},
            ]
            if cr.get("combo_min") is not None:
                rows.append({"Concepto": "Prima call objetivo máxima", "Valor": f"${cr['premium_max']:.2f}"})
                rows.append({"Concepto": "Combo (débito) MÍNIMO a pagar", "Valor": f"${cr['combo_min']:.2f}"})
            if cr.get("real_mid") is not None:
                rows += [
                    {"Concepto": "Bid / Ask real", "Valor": f"${cr['real_bid']:.2f} / ${cr['real_ask']:.2f}"},
                    {"Concepto": "Mid real", "Valor": f"${cr['real_mid']:.2f}"},
                    {"Concepto": "Combo REAL (precio − mid real)", "Valor": f"${cr['real_combo']:.2f}"},
                    {"Concepto": "¿Cumple el objetivo con precios reales?", "Valor": "✅ Sí" if cr["cumple_real"] else "❌ No"},
                ]
            else:
                rows.append({"Concepto": "Datos reales de mercado",
                              "Valor": "No encontrados para ese strike/vencimiento exactos en Alpaca"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── Escaneo ────────────────────────────────────────────────────────
    st.markdown("### 🚀 Ejecutar Escaneo")

    with st.expander("🧪 Prueba rápida con universo reducido (opcional)"):
        st.caption(
            "Cada escaneo completo del universo genera cientos o miles de "
            "llamadas de red. Si estás ajustando parámetros (extrínseco, OI, "
            "tendencia...) y vas a relanzar el escaneo varias veces "
            "seguidas, prueba primero aquí con un puñado de tickers conocidos "
            "— así no repites la carga completa cada vez que cambias un valor. "
            "Cuando los parámetros te convenzan, deja esto vacío y lanza el "
            "escaneo completo. Si además quieres forzar datos 100% frescos "
            "(ignorando lo ya cacheado), usa el botón '🧹 Resetear Todo' arriba."
        )
        test_tickers_raw = st.text_input(
            "Tickers de prueba (separados por coma o espacio)",
            value="", placeholder="AAPL, MSFT, NVDA, KO",
        )
    test_tickers = [
        _clean_ticker(t) for t in test_tickers_raw.replace(",", " ").split()
    ] if test_tickers_raw.strip() else []
    test_tickers = [t for t in test_tickers if t]

    scan_universe = test_tickers if test_tickers else tickers_all

    scan_btn = st.button(
        "🎯 INICIAR ESCANEO",
        type="primary",
        use_container_width=True,
        disabled=len(scan_universe) == 0,
    )

    if test_tickers:
        st.caption(f"🧪 Modo prueba activo: se escanearán solo **{len(scan_universe)}** tickers.")
    else:
        st.caption(
            f"ℹ️ Se escanearán **{len(scan_universe):,}** tickers · Fase 1 en paralelo (rápida) "
            f"→ Fase 2 secuencial solo sobre los que sobrevivan a precio/SMA30 "
            f"(más lenta, ~1-2s por ticker)."
        )

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
            st.warning(
                "⚠️ Ningún ticker cumplió todos los filtros. Mira el embudo de "
                "diagnóstico más abajo para ver en qué fase/paso se están cayendo."
            )

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

    df_all = st.session_state["results"]
    ts = st.session_state["scan_ts"]

    # ── Filtros adicionales sobre resultados ───────────────────────────
    st.markdown("#### 🔧 Filtros adicionales sobre resultados")
    st.caption(
        "Se aplican sobre los candidatos ya encontrados en el último escaneo — "
        "no relanzan la descarga de datos, solo acotan la tabla/gráficos de "
        "abajo. Rank conserva la posición del escaneo completo (sin re-numerar)."
    )
    colrf1, colrf2, colrf3, colrf4 = st.columns(4)
    with colrf1:
        use_min_prot_filter = st.checkbox("Prot. intrínseca mínima (%)", value=False)
        min_prot_value = st.number_input(
            "Valor mínimo (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.5,
            disabled=not use_min_prot_filter, key="min_prot_value",
        )
    with colrf2:
        use_min_delta_filter = st.checkbox("Delta mínimo", value=False)
        min_delta_value = st.number_input(
            "Valor mínimo", min_value=0.0, max_value=1.0, value=0.85, step=0.01,
            disabled=not use_min_delta_filter, key="min_delta_value",
            help="Filas sin delta calculable (IV no disponible) se excluyen al "
                 "activar este filtro, ya que no se puede comparar."
        )
    with colrf3:
        use_min_oi_post_filter = st.checkbox("OI mínimo", value=False)
        min_oi_post_value = st.number_input(
            "Valor mínimo", min_value=0, max_value=100000, value=100, step=50,
            disabled=not use_min_oi_post_filter, key="min_oi_post_value",
        )
    with colrf4:
        use_min_vol_post_filter = st.checkbox("Volumen mínimo", value=False)
        min_vol_post_value = st.number_input(
            "Valor mínimo", min_value=0, max_value=100000, value=10, step=10,
            disabled=not use_min_vol_post_filter, key="min_vol_post_value",
            help="'Volumen' aquí es una aproximación de Alpaca (tamaño de la última "
                 "operación), no el volumen acumulado del día."
        )

    df = df_all.copy()
    if use_min_prot_filter:
        df = df[df["Prot_Intrinseca_%"] >= min_prot_value]
    if use_min_delta_filter:
        df = df[df["Delta"].notna() & (df["Delta"] >= min_delta_value)]
    if use_min_oi_post_filter:
        df = df[df["OI"] >= min_oi_post_value]
    if use_min_vol_post_filter:
        df = df[df["Volumen"] >= min_vol_post_value]

    any_post_filter = use_min_prot_filter or use_min_delta_filter or use_min_oi_post_filter or use_min_vol_post_filter
    if any_post_filter:
        st.caption(f"ℹ️ Mostrando **{len(df)}** de {len(df_all)} candidatos tras aplicar estos filtros.")

    st.divider()

    if df.empty:
        st.warning("⚠️ Ningún candidato del último escaneo cumple estos filtros adicionales. Ajusta los valores de arriba.")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📊 Candidatos totales", len(df))
    k2.metric("🛡️ Prot. intrínseca máx.", f"{df['Prot_Intrinseca_%'].max():.2f}%")
    k3.metric("💰 Extrínseco medio", f"{df['Extrínseco_%'].mean():.3f}%")
    k4.metric("🕐 Escaneo", ts.strftime("%H:%M:%S"))

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📋 Ranking Completo", "🔍 Detalle", "📈 Gráficos"])

    with tab1:
        cols_show = [
            "Rank", "Ticker", "Precio", "Strike", "Prot_Intrinseca_%",
            "Extrínseco_%", "Prima_Mid", "Bid", "Ask",
            "Delta", "DTE", "Vencimiento",
            "IV_%", "RV_%", "IV_RV",
            "OI", "Volumen", "Spread_%",
            "SMA30", "Dist_SMA30_%", "SMA30_Sube", "SMA10", "SMA10>SMA30",
            "PCR",
        ]
        cols_show = [c for c in cols_show if c in df.columns]

        def color_downside(val):
            if val >= 15:
                return "background-color:#1e3a1e; color:#6daa45"
            if val >= 10:
                return "background-color:#3a3a1e; color:#e8af34"
            return ""

        def color_iv_rv(val):
            # Un IV muy por encima de la RV suele indicar riesgo de evento
            # (FDA, litigio, M&A rumor) que ni el filtro de earnings ni el
            # de tendencia detectan. Es solo un aviso visual, no un filtro:
            # a veces el IV alto es legítimo (nombre volátil de siempre).
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

        styler = df[cols_show].style.map(color_downside, subset=["Prot_Intrinseca_%"])
        if "IV_RV" in cols_show:
            styler = styler.map(color_iv_rv, subset=["IV_RV"])

        st.dataframe(
            styler,
            use_container_width=True, height=550,
        )
        if "IV_RV" in cols_show:
            st.caption(
                "🟧🟥 IV_RV resaltado = la IV de la opción supera con creces la "
                "volatilidad realizada reciente. No es necesariamente malo, pero "
                "conviene mirar por qué antes de operar (catalizador conocido, "
                "M&A, litigio, evento regulatorio...)."
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
| 🛡️ Protección intrínseca | **{row['Prot_Intrinseca_%']}%** |
| 📐 Delta | {row['Delta']} |
""")

            with col_b:
                st.markdown("### 📊 Contexto")
                st.markdown(f"""
| Concepto | Valor |
|---|---|
| 📈 IV / RV | {row['IV_%']}% / {row['RV_%']}% (ratio: {row['IV_RV']}) |
| 💧 OI / Volumen | {row['OI']:,} / {row['Volumen']:,} |
| 〰️ Spread bid-ask | {row['Spread_%']}% |
| 📐 SMA30 | {row['SMA30']} ({row['Dist_SMA30_%']}% sobre SMA) |
| ↗️ SMA30 subiendo | {row['SMA30_Sube']} |
| ⏩ SMA10 | {row.get('SMA10', 'N/D')} (> SMA30: {row.get('SMA10>SMA30', 'N/D')}) |
| 🗳️ PCR | {row['PCR']} |
""")

            st.markdown("---")
            st.markdown("### 💡 Interpretación")

            prot = row["Prot_Intrinseca_%"]
            ext = row["Extrínseco_%"]

            if prot >= 10:
                st.success(f"🟢 **Protección intrínseca alta**: el precio puede caer un {prot:.1f}% antes de que la call salga del dinero.")
            elif prot >= 5:
                st.warning(f"🟡 **Protección intrínseca moderada**: {prot:.1f}% de margen hasta el strike.")
            else:
                st.error(f"🔴 **Protección intrínseca baja**: solo {prot:.1f}% de margen hasta el strike.")

            st.info(
                f"Con un extrínseco del **{ext}%** sobre un precio de **${row['Precio']}**, "
                f"cobras **${row['Extrínseco_$']}** por acción de prima pura (valor tiempo). "
                f"Si el subyacente cierra por encima del strike **${row['Strike']}** al "
                f"vencimiento, te quedas esa prima íntegra."
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
                df, x="Extrínseco_%", y="Prot_Intrinseca_%", text="Ticker",
                color="Prot_Intrinseca_%",
                color_continuous_scale=["#dd6974", "#e8af34", "#6daa45"],
                title="Protección Intrínseca vs Extrínseco%",
                template="plotly_dark", height=420,
                labels={"Extrínseco_%": "Extrínseco (%)", "Prot_Intrinseca_%": "Protección Intrínseca (%)"},
            )
            fig_scatter.update_traces(textposition="top center", marker_size=10)
            st.plotly_chart(fig_scatter, use_container_width=True)

        fig_bar = px.bar(
            df.head(20), x="Ticker", y="Prot_Intrinseca_%",
            color="Prot_Intrinseca_%",
            color_continuous_scale=["#dd6974", "#e8af34", "#6daa45"],
            title="Top 20 — Protección Intrínseca % (mayor = más deep ITM)",
            template="plotly_dark", height=400,
            labels={"Prot_Intrinseca_%": "Protección Intrínseca (%)"},
        )
        st.plotly_chart(fig_bar, use_container_width=True)

if __name__ == "__main__":
    main()
