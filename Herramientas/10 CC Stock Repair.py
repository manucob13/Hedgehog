"""
CC Stock Repair
===============
Objetivo: bajar el coste base (CB) de acciones en recuperación vendiendo
covered calls y elegir el STRIKE (y ver su delta) alejándote del ATM lo justo
para llevar tu ratio de ajustes/primas desde el valor actual hasta el objetivo.

IDEA: NORMALIZAR DESDE EL ATM
    Tú pones:
      - Ratio ACTUAL en ATM (de tu Excel, p. ej. 61 %)
      - Ratio OBJETIVO (p. ej. 50 %)
    Los dos juntos fijan la REDUCCIÓN RELATIVA que hay que conseguir:
        reducción necesaria = objetivo / actual   (50 / 61 = 0,82)
    El modelo NO intenta adivinar el nivel del ratio (eso lo sabes tú, lo
    mides en tu Excel). Solo estima CUÁNTO MEJORA el ratio al alejarte del
    ATM, usando IV y volatilidad real (ATR):

        ratio(K) / ratio(ATM)  =  [ C(K, σ_real) / C(K, σ_IV) ]  /  [ ATM, igual ]

    donde C es el valor Black-Scholes de la call (sin deriva) con el
    movimiento esperado de la volatilidad real o de la IV. Con una IV plana
    (la del ATM) la curva es lisa y no depende del ruido de primas de 0,01.
    Ratio ajustado de cada strike = ratio actual × esa reducción.

    ρ (rho) = movimiento que cobra la IV / movimiento real (ATR), en el
    horizonte de la operación (365 días naturales para la IV, 252 días de
    mercado para el ATR: la comparación ya corrige la base de tiempo).
    Aproximadamente:  reducción(x) ≈ exp(-½ (ρ² - 1) x²)  con x = distancia
    en movimientos esperados. Si ρ ≈ 1 no hay forma de bajar el ratio sin
    quedarte sin prima; si ρ es alto, bajarlo es barato.

DISTANCIA Y TOPES (en MOVIMIENTOS ESPERADOS, EM)
    EM = precio × media ponderada del movimiento de la IV y el de la
    volatilidad real (peso de la IV: 0,5 por defecto, ajustable).
    Distancia de un strike = (strike - precio) / EM.
      - TOPE (por defecto 0,75 EM): nunca te alejas más que el strike
        listado más cercano a esa distancia (delta ≈ 0,2 - 0,25).
      - MÍNIMO (por defecto 0): distancia mínima al ATM; ponlo, p. ej., en
        0,5 si no quieres strikes casi ATM.

RECOMENDACIÓN (tres casos)
    1) ✅ Objetivo alcanzado: el strike más cercano al ATM (≥ mínimo) cuyo
       ratio ajustado es <= objetivo, siempre que no pase del tope.
    2) ⚠️ Tope: el objetivo exige alejarse más que el tope; se usa el strike
       más cercano al tope y se avisa del ratio que sí se consigue, de lo
       que mejora y de la prima que se sacrifica.
    3) ⚠️ Sin mejora: ρ <= 1 (la IV no cobra más que lo que se mueve la
       acción). Alejarse no mejora el ratio según el modelo; se muestra el
       strike más cercano al ATM permitido (respetando el mínimo).
    Cambiar ratio actual, objetivo, tope o mínimo recalcula la
    recomendación al instante, sin volver a descargar datos.

FECHAS
    Fecha de entrada y de salida (vencimiento). DTE = días naturales entre
    ambas; los días de mercado (lunes-viernes, sin festivos) se usan para la
    volatilidad real. Si no existe el vencimiento exacto, puedes usar el más
    cercano disponible (casilla activada por defecto; la herramienta siempre
    indica qué vencimiento ha usado).
    Aviso: la prima, la IV y el delta de la cadena son los de HOY. Si la
    fecha de entrada es futura, el DTE se cuenta desde esa fecha pero los
    precios de las opciones siguen siendo los actuales.

VOLATILIDAD REAL (dos fuentes, eliges cuál usar)
    - ATR: ATR(14) de Wilder / precio, convertido a desviación típica diaria
      dividiendo por un factor (1.5 por defecto, ajustable) y anualizado
      con raíz de 252.
    - Cierres 20d: desviación típica de los retornos logarítmicos de los
      últimos 20 cierres, anualizada.
    Las dos se muestran siempre para que puedas contrastarlas.

DELTA
    Se toma de Alpaca si la cadena lo trae; si no, se calcula con
    Black-Scholes usando la IV de cada strike. La columna Δ_fuente indica
    cuál se usó en cada fila.

PRIMA VS ATM
    Prima_vs_ATM_% = valor extrínseco del strike / valor extrínseco del
    strike más cercano al precio. Te dice cuánta prima sacrificas.

DATOS
    Opciones y precio en vivo: Alpaca (utils/utils_alpaca.py).
    Histórico diario para ATR / volatilidad real: yfinance.

LÍMITES DEL MODELO
    El modelo estima la FORMA de la mejora al alejarte del ATM, no tu ratio
    real: las colas gordas, los gaps y las tendencias hacen que lo que mides
    en el Excel difiera. Contrástalo ciclo a ciclo y ajusta objetivo, tope
    y mínimo.
"""

import math
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, date
import warnings
import socket
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.utils import check_password
from utils.utils_alpaca import (
    get_option_expirations,
    get_option_chain,
    get_live_price,
)

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cc_stock_repair")

SOCKET_TIMEOUT = 15
socket.setdefaulttimeout(SOCKET_TIMEOUT)

RISK_FREE_RATE = 0.045         # solo para el delta Black-Scholes de reserva
TRADING_DAYS = 252
DEFAULT_RATIO_ACTUAL = 61.0    # % — tu ratio actual en ATM (Excel)
DEFAULT_TARGET_RATIO = 50.0    # % — ratio objetivo
DEFAULT_CAP_EM = 0.75          # tope de distancia (movimientos esperados)
DEFAULT_MIN_EM = 0.0           # distancia mínima (movimientos esperados)
DEFAULT_EM_IV_WEIGHT = 0.5     # peso de la IV en el EM (el resto, volatilidad real)
DEFAULT_ATR_PERIOD = 14
DEFAULT_ATR_TO_SIGMA = 1.5     # ATR ≈ 1.5 × desviación típica diaria (ajustable)
DEFAULT_MAX_OTM_PCT = 25.0     # strikes hasta +25 % sobre el precio
DEFAULT_SPREAD_MAX_PCT = 50    # spread máximo, % de la prima
DEFAULT_MIN_OI = 50            # OI mínimo
ATM_LOOKBACK_PCT = 2.0         # strikes hasta 2 % por debajo del precio (referencia ATM)
REQ_DIST_MAX_EM = 6.0          # límite de búsqueda de la distancia necesaria

STATUS_LABELS = {
    "ok": "✅ Objetivo alcanzado",
    "tope": "⚠️ Tope: no llega",
    "sin_mejora": "⚠️ Sin mejora (IV ≈ real)",
}

# ======================================================================
# 0. UTILIDADES
# ======================================================================

def _clean_ticker(raw):
    if raw is None:
        return None
    t = str(raw).strip().replace("&amp;", "&").upper()
    if not t or t in ("-", "NAN", "N/A"):
        return None
    return t.replace(" ", "")

def _nz(x):
    """None -> NaN (para construir Series numéricas)."""
    return np.nan if x is None else x

def parse_tickers(raw):
    out = []
    for t in str(raw or "").replace(",", " ").replace(";", " ").split():
        t = _clean_ticker(t)
        if t and t not in out:
            out.append(t)
    return out

def parse_cost_basis(raw):
    """'SOFI=14.2, HOOD=30' -> {'SOFI': 14.2, 'HOOD': 30.0}. Punto decimal."""
    out = {}
    if not raw or not str(raw).strip():
        return out
    for part in str(raw).replace(";", ",").split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        t = _clean_ticker(k)
        try:
            val = float(v.strip())
        except Exception:
            continue
        if t and val > 0:
            out[t] = val
    return out

# ======================================================================
# 1. DATOS DIARIOS + ATR / VOLATILIDAD REAL (yfinance)
# ======================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def get_daily_data(ticker):
    """Histórico diario AJUSTADO vía yf.Ticker().history(). Ventana de 300
    días calendario; mínimo 40 filas para poder calcular ATR y volatilidad."""
    try:
        end = datetime.now() + timedelta(days=1)
        start = end - timedelta(days=300)
        data = yf.Ticker(ticker).history(start=start, end=end, interval="1d", auto_adjust=True)
        if data is None or data.empty:
            return None
        data = data.dropna(subset=["Close", "High", "Low"])
        if len(data) < 40:
            return None
        data.index = pd.to_datetime(data.index)
        return data
    except Exception as e:
        logger.warning(f"[daily_data] {ticker}: {e}")
        return None

def compute_vol_metrics(data, atr_period, atr_to_sigma):
    """ATR de Wilder y dos estimaciones de volatilidad real anualizada (%):
    una derivada del ATR y otra de los cierres de los últimos 20 días."""
    try:
        high, low, close = data["High"], data["Low"], data["Close"]
        prev = close.shift(1)
        tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0 / atr_period, adjust=False).mean()

        atr_now = float(atr.iloc[-1])
        last_close = float(close.iloc[-1])
        if last_close <= 0 or atr_now <= 0:
            return None

        atr_pct = atr_now / last_close * 100
        sigma_daily = (atr_pct / 100) / atr_to_sigma
        rv_atr = sigma_daily * np.sqrt(TRADING_DAYS) * 100

        rets = np.log(close / prev).dropna()
        rv20 = float(rets.iloc[-20:].std() * np.sqrt(TRADING_DAYS) * 100) if len(rets) >= 20 else None

        return {
            "atr": round(atr_now, 3),
            "atr_pct": round(atr_pct, 2),
            "rv_atr": round(float(rv_atr), 2),
            "rv20": round(rv20, 2) if rv20 is not None else None,
            "last_close": last_close,
        }
    except Exception as e:
        logger.warning(f"[vol_metrics] {e}")
        return None

# ======================================================================
# 2. BLACK-SCHOLES Y MODELO DE REDUCCIÓN RELATIVA DEL RATIO
# ======================================================================

def _ncdf(z):
    """CDF normal estándar, precisa también en las colas (erfc)."""
    return 0.5 * math.erfc(-z / math.sqrt(2.0))

def bs_call_price(S, K, T, sigma, r=0.0):
    """Valor Black-Scholes de una call europea. Con r=0 equivale a esperar
    que la acción no tenga deriva: E[max(S_T - K, 0)]. Pasando T=1 el
    parámetro sigma es directamente la desviación típica total del periodo."""
    try:
        if S <= 0 or K <= 0:
            return 0.0
        if T <= 0 or sigma <= 0:
            return max(S - K, 0.0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return float(S * _ncdf(d1) - K * np.exp(-r * T) * _ncdf(d2))
    except Exception:
        return max(S - K, 0.0)

def bs_call_delta(S, K, T_years, sigma, r=RISK_FREE_RATE):
    """Delta de una call europea (Black-Scholes): N(d1), entre 0 y 1."""
    try:
        if T_years <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return None
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T_years) / (sigma * np.sqrt(T_years))
        return round(float(_ncdf(d1)), 3)
    except Exception:
        return None

def model_ratio(S, K, s_iv, s_rv):
    """Coste esperado de recompra (movimiento real) / prima (movimiento de la
    IV plana del ATM), para una call de strike K. s_iv y s_rv son las
    desviaciones típicas TOTALES del periodo (no anualizadas)."""
    p_iv = bs_call_price(S, K, 1.0, s_iv)
    p_rv = bs_call_price(S, K, 1.0, s_rv)
    if p_iv <= 1e-12:
        return np.nan
    return p_rv / p_iv

def rel_ratio(S, K, s_iv, s_rv):
    """Ratio del strike K relativo al ratio en ATM (K = precio). 1,0 en el
    ATM; por debajo de 1 mejora al alejarse (solo si ρ = s_iv/s_rv > 1)."""
    den = model_ratio(S, S, s_iv, s_rv)
    num = model_ratio(S, K, s_iv, s_rv)
    if pd.isna(den) or den <= 0 or pd.isna(num):
        return np.nan
    return num / den

def required_distance(S, s_iv, s_rv, em, rel_target, hi_em=REQ_DIST_MAX_EM):
    """Distancia al ATM (en EM) a la que el ratio relativo baja hasta
    rel_target. 0 si el objetivo ya se cumple; None si no se alcanza ni
    a hi_em movimientos esperados (p. ej. cuando ρ <= 1)."""
    if rel_target >= 1.0:
        return 0.0

    def f(d):
        return rel_ratio(S, S + d * em, s_iv, s_rv)

    f_hi = f(hi_em)
    if pd.isna(f_hi) or f_hi > rel_target:
        return None
    lo, hi = 0.0, hi_em
    for _ in range(50):
        mid = (lo + hi) / 2
        v = f(mid)
        if pd.isna(v) or v > rel_target:
            lo = mid
        else:
            hi = mid
    return round(hi, 3)

# ======================================================================
# 3. TABLA DE STRIKES: prima, delta, distancia en EM y reducción relativa
# ======================================================================

def build_strike_table(calls, S, dte_cal, n_td, sigma_real, p, cost_basis=None):
    """A partir de la cadena de calls del vencimiento devuelve
    (DataFrame de strikes, info) con info = iv_atm, s_iv, s_rv, em, rho.
    Devuelve (None, info) si no hay calls cotizadas en el rango o no hay IV."""
    info = {"iv_atm": None, "s_iv": None, "s_rv": None, "em": None, "rho": None}
    try:
        c = calls.copy().reset_index(drop=True)
        c["strike"] = pd.to_numeric(c["strike"], errors="coerce")
        c = c.dropna(subset=["strike"])
        for col in ("bid", "ask"):
            c[col] = pd.to_numeric(c[col], errors="coerce").fillna(0)
        c["mid"] = (c["bid"] + c["ask"]) / 2
        c = c[(c["bid"] > 0) & (c["ask"] > 0) & (c["mid"] > 0)].reset_index(drop=True)
        if c.empty:
            return None, info

        if "impliedVolatility" in c.columns:
            c["iv"] = pd.to_numeric(c["impliedVolatility"], errors="coerce")
        else:
            c["iv"] = np.nan

        # IV del ATM: strike más cercano al precio con IV válida
        with_iv = c[c["iv"] > 0]
        if with_iv.empty:
            return None, info
        idx = (with_iv["strike"] - S).abs().idxmin()
        iv_atm = float(with_iv.loc[idx, "iv"]) * 100

        # Movimientos totales del periodo (IV: días naturales/365; real: días de mercado/252)
        s_iv = (iv_atm / 100) * np.sqrt(max(dte_cal, 1) / 365.0)
        s_rv = sigma_real * np.sqrt(max(n_td, 1) / TRADING_DAYS)
        w = float(p["em_iv_weight"])
        em = S * (w * s_iv + (1 - w) * s_rv)
        rho = s_iv / s_rv if s_rv > 0 else None
        info.update({"iv_atm": iv_atm, "s_iv": s_iv, "s_rv": s_rv, "em": em, "rho": rho})
        if em <= 0 or rho is None:
            return None, info

        # Rango de strikes a evaluar
        lo = S * (1 - ATM_LOOKBACK_PCT / 100)
        hi = S * (1 + p["max_otm_pct"] / 100)
        c = c[(c["strike"] >= lo) & (c["strike"] <= hi)].reset_index(drop=True)
        if c.empty:
            return None, info

        # Valor extrínseco y referencia ATM (antes de los filtros de liquidez)
        c["ext"] = c["mid"] - np.maximum(S - c["strike"], 0.0)
        j = (c["strike"] - S).abs().idxmin()
        atm_ext = float(c.loc[j, "ext"])
        if atm_ext <= 0:
            atm_ext = float(c.loc[j, "mid"])

        c["oi"] = pd.to_numeric(c["openInterest"], errors="coerce").fillna(0) if "openInterest" in c.columns else 0
        c["vol"] = pd.to_numeric(c["volume"], errors="coerce").fillna(0) if "volume" in c.columns else 0

        # Delta: Alpaca si la cadena lo trae; si no, Black-Scholes con la IV del strike
        if "delta" in c.columns:
            d_alp = pd.to_numeric(c["delta"], errors="coerce")
            d_alp = d_alp.where((d_alp > 0) & (d_alp <= 1))
        else:
            d_alp = pd.Series(np.nan, index=c.index)
        T_delta = max(dte_cal, 1) / 365.0
        d_bs = pd.Series(
            [
                _nz(bs_call_delta(S, float(k), T_delta, float(v))) if (pd.notna(v) and v > 0) else np.nan
                for k, v in zip(c["strike"], c["iv"])
            ],
            index=c.index,
            dtype=float,
        )
        c["delta"] = d_alp.where(d_alp.notna(), d_bs)
        c["delta_src"] = np.where(d_alp.notna(), "Alpaca", np.where(d_bs.notna(), "BS", "—"))

        # Reducción relativa del ratio respecto al ATM (modelo liso, IV plana)
        den = model_ratio(S, S, s_iv, s_rv)
        if pd.notna(den) and den > 0:
            c["rel"] = [model_ratio(S, float(k), s_iv, s_rv) / den for k in c["strike"]]
        else:
            c["rel"] = np.nan

        c["spread_pct"] = (c["ask"] - c["bid"]) / c["mid"] * 100

        # Filtros de liquidez opcionales
        if p["use_oi_filter"]:
            c = c[c["oi"] >= p["min_oi"]]
        if p["use_spread_filter"]:
            c = c[c["spread_pct"] <= p["spread_max_pct"]]
        c = c.reset_index(drop=True)
        if c.empty:
            return None, info

        out = pd.DataFrame({
            "Strike": c["strike"].round(2),
            "OTM_%": ((c["strike"] - S) / S * 100).round(2),
            "Dist_EM": ((c["strike"] - S) / em).round(2),
            "Delta": c["delta"].round(3),
            "Δ_fuente": c["delta_src"],
            "Prima_Mid": c["mid"].round(2),
            "Bid": c["bid"].round(2),
            "Ask": c["ask"].round(2),
            "Prima_%": (c["mid"] / S * 100).round(2),
            "Prima_vs_ATM_%": (c["ext"] / atm_ext * 100).round(0),
            "Prima_anual_%": (c["mid"] / S * 100 * 365 / max(dte_cal, 1)).round(1),
            "IV_%": (c["iv"] * 100).round(1),
            "Rel_ratio": c["rel"].round(4),
            "OI": c["oi"].astype(int),
            "Volumen": c["vol"].astype(int),
            "Spread_%": c["spread_pct"].round(1),
        })
        if cost_basis:
            out["Prima/CB_%"] = (c["mid"] / cost_basis * 100).round(2)
            out["Bajo_CB"] = np.where(c["strike"] < cost_basis, "⚠️", "")
        out = out.sort_values("Strike").reset_index(drop=True)
        return out, info
    except Exception as e:
        logger.warning(f"[strike_table] {e}")
        return None, info

def add_adjusted(df, ratio_actual):
    """Añade el ratio ajustado (ratio actual × reducción relativa del modelo)
    y el neto ajustado por acción (prima × (1 - ratio ajustado))."""
    d = df.copy()
    d["Ratio_aj_%"] = (d["Rel_ratio"] * ratio_actual).round(1)
    d["Neto_aj"] = (d["Prima_Mid"] * (1 - d["Ratio_aj_%"] / 100)).round(2)
    return d

def pick_strike(res, df_adj, ratio_actual, target, cap_em, min_em):
    """Devuelve (fila recomendada, estado, distancia necesaria en EM).
    Estados: 'ok' (objetivo alcanzado dentro del tope), 'tope' (hay que
    quedarse en el tope), 'sin_mejora' (ρ <= 1: alejarse no mejora)."""
    S = res["price"]
    rel_target = target / ratio_actual
    x_need = required_distance(S, res["s_iv"], res["s_rv"], res["em"], rel_target)

    df_min = df_adj[df_adj["Dist_EM"] >= min_em - 1e-9]
    if df_min.empty:
        df_min = df_adj[df_adj["Dist_EM"] >= 0]
    if df_min.empty:
        df_min = df_adj
    df_min = df_min.sort_values("Strike")

    # Strike listado más cercano al tope (dentro de lo permitido por el mínimo)
    cap_row = df_min.loc[(df_min["Dist_EM"] - cap_em).abs().idxmin()]

    reachable = df_min[df_min["Ratio_aj_%"] <= target]
    if not reachable.empty:
        cand = reachable.iloc[0]
        if cand["Strike"] <= cap_row["Strike"]:
            return cand, "ok", x_need

    if res["rho"] is not None and res["rho"] <= 1.0:
        return df_min.iloc[0], "sin_mejora", x_need

    return cap_row, "tope", x_need

# ======================================================================
# 4. ANÁLISIS DE UN TICKER
# ======================================================================

def analyze_ticker(ticker, entry_date, exit_date, p, cost_basis=None):
    out = {"ticker": ticker, "error": None}

    data = get_daily_data(ticker)
    if data is None:
        out["error"] = "Sin datos diarios suficientes (yfinance)."
        return out

    vol = compute_vol_metrics(data, p["atr_period"], p["atr_to_sigma"])
    if vol is None:
        out["error"] = "No se pudo calcular el ATR / la volatilidad real."
        return out

    try:
        S = float(get_live_price(ticker, vol["last_close"]))
    except Exception:
        S = vol["last_close"]
    if not S or S <= 0:
        S = vol["last_close"]

    expirations = get_option_expirations(ticker)
    if not expirations:
        out["error"] = "Sin vencimientos de opciones listados en Alpaca."
        return out

    valid = sorted(e for e in expirations if e > entry_date)
    if not valid:
        out["error"] = "No hay vencimientos posteriores a la fecha de entrada."
        return out

    if exit_date in valid:
        exp = exit_date
    elif p["use_nearest_exp"]:
        exp = min(valid, key=lambda e: abs((e - exit_date).days))
    else:
        nearest = sorted(sorted(valid, key=lambda e: abs((e - exit_date).days))[:5])
        out["error"] = (
            f"No existe el vencimiento exacto {exit_date.strftime('%d/%m/%Y')}. Más cercanos: "
            + ", ".join(e.strftime("%d/%m/%Y") for e in nearest) + "."
        )
        return out

    dte_cal = (exp - entry_date).days
    n_td = max(int(np.busday_count(entry_date, exp)), 1)

    calls, _ = get_option_chain(ticker, exp)
    if calls is None or calls.empty:
        out["error"] = "Cadena de calls vacía/no disponible en Alpaca para ese vencimiento."
        return out

    rv_used = vol["rv_atr"] if p["vol_source"] == "ATR" else vol["rv20"]
    if rv_used is None:
        rv_used = vol["rv_atr"]
    sigma_real = rv_used / 100.0

    table, info = build_strike_table(calls, S, dte_cal, n_td, sigma_real, p, cost_basis)
    if info["iv_atm"] is None:
        out["error"] = ("No hay IV válida en la cadena (bid>0, ask>0 e IV>0): no se puede calcular "
                        "el movimiento esperado.")
        return out
    if table is None or table.empty:
        out["error"] = (
            "Ninguna call cotizada (bid>0 y ask>0) cumple el rango de strikes y los filtros de "
            "liquidez. Prueba a relajar el spread/OI o a ampliar el rango OTM en Ajustes avanzados."
        )
        return out

    out.update({
        "price": round(S, 2),
        "expiration": exp,
        "requested_exit": exit_date,
        "entry_date": entry_date,
        "dte_cal": dte_cal,
        "n_td": n_td,
        "iv_atm": round(info["iv_atm"], 1),
        "rv_atr": vol["rv_atr"],
        "rv20": vol["rv20"],
        "rv_used": round(rv_used, 1),
        "vol_source": p["vol_source"],
        "s_iv": info["s_iv"],
        "s_rv": info["s_rv"],
        "em": info["em"],
        "rho": info["rho"],
        "atr": vol["atr"],
        "atr_pct": vol["atr_pct"],
        "em_real": round(S * info["s_rv"], 2),
        "em_iv": round(S * info["s_iv"], 2),
        "table": table,
        "cost_basis": cost_basis,
        "daily": data,
    })
    return out

# ======================================================================
# 5. GRÁFICOS
# ======================================================================

def plot_strike_curve(df, rec, target, ratio_actual, ticker, res, cap_em, min_em):
    """Barras: prima. Línea verde: neto ajustado ($). Línea ámbar: ratio
    ajustado (eje derecho) frente al objetivo (roja) y al ratio actual
    (gris). Estrella: strike recomendado. Línea vertical: tope de distancia."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=df["Strike"], y=df["Prima_Mid"], name="Prima mid ($)",
        marker_color="#5b9bd5", opacity=0.7,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df["Strike"], y=df["Neto_aj"], mode="lines+markers", name="Neto ajustado ($)",
        line=dict(color="#6daa45", width=2),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df["Strike"], y=df["Ratio_aj_%"], mode="lines+markers", name="Ratio ajustado (%)",
        line=dict(color="#e8af34", width=2),
    ), secondary_y=True)
    x_rng = [df["Strike"].min(), df["Strike"].max()]
    fig.add_trace(go.Scatter(
        x=x_rng, y=[target, target], mode="lines",
        name=f"Objetivo {target:.0f}%", line=dict(color="#dd6974", dash="dash", width=2),
    ), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=x_rng, y=[ratio_actual, ratio_actual], mode="lines",
        name=f"Actual ATM {ratio_actual:.0f}%", line=dict(color="#999999", dash="dot", width=1),
    ), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=[rec["Strike"]], y=[rec["Ratio_aj_%"]], mode="markers", name="Recomendado",
        marker=dict(symbol="star", size=17, color="#ffffff", line=dict(color="#e8af34", width=2)),
    ), secondary_y=True)
    cap_k = res["price"] + cap_em * res["em"]
    fig.add_vline(x=float(cap_k), line=dict(color="#dd6974", width=1, dash="dot"),
                  annotation_text=f"Tope {cap_em:.2f} EM", annotation_position="top")
    if min_em > 0:
        min_k = res["price"] + min_em * res["em"]
        fig.add_vline(x=float(min_k), line=dict(color="#5b9bd5", width=1, dash="dot"),
                      annotation_text=f"Mín. {min_em:.2f} EM", annotation_position="top")
    fig.update_xaxes(title_text="Strike")
    fig.update_yaxes(title_text="$ por acción", secondary_y=False)
    fig.update_yaxes(title_text="Ratio ajustado (%)", secondary_y=True)
    fig.update_layout(
        title=f"{ticker} — prima, neto y ratio ajustado por strike",
        template="plotly_dark", height=470, hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
    )
    return fig

def plot_price(res, rec, cap_em, days=90):
    """Velas + línea del precio, strike recomendado, strike del tope, coste
    base (si lo diste) y banda del movimiento esperado (precio ± EM)."""
    try:
        data = res["daily"].copy()
        if getattr(data.index, "tz", None) is not None:
            data.index = data.index.tz_localize(None)
        recent = data.iloc[-days:]

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=recent.index, open=recent["Open"], high=recent["High"],
            low=recent["Low"], close=recent["Close"], name=res["ticker"],
            increasing_line_color="#6daa45", decreasing_line_color="#dd6974",
        ))

        S = res["price"]
        em = res["em"]
        cap_k = S + cap_em * em
        fig.add_hrect(y0=S - em, y1=S + em, fillcolor="rgba(91,155,213,0.10)", line_width=0)
        fig.add_hline(y=S, line=dict(color="#ffffff", width=1),
                      annotation_text=f"Precio {S:.2f}", annotation_position="left",
                      annotation_font=dict(size=10))
        fig.add_hline(y=float(rec["Strike"]), line=dict(color="#e8af34", width=2, dash="dot"),
                      annotation_text=f"Strike {rec['Strike']:.2f}", annotation_position="right",
                      annotation_font=dict(size=11, color="#e8af34"))
        fig.add_hline(y=float(cap_k), line=dict(color="#dd6974", width=1, dash="dot"),
                      annotation_text=f"Tope {cap_k:.2f}", annotation_position="left",
                      annotation_font=dict(size=10, color="#dd6974"))
        y_vals = [float(recent["Low"].min()), float(recent["High"].max()),
                  S - em, S + em, float(rec["Strike"]), float(cap_k)]
        if res.get("cost_basis"):
            cb = float(res["cost_basis"])
            fig.add_hline(y=cb, line=dict(color="#dd6974", width=1.5, dash="dash"),
                          annotation_text=f"CB {cb:.2f}", annotation_position="right",
                          annotation_font=dict(size=10, color="#dd6974"))
            y_vals.append(cb)

        pad = (max(y_vals) - min(y_vals)) * 0.05
        fig.update_layout(
            title=f"{res['ticker']} — precio, strike recomendado y movimiento esperado (±{em:.2f})",
            template="plotly_dark", height=450, xaxis_rangeslider_visible=False,
            yaxis=dict(range=[min(y_vals) - pad, max(y_vals) + pad]),
            margin=dict(r=110),
        )
        return fig
    except Exception as e:
        logger.warning(f"[plot_price] {e}")
        return None

# ======================================================================
# 6. RESULTADOS
# ======================================================================

def render_results(ratio_actual, target, cap_em, min_em):
    st.markdown("### 📊 Resultados")

    results = st.session_state.get("cc_results")
    if not results:
        st.info("👆 Escribe tus tickers, revisa las fechas y pulsa **CALCULAR STRIKES**.")
        return

    ts = st.session_state.get("cc_ts")
    if ts:
        st.caption(f"🕐 Último cálculo: {ts.strftime('%d/%m/%Y %H:%M:%S')} · ratio actual, objetivo, "
                   f"tope y mínimo se aplican al instante, sin recalcular datos.")

    ok = [r for r in results if not r.get("error")]
    bad = [r for r in results if r.get("error")]
    for r in bad:
        st.warning(f"⚠️ **{r['ticker']}**: {r['error']}")
    if not ok:
        return

    if target >= ratio_actual:
        st.info("ℹ️ El ratio objetivo no es menor que tu ratio actual: no hace falta alejarse del ATM; "
                "solo se aplica la distancia mínima.")

    # ── Resumen: una fila por ticker ───────────────────────────────────
    rows = []
    picks = {}
    for r in ok:
        adj = add_adjusted(r["table"], ratio_actual)
        rec, status, x_need = pick_strike(r, adj, ratio_actual, target, cap_em, min_em)
        picks[r["ticker"]] = (adj, rec, status, x_need)
        rows.append({
            "Ticker": r["ticker"],
            "Precio": r["price"],
            "Vencimiento": r["expiration"].strftime("%d/%m/%Y"),
            "DTE": r["dte_cal"],
            "IV_ATM_%": r["iv_atm"],
            "Vol_real_%": r["rv_used"],
            "ρ (IV/real)": round(r["rho"], 2),
            "Dist_necesaria_EM": x_need,
            "Strike": float(rec["Strike"]),
            "Dist_EM": float(rec["Dist_EM"]),
            "OTM_%": float(rec["OTM_%"]),
            "Delta": rec["Delta"],
            "Prima": float(rec["Prima_Mid"]),
            "Prima_vs_ATM_%": float(rec["Prima_vs_ATM_%"]),
            "Ratio_aj_%": float(rec["Ratio_aj_%"]),
            "Estado": STATUS_LABELS[status],
        })
    st.markdown(
        f"#### 🎯 Strike recomendado por ticker (de {ratio_actual:.0f}% a {target:.0f}%, "
        f"tope {cap_em:.2f} EM)"
    )
    summary_df = pd.DataFrame(rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Descargar resumen CSV",
        summary_df.to_csv(index=False).encode("utf-8"),
        f"cc_stock_repair_resumen_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        "text/csv",
    )

    st.divider()

    # ── Detalle de un ticker ───────────────────────────────────────────
    st.markdown("#### 🔍 Detalle")
    selected = st.selectbox("Ticker", options=[r["ticker"] for r in ok], key="cc_detail_ticker")
    res = next(r for r in ok if r["ticker"] == selected)
    df, rec, status, x_need = picks[selected]

    if res["expiration"] != res["requested_exit"]:
        st.info(
            f"ℹ️ No existe el vencimiento {res['requested_exit'].strftime('%d/%m/%Y')}: "
            f"se ha usado el más cercano, **{res['expiration'].strftime('%d/%m/%Y')}**."
        )
    if res["entry_date"] > date.today():
        st.caption("ℹ️ La fecha de entrada es futura: el DTE se cuenta desde ella, pero prima, IV y "
                   "delta son los de hoy.")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("💲 Precio", f"${res['price']:.2f}")
    m2.metric("⏱️ DTE / días mercado", f"{res['dte_cal']} / {res['n_td']}",
              help=f"Vencimiento {res['expiration'].strftime('%d/%m/%Y')}.")
    m3.metric("📈 IV ATM", f"{res['iv_atm']:.1f}%")
    m4.metric(f"📉 Vol. real ({res['vol_source']})", f"{res['rv_used']:.1f}%",
              help=(f"ATR(14) = ${res['atr']:.2f} ({res['atr_pct']:.2f}% del precio). "
                    f"Vol. real vía ATR: {res['rv_atr']:.1f}% · vía cierres 20d: "
                    f"{res['rv20']:.1f}%" if res["rv20"] else
                    f"ATR(14) = ${res['atr']:.2f} ({res['atr_pct']:.2f}% del precio)."))
    m5.metric("⚖️ ρ (IV / real)", f"{res['rho']:.2f}",
              help="Movimiento que cobra la IV dividido por el movimiento real (ATR) en el horizonte "
                   "de la operación (ya corrige 365 días naturales vs 252 de mercado). Cuanto más "
                   "cerca de 1, menos margen hay para bajar el ratio alejándote.")

    n1, n2, n3 = st.columns(3)
    n1.metric("↔️ Movimiento esperado (EM)", f"±${res['em']:.2f}",
              help=f"Media de IV (±{res['em_iv']:.2f}) y volatilidad real (±{res['em_real']:.2f}).")
    n2.metric("📏 Distancia necesaria",
              f"{x_need:.2f} EM" if x_need is not None else "no alcanzable",
              help=f"Distancia al ATM a la que el modelo baja el ratio de {ratio_actual:.0f}% a "
                   f"{target:.0f}% (reducción relativa {target / ratio_actual:.2f}).")
    n3.metric("🧢 Tope / mínimo", f"{cap_em:.2f} / {min_em:.2f} EM")

    st.markdown("##### 🎯 Strike recomendado")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("🎯 Strike", f"${rec['Strike']:.2f}",
              delta=f"{rec['Dist_EM']:.2f} EM · {rec['OTM_%']:+.1f}% vs precio", delta_color="off")
    k2.metric("📐 Delta", f"{rec['Delta']:.3f}" if pd.notna(rec["Delta"]) else "n/d",
              help=f"Fuente: {rec['Δ_fuente']}.")
    k3.metric("💵 Prima", f"${rec['Prima_Mid']:.2f}",
              delta=f"{rec['Prima_vs_ATM_%']:.0f}% de la del ATM", delta_color="off")
    k4.metric("⚖️ Ratio ajustado", f"{rec['Ratio_aj_%']:.1f}%",
              delta=f"objetivo {target:.0f}%", delta_color="off")
    k5.metric("💰 Neto ajustado", f"${rec['Neto_aj']:.2f}",
              help="Prima × (1 − ratio ajustado), por acción.")

    improvement = ratio_actual - float(rec["Ratio_aj_%"])
    if status == "ok":
        if x_need is not None and x_need <= 0:
            st.success(
                f"✅ Tu objetivo ya se cumple en el ATM. Call **{rec['Strike']:.2f}** "
                f"({res['expiration'].strftime('%d/%m/%Y')}, DTE {res['dte_cal']}): prima "
                f"${rec['Prima_Mid']:.2f}, ratio ajustado {rec['Ratio_aj_%']:.1f}%."
            )
        else:
            need_txt = f"≈ {x_need:.2f} EM" if x_need is not None else "algo más que"
            st.success(
                f"✅ Para bajar tu ratio en ATM de {ratio_actual:.0f}% a {target:.0f}% hace falta "
                f"alejarse {need_txt}. Vender la **call {rec['Strike']:.2f}** "
                f"({res['expiration'].strftime('%d/%m/%Y')}, DTE {res['dte_cal']}): cobras "
                f"${rec['Prima_Mid']:.2f} por acción (${rec['Prima_Mid'] * 100:.0f} por contrato, "
                f"{rec['Prima_vs_ATM_%']:.0f}% de la prima del ATM) y el ratio ajustado queda en "
                f"{rec['Ratio_aj_%']:.1f}% (≤ {target:.0f}%)."
            )
    elif status == "tope":
        need_txt = (f"haría falta alejarse {x_need:.2f} EM" if x_need is not None
                    else "el modelo no lo alcanza dentro de rango")
        st.warning(
            f"⚠️ Este ciclo no llega al {target:.0f}% dentro del tope de {cap_em:.2f} EM ({need_txt}). "
            f"Se usa el strike más cercano al tope, **call {rec['Strike']:.2f}**: ratio ajustado "
            f"{rec['Ratio_aj_%']:.1f}% (mejora de {improvement:.1f} puntos sobre tu {ratio_actual:.0f}%) "
            f"con el {rec['Prima_vs_ATM_%']:.0f}% de la prima del ATM. Si la mejora es pequeña frente a "
            f"la prima que cedes, este ciclo no compensa alejarse."
        )
    else:
        st.warning(
            f"⚠️ La IV no cobra más que lo que la acción se mueve (ρ = {res['rho']:.2f}): según el "
            f"modelo alejarte del ATM no mejora el ratio. Se muestra el strike más cercano al ATM "
            f"permitido (mínimo {min_em:.2f} EM), **call {rec['Strike']:.2f}**, con ratio ajustado "
            f"{rec['Ratio_aj_%']:.1f}%."
        )

    cb = res.get("cost_basis")
    if cb:
        cbm1, cbm2 = st.columns(2)
        cbm1.metric("📐 Coste base", f"${cb:.2f}")
        cbm2.metric("📉 Prima / CB", f"{rec['Prima_Mid'] / cb * 100:.2f}%",
                    help="Cuánto baja tu coste base este ciclo si te quedas la prima completa.")
        if rec["Strike"] < cb:
            st.warning("⚠️ El strike recomendado está por debajo de tu coste base: si te ejercieran, "
                       "realizarías pérdida.")

    tab1, tab2, tab3 = st.tabs(["📋 Tabla de strikes", "📊 Prima vs ratio", "📈 Precio"])

    with tab1:
        cols = [
            "Strike", "OTM_%", "Dist_EM", "Delta", "Δ_fuente", "Prima_Mid", "Bid", "Ask", "Prima_%",
            "Prima_vs_ATM_%", "Prima_anual_%", "IV_%", "Ratio_aj_%", "Neto_aj",
            "OI", "Volumen", "Spread_%",
        ]
        if "Prima/CB_%" in df.columns:
            cols += ["Prima/CB_%", "Bajo_CB"]
        view = df[cols].copy()
        view.insert(0, "Rec.", np.where(view["Strike"] == float(rec["Strike"]), "⭐", ""))

        def color_ratio(val):
            try:
                if pd.isna(val):
                    return ""
                if val <= target:
                    return "background-color:#1e3a1e; color:#6daa45"
                if val <= ratio_actual:
                    return "background-color:#3a3a1e; color:#e8af34"
                return "background-color:#3a1e1e; color:#dd6974"
            except Exception:
                return ""

        st.dataframe(view.style.map(color_ratio, subset=["Ratio_aj_%"]),
                     use_container_width=True, height=480, hide_index=True)
        st.caption(
            "Ratio_aj_% = tu ratio actual × reducción relativa que estima el modelo al alejarte del ATM. "
            "Verde: ≤ objetivo · ámbar: entre objetivo y tu ratio actual · rojo: peor que el ATM. "
            "Neto_aj = prima × (1 − ratio ajustado). Dist_EM = (strike − precio) / EM. "
            "Prima_vs_ATM_% compara el valor extrínseco con el del strike más cercano al precio."
        )
        st.download_button(
            f"⬇️ Descargar tabla de {selected} (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            f"cc_stock_repair_{selected}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
        )

    with tab2:
        st.plotly_chart(plot_strike_curve(df, rec, target, ratio_actual, selected, res, cap_em, min_em),
                        use_container_width=True)
        st.caption(
            "Al alejarte del ATM el ratio ajustado baja (más estable) pero la prima y el neto también. "
            "La estrella marca el strike recomendado; la línea roja vertical, el tope de distancia."
        )

    with tab3:
        days = st.select_slider("Días visibles", options=[30, 60, 90, 120, 180, 250], value=90, key="cc_days")
        fig = plot_price(res, rec, cap_em, days=days)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Banda azul: precio ± movimiento esperado (±{res['em']:.2f}). "
                f"Línea roja punteada: strike a la distancia tope ({cap_em:.2f} EM)."
            )
        else:
            st.warning("No se pudo cargar el gráfico.")

    with st.expander("ℹ️ Cómo se calcula"):
        st.markdown(
            "- Tú das el **ratio actual en ATM** y el **objetivo**: eso fija la reducción relativa "
            "necesaria (objetivo / actual).\n"
            "- El modelo estima **cuánto mejora el ratio al alejarte del ATM**: compara el coste "
            "esperado de recompra con la volatilidad real frente a la prima que cobra la IV, strike a strike.\n"
            "- **ρ** = movimiento que cobra la IV / movimiento real. Con ρ cercano a 1 no hay forma "
            "de bajar el ratio sin perder casi toda la prima; con ρ alto es barato.\n"
            "- **EM** = precio × media del movimiento de la IV y el de la volatilidad real. "
            "El **tope** y el **mínimo** se miden en EM.\n"
            "- **Recomendación**: el strike más cercano al ATM (≥ mínimo) que alcanza el objetivo sin pasar "
            "del tope; si no llega, el strike más cercano al tope, con aviso.\n"
            "- Es una estimación de la forma de la mejora: contrástala con tu Excel y ajusta."
        )

# ======================================================================
# 7. INTERFAZ STREAMLIT
# ======================================================================

def reset_everything():
    for fn in (get_daily_data, get_option_expirations):
        try:
            fn.clear()
        except Exception:
            pass
    for key in ("cc_results", "cc_ts"):
        st.session_state.pop(key, None)

def main():
    st.set_page_config(page_title="CC Stock Repair", page_icon="🛠️", layout="wide")

    if not check_password():
        st.stop()

    st.title("🛠️ CC Stock Repair")
    st.markdown(
        "**Covered calls para bajar el coste base: strike y delta alejándote del ATM lo justo para "
        "llevar tu ratio actual hasta el objetivo · prima de Alpaca · IV y ATR**"
    )
    st.caption(
        "⚙️ Opciones y precio vía Alpaca · ATR / volatilidad real vía yfinance · "
        "ratio actual, objetivo, tope y mínimo son tuyos: cámbialos cuando quieras."
    )

    col_title, col_reset = st.columns([5, 1])
    with col_reset:
        if st.button("🧹 Resetear", use_container_width=True,
                     help="Vacía el caché de históricos y vencimientos y borra los resultados."):
            reset_everything()
            st.success("Caché y resultados borrados.")
            st.rerun()

    st.divider()

    # ── Parámetros ─────────────────────────────────────────────────────
    st.markdown("### ⚙️ Parámetros")
    c1, c2, c3 = st.columns([1.5, 1.2, 1.3])

    with c1:
        tickers_raw = st.text_input(
            "🏷️ Ticker(s)", value="", placeholder="SOFI, HOOD, F",
            help="Uno o varios, separados por coma o espacio.",
        )
        cost_basis_raw = st.text_input(
            "📐 Coste base (opcional)", value="", placeholder="SOFI=14.2, HOOD=30",
            help="Formato TICKER=precio, separados por comas, con punto decimal. Si lo das, la tabla "
                 "muestra Prima/CB % y avisa de strikes por debajo de tu coste base.",
        )
        vol_source = st.radio(
            "📉 Volatilidad real basada en",
            options=["ATR", "Cierres 20d"], index=0, horizontal=True,
            help="ATR: ATR(14) convertido a volatilidad anual. Cierres 20d: desviación típica de los "
                 "retornos de los últimos 20 cierres. Las dos se muestran siempre en el detalle.",
        )

    with c2:
        entry_date = st.date_input(
            "📅 Fecha de entrada", value=date.today(),
            min_value=date.today(), max_value=date.today() + timedelta(days=179),
            format="DD/MM/YYYY",
            help="Día en el que abrirías la covered call.",
        )
        exit_date = st.date_input(
            "🏁 Fecha de salida (vencimiento)",
            value=entry_date + timedelta(days=15),
            min_value=entry_date + timedelta(days=1), max_value=entry_date + timedelta(days=180),
            format="DD/MM/YYYY",
            help="Vencimiento objetivo. DTE = días naturales entre entrada y salida.",
        )
        use_nearest_exp = st.checkbox(
            "Usar el vencimiento más cercano si no existe el exacto", value=True,
            help="Si está desactivado y no existe ese vencimiento, el ticker se descarta y se te "
                 "indican los más cercanos.",
        )
        dte_preview = (exit_date - entry_date).days
        n_td_preview = max(int(np.busday_count(entry_date, exit_date)), 1)
        st.caption(
            f"📆 Entrada **{entry_date.strftime('%d/%m/%Y')}** → salida "
            f"**{exit_date.strftime('%d/%m/%Y')}** · DTE **{dte_preview}** días "
            f"({n_td_preview} de mercado)."
        )

    with c3:
        r1, r2 = st.columns(2)
        with r1:
            ratio_actual = st.number_input(
                "📌 Ratio actual en ATM (%)", min_value=5.0, max_value=300.0,
                value=DEFAULT_RATIO_ACTUAL, step=1.0,
                help="Tu ratio ajustes/primas medido en el Excel con calls ATM (hoy 61%). Es el punto "
                     "de partida: el modelo solo estima cuánto mejora al alejarte.",
            )
        with r2:
            target_ratio = st.number_input(
                "🎯 Ratio objetivo (%)", min_value=5.0, max_value=300.0,
                value=DEFAULT_TARGET_RATIO, step=1.0,
                help="Ratio al que quieres bajar. Reducción relativa necesaria = objetivo / actual.",
            )
        d1, d2 = st.columns(2)
        with d1:
            cap_em = st.number_input(
                "🧢 Tope de distancia (EM)", min_value=0.1, max_value=3.0,
                value=DEFAULT_CAP_EM, step=0.05,
                help="Nunca te alejas más que el strike más cercano a esta distancia, medida en "
                     "movimientos esperados. 0,75 EM suele dar delta ≈ 0,2-0,25.",
            )
        with d2:
            min_em = st.number_input(
                "📍 Distancia mínima (EM)", min_value=0.0, max_value=3.0,
                value=DEFAULT_MIN_EM, step=0.05,
                help="Distancia mínima al ATM. Con 0 se permite el strike más cercano al precio; "
                     "súbela si no quieres strikes casi ATM.",
            )
        if min_em > cap_em:
            st.warning("⚠️ La distancia mínima es mayor que el tope: se intercambian.")
            min_em, cap_em = cap_em, min_em

    with st.expander("🔧 Ajustes avanzados"):
        a1, a2, a3 = st.columns(3)
        with a1:
            atr_period = st.number_input("Periodo del ATR", min_value=5, max_value=60,
                                         value=DEFAULT_ATR_PERIOD, step=1)
            atr_to_sigma = st.number_input(
                "Factor ATR → σ diaria", min_value=1.0, max_value=2.5,
                value=DEFAULT_ATR_TO_SIGMA, step=0.05,
                help="σ diaria ≈ ATR% / factor. Teóricamente el rango diario es ~1.6 veces σ; con gaps "
                     "el ATR sube algo más. Si la vol. por ATR te sale muy distinta de la de cierres 20d "
                     "de forma sistemática, ajusta este factor.",
            )
            em_iv_weight = st.number_input(
                "Peso de la IV en el EM (0-1)", min_value=0.0, max_value=1.0,
                value=DEFAULT_EM_IV_WEIGHT, step=0.1,
                help="EM = precio × (peso × movimiento IV + (1 − peso) × movimiento real). "
                     "0,5 = media de ambos.",
            )
        with a2:
            max_otm_pct = st.number_input("Rango máximo de strikes (% sobre el precio)",
                                          min_value=2.0, max_value=100.0,
                                          value=DEFAULT_MAX_OTM_PCT, step=1.0)
            use_spread_filter = st.checkbox("Exigir spread ≤ X% de la prima", value=True)
            spread_max_pct = st.number_input("Spread máximo (%)", min_value=10, max_value=300,
                                             value=DEFAULT_SPREAD_MAX_PCT, step=10,
                                             disabled=not use_spread_filter)
        with a3:
            use_oi_filter = st.checkbox(
                "Exigir OI mínimo", value=True,
                help="Si te descarta strikes con bid/ask reales porque el OI de Alpaca viene en 0, "
                     "desactívalo.",
            )
            min_oi = st.number_input("OI mínimo", min_value=1, max_value=100000,
                                     value=DEFAULT_MIN_OI, step=10, disabled=not use_oi_filter)

    params = {
        "atr_period": int(atr_period),
        "atr_to_sigma": float(atr_to_sigma),
        "em_iv_weight": float(em_iv_weight),
        "max_otm_pct": float(max_otm_pct),
        "vol_source": vol_source,
        "use_nearest_exp": use_nearest_exp,
        "use_spread_filter": use_spread_filter,
        "spread_max_pct": float(spread_max_pct),
        "use_oi_filter": use_oi_filter,
        "min_oi": int(min_oi),
    }

    st.divider()

    # ── Cálculo ────────────────────────────────────────────────────────
    tickers = parse_tickers(tickers_raw)
    cost_basis_map = parse_cost_basis(cost_basis_raw)

    calc_btn = st.button("🛠️ CALCULAR STRIKES", type="primary", use_container_width=True,
                         disabled=(len(tickers) == 0) or (exit_date <= entry_date))
    if not tickers:
        st.caption("ℹ️ Escribe al menos un ticker.")

    if calc_btn:
        results = []
        progress = st.progress(0)
        status = st.empty()
        for i, t in enumerate(tickers, start=1):
            status.text(f"🔍 Analizando {t} ({i}/{len(tickers)})...")
            try:
                results.append(analyze_ticker(t, entry_date, exit_date, params, cost_basis_map.get(t)))
            except Exception as e:
                logger.warning(f"[analyze] {t}: {e}")
                results.append({"ticker": t, "error": f"Error inesperado: {e}"})
            progress.progress(i / len(tickers))
        progress.empty()
        status.empty()
        st.session_state["cc_results"] = results
        st.session_state["cc_ts"] = datetime.now()

    st.divider()

    render_results(float(ratio_actual), float(target_ratio), float(cap_em), float(min_em))

if __name__ == "__main__":
    main()
