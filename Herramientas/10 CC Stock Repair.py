"""
CC Stock Repair
===============
Objetivo: analizar covered calls normalizando la curva de ratios desde el
ratio ATM actual que introduces. Se muestran tres alternativas: principal,
delta cercana a 0.20 y compromiso entre prima y ratio objetivo.

QUÉ ES EL RATIO
    ratio = coste de recompra / prima cobrada
El ratio actual de tu Excel (por ejemplo, 61 %) se introduce como ancla ATM.
A partir de ese valor, el modelo estima cómo cambia el ratio al subir cada
strike. El ratio normalizado no es un cálculo contable del Excel, sino una
curva estimada calibrada al valor actual que introduces:

  - PRIMA: mid (bid+ask)/2 de la call en Alpaca. Ya lleva dentro la IV que
    cobra el mercado.
  - COSTE ESPERADO DE RECOMPRA: lo que valdría la call al vencimiento si la
    acción se mueve con su VOLATILIDAD REAL. Se calcula como el valor
    Black-Scholes de esa call usando la volatilidad real (sin deriva) y los
    días de mercado entre la fecha de entrada y la de salida.
  - RATIO ESTIMADO = coste esperado / prima.
  - NETO ESPERADO  = prima - coste esperado (por acción).

VOLATILIDAD REAL (dos fuentes, eliges cuál usar en el cálculo):
  - ATR: ATR(14) de Wilder / precio, convertido a desviación típica diaria
    dividiendo por un factor (ATR_TO_SIGMA, 1.5 por defecto, ajustable) y
    anualizado con raíz de 252.
  - Cierres 20d: desviación típica de los retornos logarítmicos de los
    últimos 20 cierres, anualizada.
  Las dos se muestran siempre para que puedas contrastarlas.

NORMALIZACIÓN Y RECOMENDACIONES
    El ratio actual introducido se asigna al strike cotizado más cercano al
    precio spot (ATM). Los ratios del resto de strikes se escalan en relación
    con el ratio bruto del modelo en ATM. Se muestran tres alternativas:
      1) Principal: strike más cercano al ATM que alcanza el objetivo; si
         ninguno lo alcanza, fallback a delta cercana a 0.20.
      2) Conservadora: delta más cercana a 0.20.
      3) Compromiso: mayor prima dentro del margen configurable sobre objetivo.

FECHAS
    Fecha de entrada y de salida (vencimiento). DTE = días naturales entre
    ambas; los días de mercado (lunes-viernes, sin festivos) se usan para la
    volatilidad. Si no existe el vencimiento exacto, puedes usar el más
    cercano disponible (casilla activada por defecto; la herramienta siempre
    indica qué vencimiento ha usado).
    Aviso: la prima, la IV y el delta de la cadena son los de HOY. Si la
    fecha de entrada es futura, el DTE se cuenta desde esa fecha pero los
    precios de las opciones siguen siendo los actuales.

DELTA
    Se toma de Alpaca si la cadena lo trae; si no, se calcula con
    Black-Scholes usando la IV de cada strike. La columna Δ_fuente indica
    cuál se usó en cada fila.

DATOS
    Opciones y precio en vivo: Alpaca (utils/utils_alpaca.py).
    Histórico diario para ATR / volatilidad real: yfinance.

LÍMITES DEL MODELO
    Es una estimación sin deriva con la volatilidad reciente: las colas
    gordas, los gaps y las tendencias hacen que tu ratio real difiera.
    Compara de vez en cuando con tu Excel y ajusta el ratio objetivo.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, date
from statistics import NormalDist
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

_N = NormalDist()

RISK_FREE_RATE = 0.045        # solo para el delta Black-Scholes de reserva
TRADING_DAYS = 252
DEFAULT_TARGET_RATIO = 50.0   # % — ratio objetivo por defecto
DEFAULT_CURRENT_RATIO = 61.0
DEFAULT_ATR_PERIOD = 14
DEFAULT_ATR_TO_SIGMA = 1.5    # ATR ≈ 1.5 × desviación típica diaria (ajustable)
DEFAULT_MAX_OTM_PCT = 25.0    # strikes hasta +25 % sobre el precio
ATM_LOOKBACK_PCT = 2.0        # se incluyen strikes hasta 2 % por debajo del precio (referencia ATM)

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

def _safe_int(x, default=0):
    try:
        v = pd.to_numeric(x, errors="coerce")
        if v is None or pd.isna(v):
            return default
        return int(v)
    except Exception:
        return default

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


def parse_ratio_input(raw, tickers, default=61.0):
    """
    Admite un ratio único para todos los tickers ("61") o valores por ticker
    ("SOFI=61, HOOD=58"). Devuelve {ticker: ratio_pct}.
    """
    text = str(raw or "").strip()
    if not text:
        return {t: float(default) for t in tickers}

    if "=" not in text:
        try:
            value = float(text.replace("%", "").strip())
            if 0 < value <= 300:
                return {t: value for t in tickers}
        except Exception:
            pass
        return {}

    out = {}
    for part in text.replace(";", ",").split(","):
        if "=" not in part:
            continue
        ticker, value = part.split("=", 1)
        ticker = _clean_ticker(ticker)
        try:
            ratio = float(value.replace("%", "").strip())
        except Exception:
            continue
        if ticker and 0 < ratio <= 300:
            out[ticker] = ratio
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
# 2. BLACK-SCHOLES (precio y delta de una call)
# ======================================================================

def bs_call_price(S, K, T, sigma, r=0.0):
    """Valor Black-Scholes de una call europea. Con r=0 equivale a esperar
    que la acción no tenga deriva: E[max(S_T - K, 0)]."""
    try:
        if S <= 0 or K <= 0:
            return 0.0
        if T <= 0 or sigma <= 0:
            return max(S - K, 0.0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return float(S * _N.cdf(d1) - K * np.exp(-r * T) * _N.cdf(d2))
    except Exception:
        return max(S - K, 0.0)

def bs_call_delta(S, K, T_years, sigma, r=RISK_FREE_RATE):
    """Delta de una call europea (Black-Scholes): N(d1), entre 0 y 1."""
    try:
        if T_years <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return None
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T_years) / (sigma * np.sqrt(T_years))
        return round(float(_N.cdf(d1)), 3)
    except Exception:
        return None

# ======================================================================
# 3. TABLA DE STRIKES: prima, delta, coste esperado, ratio y neto
# ======================================================================


def build_strike_table(calls, S, dte_cal, n_td, sigma_real, p,
                       cost_basis=None, current_ratio=61.0):
    """
    Calcula ratios brutos por strike y los NORMALIZA desde el ATM:
        ratio_normalizado(K) = ratio_modelo(K) / ratio_modelo(ATM) * ratio_actual
    Así, el strike ATM queda anclado exactamente al ratio actual introducido
    por el usuario. La selección posterior se hace solo entre strikes ATM/OTM.
    """
    try:
        c = calls.copy().reset_index(drop=True)
        c["strike"] = pd.to_numeric(c["strike"], errors="coerce")
        c = c.dropna(subset=["strike"])

        for col in ("bid", "ask"):
            c[col] = pd.to_numeric(c[col], errors="coerce").fillna(0)

        c["mid"] = (c["bid"] + c["ask"]) / 2
        c = c[(c["bid"] > 0) & (c["ask"] > 0) & (c["mid"] > 0)].reset_index(drop=True)
        if c.empty:
            return None, None, None

        if "impliedVolatility" in c.columns:
            c["iv"] = pd.to_numeric(c["impliedVolatility"], errors="coerce")
        else:
            c["iv"] = np.nan

        with_iv = c[c["iv"] > 0]
        if with_iv.empty:
            iv_atm = None
        else:
            idx_iv = (with_iv["strike"] - S).abs().idxmin()
            iv_atm = float(with_iv.loc[idx_iv, "iv"]) * 100

        c["oi"] = (
            pd.to_numeric(c["openInterest"], errors="coerce").fillna(0)
            if "openInterest" in c.columns else 0
        )
        c["vol"] = (
            pd.to_numeric(c["volume"], errors="coerce").fillna(0)
            if "volume" in c.columns else 0
        )

        if "delta" in c.columns:
            d_alp = pd.to_numeric(c["delta"], errors="coerce")
            d_alp = d_alp.where((d_alp > 0) & (d_alp <= 1))
        else:
            d_alp = pd.Series(np.nan, index=c.index)

        T_delta = max(dte_cal, 1) / 365.0
        d_bs = pd.Series(
            [
                _nz(bs_call_delta(S, float(k), T_delta, float(v)))
                if pd.notna(v) and v > 0 else np.nan
                for k, v in zip(c["strike"], c["iv"])
            ],
            index=c.index,
            dtype=float,
        )
        c["delta"] = d_alp.where(d_alp.notna(), d_bs)
        c["delta_src"] = np.where(
            d_alp.notna(), "Alpaca",
            np.where(d_bs.notna(), "BS", "—")
        )

        T_td = max(n_td, 1) / TRADING_DAYS
        c["coste"] = [
            bs_call_price(S, float(k), T_td, sigma_real)
            for k in c["strike"]
        ]
        c["ratio_raw"] = c["coste"] / c["mid"] * 100
        c["neto"] = c["mid"] - c["coste"]
        c["spread_pct"] = (c["ask"] - c["bid"]) / c["mid"] * 100

        # Ancla: strike cotizado más cercano al precio spot actual.
        atm_idx = (c["strike"] - S).abs().idxmin()
        atm_raw_ratio = float(c.loc[atm_idx, "ratio_raw"])
        if not np.isfinite(atm_raw_ratio) or atm_raw_ratio <= 0:
            return None, iv_atm, None

        c["ratio"] = (c["ratio_raw"] / atm_raw_ratio) * float(current_ratio)

        # Limitar el universo de strikes. Conservamos algo de ITM en la tabla
        # para contexto, pero las recomendaciones solo usan ATM/OTM.
        lo = S * (1 - ATM_LOOKBACK_PCT / 100)
        hi = S * (1 + p["max_otm_pct"] / 100)
        c = c[(c["strike"] >= lo) & (c["strike"] <= hi)].copy()

        if p["use_oi_filter"]:
            c = c[c["oi"] >= p["min_oi"]]
        if p["use_spread_filter"]:
            c = c[c["spread_pct"] <= p["spread_max_pct"]]
        c = c.reset_index(drop=True)
        if c.empty:
            return None, iv_atm, atm_raw_ratio

        out = pd.DataFrame({
            "Strike": c["strike"].round(2),
            "OTM_%": ((c["strike"] - S) / S * 100).round(2),
            "Delta": c["delta"].round(3),
            "Δ_fuente": c["delta_src"],
            "Prima_Mid": c["mid"].round(2),
            "Bid": c["bid"].round(2),
            "Ask": c["ask"].round(2),
            "Prima_%": (c["mid"] / S * 100).round(2),
            "Prima_anual_%": (c["mid"] / S * 100 * 365 / max(dte_cal, 1)).round(1),
            "IV_%": (c["iv"] * 100).round(1),
            "Recompra_esp": c["coste"].round(2),
            "Neto_esp": c["neto"].round(2),
            "Ratio_modelo_%": c["ratio_raw"].round(1),
            "Ratio_est_%": c["ratio"].round(1),
            "OI": c["oi"].astype(int),
            "Volumen": c["vol"].astype(int),
            "Spread_%": c["spread_pct"].round(1),
        })
        if cost_basis:
            out["Prima/CB_%"] = (c["mid"] / cost_basis * 100).round(2)
            out["Bajo_CB"] = np.where(c["strike"] < cost_basis, "⚠️", "")

        out = out.sort_values("Strike").reset_index(drop=True)
        return out, iv_atm, atm_raw_ratio
    except Exception as e:
        logger.warning(f"[strike_table] {e}")
        return None, None, None


def _otm_candidates(df):
    """Solo strikes ATM o superiores; ordenados desde el ATM hacia arriba."""
    x = df[pd.to_numeric(df["OTM_%"], errors="coerce") >= -0.01].copy()
    return x.sort_values("Strike").reset_index(drop=True)


def select_recommendations(df, target_ratio, compromise_buffer=10.0):
    """
    Devuelve tres alternativas:
      1) Principal: strike más cercano al ATM que cumple el ratio objetivo.
         Si ninguno cumple, fallback al delta más cercano a 0.20.
      2) Conservadora: delta más cercano a 0.20.
      3) Compromiso: mayor prima dentro de target + buffer, priorizando
         opciones que quedan ligeramente por encima del objetivo.
    """
    candidates = _otm_candidates(df)
    if candidates.empty:
        candidates = df.sort_values("Strike").copy()

    delta_rows = candidates[pd.to_numeric(candidates["Delta"], errors="coerce").notna()].copy()
    if not delta_rows.empty:
        delta_rows["_delta_gap"] = (delta_rows["Delta"] - 0.20).abs()
        conservative = delta_rows.sort_values(
            ["_delta_gap", "Strike"], ascending=[True, True]
        ).iloc[0].drop(labels=["_delta_gap"])
    else:
        conservative = candidates.sort_values("Strike", ascending=False).iloc[0]

    qualifying = candidates[candidates["Ratio_est_%"] <= target_ratio]
    if not qualifying.empty:
        principal = qualifying.sort_values("Strike").iloc[0]
        principal_meets = True
    else:
        principal = conservative
        principal_meets = False

    slightly_above = candidates[
        (candidates["Ratio_est_%"] > target_ratio)
        & (candidates["Ratio_est_%"] <= target_ratio + compromise_buffer)
    ]
    if not slightly_above.empty:
        # Dentro del margen, prima máxima; desempate por ratio más bajo.
        compromise = slightly_above.sort_values(
            ["Prima_Mid", "Ratio_est_%"], ascending=[False, True]
        ).iloc[0]
    else:
        # Si no hay ninguno en el margen, elige el ratio más cercano al objetivo.
        # En empate, prioriza la prima más alta.
        candidates = candidates.copy()
        candidates["_ratio_gap"] = (candidates["Ratio_est_%"] - target_ratio).abs()
        compromise = candidates.sort_values(
            ["_ratio_gap", "Prima_Mid"], ascending=[True, False]
        ).iloc[0].drop(labels=["_ratio_gap"])

    return {
        "principal": principal,
        "principal_meets": principal_meets,
        "conservative": conservative,
        "compromise": compromise,
    }

# 4. ANÁLISIS DE UN TICKER
# ======================================================================


def analyze_ticker(ticker, entry_date, exit_date, p, cost_basis=None, current_ratio=61.0):
    out = {"ticker": ticker, "error": None, "current_ratio": current_ratio}

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

    table, iv_atm, atm_raw_ratio = build_strike_table(
        calls, S, dte_cal, n_td, sigma_real, p, cost_basis, current_ratio
    )
    if table is None or table.empty:
        out["error"] = (
            "Ninguna call cotizada cumple el rango de strikes y los filtros de liquidez. "
            "Prueba a relajar spread/OI o ampliar el rango OTM."
        )
        return out

    em_real = S * sigma_real * np.sqrt(n_td / TRADING_DAYS)
    em_iv = S * (iv_atm / 100) * np.sqrt(dte_cal / 365.0) if iv_atm else None

    out.update({
        "price": round(S, 2),
        "expiration": exp,
        "requested_exit": exit_date,
        "entry_date": entry_date,
        "dte_cal": dte_cal,
        "n_td": n_td,
        "iv_atm": round(iv_atm, 1) if iv_atm else None,
        "rv_atr": vol["rv_atr"],
        "rv20": vol["rv20"],
        "rv_used": round(rv_used, 1),
        "vol_source": p["vol_source"],
        "iv_rv": round(iv_atm / rv_used, 2) if (iv_atm and rv_used > 0) else None,
        "atr": vol["atr"],
        "atr_pct": vol["atr_pct"],
        "em_real": round(em_real, 2),
        "em_iv": round(em_iv, 2) if em_iv else None,
        "table": table,
        "cost_basis": cost_basis,
        "daily": data,
        "current_ratio": float(current_ratio),
        "atm_raw_ratio": round(float(atm_raw_ratio), 2) if atm_raw_ratio else None,
    })
    return out

# ======================================================================
# 5. GRÁFICOS
# ======================================================================

def plot_strike_curve(df, rec, target, ticker):
    """Barras: prima. Línea verde: neto esperado ($). Línea ámbar: ratio
    estimado (eje derecho) frente al objetivo (línea roja). Estrella: strike
    recomendado."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=df["Strike"], y=df["Prima_Mid"], name="Prima mid ($)",
        marker_color="#5b9bd5", opacity=0.7,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df["Strike"], y=df["Neto_esp"], mode="lines+markers", name="Neto esperado ($)",
        line=dict(color="#6daa45", width=2),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df["Strike"], y=df["Ratio_est_%"], mode="lines+markers", name="Ratio estimado (%)",
        line=dict(color="#e8af34", width=2),
    ), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=[df["Strike"].min(), df["Strike"].max()], y=[target, target], mode="lines",
        name=f"Objetivo {target:.0f}%", line=dict(color="#dd6974", dash="dash", width=2),
    ), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=[rec["Strike"]], y=[rec["Ratio_est_%"]], mode="markers", name="Recomendado",
        marker=dict(symbol="star", size=17, color="#ffffff", line=dict(color="#e8af34", width=2)),
    ), secondary_y=True)
    fig.add_vline(x=float(rec["Strike"]), line=dict(color="#ffffff", width=1, dash="dot"))
    fig.update_xaxes(title_text="Strike")
    fig.update_yaxes(title_text="$ por acción", secondary_y=False)
    fig.update_yaxes(title_text="Ratio estimado (%)", secondary_y=True)
    fig.update_layout(
        title=f"{ticker} — prima, neto esperado y ratio estimado por strike",
        template="plotly_dark", height=470, hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
    )
    return fig

def plot_price(res, rec, days=90):
    """Velas + línea del precio, strike recomendado, coste base (si lo diste)
    y banda del movimiento esperado (precio ± movimiento con vol. real)."""
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
        em = res["em_real"]
        fig.add_hrect(y0=S - em, y1=S + em, fillcolor="rgba(91,155,213,0.10)", line_width=0)
        fig.add_hline(y=S, line=dict(color="#ffffff", width=1),
                      annotation_text=f"Precio {S:.2f}", annotation_position="left",
                      annotation_font=dict(size=10))
        fig.add_hline(y=float(rec["Strike"]), line=dict(color="#e8af34", width=2, dash="dot"),
                      annotation_text=f"Strike {rec['Strike']:.2f}", annotation_position="right",
                      annotation_font=dict(size=11, color="#e8af34"))
        y_vals = [float(recent["Low"].min()), float(recent["High"].max()),
                  S - em, S + em, float(rec["Strike"])]
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


def render_results(target, compromise_buffer=10.0):
    st.markdown("### 📊 Resultados")
    results = st.session_state.get("cc_results")
    if not results:
        st.info("👆 Escribe tus tickers, revisa las fechas y pulsa **CALCULAR STRIKES**.")
        return

    ts = st.session_state.get("cc_ts")
    if ts:
        st.caption(
            f"🕐 Último cálculo: {ts.strftime('%d/%m/%Y %H:%M:%S')} · "
            "puedes cambiar el ratio objetivo y las tres alternativas se actualizan sin descargar datos."
        )

    ok = [r for r in results if not r.get("error")]
    bad = [r for r in results if r.get("error")]
    for r in bad:
        st.warning(f"⚠️ **{r['ticker']}**: {r['error']}")
    if not ok:
        return

    recs = {}
    summary_rows = []
    for r in ok:
        choices = select_recommendations(r["table"], target, compromise_buffer)
        recs[r["ticker"]] = choices
        main = choices["principal"]
        summary_rows.append({
            "Ticker": r["ticker"],
            "Precio": r["price"],
            "Ratio ATM actual": r["current_ratio"],
            "Objetivo": target,
            "Vencimiento": r["expiration"].strftime("%d/%m/%Y"),
            "DTE": r["dte_cal"],
            "Strike principal": float(main["Strike"]),
            "Delta": main["Delta"],
            "Prima": float(main["Prima_Mid"]),
            "Ratio estimado": float(main["Ratio_est_%"]),
            "Estado": "Cumple objetivo" if choices["principal_meets"] else "Fallback ~0.20Δ",
        })

    st.markdown("#### 🎯 Resumen por ticker")
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Descargar resumen CSV",
        pd.DataFrame(summary_rows).to_csv(index=False).encode("utf-8"),
        f"cc_stock_repair_resumen_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        "text/csv",
    )
    st.divider()

    st.markdown("#### 🔍 Análisis detallado")
    selected = st.selectbox(
        "Ticker", options=[r["ticker"] for r in ok], key="cc_detail_ticker"
    )
    res = next(r for r in ok if r["ticker"] == selected)
    choices = recs[selected]
    main = choices["principal"]
    conservative = choices["conservative"]
    compromise = choices["compromise"]
    df = res["table"]

    if res["expiration"] != res["requested_exit"]:
        st.info(
            f"ℹ️ No existe el vencimiento {res['requested_exit'].strftime('%d/%m/%Y')}: "
            f"se ha usado el más cercano, **{res['expiration'].strftime('%d/%m/%Y')}**."
        )
    if res["entry_date"] > date.today():
        st.caption(
            "ℹ️ La fecha de entrada es futura: DTE se cuenta desde ella, "
            "pero prima, IV y delta son los de hoy."
        )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("💲 Precio", f"${res['price']:.2f}")
    m2.metric("Ratio ATM actual", f"{res['current_ratio']:.1f}%")
    m3.metric("🎯 Ratio objetivo", f"{target:.1f}%")
    m4.metric("📈 IV ATM", f"{res['iv_atm']:.1f}%" if res["iv_atm"] else "n/d")
    m5.metric("📉 Vol. real", f"{res['rv_used']:.1f}%")

    st.caption(
        f"ATM anclado al ratio que has introducido ({res['current_ratio']:.1f}%). "
        f"El modelo calcula cómo cambia ese ratio al subir el strike. "
        f"Ratio bruto del modelo en ATM antes de normalizar: {res['atm_raw_ratio']:.1f}%."
    )

    def show_choice(title, row, note, status=None):
        with st.container(border=True):
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Strike", f"${float(row['Strike']):.2f}",
                      delta=f"{float(row['OTM_%']):+.2f}% vs spot", delta_color="off")
            c2.metric("Delta", f"{float(row['Delta']):.3f}" if pd.notna(row["Delta"]) else "n/d")
            c3.metric("Prima mid", f"${float(row['Prima_Mid']):.2f}",
                      delta=f"${float(row['Prima_Mid']) * 100:.0f}/contrato", delta_color="off")
            c4.metric("Ratio normalizado", f"{float(row['Ratio_est_%']):.1f}%",
                      delta=f"{float(row['Ratio_est_%']) - target:+.1f} pp vs objetivo",
                      delta_color="inverse")
            c5.metric("Neto estimado", f"${float(row['Neto_esp']):.2f}")
            st.markdown(f"**{title}**")
            st.caption(note)
            if status:
                st.caption(status)

    st.markdown("##### 1. Principal — prima prioritaria cumpliendo el objetivo")
    if choices["principal_meets"]:
        show_choice(
            "Strike más cercano al ATM que cumple el ratio",
            main,
            "Selecciona el strike más bajo (más cercano al ATM) cuyo ratio normalizado sea "
            f"≤ {target:.1f}%. Entre los que cumplen, prioriza la prima más alta por cercanía al ATM.",
            "Objetivo alcanzado."
        )
    else:
        show_choice(
            "Fallback — delta más cercana a 0.20",
            main,
            "Ningún strike disponible alcanza el objetivo. Como alternativa, se selecciona "
            "la call OTM cuya delta está más cerca de 0.20; no se aplica 0.20 como filtro mínimo.",
            "El ratio puede quedar por encima del objetivo."
        )
        st.warning(
            f"Ninguna call ATM/OTM alcanza el {target:.1f}%. Se muestra el fallback cercano a 0.20 delta."
        )

    st.markdown("##### 2. Conservadora — delta cercana a 0.20")
    show_choice(
        "Alternativa de menor delta",
        conservative,
        "Busca la delta más próxima a 0.20 entre las calls ATM/OTM. Es una referencia de menor "
        "probabilidad de terminar ITM, no una garantía de conservar las acciones."
    )

    st.markdown(f"##### 3. Compromiso — más prima aceptando hasta +{compromise_buffer:.1f} puntos")
    show_choice(
        "Alternativa de compromiso",
        compromise,
        "Busca la prima más alta entre las opciones cuyo ratio queda por encima del objetivo, "
        f"pero no más de {compromise_buffer:.1f} puntos. Si no existe una opción en ese margen, "
        "repite la principal."
    )

    if res.get("cost_basis"):
        cb = float(res["cost_basis"])
        st.metric("📐 Coste base", f"${cb:.2f}")
        if float(main["Strike"]) < cb:
            st.warning(
                "El strike principal está por debajo de tu coste base: si te asignan, "
                "realizarías una pérdida respecto a ese coste."
            )

    tab1, tab2, tab3 = st.tabs(["📋 Tabla de strikes", "📊 Prima vs ratio", "📈 Precio"])
    with tab1:
        cols = [
            "Strike", "OTM_%", "Delta", "Δ_fuente", "Prima_Mid", "Bid", "Ask",
            "Prima_%", "Prima_anual_%", "IV_%", "Recompra_esp", "Neto_esp",
            "Ratio_modelo_%", "Ratio_est_%", "OI", "Volumen", "Spread_%"
        ]
        if "Prima/CB_%" in df.columns:
            cols += ["Prima/CB_%", "Bajo_CB"]
        view = df[cols].copy()
        view.insert(0, "Rec.", "")
        for label, row in [
            ("1 Principal", main), ("2 Delta 0.20", conservative),
            ("3 Compromiso", compromise)
        ]:
            view.loc[view["Strike"] == float(row["Strike"]), "Rec."] += label + " "
        def color_ratio(val):
            try:
                if pd.isna(val):
                    return ""
                if val <= target:
                    return "background-color:#1e3a1e; color:#6daa45"
                if val <= target + compromise_buffer:
                    return "background-color:#3a3a1e; color:#e8af34"
                return "background-color:#3a1e1e; color:#dd6974"
            except Exception:
                return ""
        st.dataframe(
            view.style.map(color_ratio, subset=["Ratio_est_%"]),
            use_container_width=True, height=480, hide_index=True
        )
        st.caption(
            "Ratio_modelo_% es el ratio bruto de Black-Scholes. Ratio_est_% es ese ratio "
            "normalizado para que el strike ATM coincida con tu ratio actual. "
            "Verde: objetivo alcanzado; ámbar: hasta el margen de compromiso; rojo: por encima."
        )
        st.download_button(
            f"⬇️ Descargar tabla de {selected} (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            f"cc_stock_repair_{selected}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
        )

    with tab2:
        st.plotly_chart(
            plot_strike_curve(df, main, target, selected),
            use_container_width=True
        )
        st.caption(
            "La curva de ratio usa el ratio normalizado desde el ATM actual. "
            "La línea horizontal marca tu objetivo."
        )

    with tab3:
        days = st.select_slider(
            "Días visibles", options=[30, 60, 90, 120, 180, 250],
            value=90, key="cc_days"
        )
        fig = plot_price(res, main, days=days)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            em_iv_txt = (
                f" · movimiento esperado por IV ±{res['em_iv']:.2f}"
                if res.get("em_iv") else ""
            )
            st.caption(
                f"Banda azul: precio ± movimiento esperado con volatilidad real "
                f"(±{res['em_real']:.2f}){em_iv_txt}."
            )
        else:
            st.warning("No se pudo cargar el gráfico.")

    with st.expander("ℹ️ Cómo se calcula la normalización"):
        st.markdown(
            "- **Ratio actual ATM**: el valor que introduces (por ejemplo, 61%). "
            "Se asigna al strike cotizado más cercano al precio actual.\n"
            "- **Ratio modelo**: coste esperado de recompra / prima mid, calculado "
            "con Black-Scholes y la volatilidad real seleccionada.\n"
            "- **Ratio normalizado por strike**: ratio modelo del strike dividido por "
            "el ratio modelo ATM y multiplicado por tu ratio actual. Por tanto, "
            "el ATM queda anclado exactamente al ratio que has introducido.\n"
            "- **Principal**: strike ATM/OTM más cercano al precio que alcanza el objetivo; "
            "si ninguno lo consigue, fallback a delta cercana a 0.20.\n"
            "- **Conservadora**: delta más cercana a 0.20.\n"
            "- **Compromiso**: mayor prima dentro del margen configurable por encima del objetivo.\n\n"
            "**Importante:** esta normalización adapta la curva del modelo a tu ratio histórico "
            "actual; no predice el ratio contable exacto después de una operación. El resultado "
            "es una estimación, no una garantía de recompra ni de conservar las acciones."
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
        "**Covered calls para bajar el coste base: strike y delta a partir de tu ratio objetivo · "
        "prima de Alpaca · volatilidad real por ATR**"
    )
    st.caption(
        "⚙️ Opciones y precio vía Alpaca · ATR / volatilidad real vía yfinance · "
        "ratio actual ATM normalizado hacia el objetivo."
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
    c1, c2, c3 = st.columns([1.6, 1.2, 1.2])

    with c1:
        tickers_raw = st.text_input(
            "🏷️ Ticker(s)", value="", placeholder="SOFI, HOOD, F",
            help="Uno o varios, separados por coma o espacio.",
        )
        current_ratio_raw = st.text_input(
            "📍 Ratio actual ATM (%)", value=str(DEFAULT_CURRENT_RATIO),
            help="Tu ratio actual de Excel, que se asigna al strike más cercano al precio de hoy. "
                 "Un valor (61) se aplica a todos los tickers; también puedes indicar SOFI=61, HOOD=58.",
        )
        target_ratio = st.number_input(
            "🎯 Ratio objetivo (%)", min_value=5.0, max_value=150.0,
            value=DEFAULT_TARGET_RATIO, step=1.0,
            help="Objetivo del ratio normalizado que quieres alcanzar al subir el strike desde el ATM.",
        )
        cost_basis_raw = st.text_input(
            "📐 Coste base (opcional)", value="", placeholder="SOFI=14.2, HOOD=30",
            help="Formato TICKER=precio, separados por comas, con punto decimal. Si lo das, la tabla "
                 "muestra Prima/CB % y avisa de strikes por debajo de tu coste base.",
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

    with c3:
        vol_source = st.radio(
            "📉 Volatilidad real basada en",
            options=["ATR", "Cierres 20d"], index=0, horizontal=True,
            help="ATR: ATR(14) convertido a volatilidad anual. Cierres 20d: desviación típica de los "
                 "retornos de los últimos 20 cierres. Las dos se muestran siempre en el detalle.",
        )
        dte_preview = (exit_date - entry_date).days
        n_td_preview = max(int(np.busday_count(entry_date, exit_date)), 1)
        st.caption(
            f"📆 Entrada **{entry_date.strftime('%d/%m/%Y')}** → salida "
            f"**{exit_date.strftime('%d/%m/%Y')}** · DTE **{dte_preview}** días "
            f"({n_td_preview} de mercado)."
        )

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
            max_otm_pct = st.number_input("Rango máximo de strikes (% sobre el precio)",
                                          min_value=2.0, max_value=100.0,
                                          value=DEFAULT_MAX_OTM_PCT, step=1.0)
        with a2:
            use_spread_filter = st.checkbox("Exigir spread ≤ X% de la prima", value=True)
            spread_max_pct = st.number_input("Spread máximo (%)", min_value=10, max_value=300,
                                             value=50, step=10, disabled=not use_spread_filter)
        with a3:
            use_oi_filter = st.checkbox(
                "Exigir OI mínimo", value=False,
                help="Desactivado por defecto: el OI de Alpaca a veces viene en 0 incluso con bid/ask reales.",
            )
            min_oi = st.number_input("OI mínimo", min_value=1, max_value=100000,
                                     value=50, step=10, disabled=not use_oi_filter)
            compromise_buffer = st.number_input(
                "Margen máximo de compromiso (puntos %)", min_value=1.0, max_value=30.0,
                value=10.0, step=1.0,
                help="La tercera alternativa puede quedar hasta este número de puntos por encima del objetivo."
            )

    params = {
        "atr_period": int(atr_period),
        "atr_to_sigma": float(atr_to_sigma),
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
    current_ratio_map = parse_ratio_input(current_ratio_raw, tickers)

    missing_ratios = [t for t in tickers if t not in current_ratio_map]
    calc_btn = st.button(
        "🛠️ CALCULAR STRIKES", type="primary", use_container_width=True,
        disabled=(len(tickers) == 0) or (exit_date <= entry_date) or bool(missing_ratios)
    )
    if not tickers:
        st.caption("ℹ️ Escribe al menos un ticker.")
    if missing_ratios:
        st.error(
            "Falta el ratio actual para: " + ", ".join(missing_ratios)
            + ". Introduce un valor único para todos o un valor por ticker, por ejemplo SOFI=61, HOOD=58."
        )

    if calc_btn:
        results = []
        progress = st.progress(0)
        status = st.empty()
        for i, t in enumerate(tickers, start=1):
            status.text(f"🔍 Analizando {t} ({i}/{len(tickers)})...")
            try:
                results.append(analyze_ticker(
                    t, entry_date, exit_date, params,
                    cost_basis_map.get(t), current_ratio_map[t]
                ))
            except Exception as e:
                logger.warning(f"[analyze] {t}: {e}")
                results.append({"ticker": t, "error": f"Error inesperado: {e}"})
            progress.progress(i / len(tickers))
        progress.empty()
        status.empty()
        st.session_state["cc_results"] = results
        st.session_state["cc_ts"] = datetime.now()

    st.divider()

    render_results(float(target_ratio), float(compromise_buffer))

if __name__ == "__main__":
    main()
