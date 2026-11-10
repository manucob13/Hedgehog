"""
Collar Screener (riesgo cero) — v2: DTE 30-60, solo crédito, margen IBKR aprox.
===============================================================================
Estructura (por 100 acciones, mismo vencimiento):

    + 100 acciones
    + 1 long put ATM (el strike más cercano al precio)
    - 1 short call OTM: el strike MÁS LEJANO cuya prima financia el put

VENCIMIENTOS: todos los que caen entre DTE desde (30) y DTE hasta (60).

OPCIÓN "SOLO CRÉDITO" (activada por defecto): el call debe ingresar más que lo
que cuesta el put (crédito neto >= 0,5 céntimos por acción). Si se desactiva,
se acepta un débito pequeño (tolerancia configurable).

FILTROS DE TENDENCIA (opcionales, Fase 1): precio > SMA200, SMA50 > SMA200,
momentum 12-1 meses, cercanía al máximo de 52 semanas y volatilidad máxima.

MARGEN (aproximado, cuenta margin IBKR, Reg T, acciones de EE. UU.):
  - Margen inicial  = 50% x valor de las acciones + importe ITM del call
                      (el call es OTM, así que el segundo término es 0)
  - Margen mantenim.= mínimo( 10% x strike put + importe OTM del put ,
                              25% x strike call )  x 100
  Fuente: tabla "US Options Margin - Collar" de IBKR. IBKR puede exigir más
  en acciones concretas (volatilidad, concentración) y optimiza el margen del
  conjunto de la cartera: es una estimación, no el margen real.

TABLAS: ranking por ganancia máxima % y ranking por tiempo (DTE ascendente).
Incluye el midprice del put y del call (por acción) para usarlo después en IBKR.
No hay llamada a IBKR en este código. Solo screener.

GRÁFICOS: velas diarias de hasta 6 meses con SMA30 y bandas de Bollinger.

AVISO: no hay filtro de dividendos. Un call corto ITM cerca de un ex-dividend
tiene riesgo de ejercicio anticipado.

FASES: Fase 1 (paralela, yfinance): precio y filtros. Fase 2 (secuencial,
Alpaca): vencimientos, cadenas y collar. Precio de opción = midprice.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import csv
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import warnings
import socket
import logging
import plotly.graph_objects as go

from utils.utils import check_password
from utils.tickers import create_tickers_universe
from utils.utils_alpaca import (
    get_option_expirations,
    get_option_chain,
    get_live_price,
)

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("collar_screener")

SOCKET_TIMEOUT = 15
socket.setdefaulttimeout(SOCKET_TIMEOUT)

MAX_WORKERS = 3
ATM_MAX_DIST_PCT = 2.0     # el put ATM no puede estar a más de este % del precio
MIN_CREDIT = 0.005         # crédito mínimo por acción en modo "solo crédito"
CHART_DAYS = 183           # ~6 meses de gráfico

CBOE_WEEKLYS_CSV_URL = "https://www.cboe.com/available_weeklys/get_csv_download/"

CBOE_SECTION_MARKERS = {
    "available weeklys - exchange traded products (etfs and etns)": "etf",
    "available weeklys - equity": "equity",
}

CBOE_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36"
}

REASON_ORDER = [
    "ok", "no_price_data", "price_out_of_range", "technical_filter",
    "no_expirations", "no_expirations_in_window",
    "no_chain", "no_collar", "error",
]

REASON_LABELS = {
    "ok": "✅ Con al menos un collar válido",
    "no_price_data": "Sin precio disponible (Fase 1)",
    "price_out_of_range": "Precio fuera de rango (Fase 1)",
    "technical_filter": "Descartado por filtros de tendencia/momentum/volatilidad (Fase 1)",
    "no_expirations": "Sin vencimientos listados en Alpaca (Fase 2)",
    "no_expirations_in_window": "Sin vencimientos en el rango de DTE (Fase 2)",
    "no_chain": "Cadenas vacías/no disponibles (Fase 2)",
    "no_collar": "Ningún collar cumple crédito/ganancia/OI/spread (Fase 2)",
    "error": "Excepción no controlada",
}

_debug_lock = Lock()
_debug_samples = {}
_MAX_DEBUG_SAMPLES = 5


def _record_debug(reason, msg):
    with _debug_lock:
        bucket = _debug_samples.setdefault(reason, [])
        if len(bucket) < _MAX_DEBUG_SAMPLES:
            bucket.append(str(msg)[:200])


def _reset_debug():
    with _debug_lock:
        _debug_samples.clear()


def _clean_ticker(raw):
    if raw is None:
        return None
    t = str(raw).strip().replace("&amp;", "&").upper()
    if not t or t in ("-", "NAN", "N/A", ""):
        return None
    return t.replace(" ", "")


def _iv_or_none(x):
    try:
        v = float(x)
        if np.isnan(v) or v <= 0:
            return None
        return v
    except Exception:
        return None


# ======================================================================
# 0. UNIVERSO
# ======================================================================

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_cboe_weeklys_tickers():
    """Tickers de 'Available Weeklys' de Cboe (ETFs y acciones). Cache 24h.
    Devuelve un set (vacío si algo falla)."""
    try:
        resp = requests.get(CBOE_WEEKLYS_CSV_URL, headers=CBOE_REQUEST_HEADERS, timeout=20)
        resp.raise_for_status()
        text = resp.content.decode("utf-8-sig", errors="replace")
    except Exception as e:
        logger.warning(f"[cboe_weeklys] descarga falló: {e}")
        return set()

    tickers = set()
    current_section = None
    try:
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            low = line.lower()
            matched = None
            for marker, name in CBOE_SECTION_MARKERS.items():
                if low.startswith(marker):
                    matched = name
                    break
            if matched:
                current_section = matched
                continue

            if current_section not in ("etf", "equity"):
                continue

            try:
                row = next(csv.reader([line]))
            except Exception:
                continue
            if not row:
                continue

            ticker = _clean_ticker(row[0])
            if ticker:
                tickers.add(ticker)
    except Exception as e:
        logger.warning(f"[cboe_weeklys] parseo falló: {e}")
        return set()

    if len(tickers) < 50:
        logger.warning(f"[cboe_weeklys] solo {len(tickers)} tickers extraídos — se descarta")
        return set()

    return tickers


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_full_universe():
    """Universo: base fija + Russell 1000 + Cboe Weeklys nuevos (best-effort)."""
    df = create_tickers_universe(include_russell1000=True)
    if df is None or df.empty:
        df = pd.DataFrame(columns=["Ticker", "Type"])
    else:
        df = df.copy()

    df["Ticker"] = df["Ticker"].apply(_clean_ticker)
    df = df.dropna(subset=["Ticker"]).drop_duplicates("Ticker").reset_index(drop=True)

    existing = set(df["Ticker"])
    cboe_tickers = get_cboe_weeklys_tickers()
    cboe_new = sorted(t for t in cboe_tickers if t and t not in existing)

    if cboe_new:
        cboe_df = pd.DataFrame({"Ticker": cboe_new, "Type": "CboeWeeklys"})
        df = pd.concat([df, cboe_df], ignore_index=True)
        df = df.drop_duplicates("Ticker").reset_index(drop=True)

    n_r = int((df["Type"] == "Russell1000").sum()) if "Type" in df.columns else 0
    n_cboe_new = len(cboe_new)
    meta = {
        "r1000_ok": n_r > 0,
        "r1000_count": n_r,
        "cboe_ok": len(cboe_tickers) > 0,
        "cboe_total_found": len(cboe_tickers),
        "cboe_new_count": n_cboe_new,
        "extra_count": len(df) - n_r - n_cboe_new,
        "total_count": len(df),
    }
    return df[["Ticker"]].sort_values("Ticker").reset_index(drop=True), meta


def refresh_universe():
    get_full_universe.clear()
    return get_full_universe()


# ======================================================================
# 1. FASE 1 — precio y filtros de tendencia (paralela)
# ======================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def get_price_data(ticker):
    """Histórico diario (2 años) vía yfinance y los indicadores para los filtros
    opcionales. Devuelve un dict (valores None si no hay historial suficiente)
    o None si falla la descarga."""
    try:
        h = yf.Ticker(ticker).history(period="2y", interval="1d", auto_adjust=False)
        if h is None or h.empty:
            _record_debug("no_price_data", f"{ticker}: descarga vacía")
            return None
        close = h["Close"].dropna()
        if close.empty:
            _record_debug("no_price_data", f"{ticker}: sin cierres")
            return None

        n = len(close)
        price = float(close.iloc[-1])
        data = {"price": price, "sma50": None, "sma200": None,
                "mom_12_1": None, "dist_high_pct": None, "vol_pct": None}

        if n >= 50:
            data["sma50"] = float(close.iloc[-50:].mean())
        if n >= 200:
            data["sma200"] = float(close.iloc[-200:].mean())
        if n >= 253:
            # Momentum 12-1: rentabilidad de hace ~252 sesiones a hace ~21 sesiones
            data["mom_12_1"] = float(close.iloc[-22] / close.iloc[-253] - 1) * 100
        if n >= 240:
            high52 = float(close.iloc[-252:].max())
            data["dist_high_pct"] = (high52 - price) / high52 * 100
        if n >= 120:
            rets = np.log(close.iloc[-252:] / close.iloc[-252:].shift(1)).dropna()
            data["vol_pct"] = float(rets.std() * np.sqrt(252) * 100)
        return data
    except Exception as e:
        _record_debug("no_price_data", f"{ticker}: {e}")
        return None


def passes_technical(ticker, d, params):
    """Aplica los filtros técnicos activados. Devuelve (ok, motivo)."""
    price = d["price"]

    if params["use_trend"]:
        if d["sma200"] is None:
            return False, "historial insuficiente para SMA200"
        if price <= d["sma200"]:
            return False, f"precio {price:.2f} <= SMA200 {d['sma200']:.2f}"
    if params["use_sma_cross"]:
        if d["sma50"] is None or d["sma200"] is None:
            return False, "historial insuficiente para SMA50/SMA200"
        if d["sma50"] <= d["sma200"]:
            return False, f"SMA50 {d['sma50']:.2f} <= SMA200 {d['sma200']:.2f}"
    if params["use_momentum"]:
        if d["mom_12_1"] is None:
            return False, "historial insuficiente para momentum 12-1"
        if d["mom_12_1"] <= params["min_momentum"]:
            return False, f"momentum 12-1 = {d['mom_12_1']:.1f}%"
    if params["use_high52"]:
        if d["dist_high_pct"] is None:
            return False, "historial insuficiente para máximo 52s"
        if d["dist_high_pct"] > params["max_dist_high"]:
            return False, f"{d['dist_high_pct']:.1f}% por debajo del máximo 52s"
    if params["use_vol"]:
        if d["vol_pct"] is None:
            return False, "historial insuficiente para volatilidad"
        if d["vol_pct"] > params["max_vol"]:
            return False, f"volatilidad realizada {d['vol_pct']:.0f}%"
    return True, ""


def phase1_price_filter(ticker, params):
    try:
        data = get_price_data(ticker)
        if data is None:
            return None, "no_price_data"
        price = data["price"]
        if not (params["min_price"] <= price <= params["max_price"]):
            _record_debug("price_out_of_range", f"{ticker}: precio = {price}")
            return None, "price_out_of_range"

        ok, why = passes_technical(ticker, data, params)
        if not ok:
            _record_debug("technical_filter", f"{ticker}: {why}")
            return None, "technical_filter"

        return {"Ticker": ticker, "current_price": round(price, 2)}, "ok"
    except Exception as e:
        _record_debug("error", f"{ticker}: {e}")
        return None, "error"


# ======================================================================
# 2. COLLAR: put ATM + call más lejano que lo financia
# ======================================================================

def _prep_leg(df, apply_oi, min_oi, apply_spread, spread_max_pct):
    """Limpia una pata: bid>0, ask>0, mid, OI y spread (filtros internos)."""
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    d["bid"] = pd.to_numeric(d["bid"], errors="coerce").fillna(0)
    d["ask"] = pd.to_numeric(d["ask"], errors="coerce").fillna(0)
    d["mid"] = (d["bid"] + d["ask"]) / 2
    d = d[(d["bid"] > 0) & (d["ask"] > 0) & (d["mid"] > 0)]
    if d.empty:
        return pd.DataFrame()

    if apply_oi:
        d["oi"] = pd.to_numeric(
            d.get("openInterest", pd.Series(0, index=d.index)), errors="coerce"
        ).fillna(0)
        d = d[d["oi"] >= min_oi]
        if d.empty:
            return pd.DataFrame()

    if apply_spread:
        d = d[(d["ask"] - d["bid"]) <= d["mid"] * (spread_max_pct / 100)]
        if d.empty:
            return pd.DataFrame()

    d["iv"] = pd.to_numeric(d["impliedVolatility"], errors="coerce") \
        if "impliedVolatility" in d.columns else np.nan
    return d


def approx_margin(price, kp, kc, params):
    """Margen aproximado IBKR (Reg T, cuenta margin, EE. UU.) por 100 acciones.
    Inicial    = % inicial x valor acciones + importe ITM del call
    Mantenim.  = min(10% x K put + importe OTM del put, 25% x K call)"""
    itm_call = max(price - kc, 0.0)
    otm_put = max(price - kp, 0.0)
    initial = (params["margin_init_pct"] / 100.0 * price + itm_call) * 100
    maint = min(params["margin_maint_put_pct"] / 100.0 * kp + otm_put,
                params["margin_maint_call_pct"] / 100.0 * kc) * 100
    return round(initial, 2), round(maint, 2)


def find_collar(ticker, exp, calls, puts, price, params):
    """Devuelve el collar (dict) del vencimiento o None.
    Put = strike más cercano al precio. Call = strike más lejano (> precio)
    que financia el put (crédito si 'solo crédito', o débito dentro de la
    tolerancia). Se exige ganancia máx. >= ganancia mínima."""
    try:
        dte = (exp - date.today()).days
        if dte <= 0:
            return None

        p = _prep_leg(puts, params["use_oi_filter"], params["min_oi"],
                      params["use_spread_filter"], params["spread_max_pct"])
        c = _prep_leg(calls, params["use_oi_filter"], params["min_oi"],
                      params["use_spread_filter"], params["spread_max_pct"])
        if p.empty or c.empty:
            return None

        # Put ATM: el strike más cercano al precio
        p = p.assign(dist=(p["strike"] - price).abs())
        pr = p.sort_values("dist").iloc[0]
        if pr["dist"] / price * 100 > ATM_MAX_DIST_PCT:
            return None

        kp = float(pr["strike"])
        put_mid = float(pr["mid"])

        if params["only_credit"]:
            min_call_mid = put_mid + MIN_CREDIT - 1e-9
        else:
            min_call_mid = put_mid - price * params["debit_tol_pct"] / 100.0

        ok = c[(c["strike"] > price) & (c["mid"] >= min_call_mid)]
        if ok.empty:
            return None
        cr = ok.sort_values("strike", ascending=False).iloc[0]

        kc = float(cr["strike"])
        call_mid = float(cr["mid"])

        net_cost = put_mid - call_mid                 # + débito / - crédito
        max_gain = kc - price - net_cost
        if max_gain <= 0:
            return None
        gain_pct = max_gain / price * 100
        if gain_pct < params["min_gain"]:
            return None

        iv_put = _iv_or_none(pr["iv"])
        iv_call = _iv_or_none(cr["iv"])
        skew = round((iv_call - iv_put) * 100, 1) if (iv_put and iv_call) else None

        if net_cost < -0.0049:
            tipo = "Crédito"
        elif net_cost > 0.0049:
            tipo = "Débito"
        else:
            tipo = "Cero"

        margin_init, margin_maint = approx_margin(price, kp, kc, params)

        return {
            "Ticker": ticker,
            "Precio": round(price, 2),
            "Vencimiento": exp.isoformat(),
            "DTE": dte,
            "Strike_Put": kp,
            "Put_Mid": round(put_mid, 3),
            "Strike_Call": kc,
            "Call_Mid": round(call_mid, 3),
            "Tipo": tipo,
            "Coste_Neto_$": round(net_cost * 100, 2),
            "Coste_Neto_%": round(net_cost / price * 100, 2),
            "Ganancia_Max_$": round(max_gain * 100, 2),
            "Ganancia_Max_%": round(gain_pct, 2),
            "Call_Skew": skew,
            "Margen_Inicial_$": margin_init,
            "Margen_Mant_$": margin_maint,
            "_coste_acc": round(net_cost, 4),   # interno (gráfico)
        }
    except Exception as e:
        _record_debug("error", f"find_collar {ticker} {exp}: {e}")
        return None


# ======================================================================
# 3. FASE 2 — opciones (SECUENCIAL)
# ======================================================================

def phase2_collar_scan(survivor, params):
    ticker = survivor["Ticker"]
    fallback_price = survivor["current_price"]
    try:
        expirations = get_option_expirations(ticker)
        if not expirations:
            return None, "no_expirations"

        today = date.today()
        valid = sorted(
            e for e in expirations
            if params["min_dte"] <= (e - today).days <= params["max_dte"]
        )
        if not valid:
            return None, "no_expirations_in_window"

        price = get_live_price(ticker, fallback_price)

        rows = []
        chains_ok = 0
        for exp in valid:
            try:
                calls, puts = get_option_chain(ticker, exp)
            except Exception as e:
                _record_debug("no_chain", f"{ticker} {exp}: {e}")
                continue
            if calls is None or puts is None or calls.empty or puts.empty:
                continue
            chains_ok += 1
            row = find_collar(ticker, exp, calls, puts, price, params)
            if row:
                rows.append(row)

        if chains_ok == 0:
            return None, "no_chain"
        if not rows:
            return None, "no_collar"
        return rows, "ok"
    except Exception as e:
        _record_debug("error", f"{ticker}: {e}")
        return None, "error"


# ======================================================================
# 4. ORQUESTADOR
# ======================================================================

def run_screener(tickers, params, progress_bar, status_text):
    _reset_debug()
    funnel = {r: 0 for r in REASON_ORDER}
    total = len(tickers)

    status_text.text(f"🔍 Fase 1/2 — precio y filtros: 0/{total}")
    survivors = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(phase1_price_filter, t, params): t for t in tickers}
        done = 0
        for future in as_completed(futures):
            done += 1
            progress_bar.progress(done / total * 0.3)
            status_text.text(f"🔍 Fase 1/2 — precio y filtros: {done}/{total} · supervivientes: {len(survivors)}")
            r, reason = None, "error"
            try:
                r, reason = future.result(timeout=SOCKET_TIMEOUT + 10)
            except Exception as e:
                _record_debug("error", f"timeout/fallo esperando resultado: {e}")
            funnel[reason] = funnel.get(reason, 0) + 1
            if r is not None:
                survivors.append(r)

    survivors.sort(key=lambda s: s["Ticker"])

    results = []
    tickers_ok = 0
    n_survivors = len(survivors)
    for i, s in enumerate(survivors, start=1):
        progress_bar.progress(0.3 + (i / max(n_survivors, 1)) * 0.7)
        status_text.text(
            f"🔍 Fase 2/2 — opciones: {i}/{n_survivors} ({s['Ticker']}) · "
            f"tickers con collar: {tickers_ok} · collares: {len(results)}"
        )
        rows, reason = phase2_collar_scan(s, params)
        funnel[reason] = funnel.get(reason, 0) + 1
        if rows:
            tickers_ok += 1
            results.extend(rows)

    status_text.text(f"✅ Completado: {len(results)} collares en {tickers_ok} tickers")

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(["Ganancia_Max_%", "Ticker"], ascending=[False, True]).reset_index(drop=True)

    with _debug_lock:
        debug_snapshot = {k: list(v) for k, v in _debug_samples.items()}

    return df, funnel, debug_snapshot


def reset_everything():
    get_price_data.clear()
    get_chart_data.clear()
    get_option_expirations.clear()
    get_cboe_weeklys_tickers.clear()
    for key in ("results", "funnel", "debug_snapshot", "scan_ts", "scanned_total"):
        st.session_state.pop(key, None)


# ======================================================================
# 5. GRÁFICOS
# ======================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def get_chart_data(ticker):
    """Velas diarias (1 año: margen para calentar las medias) vía yfinance."""
    try:
        h = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=False)
        if h is None or h.empty:
            return None
        h = h.dropna(subset=["Close"])
        return h[["Open", "High", "Low", "Close"]]
    except Exception as e:
        logger.warning(f"[chart] {ticker}: {e}")
        return None


def build_price_chart(ticker, bb_period, bb_std, strikes=None, height=460):
    """Velas diarias de hasta 6 meses + SMA30 + bandas de Bollinger.
    strikes = {"put": kp, "call": kc} dibuja líneas horizontales (opcional).
    Devuelve (figura, texto_resumen) o (None, mensaje)."""
    h = get_chart_data(ticker)
    if h is None or len(h) < max(bb_period, 30) + 5:
        return None, f"{ticker}: sin datos suficientes para el gráfico."

    close = h["Close"]
    sma30 = close.rolling(30).mean()
    mid = close.rolling(bb_period).mean()
    sd = close.rolling(bb_period).std(ddof=0)
    upper = mid + bb_std * sd
    lower = mid - bb_std * sd

    mask = h.index >= (h.index[-1] - pd.Timedelta(days=CHART_DAYS))
    hv = h.loc[mask]
    x = hv.index

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=upper.loc[mask], mode="lines", name=f"Bollinger sup ({bb_period}, {bb_std:g})",
                             line=dict(color="rgba(91,155,213,0.8)", width=1)))
    fig.add_trace(go.Scatter(x=x, y=lower.loc[mask], mode="lines", name="Bollinger inf",
                             line=dict(color="rgba(91,155,213,0.8)", width=1),
                             fill="tonexty", fillcolor="rgba(91,155,213,0.10)"))
    fig.add_trace(go.Scatter(x=x, y=mid.loc[mask], mode="lines", name=f"Media Bollinger ({bb_period})",
                             line=dict(color="rgba(91,155,213,0.6)", width=1, dash="dot")))
    fig.add_trace(go.Candlestick(x=x, open=hv["Open"], high=hv["High"], low=hv["Low"], close=hv["Close"],
                                 name="Precio",
                                 increasing_line_color="#6daa45", decreasing_line_color="#dd6974"))
    fig.add_trace(go.Scatter(x=x, y=sma30.loc[mask], mode="lines", name="SMA30",
                             line=dict(color="#e8af34", width=2)))

    if strikes:
        fig.add_hline(y=strikes["put"], line=dict(color="#dd6974", width=1, dash="dash"),
                      annotation_text=f"Put {strikes['put']:.2f}", annotation_position="bottom left")
        fig.add_hline(y=strikes["call"], line=dict(color="#5b9bd5", width=1, dash="dash"),
                      annotation_text=f"Call {strikes['call']:.2f}", annotation_position="top left")

    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_layout(
        title=f"{ticker} — diario, últimos 6 meses",
        template="plotly_dark", height=height, hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=70, b=10),
        yaxis_title="Precio ($)",
    )

    last = float(close.iloc[-1])
    s30 = float(sma30.iloc[-1]) if pd.notna(sma30.iloc[-1]) else None
    up, lo = float(upper.iloc[-1]), float(lower.iloc[-1])
    pctb = (last - lo) / (up - lo) if up > lo else None
    parts = [f"Cierre {last:.2f}"]
    if s30:
        parts.append(f"SMA30 {s30:.2f} ({(last / s30 - 1) * 100:+.1f}%)")
    if pctb is not None:
        parts.append(f"%B {pctb:.0%}")
    return fig, " · ".join(parts)


def plot_payoff(row):
    """Resultado al vencimiento del collar (por acción)."""
    S = float(row["Precio"])
    kp = float(row["Strike_Put"])
    kc = float(row["Strike_Call"])
    net_cost = float(row["_coste_acc"])

    xs = np.linspace(S * 0.7, S * 1.3, 200)
    pnl = (xs - S) + np.maximum(kp - xs, 0) - np.maximum(xs - kc, 0) - net_cost

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=xs - S, mode="lines", name="Solo acciones",
                             line=dict(color="#999999", dash="dash", width=1)))
    fig.add_trace(go.Scatter(x=xs, y=pnl, mode="lines", name="Collar",
                             line=dict(color="#6daa45", width=3)))
    fig.add_vline(x=S, line=dict(color="#ffffff", width=1), annotation_text=f"Precio {S:.2f}")
    fig.add_vline(x=kp, line=dict(color="#dd6974", width=1, dash="dot"),
                  annotation_text=f"Put {kp:.2f}", annotation_position="bottom left")
    fig.add_vline(x=kc, line=dict(color="#5b9bd5", width=1, dash="dot"),
                  annotation_text=f"Call {kc:.2f}", annotation_position="top right")
    fig.add_hline(y=0, line=dict(color="#666666", width=1))
    fig.update_layout(
        title=f"{row['Ticker']} — resultado al vencimiento {row['Vencimiento']} (por acción)",
        template="plotly_dark", height=380, hovermode="x unified",
        xaxis_title="Precio de la acción al vencimiento",
        yaxis_title="Resultado ($ por acción)",
    )
    return fig


# ======================================================================
# 6. INTERFAZ
# ======================================================================

def main():
    st.set_page_config(page_title="Collar Screener", page_icon="🧱", layout="wide")

    if not check_password():
        st.stop()

    st.title("🧱 Collar Screener — crédito, DTE 30-60")
    st.markdown(
        "**Acciones + put ATM + call más lejano que lo financia · "
        "midprices y margen IBKR aproximado en la tabla**"
    )

    col_title, col_reset = st.columns([5, 1])
    with col_reset:
        if st.button("🧹 Resetear Todo", use_container_width=True,
                     help="Vacía el caché y borra los resultados del último escaneo."):
            reset_everything()
            st.success("Caché y resultados borrados.")
            st.rerun()

    st.divider()

    # ── Universo ───────────────────────────────────────────────────────
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
            st.success(f"✅ **{meta['total_count']:,} tickers** en el universo")
        else:
            st.warning(f"⚠️ Solo universo adicional: {meta['total_count']:,} tickers")

    # ── Parámetros ─────────────────────────────────────────────────────
    st.markdown("### ⚙️ Parámetros")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**📆 Vencimientos y ganancia**")
        a, b = st.columns(2)
        min_dte = a.number_input("DTE desde", 1, 365, 30, 5)
        max_dte = b.number_input("DTE hasta", 2, 365, 60, 5)
        if min_dte > max_dte:
            min_dte, max_dte = max_dte, min_dte
        min_gain = st.number_input(
            "Ganancia máx. mínima (%)", 0.0, 500.0, 0.0, 0.5,
            help="Con 'solo crédito' y put ATM, el call suele quedar a pocos puntos "
                 "del precio: una ganancia mínima alta puede dejar el resultado vacío.",
        )

        st.markdown("**🧱 Riesgo del collar**")
        only_credit = st.checkbox(
            "Solo crédito", value=True,
            help="El call debe ingresar más que lo que cuesta el put (crédito neto). "
                 "Desmárcalo para aceptar un débito pequeño.",
        )
        debit_tol_pct = st.number_input(
            "Débito máximo (% del precio de la acción)", 0.0, 10.0, 0.5, 0.1,
            disabled=only_credit,
            help="Solo se usa si 'Solo crédito' está desactivado.",
        )
        pc1, pc2 = st.columns(2)
        min_price = pc1.number_input("Precio mínimo ($)", 1, 10000, 15, 1)
        max_price = pc2.number_input("Precio máximo ($)", 1, 10000, 100, 1)
        if min_price > max_price:
            min_price, max_price = max_price, min_price

    with c2:
        st.markdown("**💧 Liquidez (filtro interno, ambas patas)**")
        use_oi_filter = st.checkbox("Exigir OI mínimo", value=True)
        min_oi = st.number_input("OI mínimo", 1, 10000, 50, 10, disabled=not use_oi_filter)
        use_spread_filter = st.checkbox("Exigir spread ≤ X% de la prima", value=True)
        spread_max_pct = st.number_input("Spread máximo (%)", 10, 300, 70, 10,
                                         disabled=not use_spread_filter)

        with st.expander("🏦 Margen IBKR aproximado (cuenta margin, Reg T)"):
            st.caption("Inicial = % x valor de las acciones (el call es OTM, no suma importe ITM). "
                       "Mantenimiento = mín(% x strike put + importe OTM del put; % x strike call). "
                       "Ajusta los % si tu broker exige más en alguna acción.")
            margin_init_pct = st.number_input("Margen inicial (% del valor de las acciones)",
                                              10.0, 100.0, 50.0, 5.0)
            margin_maint_put_pct = st.number_input("Mantenimiento: % del strike put", 1.0, 50.0, 10.0, 1.0)
            margin_maint_call_pct = st.number_input("Mantenimiento: % del strike call", 1.0, 50.0, 25.0, 1.0)

    st.markdown("**📈 Filtros de tendencia (Fase 1, opcionales)**")
    st.caption("Descartan activos en caída antes de consultar las opciones, así que además "
               "aceleran el escaneo. Todos desactivados por defecto. Un activo con historial "
               "insuficiente para un filtro activo se descarta.")
    g1, g2, g3 = st.columns(3)
    with g1:
        use_trend = st.checkbox("Precio > media de 200 días", value=False,
                                help="Seguimiento de tendencia (Faber 2007).")
        use_sma_cross = st.checkbox("Media 50d > media 200d", value=False,
                                    help="Tendencia alcista confirmada (golden cross).")
    with g2:
        use_momentum = st.checkbox("Momentum 12-1 meses >", value=False,
                                   help="Rentabilidad de hace 12 meses a hace 1 mes "
                                        "(Jegadeesh y Titman 1993).")
        min_momentum = st.number_input("Momentum mínimo (%)", -50.0, 200.0, 0.0, 5.0,
                                       disabled=not use_momentum)
    with g3:
        use_high52 = st.checkbox("Cerca del máximo de 52 semanas", value=False,
                                 help="George y Hwang (2004).")
        max_dist_high = st.number_input("Máx. distancia al máximo (%)", 1.0, 80.0, 15.0, 1.0,
                                        disabled=not use_high52)
    use_vol = st.checkbox("Limitar volatilidad realizada (1 año)", value=False,
                          help="Evita el sesgo del screener hacia activos muy volátiles.")
    max_vol = st.number_input("Volatilidad anualizada máxima (%)", 10.0, 300.0, 60.0, 5.0,
                              disabled=not use_vol)

    params = {
        "min_dte": int(min_dte), "max_dte": int(max_dte), "min_gain": min_gain,
        "only_credit": only_credit, "debit_tol_pct": debit_tol_pct,
        "min_price": min_price, "max_price": max_price,
        "use_oi_filter": use_oi_filter, "min_oi": min_oi,
        "use_spread_filter": use_spread_filter, "spread_max_pct": spread_max_pct,
        "margin_init_pct": margin_init_pct,
        "margin_maint_put_pct": margin_maint_put_pct,
        "margin_maint_call_pct": margin_maint_call_pct,
        "use_trend": use_trend, "use_sma_cross": use_sma_cross,
        "use_momentum": use_momentum, "min_momentum": min_momentum,
        "use_high52": use_high52, "max_dist_high": max_dist_high,
        "use_vol": use_vol, "max_vol": max_vol,
    }

    st.divider()

    # ── Escaneo ────────────────────────────────────────────────────────
    st.markdown("### 🚀 Escaneo")
    with st.expander("🧪 Prueba rápida con universo reducido (opcional)"):
        test_raw = st.text_input("Tickers de prueba (coma o espacio)", value="",
                                 placeholder="F, PFE, T, BAC")
    test_tickers = [_clean_ticker(t) for t in test_raw.replace(",", " ").split()] if test_raw.strip() else []
    test_tickers = [t for t in test_tickers if t]
    scan_universe = test_tickers if test_tickers else tickers_all

    scan_btn = st.button("🎯 INICIAR ESCANEO", type="primary", use_container_width=True,
                         disabled=(len(scan_universe) == 0))
    if test_tickers:
        st.caption(f"🧪 Modo prueba: **{len(scan_universe)}** tickers.")
    else:
        st.caption(f"ℹ️ Se escanearán hasta **{len(scan_universe):,}** tickers.")

    if scan_btn:
        progress_bar = st.progress(0)
        status_text = st.empty()
        df_results, funnel, debug_snapshot = run_screener(
            scan_universe, params, progress_bar, status_text
        )
        progress_bar.empty()

        st.session_state["results"] = df_results
        st.session_state["funnel"] = funnel
        st.session_state["debug_snapshot"] = debug_snapshot
        st.session_state["scan_ts"] = datetime.now()
        st.session_state["scanned_total"] = len(scan_universe)

        if df_results.empty:
            st.warning("⚠️ Ningún collar cumplió los filtros. Mira el embudo de diagnóstico.")

    # ── Embudo ─────────────────────────────────────────────────────────
    if "funnel" in st.session_state:
        with st.expander("🔎 Embudo de diagnóstico"):
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
                with st.expander("🐛 Mensajes de error (muestras)"):
                    for code in REASON_ORDER:
                        for m in debug_snapshot.get(code, []):
                            st.code(f"[{code}] {m}", language=None)

    st.divider()
    render_results()


REQUIRED_COLS = {
    "Ticker", "Precio", "Vencimiento", "DTE", "Strike_Put", "Put_Mid", "Strike_Call",
    "Call_Mid", "Tipo", "Coste_Neto_$", "Coste_Neto_%", "Ganancia_Max_$",
    "Ganancia_Max_%", "Call_Skew", "Margen_Inicial_$", "Margen_Mant_$", "_coste_acc",
}


def render_results():
    st.markdown("### 📊 Resultados")

    # Resultados de una versión anterior del código (otras columnas): se descartan.
    stale = st.session_state.get("results")
    if stale is not None and not stale.empty and not REQUIRED_COLS.issubset(stale.columns):
        for key in ("results", "funnel", "debug_snapshot", "scan_ts", "scanned_total"):
            st.session_state.pop(key, None)
        st.warning("⚠️ Había resultados de una versión anterior del screener. "
                   "Se han descartado: lanza un nuevo escaneo.")
        return

    if "results" not in st.session_state or st.session_state["results"].empty:
        st.info("👆 Configura los parámetros y pulsa **INICIAR ESCANEO**.")
        return

    df_all = st.session_state["results"].copy()
    ts = st.session_state["scan_ts"]

    # Filtros rápidos sobre resultados
    f1, f2 = st.columns([1, 3])
    with f1:
        use_skew = st.checkbox("Call skew ≥ 0", value=False,
                               help="IV call − IV put, en puntos. Excluye los sin dato.")
    with f2:
        pick = st.multiselect("Tickers (vacío = todos)", sorted(df_all["Ticker"].unique()))

    df = df_all
    if pick:
        df = df[df["Ticker"].isin(pick)]
    if use_skew:
        df = df[df["Call_Skew"].notna() & (df["Call_Skew"] >= 0)]

    if df.empty:
        st.warning("⚠️ Ningún collar cumple estos filtros.")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Collares", len(df))
    k2.metric("Tickers", df["Ticker"].nunique())
    k3.metric("Mejor ganancia", f"{df['Ganancia_Max_%'].max():.1f}%")
    k4.metric("Con crédito", int((df["Tipo"] == "Crédito").sum()))
    st.caption(f"Último escaneo: {ts.strftime('%H:%M:%S')}")

    cols_show = [
        "Ticker", "Precio", "Vencimiento", "DTE",
        "Strike_Put", "Put_Mid", "Strike_Call", "Call_Mid",
        "Tipo", "Coste_Neto_$", "Coste_Neto_%",
        "Ganancia_Max_$", "Ganancia_Max_%", "Call_Skew",
        "Margen_Inicial_$", "Margen_Mant_$",
    ]
    names = {
        "Strike_Put": "Put (ATM)", "Put_Mid": "Put mid",
        "Strike_Call": "Call", "Call_Mid": "Call mid",
        "Coste_Neto_$": "Coste neto $", "Coste_Neto_%": "Coste neto %",
        "Ganancia_Max_$": "Ganancia máx. $", "Ganancia_Max_%": "Ganancia máx. %",
        "Call_Skew": "Call skew",
        "Margen_Inicial_$": "Margen inicial $", "Margen_Mant_$": "Margen mant. $",
    }

    def color_tipo(v):
        if v == "Crédito":
            return "background-color:#1e3a1e; color:#6daa45; font-weight:600"
        if v == "Cero":
            return "background-color:#1e3a1e; color:#a8d08d; font-weight:600"
        if v == "Débito":
            return "background-color:#3a3a1e; color:#e8af34; font-weight:600"
        return ""

    def show_table(r, key, file_prefix):
        view = r[cols_show].rename(columns=names)
        styler = (
            view.style
            .format({
                "Precio": "${:,.2f}",
                "Put (ATM)": "{:,.2f}", "Put mid": "{:.3f}",
                "Call": "{:,.2f}", "Call mid": "{:.3f}",
                "Coste neto $": "{:+,.0f}", "Coste neto %": "{:+.2f}%",
                "Ganancia máx. $": "${:,.0f}", "Ganancia máx. %": "{:.1f}%",
                "Call skew": "{:+.1f}",
                "Margen inicial $": "${:,.0f}", "Margen mant. $": "${:,.0f}",
            }, na_rep="–")
            .map(color_tipo, subset=["Tipo"])
        )
        st.dataframe(styler, use_container_width=True, hide_index=True, height=500)
        st.download_button(
            "⬇️ Descargar CSV", view.to_csv(index=False).encode("utf-8"),
            f"{file_prefix}_{ts.strftime('%Y%m%d_%H%M')}.csv", "text/csv", key=f"dl_{key}",
        )

    with st.expander("⚙️ Opciones de los gráficos (Bollinger)"):
        bcol1, bcol2 = st.columns(2)
        bb_period = bcol1.number_input("Periodo de Bollinger", 5, 100, 20, 1, key="bb_period")
        bb_std = bcol2.number_input("Desviaciones típicas", 0.5, 4.0, 2.0, 0.5, key="bb_std")

    tab1, tab2, tab3, tab4 = st.tabs(["🥇 Por ganancia", "⏱️ Por tiempo", "📈 Gráficos", "🔍 Detalle"])

    with tab1:
        st.caption("Ordenado por ganancia máxima % (mayor primero). Por 100 acciones. "
                   "Coste neto: positivo = débito, negativo = crédito. Mid = precio medio por acción. "
                   "Margen = estimación IBKR (Reg T), no el margen real.")
        show_table(df.sort_values(["Ganancia_Max_%", "Ticker"], ascending=[False, True]),
                   "gan", "collar_ganancia")

    with tab2:
        st.caption("Ordenado por DTE (vencimiento más cercano primero) y, a igual DTE, por ganancia %.")
        show_table(df.sort_values(["DTE", "Ganancia_Max_%"], ascending=[True, False]),
                   "tiempo", "collar_tiempo")

    with tab3:
        st.caption("Velas diarias, 6 meses, SMA30 y bandas de Bollinger. "
                   "%B: posición del cierre dentro de las bandas (0% = banda inferior, 100% = superior).")
        ticker_opts = list(dict.fromkeys(
            df.sort_values("Ganancia_Max_%", ascending=False)["Ticker"].tolist()
        ))
        chosen = st.multiselect("Activos a graficar", ticker_opts, default=ticker_opts[:4],
                                key="chart_tickers")
        if len(chosen) > 12:
            st.info("Se muestran los 12 primeros.")
            chosen = chosen[:12]
        for i in range(0, len(chosen), 2):
            cols = st.columns(2)
            for col, tk in zip(cols, chosen[i:i + 2]):
                with col:
                    fig, info = build_price_chart(tk, int(bb_period), float(bb_std), height=380)
                    if fig is None:
                        st.warning(info)
                    else:
                        st.plotly_chart(fig, use_container_width=True, key=f"pc_{tk}")
                        st.caption(info)

    with tab4:
        d = df.sort_values("Ganancia_Max_%", ascending=False)
        labels = {
            int(i): (f"{r['Ticker']} · {r['Vencimiento']} ({r['DTE']}d) · "
                     f"Put {r['Strike_Put']:.2f} / Call {r['Strike_Call']:.2f} · "
                     f"{r['Ganancia_Max_%']:.1f}%")
            for i, r in d.iterrows()
        }
        sel = st.selectbox("Collar", options=list(labels.keys()), format_func=lambda k: labels[k])
        row = d.loc[sel]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Coste neto", f"{row['Coste_Neto_$']:+,.0f} $", row["Tipo"], delta_color="off")
        m2.metric("Ganancia máx.", f"${row['Ganancia_Max_$']:,.0f}", f"{row['Ganancia_Max_%']:.1f}%")
        skew = f"{row['Call_Skew']:+.1f} pts" if pd.notna(row["Call_Skew"]) else "n/d"
        m3.metric("Call skew", skew)
        m4.metric("Vencimiento", f"{row['DTE']} días", row["Vencimiento"], delta_color="off")

        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Put mid", f"{row['Put_Mid']:.3f}", f"Strike {row['Strike_Put']:.2f}", delta_color="off")
        n2.metric("Call mid", f"{row['Call_Mid']:.3f}", f"Strike {row['Strike_Call']:.2f}", delta_color="off")
        n3.metric("Margen inicial (aprox.)", f"${row['Margen_Inicial_$']:,.0f}")
        n4.metric("Margen mant. (aprox.)", f"${row['Margen_Mant_$']:,.0f}")

        fig, info = build_price_chart(
            row["Ticker"], int(bb_period), float(bb_std),
            strikes={"put": float(row["Strike_Put"]), "call": float(row["Strike_Call"])},
        )
        if fig is None:
            st.warning(info)
        else:
            st.plotly_chart(fig, use_container_width=True, key=f"pd_{sel}")
            st.caption(info)

        st.plotly_chart(plot_payoff(row), use_container_width=True, key=f"po_{sel}")
        st.caption("Sin filtro de dividendos: revisa el riesgo de ejercicio anticipado del call "
                   "cerca de un ex-dividend.")


if __name__ == "__main__":
    main()
