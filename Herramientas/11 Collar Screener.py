"""
Collar Screener (coste cero) — Call Skew
=========================================
Objetivo: encontrar COLLARS de coste cero (o casi) sobre acciones:

    + 100 acciones
    + 1 long put  ATM o hasta X% OTM (por defecto 2%)         -> protección
    - 1 short call OTM, como mínimo Y% OTM (por defecto 5%)   -> financia el put
    Mismo vencimiento para ambas patas.

La idea es aprovechar el CALL SKEW: vencimientos donde el call OTM se paga
tanto (o más) que el put de protección, de modo que el recorrido alcista
queda asegurado con riesgo (casi) nulo.

SELECCIÓN:
- Para CADA put candidato (strike <= precio y >= precio x (1 - put_max_otm))
  se busca el call de strike MÁS LEJANO (>= call_min_otm) cuya prima cubra
  el put dentro de la tolerancia de débito:

        prima_call >= prima_put - tolerancia_debito
        tolerancia_debito = precio_accion x debito_max_pct / 100

  Se devuelve UNA fila por combinación (vencimiento, put).
- Los créditos netos (call más caro que el put) se aceptan siempre.

MÉTRICAS (por acción, sobre el PRECIO de la acción):
- Coste_Neto      = prima_put - prima_call (positivo = débito, negativo = crédito)
- Ganancia_Max    = strike_call - precio - coste_neto
- Ganancia_Max_%  = Ganancia_Max / precio x 100
- Ganancia_Anualizada_% = Ganancia_Max_% x 365 / DTE   (simple, no compuesta)
- Perdida_Max_%   = (precio - strike_put + coste_neto) / precio x 100
- Coste_Conservador_% : comprar el put al ASK y vender el call al BID
  (para ver si el coste cero aguanta con spreads anchos).
- Call_Skew       = IV del call vendido - IV del put comprado, en PUNTOS de
  volatilidad (informativo; positivo = el call está más caro en vol).

RANKINGS:
  1) Ganancia máxima % (distancia del call)
  2) Ganancia máxima anualizada % (tiempo y ganancia)
  El tope de filas por ticker se aplica a CADA ranking (se conservan las
  mejores N por ganancia % y las mejores N por ganancia anualizada).

FILTROS (UI): rango de precio, DTE mínimo/máximo, OI mínimo y spread máximo
(aplicados a ambas patas). NO hay filtros de SMA, earnings ni dividendos.

AVISO: sin filtro de dividendos, un call corto ITM cerca de un ex-dividend
tiene riesgo de ejercicio anticipado. Revísalo antes de operar.

FASES: Fase 1 (paralela, yfinance): precio y rango. Fase 2 (secuencial,
Alpaca): vencimientos, cadenas y combinaciones.
PRECIO DE OPCIÓN: midprice (bid+ask)/2.
DATOS: opciones vía Alpaca — utils/utils_alpaca.py.
Solo screener (sin envío de órdenes).
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import csv
from datetime import datetime, date
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
    get_option_chain,
    get_live_price,
)

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("collar_screener")

SOCKET_TIMEOUT = 15
socket.setdefaulttimeout(SOCKET_TIMEOUT)

MAX_WORKERS = 3

_N = NormalDist()
RISK_FREE_RATE = 0.045

# CSV público de Cboe "Available Weeklys"
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
    "ok", "no_price_data", "price_out_of_range",
    "no_expirations", "no_expirations_in_window",
    "no_chain", "no_collar", "error",
]

REASON_LABELS = {
    "ok": "✅ Con al menos un collar válido",
    "no_price_data": "Sin precio disponible (Fase 1)",
    "price_out_of_range": "Precio fuera de rango (Fase 1)",
    "no_expirations": "Sin vencimientos de opciones listados en Alpaca (Fase 2)",
    "no_expirations_in_window": "Sin vencimientos dentro del rango de DTE (Fase 2)",
    "no_chain": "Cadenas vacías/no disponibles en los vencimientos (Fase 2)",
    "no_collar": "Ningún collar cumple prima/OI/spread/distancias (Fase 2)",
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


def _safe_int(x, default=0):
    """int() tolerante a None/NaN."""
    try:
        v = pd.to_numeric(x, errors="coerce")
        if v is None or pd.isna(v):
            return default
        return int(v)
    except Exception:
        return default


def _iv_or_none(x):
    try:
        v = float(x)
        if np.isnan(v) or v <= 0:
            return None
        return v
    except Exception:
        return None


# ======================================================================
# 0. UNIVERSO (idéntico al CSP screener)
# ======================================================================

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_cboe_weeklys_tickers():
    """Descarga el CSV público de 'Available Weeklys' de Cboe y extrae los
    tickers de ETFs/ETNs y Acciones (índices ignorados). Cacheado 24h.
    Devuelve un set (vacío si algo falla: nunca es un error fatal)."""
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
    """Universo completo: base fija + Russell 1000 dinámico + Cboe Weeklys
    nuevos (aditivo y best-effort)."""
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
# 1. BLACK-SCHOLES DELTA
# ======================================================================

def bs_put_delta(S, K, T_years, sigma, r=RISK_FREE_RATE):
    """Delta de un put europeo: N(d1) - 1."""
    try:
        if T_years <= 0 or sigma is None or sigma <= 0 or S <= 0 or K <= 0:
            return None
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_years) / (sigma * np.sqrt(T_years))
        return round(float(_N.cdf(d1) - 1), 3)
    except Exception:
        return None


def bs_call_delta(S, K, T_years, sigma, r=RISK_FREE_RATE):
    """Delta de un call europeo: N(d1)."""
    try:
        if T_years <= 0 or sigma is None or sigma <= 0 or S <= 0 or K <= 0:
            return None
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_years) / (sigma * np.sqrt(T_years))
        return round(float(_N.cdf(d1)), 3)
    except Exception:
        return None


# ======================================================================
# 2. FASE 1 — precio (paralela)
# ======================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def get_last_price(ticker):
    """Último cierre diario vía yfinance (ligero: 5 días). None si falla."""
    try:
        h = yf.Ticker(ticker).history(period="5d", interval="1d", auto_adjust=False)
        if h is None or h.empty:
            _record_debug("no_price_data", f"{ticker}: descarga vacía")
            return None
        close = h["Close"].dropna()
        if close.empty:
            _record_debug("no_price_data", f"{ticker}: sin cierres")
            return None
        return float(close.iloc[-1])
    except Exception as e:
        _record_debug("no_price_data", f"{ticker}: {e}")
        return None


def phase1_price_filter(ticker, params):
    try:
        price = get_last_price(ticker)
        if price is None:
            return None, "no_price_data"
        if not (params["min_price"] <= price <= params["max_price"]):
            _record_debug("price_out_of_range", f"{ticker}: precio = {price}")
            return None, "price_out_of_range"
        return {"Ticker": ticker, "current_price": round(price, 2)}, "ok"
    except Exception as e:
        _record_debug("error", f"{ticker}: {e}")
        return None, "error"


# ======================================================================
# 3. COLLARS: put ATM/≤X% OTM + call ≥Y% OTM, mismo vencimiento
# ======================================================================

def _prep_leg(df, apply_oi, min_oi, apply_spread, spread_max_pct):
    """Limpia una pata (calls o puts): bid>0, ask>0, mid, OI, spread, IV."""
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    d["bid"] = pd.to_numeric(d["bid"], errors="coerce").fillna(0)
    d["ask"] = pd.to_numeric(d["ask"], errors="coerce").fillna(0)
    d["mid"] = (d["bid"] + d["ask"]) / 2
    d = d[(d["bid"] > 0) & (d["ask"] > 0) & (d["mid"] > 0)]
    if d.empty:
        return pd.DataFrame()

    d["oi"] = pd.to_numeric(
        d.get("openInterest", pd.Series(0, index=d.index)), errors="coerce"
    ).fillna(0)
    if apply_oi:
        d = d[d["oi"] >= min_oi]
        if d.empty:
            return pd.DataFrame()

    d["spread_dollar"] = d["ask"] - d["bid"]
    d["spread_pct"] = d["spread_dollar"] / d["mid"] * 100
    if apply_spread:
        d = d[d["spread_dollar"] <= d["mid"] * (spread_max_pct / 100)]
        if d.empty:
            return pd.DataFrame()

    if "impliedVolatility" in d.columns:
        d["iv"] = pd.to_numeric(d["impliedVolatility"], errors="coerce")
    else:
        d["iv"] = np.nan
    return d


def find_collars(ticker, exp, calls, puts, price, params):
    """Para cada put candidato devuelve el collar con el call de strike más
    lejano (>= call_min_otm) cuya prima cubre el put dentro de la
    tolerancia de débito. Devuelve una lista de filas (dicts)."""
    try:
        dte = (exp - date.today()).days
        if dte <= 0:
            return []
        T_years = dte / 365.0

        p = _prep_leg(puts, params["use_oi_filter"], params["min_oi"],
                      params["use_spread_filter"], params["spread_max_pct"])
        c = _prep_leg(calls, params["use_oi_filter"], params["min_oi"],
                      params["use_spread_filter"], params["spread_max_pct"])
        if p.empty or c.empty:
            return []

        p = p[(p["strike"] <= price) & (p["strike"] >= price * (1 - params["put_max_otm"] / 100))]
        c = c[c["strike"] >= price * (1 + params["call_min_otm"] / 100)]
        if p.empty or c.empty:
            return []

        tol = price * params["debit_tol_pct"] / 100.0  # $ por acción

        rows = []
        for _, pr in p.iterrows():
            ok = c[c["mid"] >= pr["mid"] - tol]
            if ok.empty:
                continue
            cr = ok.sort_values("strike", ascending=False).iloc[0]

            kp = float(pr["strike"])
            kc = float(cr["strike"])
            put_mid = float(pr["mid"])
            call_mid = float(cr["mid"])

            net_cost = put_mid - call_mid
            max_gain = kc - price - net_cost
            if max_gain <= 0:
                continue
            max_loss = price - kp + net_cost
            cons_cost = float(pr["ask"]) - float(cr["bid"])

            iv_put = _iv_or_none(pr["iv"])
            iv_call = _iv_or_none(cr["iv"])
            call_skew = round((iv_call - iv_put) * 100, 2) if (iv_put and iv_call) else None

            gain_pct = max_gain / price * 100

            rows.append({
                "Ticker": ticker,
                "Precio": round(price, 2),
                "Vencimiento": exp.isoformat(),
                "DTE": dte,
                "Strike_Put": kp,
                "Put_OTM_%": round((price - kp) / price * 100, 2),
                "Put_Mid": round(put_mid, 2),
                "Put_Ask": round(float(pr["ask"]), 2),
                "Strike_Call": kc,
                "Call_OTM_%": round((kc - price) / price * 100, 2),
                "Call_Mid": round(call_mid, 2),
                "Call_Bid": round(float(cr["bid"]), 2),
                "Coste_Neto": round(net_cost, 2),
                "Coste_Neto_%": round(net_cost / price * 100, 3),
                "Coste_Conservador_%": round(cons_cost / price * 100, 3),
                "Ganancia_Max_$": round(max_gain * 100, 2),
                "Ganancia_Max_%": round(gain_pct, 2),
                "Ganancia_Anualizada_%": round(gain_pct * 365.0 / dte, 2),
                "Perdida_Max_%": round(max_loss / price * 100, 2),
                "Capital_Neto": round((price + net_cost) * 100, 2),
                "IV_Put_%": round(iv_put * 100, 2) if iv_put else None,
                "IV_Call_%": round(iv_call * 100, 2) if iv_call else None,
                "Call_Skew": call_skew,
                "Delta_Put": bs_put_delta(price, kp, T_years, iv_put),
                "Delta_Call": bs_call_delta(price, kc, T_years, iv_call),
                "OI_Put": _safe_int(pr["oi"]),
                "OI_Call": _safe_int(cr["oi"]),
                "Spread_Put_%": round(float(pr["spread_pct"]), 2),
                "Spread_Call_%": round(float(cr["spread_pct"]), 2),
            })
        return rows
    except Exception as e:
        _record_debug("error", f"find_collars {ticker} {exp}: {e}")
        return []


def _sample_evenly(items, n):
    """Si hay más de n elementos, devuelve n repartidos uniformemente
    (incluye el primero y el último). n<=0 = sin límite."""
    if n is None or n <= 0 or len(items) <= n:
        return items
    idx = sorted(set(np.linspace(0, len(items) - 1, n).round().astype(int).tolist()))
    return [items[i] for i in idx]


# ======================================================================
# 4. FASE 2 — opciones (SECUENCIAL)
# ======================================================================

def phase2_collar_scan(survivor, params):
    """Recorre los vencimientos dentro de [min_dte, max_dte] y devuelve
    (lista de filas, motivo)."""
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
        valid = _sample_evenly(valid, params["max_expirations"])

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
            rows.extend(find_collars(ticker, exp, calls, puts, price, params))

        if chains_ok == 0:
            return None, "no_chain"
        if not rows:
            return None, "no_collar"

        # Tope por ticker, aplicado a CADA ranking (unión de ambos tops)
        max_n = params["max_rows_per_ticker"]
        if max_n and max_n > 0 and len(rows) > max_n:
            tdf = pd.DataFrame(rows)
            top_gain = tdf.sort_values("Ganancia_Max_%", ascending=False).head(max_n)
            top_ann = tdf.sort_values("Ganancia_Anualizada_%", ascending=False).head(max_n)
            tdf = pd.concat([top_gain, top_ann]).drop_duplicates(
                subset=["Vencimiento", "Strike_Put", "Strike_Call"]
            )
            rows = tdf.to_dict("records")

        return rows, "ok"
    except Exception as e:
        _record_debug("error", f"{ticker}: {e}")
        return None, "error"


# ======================================================================
# 5. ORQUESTADOR: Fase 1 (paralela) → Fase 2 (secuencial)
# ======================================================================

def run_screener(tickers, params, progress_bar, status_text):
    _reset_debug()
    funnel = {r: 0 for r in REASON_ORDER}
    total = len(tickers)

    status_text.text(f"🔍 Fase 1/2 — precio: 0/{total}")
    survivors = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(phase1_price_filter, t, params): t for t in tickers}
        done = 0
        for future in as_completed(futures):
            done += 1
            progress_bar.progress(done / total * 0.3)
            status_text.text(f"🔍 Fase 1/2 — precio: {done}/{total} · supervivientes: {len(survivors)}")
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
        funnel[reason] = funnel.get(reason, 0) + 1   # el embudo cuenta TICKERS
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
    get_last_price.clear()
    get_option_expirations.clear()
    get_cboe_weeklys_tickers.clear()
    for key in ("results", "funnel", "debug_snapshot", "scan_ts", "scanned_total"):
        st.session_state.pop(key, None)


# ======================================================================
# 6. GRÁFICOS
# ======================================================================

def plot_payoff(row):
    """Perfil de resultado al vencimiento del collar (por acción)."""
    S = float(row["Precio"])
    kp = float(row["Strike_Put"])
    kc = float(row["Strike_Call"])
    net_cost = float(row["Coste_Neto"])

    xs = np.linspace(S * 0.6, S * 1.5, 200)
    pnl = (xs - S) + np.maximum(kp - xs, 0) - np.maximum(xs - kc, 0) - net_cost
    pnl_stock = xs - S

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=pnl_stock, mode="lines", name="Solo acciones",
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
        template="plotly_dark", height=420, hovermode="x unified",
        xaxis_title="Precio de la acción al vencimiento",
        yaxis_title="Resultado ($ por acción)",
    )
    return fig


# ======================================================================
# 7. INTERFAZ STREAMLIT
# ======================================================================

def main():
    st.set_page_config(page_title="Collar Screener", page_icon="🧱", layout="wide")

    if not check_password():
        st.stop()

    st.title("🧱 Collar Screener — coste cero (Call Skew)")
    st.markdown(
        "**Long put ATM / ≤X% OTM + short call ≥Y% OTM, mismo vencimiento · "
        "prima del call ≥ prima del put (con tolerancia de débito) · "
        "recorrido alcista asegurado con riesgo acotado**"
    )
    st.caption(
        "⚙️ Opciones vía Alpaca · precio vía yfinance · midprice · "
        "filtros de OI y spread en ambas patas · sin filtros de SMA, earnings ni dividendos"
    )

    col_title, col_reset = st.columns([5, 1])
    with col_reset:
        if st.button("🧹 Resetear Todo", use_container_width=True,
                     help="Vacía el caché de precios y vencimientos y borra los "
                          "resultados del último escaneo."):
            reset_everything()
            st.success("Caché y resultados borrados — listo para un escaneo limpio.")
            st.rerun()

    st.divider()

    # ── Universo ───────────────────────────────────────────────────────
    st.markdown("### 📂 Universo de Tickers")
    col_info, col_btn = st.columns([4, 1])
    with col_btn:
        if st.button("🔄 Actualizar Universo", type="primary", use_container_width=True):
            with st.spinner("Reconstruyendo universo (incluye descarga de Cboe Weeklys)..."):
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
            if meta.get("cboe_ok"):
                cboe_txt = (
                    f" + Cboe Weeklys nuevos: {meta['cboe_new_count']:,} "
                    f"(de {meta['cboe_total_found']:,} encontrados en el CSV)"
                )
            else:
                cboe_txt = " · ⚠️ CSV de Cboe Weeklys no disponible en este refresco"
            st.success(
                f"✅ **{meta['total_count']:,} tickers** "
                f"(Russell 1000: {meta['r1000_count']:,} + Adicionales: {meta['extra_count']:,}{cboe_txt})"
            )
        else:
            st.warning(f"⚠️ Solo universo adicional: {meta['total_count']:,} tickers")

    st.divider()

    # ── Parámetros ─────────────────────────────────────────────────────
    st.markdown("### ⚙️ Parámetros")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**🧱 Estructura del collar**")
        debit_tol_pct = st.number_input(
            "Débito máximo (% del precio de la acción)", min_value=0.0, max_value=10.0,
            value=1.0, step=0.1,
            help="Tolerancia: se acepta que el put cueste hasta este % del precio de la "
                 "acción más que el call. Los créditos (call más caro que el put) se "
                 "aceptan siempre. 0 = el call debe valer al menos lo que el put.",
        )
        put_max_otm = st.number_input(
            "Put: distancia OTM máxima (%)", min_value=0.0, max_value=10.0,
            value=2.0, step=0.5,
            help="Se prueban todos los puts con strike <= precio y hasta este % por debajo.",
        )
        call_min_otm = st.number_input(
            "Call: distancia OTM mínima (%)", min_value=0.0, max_value=50.0,
            value=5.0, step=0.5,
            help="El call vendido debe estar al menos este % por encima del precio. "
                 "Para cada put se elige el strike más lejano que cubre el coste.",
        )
        max_rows_per_ticker = st.number_input(
            "Máx. collares por ticker y ranking (0 = sin tope)", min_value=0, max_value=100,
            value=10, step=1,
            help="Se conservan los N mejores por ganancia % y los N mejores por "
                 "ganancia anualizada (unión de ambos).",
        )

        st.markdown("**💲 Precio del subyacente**")
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            min_price = st.number_input("Precio mínimo ($)", min_value=1, max_value=10000, value=15, step=1)
        with pcol2:
            max_price = st.number_input("Precio máximo ($)", min_value=1, max_value=10000, value=100, step=1)
        if min_price > max_price:
            st.error(f"⚠️ El precio mínimo (${min_price:,}) es mayor que el máximo (${max_price:,}) — se intercambian.")
            min_price, max_price = max_price, min_price
        st.caption(f"✅ Rango activo: **${min_price:,} — ${max_price:,}**")

    with c2:
        st.markdown("**📆 Vencimientos**")
        include_short = st.checkbox(
            "Incluir vencimientos de menos de 30 días", value=False,
            help="Por defecto solo se miran vencimientos desde el DTE mínimo. Actívalo "
                 "para ver también los más cortos (desde 1 día).",
        )
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            min_dte = st.number_input("DTE mínimo", min_value=1, max_value=730, value=30, step=5,
                                      disabled=include_short)
        with dcol2:
            max_dte = st.number_input("DTE máximo", min_value=2, max_value=800, value=730, step=30)
        if include_short:
            min_dte = 1
        if min_dte > max_dte:
            st.error("⚠️ El DTE mínimo es mayor que el máximo — se intercambian.")
            min_dte, max_dte = max_dte, min_dte
        max_expirations = st.number_input(
            "Máx. vencimientos por ticker (0 = todos)", min_value=0, max_value=100,
            value=0, step=1,
            help="Si el ticker tiene más vencimientos en el rango, se muestrean de forma "
                 "uniforme (incluye el más cercano y el más lejano). Úsalo para acelerar "
                 "el escaneo: hay una llamada a Alpaca por vencimiento.",
        )
        st.caption(f"📅 Vencimientos entre **{min_dte}** y **{max_dte}** días.")

        st.markdown("**💧 Liquidez (ambas patas)**")
        use_oi_filter = st.checkbox(
            "Exigir OI mínimo en put y call", value=True,
            help="En LEAPS el OI suele ser bajo: relájalo si te descarta candidatos válidos.",
        )
        min_oi = st.number_input("OI mínimo", min_value=1, max_value=10000, value=50, step=10,
                                 disabled=not use_oi_filter)
        use_spread_filter = st.checkbox(
            "Exigir spread ≤ X% de la prima (put y call)", value=True,
            help="En LEAPS los spreads son anchos: revisa Spread_Put_% y Spread_Call_% "
                 "y el Coste_Conservador_% antes de operar.",
        )
        spread_max_pct = st.number_input("Spread máximo (% de la prima)", min_value=10, max_value=300,
                                         value=70, step=10, disabled=not use_spread_filter)

    params = {
        "debit_tol_pct": debit_tol_pct,
        "put_max_otm": put_max_otm,
        "call_min_otm": call_min_otm,
        "max_rows_per_ticker": int(max_rows_per_ticker),
        "min_price": min_price,
        "max_price": max_price,
        "min_dte": int(min_dte),
        "max_dte": int(max_dte),
        "max_expirations": int(max_expirations),
        "use_oi_filter": use_oi_filter,
        "min_oi": min_oi,
        "use_spread_filter": use_spread_filter,
        "spread_max_pct": spread_max_pct,
    }

    st.divider()

    # ── Escaneo ────────────────────────────────────────────────────────
    st.markdown("### 🚀 Ejecutar Escaneo")

    with st.expander("🧪 Prueba rápida con universo reducido (opcional)"):
        test_tickers_raw = st.text_input(
            "Tickers de prueba (separados por coma o espacio)", value="",
            placeholder="F, PFE, T, BAC",
        )
    test_tickers = [
        _clean_ticker(t) for t in test_tickers_raw.replace(",", " ").split()
    ] if test_tickers_raw.strip() else []
    test_tickers = [t for t in test_tickers if t]

    scan_universe = test_tickers if test_tickers else tickers_all

    scan_btn = st.button("🎯 INICIAR ESCANEO", type="primary", use_container_width=True,
                         disabled=(len(scan_universe) == 0))

    if test_tickers:
        st.caption(f"🧪 Modo prueba activo: se escanearán solo **{len(scan_universe)}** tickers.")
    else:
        st.caption(
            f"ℹ️ Se escanearán hasta **{len(scan_universe):,}** tickers · Fase 1 en paralelo → "
            f"Fase 2 secuencial. Hay una llamada a Alpaca por vencimiento y ticker: con todos "
            f"los vencimientos hasta 2 años el escaneo puede ser largo (usa el modo prueba o "
            f"limita los vencimientos por ticker)."
        )

    if scan_btn:
        if not scan_universe:
            st.error("⚠️ El universo de tickers está vacío — pulsa 'Actualizar Universo'.")
            return

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

        if not df_results.empty:
            n_tk = df_results["Ticker"].nunique()
            st.success(
                f"✅ **{len(df_results)} collares en {n_tk} tickers** encontrados "
                f"sobre {len(scan_universe):,} tickers analizados"
            )
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


def render_results():
    """Resultados guardados en session_state: filtros post-escaneo,
    métricas y pestañas (Ranking 1, Ranking 2, Detalle, Gráficos)."""
    st.markdown("### 📊 Resultados")

    if "results" not in st.session_state or st.session_state["results"].empty:
        st.info("👆 Configura los parámetros y pulsa **INICIAR ESCANEO**.")
        return

    df_all = st.session_state["results"].copy()
    ts = st.session_state["scan_ts"]

    st.markdown("#### 🔧 Filtros adicionales sobre resultados")
    st.caption("Se aplican sobre los collares ya encontrados — no relanzan la descarga de datos.")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        use_min_gain = st.checkbox("Ganancia máx. mínima (%)", value=False)
        min_gain = st.number_input("Valor mínimo (%)", min_value=0.0, max_value=500.0, value=8.0,
                                   step=1.0, disabled=not use_min_gain, key="min_gain_value")
    with f2:
        use_cons_filter = st.checkbox("Coste conservador máximo (%)", value=False,
                                      help="Coste de comprar el put al ASK y vender el call al BID, "
                                           "como % del precio. Filtra los collares cuyo coste cero "
                                           "desaparece con el spread.")
        max_cons = st.number_input("Valor máximo (%)", min_value=0.0, max_value=20.0, value=2.0,
                                   step=0.25, disabled=not use_cons_filter, key="max_cons_value")
    with f3:
        use_skew_filter = st.checkbox("Call skew mínimo (pts)", value=False,
                                      help="Call_Skew = IV del call vendido - IV del put comprado. "
                                           "Solo se muestran los collares con al menos este valor; "
                                           "los que no tienen skew calculable se excluyen.")
        min_skew = st.number_input("Valor mínimo (pts)", min_value=-50.0, max_value=50.0, value=0.0,
                                   step=0.5, disabled=not use_skew_filter, key="min_skew_value")
    with f4:
        dte_lo, dte_hi = int(df_all["DTE"].min()), int(df_all["DTE"].max())
        if dte_lo < dte_hi:
            dte_range = st.slider("Rango de DTE", min_value=dte_lo, max_value=dte_hi,
                                  value=(dte_lo, dte_hi), key="dte_range")
        else:
            dte_range = (dte_lo, dte_hi)
            st.caption(f"DTE único: {dte_lo}")

    df = df_all.copy()
    df = df[(df["DTE"] >= dte_range[0]) & (df["DTE"] <= dte_range[1])]
    if use_min_gain:
        df = df[df["Ganancia_Max_%"] >= min_gain]
    if use_cons_filter:
        df = df[df["Coste_Conservador_%"] <= max_cons]
    if use_skew_filter:
        df = df[df["Call_Skew"].notna() & (df["Call_Skew"] >= min_skew)]

    st.caption(f"ℹ️ Mostrando **{len(df)}** de {len(df_all)} collares.")
    st.divider()

    if df.empty:
        st.warning("⚠️ Ningún collar del último escaneo cumple estos filtros adicionales. Ajusta los valores de arriba.")
        return

    n_tickers = df["Ticker"].nunique()
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("📊 Collares", len(df))
    k2.metric("🏷️ Tickers", n_tickers)
    k3.metric("🚀 Ganancia máx. %", f"{df['Ganancia_Max_%'].max():.2f}%")
    k4.metric("📆 Mejor anualizada", f"{df['Ganancia_Anualizada_%'].max():.2f}%")
    k5.metric("🕐 Escaneo", ts.strftime("%H:%M:%S"))

    st.divider()

    cols_show = [
        "Rank", "Ticker", "Precio", "Vencimiento", "DTE",
        "Strike_Put", "Put_OTM_%", "Put_Mid", "Strike_Call", "Call_OTM_%", "Call_Mid",
        "Coste_Neto", "Coste_Neto_%", "Coste_Conservador_%",
        "Ganancia_Max_$", "Ganancia_Max_%", "Ganancia_Anualizada_%", "Perdida_Max_%",
        "Call_Skew", "IV_Put_%", "IV_Call_%", "Delta_Put", "Delta_Call",
        "OI_Put", "OI_Call", "Spread_Put_%", "Spread_Call_%", "Capital_Neto",
    ]

    def color_cost(val):
        try:
            if val is None or pd.isna(val):
                return ""
            if val <= 0:
                return "background-color:#1e3a1e; color:#6daa45"      # crédito
            if val <= 0.5:
                return "background-color:#3a3a1e; color:#e8af34"      # débito pequeño
        except Exception:
            pass
        return ""

    def make_ranked(sort_col):
        r = df.sort_values([sort_col, "Ticker"], ascending=[False, True]).reset_index(drop=True)
        r.insert(0, "Rank", range(1, len(r) + 1))
        return r

    def show_ranking(r, key_prefix, file_prefix):
        cs = [c for c in cols_show if c in r.columns]
        styler = r[cs].style.map(color_cost, subset=["Coste_Neto_%"])
        st.dataframe(styler, use_container_width=True, height=550)
        st.download_button(
            "⬇️ Descargar CSV", r.to_csv(index=False).encode("utf-8"),
            f"{file_prefix}_{ts.strftime('%Y%m%d_%H%M')}.csv", "text/csv",
            key=f"dl_{key_prefix}",
        )

    tab1, tab2, tab3, tab4 = st.tabs([
        "🥇 Ranking 1: ganancia máx. %", "⏱️ Ranking 2: ganancia anualizada",
        "🔍 Detalle", "📈 Gráficos",
    ])

    with tab1:
        st.caption("Ordenado por la ganancia máxima % (distancia del call + crédito/débito neto).")
        show_ranking(make_ranked("Ganancia_Max_%"), "r1", "collar_rank_ganancia")

    with tab2:
        st.caption("Ordenado por la ganancia máxima anualizada % (ganancia % × 365 / DTE, simple).")
        show_ranking(make_ranked("Ganancia_Anualizada_%"), "r2", "collar_rank_anualizada")

    with tab3:
        ticker_options = list(dict.fromkeys(
            df.sort_values("Ganancia_Max_%", ascending=False)["Ticker"].tolist()
        ))
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            selected = st.selectbox("Selecciona un ticker", options=ticker_options, key="detail_ticker")
        row = None
        if selected:
            tdf = df[df["Ticker"] == selected].sort_values(["Vencimiento", "Strike_Put"], ascending=[True, False])
            labels = {
                int(i): (f"{r['Vencimiento']} · Put {r['Strike_Put']:.2f} / Call {r['Strike_Call']:.2f} · "
                         f"{r['Ganancia_Max_%']:.1f}% · coste {r['Coste_Neto_%']:+.2f}%")
                for i, r in tdf.iterrows()
            }
            with dcol2:
                sel_idx = st.selectbox(
                    f"Collar ({len(tdf)} disponibles)",
                    options=list(labels.keys()),
                    format_func=lambda k: labels[k],
                    key=f"detail_combo_{selected}",
                )
            row = tdf.loc[sel_idx]

        if row is not None:
            skew_txt = f"{row['Call_Skew']:.2f} pts" if pd.notna(row.get("Call_Skew")) else "n/d"
            st.markdown(f"## {selected} — Collar {row['Vencimiento']} (DTE {row['DTE']}d)")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("### 📌 Trade Setup (100 acciones)")
                st.markdown(f"""
| Concepto | Valor |
|---|---|
| 💲 Precio de la acción | **${row['Precio']}** |
| 🛡️ Compra put | **K ${row['Strike_Put']}** ({row['Put_OTM_%']}% OTM) · mid ${row['Put_Mid']} |
| 💰 Vende call | **K ${row['Strike_Call']}** ({row['Call_OTM_%']}% OTM) · mid ${row['Call_Mid']} |
| 💵 Coste neto (mid) | **${row['Coste_Neto']}**/acción ({row['Coste_Neto_%']:+.3f}%) · positivo = débito |
| 🧮 Coste conservador (ask put / bid call) | {row['Coste_Conservador_%']:+.3f}% del precio |
| 💼 Capital neto (acciones + coste) | ${row['Capital_Neto']:,.2f} |
""")
            with col_b:
                st.markdown("### 📊 Resultado y contexto")
                st.markdown(f"""
| Concepto | Valor |
|---|---|
| 🚀 Ganancia máxima | **${row['Ganancia_Max_$']:,.2f}** ({row['Ganancia_Max_%']}%) |
| 📆 Ganancia anualizada | **{row['Ganancia_Anualizada_%']}%** |
| 🛑 Pérdida máxima | {row['Perdida_Max_%']}% del precio |
| 📉 Call skew (IV call − IV put) | {skew_txt} |
| 📈 IV put / IV call | {row['IV_Put_%']}% / {row['IV_Call_%']}% |
| 📐 Delta put / call | {row['Delta_Put']} / {row['Delta_Call']} |
| 💧 OI put / call | {row['OI_Put']:,} / {row['OI_Call']:,} |
| 〰️ Spread put / call | {row['Spread_Put_%']}% / {row['Spread_Call_%']}% |
""")
            st.plotly_chart(plot_payoff(row), use_container_width=True)
            st.info(
                f"Si la acción cierra por encima de **${row['Strike_Call']}**, ganas como máximo "
                f"**${row['Ganancia_Max_$']:,.2f}** por contrato; si cae por debajo de "
                f"**${row['Strike_Put']}**, la pérdida queda acotada a "
                f"**{row['Perdida_Max_%']}%** del precio actual. "
                f"Recuerda: sin filtro de dividendos, revisa el riesgo de ejercicio anticipado "
                f"del call cerca de un ex-dividend."
            )

    with tab4:
        fig_scatter = px.scatter(
            df, x="DTE", y="Ganancia_Max_%", color="Ticker",
            size=df["Ganancia_Anualizada_%"].clip(lower=1),
            title="Ganancia máxima % vs DTE (tamaño = ganancia anualizada)",
            template="plotly_dark", height=450,
            hover_data=["Ticker", "Vencimiento", "Strike_Put", "Strike_Call",
                        "Coste_Neto_%", "Ganancia_Anualizada_%", "Call_Skew"],
        )
        fig_scatter.update_layout(showlegend=n_tickers <= 15)
        st.plotly_chart(fig_scatter, use_container_width=True)

        best_per_ticker = (
            df.groupby("Ticker", as_index=False)
              .agg(Gan_max=("Ganancia_Max_%", "max"), Collares=("Strike_Call", "count"))
              .sort_values("Gan_max", ascending=False)
              .head(20)
        )
        fig_bar = px.bar(
            best_per_ticker, x="Ticker", y="Gan_max", color="Gan_max", text="Collares",
            color_continuous_scale=["#dd6974", "#e8af34", "#6daa45"],
            title="Top 20 tickers — mejor ganancia máx. % (el número = collares válidos del ticker)",
            template="plotly_dark", height=400,
            labels={"Gan_max": "Ganancia máx. (%)"},
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        fig_skew = px.scatter(
            df[df["Call_Skew"].notna()], x="Call_Skew", y="Coste_Neto_%", color="Ticker",
            title="Call skew vs coste neto % (negativo = crédito)",
            template="plotly_dark", height=400,
            hover_data=["Ticker", "Vencimiento", "Strike_Put", "Strike_Call"],
        )
        fig_skew.update_layout(showlegend=n_tickers <= 15)
        st.plotly_chart(fig_skew, use_container_width=True)


if __name__ == "__main__":
    main()
