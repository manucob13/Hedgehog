"""
Deep ITM Covered Call Screener
================================
Objetivo: encontrar covered calls deep ITM con extrínseco 0.85%-1.00% semanal.

FILTROS DUROS:
- Precio del subyacente: rango configurable
- Close > SMA30 (tendencia alcista)
- PCR < 1.0 (sesgo alcista)
- OI > 100 (liquidez mínima)
- Sin earnings en los próximos 7 días
- Extrínseco entre 0.85% y 1.00% del precio del subyacente
- Bid > 0 (opción negociable)
- Vencimiento = próximo viernes (DTE ≤ 7)

RANKING: mayor downside protection % primero
         (más deep ITM = más protección = mejor)

PRECIO DE OPCIÓN: midprice (bid+ask)/2 siempre
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
import plotly.graph_objects as go
import plotly.express as px

from utils.utils import check_password
from utils.tickers import create_tickers_universe

warnings.filterwarnings('ignore')

_lock = Lock()
_N   = NormalDist()
RISK_FREE_RATE = 0.045


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
# 1. DATOS DIARIOS
# ======================================================================

def get_daily_data(ticker):
    try:
        end   = datetime.now() + timedelta(days=1)
        start = end - timedelta(days=120)
        with _lock:
            data = yf.download(
                ticker, start=start, end=end,
                interval="1d", auto_adjust=False,
                multi_level_index=False, progress=False
            )
        if data is None or data.empty or len(data) < 35:
            return None
        data.index = pd.to_datetime(data.index)
        return data
    except Exception:
        return None


# ======================================================================
# 2. SMA30
# ======================================================================

def get_sma30(close):
    """Devuelve (sma30_valor, dist_pct, slope_positivo) o (None, None, None)."""
    try:
        if len(close) < 35:
            return None, None, None
        sma   = close.rolling(30).mean()
        sma_now  = float(sma.iloc[-1])
        sma_prev = float(sma.iloc[-6])   # 5 días atrás
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
    """Devuelve la fecha del próximo viernes (incluyendo hoy si es viernes)."""
    today = date.today()
    days_ahead = (4 - today.weekday()) % 7   # 4 = viernes
    if days_ahead == 0:
        days_ahead = 7  # si hoy es viernes, el PRÓXIMO viernes
    return today + timedelta(days=days_ahead)


def select_friday_expiration(expirations):
    """
    Busca en la lista de vencimientos el próximo viernes con DTE ≤ 7.
    Devuelve (exp_str, dte) o (None, None).
    """
    target = next_friday()
    today  = date.today()
    for exp_str in expirations:
        exp = datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp - today).days
        if exp == target and 1 <= dte <= 7:
            return exp_str, dte
    # Fallback: cualquier viernes con DTE ≤ 7
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
    """True si hay earnings en los próximos 7 días."""
    try:
        cal = stock.calendar
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
# 7. PUT/CALL RATIO
# ======================================================================

def get_pcr(stock, exp_str, current_price, range_pct=15):
    """PCR basado en OI de la cadena de opciones (±15% del precio)."""
    try:
        chain  = stock.option_chain(exp_str)
        lo     = current_price * (1 - range_pct / 100)
        hi     = current_price * (1 + range_pct / 100)

        calls  = chain.calls
        puts   = chain.puts

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
                             min_oi):
    """
    Entre todos los strikes ITM con extrínseco en el rango objetivo:
    - Calcula midprice, intrínseco, extrínseco
    - Filtra por OI > min_oi y bid > 0
    - Devuelve el strike con MAYOR downside protection
      (más deep ITM = más alejado del precio actual)
    """
    try:
        itm = calls_df[calls_df["strike"] < current_price].copy()
        if itm.empty:
            return None

        # Usar midprice siempre
        itm["bid"]  = pd.to_numeric(itm["bid"], errors="coerce").fillna(0)
        itm["ask"]  = pd.to_numeric(itm["ask"], errors="coerce").fillna(0)
        itm["mid"]  = (itm["bid"] + itm["ask"]) / 2

        # Filtro: bid > 0 y mid > 0
        itm = itm[(itm["bid"] > 0) & (itm["mid"] > 0)]
        if itm.empty:
            return None

        # Filtro OI
        itm["oi"] = pd.to_numeric(
            itm.get("openInterest", pd.Series(0, index=itm.index)),
            errors="coerce"
        ).fillna(0)
        itm = itm[itm["oi"] >= min_oi]
        if itm.empty:
            return None

        # Métricas
        itm["intrinsic"]     = current_price - itm["strike"]
        itm["extrinsic"]     = itm["mid"] - itm["intrinsic"]
        itm["extrinsic_pct"] = itm["extrinsic"] / current_price * 100
        itm["spread_pct"]    = (itm["ask"] - itm["bid"]) / itm["mid"] * 100
        itm["downside_prot"] = itm["intrinsic"] / current_price * 100  # = protección bajista

        # Filtro extrínseco objetivo
        candidates = itm[
            (itm["extrinsic_pct"] >= extrinsic_min_pct) &
            (itm["extrinsic_pct"] <= extrinsic_max_pct) &
            (itm["extrinsic"] > 0)
        ]

        if candidates.empty:
            return None

        # Ranking: mayor downside_prot = más deep ITM = mejor
        best = candidates.sort_values("downside_prot", ascending=False).iloc[0]

        # Delta para ese strike
        T_years = dte_calendar / 365.0
        iv      = float(best.get("impliedVolatility") or 0)
        delta   = bs_delta(current_price, float(best["strike"]), T_years, iv) if iv > 0 else None

        return {
            "strike":        float(best["strike"]),
            "mid":           round(float(best["mid"]), 2),
            "bid":           round(float(best["bid"]), 2),
            "ask":           round(float(best["ask"]), 2),
            "intrinsic":     round(float(best["intrinsic"]), 2),
            "extrinsic":     round(float(best["extrinsic"]), 2),
            "extrinsic_pct": round(float(best["extrinsic_pct"]), 3),
            "downside_prot": round(float(best["downside_prot"]), 2),
            "spread_pct":    round(float(best["spread_pct"]), 2),
            "oi":            int(best["oi"]),
            "volume":        int(pd.to_numeric(best.get("volume", 0), errors="coerce") or 0),
            "iv_pct":        round(iv * 100, 2) if iv > 0 else None,
            "delta":         delta,
        }
    except Exception:
        return None


# ======================================================================
# 9. ANÁLISIS DE UN TICKER
# ======================================================================

def analyze_ticker(ticker, params):
    try:
        # ── Datos diarios ──────────────────────────────────────────────
        data = get_daily_data(ticker)
        if data is None:
            return None

        close        = data["Close"]
        current_price = float(close.iloc[-1])

        # Filtro precio
        if not (params["min_price"] <= current_price <= params["max_price"]):
            return None

        # SMA30
        sma30, dist_sma_pct, slope_up = get_sma30(close)
        if sma30 is None:
            return None

        # FILTRO DURO: Close > SMA30
        if current_price <= sma30:
            return None

        # RV10
        rv = get_rv10(close)

        # ── Opciones ───────────────────────────────────────────────────
        stock = yf.Ticker(ticker)

        # FILTRO DURO: No earnings esta semana
        if has_earnings_this_week(stock):
            return None

        expirations = stock.options
        if not expirations:
            return None

        exp_str, dte = select_friday_expiration(expirations)
        if exp_str is None:
            return None

        # PCR
        pcr = get_pcr(stock, exp_str, current_price)

        # FILTRO DURO: PCR < 1.0 (alcista)
        if pcr is not None and pcr >= 1.0:
            return None

        # Cadena de calls
        chain = stock.option_chain(exp_str)
        calls = chain.calls
        if calls is None or calls.empty:
            return None

        # Candidato deep ITM
        candidate = find_deep_itm_candidate(
            calls, current_price, dte,
            params["extrinsic_min"],
            params["extrinsic_max"],
            params["min_oi"],
        )
        if candidate is None:
            return None

        # ── Métricas adicionales ───────────────────────────────────────
        iv_rv_ratio = (
            round(candidate["iv_pct"] / rv, 3)
            if (candidate["iv_pct"] and rv and rv > 0)
            else None
        )

        annualized = round(
            candidate["extrinsic_pct"] * (365 / dte), 1
        ) if dte > 0 else None

        breakeven = round(current_price - candidate["mid"], 2)

        # ── Resultado ──────────────────────────────────────────────────
        return {
            # Identificación
            "Ticker":          ticker,
            "Precio":          round(current_price, 2),
            "Vencimiento":     exp_str,
            "DTE":             dte,
            # Datos del strike (ranking principal)
            "Strike":          candidate["strike"],
            "Downside_Prot_%": candidate["downside_prot"],   # ← ranking principal
            "Extrínseco_%":    candidate["extrinsic_pct"],
            "Prima_Mid":       candidate["mid"],
            "Bid":             candidate["bid"],
            "Ask":             candidate["ask"],
            "Intrínseco":      candidate["intrinsic"],
            "Extrínseco_$":    candidate["extrinsic"],
            "Breakeven":       breakeven,
            # Griegos e IV
            "Delta":           candidate["delta"],
            "IV_%":            candidate["iv_pct"],
            "RV_%":            rv,
            "IV_RV":           iv_rv_ratio,
            # Retorno
            "Ret_Anualizado_%": annualized,
            # Liquidez
            "OI":              candidate["oi"],
            "Volumen":         candidate["volume"],
            "Spread_%":        candidate["spread_pct"],
            # Tendencia
            "SMA30":           sma30,
            "Dist_SMA30_%":    dist_sma_pct,
            "SMA30_Sube":      slope_up,
            # Sentimiento
            "PCR":             pcr,
        }

    except Exception:
        return None


# ======================================================================
# 10. SCREENER EN PARALELO
# ======================================================================

def run_screener(tickers, params, progress_bar, status_text):
    results = []
    total   = len(tickers)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(analyze_ticker, t, params): t for t in tickers}
        done = 0
        for future in as_completed(futures):
            done += 1
            progress_bar.progress(done / total)
            status_text.text(f"🔍 {done}/{total} — encontrados: {len(results)}")
            r = future.result()
            if r is not None:
                results.append(r)

    status_text.text(f"✅ Completado: {len(results)} candidatos")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Ranking: mayor Downside_Prot_% primero
    df = df.sort_values("Downside_Prot_%", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))

    return df


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
            low=data["Low"],   close=data["Close"],
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
    st.set_page_config(
        page_title="Deep ITM CC Screener",
        page_icon="🎯",
        layout="wide",
    )

    if not check_password():
        st.stop()

    st.title("🎯 Deep ITM Covered Call Screener")
    st.markdown(
        "**Objetivo: extrínseco 0.85%-1.00% semanal · "
        "Vencimiento próximo viernes · "
        "Ranking por mayor downside protection**"
    )
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
                f"(Russell 1000: {meta['r1000_count']:,} "
                f"+ Adicionales: {meta['extra_count']:,})"
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
            min_value=0.50, max_value=2.00,
            value=(0.85, 1.00), step=0.05,
            help="Filtro duro: solo strikes con extrínseco en este rango"
        )

        st.markdown("**💲 Precio del subyacente**")
        min_price, max_price = st.slider(
            "Rango de precio ($)",
            min_value=5, max_value=1000,
            value=(20, 500), step=5,
            help="Filtro duro: solo acciones en este rango de precio"
        )

    with c2:
        st.markdown("**💧 Liquidez mínima**")
        min_oi = st.number_input(
            "OI mínimo del strike",
            min_value=10, max_value=10000,
            value=100, step=50,
            help="Filtro duro: open interest mínimo para que el strike sea negociable"
        )

        st.markdown("**📅 Próximo viernes**")
        today    = date.today()
        friday   = next_friday()
        dte_days = (friday - today).days
        st.info(f"📅 Próximo viernes: **{friday.strftime('%d %b %Y')}** (DTE: {dte_days} días)")

    with c3:
        st.markdown("**ℹ️ Filtros activos (siempre)**")
        st.info("✅ Close > SMA30 (tendencia alcista)")
        st.info("✅ PCR < 1.0 (sesgo alcista)")
        st.info("✅ Sin earnings en los próximos 7 días")
        st.info("✅ Bid > 0 (opción negociable)")
        st.info("✅ Midprice = (Bid + Ask) / 2")

    params = {
        "extrinsic_min": extrinsic_min,
        "extrinsic_max": extrinsic_max,
        "min_price":     min_price,
        "max_price":     max_price,
        "min_oi":        min_oi,
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

    if scan_btn:
        progress_bar = st.progress(0)
        status_text  = st.empty()
        df_results   = run_screener(tickers_all, params, progress_bar, status_text)
        progress_bar.empty()

        if not df_results.empty:
            st.session_state["results"] = df_results
            st.session_state["scan_ts"] = datetime.now()
            st.success(
                f"✅ **{len(df_results)} candidatos** encontrados "
                f"sobre {len(tickers_all):,} tickers analizados"
            )
        else:
            st.warning("⚠️ Ningún ticker cumplió todos los filtros. Prueba a ampliar el rango de extrínseco u OI.")
            st.session_state.pop("results", None)

    st.divider()

    # ── Resultados ─────────────────────────────────────────────────────
    st.markdown("### 📊 Resultados")

    if "results" not in st.session_state:
        st.info("👆 Configura los parámetros y pulsa **INICIAR ESCANEO**.")
        return

    df = st.session_state["results"]
    ts = st.session_state["scan_ts"]

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📊 Candidatos totales", len(df))
    k2.metric("🛡️ Downside prot. máx.", f"{df['Downside_Prot_%'].max():.2f}%")
    k3.metric("💰 Extrínseco medio", f"{df['Extrínseco_%'].mean():.3f}%")
    k4.metric("🕐 Escaneo", ts.strftime("%H:%M:%S"))

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📋 Ranking Completo", "🔍 Detalle", "📈 Gráficos"])

    # ── Tab 1: Tabla ───────────────────────────────────────────────────
    with tab1:
        # Columnas a mostrar en orden lógico
        cols_show = [
            "Rank", "Ticker", "Precio", "Strike", "Downside_Prot_%",
            "Extrínseco_%", "Prima_Mid", "Bid", "Ask",
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
            use_container_width=True,
            height=550,
        )

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar CSV",
            csv,
            f"deep_itm_cc_{ts.strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
        )

    # ── Tab 2: Detalle ─────────────────────────────────────────────────
    with tab2:
        selected = st.selectbox(
            "Selecciona un ticker para ver el detalle",
            options=df["Ticker"].tolist(),
        )
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

    # ── Tab 3: Gráficos ────────────────────────────────────────────────
    with tab3:
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            ticker_chart = st.selectbox(
                "Ver gráfico de precio",
                options=df["Ticker"].tolist(),
                key="chart_select",
            )
            if ticker_chart:
                fig = plot_price(ticker_chart)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No se pudo cargar el gráfico.")

        with col_g2:
            # Scatter: Downside protection vs Extrínseco
            fig_scatter = px.scatter(
                df,
                x="Extrínseco_%",
                y="Downside_Prot_%",
                text="Ticker",
                color="Downside_Prot_%",
                color_continuous_scale=["#dd6974", "#e8af34", "#6daa45"],
                title="Downside Protection vs Extrínseco%",
                template="plotly_dark",
                height=420,
                labels={
                    "Extrínseco_%": "Extrínseco (%)",
                    "Downside_Prot_%": "Downside Protection (%)",
                },
            )
            fig_scatter.update_traces(textposition="top center", marker_size=10)
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Bar chart ranking
        fig_bar = px.bar(
            df.head(20),
            x="Ticker",
            y="Downside_Prot_%",
            color="Downside_Prot_%",
            color_continuous_scale=["#dd6974", "#e8af34", "#6daa45"],
            title="Top 20 — Downside Protection % (mayor = más deep ITM)",
            template="plotly_dark",
            height=400,
            labels={"Downside_Prot_%": "Downside Protection (%)"},
        )
        st.plotly_chart(fig_bar, use_container_width=True)


if __name__ == "__main__":
    main()
