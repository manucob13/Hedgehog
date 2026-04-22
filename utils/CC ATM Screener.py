import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import warnings
import plotly.graph_objects as go
import plotly.express as px
from utils.utils import check_password

warnings.filterwarnings('ignore')

# ============================================================
# UNIVERSO ETFs - Alta liquidez en opciones, sin dividendos
# ============================================================
ETF_UNIVERSE = {
    # Tier 1 - Máxima liquidez
    "SPY":  "SPDR S&P 500 ETF",
    "QQQ":  "Invesco Nasdaq-100 ETF",
    "IWM":  "iShares Russell 2000 ETF",
    # Tier 2 - Muy líquidos
    "GLD":  "SPDR Gold Shares",
    "SLV":  "iShares Silver Trust",
    "TLT":  "iShares 20+ Year Treasury Bond",
    "HYG":  "iShares iBoxx $ High Yield Corp Bond",
    "XLF":  "Financial Select Sector SPDR",
    "XLE":  "Energy Select Sector SPDR",
    "EEM":  "iShares MSCI Emerging Markets",
    "SMH":  "VanEck Semiconductor ETF",
    "XBI":  "SPDR S&P Biotech ETF",
    "ARKK": "ARK Innovation ETF",
    "GDX":  "VanEck Gold Miners ETF",
    "KWEB": "KraneShares CSI China Internet",
    # Tier 3 - Líquidos con opciones semanales
    "SOXX": "iShares Semiconductor ETF",
    "XOP":  "SPDR S&P Oil & Gas Exploration",
    "GDXJ": "VanEck Junior Gold Miners",
    "IAU":  "iShares Gold Trust",
    "USO":  "United States Oil Fund",
    "IBIT": "iShares Bitcoin Trust",
    "XLK":  "Technology Select Sector SPDR",
    "XLV":  "Health Care Select Sector SPDR",
    "XLU":  "Utilities Select Sector SPDR",
    "XLI":  "Industrial Select Sector SPDR",
    "XLP":  "Consumer Staples Select Sector SPDR",
    "XLY":  "Consumer Discretionary SPDR",
    "VXX":  "iPath Series B S&P 500 VIX",
    "DIA":  "SPDR Dow Jones Industrial Average",
    "EWJ":  "iShares MSCI Japan ETF",
    "FXI":  "iShares China Large-Cap ETF",
}

_yfinance_lock = Lock()

# ============================================================
# FUNCIONES DE CÁLCULO
# ============================================================

def download_daily_data(ticker, period="1y"):
    """Descarga datos diarios para un ticker"""
    try:
        end = datetime.now() + timedelta(days=1)
        period_days = {'3mo': 90, '6mo': 180, '1y': 365, '2y': 730}
        days = period_days.get(period, 365)
        start = end - timedelta(days=days)
        with _yfinance_lock:
            data = yf.download(
                ticker, start=start, end=end,
                interval="1d", auto_adjust=False,
                multi_level_index=False, progress=False
            )
        if data is None or data.empty or len(data) < 20:
            return None
        data.index = pd.to_datetime(data.index)
        return data
    except:
        return None


def calculate_efficiency_ratio(prices, period=21):
    """ER = movimiento_neto / suma_movimientos. Bajo = lateral."""
    try:
        if len(prices) < period + 1:
            return None
        recent = prices.iloc[-period:]
        net_move = abs(recent.iloc[-1] - recent.iloc[0])
        total_path = sum(abs(recent.iloc[i] - recent.iloc[i-1]) for i in range(1, len(recent)))
        if total_path == 0:
            return None
        return round(float(net_move / total_path), 4)
    except:
        return None


def calculate_fdi(prices, period=50):
    """Fractal Dimension Index. Alto (>1.4) = lateral/caótico."""
    try:
        if len(prices) < period + 1:
            return None
        p = prices.iloc[-period:].values.astype(float)
        n = len(p)
        hi = max(p)
        lo = min(p)
        if hi == lo:
            return None
        total_len = sum(np.sqrt(1 + ((p[i] - p[i-1]) / ((hi - lo) / n)) ** 2) for i in range(1, n))
        fdi = 1 + (np.log(total_len) + np.log(2)) / np.log(2 * n)
        return round(float(fdi), 4)
    except:
        return None


def calculate_permutation_entropy(prices, window, m=3, tau=1):
    """PE en una ventana. Más alto = más caótico/lateral."""
    try:
        p = prices.iloc[-window:].values.astype(float)
        if len(p) < m + (m-1)*tau:
            return None
        from itertools import permutations
        import math
        perms = {}
        n = len(p) - (m - 1) * tau
        for i in range(n):
            pattern = tuple(np.argsort([p[i + j*tau] for j in range(m)]))
            perms[pattern] = perms.get(pattern, 0) + 1
        total = sum(perms.values())
        pe = -sum((c/total) * math.log2(c/total) for c in perms.values() if c > 0)
        max_pe = math.log2(math.factorial(m))
        return round(pe / max_pe, 4) if max_pe > 0 else None
    except:
        return None


def calculate_pe_multiscale(prices):
    """PE en 3 ventanas: 10, 30, 90 días. Divergencia baja = régimen estable."""
    try:
        pe10 = calculate_permutation_entropy(prices, window=min(10, len(prices)))
        pe30 = calculate_permutation_entropy(prices, window=min(30, len(prices)))
        pe90 = calculate_permutation_entropy(prices, window=min(90, len(prices)))
        if any(x is None for x in [pe10, pe30, pe90]):
            return None, None, None, None
        divergence = round(float(np.std([pe10, pe30, pe90])), 4)
        return pe10, pe30, pe90, divergence
    except:
        return None, None, None, None


def calculate_rvr(prices):
    """Realized Volatility Ratio = HV5 / HV20. Cerca de 1.0 = estable."""
    try:
        if len(prices) < 22:
            return None
        rets = np.log(prices / prices.shift(1)).dropna()
        hv5  = float(rets.iloc[-5:].std()  * np.sqrt(252) * 100)
        hv20 = float(rets.iloc[-20:].std() * np.sqrt(252) * 100)
        if hv20 == 0:
            return None
        return round(hv5 / hv20, 4)
    except:
        return None


def calculate_r2(prices, period=60):
    """R² de regresión lineal. Bajo = lateral (no hay tendencia lineal)."""
    try:
        if len(prices) < period:
            return None
        y = prices.iloc[-period:].values.astype(float)
        x = np.arange(len(y))
        coeffs = np.polyfit(x, y, 1)
        y_pred = np.polyval(coeffs, x)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return round(float(r2), 4)
    except:
        return None


def score_er(er):
    if er is None: return 0
    if er < 0.20: return 100
    if er < 0.30: return 80
    if er < 0.40: return 60
    if er < 0.50: return 30
    return 0

def score_fdi(fdi):
    if fdi is None: return 0
    if fdi > 1.45: return 100
    if fdi > 1.35: return 80
    if fdi > 1.25: return 60
    if fdi > 1.15: return 30
    return 0

def score_pe_div(div):
    if div is None: return 0
    if div < 0.03: return 100
    if div < 0.05: return 80
    if div < 0.08: return 60
    if div < 0.12: return 30
    return 0

def score_rvr(rvr):
    if rvr is None: return 0
    if 0.85 <= rvr <= 1.15: return 100
    if 0.70 <= rvr <= 1.30: return 70
    if 0.55 <= rvr <= 1.50: return 40
    return 0

def score_r2(r2):
    if r2 is None: return 0
    if r2 < 0.20: return 100
    if r2 < 0.35: return 80
    if r2 < 0.50: return 60
    if r2 < 0.65: return 30
    return 0


def calculate_score(er, fdi, pe_div, rvr, r2):
    """Score ponderado 0-100."""
    s_er  = score_er(er)   * 0.25
    s_fdi = score_fdi(fdi) * 0.25
    s_pe  = score_pe_div(pe_div) * 0.25
    s_rvr = score_rvr(rvr) * 0.15
    s_r2  = score_r2(r2)   * 0.10
    return round(s_er + s_fdi + s_pe + s_rvr + s_r2, 1)


def get_semaphore(score):
    if score >= 70: return "🟢"
    if score >= 50: return "🟡"
    return "🔴"


def get_atm_iv(ticker, current_price):
    """Obtiene IV ATM del vencimiento mensual más próximo."""
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return None, None

        today = datetime.now().date()

        def is_third_friday(d):
            return d.weekday() == 4 and 15 <= d.day <= 21

        monthly = []
        for exp_str in expirations:
            try:
                exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
                if exp_date > today and is_third_friday(exp_date):
                    monthly.append((exp_date, exp_str))
            except:
                continue

        if not monthly:
            monthly = [(datetime.strptime(expirations[0], '%Y-%m-%d').date(), expirations[0])]

        monthly.sort(key=lambda x: x[0])
        first_date, first_str = monthly[0]
        days_to_exp = (first_date - today).days

        if days_to_exp < 7 and len(monthly) >= 2:
            first_date, first_str = monthly[1]
            days_to_exp = (first_date - today).days

        chain = stock.option_chain(first_str)
        calls = chain.calls

        calls = calls[calls['strike'].notna() & calls['impliedVolatility'].notna()]
        if calls.empty:
            return None, days_to_exp

        calls['dist'] = abs(calls['strike'] - current_price)
        atm_call = calls.loc[calls['dist'].idxmin()]
        iv_atm = round(float(atm_call['impliedVolatility']) * 100, 1)
        return iv_atm, days_to_exp
    except:
        return None, None


def analyze_etf(ticker):
    """Analiza un ETF individual."""
    try:
        data = download_daily_data(ticker, period="1y")
        if data is None or len(data) < 100:
            return None

        close = data['Close']
        current_price = float(close.iloc[-1])

        er   = calculate_efficiency_ratio(close, period=21)
        fdi  = calculate_fdi(close, period=50)
        pe10, pe30, pe90, pe_div = calculate_pe_multiscale(close)
        rvr  = calculate_rvr(close)
        r2   = calculate_r2(close, period=60)

        score = calculate_score(er, fdi, pe_div, rvr, r2)
        semaphore = get_semaphore(score)

        iv_atm, dte = get_atm_iv(ticker, current_price)

        # Prima estimada ATM (simplificada: IV * precio * sqrt(DTE/365))
        prima_est = None
        if iv_atm is not None and dte is not None and dte > 0:
            prima_est = round(current_price * (iv_atm/100) * np.sqrt(dte/365), 2)

        return {
            'Semáforo':   semaphore,
            'Ticker':     ticker,
            'Nombre':     ETF_UNIVERSE.get(ticker, ticker),
            'Precio':     round(current_price, 2),
            'Score':      score,
            'ER_21d':     er,
            'FDI_50d':    fdi,
            'PE_10d':     pe10,
            'PE_30d':     pe30,
            'PE_90d':     pe90,
            'PE_Div':     pe_div,
            'RVR':        rvr,
            'R2_60d':     r2,
            'IV_ATM_%':   iv_atm,
            'DTE':        dte,
            'Prima_Est':  prima_est,
        }
    except:
        return None


def run_screener(tickers, progress_bar, status_text):
    results = []
    total = len(tickers)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(analyze_etf, t): t for t in tickers}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            progress_bar.progress(completed / total)
            status_text.text(f"🔍 Analizando {completed}/{total}...")
            result = future.result()
            if result is not None:
                results.append(result)
    status_text.text(f"✅ Completado: {len(results)} ETFs analizados")
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    df = df.sort_values('Score', ascending=False).reset_index(drop=True)
    return df


def plot_radar(row):
    """Gráfico radar del scoring para un ETF."""
    categories = ['ER (lateral)', 'FDI (fractal)', 'PE Estabilidad', 'RVR (vol ratio)', 'R² (no trend)']
    values = [
        score_er(row.get('ER_21d')),
        score_fdi(row.get('FDI_50d')),
        score_pe_div(row.get('PE_Div')),
        score_rvr(row.get('RVR')),
        score_r2(row.get('R2_60d')),
    ]
    values_norm = [v / 100 for v in values]
    values_norm += values_norm[:1]
    categories_plot = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_norm, theta=categories_plot,
        fill='toself', name=row['Ticker'],
        line=dict(color='#4f98a3', width=2),
        fillcolor='rgba(79,152,163,0.25)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickformat='.0%', gridcolor='#393836'),
            angularaxis=dict(gridcolor='#393836'),
            bgcolor='#1c1b19'
        ),
        showlegend=False,
        template='plotly_dark',
        height=320,
        margin=dict(l=40, r=40, t=40, b=40),
        title=dict(text=f"{row['Ticker']} — Score {row['Score']}/100", font=dict(size=14))
    )
    return fig


def plot_price_chart(ticker, ticker_name):
    """Gráfico de velas + HV5 overlay."""
    try:
        data = download_daily_data(ticker, period="6mo")
        if data is None or data.empty:
            return None
        rets = np.log(data['Close'] / data['Close'].shift(1))
        hv20 = rets.rolling(20).std() * np.sqrt(252) * 100

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'], high=data['High'],
            low=data['Low'], close=data['Close'],
            name=ticker,
            increasing_line_color='#6daa45',
            decreasing_line_color='#dd6974'
        ))
        fig.add_trace(go.Scatter(
            x=data.index, y=hv20,
            mode='lines', name='HV20 (%)',
            line=dict(color='#e8af34', width=1.5, dash='dot'),
            yaxis='y2'
        ))
        fig.update_layout(
            title=f'{ticker} — {ticker_name}',
            template='plotly_dark',
            height=420,
            xaxis_rangeslider_visible=False,
            yaxis=dict(title='Precio ($)', side='left'),
            yaxis2=dict(title='HV20 (%)', overlaying='y', side='right', showgrid=False),
            hovermode='x unified',
            legend=dict(orientation='h', y=1.02)
        )
        return fig
    except:
        return None


# ============================================================
# INTERFAZ PRINCIPAL
# ============================================================
def main():
    st.set_page_config(
        page_title="CC ATM Screener",
        page_icon="📊",
        layout="wide"
    )

    if not check_password():
        st.stop()

    st.title("📊 CC ATM Screener — ETFs Laterales")
    st.markdown("**Detecta ETFs en régimen lateral/baja subida para Covered Calls ATM · Alta liquidez en opciones**")
    st.markdown("---")

    # ============= CONFIGURACIÓN =============
    st.markdown("### ⚙️ Configuración")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🎯 Score mínimo para candidato CC**")
        min_score = st.slider("Score mínimo (0–100)", 0, 100, 60, step=5,
            help="Score ponderado: ER 25% + FDI 25% + PE 25% + RVR 15% + R² 10%")
        st.markdown("**📋 ETFs a analizar**")
        all_tickers = list(ETF_UNIVERSE.keys())
        selected_tickers = st.multiselect(
            "Selecciona ETFs",
            options=all_tickers,
            default=all_tickers,
            format_func=lambda x: f"{x} — {ETF_UNIVERSE[x]}"
        )

    with col2:
        st.markdown("**📐 Filtros de métricas**")
        max_er = st.slider("ER máximo (lateralidad)", 0.10, 0.80, 0.45, step=0.05,
            help="Efficiency Ratio < 0.3 = muy lateral")
        min_fdi = st.slider("FDI mínimo (fractalidad)", 1.0, 1.6, 1.2, step=0.05,
            help="FDI > 1.4 = muy lateral/fractal")
        max_r2 = st.slider("R² máximo (sin tendencia)", 0.1, 0.9, 0.55, step=0.05,
            help="R² < 0.3 = sin tendencia lineal clara")

    with col3:
        st.markdown("**⚡ Filtros adicionales**")
        max_rvr = st.slider("RVR máximo (estabilidad vol)", 0.5, 2.5, 1.5, step=0.1,
            help="HV5/HV20 — cerca de 1.0 es estable")
        min_rvr = st.slider("RVR mínimo", 0.3, 1.0, 0.5, step=0.05)
        max_pe_div = st.slider("PE Divergencia máxima", 0.01, 0.30, 0.12, step=0.01,
            help="Divergencia entre PE10/PE30/PE90 — baja = régimen estable")
        apply_iv = st.checkbox("Solo ETFs con IV ATM disponible", value=False)

    st.markdown("---")

    # ============= RESUMEN FILTROS =============
    with st.expander("📋 Resumen de filtros activos", expanded=False):
        fc1, fc2 = st.columns(2)
        with fc1:
            st.info(f"Score mínimo: **{min_score}**")
            st.info(f"ER máximo: **{max_er}**")
            st.info(f"FDI mínimo: **{min_fdi}**")
        with fc2:
            st.info(f"R² máximo: **{max_r2}**")
            st.info(f"RVR rango: **{min_rvr} — {max_rvr}**")
            st.info(f"PE Divergencia máx: **{max_pe_div}**")

    st.markdown("---")

    # ============= ESCANEO =============
    st.markdown("### 🚀 Ejecutar Escaneo")
    scan_btn = st.button("🚀 INICIAR ESCANEO CC ATM",
                         type="primary", use_container_width=True,
                         disabled=len(selected_tickers) == 0)

    if scan_btn:
        progress_bar = st.progress(0)
        status_text  = st.empty()
        df_raw = run_screener(selected_tickers, progress_bar, status_text)
        progress_bar.empty()

        if not df_raw.empty:
            # Aplicar filtros adicionales
            df_filtered = df_raw.copy()
            df_filtered = df_filtered[df_filtered['Score'] >= min_score]
            df_filtered = df_filtered[df_filtered['ER_21d'].apply(lambda x: x is not None and x <= max_er)]
            df_filtered = df_filtered[df_filtered['FDI_50d'].apply(lambda x: x is not None and x >= min_fdi)]
            df_filtered = df_filtered[df_filtered['R2_60d'].apply(lambda x: x is not None and x <= max_r2)]
            df_filtered = df_filtered[df_filtered['RVR'].apply(lambda x: x is not None and min_rvr <= x <= max_rvr)]
            df_filtered = df_filtered[df_filtered['PE_Div'].apply(lambda x: x is not None and x <= max_pe_div)]
            if apply_iv:
                df_filtered = df_filtered[df_filtered['IV_ATM_%'].notna()]

            st.session_state['scan_results_raw']      = df_raw
            st.session_state['scan_results_filtered'] = df_filtered
            st.session_state['scan_timestamp']        = datetime.now()

            st.success(f"✅ Escaneo completado — **{len(df_raw)}** ETFs analizados | **{len(df_filtered)}** candidatos CC ATM")
        else:
            st.warning("⚠️ No se obtuvieron datos. Verifica la conexión.")

    st.markdown("---")

    # ============= RESULTADOS =============
    st.markdown("### 📈 Resultados")

    if 'scan_results_filtered' in st.session_state:
        df_show = st.session_state['scan_results_filtered']
        df_raw  = st.session_state['scan_results_raw']
        ts      = st.session_state['scan_timestamp']

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("🟢 Candidatos CC", len(df_show[df_show['Score'] >= 70]))
        k2.metric("🟡 Candidatos Moderados", len(df_show[(df_show['Score'] >= 50) & (df_show['Score'] < 70)]))
        k3.metric("📊 Total analizados", len(df_raw))
        k4.metric("🕐 Último escaneo", ts.strftime("%H:%M:%S"))

        st.markdown("---")

        # Tabla principal
        tab1, tab2, tab3 = st.tabs(["📊 Tabla Completa", "🎯 Top Candidatos", "📉 Gráficos"])

        with tab1:
            df_display = df_show.copy()
            # Formato visual
            def color_score(val):
                if val >= 70: return 'background-color: #1e3a1e; color: #6daa45'
                if val >= 50: return 'background-color: #3a3a1e; color: #e8af34'
                return 'background-color: #3a1e1e; color: #dd6974'

            st.dataframe(
                df_display[[
                    'Semáforo','Ticker','Nombre','Precio',
                    'Score','ER_21d','FDI_50d','PE_Div',
                    'RVR','R2_60d','IV_ATM_%','DTE','Prima_Est'
                ]].style.applymap(color_score, subset=['Score']),
                use_container_width=True,
                height=500
            )

            # Descarga CSV
            csv = df_display.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Descargar CSV", csv, "cc_atm_screener.csv", "text/csv")

        with tab2:
            top = df_show[df_show['Score'] >= 70].head(10)
            if top.empty:
                st.info("No hay candidatos con score ≥ 70. Reduce el filtro de score mínimo.")
            else:
                for _, row in top.iterrows():
                    with st.expander(f"{row['Semáforo']} **{row['Ticker']}** — {row['Nombre']} | Score: {row['Score']} | IV ATM: {row['IV_ATM_%']}%"):
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            st.markdown(f"""
| Métrica | Valor |
|---|---|
| 💰 Precio | **${row['Precio']}** |
| 🎯 Score | **{row['Score']}/100** |
| 📐 ER 21d | `{row['ER_21d']}` |
| 🌀 FDI 50d | `{row['FDI_50d']}` |
| 🔀 PE Div | `{row['PE_Div']}` |
| ⚡ RVR | `{row['RVR']}` |
| 📏 R² 60d | `{row['R2_60d']}` |
| 📅 DTE | **{row['DTE']} días** |
| 📊 IV ATM | **{row['IV_ATM_%']}%** |
| 💵 Prima Est. | **${row['Prima_Est']}** |
""")
                        with c2:
                            fig_radar = plot_radar(row)
                            st.plotly_chart(fig_radar, use_container_width=True)

        with tab3:
            if df_show.empty:
                st.info("Ejecuta el escaneo primero.")
            else:
                selected_chart = st.selectbox(
                    "Selecciona ETF para ver gráfico de precios",
                    options=df_show['Ticker'].tolist()
                )
                if selected_chart:
                    row = df_show[df_show['Ticker'] == selected_chart].iloc[0]
                    fig_price = plot_price_chart(selected_chart, row['Nombre'])
                    if fig_price:
                        st.plotly_chart(fig_price, use_container_width=True)
                    else:
                        st.warning("No se pudo cargar el gráfico de precios.")

                # Score bar chart
                fig_scores = px.bar(
                    df_show.sort_values('Score', ascending=True).tail(20),
                    x='Score', y='Ticker', orientation='h',
                    color='Score',
                    color_continuous_scale=['#dd6974','#e8af34','#6daa45'],
                    range_color=[0, 100],
                    title='Scores de lateralidad — Top 20',
                    template='plotly_dark',
                    height=500,
                    labels={'Score': 'Score CC ATM', 'Ticker': ''}
                )
                fig_scores.add_vline(x=70, line_dash='dash', line_color='#6daa45',
                                     annotation_text="Umbral óptimo (70)")
                fig_scores.add_vline(x=50, line_dash='dash', line_color='#e8af34',
                                     annotation_text="Umbral moderado (50)")
                st.plotly_chart(fig_scores, use_container_width=True)
    else:
        st.info("👆 Configura los parámetros y pulsa **INICIAR ESCANEO** para analizar los ETFs.")

    # ============= LEYENDA =============
    st.markdown("---")
    with st.expander("ℹ️ Guía de métricas y scoring", expanded=False):
        st.markdown("""
### Sistema de Scoring para CC ATM (0–100)

| Métrica | Peso | Descripción | Ideal para CC |
|---|---|---|---|
| **ER (Efficiency Ratio)** | 25% | Movimiento neto / suma movimientos. Bajo = lateral | ER < 0.30 = 100 pts |
| **FDI (Fractal Dimension)** | 25% | Dimensión fractal del precio. Alto = caótico/lateral | FDI > 1.45 = 100 pts |
| **PE Divergencia** | 25% | Divergencia entre PE de 10/30/90 días. Baja = régimen estable | Div < 0.03 = 100 pts |
| **RVR (Vol Ratio)** | 15% | HV5/HV20. Cerca de 1.0 = vol estable, sin expansión | 0.85–1.15 = 100 pts |
| **R² (60d)** | 10% | R² regresión lineal. Bajo = sin tendencia clara | R² < 0.20 = 100 pts |

### Semáforo
- 🟢 **Score ≥ 70** → Candidato óptimo para CC ATM
- 🟡 **Score 50–69** → Candidato moderado, monitorizar
- 🔴 **Score < 50** → No recomendado actualmente

### Columnas de Opciones
- **IV ATM %**: Volatilidad implícita del strike ATM del próximo vencimiento mensual
- **DTE**: Días al vencimiento mensual seleccionado
- **Prima Est.**: Estimación prima ATM = Precio × (IV/100) × √(DTE/365)
""")


if __name__ == "__main__":
    main()
