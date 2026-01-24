# pages/ZScore_MACDV_Analyzer.py

import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats
import warnings

# Importación de seguridad (ajusta según tu estructura de carpetas)
try:
    from utils.utils import check_password
except ImportError:
    def check_password(): return True

warnings.filterwarnings('ignore')

# 1. NUEVA PALETA DE COLORES DE ALTO CONTRASTE
REGIME_COLORS_ZSCORE = {
    'SOBRECOMPRA_FUERTE': '#FF0000', # Rojo Puro (Peligro/Extremo)
    'SOBRECOMPRA_DEBIL':  '#FFA500', # Naranja (Agotamiento)
    'ALCISTA':            '#00FF00', # Verde Lima (Saludable)
    'RANGO':              '#808080', # Gris (Neutral)
    'BAJISTA':            '#4169E1', # Azul Real (Debilidad)
    'SOBREVENTA_DEBIL':   '#DA70D6', # Orquídea (Rebote potencial)
    'SOBREVENTA_FUERTE':  '#8B008B'  # Magenta Oscuro (Pánico)
}

# ============================================================================
# FUNCIONES DE CÁLCULO
# ============================================================================

def calculate_sma(data, period):
    return data.rolling(window=period).mean()

def calculate_donchian(df, period=20):
    upper = df['High'].rolling(window=period).max()
    lower = df['Low'].rolling(window=period).min()
    middle = (upper + lower) / 2
    return upper, middle, lower

def calculate_z_score_macdv(df, fast=12, slow=26, signal=9, z_window=20):
    h_l = df['High'] - df['Low']
    h_pc = np.abs(df['High'] - df['Close'].shift(1))
    l_pc = np.abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    atr = tr.rolling(window=slow).mean()
    
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd_v = ((ema_fast - ema_slow) / atr) * 100
    
    signal_line = macd_v.ewm(span=signal, adjust=False).mean()
    momentum_dist = macd_v - signal_line
    
    m_mean = momentum_dist.rolling(window=z_window).mean()
    m_std = momentum_dist.rolling(window=z_window).std()
    z_score = (momentum_dist - m_mean) / m_std
    
    def rolling_kurtosis(series, window):
        return series.rolling(window).apply(lambda x: stats.kurtosis(x, fisher=True, nan_policy='omit'), raw=False)
    
    kurtosis = rolling_kurtosis(momentum_dist, z_window)
    kurtosis_factor = 1 + (kurtosis / 10).clip(-0.5, 0.5)
    
    df['MACD_V'] = macd_v
    df['MACD_V_Signal'] = signal_line
    df['Z_Score_Adjusted'] = z_score / kurtosis_factor
    df['Kurtosis'] = kurtosis
    return df

# ============================================================================
# LÓGICA DE TRANSICIÓN OPTIMIZADA
# ============================================================================

def classify_regime_zscore(df):
    regimes = []
    # Usamos una media móvil del diferencial para suavizar transiciones
    macd_v_trend = df['MACD_V'].diff().rolling(window=3).mean() 
    
    for idx, row in df.iterrows():
        z = row['Z_Score_Adjusted']
        price = row['Close']
        sma50 = row['SMA_50']
        diff = macd_v_trend.loc[idx]
        
        if pd.isna(z) or pd.isna(price) or pd.isna(sma50):
            regimes.append('RANGO'); continue

        # Zona Neutral
        if -0.75 <= z <= 0.75:
            regimes.append('RANGO'); continue
        
        # LÓGICA ALCISTA
        if price > sma50:
            if z > 2.0:
                # Solo FUERTE si el momentum acelera (diff > 0)
                regime = 'SOBRECOMPRA_FUERTE' if diff > 0 else 'SOBRECOMPRA_DEBIL'
            elif 1.5 < z <= 2.0:
                # Siempre DEBIL en esta zona de enfriamiento
                regime = 'SOBRECOMPRA_DEBIL'
            elif 0.75 < z <= 1.5:
                regime = 'ALCISTA'
            else:
                regime = 'RANGO'
        
        # LÓGICA BAJISTA
        elif price < sma50:
            if z < -2.0:
                regime = 'SOBREVENTA_FUERTE' if diff < 0 else 'SOBREVENTA_DEBIL'
            elif -2.0 <= z < -1.5:
                regime = 'SOBREVENTA_DEBIL'
            elif -1.5 <= z < -0.75:
                regime = 'BAJISTA'
            else:
                regime = 'RANGO'
        else:
            regime = 'RANGO'
            
        regimes.append(regime)
    return regimes

# ============================================================================
# DESCARGA DE DATOS (Mantiene inputs originales)
# ============================================================================

@st.cache_data(ttl=3600)
def download_weekly_data(ticker, start_date=None, years_back=None):
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365*(years_back if years_back else 7))).strftime('%Y-%m-%d')
    else:
        start_date = start_date.strftime('%Y-%m-%d')
        
    data = yf.download(ticker, start=start_date, interval='1wk', progress=False)
    if data.empty: return None
    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
    
    df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['SMA_20'] = calculate_sma(df['Close'], 20)
    df['SMA_50'] = calculate_sma(df['Close'], 50)
    df['Donchian_Upper'], df['Donchian_Middle'], df['Donchian_Lower'] = calculate_donchian(df, period=20)
    df = calculate_z_score_macdv(df)
    return df.dropna()

# ============================================================================
# INTERFAZ Y DASHBOARD
# ============================================================================

st.set_page_config(layout="wide", page_title="Z-Score MACD-V Analyzer")
st.title("📊 Z-Score MACD-V Analyzer")
st.markdown("### *Optimización de Transiciones y Contraste Visual - v3.5*")

# Controles en la página principal
col_cfg1, col_cfg2, col_cfg3, col_cfg4 = st.columns([2, 2, 2, 2])

with col_cfg1:
    ticker = st.text_input("🎯 Ticker Symbol", value="AAPL").upper()
with col_cfg2:
    lookback_months = st.slider("📅 Meses a visualizar", 1, 24, 6)
with col_cfg3:
    years_options = {"3 años": 3, "5 años": 5, "7 años": 7, "10 años": 10}
    years_label = st.selectbox("📊 Histórico de descarga", list(years_options.keys()), index=2)
    years_back = years_options[years_label]
with col_cfg4:
    use_custom = st.checkbox("📆 Fecha custom", value=False)
    start_custom = st.date_input("Inicio", value=datetime(2018, 1, 1)) if use_custom else None

st.markdown("---")

if st.button("🚀 ANALIZAR", type="primary", use_container_width=True):
    with st.spinner(f"Analizando {ticker}..."):
        df = download_weekly_data(ticker, start_date=start_custom if use_custom else None, years_back=years_back)
        if df is not None:
            df['Regime_ZScore'] = classify_regime_zscore(df)
            st.session_state['df'] = df
            st.session_state['ticker_name'] = ticker

if 'df' in st.session_state:
    df = st.session_state['df']
    tk = st.session_state['ticker_name']
    df_recent = df.tail(int(lookback_months * 4.33))
    current = df.iloc[-1]

    # Métricas
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("RÉGIMEN", current['Regime_ZScore'])
    m2.metric("PRECIO", f"${current['Close']:.2f}")
    m3.metric("Z-SCORE ADJ", f"{current['Z_Score_Adjusted']:.2f}σ")
    m4.metric("KURTOSIS", f"{current['Kurtosis']:.2f}")
    m5.metric("FECHA", df.index[-1].strftime('%Y-%m-%d'))

    # Gráficos
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 16), facecolor='#0E1117')
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.3)

    # 1. Precio con Regímenes
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#1A1D29')
    ax1.plot(df_recent.index, df_recent['Close'], color='white', alpha=0.3)
    ax1.plot(df_recent.index, df_recent['SMA_50'], color='#BD93F9', label='SMA 50', linewidth=2)
    
    for regime, color in REGIME_COLORS_ZSCORE.items():
        mask = df_recent['Regime_ZScore'] == regime
        if mask.any():
            ax1.scatter(df_recent[mask].index, df_recent[mask]['Close'], c=color, label=regime, s=100, edgecolors='black', zorder=5)
    ax1.legend(loc='upper left', ncol=4)
    ax1.set_title(f"Análisis {tk} - Tendencia y Extremos", fontsize=18)

    # 2. Z-Score con zonas
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_facecolor('#1A1D29')
    ax2.plot(df_recent.index, df_recent['Z_Score_Adjusted'], color='cyan', linewidth=2)
    ax2.axhline(2.0, color='red', linestyle='--')
    ax2.axhline(1.5, color='orange', linestyle=':')
    ax2.axhline(-1.5, color='orchid', linestyle=':')
    ax2.axhline(-2.0, color='purple', linestyle='--')
    ax2.set_ylabel("Z-Score Adj")

    # 3. Timeline de Estados
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.set_facecolor('#1A1D29')
    order = list(REGIME_COLORS_ZSCORE.keys())
    y_vals = [order.index(r) for r in df_recent['Regime_ZScore']]
    ax3.scatter(df_recent.index, y_vals, c=[REGIME_COLORS_ZSCORE[r] for r in df_recent['Regime_ZScore']], s=80, marker='s')
    ax3.set_yticks(range(len(order)))
    ax3.set_yticklabels(order)
    ax3.grid(True, alpha=0.1)

    st.pyplot(fig)
