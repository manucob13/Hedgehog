import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Colores de régimen para el dashboard
REGIME_COLORS_ZSCORE = {
    'SOBRECOMPRA_FUERTE': '#FF4444',
    'SOBRECOMPRA_DEBIL':  '#FF8888',
    'ALCISTA':            '#4ECDC4',
    'RANGO':              '#FFD93D',
    'BAJISTA':            '#EE5A6F',
    'SOBREVENTA_DEBIL':   '#BB77DD',
    'SOBREVENTA_FUERTE':  '#9D4EDD'
}

# ============================================================================
# FUNCIONES DE CÁLCULO TÉCNICO
# ============================================================================

def calculate_sma(data, period):
    return data.rolling(window=period).mean()

def calculate_donchian(df, period=20):
    upper = df['High'].rolling(window=period).max()
    lower = df['Low'].rolling(window=period).min()
    middle = (upper + lower) / 2
    return upper, middle, lower

def calculate_z_score_macdv(df, fast=12, slow=26, signal=9, z_window=20):
    """Calcula el Z-Score del MACD-V con ajuste por curtosis"""
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
    z_score_adjusted = z_score / kurtosis_factor
    
    df['MACD_V'] = macd_v
    df['MACD_V_Signal'] = signal_line
    df['Z_Score_Adjusted'] = z_score_adjusted
    df['Kurtosis'] = kurtosis
    
    return df

# ============================================================================
# LÓGICA DE TRANSICIÓN DE ESTADOS (OPTIMIZADA)
# ============================================================================

def classify_regime_zscore(df):
    """Clasifica el régimen de mercado asegurando transiciones lógicas"""
    regimes = []
    macd_diff = df['MACD_V'].diff()
    
    for idx, row in df.iterrows():
        z = row['Z_Score_Adjusted']
        price = row['Close']
        sma50 = row['SMA_50']
        macd_v = row['MACD_V']
        macd_signal = row['MACD_V_Signal']
        macd_v_diff = macd_diff.loc[idx]
        
        if pd.isna(z) or pd.isna(price) or pd.isna(sma50):
            regimes.append('RANGO')
            continue
        
        # Filtro de zona neutral
        if -0.75 <= z <= 0.75:
            regimes.append('RANGO')
            continue
        
        # CONTEXTO ALCISTA
        if price > sma50:
            if z > 1.5:
                # Solo es FUERTE si el Z > 2 Y el momentum sigue acelerando
                if z > 2.0 and macd_v_diff > 0 and macd_v > macd_signal:
                    regime = 'SOBRECOMPRA_FUERTE'
                # Si está en el "buffer" (1.5 a 2.0) o el momentum se frena, es DÉBIL
                else:
                    regime = 'SOBRECOMPRA_DEBIL'
            elif 0.75 < z <= 1.5:
                regime = 'ALCISTA'
            else:
                regime = 'RANGO'
                
        # CONTEXTO BAJISTA
        elif price < sma50:
            if z < -1.5:
                # Solo es FUERTE si Z < -2 Y la caída acelera
                if z < -2.0 and macd_v_diff < 0 and macd_v < macd_signal:
                    regime = 'SOBREVENTA_FUERTE'
                else:
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
# DESCARGA Y DASHBOARD
# ============================================================================

@st.cache_data(ttl=3600)
def download_weekly_data(ticker, years_back=7):
    start_date = (datetime.now() - timedelta(days=365*years_back)).strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_date, interval='1wk', progress=False)
    
    if data.empty: return None
    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
    
    df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['SMA_20'] = calculate_sma(df['Close'], 20)
    df['SMA_50'] = calculate_sma(df['Close'], 50)
    df['Donchian_Upper'], df['Donchian_Middle'], df['Donchian_Lower'] = calculate_donchian(df, period=20)
    df = calculate_z_score_macdv(df)
    return df.dropna()

def plot_zscore_dashboard(df_recent, ticker):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 16), facecolor='#0E1117')
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)
    
    # 1. PRECIO
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#1A1D29')
    ax1.plot(df_recent.index, df_recent['Close'], color='white', linewidth=1.5, alpha=0.8)
    
    for regime_name, color in REGIME_COLORS_ZSCORE.items():
        mask = df_recent['Regime_ZScore'] == regime_name
        ax1.scatter(df_recent[mask].index, df_recent[mask]['Close'], c=color, s=60, label=regime_name)

    ax1.plot(df_recent.index, df_recent['SMA_50'], color='#BD93F9', linestyle='--', label='SMA 50')
    ax1.set_title(f"Análisis de Régimen: {ticker}", fontsize=20)
    ax1.legend(loc='upper left', ncol=3)

    # 2. Z-SCORE
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_facecolor('#1A1D29')
    ax2.plot(df_recent.index, df_recent['Z_Score_Adjusted'], color='#00D9FF')
    ax2.axhline(2.0, color='red', linestyle='--')
    ax2.axhline(-2.0, color='purple', linestyle='--')
    ax2.axhline(0, color='gray', alpha=0.5)
    ax2.set_ylabel("Z-Score Adj")

    # 3. MACD-V
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.set_facecolor('#1A1D29')
    ax3.plot(df_recent.index, df_recent['MACD_V'], color='#4ECDC4', label='MACD-V')
    ax3.plot(df_recent.index, df_recent['MACD_V_Signal'], color='#FFB86C', linestyle=':')
    ax3.legend()

    # 4. REGIMEN TIMELINE
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.set_facecolor('#1A1D29')
    regime_list = list(REGIME_COLORS_ZSCORE.keys())
    y_vals = [regime_list.index(r) for r in df_recent['Regime_ZScore']]
    ax4.scatter(df_recent.index, y_vals, c=[REGIME_COLORS_ZSCORE[r] for r in df_recent['Regime_ZScore']], s=100)
    ax4.set_yticks(range(len(regime_list)))
    ax4.set_yticklabels(regime_list)
    
    return fig

# ============================================================================
# APP PRINCIPAL
# ============================================================================

st.set_page_config(layout="wide")
st.title("📊 Z-Score MACD-V Pro")

ticker = st.sidebar.text_input("Ticker", "AAPL").upper()
lookback = st.sidebar.slider("Semanas a visualizar", 20, 200, 100)

if st.sidebar.button("Ejecutar Análisis"):
    df = download_weekly_data(ticker)
    if df is not None:
        df['Regime_ZScore'] = classify_regime_zscore(df)
        df_plot = df.tail(lookback)
        
        # Métricas
        curr = df.iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("Régimen Actual", curr['Regime_ZScore'])
        c2.metric("Z-Score Adj", f"{curr['Z_Score_Adjusted']:.2f}σ")
        c3.metric("Precio", f"${curr['Close']:.2f}")
        
        # Dashboard
        fig = plot_zscore_dashboard(df_plot, ticker)
        st.pyplot(fig)
    else:
        st.error("No se pudieron obtener datos.")
