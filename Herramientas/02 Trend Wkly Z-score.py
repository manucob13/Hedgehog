import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# PALETA DE COLORES (REGÍMENES FINALES)
# ============================================================================

REGIME_COLORS = {
    'ALCISTA_FUERTE': '#00FF00',
    'ALCISTA': '#2ECC71',
    'ALCISTA_EXTREMO_RIESGO': '#FFA500',
    'BAJISTA_FUERTE': '#4169E1',
    'BAJISTA': '#3498DB',
    'BAJISTA_EXTREMO_RIESGO': '#9B59B6',
    'RANGO': '#7F8C8D',
    'RANGO_EXTREMO': '#E74C3C'
}

# ============================================================================
# FUNCIONES DE CÁLCULO
# ============================================================================

def calculate_sma(data, period):
    return data.rolling(window=period).mean()

def calculate_z_score_macdv(df, fast=12, slow=26, signal=9, z_window=20):

    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low'] - df['Close'].shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(slow).mean()

    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()

    macd_v = ((ema_fast - ema_slow) / atr) * 100
    signal_line = macd_v.ewm(span=signal, adjust=False).mean()
    momentum = macd_v - signal_line

    mean = momentum.rolling(z_window).mean()
    std = momentum.rolling(z_window).std()
    z = (momentum - mean) / std

    kurt = momentum.rolling(z_window).apply(
        lambda x: stats.kurtosis(x, fisher=True, nan_policy='omit'),
        raw=False
    )

    adj_factor = 1 + (kurt / 10).clip(-0.5, 0.5)

    df['MACD_V'] = macd_v
    df['MACD_V_Signal'] = signal_line
    df['Z_Score_Adjusted'] = z / adj_factor
    df['Kurtosis'] = kurt

    return df

# ============================================================================
# CLASIFICACIÓN DE REGÍMENES (LÓGICA ROBUSTA)
# ============================================================================

def classify_regime(df):

    regimes = []
    macd_trend = df['MACD_V'].diff().rolling(3).mean()

    prev_regime = 'RANGO'
    confirm = 0

    for i in range(len(df)):

        price = df['Close'].iloc[i]
        sma50 = df['SMA_50'].iloc[i]
        z = df['Z_Score_Adjusted'].iloc[i]
        m_trend = macd_trend.iloc[i]

        if np.isnan(price) or np.isnan(sma50) or np.isnan(z):
            regimes.append(prev_regime)
            continue

        # 1️⃣ RÉGIMEN ESTRUCTURAL
        if price > sma50 and sma50 > df['SMA_50'].iloc[i-3]:
            structural = 'ALCISTA'
        elif price < sma50 and sma50 < df['SMA_50'].iloc[i-3]:
            structural = 'BAJISTA'
        else:
            structural = 'RANGO'

        # 2️⃣ MOMENTUM
        if m_trend > 0:
            momentum = 'ACELERANDO'
        elif m_trend < 0:
            momentum = 'DESACELERANDO'
        else:
            momentum = 'NEUTRO'

        # 3️⃣ EXTREMOS
        if z > 2:
            extreme = 'SOBRECOMPRA_FUERTE'
        elif z > 1.5:
            extreme = 'SOBRECOMPRA'
        elif z < -2:
            extreme = 'SOBREVENTA_FUERTE'
        elif z < -1.5:
            extreme = 'SOBREVENTA'
        else:
            extreme = 'NORMAL'

        # 4️⃣ COMPOSICIÓN FINAL
        if structural == 'ALCISTA':
            if extreme.startswith('SOBRECOMPRA') and momentum == 'DESACELERANDO':
                new_regime = 'ALCISTA_EXTREMO_RIESGO'
            elif momentum == 'ACELERANDO':
                new_regime = 'ALCISTA_FUERTE'
            else:
                new_regime = 'ALCISTA'

        elif structural == 'BAJISTA':
            if extreme.startswith('SOBREVENTA') and momentum == 'DESACELERANDO':
                new_regime = 'BAJISTA_EXTREMO_RIESGO'
            elif momentum == 'ACELERANDO':
                new_regime = 'BAJISTA_FUERTE'
            else:
                new_regime = 'BAJISTA'

        else:
            new_regime = 'RANGO_EXTREMO' if extreme != 'NORMAL' else 'RANGO'

        # 5️⃣ HISTERESIS (CONFIRMACIÓN)
        if new_regime != prev_regime:
            confirm += 1
            if confirm < 2:
                new_regime = prev_regime
        else:
            confirm = 0

        prev_regime = new_regime
        regimes.append(new_regime)

    return regimes

# ============================================================================
# DESCARGA DE DATOS
# ============================================================================

@st.cache_data(ttl=3600)
def download_weekly_data(ticker, years_back=7):

    start = (datetime.now() - timedelta(days=365 * years_back)).strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start, interval='1wk', progress=False)

    if df.empty:
        return None

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df['SMA_50'] = calculate_sma(df['Close'], 50)

    df = calculate_z_score_macdv(df)
    df.dropna(inplace=True)

    return df

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(layout="wide", page_title="Z-Score MACD-V Analyzer")
st.title("📊 Z-Score MACD-V Regime Analyzer (Pro)")

ticker = st.text_input("Ticker", "AAPL").upper()
years_back = st.slider("Histórico (años)", 3, 12, 7)

if st.button("🚀 ANALIZAR", use_container_width=True):

    df = download_weekly_data(ticker, years_back)

    if df is not None:

        df['Regime'] = classify_regime(df)
        current = df.iloc[-1]

        # ================= METRICS =================
        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("RÉGIMEN", current['Regime'])
        c2.metric("PRECIO", f"${current['Close']:.2f}")
        c3.metric("Z-SCORE", f"{current['Z_Score_Adjusted']:.2f}")
        c4.metric("MACD-V", f"{current['MACD_V']:.2f}")
        c5.metric("KURTOSIS", f"{current['Kurtosis']:.2f}")

        # ================= CHART =================
        plt.style.use('dark_background')
        fig, axs = plt.subplots(3, 1, figsize=(18, 18), sharex=True)

        # PRECIO
        axs[0].plot(df.index, df['Close'], color='white', alpha=0.4)
        axs[0].plot(df.index, df['SMA_50'], color='violet')

        for r, c in REGIME_COLORS.items():
            m = df['Regime'] == r
            axs[0].scatter(df[m].index, df[m]['Close'], c=c, s=70, label=r)

        axs[0].legend(ncol=3)
        axs[0].set_title(f"{ticker} — Precio y Regímenes")

        # Z-SCORE
        axs[1].plot(df.index, df['Z_Score_Adjusted'], color='cyan')
        axs[1].axhline(2, color='red', linestyle='--')
        axs[1].axhline(-2, color='purple', linestyle='--')
        axs[1].set_title("Z-Score Ajustado")

        # MACD-V
        axs[2].plot(df.index, df['MACD_V'], label='MACD-V', color='#4ECDC4')
        axs[2].plot(df.index, df['MACD_V_Signal'], label='Signal', linestyle='--')

        axs[2].annotate(
            f"Último MACD-V: {current['MACD_V']:.2f}",
            xy=(df.index[-1], current['MACD_V']),
            xytext=(-120, 30),
            textcoords='offset points',
            arrowprops=dict(arrowstyle="->"),
            fontsize=10
        )

        axs[2].legend()
        axs[2].set_title("MACD-V")

        st.pyplot(fig)
