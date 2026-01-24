import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# COLORES DE REGÍMENES
# ============================================================================

REGIME_COLORS = {
    "ALCISTA_FUERTE": "#00FF00",
    "ALCISTA": "#2ECC71",
    "ALCISTA_EXTREMO_RIESGO": "#F39C12",
    "BAJISTA_FUERTE": "#4169E1",
    "BAJISTA": "#3498DB",
    "BAJISTA_EXTREMO_RIESGO": "#9B59B6",
    "RANGO": "#7F8C8D",
    "RANGO_EXTREMO": "#E74C3C",
}

# ============================================================================
# INDICADORES
# ============================================================================

def calculate_z_score_macdv(df, fast=12, slow=26, signal=9, z_window=20):

    close = df["Close"].astype(float).squeeze()
    high = df["High"].astype(float).squeeze()
    low = df["Low"].astype(float).squeeze()

    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(slow).mean()

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_v = ((ema_fast - ema_slow) / atr) * 100
    signal_line = macd_v.ewm(span=signal, adjust=False).mean()
    momentum = macd_v - signal_line

    mean = momentum.rolling(z_window).mean()
    std = momentum.rolling(z_window).std()
    z = (momentum - mean) / std

    kurt = momentum.rolling(z_window).apply(
        lambda x: stats.kurtosis(x, fisher=True, nan_policy="omit"),
        raw=False,
    )

    adj_factor = 1 + (kurt / 10).clip(-0.5, 0.5)

    df["MACD_V"] = macd_v
    df["MACD_V_Signal"] = signal_line
    df["Z_Score_Adjusted"] = z / adj_factor
    df["Kurtosis"] = kurt

    return df

# ============================================================================
# CLASIFICACIÓN DE REGÍMENES
# ============================================================================

def classify_regime(df):

    regimes = []
    macd_trend = df["MACD_V"].diff().rolling(3).mean()

    prev_regime = "RANGO"
    confirm = 0

    for i in range(len(df)):

        price = df["Close"].iloc[i]
        sma50 = df["SMA_50"].iloc[i]
        z = df["Z_Score_Adjusted"].iloc[i]
        m_trend = macd_trend.iloc[i]

        if np.isnan(price) or np.isnan(sma50) or np.isnan(z):
            regimes.append(prev_regime)
            continue

        # 1️⃣ ESTRUCTURA
        if i >= 3 and price > sma50 and sma50 > df["SMA_50"].iloc[i - 3]:
            structural = "ALCISTA"
        elif i >= 3 and price < sma50 and sma50 < df["SMA_50"].iloc[i - 3]:
            structural = "BAJISTA"
        else:
            structural = "RANGO"

        # 2️⃣ MOMENTUM
        if m_trend > 0:
            momentum = "ACELERANDO"
        elif m_trend < 0:
            momentum = "DESACELERANDO"
        else:
            momentum = "NEUTRO"

        # 3️⃣ EXTREMOS
        if z > 2:
            extreme = "SOBRECOMPRA"
        elif z < -2:
            extreme = "SOBREVENTA"
        else:
            extreme = "NORMAL"

        # 4️⃣ COMPOSICIÓN
        if structural == "ALCISTA":
            if extreme == "SOBRECOMPRA" and momentum == "DESACELERANDO":
                new_regime = "ALCISTA_EXTREMO_RIESGO"
            elif momentum == "ACELERANDO":
                new_regime = "ALCISTA_FUERTE"
            else:
                new_regime = "ALCISTA"

        elif structural == "BAJISTA":
            if extreme == "SOBREVENTA" and momentum == "DESACELERANDO":
                new_regime = "BAJISTA_EXTREMO_RIESGO"
            elif momentum == "ACELERANDO":
                new_regime = "BAJISTA_FUERTE"
            else:
                new_regime = "BAJISTA"

        else:
            new_regime = "RANGO_EXTREMO" if extreme != "NORMAL" else "RANGO"

        # 5️⃣ HISTERESIS
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
# DESCARGA
# ============================================================================

@st.cache_data(ttl=3600)
def download_weekly_data(ticker, years_back):

    start = (datetime.now() - timedelta(days=365 * years_back)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, interval="1wk", progress=False)

    if df.empty:
        return None

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df["SMA_50"] = df["Close"].rolling(50).mean()

    df = calculate_z_score_macdv(df)
    df.dropna(inplace=True)

    return df

# ============================================================================
# UI
# ============================================================================

st.set_page_config(layout="wide", page_title="Weekly Z-Score MACD-V")
st.title("📊 Weekly Z-Score MACD-V Regime Analyzer")

ticker = st.text_input("Ticker", "AAPL").upper()
years_back = st.slider("Histórico descargado (años)", 3, 12, 7)
lookback_months = st.slider("Meses a visualizar", 1, 36, 6)

if st.button("🚀 ANALIZAR", use_container_width=True):

    df = download_weekly_data(ticker, years_back)

    if df is not None:

        df["Regime"] = classify_regime(df)
        df_plot = df.tail(int(lookback_months * 4.33))
        current = df.iloc[-1]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("RÉGIMEN", current["Regime"])
        c2.metric("PRECIO", f"${current['Close']:.2f}")
        c3.metric("Z-SCORE", f"{current['Z_Score_Adjusted']:.2f}")
        c4.metric("MACD-V", f"{current['MACD_V']:.2f}")
        c5.metric("KURTOSIS", f"{current['Kurtosis']:.2f}")

        plt.style.use("dark_background")
        fig, axs = plt.subplots(3, 1, figsize=(18, 18), sharex=True)

        axs[0].plot(df_plot.index, df_plot["Close"], color="white", alpha=0.4)
        axs[0].plot(df_plot.index, df_plot["SMA_50"], color="violet")

        for r, c in REGIME_COLORS.items():
            m = df_plot["Regime"] == r
            axs[0].scatter(df_plot[m].index, df_plot[m]["Close"], c=c, s=60)

        axs[0].set_title(f"{ticker} — Precio y Regímenes")

        axs[1].plot(df_plot.index, df_plot["Z_Score_Adjusted"], color="cyan")
        axs[1].axhline(2, color="red", linestyle="--")
        axs[1].axhline(-2, color="purple", linestyle="--")

        axs[2].plot(df_plot.index, df_plot["MACD_V"], label="MACD-V")
        axs[2].plot(df_plot.index, df_plot["MACD_V_Signal"], linestyle="--", label="Signal")
        axs[2].legend()

        st.pyplot(fig)
