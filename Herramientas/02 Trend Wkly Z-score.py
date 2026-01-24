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
    df["MACD_V_Histogram"] = macd_v - signal_line
    df["Z_Score_Adjusted"] = z / adj_factor
    df["Kurtosis"] = kurt

    return df

# ============================================================================
# CLASIFICACIÓN DE REGÍMENES (OPTIMIZADA PARA SWING TRADING)
# ============================================================================

def classify_regime(df):

    regimes = []
    
    # Tendencia del histograma MACD-V (más reactivo)
    hist_trend = df["MACD_V_Histogram"].diff().rolling(2).mean()
    
    # Pendiente de SMA50
    sma_slope = (df["SMA_50"] - df["SMA_50"].shift(3)) / df["SMA_50"].shift(3) * 100

    prev_regime = "RANGO"
    confirm = 0

    for i in range(len(df)):

        # Extracción segura de valores escalares
        try:
            price = float(df["Close"].iloc[i])
            sma50 = float(df["SMA_50"].iloc[i])
            z = float(df["Z_Score_Adjusted"].iloc[i])
            histogram = float(df["MACD_V_Histogram"].iloc[i])
            h_trend = float(hist_trend.iloc[i])
            slope = float(sma_slope.iloc[i])
        except (ValueError, TypeError):
            regimes.append(prev_regime)
            continue

        # Protección NaN
        if pd.isna(price) or pd.isna(sma50) or pd.isna(z) or pd.isna(h_trend) or pd.isna(slope):
            regimes.append(prev_regime)
            continue

        # Evitar índices negativos
        if i < 5:
            regimes.append(prev_regime)
            continue

        # 1️⃣ RÉGIMEN ESTRUCTURAL (con pendiente de SMA)
        if price > sma50 and slope > 0.3:  # Tendencia alcista clara
            structural = "ALCISTA"
        elif price < sma50 and slope < -0.3:  # Tendencia bajista clara
            structural = "BAJISTA"
        else:
            structural = "RANGO"

        # 2️⃣ MOMENTUM (basado en histograma MACD-V)
        if h_trend > 0 and histogram > 0:
            momentum = "ACELERANDO"
        elif h_trend < 0 and histogram < 0:
            momentum = "DESACELERANDO"
        elif h_trend > 0 and histogram < 0:
            momentum = "RECUPERANDO"
        elif h_trend < 0 and histogram > 0:
            momentum = "DEBILITANDO"
        else:
            momentum = "NEUTRO"

        # 3️⃣ EXTREMOS (Z-SCORE)
        if z > 2:
            extreme = "SOBRECOMPRA"
        elif z < -2:
            extreme = "SOBREVENTA"
        else:
            extreme = "NORMAL"

        # 4️⃣ COMPOSICIÓN FINAL
        if structural == "ALCISTA":
            if extreme == "SOBRECOMPRA" and momentum in ["DESACELERANDO", "DEBILITANDO"]:
                new_regime = "ALCISTA_EXTREMO_RIESGO"
            elif momentum in ["ACELERANDO", "RECUPERANDO"]:
                new_regime = "ALCISTA_FUERTE"
            else:
                new_regime = "ALCISTA"

        elif structural == "BAJISTA":
            if extreme == "SOBREVENTA" and momentum in ["DESACELERANDO", "DEBILITANDO"]:
                new_regime = "BAJISTA_EXTREMO_RIESGO"
            elif momentum in ["ACELERANDO", "RECUPERANDO"]:
                new_regime = "BAJISTA_FUERTE"
            else:
                new_regime = "BAJISTA"

        else:  # RANGO
            new_regime = "RANGO_EXTREMO" if extreme != "NORMAL" else "RANGO"

        # 5️⃣ HISTERESIS (reducida para swing trading de 15 días)
        if new_regime != prev_regime:
            confirm += 1
            if confirm < 2:  # Solo 2 semanas de confirmación
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
def download_weekly_data(ticker, years_back):

    start = (datetime.now() - timedelta(days=365 * years_back)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, interval="1wk", progress=False)

    if df.empty:
        return None

    # Resetear índice para evitar problemas con MultiIndex
    df = df.reset_index()
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    df.set_index("Date", inplace=True)
    
    df["SMA_50"] = df["Close"].rolling(50).mean()

    df = calculate_z_score_macdv(df)
    df.dropna(inplace=True)

    return df

# ============================================================================
# UI STREAMLIT
# ============================================================================

st.set_page_config(layout="wide", page_title="Weekly Z-Score MACD-V")
st.title("📊 Weekly Z-Score MACD-V Regime Analyzer (Swing Trading)")

ticker = st.text_input("Ticker", "AAPL").upper()
years_back = st.slider("Histórico descargado (años)", 3, 12, 7)
lookback_months = st.slider("Meses a visualizar", 1, 36, 6)

if st.button("🚀 ANALIZAR", use_container_width=True):

    df = download_weekly_data(ticker, years_back)

    if df is not None:

        df["Regime"] = classify_regime(df)
        df_plot = df.tail(int(lookback_months * 4.33))
        current = df.iloc[-1]

        # ================= MÉTRICAS =================
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        
        # Convertir régimen a string legible
        regime_name = str(current["Regime"]).replace("_", " ")
        
        c1.metric("RÉGIMEN", regime_name)
        c2.metric("PRECIO", f"${float(current['Close']):.2f}")
        c3.metric("Z-SCORE", f"{float(current['Z_Score_Adjusted']):.2f}")
        c4.metric("MACD-V", f"{float(current['MACD_V']):.2f}")
        c5.metric("HISTOGRAM", f"{float(current['MACD_V_Histogram']):.2f}")
        c6.metric("KURTOSIS", f"{float(current['Kurtosis']):.2f}")

        # ================= GRÁFICOS =================
        plt.style.use("dark_background")
        fig, axs = plt.subplots(4, 1, figsize=(18, 20), sharex=True)

        # PRECIO
        axs[0].plot(df_plot.index, df_plot["Close"], color="white", alpha=0.4, linewidth=1.5)
        axs[0].plot(df_plot.index, df_plot["SMA_50"], color="violet", linewidth=2, label="SMA 50")

        for r, c in REGIME_COLORS.items():
            m = df_plot["Regime"] == r
            axs[0].scatter(df_plot[m].index, df_plot[m]["Close"], c=c, s=80, alpha=0.8, edgecolors='white', linewidths=0.5)

        axs[0].set_title(f"{ticker} — Precio y Regímenes", fontsize=14, fontweight='bold')
        axs[0].legend()
        axs[0].grid(alpha=0.3)

        # Z-SCORE
        axs[1].plot(df_plot.index, df_plot["Z_Score_Adjusted"], color="cyan", linewidth=2)
        axs[1].axhline(2, color="red", linestyle="--", alpha=0.7, label="Sobrecompra")
        axs[1].axhline(-2, color="purple", linestyle="--", alpha=0.7, label="Sobreventa")
        axs[1].axhline(0, color="gray", linestyle="-", alpha=0.3)
        axs[1].fill_between(df_plot.index, 2, df_plot["Z_Score_Adjusted"], where=(df_plot["Z_Score_Adjusted"]>2), color="red", alpha=0.2)
        axs[1].fill_between(df_plot.index, -2, df_plot["Z_Score_Adjusted"], where=(df_plot["Z_Score_Adjusted"]<-2), color="purple", alpha=0.2)
        axs[1].set_title("Z-Score Ajustado por Curtosis", fontsize=14, fontweight='bold')
        axs[1].legend()
        axs[1].grid(alpha=0.3)

        # MACD-V
        axs[2].plot(df_plot.index, df_plot["MACD_V"], label="MACD-V", color="lime", linewidth=2)
        axs[2].plot(df_plot.index, df_plot["MACD_V_Signal"], linestyle="--", label="Signal", color="orange", linewidth=2)
        axs[2].axhline(0, color="gray", linestyle="-", alpha=0.3)
        axs[2].legend()
        axs[2].set_title("MACD-V Normalizado por ATR", fontsize=14, fontweight='bold')
        axs[2].grid(alpha=0.3)

        # HISTOGRAMA MACD-V
        colors = ['green' if x > 0 else 'red' for x in df_plot["MACD_V_Histogram"]]
        axs[3].bar(df_plot.index, df_plot["MACD_V_Histogram"], color=colors, alpha=0.6, width=5)
        axs[3].axhline(0, color="white", linestyle="-", alpha=0.5)
        axs[3].set_title("Histograma MACD-V (Momentum Puro)", fontsize=14, fontweight='bold')
        axs[3].grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        
        # ================= TABLA DE REGÍMENES RECIENTES =================
        st.subheader("📋 Últimos 10 Regímenes")
        recent = df[["Close", "SMA_50", "Z_Score_Adjusted", "MACD_V_Histogram", "Regime"]].tail(10).copy()
        recent["Close"] = recent["Close"].apply(lambda x: f"${float(x):.2f}")
        recent["SMA_50"] = recent["SMA_50"].apply(lambda x: f"${float(x):.2f}")
        recent["Z_Score_Adjusted"] = recent["Z_Score_Adjusted"].apply(lambda x: f"{float(x):.2f}")
        recent["MACD_V_Histogram"] = recent["MACD_V_Histogram"].apply(lambda x: f"{float(x):.2f}")
        st.dataframe(recent, use_container_width=True)

    else:
        st.error("❌ No se pudo descargar data. Verifica el ticker.")
