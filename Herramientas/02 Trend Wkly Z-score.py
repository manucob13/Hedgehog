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
# COLORES DE REGÍMENES (SISTEMA SIMPLIFICADO)
# ============================================================================

REGIME_COLORS = {
    "ALCISTA_FUERTE": "#00FF00",      # Verde brillante
    "ALCISTA": "#90EE90",             # Verde claro
    "ALCISTA_RIESGO": "#FF4500",      # Rojo-naranja (sobrecompra)
    "BAJISTA_FUERTE": "#DC143C",      # Rojo oscuro
    "BAJISTA": "#FF6B6B",             # Rojo claro
    "BAJISTA_RIESGO": "#FF1493",      # Rosa fuerte (sobreventa)
    "RANGO": "#FFD700",               # Amarillo dorado
}

# Nombres cortos para display
REGIME_NAMES = {
    "ALCISTA_FUERTE": "🟢 ALC+",
    "ALCISTA": "🟢 ALC",
    "ALCISTA_RIESGO": "🔴 ALC⚠",
    "BAJISTA_FUERTE": "🔴 BAJ+",
    "BAJISTA": "🔴 BAJ",
    "BAJISTA_RIESGO": "🔴 BAJ⚠",
    "RANGO": "🟡 RNG",
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
    histogram = macd_v - signal_line

    mean = histogram.rolling(z_window).mean()
    std = histogram.rolling(z_window).std()
    z = (histogram - mean) / std

    kurt = histogram.rolling(z_window).apply(
        lambda x: stats.kurtosis(x, fisher=True, nan_policy="omit"),
        raw=False,
    )

    adj_factor = 1 + (kurt / 10).clip(-0.5, 0.5)

    df["MACD_V"] = macd_v
    df["MACD_V_Signal"] = signal_line
    df["MACD_V_Histogram"] = histogram
    df["Z_Score_Adjusted"] = z / adj_factor
    df["Kurtosis"] = kurt

    return df

# ============================================================================
# CLASIFICACIÓN DE REGÍMENES (LÓGICA MEJORADA)
# ============================================================================

def classify_regime(df):

    regimes = []
    
    # Pendiente de SMA50 (en porcentaje)
    sma_slope = (df["SMA_50"] - df["SMA_50"].shift(5)) / df["SMA_50"].shift(5) * 100
    
    # Tendencia del MACD-V
    macd_trend = df["MACD_V"].diff().rolling(3).mean()

    prev_regime = "RANGO"
    confirm = 0

    for i in range(len(df)):

        # Extracción segura de valores escalares
        try:
            price = float(df["Close"].iloc[i])
            sma50 = float(df["SMA_50"].iloc[i])
            z = float(df["Z_Score_Adjusted"].iloc[i])
            macd_v = float(df["MACD_V"].iloc[i])
            histogram = float(df["MACD_V_Histogram"].iloc[i])
            m_trend = float(macd_trend.iloc[i])
            slope = float(sma_slope.iloc[i])
        except (ValueError, TypeError):
            regimes.append(prev_regime)
            continue

        # Protección NaN
        if any(pd.isna(x) for x in [price, sma50, z, macd_v, histogram, m_trend, slope]):
            regimes.append(prev_regime)
            continue

        # Evitar índices negativos
        if i < 6:
            regimes.append(prev_regime)
            continue

        # ========== LÓGICA DE CLASIFICACIÓN ==========
        
        # 1. Detectar tendencia estructural fuerte
        price_above_sma = price > sma50
        sma_trending_up = slope > 0.5
        sma_trending_down = slope < -0.5
        
        # 2. MACD-V momentum
        macd_positive = macd_v > 0
        macd_accelerating = m_trend > 0
        
        # 3. Extremos (Z-Score)
        sobrecompra = z > 2.0
        sobreventa = z < -2.0
        
        # 4. Histograma
        hist_positive = histogram > 0
        
        # ========== REGÍMENES ALCISTAS ==========
        if price_above_sma and sma_trending_up:
            if sobrecompra and not macd_accelerating:
                new_regime = "ALCISTA_RIESGO"  # Sobrecompra - riesgo de corrección
            elif macd_positive and hist_positive and macd_accelerating:
                new_regime = "ALCISTA_FUERTE"  # Todo alineado
            else:
                new_regime = "ALCISTA"  # Tendencia pero sin fuerza
        
        # ========== REGÍMENES BAJISTAS ==========
        elif not price_above_sma and sma_trending_down:
            if sobreventa and macd_accelerating:
                new_regime = "BAJISTA_RIESGO"  # Sobreventa - posible rebote
            elif not macd_positive and not hist_positive and not macd_accelerating:
                new_regime = "BAJISTA_FUERTE"  # Todo bajista
            else:
                new_regime = "BAJISTA"  # Tendencia bajista moderada
        
        # ========== RANGO (sin tendencia clara) ==========
        else:
            new_regime = "RANGO"

        # ========== HISTERESIS (3 períodos para evitar ruido) ==========
        if new_regime != prev_regime:
            confirm += 1
            if confirm < 3:
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

    df = df.reset_index()
    df = df[["Date", "Open", "High", "Low", "Close"]].copy()
    df.set_index("Date", inplace=True)
    
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df = calculate_z_score_macdv(df)
    df.dropna(inplace=True)

    return df

# ============================================================================
# UI STREAMLIT
# ============================================================================

st.set_page_config(layout="wide", page_title="Weekly Z-Score MACD-V")
st.title("📊 Weekly Z-Score MACD-V Regime Analyzer")

ticker = st.text_input("Ticker", "AAPL").upper()
years_back = st.slider("Histórico (años)", 3, 12, 7)
lookback_months = st.slider("Meses a visualizar", 1, 36, 6)

if st.button("🚀 ANALIZAR", use_container_width=True):

    df = download_weekly_data(ticker, years_back)

    if df is not None:

        df["Regime"] = classify_regime(df)
        df_plot = df.tail(int(lookback_months * 4.33))
        current = df.iloc[-1]

        # ================= MÉTRICAS =================
        c1, c2, c3, c4 = st.columns(4)
        
        regime_key = str(current["Regime"])
        regime_display = REGIME_NAMES.get(regime_key, regime_key)
        
        c1.metric("RÉGIMEN", regime_display)
        c2.metric("PRECIO", f"${float(current['Close']):.2f}")
        c3.metric("Z-SCORE", f"{float(current['Z_Score_Adjusted']):.2f}")
        c4.metric("MACD-V", f"{float(current['MACD_V']):.2f}")

        # ================= LEYENDA DE COLORES =================
        st.markdown("### 🎨 Leyenda de Regímenes")
        cols = st.columns(7)
        for idx, (regime, color) in enumerate(REGIME_COLORS.items()):
            with cols[idx]:
                st.markdown(
                    f'<div style="background-color:{color};padding:6px;border-radius:4px;text-align:center;font-weight:bold;color:black;font-size:12px;">'
                    f'{REGIME_NAMES[regime]}</div>',
                    unsafe_allow_html=True
                )

        # ================= GRÁFICOS MÁS PEQUEÑOS =================
        plt.style.use("dark_background")
        fig, axs = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

        # ========== GRÁFICO 1: PRECIO ==========
        axs[0].plot(df_plot.index, df_plot["Close"], color="white", alpha=0.5, linewidth=1.2)
        axs[0].plot(df_plot.index, df_plot["SMA_50"], color="cyan", linewidth=1.5, alpha=0.7)

        for r, c in REGIME_COLORS.items():
            m = df_plot["Regime"] == r
            if m.any():
                axs[0].scatter(df_plot[m].index, df_plot[m]["Close"], c=c, s=70, alpha=0.9, 
                             edgecolors='black', linewidths=1, zorder=5)

        # Valor actual en la derecha
        last_price = float(df_plot["Close"].iloc[-1])
        axs[0].text(1.01, last_price, f'${last_price:.2f}', 
                   transform=axs[0].get_yaxis_transform(), 
                   fontsize=9, color='white', va='center', fontweight='bold')
        
        axs[0].set_title(f"{ticker} — Precio y Regímenes", fontsize=11, fontweight='bold', pad=8)
        axs[0].grid(alpha=0.2, linestyle='--')
        axs[0].set_ylabel("Precio ($)", fontsize=9)
        axs[0].yaxis.tick_right()
        axs[0].yaxis.set_label_position("right")
        axs[0].tick_params(labelsize=8)

        # ========== GRÁFICO 2: Z-SCORE ==========
        axs[1].plot(df_plot.index, df_plot["Z_Score_Adjusted"], color="lime", linewidth=1.5)
        axs[1].axhline(2, color="red", linestyle="--", alpha=0.6, linewidth=1.2)
        axs[1].axhline(-2, color="magenta", linestyle="--", alpha=0.6, linewidth=1.2)
        axs[1].axhline(0, color="gray", linestyle="-", alpha=0.4)
        axs[1].fill_between(df_plot.index, 2, df_plot["Z_Score_Adjusted"], 
                           where=(df_plot["Z_Score_Adjusted"]>2), color="red", alpha=0.15)
        axs[1].fill_between(df_plot.index, -2, df_plot["Z_Score_Adjusted"], 
                           where=(df_plot["Z_Score_Adjusted"]<-2), color="magenta", alpha=0.15)
        
        # Valor actual en la derecha
        last_z = float(df_plot["Z_Score_Adjusted"].iloc[-1])
        axs[1].text(1.01, last_z, f'{last_z:.2f}σ', 
                   transform=axs[1].get_yaxis_transform(), 
                   fontsize=9, color='lime', va='center', fontweight='bold')
        
        axs[1].set_title("Z-Score Ajustado", fontsize=11, fontweight='bold', pad=8)
        axs[1].grid(alpha=0.2, linestyle='--')
        axs[1].set_ylabel("Z-Score", fontsize=9)
        axs[1].yaxis.tick_right()
        axs[1].yaxis.set_label_position("right")
        axs[1].tick_params(labelsize=8)

        # ========== GRÁFICO 3: MACD-V ==========
        axs[2].plot(df_plot.index, df_plot["MACD_V"], color="yellow", linewidth=1.5)
        axs[2].plot(df_plot.index, df_plot["MACD_V_Signal"], linestyle="--", 
                   color="orange", linewidth=1.5, alpha=0.7)
        axs[2].axhline(0, color="gray", linestyle="-", alpha=0.4)
        
        # Valor actual en la derecha
        last_macd = float(df_plot["MACD_V"].iloc[-1])
        axs[2].text(1.01, last_macd, f'{last_macd:.2f}', 
                   transform=axs[2].get_yaxis_transform(), 
                   fontsize=9, color='yellow', va='center', fontweight='bold')
        
        axs[2].set_title("MACD-V", fontsize=11, fontweight='bold', pad=8)
        axs[2].grid(alpha=0.2, linestyle='--')
        axs[2].set_ylabel("MACD-V", fontsize=9)
        axs[2].set_xlabel("Fecha", fontsize=9)
        axs[2].yaxis.tick_right()
        axs[2].yaxis.set_label_position("right")
        axs[2].tick_params(labelsize=8)

        plt.tight_layout()
        st.pyplot(fig)

    else:
        st.error("❌ No se pudo descargar data. Verifica el ticker.")
