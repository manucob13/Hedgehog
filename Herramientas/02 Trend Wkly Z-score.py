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
# COLORES DE REGÍMENES (CLAROS Y DISTINTIVOS)
# ============================================================================

REGIME_COLORS = {
    "ALCISTA_FUERTE": "#00FF00",      # Verde neón
    "ALCISTA": "#7FFF00",             # Verde chartreuse
    "ALCISTA_RIESGO": "#FF8C00",      # Naranja oscuro
    "BAJISTA_FUERTE": "#FF0000",      # Rojo puro
    "BAJISTA": "#FF6347",             # Tomate
    "BAJISTA_RIESGO": "#FF1493",      # Rosa profundo
    "RANGO": "#FFD700",               # Dorado
}

# Nombres cortos para display
REGIME_NAMES = {
    "ALCISTA_FUERTE": "ALCISTA+",
    "ALCISTA": "ALCISTA",
    "ALCISTA_RIESGO": "ALC RISK",
    "BAJISTA_FUERTE": "BAJISTA+",
    "BAJISTA": "BAJISTA",
    "BAJISTA_RIESGO": "BAJ RISK",
    "RANGO": "RANGO",
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
# CLASIFICACIÓN DE REGÍMENES (LÓGICA REVISADA Y ESTRICTA)
# ============================================================================

def classify_regime(df):

    regimes = []
    
    # Pendiente de SMA50 (más suave, 8 períodos)
    sma_slope = (df["SMA_50"] - df["SMA_50"].shift(8)) / df["SMA_50"].shift(8) * 100
    
    # Distancia del precio a la SMA
    price_distance = (df["Close"] - df["SMA_50"]) / df["SMA_50"] * 100

    prev_regime = "RANGO"
    confirm = 0

    for i in range(len(df)):

        try:
            price = float(df["Close"].iloc[i])
            sma50 = float(df["SMA_50"].iloc[i])
            z = float(df["Z_Score_Adjusted"].iloc[i])
            macd_v = float(df["MACD_V"].iloc[i])
            histogram = float(df["MACD_V_Histogram"].iloc[i])
            slope = float(sma_slope.iloc[i])
            dist = float(price_distance.iloc[i])
        except (ValueError, TypeError):
            regimes.append(prev_regime)
            continue

        if any(pd.isna(x) for x in [price, sma50, z, macd_v, histogram, slope, dist]):
            regimes.append(prev_regime)
            continue

        if i < 10:
            regimes.append(prev_regime)
            continue

        # ========== DETECCIÓN DE RIESGO (PRIORIDAD MÁXIMA) ==========
        if abs(z) > 2.0:
            if z > 2.0:
                new_regime = "ALCISTA_RIESGO"  # Sobrecompra extrema
            else:
                new_regime = "BAJISTA_RIESGO"  # Sobreventa extrema
        
        # ========== TENDENCIAS CLARAS ==========
        elif price > sma50 and slope > 1.0 and dist > 2.0:
            # Alcista fuerte: precio muy por encima, SMA subiendo fuerte
            if macd_v > 0 and histogram > 0:
                new_regime = "ALCISTA_FUERTE"
            else:
                new_regime = "ALCISTA"
        
        elif price < sma50 and slope < -1.0 and dist < -2.0:
            # Bajista fuerte: precio muy por debajo, SMA bajando fuerte
            if macd_v < 0 and histogram < 0:
                new_regime = "BAJISTA_FUERTE"
            else:
                new_regime = "BAJISTA"
        
        # ========== TENDENCIAS MODERADAS ==========
        elif price > sma50 and slope > 0.3:
            new_regime = "ALCISTA"
        
        elif price < sma50 and slope < -0.3:
            new_regime = "BAJISTA"
        
        # ========== RANGO (SIN TENDENCIA) ==========
        else:
            new_regime = "RANGO"

        # ========== HISTERESIS (4 períodos) ==========
        if new_regime != prev_regime:
            confirm += 1
            if confirm < 4:
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
                    f'<div style="background-color:{color};padding:6px;border-radius:4px;text-align:center;font-weight:bold;color:black;font-size:11px;">'
                    f'{REGIME_NAMES[regime]}</div>',
                    unsafe_allow_html=True
                )

        # ================= GRÁFICOS =================
        plt.style.use("dark_background")
        fig, axs = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

        # ========== GRÁFICO 1: PRECIO ==========
        axs[0].plot(df_plot.index, df_plot["Close"], color="white", alpha=0.5, linewidth=1.5)
        axs[0].plot(df_plot.index, df_plot["SMA_50"], color="cyan", linewidth=2, alpha=0.7)

        for r, c in REGIME_COLORS.items():
            m = df_plot["Regime"] == r
            if m.any():
                axs[0].scatter(df_plot[m].index, df_plot[m]["Close"], c=c, s=120, alpha=1, 
                             edgecolors='black', linewidths=2, zorder=5)

        last_price = float(df_plot["Close"].iloc[-1])
        axs[0].text(1.01, last_price, f'${last_price:.2f}', 
                   transform=axs[0].get_yaxis_transform(), 
                   fontsize=10, color='white', va='center', fontweight='bold')
        
        axs[0].set_title(f"{ticker} — Precio y Regímenes", fontsize=12, fontweight='bold', pad=10)
        axs[0].grid(alpha=0.25, linestyle='--')
        axs[0].set_ylabel("Precio ($)", fontsize=10)
        axs[0].yaxis.tick_right()
        axs[0].yaxis.set_label_position("right")
        axs[0].tick_params(labelsize=9)

        # ========== GRÁFICO 2: Z-SCORE Y MACD-V COMBINADOS ==========
        ax2_left = axs[1]
        ax2_right = ax2_left.twinx()
        
        # Z-Score en eje izquierdo
        ax2_left.plot(df_plot.index, df_plot["Z_Score_Adjusted"], color="lime", linewidth=2, label="Z-Score")
        ax2_left.axhline(2, color="red", linestyle="--", alpha=0.6, linewidth=1.5)
        ax2_left.axhline(-2, color="magenta", linestyle="--", alpha=0.6, linewidth=1.5)
        ax2_left.axhline(0, color="gray", linestyle="-", alpha=0.4)
        ax2_left.fill_between(df_plot.index, 2, df_plot["Z_Score_Adjusted"], 
                             where=(df_plot["Z_Score_Adjusted"]>2), color="red", alpha=0.15)
        ax2_left.fill_between(df_plot.index, -2, df_plot["Z_Score_Adjusted"], 
                             where=(df_plot["Z_Score_Adjusted"]<-2), color="magenta", alpha=0.15)
        
        # MACD-V en eje derecho (SOLO LA LÍNEA PRINCIPAL, SIN SIGNAL)
        ax2_right.plot(df_plot.index, df_plot["MACD_V"], color="yellow", linewidth=2, label="MACD-V")
        ax2_right.axhline(0, color="gray", linestyle="-", alpha=0.4)
        
        # Valores actuales
        last_z = float(df_plot["Z_Score_Adjusted"].iloc[-1])
        last_macd = float(df_plot["MACD_V"].iloc[-1])
        
        ax2_left.text(-0.01, last_z, f'{last_z:.2f}σ', 
                     transform=ax2_left.get_yaxis_transform(), 
                     fontsize=10, color='lime', va='center', fontweight='bold', ha='right')
        
        ax2_right.text(1.01, last_macd, f'{last_macd:.2f}', 
                      transform=ax2_right.get_yaxis_transform(), 
                      fontsize=10, color='yellow', va='center', fontweight='bold')
        
        ax2_left.set_title("Z-Score (izq) y MACD-V (der)", fontsize=12, fontweight='bold', pad=10)
        ax2_left.grid(alpha=0.25, linestyle='--')
        ax2_left.set_ylabel("Z-Score", fontsize=10, color='lime')
        ax2_right.set_ylabel("MACD-V", fontsize=10, color='yellow')
        ax2_left.set_xlabel("Fecha", fontsize=10)
        ax2_left.tick_params(axis='y', labelcolor='lime', labelsize=9)
        ax2_right.tick_params(axis='y', labelcolor='yellow', labelsize=9)
        ax2_left.tick_params(axis='x', labelsize=9)

        plt.tight_layout()
        st.pyplot(fig)

    else:
        st.error("❌ No se pudo descargar data. Verifica el ticker.")
