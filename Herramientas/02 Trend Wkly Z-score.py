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
    "ALCISTA": "#00FF00",             # Verde brillante
    "ALCISTA_RIESGO-": "#9370DB",     # Morado medio
    "ALCISTA_RIESGO+": "#4B0082",     # Índigo oscuro
    "BAJISTA": "#FF0000",             # Rojo brillante
    "BAJISTA_RIESGO-": "#9370DB",     # Morado medio (mismo que alcista)
    "BAJISTA_RIESGO+": "#4B0082",     # Índigo oscuro (mismo que alcista)
    "RANGO": "#FFD700",               # Dorado
}

# Nombres simplificados para tabla - CORREGIDO
REGIME_SIMPLE = {
    "ALCISTA": "ALCISTA",
    "ALCISTA_RIESGO-": "RIESGO-",
    "ALCISTA_RIESGO+": "RIESGO+",
    "BAJISTA": "BAJISTA",
    "BAJISTA_RIESGO-": "RIESGO-",
    "BAJISTA_RIESGO+": "RIESGO+",
    "RANGO": "RANGO",
}

# Leyenda consolidada (solo 5 colores únicos)
LEGEND_COLORS = {
    "ALCISTA": "#00FF00",
    "BAJISTA": "#FF0000",
    "RANGO": "#FFD700",
    "RIESGO-": "#9370DB",
    "RIESGO+": "#4B0082",
}

# ============================================================================
# INDICADORES
# ============================================================================

def calculate_indicators(df, fast=12, slow=26, signal=9, z_window=20, z_price_window=50):

    close = df["Close"].astype(float).squeeze()
    high = df["High"].astype(float).squeeze()
    low = df["Low"].astype(float).squeeze()

    # MACD-V
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

    # Z-Score del histograma MACD-V
    mean = histogram.rolling(z_window).mean()
    std = histogram.rolling(z_window).std()
    z_macd = (histogram - mean) / std

    kurt = histogram.rolling(z_window).apply(
        lambda x: stats.kurtosis(x, fisher=True, nan_policy="omit"),
        raw=False,
    )

    adj_factor = 1 + (kurt / 10).clip(-0.5, 0.5)
    z_macd_adjusted = z_macd / adj_factor

    # Z-Score del PRECIO
    price_mean = close.rolling(z_price_window).mean()
    price_std = close.rolling(z_price_window).std()
    z_price = (close - price_mean) / price_std

    df["MACD_V"] = macd_v
    df["MACD_V_Signal"] = signal_line
    df["MACD_V_Histogram"] = histogram
    df["Z_Score_MACD"] = z_macd_adjusted
    df["Z_Score_Price"] = z_price
    df["Kurtosis"] = kurt

    return df

# ============================================================================
# CLASIFICACIÓN POR MACD-V
# ============================================================================

def classify_by_macdv(df):
    regimes = []
    prev_regime = "RANGO"
    confirm = 0

    for i in range(len(df)):
        try:
            price = float(df["Close"].iloc[i])
            sma50 = float(df["SMA_50"].iloc[i])
            z_macd = float(df["Z_Score_MACD"].iloc[i])
            macd_v = float(df["MACD_V"].iloc[i])
        except (ValueError, TypeError):
            regimes.append(prev_regime)
            continue

        if any(pd.isna(x) for x in [price, sma50, z_macd, macd_v]):
            regimes.append(prev_regime)
            continue

        if i < 10:
            regimes.append(prev_regime)
            continue

        # Lado del mercado
        above_sma = price > sma50

        # Detectar RIESGO primero (prioridad)
        if z_macd > 2.0:  # Sobrecompra
            if macd_v < 150:
                new_regime = "ALCISTA_RIESGO-"
            else:
                new_regime = "ALCISTA_RIESGO+"
        elif z_macd < -2.0:  # Sobreventa
            if macd_v > -150:
                new_regime = "BAJISTA_RIESGO-"
            else:
                new_regime = "BAJISTA_RIESGO+"
        # Clasificación por MACD-V
        elif -50 <= macd_v <= 50:
            new_regime = "RANGO"
        elif 50 < macd_v <= 150:
            if above_sma:
                new_regime = "ALCISTA"
            else:
                new_regime = "RANGO"  # Conflicto
        elif -150 <= macd_v < -50:
            if not above_sma:
                new_regime = "BAJISTA"
            else:
                new_regime = "RANGO"  # Conflicto
        elif macd_v > 150:
            new_regime = "ALCISTA" if above_sma else "RANGO"
        elif macd_v < -150:
            new_regime = "BAJISTA" if not above_sma else "RANGO"
        else:
            new_regime = "RANGO"

        # Histeresis
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
# CLASIFICACIÓN POR Z-SCORE DE PRECIO
# ============================================================================

def classify_by_zscore(df):
    regimes = []
    prev_regime = "RANGO"
    confirm = 0

    for i in range(len(df)):
        try:
            price = float(df["Close"].iloc[i])
            sma50 = float(df["SMA_50"].iloc[i])
            z_macd = float(df["Z_Score_MACD"].iloc[i])
            z_price = float(df["Z_Score_Price"].iloc[i])
            macd_v = float(df["MACD_V"].iloc[i])
        except (ValueError, TypeError):
            regimes.append(prev_regime)
            continue

        if any(pd.isna(x) for x in [price, sma50, z_macd, z_price, macd_v]):
            regimes.append(prev_regime)
            continue

        if i < 10:
            regimes.append(prev_regime)
            continue

        above_sma = price > sma50

        # Detectar RIESGO primero (igual que MACD-V)
        if z_macd > 2.0:
            if macd_v < 150:
                new_regime = "ALCISTA_RIESGO-"
            else:
                new_regime = "ALCISTA_RIESGO+"
        elif z_macd < -2.0:
            if macd_v > -150:
                new_regime = "BAJISTA_RIESGO-"
            else:
                new_regime = "BAJISTA_RIESGO+"
        # Clasificación por Z-Score de PRECIO
        elif z_price > 0.5:
            if above_sma:
                new_regime = "ALCISTA"
            else:
                new_regime = "RANGO"
        elif z_price < -0.5:
            if not above_sma:
                new_regime = "BAJISTA"
            else:
                new_regime = "RANGO"
        else:
            new_regime = "RANGO"

        # Histeresis
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

@st.cache_data(ttl=3600, show_spinner=False)
def download_weekly_data(ticker, years_back):
    start = (datetime.now() - timedelta(days=365 * years_back)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, interval="1wk", progress=False)

    if df.empty:
        return None

    df = df.reset_index()
    df = df[["Date", "Open", "High", "Low", "Close"]].copy()
    df.set_index("Date", inplace=True)
    
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df = calculate_indicators(df)
    df.dropna(inplace=True)

    return df

# ============================================================================
# UI STREAMLIT
# ============================================================================

st.set_page_config(layout="wide", page_title="Weekly Z-Score MACD-V")
st.title("📊 Weekly Z-Score MACD-V Regime Analyzer")

# Mantener estado en session_state para evitar que desaparezca
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'ticker' not in st.session_state:
    st.session_state.ticker = "AAPL"

col_input1, col_input2, col_input3 = st.columns([2, 1, 1])
with col_input1:
    ticker = st.text_input("Ticker", st.session_state.ticker).upper()
with col_input2:
    years_back = st.slider("Histórico (años)", 3, 12, 5)
with col_input3:
    lookback_months = st.slider("Meses a visualizar", 1, 36, 12)

if st.button("🚀 ANALIZAR", use_container_width=True):
    st.session_state.ticker = ticker
    st.session_state.df = download_weekly_data(ticker, years_back)
    st.session_state.analyzed = True

if st.session_state.analyzed and st.session_state.df is not None:
    
    df = st.session_state.df
    
    if "Regime_MACDV" not in df.columns:
        df["Regime_MACDV"] = classify_by_macdv(df)
        df["Regime_ZScore"] = classify_by_zscore(df)
        st.session_state.df = df
    
    df_plot = df.tail(int(lookback_months * 4.33))
    current = df.iloc[-1]

    # ================= TABLA COMPARATIVA - CORREGIDA =================
    # Extraer correctamente el valor del régimen
    if isinstance(current["Regime_MACDV"], str):
        regime_macdv_full = current["Regime_MACDV"]
    else:
        regime_macdv_full = current["Regime_MACDV"].iloc[0] if hasattr(current["Regime_MACDV"], 'iloc') else str(current["Regime_MACDV"])
    
    if isinstance(current["Regime_ZScore"], str):
        regime_zscore_full = current["Regime_ZScore"]
    else:
        regime_zscore_full = current["Regime_ZScore"].iloc[0] if hasattr(current["Regime_ZScore"], 'iloc') else str(current["Regime_ZScore"])
    
    # Simplificar los nombres usando el diccionario
    regime_macdv_simple = REGIME_SIMPLE.get(regime_macdv_full, regime_macdv_full)
    regime_zscore_simple = REGIME_SIMPLE.get(regime_zscore_full, regime_zscore_full)
    
    comparison_data = {
        "Método": ["MACD-V", "Z-Score Precio"],
        "Régimen": [regime_macdv_simple, regime_zscore_simple],
        "Precio": [f"${float(current['Close']):.2f}", f"${float(current['Close']):.2f}"],
        "MACD-V": [f"{float(current['MACD_V']):.2f}", f"{float(current['MACD_V']):.2f}"],
        "Z-Score MACD": [f"{float(current['Z_Score_MACD']):.2f}", f"{float(current['Z_Score_MACD']):.2f}"],
        "Z-Score Precio": [f"{float(current['Z_Score_Price']):.2f}", f"{float(current['Z_Score_Price']):.2f}"],
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # ================= LEYENDA CONSOLIDADA =================
    st.markdown("### 🎨 Leyenda")
    cols = st.columns(5)
    for idx, (regime, color) in enumerate(LEGEND_COLORS.items()):
        with cols[idx]:
            st.markdown(
                f'<div style="background-color:{color};padding:4px;border-radius:3px;text-align:center;font-weight:bold;color:white;font-size:10px;">'
                f'{regime}</div>',
                unsafe_allow_html=True
            )

    # ================= GRÁFICOS =================
    plt.style.use("dark_background")
    fig, axs = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    # ========== GRÁFICO 1: PRECIO + MACD-V ==========
    axs[0].plot(df_plot.index, df_plot["Close"], color="white", alpha=0.5, linewidth=1.5)
    axs[0].plot(df_plot.index, df_plot["SMA_50"], color="cyan", linewidth=2, alpha=0.7)

    for r, c in REGIME_COLORS.items():
        m = df_plot["Regime_MACDV"] == r
        if m.any():
            axs[0].scatter(df_plot[m].index, df_plot[m]["Close"], c=c, s=90, alpha=0.95, 
                         edgecolors='black', linewidths=1.5, zorder=5)

    last_price = float(df_plot["Close"].iloc[-1])
    axs[0].text(1.01, last_price, f'${last_price:.2f}', 
               transform=axs[0].get_yaxis_transform(), 
               fontsize=9, color='white', va='center', fontweight='bold')
    
    axs[0].set_title(f"{ticker} — Método MACD-V", fontsize=11, fontweight='bold', pad=8)
    axs[0].grid(alpha=0.25, linestyle='--')
    axs[0].set_ylabel("Precio ($)", fontsize=9)
    axs[0].yaxis.tick_right()
    axs[0].yaxis.set_label_position("right")
    axs[0].tick_params(labelsize=8)

    # ========== GRÁFICO 2: PRECIO + Z-SCORE ==========
    axs[1].plot(df_plot.index, df_plot["Close"], color="white", alpha=0.5, linewidth=1.5)
    axs[1].plot(df_plot.index, df_plot["SMA_50"], color="cyan", linewidth=2, alpha=0.7)

    for r, c in REGIME_COLORS.items():
        m = df_plot["Regime_ZScore"] == r
        if m.any():
            axs[1].scatter(df_plot[m].index, df_plot[m]["Close"], c=c, s=90, alpha=0.95, 
                         edgecolors='black', linewidths=1.5, zorder=5)

    axs[1].text(1.01, last_price, f'${last_price:.2f}', 
               transform=axs[1].get_yaxis_transform(), 
               fontsize=9, color='white', va='center', fontweight='bold')
    
    axs[1].set_title(f"{ticker} — Método Z-Score Precio", fontsize=11, fontweight='bold', pad=8)
    axs[1].grid(alpha=0.25, linestyle='--')
    axs[1].set_ylabel("Precio ($)", fontsize=9)
    axs[1].yaxis.tick_right()
    axs[1].yaxis.set_label_position("right")
    axs[1].tick_params(labelsize=8)

    # ========== GRÁFICO 3: Z-SCORE MACD (izq) y MACD-V (der) CON COLORES CONDICIONALES ==========
    ax3_left = axs[2]
    ax3_right = ax3_left.twinx()
    
    # Z-Score MACD con colores condicionales
    z_colors = ['#FF0000' if abs(z) > 2 else '#87CEEB' for z in df_plot["Z_Score_MACD"]]
    for i in range(len(df_plot) - 1):
        ax3_left.plot(df_plot.index[i:i+2], df_plot["Z_Score_MACD"].iloc[i:i+2], 
                     color=z_colors[i], linewidth=2)
    
    ax3_left.axhline(2, color="red", linestyle="--", alpha=0.6, linewidth=1.5)
    ax3_left.axhline(-2, color="red", linestyle="--", alpha=0.6, linewidth=1.5)
    ax3_left.axhline(0, color="gray", linestyle="-", alpha=0.4)
    ax3_left.fill_between(df_plot.index, 2, df_plot["Z_Score_MACD"], 
                         where=(df_plot["Z_Score_MACD"]>2), color="red", alpha=0.15)
    ax3_left.fill_between(df_plot.index, -2, df_plot["Z_Score_MACD"], 
                         where=(df_plot["Z_Score_MACD"]<-2), color="red", alpha=0.15)
    
    # MACD-V con colores condicionales
    macd_colors = ['#FF0000' if abs(m) > 150 else '#FFFFFF' for m in df_plot["MACD_V"]]
    for i in range(len(df_plot) - 1):
        ax3_right.plot(df_plot.index[i:i+2], df_plot["MACD_V"].iloc[i:i+2], 
                      color=macd_colors[i], linewidth=2)
    
    ax3_right.axhline(0, color="gray", linestyle="-", alpha=0.4)
    ax3_right.axhline(50, color="green", linestyle=":", alpha=0.4, linewidth=1)
    ax3_right.axhline(-50, color="red", linestyle=":", alpha=0.4, linewidth=1)
    ax3_right.axhline(150, color="red", linestyle="--", alpha=0.6, linewidth=1.5)
    ax3_right.axhline(-150, color="red", linestyle="--", alpha=0.6, linewidth=1.5)
    
    last_z = float(df_plot["Z_Score_MACD"].iloc[-1])
    last_macd = float(df_plot["MACD_V"].iloc[-1])
    
    z_color_final = '#FF0000' if abs(last_z) > 2 else '#87CEEB'
    macd_color_final = '#FF0000' if abs(last_macd) > 150 else '#FFFFFF'
    
    ax3_left.text(-0.01, last_z, f'{last_z:.2f}σ', 
                 transform=ax3_left.get_yaxis_transform(), 
                 fontsize=9, color=z_color_final, va='center', fontweight='bold', ha='right')
    
    ax3_right.text(1.01, last_macd, f'{last_macd:.2f}', 
                  transform=ax3_right.get_yaxis_transform(), 
                  fontsize=9, color=macd_color_final, va='center', fontweight='bold')
    
    ax3_left.set_title("Z-Score MACD (izq) y MACD-V (der)", fontsize=11, fontweight='bold', pad=8)
    ax3_left.grid(alpha=0.25, linestyle='--')
    ax3_left.set_ylabel("Z-Score MACD", fontsize=9, color='#87CEEB')
    ax3_right.set_ylabel("MACD-V", fontsize=9, color='#FFFFFF')
    ax3_left.set_xlabel("Fecha", fontsize=9)
    ax3_left.tick_params(axis='y', labelcolor='#87CEEB', labelsize=8)
    ax3_right.tick_params(axis='y', labelcolor='#FFFFFF', labelsize=8)
    ax3_left.tick_params(axis='x', labelsize=8)

    plt.tight_layout()
    st.pyplot(fig)

    # ================= LÓGICA DE CLASIFICACIÓN =================
    st.markdown("---")
    st.markdown("### 📖 Lógica de Clasificación")
    
    col_logic1, col_logic2 = st.columns(2)
    
    with col_logic1:
        st.markdown("#### 🔹 Método MACD-V")
        st.markdown("""
        **Filtro 1: Precio vs SMA50**
        - Precio > SMA50 → Lado alcista
        - Precio < SMA50 → Lado bajista
        
        **Filtro 2: MACD-V (rango de valores)**
        - MACD-V entre -50 y +50 → **RANGO**
        - MACD-V entre 50 y 150 (y precio > SMA50) → **ALCISTA**
        - MACD-V entre -50 y -150 (y precio < SMA50) → **BAJISTA**
        
        **Filtro 3: Z-Score MACD (extremos)**
        - Z-Score > +2 y MACD-V < 150 → **RIESGO-**
        - Z-Score > +2 y MACD-V ≥ 150 → **RIESGO+**
        - Z-Score < -2 y MACD-V > -150 → **RIESGO-**
        - Z-Score < -2 y MACD-V ≤ -150 → **RIESGO+**
        
        **Confirmación:** 2 períodos consecutivos
        """)
    
    with col_logic2:
        st.markdown("#### 🔹 Método Z-Score Precio")
        st.markdown("""
        **Filtro 1: Precio vs SMA50**
        - Precio > SMA50 → Lado alcista
        - Precio < SMA50 → Lado bajista
        
        **Filtro 2: Z-Score Precio**
        - Z-Score Precio > +0.5 (y precio > SMA50) → **ALCISTA**
        - Z-Score Precio < -0.5 (y precio < SMA50) → **BAJISTA**
        - Z-Score Precio entre -0.5 y +0.5 → **RANGO**
        
        **Filtro 3: Z-Score MACD (extremos)**
        - Z-Score MACD > +2 y MACD-V < 150 → **RIESGO-**
        - Z-Score MACD > +2 y MACD-V ≥ 150 → **RIESGO+**
        - Z-Score MACD < -2 y MACD-V > -150 → **RIESGO-**
        - Z-Score MACD < -2 y MACD-V ≤ -150 → **RIESGO+**
        
        **Confirmación:** 2 períodos consecutivos
        """)

elif st.session_state.analyzed and st.session_state.df is None:
    st.error("❌ No se pudo descargar data. Verifica el ticker.")
