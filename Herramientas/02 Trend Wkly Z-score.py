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
# AUTENTICACIÓN - Importar desde utils
# ============================================================================

try:
    from utils.utils import check_password
except ImportError:
    # Fallback si no existe el módulo utils
    def check_password():
        """Verifica si el usuario tiene credenciales correctas."""
        
        def login_form():
            """Formulario de login en el sidebar."""
            with st.sidebar:
                st.markdown("### 🔑 Iniciar Sesión")
                username = st.text_input("Usuario", key="username_input")
                password = st.text_input("Contraseña", type="password", key="password_input")
                
                if st.button("Login", use_container_width=True):
                    if username in st.secrets.get("passwords", {}) and password == st.secrets["passwords"][username]:
                        st.session_state["password_correct"] = True
                        st.session_state["username"] = username
                        st.rerun()
                    else:
                        st.error("❌ Usuario o contraseña incorrectos")
        
        # Si ya está autenticado, mostrar usuario en sidebar
        if st.session_state.get("password_correct", False):
            with st.sidebar:
                st.success(f"✅ Sesión activa: {st.session_state.get('username', 'Usuario')}")
                if st.button("🚪 Cerrar Sesión", use_container_width=True):
                    st.session_state["password_correct"] = False
                    if "username" in st.session_state:
                        del st.session_state["username"]
                    st.rerun()
            return True
        
        # Mostrar formulario de login
        login_form()
        return False

# ============================================================================
# COLORES DE REGÍMENES
# ============================================================================

REGIME_COLORS = {
    "ALCISTA": "#00FF00",      # Verde brillante
    "BAJISTA": "#FF0000",      # Rojo brillante
    "RANGO": "#FFD700",        # Dorado
    "RIESGO-": "#9370DB",      # Morado medio
    "RIESGO+": "#4B0082",      # Índigo oscuro
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

    # Z-Score del histograma MACD-V (para detectar RIESGO)
    mean = histogram.rolling(z_window).mean()
    std = histogram.rolling(z_window).std()
    z_macd = (histogram - mean) / std

    kurt = histogram.rolling(z_window).apply(
        lambda x: stats.kurtosis(x, fisher=True, nan_policy="omit"),
        raw=False,
    )

    adj_factor = 1 + (kurt / 10).clip(-0.5, 0.5)
    z_macd_adjusted = z_macd / adj_factor

    # Z-Score del PRECIO (para clasificación alternativa)
    price_mean = close.rolling(z_price_window).mean()
    price_std = close.rolling(z_price_window).std()
    z_price = (close - price_mean) / price_std

    df["MACD_V"] = macd_v
    df["MACD_V_Signal"] = signal_line
    df["MACD_V_Histogram"] = histogram
    df["Risk_Indicator"] = z_macd_adjusted  # Renombrado para claridad
    df["Price_Deviation"] = z_price  # Renombrado para claridad
    df["Kurtosis"] = kurt

    return df

# ============================================================================
# CLASIFICACIÓN POR MACD-V (MEJORADA)
# ============================================================================

def classify_by_macdv(df):
    regimes = []

    for i in range(len(df)):
        try:
            price = float(df["Close"].iloc[i])
            sma50 = float(df["SMA_50"].iloc[i])
            risk_ind = float(df["Risk_Indicator"].iloc[i])
            macd_v = float(df["MACD_V"].iloc[i])
        except (ValueError, TypeError):
            regimes.append("RANGO")
            continue

        if any(pd.isna(x) for x in [price, sma50, risk_ind, macd_v]):
            regimes.append("RANGO")
            continue

        if i < 10:
            regimes.append("RANGO")
            continue

        # PASO 1: Detectar condiciones extremas de RIESGO (máxima prioridad)
        macd_extreme = abs(macd_v) > 150
        risk_extreme = abs(risk_ind) > 2.0

        if macd_extreme and risk_extreme:
            regime = "RIESGO+"
        elif macd_extreme or risk_extreme:
            regime = "RIESGO-"
        
        # PASO 2: Verificar si MACD-V está en zona neutra (RANGO)
        elif -50 <= macd_v <= 50:
            regime = "RANGO"
        
        # PASO 3: Clasificar según lado del mercado + MACD-V
        else:
            above_sma = price > sma50
            
            if above_sma and 50 < macd_v <= 150:
                regime = "ALCISTA"
            elif not above_sma and -150 <= macd_v < -50:
                regime = "BAJISTA"
            else:
                # Caso: precio y MACD-V en lados opuestos → RANGO
                regime = "RANGO"

        regimes.append(regime)

    return regimes

# ============================================================================
# CLASIFICACIÓN POR Z-SCORE DE PRECIO (MEJORADA)
# ============================================================================

def classify_by_zscore(df):
    regimes = []

    for i in range(len(df)):
        try:
            price = float(df["Close"].iloc[i])
            sma50 = float(df["SMA_50"].iloc[i])
            risk_ind = float(df["Risk_Indicator"].iloc[i])
            price_dev = float(df["Price_Deviation"].iloc[i])
            macd_v = float(df["MACD_V"].iloc[i])
        except (ValueError, TypeError):
            regimes.append("RANGO")
            continue

        if any(pd.isna(x) for x in [price, sma50, risk_ind, price_dev, macd_v]):
            regimes.append("RANGO")
            continue

        if i < 10:
            regimes.append("RANGO")
            continue

        # PASO 1: Detectar condiciones extremas de RIESGO (máxima prioridad)
        macd_extreme = abs(macd_v) > 150
        risk_extreme = abs(risk_ind) > 2.0

        if macd_extreme and risk_extreme:
            regime = "RIESGO+"
        elif macd_extreme or risk_extreme:
            regime = "RIESGO-"
        
        # PASO 2: Verificar si Price Deviation está en zona neutra (RANGO)
        elif -0.5 <= price_dev <= 0.5:
            regime = "RANGO"
        
        # PASO 3: Clasificar según lado del mercado + Price Deviation
        else:
            above_sma = price > sma50
            
            if above_sma and price_dev > 0.5:
                regime = "ALCISTA"
            elif not above_sma and price_dev < -0.5:
                regime = "BAJISTA"
            else:
                # Caso: precio y desviación en lados opuestos → RANGO
                regime = "RANGO"

        regimes.append(regime)

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

def main():
    st.set_page_config(layout="wide", page_title="Weekly Regime Analyzer")
    st.title("📊 Weekly Regime Analyzer — Comparación de Métodos")

    # Mantener estado en session_state
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
        
        # Recalcular clasificaciones
        df["Regime_MACDV"] = classify_by_macdv(df)
        df["Regime_ZScore"] = classify_by_zscore(df)
        st.session_state.df = df
        
        df_plot = df.tail(int(lookback_months * 4.33))
        current = df.iloc[-1]

        # ================= RESUMEN EJECUTIVO =================
        st.markdown("## 📋 Resumen Actual")
        
        # Extraer valores actuales
        precio = float(current["Close"])
        sma50 = float(current["SMA_50"])
        macd_v = float(current["MACD_V"])
        risk_ind = float(current["Risk_Indicator"])
        price_dev = float(current["Price_Deviation"])
        
        # Extraer regímenes
        regime_macdv = str(current["Regime_MACDV"])
        regime_zscore = str(current["Regime_ZScore"])
        
        # Determinar condiciones
        precio_lado = "ALCISTA (> SMA50)" if precio > sma50 else "BAJISTA (< SMA50)"
        macd_extreme = abs(macd_v) > 150
        risk_extreme = abs(risk_ind) > 2.0
        
        # TABLA 1: Valores Actuales
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Valores de Indicadores")
            values_data = {
                "Indicador": [
                    "Precio",
                    "SMA 50",
                    "Lado del Mercado",
                    "MACD-V",
                    "Risk Indicator (Z-MACD)",
                    "Price Deviation (Z-Price)"
                ],
                "Valor": [
                    f"${precio:.2f}",
                    f"${sma50:.2f}",
                    precio_lado,
                    f"{macd_v:.2f}",
                    f"{risk_ind:.2f}",
                    f"{price_dev:.2f}"
                ],
                "Estado": [
                    "✓",
                    "✓",
                    "🔵 ALCISTA" if precio > sma50 else "🔴 BAJISTA",
                    "⚠️ EXTREMO" if macd_extreme else "✓ Normal",
                    "⚠️ EXTREMO" if risk_extreme else "✓ Normal",
                    f"{'⬆️' if price_dev > 0.5 else '⬇️' if price_dev < -0.5 else '↔️'}"
                ]
            }
            values_df = pd.DataFrame(values_data)
            st.dataframe(values_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("### 🎯 Clasificación por Método")
            
            # Determinar nivel de riesgo
            if macd_extreme and risk_extreme:
                nivel_riesgo = "RIESGO+ (Ambos extremos)"
                color_riesgo = REGIME_COLORS["RIESGO+"]
            elif macd_extreme or risk_extreme:
                nivel_riesgo = "RIESGO- (Un extremo)"
                color_riesgo = REGIME_COLORS["RIESGO-"]
            else:
                nivel_riesgo = "Sin Riesgo Extremo"
                color_riesgo = "#00FF00"
            
            regime_data = {
                "Método": [
                    "MACD-V",
                    "Z-Score Precio",
                    "⚠️ Nivel de Riesgo"
                ],
                "Régimen": [
                    regime_macdv,
                    regime_zscore,
                    nivel_riesgo
                ],
                "Color": [
                    REGIME_COLORS.get(regime_macdv, "#FFFFFF"),
                    REGIME_COLORS.get(regime_zscore, "#FFFFFF"),
                    color_riesgo
                ]
            }
            
            regime_df = pd.DataFrame(regime_data)
            
            # Mostrar con colores
            for idx, row in regime_df.iterrows():
                st.markdown(
                    f'<div style="background-color:{row["Color"]};padding:8px;border-radius:5px;'
                    f'margin:4px 0;text-align:center;font-weight:bold;color:white;">'
                    f'{row["Método"]}: {row["Régimen"]}</div>',
                    unsafe_allow_html=True
                )

        # ================= ANÁLISIS DETALLADO =================
        st.markdown("---")
        st.markdown("## 🔬 Análisis Detallado por Método")
        
        col_analysis1, col_analysis2 = st.columns(2)
        
        with col_analysis1:
            st.markdown("### 🔹 Método 1: MACD-V")
            
            # Determinar qué condición se cumplió
            if macd_extreme and risk_extreme:
                condicion = "✅ PASO 1: Ambos indicadores extremos → RIESGO+"
            elif macd_extreme or risk_extreme:
                condicion = "✅ PASO 1: Un indicador extremo → RIESGO-"
            elif -50 <= macd_v <= 50:
                condicion = "✅ PASO 2: MACD-V en zona neutra [-50, +50] → RANGO"
            elif precio > sma50 and 50 < macd_v <= 150:
                condicion = "✅ PASO 3: Precio > SMA50 Y MACD-V > 50 → ALCISTA"
            elif precio < sma50 and -150 <= macd_v < -50:
                condicion = "✅ PASO 3: Precio < SMA50 Y MACD-V < -50 → BAJISTA"
            else:
                condicion = "✅ PASO 3: Señales mixtas (precio y MACD-V en lados opuestos) → RANGO"
            
            st.markdown(f"""
            **Resultado:** `{regime_macdv}`
            
            **Lógica aplicada:**
            {condicion}
            
            **Rangos de clasificación:**
            - RIESGO+: |MACD-V| > 150 **Y** |Risk Ind.| > 2.0
            - RIESGO-: |MACD-V| > 150 **O** |Risk Ind.| > 2.0
            - RANGO: MACD-V en [-50, +50]
            - ALCISTA: Precio > SMA50 **Y** MACD-V en (50, 150]
            - BAJISTA: Precio < SMA50 **Y** MACD-V en [-150, -50)
            
            **Ventaja:** Usa momentum normalizado por volatilidad
            """)
        
        with col_analysis2:
            st.markdown("### 🔹 Método 2: Z-Score Precio")
            
            # Determinar qué condición se cumplió
            if macd_extreme and risk_extreme:
                condicion = "✅ PASO 1: Ambos indicadores extremos → RIESGO+"
            elif macd_extreme or risk_extreme:
                condicion = "✅ PASO 1: Un indicador extremo → RIESGO-"
            elif -0.5 <= price_dev <= 0.5:
                condicion = "✅ PASO 2: Price Deviation en zona neutra [-0.5, +0.5] → RANGO"
            elif precio > sma50 and price_dev > 0.5:
                condicion = "✅ PASO 3: Precio > SMA50 Y Price Dev. > 0.5 → ALCISTA"
            elif precio < sma50 and price_dev < -0.5:
                condicion = "✅ PASO 3: Precio < SMA50 Y Price Dev. < -0.5 → BAJISTA"
            else:
                condicion = "✅ PASO 3: Señales mixtas (precio y desviación en lados opuestos) → RANGO"
            
            st.markdown(f"""
            **Resultado:** `{regime_zscore}`
            
            **Lógica aplicada:**
            {condicion}
            
            **Rangos de clasificación:**
            - RIESGO+: |MACD-V| > 150 **Y** |Risk Ind.| > 2.0
            - RIESGO-: |MACD-V| > 150 **O** |Risk Ind.| > 2.0
            - RANGO: Price Deviation en [-0.5, +0.5]
            - ALCISTA: Precio > SMA50 **Y** Price Dev. > +0.5
            - BAJISTA: Precio < SMA50 **Y** Price Dev. < -0.5
            
            **Ventaja:** Usa desviación estadística del precio
            """)

        # ================= LEYENDA =================
        st.markdown("---")
        st.markdown("### 🎨 Leyenda de Regímenes")
        cols = st.columns(5)
        for idx, (regime, color) in enumerate(REGIME_COLORS.items()):
            with cols[idx]:
                st.markdown(
                    f'<div style="background-color:{color};padding:8px;border-radius:5px;'
                    f'text-align:center;font-weight:bold;color:white;">{regime}</div>',
                    unsafe_allow_html=True
                )

        # ================= GRÁFICOS =================
        st.markdown("---")
        st.markdown("## 📈 Gráficos Comparativos")
        
        plt.style.use("dark_background")
        fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # ========== GRÁFICO 1: PRECIO + MACD-V ==========
        axs[0].plot(df_plot.index, df_plot["Close"], color="white", alpha=0.5, linewidth=1.5, label="Precio")
        axs[0].plot(df_plot.index, df_plot["SMA_50"], color="cyan", linewidth=2, alpha=0.7, label="SMA 50")

        for r, c in REGIME_COLORS.items():
            m = df_plot["Regime_MACDV"] == r
            if m.any():
                axs[0].scatter(df_plot[m].index, df_plot[m]["Close"], c=c, s=90, alpha=0.95, 
                             edgecolors='black', linewidths=1.5, zorder=5, label=f"{r}")

        last_price = float(df_plot["Close"].iloc[-1])
        axs[0].text(1.01, last_price, f'${last_price:.2f}', 
                   transform=axs[0].get_yaxis_transform(), 
                   fontsize=10, color='white', va='center', fontweight='bold')
        
        axs[0].set_title(f"{ticker} — Método MACD-V", fontsize=12, fontweight='bold', pad=10)
        axs[0].grid(alpha=0.25, linestyle='--')
        axs[0].set_ylabel("Precio ($)", fontsize=10)
        axs[0].legend(loc='upper left', fontsize=8, ncol=3)
        axs[0].yaxis.tick_right()
        axs[0].yaxis.set_label_position("right")

        # ========== GRÁFICO 2: PRECIO + Z-SCORE ==========
        axs[1].plot(df_plot.index, df_plot["Close"], color="white", alpha=0.5, linewidth=1.5, label="Precio")
        axs[1].plot(df_plot.index, df_plot["SMA_50"], color="cyan", linewidth=2, alpha=0.7, label="SMA 50")

        for r, c in REGIME_COLORS.items():
            m = df_plot["Regime_ZScore"] == r
            if m.any():
                axs[1].scatter(df_plot[m].index, df_plot[m]["Close"], c=c, s=90, alpha=0.95, 
                             edgecolors='black', linewidths=1.5, zorder=5, label=f"{r}")

        axs[1].text(1.01, last_price, f'${last_price:.2f}', 
                   transform=axs[1].get_yaxis_transform(), 
                   fontsize=10, color='white', va='center', fontweight='bold')
        
        axs[1].set_title(f"{ticker} — Método Z-Score Precio", fontsize=12, fontweight='bold', pad=10)
        axs[1].grid(alpha=0.25, linestyle='--')
        axs[1].set_ylabel("Precio ($)", fontsize=10)
        axs[1].legend(loc='upper left', fontsize=8, ncol=3)
        axs[1].yaxis.tick_right()
        axs[1].yaxis.set_label_position("right")

        # ========== GRÁFICO 3: INDICADORES DUALES ==========
        ax3_left = axs[2]
        ax3_right = ax3_left.twinx()
        
        # Risk Indicator (izquierda) con colores condicionales
        risk_colors = ['#FF0000' if abs(z) > 2 else '#87CEEB' for z in df_plot["Risk_Indicator"]]
        for i in range(len(df_plot) - 1):
            ax3_left.plot(df_plot.index[i:i+2], df_plot["Risk_Indicator"].iloc[i:i+2], 
                         color=risk_colors[i], linewidth=2)
        
        ax3_left.axhline(2, color="red", linestyle="--", alpha=0.6, linewidth=1.5, label="Umbral ±2.0")
        ax3_left.axhline(-2, color="red", linestyle="--", alpha=0.6, linewidth=1.5)
        ax3_left.axhline(0, color="gray", linestyle="-", alpha=0.4)
        ax3_left.fill_between(df_plot.index, 2, df_plot["Risk_Indicator"], 
                             where=(df_plot["Risk_Indicator"]>2), color="red", alpha=0.15)
        ax3_left.fill_between(df_plot.index, -2, df_plot["Risk_Indicator"], 
                             where=(df_plot["Risk_Indicator"]<-2), color="red", alpha=0.15)
        
        # MACD-V (derecha) con colores condicionales
        macd_colors = ['#FF0000' if abs(m) > 150 else '#FFFFFF' for m in df_plot["MACD_V"]]
        for i in range(len(df_plot) - 1):
            ax3_right.plot(df_plot.index[i:i+2], df_plot["MACD_V"].iloc[i:i+2], 
                          color=macd_colors[i], linewidth=2)
        
        ax3_right.axhline(0, color="gray", linestyle="-", alpha=0.4)
        ax3_right.axhline(50, color="green", linestyle=":", alpha=0.4, linewidth=1, label="Zona ±50")
        ax3_right.axhline(-50, color="red", linestyle=":", alpha=0.4, linewidth=1)
        ax3_right.axhline(150, color="red", linestyle="--", alpha=0.6, linewidth=1.5, label="Umbral ±150")
        ax3_right.axhline(-150, color="red", linestyle="--", alpha=0.6, linewidth=1.5)
        
        # Valores actuales
        last_risk = float(df_plot["Risk_Indicator"].iloc[-1])
        last_macd = float(df_plot["MACD_V"].iloc[-1])
        
        risk_color_final = '#FF0000' if abs(last_risk) > 2 else '#87CEEB'
        macd_color_final = '#FF0000' if abs(last_macd) > 150 else '#FFFFFF'
        
        ax3_left.text(-0.01, last_risk, f'{last_risk:.2f}σ', 
                     transform=ax3_left.get_yaxis_transform(), 
                     fontsize=10, color=risk_color_final, va='center', fontweight='bold', ha='right')
        
        ax3_right.text(1.01, last_macd, f'{last_macd:.2f}', 
                      transform=ax3_right.get_yaxis_transform(), 
                      fontsize=10, color=macd_color_final, va='center', fontweight='bold')
        
        ax3_left.set_title("Risk Indicator (izq) y MACD-V (der)", fontsize=12, fontweight='bold', pad=10)
        ax3_left.grid(alpha=0.25, linestyle='--')
        ax3_left.set_ylabel("Risk Indicator (Z-MACD)", fontsize=10, color='#87CEEB')
        ax3_right.set_ylabel("MACD-V", fontsize=10, color='#FFFFFF')
        ax3_left.set_xlabel("Fecha", fontsize=10)
        ax3_left.tick_params(axis='y', labelcolor='#87CEEB')
        ax3_right.tick_params(axis='y', labelcolor='#FFFFFF')
        ax3_left.legend(loc='upper left', fontsize=8)
        ax3_right.legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        st.pyplot(fig)

        # ================= DOCUMENTACIÓN =================
        st.markdown("---")
        st.markdown("## 📖 Documentación de Métodos")
        
        with st.expander("🔍 ¿Qué es el Risk Indicator (Z-Score MACD)?"):
            st.markdown("""
            ### Risk Indicator (Z-Score del Histograma MACD-V)
            
            **Propósito:** Detector de condiciones extremas de mercado
            
            **Cálculo:**
            1. Se toma el histograma del MACD-V (diferencia entre MACD-V y su señal)
            2. Se calcula su media y desviación estándar en las últimas 20 semanas
            3. Se normaliza: `Z = (Histograma - Media) / Desviación Estándar`
            4. Se ajusta por curtosis (exceso de eventos extremos)
            
            **Interpretación:**
            - `|Z| > 2.0` → El momentum está en niveles estadísticamente anormales (solo 5% del tiempo)
            - `|Z| < 2.0` → El momentum está dentro de rangos normales
            
            **Uso:** Se combina con MACD-V para detectar RIESGO+ o RIESGO-
            """)
        
        with st.expander("🔍 ¿Qué es el Price Deviation (Z-Score del Precio)?"):
            st.markdown("""
            ### Price Deviation (Z-Score del Precio)
            
            **Propósito:** Medir qué tan "estirado" está el precio respecto a su normalidad
            
            **Cálculo:**
            1. Se calcula el precio promedio de las últimas 50 semanas
            2. Se calcula la desviación estándar del precio en ese periodo
            3. Se normaliza: `Z = (Precio Actual - Promedio) / Desviación Estándar`
            
            **Interpretación:**
            - `Z > +0.5` → Precio significativamente arriba de su promedio
            - `Z < -0.5` → Precio significativamente abajo de su promedio
            - `-0.5 ≤ Z ≤ +0.5` → Precio cerca de su promedio (neutral)
            
            **Uso:** Método alternativo para clasificar regímenes (compite con MACD-V)
            """)
        
        with st.expander("📊 Comparación de Métodos"):
            st.markdown("""
            ### Método 1: MACD-V
            **Ventajas:**
            - Captura cambios de momentum ajustados por volatilidad
            - Responde rápido a cambios de tendencia
            - Más sensible a aceleración/desaceleración
            
            **Desventajas:**
            - Puede generar señales prematuras en mercados laterales
            - Requiere filtro adicional para evitar ruido
            
            ---
            
            ### Método 2: Z-Score Precio
            **Ventajas:**
            - Mide desviación estadística pura del precio
            - Más estable en mercados laterales
            - Identifica mejor zonas de normalización
            
            **Desventajas:**
            - Responde más lento a cambios bruscos
            - Puede tardar en confirmar tendencias nuevas
            
            ---
            
            ### ¿Cuál usar?
            - **Swing Trading agresivo:** Método MACD-V
            - **Position Trading conservador:** Método Z-Score Precio
            - **Mejor opción:** Combinar ambos para confluencias
            """)

    elif st.session_state.analyzed and st.session_state.df is None:
        st.error("❌ No se pudo descargar data. Verifica el ticker.")

if __name__ == "__main__":
    if check_password():
        main()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder.")
