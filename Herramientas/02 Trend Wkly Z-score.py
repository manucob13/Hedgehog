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

# Colores para todos los regímenes (incluido RIESGO para gráficos)
REGIME_COLORS = {
    "ALCISTA": "#00FF00",      # Verde brillante
    "BAJISTA": "#FF0000",      # Rojo brillante
    "RANGO": "#FFD700",        # Dorado
    "RIESGO": "#9370DB",       # Morado (cuando hay riesgo en gráficos)
}

# Colores para el análisis de riesgo en tabla
RISK_COLORS = {
    "RIESGO-": "#9370DB",      # Morado medio
    "RIESGO+": "#4B0082",      # Índigo oscuro
    "Sin Riesgo": "#00FF00",   # Verde
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
# CLASIFICACIÓN POR MACD-V (CON RIESGO)
# ============================================================================

def classify_by_macdv(df):
    """
    Clasifica en: ALCISTA, BAJISTA, RANGO, RIESGO
    RIESGO se muestra cuando hay condiciones extremas
    """
    regimes = []

    for i in range(len(df)):
        try:
            price = float(df["Close"].iloc[i])
            sma50 = float(df["SMA_50"].iloc[i])
            macd_v = float(df["MACD_V"].iloc[i])
            z_macd = float(df["Z_Score_MACD"].iloc[i])
        except (ValueError, TypeError):
            regimes.append("RANGO")
            continue

        if any(pd.isna(x) for x in [price, sma50, macd_v, z_macd]):
            regimes.append("RANGO")
            continue

        if i < 10:
            regimes.append("RANGO")
            continue

        # PASO 1: Detectar condiciones de RIESGO (máxima prioridad)
        macd_extreme = abs(macd_v) >= 151
        z_extreme = abs(z_macd) > 2.0

        if macd_extreme or z_extreme:
            # Si hay cualquier condición extrema → RIESGO en gráfico
            regime = "RIESGO"
        # PASO 2: Determinar lado del mercado
        else:
            above_sma = price > sma50

            # PASO 3: Clasificar por MACD-V (sin considerar RIESGO)
            if -50 <= macd_v <= 50:
                # Zona neutral
                regime = "RANGO"
            elif above_sma:  # Precio > SMA50
                if 51 <= macd_v <= 150:
                    regime = "ALCISTA"
                else:
                    # MACD-V fuera de rango alcista o en contradicción
                    regime = "RANGO"
            else:  # Precio < SMA50
                if -150 <= macd_v <= -51:
                    regime = "BAJISTA"
                else:
                    # MACD-V fuera de rango bajista o en contradicción
                    regime = "RANGO"

        regimes.append(regime)

    return regimes

# ============================================================================
# CLASIFICACIÓN POR Z-SCORE DE PRECIO (CON RIESGO)
# ============================================================================

def classify_by_zscore(df):
    """
    Clasifica en: ALCISTA, BAJISTA, RANGO, RIESGO
    RIESGO se muestra cuando hay condiciones extremas
    """
    regimes = []

    for i in range(len(df)):
        try:
            price = float(df["Close"].iloc[i])
            sma50 = float(df["SMA_50"].iloc[i])
            z_price = float(df["Z_Score_Price"].iloc[i])
            z_macd = float(df["Z_Score_MACD"].iloc[i])
            macd_v = float(df["MACD_V"].iloc[i])
        except (ValueError, TypeError):
            regimes.append("RANGO")
            continue

        if any(pd.isna(x) for x in [price, sma50, z_price, z_macd, macd_v]):
            regimes.append("RANGO")
            continue

        if i < 10:
            regimes.append("RANGO")
            continue

        # PASO 1: Detectar condiciones de RIESGO (máxima prioridad)
        macd_extreme = abs(macd_v) >= 151
        z_extreme = abs(z_macd) > 2.0

        if macd_extreme or z_extreme:
            # Si hay cualquier condición extrema → RIESGO en gráfico
            regime = "RIESGO"
        # PASO 2: Determinar lado del mercado
        else:
            above_sma = price > sma50

            # PASO 3: Clasificar por Z-Score Precio (sin considerar RIESGO)
            if -0.5 <= z_price <= 0.5:
                # Zona neutral
                regime = "RANGO"
            elif above_sma:  # Precio > SMA50
                if z_price > 0.5:
                    regime = "ALCISTA"
                else:
                    # Z-Score no confirma alcista
                    regime = "RANGO"
            else:  # Precio < SMA50
                if z_price < -0.5:
                    regime = "BAJISTA"
                else:
                    # Z-Score no confirma bajista
                    regime = "RANGO"

        regimes.append(regime)

    return regimes

# ============================================================================
# ANÁLISIS DE RIESGO (SEPARADO)
# ============================================================================

def analyze_risk(df):
    """
    Analiza condiciones de riesgo de manera independiente
    Retorna: RIESGO+, RIESGO-, o Sin Riesgo
    """
    risk_levels = []

    for i in range(len(df)):
        try:
            macd_v = float(df["MACD_V"].iloc[i])
            z_macd = float(df["Z_Score_MACD"].iloc[i])
        except (ValueError, TypeError):
            risk_levels.append("Sin Riesgo")
            continue

        if any(pd.isna(x) for x in [macd_v, z_macd]):
            risk_levels.append("Sin Riesgo")
            continue

        # Detectar condiciones extremas (umbrales corregidos)
        macd_extreme = abs(macd_v) >= 151  # Corregido: >= 151
        z_extreme = abs(z_macd) > 2.0

        if macd_extreme and z_extreme:
            risk = "RIESGO+"
        elif macd_extreme or z_extreme:
            risk = "RIESGO-"
        else:
            risk = "Sin Riesgo"

        risk_levels.append(risk)

    return risk_levels

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
    st.title("📊 Weekly Regime Analyzer — MACD-V vs Z-Score Precio")

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
        df["Risk_Level"] = analyze_risk(df)
        st.session_state.df = df
        
        df_plot = df.tail(int(lookback_months * 4.33))
        current = df.iloc[-1]

        # Extraer valores actuales
        regime_macdv = df["Regime_MACDV"].iloc[-1]
        regime_zscore = df["Regime_ZScore"].iloc[-1]
        risk_level = df["Risk_Level"].iloc[-1]
        
        precio = float(current["Close"])
        sma50 = float(current["SMA_50"])
        macd_v = float(current["MACD_V"])
        z_price = float(current["Z_Score_Price"])
        z_macd = float(current["Z_Score_MACD"])
        
        precio_above_sma = "✅ SÍ" if precio > sma50 else "❌ NO"
        
        # Determinar confluencia
        if regime_macdv == regime_zscore:
            confluencia = f"✅ Ambos {regime_macdv}"
            confluencia_color = REGIME_COLORS.get(regime_macdv, "#FFFFFF")
        else:
            confluencia = f"⚠️ Discrepan"
            confluencia_color = "#FFA500"

        # ================= 3 TABLAS EN UNA FILA =================
        col1, col2, col3 = st.columns(3)
        
        # ================= TABLA 1: MÉTODO MACD-V =================
        with col1:
            st.markdown("#### 📊 Método MACD-V")
            
            macdv_color = REGIME_COLORS.get(regime_macdv, "#FFFFFF")
            
            st.markdown(f"""
            <table style="width:100%; border-collapse: collapse; margin-bottom: 15px;">
                <tr>
                    <td colspan="2" style="padding: 10px; border: 1px solid #444; background-color: {macdv_color}; color: black; font-weight: bold; text-align: center; font-size: 18px;">
                        {regime_macdv}
                    </td>
                </tr>
                <tr>
                    <td style="padding: 6px; border: 1px solid #444; font-size: 11px; background-color: #2e2e2e;">Precio</td>
                    <td style="padding: 6px; border: 1px solid #444; font-weight: bold; font-size: 13px;">${precio:.2f}</td>
                </tr>
                <tr>
                    <td style="padding: 6px; border: 1px solid #444; font-size: 11px; background-color: #2e2e2e;">SMA 50</td>
                    <td style="padding: 6px; border: 1px solid #444; font-size: 11px;">${sma50:.2f}</td>
                </tr>
                <tr>
                    <td style="padding: 6px; border: 1px solid #444; font-size: 11px; background-color: #2e2e2e;">Precio > SMA50</td>
                    <td style="padding: 6px; border: 1px solid #444; font-size: 11px;">{precio_above_sma}</td>
                </tr>
                <tr>
                    <td style="padding: 6px; border: 1px solid #444; font-size: 11px; background-color: #2e2e2e;">MACD-V</td>
                    <td style="padding: 6px; border: 1px solid #444; font-weight: bold; font-size: 11px;">{macd_v:.2f}</td>
                </tr>
            </table>
            """, unsafe_allow_html=True)

        # ================= TABLA 2: MÉTODO Z-SCORE =================
        with col2:
            st.markdown("#### 📊 Método Z-Score")
            
            zscore_color = REGIME_COLORS.get(regime_zscore, "#FFFFFF")
            
            st.markdown(f"""
            <table style="width:100%; border-collapse: collapse; margin-bottom: 15px;">
                <tr>
                    <td colspan="2" style="padding: 10px; border: 1px solid #444; background-color: {zscore_color}; color: black; font-weight: bold; text-align: center; font-size: 18px;">
                        {regime_zscore}
                    </td>
                </tr>
                <tr>
                    <td style="padding: 6px; border: 1px solid #444; font-size: 11px; background-color: #2e2e2e;">Precio</td>
                    <td style="padding: 6px; border: 1px solid #444; font-weight: bold; font-size: 13px;">${precio:.2f}</td>
                </tr>
                <tr>
                    <td style="padding: 6px; border: 1px solid #444; font-size: 11px; background-color: #2e2e2e;">SMA 50</td>
                    <td style="padding: 6px; border: 1px solid #444; font-size: 11px;">${sma50:.2f}</td>
                </tr>
                <tr>
                    <td style="padding: 6px; border: 1px solid #444; font-size: 11px; background-color: #2e2e2e;">Precio > SMA50</td>
                    <td style="padding: 6px; border: 1px solid #444; font-size: 11px;">{precio_above_sma}</td>
                </tr>
                <tr>
                    <td style="padding: 6px; border: 1px solid #444; font-size: 11px; background-color: #2e2e2e;">Z-Score Precio</td>
                    <td style="padding: 6px; border: 1px solid #444; font-weight: bold; font-size: 11px;">{z_price:.2f}</td>
                </tr>
            </table>
            """, unsafe_allow_html=True)

        # ================= TABLA 3: CONFLUENCIA (SEMÁFORO) =================
        with col3:
            st.markdown("#### 🎯 Confluencia")
            
            st.markdown(f"""
            <div style="background-color:{confluencia_color}; padding: 50px 10px; border-radius: 8px; margin-top: 8px;
                        text-align: center; font-weight: bold; color: black; font-size: 16px; margin-bottom: 15px;">
                {confluencia}
            </div>
            """, unsafe_allow_html=True)

        # ================= ANÁLISIS DE RIESGO (MEJORADO Y ALINEADO) =================
        st.markdown("### ⚠️ Análisis de Riesgo")
        
        macd_extreme = abs(macd_v) >= 151
        z_extreme = abs(z_macd) > 2.0
        
        # Interpretaciones
        if macd_extreme and z_extreme:
            interpretacion = "⚠️⚠️ ALTA PRECAUCIÓN"
            interp_color = "#FF0000"
        elif macd_extreme:
            interpretacion = "⚠️ MACD-V sobre-extendido"
            interp_color = "#FFA500"
        elif z_extreme:
            interpretacion = "⚠️ Momentum anormal"
            interp_color = "#FFA500"
        else:
            interpretacion = "✅ Condiciones normales"
            interp_color = "#00FF00"
        
        risk_color = RISK_COLORS.get(risk_level, "#FFFFFF")
        
        col_risk1, col_risk2 = st.columns([3, 1])
        
        with col_risk1:
            st.markdown(f"""
            <table style="width:100%; border-collapse: collapse; margin-bottom: 15px;">
                <tr style="background-color: #1e1e1e;">
                    <th style="padding: 8px; text-align: left; border: 1px solid #444; font-size: 12px;">Indicador</th>
                    <th style="padding: 8px; text-align: center; border: 1px solid #444; font-size: 12px;">Valor</th>
                    <th style="padding: 8px; text-align: center; border: 1px solid #444; font-size: 12px;">¿Extremo?</th>
                    <th style="padding: 8px; text-align: left; border: 1px solid #444; font-size: 12px;">Umbral</th>
                </tr>
                <tr>
                    <td style="padding: 6px; border: 1px solid #444; font-size: 11px;">Z-Score MACD</td>
                    <td style="padding: 6px; border: 1px solid #444; text-align: center; font-weight: bold; font-size: 11px;">{z_macd:.2f}</td>
                    <td style="padding: 6px; border: 1px solid #444; text-align: center; font-size: 11px;">{"🔴 SÍ" if z_extreme else "🟢 NO"}</td>
                    <td style="padding: 6px; border: 1px solid #444; font-size: 11px;">|Z| > 2.0</td>
                </tr>
                <tr>
                    <td style="padding: 6px; border: 1px solid #444; font-size: 11px;">MACD-V</td>
                    <td style="padding: 6px; border: 1px solid #444; text-align: center; font-weight: bold; font-size: 11px;">{macd_v:.2f}</td>
                    <td style="padding: 6px; border: 1px solid #444; text-align: center; font-size: 11px;">{"🔴 SÍ" if macd_extreme else "🟢 NO"}</td>
                    <td style="padding: 6px; border: 1px solid #444; font-size: 11px;">|MACD-V| ≥ 151</td>
                </tr>
            </table>
            """, unsafe_allow_html=True)
        
        with col_risk2:
            st.markdown(f"""
            <div style="margin-top: 34px;">
                <div style="background-color: {risk_color}; padding: 20px 10px; border-radius: 8px; margin-bottom: 8px;
                            text-align: center; font-weight: bold; color: black; font-size: 18px;">
                    {risk_level}
                </div>
                <div style="background-color: {interp_color}; padding: 10px; border-radius: 8px;
                            text-align: center; font-weight: bold; color: black; font-size: 11px;">
                    {interpretacion}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ================= LEYENDA =================
        st.markdown("### 🎨 Leyenda")
        cols = st.columns(4)
        for idx, (regime, color) in enumerate(REGIME_COLORS.items()):
            with cols[idx]:
                st.markdown(
                    f'<div style="background-color:{color};padding:6px;border-radius:5px;'
                    f'text-align:center;font-weight:bold;color:black;border:2px solid white;font-size:12px;">{regime}</div>',
                    unsafe_allow_html=True
                )

        # ================= GRÁFICOS (MÁS COMPACTOS) =================
        st.markdown("---")
        plt.style.use("dark_background")
        fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True)

        # ========== GRÁFICO 1: PRECIO + MACD-V ==========
        axs[0].plot(df_plot.index, df_plot["Close"], color="white", alpha=0.5, linewidth=1.3, label="Precio")
        axs[0].plot(df_plot.index, df_plot["SMA_50"], color="cyan", linewidth=1.8, alpha=0.7, label="SMA 50")

        for r, c in REGIME_COLORS.items():
            m = df_plot["Regime_MACDV"] == r
            if m.any():
                axs[0].scatter(df_plot[m].index, df_plot[m]["Close"], c=c, s=70, alpha=0.95, 
                             edgecolors='black', linewidths=1.2, zorder=5, label=f"{r}")

        last_price = float(df_plot["Close"].iloc[-1])
        axs[0].text(1.01, last_price, f'${last_price:.2f}', 
                   transform=axs[0].get_yaxis_transform(), 
                   fontsize=8, color='white', va='center', fontweight='bold')
        
        axs[0].set_title(f"{ticker} — Método MACD-V", fontsize=10, fontweight='bold', pad=6)
        axs[0].grid(alpha=0.25, linestyle='--')
        axs[0].set_ylabel("Precio ($)", fontsize=8)
        axs[0].legend(loc='upper left', fontsize=6.5, ncol=2)
        axs[0].yaxis.tick_right()
        axs[0].yaxis.set_label_position("right")
        axs[0].tick_params(axis='both', labelsize=7)

        # ========== GRÁFICO 2: PRECIO + Z-SCORE ==========
        axs[1].plot(df_plot.index, df_plot["Close"], color="white", alpha=0.5, linewidth=1.3, label="Precio")
        axs[1].plot(df_plot.index, df_plot["SMA_50"], color="cyan", linewidth=1.8, alpha=0.7, label="SMA 50")

        for r, c in REGIME_COLORS.items():
            m = df_plot["Regime_ZScore"] == r
            if m.any():
                axs[1].scatter(df_plot[m].index, df_plot[m]["Close"], c=c, s=70, alpha=0.95, 
                             edgecolors='black', linewidths=1.2, zorder=5, label=f"{r}")

        axs[1].text(1.01, last_price, f'${last_price:.2f}', 
                   transform=axs[1].get_yaxis_transform(), 
                   fontsize=8, color='white', va='center', fontweight='bold')
        
        axs[1].set_title(f"{ticker} — Método Z-Score Precio", fontsize=10, fontweight='bold', pad=6)
        axs[1].grid(alpha=0.25, linestyle='--')
        axs[1].set_ylabel("Precio ($)", fontsize=8)
        axs[1].legend(loc='upper left', fontsize=6.5, ncol=2)
        axs[1].yaxis.tick_right()
        axs[1].yaxis.set_label_position("right")
        axs[1].tick_params(axis='both', labelsize=7)

        # ========== GRÁFICO 3: INDICADORES DE RIESGO ==========
        ax3_left = axs[2]
        ax3_right = ax3_left.twinx()
        
        # Z-Score MACD con colores condicionales
        z_colors = ['#FF0000' if abs(z) > 2 else '#87CEEB' for z in df_plot["Z_Score_MACD"]]
        for i in range(len(df_plot) - 1):
            ax3_left.plot(df_plot.index[i:i+2], df_plot["Z_Score_MACD"].iloc[i:i+2], 
                         color=z_colors[i], linewidth=1.8)
        
        ax3_left.axhline(2, color="red", linestyle="--", alpha=0.6, linewidth=1.3, label="Umbral ±2.0")
        ax3_left.axhline(-2, color="red", linestyle="--", alpha=0.6, linewidth=1.3)
        ax3_left.axhline(0, color="gray", linestyle="-", alpha=0.4)
        ax3_left.fill_between(df_plot.index, 2, df_plot["Z_Score_MACD"], 
                             where=(df_plot["Z_Score_MACD"]>2), color="red", alpha=0.15)
        ax3_left.fill_between(df_plot.index, -2, df_plot["Z_Score_MACD"], 
                             where=(df_plot["Z_Score_MACD"]<-2), color="red", alpha=0.15)
        
        # MACD-V con colores condicionales
        macd_colors = ['#FF0000' if abs(m) >= 151 else '#FFFFFF' for m in df_plot["MACD_V"]]
        for i in range(len(df_plot) - 1):
            ax3_right.plot(df_plot.index[i:i+2], df_plot["MACD_V"].iloc[i:i+2], 
                          color=macd_colors[i], linewidth=1.8)
        
        ax3_right.axhline(0, color="gray", linestyle="-", alpha=0.4)
        ax3_right.axhline(50, color="green", linestyle=":", alpha=0.4, linewidth=1, label="Zona ±50")
        ax3_right.axhline(-50, color="red", linestyle=":", alpha=0.4, linewidth=1)
        ax3_right.axhline(151, color="red", linestyle="--", alpha=0.6, linewidth=1.3, label="Umbral ±151")
        ax3_right.axhline(-151, color="red", linestyle="--", alpha=0.6, linewidth=1.3)
        
        last_z = float(df_plot["Z_Score_MACD"].iloc[-1])
        last_macd = float(df_plot["MACD_V"].iloc[-1])
        
        z_color_final = '#FF0000' if abs(last_z) > 2 else '#87CEEB'
        macd_color_final = '#FF0000' if abs(last_macd) >= 151 else '#FFFFFF'
        
        ax3_left.text(-0.01, last_z, f'{last_z:.2f}σ', 
                     transform=ax3_left.get_yaxis_transform(), 
                     fontsize=8, color=z_color_final, va='center', fontweight='bold', ha='right')
        
        ax3_right.text(1.01, last_macd, f'{last_macd:.2f}', 
                      transform=ax3_right.get_yaxis_transform(), 
                      fontsize=8, color=macd_color_final, va='center', fontweight='bold')
        
        ax3_left.set_title("Indicadores de Riesgo: Z-Score MACD (izq) y MACD-V (der)", 
                          fontsize=10, fontweight='bold', pad=6)
        ax3_left.grid(alpha=0.25, linestyle='--')
        ax3_left.set_ylabel("Z-Score MACD", fontsize=8, color='#87CEEB')
        ax3_right.set_ylabel("MACD-V", fontsize=8, color='#FFFFFF')
        ax3_left.set_xlabel("Fecha", fontsize=8)
        ax3_left.tick_params(axis='y', labelcolor='#87CEEB', labelsize=7)
        ax3_right.tick_params(axis='y', labelcolor='#FFFFFF', labelsize=7)
        ax3_left.tick_params(axis='x', labelsize=7)
        ax3_left.legend(loc='upper left', fontsize=6.5)
        ax3_right.legend(loc='upper right', fontsize=6.5)

        plt.tight_layout()
        st.pyplot(fig)

        # ================= DOCUMENTACIÓN (COMPACTA) =================
        st.markdown("---")
        with st.expander("📖 Ver Lógica de Clasificación"):
            col_logic1, col_logic2 = st.columns(2)
            
            with col_logic1:
                st.markdown("#### 🔹 Método MACD-V")
                st.markdown("""
                **PASO 1: Detectar RIESGO**
                - |MACD-V| ≥ 151 **O** |Z-Score MACD| > 2.0 → **RIESGO**
                
                **PASO 2: Determinar lado**
                - Precio > SMA50 → Lado alcista
                - Precio < SMA50 → Lado bajista
                
                **PASO 3: Clasificar**
                - Alcista: MACD-V [51, 150]
                - Bajista: MACD-V [-150, -51]
                - Rango: MACD-V [-50, +50]
                """)
            
            with col_logic2:
                st.markdown("#### 🔹 Método Z-Score Precio")
                st.markdown("""
                **PASO 1: Detectar RIESGO**
                - |MACD-V| ≥ 151 **O** |Z-Score MACD| > 2.0 → **RIESGO**
                
                **PASO 2: Determinar lado**
                - Precio > SMA50 → Lado alcista
                - Precio < SMA50 → Lado bajista
                
                **PASO 3: Clasificar**
                - Alcista: Z-Score > +0.5
                - Bajista: Z-Score < -0.5
                - Rango: Z-Score [-0.5, +0.5]
                """)

    elif st.session_state.analyzed and st.session_state.df is None:
        st.error("❌ No se pudo descargar data. Verifica el ticker.")

if __name__ == "__main__":
    if check_password():
        main()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder.")
