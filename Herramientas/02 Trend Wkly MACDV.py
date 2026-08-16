import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# AUTENTICACIÓN - Importar desde utils
# ============================================================================

try:
    from utils.utils import check_password
except ImportError:
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

        if st.session_state.get("password_correct", False):
            with st.sidebar:
                st.success(f"✅ Sesión activa: {st.session_state.get('username', 'Usuario')}")
                if st.button("🚪 Cerrar Sesión", use_container_width=True):
                    st.session_state["password_correct"] = False
                    if "username" in st.session_state:
                        del st.session_state["username"]
                    st.rerun()
            return True

        login_form()
        return False

# ============================================================================
# COLORES DE REGÍMENES
# ============================================================================

REGIME_COLORS = {
    "ALCISTA": "#00FF00",
    "BAJISTA": "#FF0000",
    "RANGO":   "#FFD700",
    "RIESGO":  "#9370DB",
}

RISK_COLORS = {
    "RIESGO+":    "#9370DB",
    "Sin Riesgo": "#00FF00",
}

# ============================================================================
# INDICADORES (solo lo necesario para MACD-V; Z-Score de precio y la
# corrección por kurtosis del Z-Score de MACD fueron removidos)
# ============================================================================

def calculate_indicators(df, fast=12, slow=26, signal=9):
    close = df["Close"].astype(float).squeeze()
    high  = df["High"].astype(float).squeeze()
    low   = df["Low"].astype(float).squeeze()

    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(slow).mean()

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_v      = ((ema_fast - ema_slow) / atr) * 100
    signal_line = macd_v.ewm(span=signal, adjust=False).mean()
    histogram   = macd_v - signal_line

    df["MACD_V"]           = macd_v
    df["MACD_V_Signal"]    = signal_line
    df["MACD_V_Histogram"] = histogram

    return df

# ============================================================================
# CLASIFICACIÓN POR MACD-V (único método de régimen)
# ============================================================================

def classify_by_macdv(df):
    regimes = []

    for i in range(len(df)):
        try:
            price  = float(df["Close"].iloc[i])
            sma50  = float(df["SMA_50"].iloc[i])
            macd_v = float(df["MACD_V"].iloc[i])
        except (ValueError, TypeError):
            regimes.append("RANGO")
            continue

        if any(pd.isna(x) for x in [price, sma50, macd_v]):
            regimes.append("RANGO")
            continue

        if i < 10:
            regimes.append("RANGO")
            continue

        above_sma = price > sma50

        if -50 <= macd_v <= 50:
            regime = "RANGO"
        elif above_sma:
            if 51 <= macd_v <= 150:
                regime = "ALCISTA"
            elif macd_v > 150:
                regime = "RIESGO"
            else:
                regime = "RANGO"
        else:
            if -150 <= macd_v <= -51:
                regime = "BAJISTA"
            elif macd_v < -150:
                regime = "RIESGO"
            else:
                regime = "RANGO"

        regimes.append(regime)

    return regimes

# ============================================================================
# ANÁLISIS DE RIESGO (depende únicamente de MACD-V)
# ============================================================================

def analyze_risk(df, macd_extreme_threshold=151):
    risk_levels = []

    for i in range(len(df)):
        try:
            macd_v = float(df["MACD_V"].iloc[i])
        except (ValueError, TypeError):
            risk_levels.append("Sin Riesgo")
            continue

        if pd.isna(macd_v):
            risk_levels.append("Sin Riesgo")
            continue

        macd_extreme = abs(macd_v) >= macd_extreme_threshold
        risk_levels.append("RIESGO+" if macd_extreme else "Sin Riesgo")

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

    # ── APLANAR MultiIndex de columnas (yfinance versiones recientes) ──────────
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # ──────────────────────────────────────────────────────────────────────────

    df = df.reset_index()
    df = df[["Date", "Open", "High", "Low", "Close"]].copy()
    df.set_index("Date", inplace=True)

    df.columns = [str(c) for c in df.columns]

    df["SMA_50"]  = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200, min_periods=1).mean()  # min_periods=1: nunca queda vacía/ausente
    df["EMA_8"]   = df["Close"].ewm(span=8,  adjust=False).mean()
    df["EMA_21"]  = df["Close"].ewm(span=21, adjust=False).mean()

    df = calculate_indicators(df)

    # Solo se exige que MACD_V y SMA_50 sean válidos para conservar la fila;
    # SMA_200 puede legítimamente no tener 200 datos aún (ticker joven o
    # 'Histórico (años)' bajo) y no debe tirar filas por eso.
    df.dropna(subset=["MACD_V", "SMA_50"], inplace=True)

    return df

# ============================================================================
# UI STREAMLIT
# ============================================================================

def main():
    st.set_page_config(layout="wide", page_title="Weekly Regime Analyzer")
    st.title("📊 Weekly Regime Analyzer — MACD-V")
    st.caption("Alcista · Bajista · Rango · Riesgo — clasificación por momentum normalizado por volatilidad (MACD-V)")

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

        df = st.session_state.df.copy()

        # Salvaguarda: si por cualquier motivo (p.ej. caché antigua de una
        # versión anterior de este script) el DataFrame no trae SMA_200,
        # se calcula aquí mismo en vez de fallar con KeyError.
        if "SMA_200" not in df.columns:
            df["SMA_200"] = df["Close"].rolling(200, min_periods=1).mean()

        df["Regime_MACDV"] = classify_by_macdv(df)
        df["Risk_Level"]   = analyze_risk(df)
        st.session_state.df = df

        df_plot = df.tail(int(lookback_months * 4.33))
        current = df.iloc[-1]

        regime_macdv = df["Regime_MACDV"].iloc[-1]
        risk_level   = df["Risk_Level"].iloc[-1]

        precio  = float(current["Close"])
        sma50   = float(current["SMA_50"])
        sma200_val = current.get("SMA_200", np.nan)
        sma200  = float(sma200_val) if pd.notna(sma200_val) else np.nan
        macd_v  = float(current["MACD_V"])

        precio_above_sma = "✅ SÍ" if precio > sma50 else "❌ NO"

        # ================= TARJETA DE ESTADO ACTUAL =================
        macdv_color = REGIME_COLORS.get(regime_macdv, "#FFFFFF")
        st.markdown(
            f'<div style="padding:10px;border:1px solid #444;background-color:{macdv_color};'
            f'color:black;font-weight:bold;text-align:center;font-size:20px;border-radius:4px;">'
            f'{regime_macdv}</div>',
            unsafe_allow_html=True,
        )

        info_cols = st.columns(5)
        info_cols[0].metric("Precio", f"${precio:.2f}")
        info_cols[1].metric("SMA 50", f"${sma50:.2f}")
        info_cols[2].metric("SMA 200", f"${sma200:.2f}" if pd.notna(sma200) else "N/A")
        info_cols[3].metric("Precio > SMA50", precio_above_sma)
        info_cols[4].metric("MACD-V", f"{macd_v:.2f}")

        # ================= ANÁLISIS DE RIESGO =================
        macd_extreme = abs(macd_v) >= 151
        risk_color   = RISK_COLORS.get(risk_level, "#FFFFFF")

        st.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <h3 style="margin: 0; padding: 0; display: inline;">⚠️ Análisis de Riesgo</h3>
            <div style="width: 35px; height: 35px; border-radius: 50%; background-color: {risk_color}; 
                        margin-left: 15px; display: flex; justify-content: center; align-items: center;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.3); font-size: 9px; font-weight: bold; color: black;">
                {risk_level.replace("Sin Riesgo", "OK").replace("RIESGO+", "R+")}
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_risk1, col_risk2 = st.columns(2)

        with col_risk1:
            st.markdown(f"""
            <table style="width:100%; border-collapse: collapse; margin-bottom: 10px;">
                <tr style="background-color: #1e1e1e;">
                    <th style="padding: 6px; text-align: left; border: 1px solid #444; font-size: 11px;">Indicador</th>
                    <th style="padding: 6px; text-align: center; border: 1px solid #444; font-size: 11px;">Valor</th>
                </tr>
                <tr>
                    <td style="padding: 4px 6px; border: 1px solid #444; font-size: 10px;">MACD-V</td>
                    <td style="padding: 4px 6px; border: 1px solid #444; text-align: center; font-weight: bold; font-size: 10px;">{macd_v:.2f}</td>
                </tr>
                <tr>
                    <td style="padding: 4px 6px; border: 1px solid #444; font-size: 10px;">¿Sobre-extendido?</td>
                    <td style="padding: 4px 6px; border: 1px solid #444; text-align: center; font-size: 10px;">{"🔴 SÍ" if macd_extreme else "🟢 NO"}</td>
                </tr>
            </table>
            """, unsafe_allow_html=True)

        with col_risk2:
            st.markdown(f"""
            <table style="width:100%; border-collapse: collapse; margin-bottom: 10px;">
                <tr style="background-color: #1e1e1e;">
                    <th style="padding: 6px; text-align: left; border: 1px solid #444; font-size: 11px;">Umbral</th>
                </tr>
                <tr>
                    <td style="padding: 4px 6px; border: 1px solid #444; font-size: 10px;">|MACD-V| ≥ 151 → RIESGO+</td>
                </tr>
                <tr>
                    <td style="padding: 4px 6px; border: 1px solid #444; font-size: 10px;">|MACD-V| ≤ 50 → RANGO</td>
                </tr>
            </table>
            """, unsafe_allow_html=True)

        # ================= LEYENDA =================
        st.markdown("### 🎨 Leyenda")
        cols = st.columns(4)
        for idx, (regime, color) in enumerate(REGIME_COLORS.items()):
            with cols[idx]:
                st.markdown(
                    f'<div style="background-color:{color};padding:5px;border-radius:5px;'
                    f'text-align:center;font-weight:bold;color:black;border:2px solid white;font-size:11px;">{regime}</div>',
                    unsafe_allow_html=True
                )

        # ================= GRÁFICOS =================
        st.markdown("---")
        plt.style.use("dark_background")
        fig, axs = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

        # ── Gráfico 1: Precio + Régimen MACD-V ──────────────────────────────
        axs[0].plot(df_plot.index, df_plot["Close"],   color="white",   alpha=0.5,  linewidth=1.3,  label="Precio")
        axs[0].plot(df_plot.index, df_plot["SMA_50"],  color="cyan",    linewidth=1.8, alpha=0.7,   label="SMA 50")
        if "SMA_200" in df_plot.columns and df_plot["SMA_200"].notna().any():
            axs[0].plot(df_plot.index, df_plot["SMA_200"], color="orange", linewidth=1.8, alpha=0.75, label="SMA 200")
        axs[0].plot(df_plot.index, df_plot["EMA_8"],   color="#FFFFFF", linewidth=1.4, alpha=0.85,  linestyle="--", label="EMA 8")
        axs[0].plot(df_plot.index, df_plot["EMA_21"],  color="#00BFFF", linewidth=1.4, alpha=0.85,  linestyle="--", label="EMA 21")

        for r, c in REGIME_COLORS.items():
            m = df_plot["Regime_MACDV"] == r
            if m.any():
                axs[0].scatter(df_plot[m].index, df_plot[m]["Close"], c=c, s=70, alpha=0.95,
                               edgecolors='black', linewidths=1.2, zorder=5, label=f"{r}")

        last_price = float(df_plot["Close"].iloc[-1])
        axs[0].text(1.01, last_price, f'${last_price:.2f}',
                    transform=axs[0].get_yaxis_transform(),
                    fontsize=8, color='white', va='center', fontweight='bold')

        axs[0].set_title(f"{ticker} — Régimen (MACD-V)", fontsize=10, fontweight='bold', pad=6)
        axs[0].grid(alpha=0.25, linestyle='--')
        axs[0].set_ylabel("Precio ($)", fontsize=8)
        axs[0].legend(loc='upper left', fontsize=6.5, ncol=3)
        axs[0].yaxis.tick_right()
        axs[0].yaxis.set_label_position("right")
        axs[0].tick_params(axis='both', labelsize=7)

        # ── Gráfico 2: MACD-V con zonas y umbral de riesgo ──────────────────
        macd_colors = ['#FF0000' if abs(m) >= 151 else '#FFFFFF' for m in df_plot["MACD_V"]]
        for i in range(len(df_plot) - 1):
            axs[1].plot(df_plot.index[i:i+2], df_plot["MACD_V"].iloc[i:i+2],
                        color=macd_colors[i], linewidth=1.8)

        axs[1].axhline(  0, color="gray",  linestyle="-",  alpha=0.4)
        axs[1].axhline( 50, color="green", linestyle=":",  alpha=0.4, linewidth=1, label="Zona ±50 (Rango)")
        axs[1].axhline(-50, color="red",   linestyle=":",  alpha=0.4, linewidth=1)
        axs[1].axhline( 151, color="red",  linestyle="--", alpha=0.6, linewidth=1.3, label="Umbral ±151 (Riesgo)")
        axs[1].axhline(-151, color="red",  linestyle="--", alpha=0.6, linewidth=1.3)
        axs[1].fill_between(df_plot.index, -50, 50, color="gold", alpha=0.08)

        last_macd = float(df_plot["MACD_V"].iloc[-1])
        macd_color_final = '#FF0000' if abs(last_macd) >= 151 else '#FFFFFF'
        axs[1].text(1.01, last_macd, f'{last_macd:.2f}',
                    transform=axs[1].get_yaxis_transform(),
                    fontsize=8, color=macd_color_final, va='center', fontweight='bold')

        axs[1].set_title("MACD-V (momentum normalizado por ATR)", fontsize=9, fontweight='bold', pad=4)
        axs[1].grid(alpha=0.25, linestyle='--')
        axs[1].set_ylabel("MACD-V", fontsize=8)
        axs[1].set_xlabel("Fecha", fontsize=8)
        axs[1].yaxis.tick_right()
        axs[1].yaxis.set_label_position("right")
        axs[1].tick_params(axis='both', labelsize=7)
        axs[1].legend(loc='upper left', fontsize=6.5)

        plt.tight_layout()
        st.pyplot(fig)

        # ================= DOCUMENTACIÓN =================
        st.markdown("---")
        with st.expander("📖 Ver Lógica de Clasificación"):
            st.markdown("""
            #### 🔹 Método MACD-V (único clasificador de régimen)

            **MACD-V** = [(EMA12 − EMA26) / ATR(26)] × 100 — es el MACD estándar
            normalizado por el rango medio real (ATR), lo que lo hace comparable
            entre activos y timeframes distintos sin necesitar recalibrar el
            umbral por ticker (a diferencia de un umbral de pendiente o retorno
            absoluto, que sí depende de la escala de volatilidad de cada acción).

            **Reglas de clasificación:**
            - |MACD-V| entre -50 y +50 → **RANGO**
            - Precio > SMA50 **y** MACD-V entre 51 y 150 → **ALCISTA**
            - Precio < SMA50 **y** MACD-V entre -150 y -51 → **BAJISTA**
            - |MACD-V| > 150 → **RIESGO** (momentum sobre-extendido; el impulso es
              tan fuerte en relación a la volatilidad reciente que estadísticamente
              es más probable un descanso/reversión que una continuación limpia)

            La **SMA 200** en el gráfico es solo de referencia visual (tendencia de
            largo plazo) — no participa en la lógica de clasificación, que usa
            exclusivamente SMA 50 + MACD-V. Con menos de 200 semanas de historial
            (ticker joven o "Histórico (años)" bajo) puede no mostrarse completa
            al inicio del gráfico.

            #### ⚠️ Análisis de Riesgo

            El semáforo de riesgo usa directamente el mismo umbral de sobre-extensión
            de MACD-V:

            - |MACD-V| ≥ 151 → **RIESGO+** (morado)
            - En cualquier otro caso → **Sin Riesgo** (verde)
            """)

    elif st.session_state.analyzed and st.session_state.df is None:
        st.error("❌ No se pudo descargar data. Verifica el ticker.")


if __name__ == "__main__":
    if check_password():
        main()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder.")
