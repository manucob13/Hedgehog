import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
# PARÁMETROS (idénticos al indicador de TradingView)
# ============================================================================

FAST_LEN  = 12
SLOW_LEN  = 26
SIG_LEN   = 9
ATR_LEN   = 26
NIVEL_NEU = 50
NIVEL_EXT = 150
EXTRA_FONDO = 50   # altura extra del fondo extremo (como en Pine)

# ============================================================================
# COLORES
# ============================================================================

# Colores de zona MACD-V (los mismos RGB que el script de Pine)
C_BLUE   = "rgb(0,70,255)"      # riesgo (> +150 / < -150)
C_GREEN  = "rgb(0,170,60)"      # alcista (+50 a +150)
C_YELLOW = "rgb(255,200,0)"     # neutral (-50 a +50)
C_RED    = "rgb(220,30,30)"     # bajista (-150 a -50)

# Fondos entre líneas (RGBA ~ transparencia 85 de Pine => alpha 0.15)
F_BLUE   = "rgba(0,70,255,0.15)"
F_GREEN  = "rgba(0,170,60,0.15)"
F_YELLOW = "rgba(255,200,0,0.15)"
F_RED    = "rgba(220,30,30,0.15)"

ZONE_COLORS = {
    "RIESGO ALTO": C_BLUE,
    "ALCISTA":     C_GREEN,
    "NEUTRAL":     C_YELLOW,
    "BAJISTA":     C_RED,
    "RIESGO BAJO": C_BLUE,
}

# Colores de régimen (clasificador con SMA 50 + MACD-V)
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

SERIES_COLORS = {
    "SMA 50":  "#00FFFF",
    "SMA 200": "#FFA500",
    "EMA 8":   "#AAAAAA",
    "EMA 21":  "#00BFFF",
}

CANDLE_UP   = "#26a69a"
CANDLE_DOWN = "#ef5350"

# ============================================================================
# INDICADORES
# ============================================================================

def rma(series, length):
    """Media móvil de Wilder (RMA) igual que ta.rma de Pine: semilla = SMA de las
    primeras `length` barras y luego alpha = 1/length."""
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) < length:
        return pd.Series(out, index=series.index)
    out[length - 1] = np.nanmean(values[:length])
    alpha = 1.0 / length
    for i in range(length, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return pd.Series(out, index=series.index)


def calculate_indicators(df, fast=FAST_LEN, slow=SLOW_LEN, signal=SIG_LEN, atr_len=ATR_LEN):
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

    # ta.atr() de TradingView = RMA del True Range (NO una media simple)
    atr = rma(tr, atr_len)

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
# ZONAS (igual que zoneCol / zoneTxt del Pine)
# ============================================================================

def classify_zone(macd_v):
    macd_v = np.asarray(macd_v, dtype=float)
    return np.select(
        [
            macd_v > NIVEL_EXT,
            macd_v > NIVEL_NEU,
            macd_v >= -NIVEL_NEU,
            macd_v >= -NIVEL_EXT,
        ],
        ["RIESGO ALTO", "ALCISTA", "NEUTRAL", "BAJISTA"],
        default="RIESGO BAJO",
    )

# ============================================================================
# CLASIFICACIÓN DE RÉGIMEN (SMA 50 + MACD-V)
# ============================================================================

def classify_by_macdv(df):
    price  = df["Close"].astype(float).to_numpy()
    sma50  = df["SMA_50"].astype(float).to_numpy()
    macd_v = df["MACD_V"].astype(float).to_numpy()

    above = price > sma50
    valid = ~(np.isnan(price) | np.isnan(sma50) | np.isnan(macd_v))

    conditions = [
        ~valid,
        np.abs(macd_v) <= NIVEL_NEU,
        above  & (macd_v > NIVEL_EXT),
        above  & (macd_v > NIVEL_NEU),
        ~above & (macd_v < -NIVEL_EXT),
        ~above & (macd_v < -NIVEL_NEU),
    ]
    choices = ["RANGO", "RANGO", "RIESGO", "ALCISTA", "RIESGO", "BAJISTA"]
    regimes = np.select(conditions, choices, default="RANGO")

    # Primeras 10 barras: sin suficiente contexto
    regimes[:10] = "RANGO"
    return list(regimes)

# ============================================================================
# ANÁLISIS DE RIESGO
# ============================================================================

def analyze_risk(df):
    macd_v = df["MACD_V"].astype(float).to_numpy()
    extreme = np.abs(macd_v) > NIVEL_EXT
    extreme = np.where(np.isnan(macd_v), False, extreme)
    return list(np.where(extreme, "RIESGO+", "Sin Riesgo"))

# ============================================================================
# DESCARGA DE DATOS (DIARIO)
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def download_daily_data(ticker, years_back):
    # +1 año extra de calentamiento para SMA200 / EMAs / ATR
    start = (datetime.now() - timedelta(days=365 * (years_back + 1))).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, interval="1d", progress=False, auto_adjust=False)

    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df = df[["Date", "Open", "High", "Low", "Close"]].copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

    df.columns = [str(c) for c in df.columns]

    df["SMA_50"]  = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200, min_periods=1).mean()
    df["EMA_8"]   = df["Close"].ewm(span=8,  adjust=False).mean()
    df["EMA_21"]  = df["Close"].ewm(span=21, adjust=False).mean()

    df = calculate_indicators(df)
    df.dropna(subset=["MACD_V", "SMA_50"], inplace=True)

    # Recortar al histórico pedido (el año extra solo servía de calentamiento)
    cutoff = pd.Timestamp(datetime.now() - timedelta(days=365 * years_back))
    df = df[df.index >= cutoff]

    return df

# ============================================================================
# UI STREAMLIT
# ============================================================================

def main():
    st.set_page_config(layout="wide", page_title="Daily Regime Analyzer")
    st.title("📊 Daily Regime Analyzer — MACD-V")
    st.caption("Alcista · Bajista · Rango · Riesgo — clasificación por momentum normalizado por volatilidad (MACD-V), timeframe diario")

    if 'analyzed' not in st.session_state:
        st.session_state.analyzed = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'ticker' not in st.session_state:
        st.session_state.ticker = "AAPL"

    col_input1, col_input2, col_input3, col_input4 = st.columns([2, 1, 1, 1])
    with col_input1:
        ticker = st.text_input("Ticker", st.session_state.ticker).upper()
    with col_input2:
        years_back = st.slider("Histórico (años)", 1, 10, 3)
    with col_input3:
        lookback_months = st.slider("Meses a visualizar", 1, 36, 8)
    with col_input4:
        modo = st.selectbox("Modo MACD-V", ["Barra", "Area", "Lineas"], index=0)

    if st.button("🚀 ANALIZAR", use_container_width=True):
        st.session_state.ticker = ticker
        st.session_state.df = download_daily_data(ticker, years_back)
        st.session_state.analyzed = True

    if st.session_state.analyzed and st.session_state.df is not None:

        df = st.session_state.df.copy()

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[df.index.notna()].sort_index()

        df["Regime_MACDV"] = classify_by_macdv(df)
        df["Risk_Level"]   = analyze_risk(df)
        df["Zone"]         = classify_zone(df["MACD_V"])
        st.session_state.df = df

        # ~21 sesiones por mes en diario
        df_plot = df.tail(max(int(lookback_months * 21), 30)).copy()
        current = df.iloc[-1]

        regime_macdv = df["Regime_MACDV"].iloc[-1]
        risk_level   = df["Risk_Level"].iloc[-1]
        zone_now     = df["Zone"].iloc[-1]

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

        info_cols = st.columns(6)
        info_cols[0].metric("Precio", f"${precio:.2f}")
        info_cols[1].metric("SMA 50", f"${sma50:.2f}")
        info_cols[2].metric("SMA 200", f"${sma200:.2f}" if pd.notna(sma200) else "N/A")
        info_cols[3].metric("Precio > SMA50", precio_above_sma)
        info_cols[4].metric("MACD-V", f"{macd_v:.2f}")
        info_cols[5].metric("Zona MACD-V", zone_now)

        # ================= ANÁLISIS DE RIESGO =================
        macd_extreme = abs(macd_v) > NIVEL_EXT
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
                    <td style="padding: 4px 6px; border: 1px solid #444; font-size: 10px;">|MACD-V| &gt; 150 → RIESGO+</td>
                </tr>
                <tr>
                    <td style="padding: 4px 6px; border: 1px solid #444; font-size: 10px;">|MACD-V| ≤ 50 → RANGO / NEUTRAL</td>
                </tr>
            </table>
            """, unsafe_allow_html=True)

        # ================= LEYENDA DE COLORES =================
        st.markdown("### 🎨 Leyenda de Régimen")
        cols = st.columns(4)
        for idx, (regime, color) in enumerate(REGIME_COLORS.items()):
            with cols[idx]:
                st.markdown(
                    f'<div style="background-color:{color};padding:5px;border-radius:5px;'
                    f'text-align:center;font-weight:bold;color:black;border:2px solid white;font-size:11px;">{regime}</div>',
                    unsafe_allow_html=True
                )

        # ================= TÍTULO DEL GRÁFICO 1 =================
        st.markdown(
            f'<div style="margin-top:14px;margin-bottom:2px;font-size:15px;'
            f'font-weight:bold;color:#FFFFFF;">{ticker} — Diario (MACD-V)</div>',
            unsafe_allow_html=True,
        )

        # ================= LEYENDA HTML =================
        legend_items_html = ""
        for name, color in SERIES_COLORS.items():
            dash = "border-top:2px dashed" if name in ("EMA 8", "EMA 21") else "border-top:3px solid"
            legend_items_html += f"""
            <div style="display:flex;align-items:center;gap:6px;margin-right:18px;">
                <div style="width:18px;{dash} {color};"></div>
                <span style="color:#DDD;font-size:12px;">{name}</span>
            </div>
            """
        st.markdown(
            f"""
            <div style="display:flex;flex-wrap:nowrap;align-items:center;
                        overflow-x:auto;padding:4px 4px 8px 4px;">
                {legend_items_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ================= GRÁFICOS =================
        macdv_abs_max = float(df_plot["MACD_V"].abs().max())
        y_limit_macdv = max(NIVEL_EXT + EXTRA_FONDO, macdv_abs_max * 1.08)

        x_min = df_plot.index.min()
        x_max = df_plot.index.max()
        x_pad = (x_max - x_min) * 0.06
        x_max_padded = x_max + x_pad

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.58, 0.42],
            vertical_spacing=0.1,
        )

        # ── Panel 1: Velas + medias + puntos de régimen ─────────────────────
        fig.add_trace(go.Candlestick(
            x=df_plot.index,
            open=df_plot["Open"], high=df_plot["High"],
            low=df_plot["Low"], close=df_plot["Close"],
            name="Precio",
            increasing=dict(line=dict(color=CANDLE_UP, width=1), fillcolor=CANDLE_UP),
            decreasing=dict(line=dict(color=CANDLE_DOWN, width=1), fillcolor=CANDLE_DOWN),
            showlegend=False,
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["SMA_50"], mode="lines", name="SMA 50",
            line=dict(color=SERIES_COLORS["SMA 50"], width=1.3), opacity=0.85,
            showlegend=False,
        ), row=1, col=1)
        if df_plot["SMA_200"].notna().any():
            fig.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot["SMA_200"], mode="lines", name="SMA 200",
                line=dict(color=SERIES_COLORS["SMA 200"], width=1.3), opacity=0.85,
                showlegend=False,
            ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["EMA_8"], mode="lines", name="EMA 8",
            line=dict(color=SERIES_COLORS["EMA 8"], width=1.1, dash="dash"), opacity=0.7,
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["EMA_21"], mode="lines", name="EMA 21",
            line=dict(color=SERIES_COLORS["EMA 21"], width=1.1, dash="dash"), opacity=0.7,
            showlegend=False,
        ), row=1, col=1)

        last_price = float(df_plot["Close"].iloc[-1])
        fig.add_annotation(
            xref="x", yref="y1", x=x_max_padded, y=last_price,
            text=f" ${last_price:,.2f} ", showarrow=False,
            font=dict(color="black", size=13, family="Arial Black"),
            align="left", xanchor="left",
            bgcolor="#FFFFFF", bordercolor="white", borderwidth=1,
            row=1, col=1,
        )

        # ── Panel 2: MACD-V estilo TradingView ──────────────────────────────
        # Fondos por zona (cada una con su color), de x_min a x_max_padded
        zone_rects = [
            (NIVEL_EXT,   y_limit_macdv, F_BLUE),
            (NIVEL_NEU,   NIVEL_EXT,     F_GREEN),
            (-NIVEL_NEU,  NIVEL_NEU,     F_YELLOW),
            (-NIVEL_EXT,  -NIVEL_NEU,    F_RED),
            (-y_limit_macdv, -NIVEL_EXT, F_BLUE),
        ]
        for y0, y1, fill in zone_rects:
            fig.add_shape(
                type="rect", x0=x_min, x1=x_max_padded, y0=y0, y1=y1,
                fillcolor=fill, line_width=0, layer="below",
                row=2, col=1,
            )

        # Líneas de referencia: ±150 discontinua, ±50 punteada, 0 sólida tenue
        for y_val, color, dash, width in [
            (0,           "gray",    "solid", 1),
            (NIVEL_NEU,   "#A5A5A5", "dot",   1.3),
            (-NIVEL_NEU,  "#A5A5A5", "dot",   1.3),
            (NIVEL_EXT,   "#8C8C8C", "dash",  1.5),
            (-NIVEL_EXT,  "#8C8C8C", "dash",  1.5),
        ]:
            fig.add_shape(
                type="line", x0=x_min, x1=x_max_padded, y0=y_val, y1=y_val,
                line=dict(color=color, width=width, dash=dash),
                row=2, col=1,
            )

        zone_colors_series = [ZONE_COLORS[z] for z in df_plot["Zone"]]

        if modo == "Barra":
            fig.add_trace(go.Bar(
                x=df_plot.index, y=df_plot["MACD_V"],
                marker=dict(color=zone_colors_series, line=dict(width=0)),
                name="MACD-V", showlegend=False,
            ), row=2, col=1)

        elif modo == "Area":
            # Un trazo por color de zona, relleno hasta 0
            fill_map = {
                C_BLUE:   "rgba(0,70,255,0.65)",
                C_GREEN:  "rgba(0,170,60,0.65)",
                C_YELLOW: "rgba(255,200,0,0.65)",
                C_RED:    "rgba(220,30,30,0.65)",
            }
            zone_color_arr = np.array(zone_colors_series)
            for col_name, fill_rgba in fill_map.items():
                mask = zone_color_arr == col_name
                if not mask.any():
                    continue
                y_masked = df_plot["MACD_V"].where(mask)
                fig.add_trace(go.Scatter(
                    x=df_plot.index, y=y_masked, mode="lines",
                    line=dict(color=col_name, width=1),
                    fill="tozeroy", fillcolor=fill_rgba,
                    connectgaps=False, showlegend=False, name="MACD-V",
                ), row=2, col=1)

        else:  # Lineas
            fig.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot["MACD_V"], mode="lines", name="MACD-V",
                line=dict(color="rgb(20,70,200)", width=2), showlegend=False,
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot["MACD_V_Signal"], mode="lines", name="Signal",
                line=dict(color="rgb(230,120,0)", width=1), showlegend=False,
            ), row=2, col=1)

        # Etiqueta de zona (esquina superior derecha, como la tabla de Pine)
        label_txt_color = "black" if zone_now == "NEUTRAL" else "white"
        fig.add_annotation(
            xref="x2 domain", yref="y2 domain", x=1.0, y=1.08,
            xanchor="right", yanchor="bottom",
            text=f"<b>{zone_now}: {macd_v:.1f}</b>",
            showarrow=False,
            font=dict(size=13, color=label_txt_color),
            bgcolor=ZONE_COLORS[zone_now], bordercolor="white", borderwidth=1,
            borderpad=4,
        )

        # Ejes
        fig.update_xaxes(range=[x_min, x_max_padded], domain=[0.0, 0.94], row=1, col=1)
        fig.update_xaxes(range=[x_min, x_max_padded], domain=[0.0, 0.94], row=2, col=1)

        # Ocultar fines de semana (diario) para evitar huecos
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

        fig.update_yaxes(range=[-y_limit_macdv, y_limit_macdv], title_text="MACD-V",
                         row=2, col=1, side="right",
                         tickvals=[-200, -150, -100, -50, 0, 50, 100, 150, 200])
        fig.update_yaxes(title_text="Precio ($)", row=1, col=1, side="right")
        fig.update_xaxes(title_text="Fecha", row=2, col=1)

        fig.add_annotation(
            xref="x2 domain", yref="y2 domain", x=0, y=1.08,
            xanchor="left", yanchor="bottom",
            text="<b>MACD-V (momentum normalizado por ATR)</b>",
            showarrow=False, font=dict(size=12, color="white"),
        )

        fig.update_layout(
            template="plotly_dark",
            height=860,
            margin=dict(l=10, r=70, t=20, b=10),
            showlegend=False,
            hovermode=False,
            dragmode=False,
            bargap=0.05,
        )
        fig.update_xaxes(fixedrange=True, rangeslider_visible=False)
        fig.update_yaxes(fixedrange=True)

        st.plotly_chart(fig, use_container_width=True, config={
            "staticPlot": True,
            "displayModeBar": False,
            "displaylogo": False,
        })

        # ================= DOCUMENTACIÓN =================
        st.markdown("---")
        with st.expander("📖 Ver Lógica de Clasificación"):
            st.markdown("""
            #### 🔹 MACD-V (Spiroglou) — timeframe diario

            **MACD-V** = [(EMA12 − EMA26) / ATR(26)] × 100, con el **ATR calculado
            con la media de Wilder (RMA)**, igual que `ta.atr` de TradingView, de modo
            que los valores coinciden con el indicador de Pine.

            **Zonas del indicador (colores de las barras):**
            - MACD-V > +150 → **RIESGO ALTO** (azul)
            - +50 a +150 → **ALCISTA** (verde)
            - -50 a +50 → **NEUTRAL** (amarillo)
            - -150 a -50 → **BAJISTA** (rojo)
            - MACD-V < -150 → **RIESGO BAJO** (azul)

            **Régimen (tarjeta superior y puntos del gráfico)**, combinando SMA 50:
            - |MACD-V| ≤ 50 → **RANGO**
            - Precio > SMA50 y MACD-V entre 50 y 150 → **ALCISTA**
            - Precio < SMA50 y MACD-V entre -150 y -50 → **BAJISTA**
            - |MACD-V| > 150 (en el sentido de la tendencia) → **RIESGO**

            La **SMA 200** y las EMAs son solo de referencia visual.

            #### ⚠️ Análisis de Riesgo
            - |MACD-V| > 150 → **RIESGO+** (morado)
            - En cualquier otro caso → **Sin Riesgo** (verde)

            **Modos del panel MACD-V:** Barra (por defecto), Area y Lineas (MACD-V + Signal).
            """)

    elif st.session_state.analyzed and st.session_state.df is None:
        st.error("❌ No se pudo descargar data. Verifica el ticker.")


if __name__ == "__main__":
    if check_password():
        main()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder.")
