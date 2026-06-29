"""
00 WEIC Calculos.py
====================
Weekly Iron Condor — Calculadora de bandas y strikes.

- Descarga SPX y VIX desde Yahoo Finance (sin Schwab)
- Calcula volatilidad Garman-Klass por régimen de VIX
- Escala la vol a 3, 4 o 5 DTE (siempre cierre el viernes)
- Permite modificar: ticker, DTE, desviaciones (stdn), ancho de spread, contratos
- Muestra las tres tranches con sus bandas y strikes sugeridos
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import yfinance as yf
import math
import warnings
warnings.filterwarnings("ignore")

from utils.utils import check_password
from utils.volatility import calcular_bandas_ic

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA
# ==============================================================================
# st.set_page_config(page_title="🦅 WEIC Calculos", layout="wide")


# ==============================================================================
# HELPERS
# ==============================================================================

def get_next_friday() -> date:
    """Devuelve el próximo viernes desde hoy."""
    today = date.today()
    days_ahead = (4 - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return today + timedelta(days=days_ahead)


def get_entry_date_for_dte(dte: int) -> date:
    """
    Dado un DTE (3, 4 o 5), calcula el día de entrada para que el vencimiento
    sea el próximo viernes.
    DTE 5 → lunes    (viernes - 5 días hábiles ≈ lunes)
    DTE 4 → martes
    DTE 3 → miércoles
    """
    friday = get_next_friday()
    # Restamos días naturales: lun=0 vie=4, así que:
    # DTE 5 → entrada lunes  → friday - 4 días
    # DTE 4 → entrada martes → friday - 3 días
    # DTE 3 → entrada miérc  → friday - 2 días
    offset = {5: 4, 4: 3, 3: 2}
    entry = friday - timedelta(days=offset[dte])
    return entry


def round_to_multiple(value: float, multiple: float = 5.0) -> int:
    """Redondea al múltiplo más cercano (hacia abajo para DW, arriba para UP)."""
    return int(round(value / multiple) * multiple)


def floor_to_multiple(value: float, multiple: float = 5.0) -> int:
    return int(math.floor(value / multiple) * multiple)


def ceil_to_multiple(value: float, multiple: float = 5.0) -> int:
    return int(math.ceil(value / multiple) * multiple)


@st.cache_data(ttl=3600)
def descargar_datos_yf(ticker_spx: str = "^GSPC", ticker_vix: str = "^VIX") -> dict:
    """
    Descarga precio actual del SPX y VIX desde Yahoo Finance.
    Usa los últimos 5 días para asegurarse de tener el último cierre.
    """
    end = datetime.now() + timedelta(days=1)
    start = end - timedelta(days=10)

    df_spx = yf.download(ticker_spx, start=start, end=end,
                          auto_adjust=False, multi_level_index=False, progress=False)
    df_vix = yf.download(ticker_vix, start=start, end=end,
                          auto_adjust=False, multi_level_index=False, progress=False)

    if df_spx.empty or df_vix.empty:
        return {"error": "No se pudieron descargar datos de Yahoo Finance."}

    last_spx   = float(df_spx["Close"].iloc[-1])
    last_open  = float(df_spx["Open"].iloc[-1])
    last_vix   = float(df_vix["Close"].iloc[-1])
    last_date  = df_spx.index[-1].strftime("%Y-%m-%d")

    return {
        "spx_close": last_spx,
        "spx_open":  last_open,
        "vix":       last_vix,
        "last_date": last_date,
        "error":     None,
    }


def render_tranche_card(tranche, strike_width: int, contratos: int):
    """
    Renderiza una card de tranche con bandas, strikes sugeridos y resumen de la orden.
    """
    entry_date = get_entry_date_for_dte(tranche.dte)
    friday     = get_next_friday()

    # Strikes sugeridos
    short_put  = floor_to_multiple(tranche.band_dw, 5)
    long_put   = short_put - strike_width
    short_call = ceil_to_multiple(tranche.band_up, 5)
    long_call  = short_call + strike_width

    dte_label  = {5: "Lunes", 4: "Martes", 3: "Miércoles"}[tranche.dte]
    color_map  = {5: "#2a78d6", 4: "#1baf7a", 3: "#eda100"}
    color      = color_map[tranche.dte]

    st.markdown(
        f"""
        <div style="border-top: 3px solid {color}; background: var(--secondary-background-color);
                    border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 0.5rem;">
            <div style="font-size:11px; color:gray; margin-bottom:2px;">
                {tranche.dte} DTE &nbsp;|&nbsp; Entrada: <b>{dte_label} {entry_date.strftime('%d/%m')}</b>
                &nbsp;→&nbsp; Cierre: <b>Viernes {friday.strftime('%d/%m')}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Banda inferior", f"{tranche.band_dw:,.0f}")
        st.metric("Banda superior", f"{tranche.band_up:,.0f}")
        st.metric("Movimiento ±", f"{tranche.move_pct:.2f}%")
    with c2:
        st.metric("Vol diaria (GK)", f"{tranche.vol_daily:.5f}")
        st.metric("Vol escalada ×√DTE", f"{tranche.vol_scaled:.5f}")

    st.markdown("**Strikes sugeridos:**")
    cols = st.columns(4)
    with cols[0]:
        st.markdown(f"<div style='text-align:center'><small>Long PUT</small><br><b>{long_put}</b></div>",
                    unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"<div style='text-align:center'><small>Short PUT</small><br><b>{short_put}</b></div>",
                    unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f"<div style='text-align:center'><small>Short CALL</small><br><b>{short_call}</b></div>",
                    unsafe_allow_html=True)
    with cols[3]:
        st.markdown(f"<div style='text-align:center'><small>Long CALL</small><br><b>{long_call}</b></div>",
                    unsafe_allow_html=True)

    st.markdown("---")

    # Tabla resumen de la orden para este tranche
    expiry_str = friday.strftime("%Y%m%d")
    orders = pd.DataFrame([
        {"Acción": "BUY",  "Strike": long_put,   "Tipo": "PUT",  "Exp": expiry_str, "Qty": contratos, "Label": "Long PUT"},
        {"Acción": "SELL", "Strike": short_put,  "Tipo": "PUT",  "Exp": expiry_str, "Qty": contratos, "Label": "Short PUT"},
        {"Acción": "SELL", "Strike": short_call, "Tipo": "CALL", "Exp": expiry_str, "Qty": contratos, "Label": "Short CALL"},
        {"Acción": "BUY",  "Strike": long_call,  "Tipo": "CALL", "Exp": expiry_str, "Qty": contratos, "Label": "Long CALL"},
    ])
    st.dataframe(orders, hide_index=True, use_container_width=True)

    return {
        "dte":        tranche.dte,
        "entry":      entry_date,
        "expiry":     friday,
        "long_put":   long_put,
        "short_put":  short_put,
        "short_call": short_call,
        "long_call":  long_call,
        "band_dw":    tranche.band_dw,
        "band_up":    tranche.band_up,
        "move_pct":   tranche.move_pct,
    }


# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================

def main_weic_calculos():

    st.markdown(
        "<h1><span style='font-size:1.5em;'>🦅</span> WEIC — Weekly Iron Condor</h1>",
        unsafe_allow_html=True
    )
    st.markdown("Calculadora de bandas y strikes para Iron Condor semanal en SPX.")
    st.markdown("---")

    # --------------------------------------------------------------------------
    # SIDEBAR — parámetros
    # --------------------------------------------------------------------------
    st.sidebar.header("⚙️ Parámetros")

    ticker_input = st.sidebar.text_input(
        "Ticker (Yahoo Finance)", value="^GSPC",
        help="Default: ^GSPC (SPX). También podés usar SPY, QQQ, etc."
    )
    ticker_vix = "^VIX"

    stdn = st.sidebar.number_input(
        "Desviaciones estándar (stdn)", min_value=1.0, max_value=4.0,
        value=2.3, step=0.1,
        help="Número de desviaciones para calcular las bandas. Default: 2.3"
    )

    strike_width = st.sidebar.selectbox(
        "Ancho del spread (puntos)", options=[10, 15, 20, 25, 30, 35, 50],
        index=3,
        help="Distancia entre el strike corto y el largo de cada pata."
    )

    contratos = st.sidebar.number_input(
        "Contratos por tranche", min_value=1, max_value=20, value=1, step=1
    )

    dte_options = st.sidebar.multiselect(
        "Tranches a calcular (DTE)",
        options=[5, 4, 3],
        default=[5, 4, 3],
        format_func=lambda x: {5: "5 DTE — Lunes→Viernes",
                                4: "4 DTE — Martes→Viernes",
                                3: "3 DTE — Miércoles→Viernes"}[x],
        help="Seleccioná los días de entrada que querés calcular."
    )
    dte_options = sorted(dte_options, reverse=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**📅 Próximo viernes:**")
    st.sidebar.info(get_next_friday().strftime("%A %d/%m/%Y"))

    # --------------------------------------------------------------------------
    # SECCIÓN 1 — Datos de mercado
    # --------------------------------------------------------------------------
    st.header("1. Datos de Mercado")

    if st.button("🔄 Actualizar datos desde Yahoo Finance", type="primary"):
        st.cache_data.clear()

    with st.spinner("Descargando SPX y VIX..."):
        datos = descargar_datos_yf(ticker_input, ticker_vix)

    if datos["error"]:
        st.error(f"❌ {datos['error']}")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("SPX Último Cierre", f"${datos['spx_close']:,.2f}")
    with col2:
        st.metric("SPX Apertura", f"${datos['spx_open']:,.2f}")
    with col3:
        st.metric("VIX", f"{datos['vix']:.2f}")
    with col4:
        st.metric("Última fecha", datos["last_date"])

    # Precio de referencia — usamos el último cierre como spot
    # (el Open del día no está disponible hasta que el mercado abre)
    spot = datos["spx_close"]
    current_vix = datos["vix"]

    st.info(
        f"📌 **Spot de referencia:** ${spot:,.2f}  |  "
        f"**VIX:** {current_vix:.2f}  |  "
        f"**Cierre:** {datos['last_date']}"
    )

    st.markdown("---")

    # --------------------------------------------------------------------------
    # SECCIÓN 2 — Volatilidad y bandas
    # --------------------------------------------------------------------------
    st.header("2. Volatilidad por Régimen de VIX")

    with st.spinner("Calculando volatilidad Garman-Klass por régimen..."):
        try:
            resultado = calcular_bandas_ic(
                current_vix = current_vix,
                current_spx = spot,
                stdn        = stdn,
                ticker      = ticker_input,
                vix_ticker  = ticker_vix,
            )
        except Exception as e:
            st.error(f"❌ Error calculando volatilidad: {e}")
            st.stop()

    # Régimen activo
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Régimen activo", resultado.regime_label)
    with col2:
        st.metric("Días históricos en régimen", f"{resultado.regime_rows:,}")
    with col3:
        st.metric("Vol diaria (Avg_252_GK21)", f"{resultado.vol_daily:.6f}")

    # Señal de operación
    signal_color = "🟢" if resultado.signal == 1 else "🔴"
    signal_text  = "SÍ OPERAR" if resultado.signal == 1 else "NO OPERAR"
    st.markdown(f"**Señal del sistema:** {signal_color} **{signal_text}**")

    with st.expander("Ver detalle de la señal"):
        signal_data = {
            "Tendencia"         : resultado.trend,
            "VIX < 25"          : "✅" if resultado.vix_lt_25 else "❌",
            "VIX < VIX WMA21"   : "✅" if resultado.vix_lt_wma21 else "❌",
            "VIX WMA21 bajando" : "✅" if resultado.wma21_falling else "❌",
            "Último dato"       : resultado.last_date,
            "Calculado"         : resultado.timestamp,
        }
        st.dataframe(
            pd.DataFrame(signal_data.items(), columns=["Campo", "Valor"]),
            hide_index=True, use_container_width=True
        )

    st.markdown("---")

    # --------------------------------------------------------------------------
    # SECCIÓN 3 — Tranches
    # --------------------------------------------------------------------------
    st.header("3. Tranches Semanales")

    if not dte_options:
        st.warning("⚠️ Seleccioná al menos un DTE en el panel lateral.")
        st.stop()

    tranche_map = {5: resultado.T1, 4: resultado.T2, 3: resultado.T3}
    resumen_rows = []

    tabs = st.tabs([
        {5: "📅 T1 — Lunes (5 DTE)",
         4: "📅 T2 — Martes (4 DTE)",
         3: "📅 T3 — Miércoles (3 DTE)"}[d]
        for d in dte_options
    ])

    for tab, dte in zip(tabs, dte_options):
        with tab:
            info = render_tranche_card(tranche_map[dte], strike_width, contratos)
            resumen_rows.append(info)

    st.markdown("---")

    # --------------------------------------------------------------------------
    # SECCIÓN 4 — Resumen consolidado
    # --------------------------------------------------------------------------
    st.header("4. Resumen Consolidado")

    friday = get_next_friday()
    df_resumen = pd.DataFrame([{
        "Tranche"      : {5: "T1 Lunes→Vier.", 4: "T2 Martes→Vier.", 3: "T3 Miérc.→Vier."}[r["dte"]],
        "DTE"          : r["dte"],
        "Entrada"      : r["entry"].strftime("%d/%m/%Y"),
        "Vencimiento"  : r["expiry"].strftime("%d/%m/%Y"),
        "Long PUT"     : r["long_put"],
        "Short PUT"    : r["short_put"],
        "Short CALL"   : r["short_call"],
        "Long CALL"    : r["long_call"],
        "Banda DW"     : f"{r['band_dw']:,.0f}",
        "Banda UP"     : f"{r['band_up']:,.0f}",
        "Movimiento ±%": f"{r['move_pct']:.2f}%",
        "Contratos"    : contratos,
        "Ancho"        : strike_width,
    } for r in resumen_rows])

    st.dataframe(df_resumen, hide_index=True, use_container_width=True)

    # Métricas globales
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Spot referencia", f"${spot:,.2f}")
    with col2:
        st.metric("VIX", f"{current_vix:.2f}")
    with col3:
        st.metric("Régimen", resultado.regime_label)
    with col4:
        st.metric("Próximo viernes", friday.strftime("%d/%m/%Y"))

    st.markdown(f"""
    > **Parámetros usados:** stdn = {stdn} | Ancho = {strike_width} pts | Contratos = {contratos}  
    > *Las bandas y strikes son estimaciones basadas en volatilidad histórica Garman-Klass.
    > Verificá siempre los precios reales antes de operar.*
    """)


# ==============================================================================
# PUNTO DE ENTRADA PROTEGIDO
# ==============================================================================

if check_password():
    main_weic_calculos()
else:
    st.title("🔒 Acceso Restringido")
    st.info("Por favor, introducí tus credenciales en el menú lateral para acceder.")
