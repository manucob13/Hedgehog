"""
00 WEIC Calculos.py
====================
Weekly Iron Condor — Calculadora de movimiento esperado SPX.

- Descarga SPX y VIX desde Yahoo Finance
- Calcula volatilidad Garman-Klass por régimen de VIX (via volatility.py)
- El usuario elige: día de entrada, día de salida y std
- Muestra solo las bandas de precio esperado (sin strikes ni spreads)
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

from utils.utils import check_password
from utils.volatility import calcular_bandas_ic


# ==============================================================================
# HELPERS
# ==============================================================================

def get_next_thursday() -> date:
    today = date.today()
    days_ahead = (3 - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return today + timedelta(days=days_ahead)


def get_next_friday() -> date:
    today = date.today()
    days_ahead = (4 - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return today + timedelta(days=days_ahead)


def calcular_dte(entrada: str, salida: str) -> int:
    """
    Calcula los DTE según día de entrada y día de salida.
    Entrada: lunes=0, martes=1, miércoles=2
    Salida:  jueves=3, viernes=4
    """
    dias = {"Lunes": 0, "Martes": 1, "Miércoles": 2, "Jueves": 3, "Viernes": 4}
    return dias[salida] - dias[entrada]


@st.cache_data(ttl=3600)
def descargar_datos_yf(ticker_spx: str = "^GSPC", ticker_vix: str = "^VIX") -> dict:
    end   = datetime.now() + timedelta(days=1)
    start = end - timedelta(days=10)

    df_spx = yf.download(ticker_spx, start=start, end=end,
                         auto_adjust=False, multi_level_index=False, progress=False)
    df_vix = yf.download(ticker_vix, start=start, end=end,
                         auto_adjust=False, multi_level_index=False, progress=False)

    if df_spx.empty or df_vix.empty:
        return {"error": "No se pudieron descargar datos de Yahoo Finance."}

    return {
        "spx_close": float(df_spx["Close"].iloc[-1]),
        "spx_open":  float(df_spx["Open"].iloc[-1]),
        "vix":       float(df_vix["Close"].iloc[-1]),
        "last_date": df_spx.index[-1].strftime("%Y-%m-%d"),
        "error":     None,
    }


# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================

def main_weic_calculos():

    st.markdown(
        "<h1><span style='font-size:1.5em;'>🦅</span> WEIC — Movimiento Esperado SPX</h1>",
        unsafe_allow_html=True,
    )
    st.markdown("Calcula las bandas de precio esperado en función de la volatilidad GK y el régimen de VIX.")
    st.markdown("---")

    # --------------------------------------------------------------------------
    # SECCIÓN 1 — Datos de mercado
    # --------------------------------------------------------------------------
    st.header("1. Datos de mercado")

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button("🔄 Actualizar datos", type="primary"):
            st.cache_data.clear()
            st.rerun()

    with st.spinner("Descargando SPX y VIX..."):
        datos = descargar_datos_yf("^GSPC", "^VIX")

    if datos["error"]:
        st.error(f"❌ {datos['error']}")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("SPX cierre", f"${datos['spx_close']:,.2f}")
    with col2:
        st.metric("SPX apertura", f"${datos['spx_open']:,.2f}")
    with col3:
        st.metric("VIX", f"{datos['vix']:.2f}")
    with col4:
        st.metric("Fecha", datos["last_date"])

    spot        = datos["spx_close"]
    current_vix = datos["vix"]

    st.markdown("---")

    # --------------------------------------------------------------------------
    # SECCIÓN 2 — Parámetros de la operación (en página)
    # --------------------------------------------------------------------------
    st.header("2. Parámetros")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        dia_entrada = st.selectbox(
            "Día de entrada",
            options=["Lunes", "Martes", "Miércoles"],
            index=1,
            help="Día en que se abre la posición.",
        )

    with col_b:
        # Día de salida solo puede ser posterior al de entrada
        salidas_posibles = ["Jueves", "Viernes"]
        # Si entra el miércoles, solo puede salir jueves o viernes (ambas ok)
        # Si entra el lunes o martes, igualmente ambas opciones son válidas
        dia_salida = st.selectbox(
            "Día de salida",
            options=salidas_posibles,
            index=1,
            help="Día en que cierra la posición.",
        )

    with col_c:
        stdn = st.number_input(
            "Desviaciones estándar (σ)",
            min_value=0.5,
            max_value=4.0,
            value=2.3,
            step=0.1,
            format="%.1f",
            help="Número de desviaciones para calcular las bandas. La vol base viene del régimen GK.",
        )

    dte = calcular_dte(dia_entrada, dia_salida)

    # Validación: entrada debe ser antes de salida
    if dte <= 0:
        st.error(f"❌ El día de salida ({dia_salida}) debe ser posterior al día de entrada ({dia_entrada}).")
        st.stop()

    # Fechas concretas
    next_thursday = get_next_thursday()
    next_friday   = get_next_friday()
    fecha_salida  = next_thursday if dia_salida == "Jueves" else next_friday

    dias_map = {"Lunes": 0, "Martes": 1, "Miércoles": 2}
    hoy = date.today()
    # Calcular fecha de entrada dentro de esta semana (próxima ocurrencia)
    dias_hasta_entrada = (dias_map[dia_entrada] - hoy.weekday()) % 7
    if dias_hasta_entrada == 0:
        dias_hasta_entrada = 7
    fecha_entrada = hoy + timedelta(days=dias_hasta_entrada)

    st.info(
        f"📅 **Entrada:** {dia_entrada} {fecha_entrada.strftime('%d/%m/%Y')}  "
        f"→  **Salida:** {dia_salida} {fecha_salida.strftime('%d/%m/%Y')}  "
        f"|  **DTE:** {dte} días"
    )

    st.markdown("---")

    # --------------------------------------------------------------------------
    # SECCIÓN 3 — Cálculo de bandas
    # --------------------------------------------------------------------------
    st.header("3. Movimiento esperado")

    with st.spinner("Calculando volatilidad GK por régimen de VIX..."):
        try:
            resultado = calcular_bandas_ic(
                current_vix=current_vix,
                current_spx=spot,
                stdn=stdn,
                ticker="^GSPC",
                vix_ticker="^VIX",
            )
        except Exception as e:
            st.error(f"❌ Error calculando volatilidad: {e}")
            st.stop()

    # Tomamos la vol diaria GK del régimen activo
    vol_diaria  = resultado.vol_daily
    vol_escalada = vol_diaria * (dte ** 0.5)
    movimiento  = spot * vol_escalada * stdn
    move_pct    = vol_escalada * stdn * 100
    banda_inf   = spot - movimiento
    banda_sup   = spot + movimiento

    # Régimen informativo
    st.caption(
        f"Régimen VIX activo: **{resultado.regime_label}**  |  "
        f"Vol diaria GK: **{vol_diaria:.5f}**  |  "
        f"Vol escalada ×√{dte}: **{vol_escalada:.5f}**  |  "
        f"σ aplicado: **{stdn}**"
    )

    # Métricas principales
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Movimiento esperado ±",
            value=f"± {move_pct:.2f}%",
            delta=f"± {movimiento:,.0f} pts",
            delta_color="off",
        )
    with col2:
        st.metric(
            label="Banda inferior",
            value=f"{banda_inf:,.2f}",
            delta=f"−{movimiento:,.0f} pts vs spot",
            delta_color="inverse",
        )
    with col3:
        st.metric(
            label="Banda superior",
            value=f"{banda_sup:,.2f}",
            delta=f"+{movimiento:,.0f} pts vs spot",
            delta_color="normal",
        )

    st.markdown("---")

    # Tabla resumen
    st.subheader("Resumen")
    df = pd.DataFrame([{
        "Entrada"           : f"{dia_entrada} {fecha_entrada.strftime('%d/%m/%Y')}",
        "Salida"            : f"{dia_salida} {fecha_salida.strftime('%d/%m/%Y')}",
        "DTE"               : dte,
        "Spot"              : f"${spot:,.2f}",
        "VIX"               : f"{current_vix:.2f}",
        "Régimen"           : resultado.regime_label,
        "Vol diaria GK"     : f"{vol_diaria:.5f}",
        "Vol escalada ×√DTE": f"{vol_escalada:.5f}",
        "σ"                 : stdn,
        "Movimiento ±%"     : f"{move_pct:.2f}%",
        "Movimiento ±pts"   : f"{movimiento:,.0f}",
        "Banda inferior"    : f"{banda_inf:,.2f}",
        "Banda superior"    : f"{banda_sup:,.2f}",
    }])

    st.dataframe(df.T.rename(columns={0: "Valor"}), use_container_width=True)

    st.caption(
        "Las bandas son estimaciones basadas en volatilidad histórica Garman-Klass. "
        "Verificá siempre los precios reales antes de operar."
    )


# ==============================================================================
# PUNTO DE ENTRADA PROTEGIDO
# ==============================================================================

if check_password():
    main_weic_calculos()
else:
    st.title("🔒 Acceso Restringido")
    st.info("Por favor, introducí tus credenciales en el menú lateral para acceder.")
