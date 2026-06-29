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
import warnings
warnings.filterwarnings("ignore")

from utils.utils import check_password, fetch_data
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
    DTE = días desde entrada (inclusive) hasta salida (inclusive).
    Lunes→Viernes = 5, Lunes→Jueves = 4
    Martes→Viernes = 4, Martes→Jueves = 3
    Miércoles→Viernes = 3, Miércoles→Jueves = 2
    """
    dias = {"Lunes": 0, "Martes": 1, "Miércoles": 2, "Jueves": 3, "Viernes": 4}
    return dias[salida] - dias[entrada] + 1


def obtener_datos_actuales() -> dict:
    """
    Usa fetch_data() de utils.py (cacheada) y extrae el último dato disponible.
    fetch_data() descarga ^GSPC + ^VIX + ^VIX3M desde 2010.
    """
    try:
        df = fetch_data()
        if df.empty:
            return {"error": "No se pudieron obtener datos."}
        last = df.iloc[-1]
        return {
            "spx_close": float(last["Close"]),
            "spx_open":  float(last["Open"]),
            "vix":       float(last["VIX"]),
            "last_date": df.index[-1].strftime("%Y-%m-%d"),
            "error":     None,
        }
    except Exception as e:
        return {"error": str(e)}


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

    with st.spinner("Cargando SPX y VIX..."):
        datos = obtener_datos_actuales()

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
    dias_hasta_entrada = (dias_map[dia_entrada] - hoy.weekday()) % 7
    # Si dias_hasta_entrada == 0 significa que hoy ES ese día → usar hoy
    fecha_entrada = hoy + timedelta(days=dias_hasta_entrada)

    st.info(
        f"📅 **Entrada:** {dia_entrada} {fecha_entrada.strftime('%d/%m/%Y')}  "
        f"→  **Salida:** {dia_salida} {fecha_salida.strftime('%d/%m/%Y')}  "
        f"|  **DTE:** {dte} días"
    )

    st.markdown("---")

    # --------------------------------------------------------------------------
    # SECCIÓN 3 — Cálculo de bandas y resumen visual
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
    vol_diaria   = resultado.vol_daily
    vol_escalada = vol_diaria * (dte ** 0.5)
    movimiento   = spot * vol_escalada * stdn
    move_pct     = vol_escalada * stdn * 100
    banda_inf    = spot - movimiento
    banda_sup    = spot + movimiento

    st.caption(
        f"Régimen VIX: **{resultado.regime_label}**  |  "
        f"Vol diaria GK: **{vol_diaria:.5f}**  |  "
        f"Vol escalada ×√{dte}: **{vol_escalada:.5f}**  |  "
        f"σ: **{stdn}**"
    )

    # Precalcular strings para evitar conflictos con f-string y formato de comas
    s_move_pct  = f"{move_pct:.2f}%"
    s_move_pts  = f"{int(round(movimiento)):,} pts".replace(",", ".")
    s_banda_inf = f"{int(round(banda_inf)):,}".replace(",", ".")
    s_banda_sup = f"{int(round(banda_sup)):,}".replace(",", ".")
    s_spot      = f"{int(round(spot)):,}".replace(",", ".")
    s_vix       = f"{current_vix:.2f}"
    s_dte       = f"{dte} días"
    s_rango     = f"{dia_entrada} → {dia_salida}"
    s_regimen   = resultado.regime_label

    html_cards = (
        '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:1rem 0 1.5rem;">'

        '<div style="background:var(--secondary-background-color);border-radius:12px;'
        'padding:1.5rem;text-align:center;border:1px solid rgba(128,128,128,0.2);">'
        '<div style="font-size:13px;color:gray;margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em;">Movimiento esperado</div>'
        '<div style="font-size:36px;font-weight:700;">± ' + s_move_pct + '</div>'
        '<div style="font-size:18px;color:gray;margin-top:6px;">± ' + s_move_pts + '</div>'
        '</div>'

        '<div style="background:var(--secondary-background-color);border-radius:12px;'
        'padding:1.5rem;text-align:center;border:2px solid #1baf7a;">'
        '<div style="font-size:13px;color:#1baf7a;margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em;">Banda inferior</div>'
        '<div style="font-size:36px;font-weight:700;color:#1baf7a;">' + s_banda_inf + '</div>'
        '<div style="font-size:15px;color:gray;margin-top:6px;">− ' + s_move_pts + ' vs spot</div>'
        '</div>'

        '<div style="background:var(--secondary-background-color);border-radius:12px;'
        'padding:1.5rem;text-align:center;border:2px solid #e05c5c;">'
        '<div style="font-size:13px;color:#e05c5c;margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em;">Banda superior</div>'
        '<div style="font-size:36px;font-weight:700;color:#e05c5c;">' + s_banda_sup + '</div>'
        '<div style="font-size:15px;color:gray;margin-top:6px;">+ ' + s_move_pts + ' vs spot</div>'
        '</div>'

        '</div>'

        '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:1rem;">'

        '<div style="background:var(--secondary-background-color);border-radius:10px;'
        'padding:1rem;text-align:center;border:1px solid rgba(128,128,128,0.15);">'
        '<div style="font-size:11px;color:gray;margin-bottom:4px;text-transform:uppercase;">Spot SPX</div>'
        '<div style="font-size:22px;font-weight:600;">' + s_spot + '</div>'
        '</div>'

        '<div style="background:var(--secondary-background-color);border-radius:10px;'
        'padding:1rem;text-align:center;border:1px solid rgba(128,128,128,0.15);">'
        '<div style="font-size:11px;color:gray;margin-bottom:4px;text-transform:uppercase;">VIX</div>'
        '<div style="font-size:22px;font-weight:600;">' + s_vix + '</div>'
        '</div>'

        '<div style="background:var(--secondary-background-color);border-radius:10px;'
        'padding:1rem;text-align:center;border:1px solid rgba(128,128,128,0.15);">'
        '<div style="font-size:11px;color:gray;margin-bottom:4px;text-transform:uppercase;">DTE</div>'
        '<div style="font-size:22px;font-weight:600;">' + s_dte + '</div>'
        '<div style="font-size:11px;color:gray;">' + s_rango + '</div>'
        '</div>'

        '<div style="background:var(--secondary-background-color);border-radius:10px;'
        'padding:1rem;text-align:center;border:1px solid rgba(128,128,128,0.15);">'
        '<div style="font-size:11px;color:gray;margin-bottom:4px;text-transform:uppercase;">Régimen</div>'
        '<div style="font-size:16px;font-weight:600;">' + s_regimen + '</div>'
        '</div>'

        '</div>'
    )

    st.markdown(html_cards, unsafe_allow_html=True)

    st.caption(
        "Estimación basada en volatilidad histórica Garman-Klass. "
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
