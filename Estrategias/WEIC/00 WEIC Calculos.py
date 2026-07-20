"""
00 WEIC Calculos.py
====================
Weekly Iron Condor — Calculadora de movimiento esperado SPX.

- Descarga SPX, VIX y VIX3M semanales desde Yahoo Finance
- Segmenta el histórico por régimen de VIX y calcula el std de los
  log-returns semanales (metodología del notebook Vol_SPX_5DTE_2_0_prep)
- Fijo a 5 DTE: entrada Lunes, salida Viernes
- El usuario ajusta: std (stdn)
- Muestra las bandas de precio esperado y un gráfico del rango proyectado
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta
import warnings
warnings.filterwarnings("ignore")

from utils.utils import check_password
from utils.volatility import calcular_rango_esperado


# ==============================================================================
# HELPERS
# ==============================================================================

def get_next_monday() -> date:
    today = date.today()
    days_ahead = (0 - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return today + timedelta(days=days_ahead)


def get_next_friday() -> date:
    today = date.today()
    days_ahead = (4 - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return today + timedelta(days=days_ahead)


@st.cache_data(ttl=3600, show_spinner=False)
def obtener_rango_esperado(stdn: float):
    """Wrapper cacheado sobre calcular_rango_esperado (1h de cache)."""
    return calcular_rango_esperado(stdn=stdn)


# ==============================================================================
# GRAFICO
# ==============================================================================

def construir_grafico(resultado, fecha_salida: date):
    """
    Grafico de velas/linea con el historico semanal reciente del SPX
    y la banda esperada (stdn) proyectada hacia la fecha de salida.
    """
    hist = resultado.hist_df.tail(26)  # ultimas ~26 semanas

    fig = go.Figure()

    # --- Historico de cierres semanales ---
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist["Close"],
        mode="lines",
        name="SPX (cierre semanal)",
        line=dict(color="#5b8def", width=2),
    ))

    fecha_proyeccion = pd.Timestamp(fecha_salida)
    ultima_fecha = hist.index[-1]

    # --- Lineas de proyeccion desde el ultimo cierre hasta la banda ---
    fig.add_trace(go.Scatter(
        x=[ultima_fecha, fecha_proyeccion],
        y=[resultado.last_close, resultado.band_up],
        mode="lines",
        name="Banda superior",
        line=dict(color="#e05c5c", width=2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=[ultima_fecha, fecha_proyeccion],
        y=[resultado.last_close, resultado.band_dw],
        mode="lines",
        name="Banda inferior",
        line=dict(color="#1baf7a", width=2, dash="dot"),
    ))

    # --- Marcadores de los puntos clave ---
    fig.add_trace(go.Scatter(
        x=[ultima_fecha],
        y=[resultado.last_close],
        mode="markers+text",
        name="Spot actual",
        marker=dict(color="#5b8def", size=10),
        text=[f"{resultado.last_close:,.0f}"],
        textposition="middle left",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[fecha_proyeccion],
        y=[resultado.band_up],
        mode="markers+text",
        marker=dict(color="#e05c5c", size=10),
        text=[f"{resultado.band_up:,.0f}"],
        textposition="top right",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[fecha_proyeccion],
        y=[resultado.band_dw],
        mode="markers+text",
        marker=dict(color="#1baf7a", size=10),
        text=[f"{resultado.band_dw:,.0f}"],
        textposition="bottom right",
        showlegend=False,
    ))

    # --- Sombreado del area de la banda ---
    fig.add_trace(go.Scatter(
        x=[fecha_proyeccion, fecha_proyeccion],
        y=[resultado.band_dw, resultado.band_up],
        mode="lines",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.update_layout(
        title=f"SPX — Rango esperado proyectado ({resultado.stdn}σ, régimen VIX {resultado.regime_label})",
        xaxis_title="Fecha",
        yaxis_title="SPX",
        height=450,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )

    return fig


# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================

def main_weic_calculos():

    st.markdown(
        "<h1><span style='font-size:1.5em;'>🦅</span> WEIC — Movimiento Esperado SPX</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "Calcula las bandas de precio esperado en función del régimen de VIX vigente, "
        "para una entrada Lunes → salida Viernes (5 DTE fijo)."
    )
    st.markdown("---")

    # --------------------------------------------------------------------------
    # SECCIÓN 1 — Parámetros
    # --------------------------------------------------------------------------
    st.header("1. Parámetros")

    col_std, col_btn = st.columns([2, 1])
    with col_std:
        stdn = st.number_input(
            "Desviaciones estándar (σ)",
            min_value=0.5,
            max_value=4.0,
            value=2.5,
            step=0.1,
            format="%.1f",
            help="Número de desviaciones para calcular las bandas, sobre el std de "
                 "log-returns semanales del régimen de VIX vigente.",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Actualizar datos", type="primary"):
            st.cache_data.clear()
            st.rerun()

    fecha_entrada = get_next_monday()
    fecha_salida  = get_next_friday()

    st.info(
        f"📅 **Entrada:** Lunes {fecha_entrada.strftime('%d/%m/%Y')}  "
        f"→  **Salida:** Viernes {fecha_salida.strftime('%d/%m/%Y')}  "
        f"|  **DTE:** 5 días (fijo)"
    )

    st.markdown("---")

    # --------------------------------------------------------------------------
    # SECCIÓN 2 — Señal de entrada
    # --------------------------------------------------------------------------
    st.header("2. Señal de entrada")

    with st.spinner("Descargando datos semanales y calculando régimen de VIX..."):
        try:
            resultado = obtener_rango_esperado(stdn)
        except Exception as e:
            st.error(f"❌ {e}")
            st.stop()

    senal = resultado.senal

    if senal.signal == 1:
        st.success(
            f"✅ **Signal = 1** — se cumplen las 4 condiciones "
            f"(Tendencia Alcista, VIX en rango, VIX < VIX_WMA_21, Contango). "
            f"Aplica al lunes **{senal.new_date}**."
        )
    else:
        fallidas = []
        if not senal.cond_tendencia:
            fallidas.append("Tendencia no es Alcista")
        if not senal.cond_vix_rango:
            fallidas.append("VIX fuera del rango 10-25")
        if not senal.cond_vix_wma:
            fallidas.append("VIX no está por debajo de su VIX_WMA_21")
        if not senal.cond_contango:
            fallidas.append("Term Structure no está en Contango")
        st.error(
            f"⛔ **Signal = 0** — falla: {', '.join(fallidas)}. "
            f"Aplica al lunes **{senal.new_date}**."
        )

    valores_ordenados = {
        "New Date":                  senal.new_date,
        "Signal":                    senal.signal,
        "Last Close":                senal.last_close,
        "Last SP500_WMA_30":         senal.last_sp500_wma30,
        "Tendencia":                 senal.tendencia,
        "Last VIX":                  senal.last_vix,
        "Last VIX_WMA_21":           senal.last_vix_wma21,
        "VIX en rango (10-25)":      senal.vix_en_rango,
        "VIX < VIX WMA21":           senal.vix_lt_wma21,
        "Term Structure":            senal.term_structure,
        "VIX WMA21 bajando (info)":  senal.vix_wma21_bajando,
        "Realized Vol (anual %)":    senal.realized_vol_anual_pct,
        "VRP (info)":                senal.vrp_positive,
    }

    condicion_senal = {
        "New Date":                 "",
        "Signal":                   "",
        "Tendencia":                "Cumple" if senal.cond_tendencia else "No cumple",
        "Last Close":               "",
        "Last SP500_WMA_30":        "",
        "Last VIX":                 "",
        "Last VIX_WMA_21":          "",
        "VIX en rango (10-25)":     "Cumple" if senal.cond_vix_rango else "No cumple",
        "VIX < VIX WMA21":          "Cumple" if senal.cond_vix_wma else "No cumple",
        "Term Structure":           "Cumple" if senal.cond_contango else "No cumple",
        "VIX WMA21 bajando (info)": "-- info --",
        "Realized Vol (anual %)":   "",
        "VRP (info)":               "-- info --",
    }

    descripciones = {
        "New Date":                 "Fecha del próximo día hábil (lunes) al que aplica la señal",
        "Signal":                   "Señal binaria: 1 = abrir Iron Condor 5DTE, 0 = no operar",
        "Last Close":               "Cierre del SP500 en la última semana cerrada",
        "Last SP500_WMA_30":        "Media móvil ponderada del SP500 (5 semanas), referencia de tendencia",
        "Tendencia":                "Alcista si el cierre está por encima de su WMA_30, si no Bajista",
        "Last VIX":                 "Cierre del VIX en la última semana",
        "Last VIX_WMA_21":          "Media móvil ponderada del VIX (aprox 1 año)",
        "VIX en rango (10-25)":     "VIX dentro del rango operativo definido (ni muy bajo ni muy alto)",
        "VIX < VIX WMA21":          "Volatilidad implícita actual por debajo de su propia media (relajándose)",
        "Term Structure":           "Contango (VIX<VIX3M, calma) o Backwardation (VIX>VIX3M, estrés)",
        "VIX WMA21 bajando (info)": "La media del VIX lleva bajando respecto a la semana anterior. No entra en la señal",
        "Realized Vol (anual %)":   "Volatilidad realizada anualizada del SP500 (últimas 4 semanas)",
        "VRP (info)":               "VIX > Vol. Realizada: prima de riesgo de volatilidad positiva, favorable para vender opciones. No entra en la señal",
    }

    tabla_senal = pd.DataFrame({
        "Field":        list(valores_ordenados.keys()),
        "Value":        list(valores_ordenados.values()),
        "Cumple Señal": [condicion_senal.get(k, "") for k in valores_ordenados.keys()],
        "Description":  [descripciones.get(k, "") for k in valores_ordenados.keys()],
    })

    st.dataframe(tabla_senal, use_container_width=True, hide_index=True)

    st.markdown("---")

    # --------------------------------------------------------------------------
    # SECCIÓN 3 — Cálculo del rango esperado
    # --------------------------------------------------------------------------
    st.header("3. Movimiento esperado")

    spot         = resultado.last_close
    current_vix  = resultado.current_vix
    movimiento   = resultado.band_up - spot
    move_pct     = resultado.move_pct / 2  # amplitud total / 2 = % a cada lado

    st.caption(
        f"Régimen VIX: **{resultado.regime_label}**  |  "
        f"Semanas en régimen: **{resultado.regime_rows}**  |  "
        f"Std log-return semanal: **{resultado.sigma_regimen:.5f}**  |  "
        f"Tendencia: **{resultado.trend}**  |  "
        f"Term structure: **{resultado.term_structure}**  |  "
        f"σ: **{resultado.stdn}**"
    )

    s_move_pct  = f"± {move_pct:.2f}%"
    s_move_pts  = f"± {int(round(movimiento)):,} pts".replace(",", ".")
    s_banda_inf = f"{int(round(resultado.band_dw)):,}".replace(",", ".")
    s_banda_sup = f"{int(round(resultado.band_up)):,}".replace(",", ".")
    s_spot      = f"{int(round(spot)):,}".replace(",", ".")
    s_vix       = f"{current_vix:.2f}"
    s_dte       = "5 días"
    s_rango     = "Lunes → Viernes"
    s_regimen   = resultado.regime_label

    html_cards = (
        '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:1rem 0 1.5rem;">'

        '<div style="background:var(--secondary-background-color);border-radius:12px;'
        'padding:1.5rem;text-align:center;border:1px solid rgba(128,128,128,0.2);">'
        '<div style="font-size:13px;color:gray;margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em;">Movimiento esperado</div>'
        '<div style="font-size:36px;font-weight:700;">' + s_move_pct + '</div>'
        '<div style="font-size:18px;color:gray;margin-top:6px;">' + s_move_pts + '</div>'
        '</div>'

        '<div style="background:var(--secondary-background-color);border-radius:12px;'
        'padding:1.5rem;text-align:center;border:2px solid #1baf7a;">'
        '<div style="font-size:13px;color:#1baf7a;margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em;">Banda inferior</div>'
        '<div style="font-size:36px;font-weight:700;color:#1baf7a;">' + s_banda_inf + '</div>'
        '<div style="font-size:15px;color:gray;margin-top:6px;">− ' + s_move_pts.replace('± ', '') + ' vs spot</div>'
        '</div>'

        '<div style="background:var(--secondary-background-color);border-radius:12px;'
        'padding:1.5rem;text-align:center;border:2px solid #e05c5c;">'
        '<div style="font-size:13px;color:#e05c5c;margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em;">Banda superior</div>'
        '<div style="font-size:36px;font-weight:700;color:#e05c5c;">' + s_banda_sup + '</div>'
        '<div style="font-size:15px;color:gray;margin-top:6px;">+ ' + s_move_pts.replace('± ', '') + ' vs spot</div>'
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

    st.markdown("---")

    # --------------------------------------------------------------------------
    # SECCIÓN 3 — Gráfico
    # --------------------------------------------------------------------------
    st.header("4. Gráfico del rango proyectado")

    fig = construir_grafico(resultado, fecha_salida)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📊 Ver estadísticas de todos los regímenes de VIX"):
        st.dataframe(
            resultado.stats_df.style.format({
                "N_semanas": "{:.0f}",
                "Mean_logret": "{:.5f}",
                "Std_logret": "{:.5f}",
                "P5_logret": "{:.5f}",
                "P95_logret": "{:.5f}",
            }),
            use_container_width=True,
        )

    st.caption(
        f"Última semana cerrada: {resultado.last_date}  |  "
        "Estimación basada en el std histórico de log-returns semanales por régimen de VIX. "
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
