"""
00 WEIC Calculos.py
====================
Weekly Iron Condor — Calculadora de movimiento esperado SPX.

- Descarga SPX, VIX y VIX3M semanales desde Yahoo Finance
- Segmenta el histórico por régimen de VIX y calcula el std de los
  log-returns semanales (metodología del notebook Vol_SPX_5DTE_2_0_prep)
- Fijo a 5 DTE: entrada Lunes, salida Viernes
- El usuario ajusta: std (stdn), por defecto 2
- Muestra las bandas de precio esperado y un gráfico del rango proyectado
"""

import html
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta
import warnings
warnings.filterwarnings("ignore")

from utils.utils import check_password
from utils.volatility import calcular_rango_esperado


# ==============================================================================
# HELPERS FECHAS
# ==============================================================================

def get_next_monday() -> date:
    today = date.today()
    days_ahead = (0 - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return today + timedelta(days=days_ahead)


def get_next_friday_from_monday(monday_date: date) -> date:
    return monday_date + timedelta(days=4)


def get_reference_monday_from_result(resultado) -> date:
    """
    Usa la fecha ya calculada por volatility.py (senal.new_date), que debería ser
    el próximo lunes aplicable a la señal. Si falla, cae al próximo lunes natural.
    """
    try:
        return pd.to_datetime(resultado.senal.new_date).date()
    except Exception:
        return get_next_monday()


# ==============================================================================
# CACHE
# ==============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def obtener_rango_esperado(stdn: float):
    """Wrapper cacheado sobre calcular_rango_esperado (1h de cache)."""
    return calcular_rango_esperado(stdn=stdn)


# ==============================================================================
# GRAFICO
# ==============================================================================

def construir_grafico(resultado, fecha_salida: date):
    """
    Grafico de linea con el historico semanal reciente del SPX
    y la banda esperada (stdn) proyectada hacia la fecha de salida.
    """
    hist = resultado.hist_df.tail(26)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist["Close"],
        mode="lines",
        name="SPX (cierre semanal)",
        line=dict(color="#5b8def", width=2),
    ))

    fecha_proyeccion = pd.Timestamp(fecha_salida)
    ultima_fecha = hist.index[-1]

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
# HELPERS DE FORMATO
# ==============================================================================

def _fmt_bool(v):
    return "Sí" if bool(v) else "No"


def _fmt_num(v, decimals=2):
    try:
        return f"{float(v):,.{decimals}f}"
    except Exception:
        return str(v)


# ==============================================================================
# ESTILOS COMPARTIDOS DE TABLAS
# ==============================================================================

def _weic_table_styles() -> str:
    return (
        '<style>'
        '.weic-table-wrap{margin-top:0.5rem;margin-bottom:1.25rem;border:1px solid rgba(128,128,128,0.18);'
        'border-radius:14px;overflow:hidden;background:rgba(255,255,255,0.02);}'
        '.weic-table-title{padding:0.7rem 1rem;font-weight:700;font-size:0.95rem;'
        'background:rgba(120,120,120,0.14);border-bottom:1px solid rgba(128,128,128,0.18);}'
        '.weic-table{width:100%;border-collapse:collapse;font-size:0.95rem;}'
        '.weic-table thead th{text-align:left;padding:0.9rem 1rem;background:rgba(120,120,120,0.10);'
        'border-bottom:1px solid rgba(128,128,128,0.18);font-weight:700;}'
        '.weic-table tbody td{padding:0.82rem 1rem;border-bottom:1px solid rgba(128,128,128,0.10);vertical-align:top;}'
        '.weic-table tbody tr:last-child td{border-bottom:none;}'
        '.weic-row-ok{background:rgba(27,175,122,0.06);}'
        '.weic-row-bad{background:rgba(224,92,92,0.06);}'
        '.weic-row-info{background:transparent;}'
        '.weic-pill{display:inline-block;padding:0.28rem 0.65rem;border-radius:999px;font-size:0.84rem;'
        'font-weight:700;white-space:nowrap;}'
        '.weic-pill-ok{background:rgba(27,175,122,0.18);color:#72e0b5;border:1px solid rgba(27,175,122,0.28);}'
        '.weic-pill-bad{background:rgba(224,92,92,0.18);color:#ff9a9a;border:1px solid rgba(224,92,92,0.28);}'
        '.weic-pill-info{background:rgba(140,140,140,0.14);color:#cfcfcf;border:1px solid rgba(160,160,160,0.18);}'
        '.col-field{width:26%;font-weight:650;}'
        '.col-value{width:20%;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-weight:600;}'
        '.col-status{width:18%;}'
        '.col-desc{width:56%;color:rgba(250,250,250,0.78);}'
        '@media (max-width:900px){.weic-table{font-size:0.88rem;}.col-desc{display:none;}'
        '.col-field{width:40%;}.col-value{width:30%;}.col-status{width:30%;}}'
        '</style>'
    )


def _badge(status: str) -> str:
    if status == "ok":
        return '<span class="weic-pill weic-pill-ok">🟢 Cumple</span>'
    if status == "bad":
        return '<span class="weic-pill weic-pill-bad">🔴 No cumple</span>'
    return '<span class="weic-pill weic-pill-info">⚪ Info</span>'


def _build_table_html(title: str, headers: list, rows_html: list) -> str:
    header_html = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    return (
        f'<div class="weic-table-wrap">'
        f'<div class="weic-table-title">{html.escape(title)}</div>'
        f'<table class="weic-table">'
        f'<thead><tr>{header_html}</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        f'</table></div>'
    )


# ==============================================================================
# TABLA 1 — DATOS BASE (Last Close, WMA, VIX, VIX WMA)
# ==============================================================================

def construir_tabla_datos_base_html(senal) -> str:
    rows = [
        {"field": "Last Close", "value": _fmt_num(senal.last_close, 2),
         "desc": "Cierre del SP500 en la última semana cerrada"},
        {"field": "Last SP500_WMA_30", "value": _fmt_num(senal.last_sp500_wma30, 2),
         "desc": "Media móvil ponderada del SP500 (5 semanas)"},
        {"field": "Last VIX", "value": _fmt_num(senal.last_vix, 2),
         "desc": "Cierre del VIX en la última semana"},
        {"field": "Last VIX_WMA_21", "value": _fmt_num(senal.last_vix_wma21, 2),
         "desc": "Media móvil ponderada del VIX (aprox. 1 año)"},
    ]

    rows_html = []
    for r in rows:
        row_html = (
            f'<tr class="weic-row-info">'
            f'<td class="col-field">{html.escape(str(r["field"]))}</td>'
            f'<td class="col-value">{html.escape(str(r["value"]))}</td>'
            f'<td class="col-desc">{html.escape(str(r["desc"]))}</td>'
            f'</tr>'
        )
        rows_html.append(row_html)

    return _build_table_html(
        "📌 Datos base",
        ["Field", "Value", "Description"],
        rows_html,
    )


# ==============================================================================
# TABLA 2 — CONDICIONES DE LA SEÑAL (con semáforo)
# ==============================================================================

def construir_tabla_condiciones_html(senal) -> str:
    rows = [
        {"field": "Signal", "value": "1" if senal.signal == 1 else "0",
         "status": "ok" if senal.signal == 1 else "bad",
         "desc": "Señal binaria: 1 = abrir Iron Condor 5DTE, 0 = no operar"},
        {"field": "Tendencia", "value": senal.tendencia,
         "status": "ok" if senal.cond_tendencia else "bad",
         "desc": "Alcista si el cierre está por encima de su WMA_30"},
        {"field": "VIX en rango (10-25)", "value": _fmt_bool(senal.vix_en_rango),
         "status": "ok" if senal.cond_vix_rango else "bad",
         "desc": "VIX dentro del rango operativo definido"},
        {"field": "VIX < VIX_WMA21", "value": _fmt_bool(senal.vix_lt_wma21),
         "status": "ok" if senal.cond_vix_wma else "bad",
         "desc": "Volatilidad implícita relajándose"},
        {"field": "Term Structure", "value": senal.term_structure,
         "status": "ok" if senal.cond_contango else "bad",
         "desc": "Contango (VIX < VIX3M) o Backwardation"},
    ]

    rows_html = []
    for r in rows:
        row_class = "weic-row-ok" if r["status"] == "ok" else "weic-row-bad"
        row_html = (
            f'<tr class="{row_class}">'
            f'<td class="col-field">{html.escape(str(r["field"]))}</td>'
            f'<td class="col-value">{html.escape(str(r["value"]))}</td>'
            f'<td class="col-status">{_badge(r["status"])}</td>'
            f'<td class="col-desc">{html.escape(str(r["desc"]))}</td>'
            f'</tr>'
        )
        rows_html.append(row_html)

    return _build_table_html(
        "🚦 Condiciones de la señal",
        ["Field", "Value", "Semáforo", "Description"],
        rows_html,
    )


# ==============================================================================
# TABLA 3 — INFORMATIVAS (no entran en la señal)
# ==============================================================================

def construir_tabla_informativas_html(senal) -> str:
    rows = [
        {"field": "New Date", "value": senal.new_date,
         "desc": "Fecha del próximo lunes al que aplica la señal"},
        {"field": "VIX WMA21 bajando", "value": _fmt_bool(senal.vix_wma21_bajando),
         "desc": "Informativo, no entra en la señal"},
        {"field": "Realized Vol (anual %)", "value": _fmt_num(senal.realized_vol_anual_pct, 2),
         "desc": "Volatilidad realizada anualizada del SP500"},
        {"field": "VRP positiva", "value": _fmt_bool(senal.vrp_positive),
         "desc": "VIX > vol. realizada, favorable para venta de opciones"},
    ]

    rows_html = []
    for r in rows:
        row_html = (
            f'<tr class="weic-row-info">'
            f'<td class="col-field">{html.escape(str(r["field"]))}</td>'
            f'<td class="col-value">{html.escape(str(r["value"]))}</td>'
            f'<td class="col-desc">{html.escape(str(r["desc"]))}</td>'
            f'</tr>'
        )
        rows_html.append(row_html)

    return _build_table_html(
        "ℹ️ Informativas (no entran en la señal)",
        ["Field", "Value", "Description"],
        rows_html,
    )


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
            value=2.0,
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

    with st.spinner("Descargando datos semanales y calculando régimen de VIX..."):
        try:
            resultado = obtener_rango_esperado(stdn)
        except Exception as e:
            st.error(f"❌ {e}")
            st.stop()

    # Fechas corregidas apoyándonos en la señal calculada
    fecha_entrada = get_reference_monday_from_result(resultado)
    fecha_salida = get_next_friday_from_monday(fecha_entrada)

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

    # --- Estilos compartidos (una sola vez) ---
    st.markdown(_weic_table_styles(), unsafe_allow_html=True)

    # --- Tabla 1: Datos base ---
    st.markdown(construir_tabla_datos_base_html(senal), unsafe_allow_html=True)

    # --- Tabla 2: Condiciones + semáforo ---
    st.markdown(construir_tabla_condiciones_html(senal), unsafe_allow_html=True)

    # --- Tabla 3: Informativas ---
    st.markdown(construir_tabla_informativas_html(senal), unsafe_allow_html=True)

    st.markdown("---")

    # --------------------------------------------------------------------------
    # SECCIÓN 3 — Cálculo del rango esperado
    # --------------------------------------------------------------------------
    st.header("3. Movimiento esperado")

    spot = resultado.last_close
    current_vix = resultado.current_vix
    movimiento = resultado.band_up - spot
    move_pct = resultado.move_pct / 2

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
    # SECCIÓN 4 — Gráfico
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
