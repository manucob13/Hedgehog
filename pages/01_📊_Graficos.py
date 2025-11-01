# pages/graficos.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_plotly_events import plotly_events 

st.set_page_config(page_title="Gráficos - HEDGEHOG", layout="wide")
st.title("📊 Gráficos de Análisis Técnico Combinados (K=2, K=3, NR/WR)")

# ==============================================================================
# INICIALIZACIÓN DE ESTADO PARA LA LÍNEA DIBUJADA (AHORA UNA SOLA LÍNEA TEMPORAL)
# ==============================================================================
if 'current_drawn_line_x' not in st.session_state:
    st.session_state['current_drawn_line_x'] = None # No hay línea dibujada inicialmente

def clear_line():
    """Borra la línea actual (la oculta)."""
    st.session_state['current_drawn_line_x'] = None

# ==============================================================================
# VERIFICAR QUE EXISTEN LOS DATOS CALCULADOS (y el resto de tu código de carga...)
# ==============================================================================

if 'datos_calculados' not in st.session_state:
    st.warning("⚠️ No hay datos calculados. Por favor, ve primero a la página principal (Home) para ejecutar los cálculos.")
    st.stop()

# --- RECUPERAR DATOS DE SESSION_STATE ---
datos = st.session_state['datos_calculados']
df_raw = datos['df_raw']
spx = datos['spx']
endog_final = datos['endog_final']
results_k2 = datos['results_k2']
results_k3 = datos['results_k3']
nr_wr_series = datos['nr_wr_series']

st.success(f"✅ Datos cargados desde memoria. ({len(spx)} días disponibles)")

# --- CONTROLES DE FECHA ---
st.sidebar.header("⚙️ Configuración del Gráfico")
fecha_final = spx.index[-1].date()
st.sidebar.info(f"📅 Última fecha disponible: {fecha_final}")
fecha_inicio_default = fecha_final - timedelta(days=90)

fecha_inicio = st.sidebar.date_input(
    "Fecha de inicio:",
    value=fecha_inicio_default,
    min_value=spx.index[0].date(),
    max_value=fecha_final
)

# --- FILTRAR DATOS POR RANGO DE FECHAS ---
fecha_inicio_dt = pd.to_datetime(fecha_inicio)
fecha_final_dt = pd.to_datetime(fecha_final)

spx_filtered = spx[(spx.index >= fecha_inicio_dt) & (spx.index <= fecha_final_dt)].copy()
spx_filtered = spx_filtered[spx_filtered.index.dayofweek < 5]

# --- PREPARACIÓN DE DATOS PARA GRÁFICO COMBINADO ---
date_labels = [d.strftime('%b %d') if i % 5 == 0 else '' for i, d in enumerate(spx_filtered.index)]
date_labels[0] = spx_filtered.index[0].strftime('%b %d')
date_labels[-1] = spx_filtered.index[-1].strftime('%b %d')

spx_filtered['RV_5d_pct'] = spx_filtered['RV_5d'] * 100
UMBRAL_RV = 0.10
spx_filtered['RV_change'] = spx_filtered['RV_5d_pct'].diff()
is_up = spx_filtered['RV_change'] >= 0

prob_baja_serie_k2 = results_k2['prob_baja_serie'].loc[spx_filtered.index].fillna(method='ffill')
prob_baja_serie_k3 = results_k3['prob_baja_serie'].loc[spx_filtered.index].fillna(method='ffill')
prob_media_serie_k3 = results_k3['prob_media_serie'].loc[spx_filtered.index].fillna(method='ffill')
prob_k3_consolidada = prob_baja_serie_k3 + prob_media_serie_k3
nr_wr_filtered = nr_wr_series.reindex(spx_filtered.index).fillna(0)
UMBRAL_ALERTA = 0.50 
UMBRAL_COMPRESION = results_k2['UMBRAL_COMPRESION']
fechas_formateadas = spx_filtered.index.strftime('%Y-%m-%d').tolist()

# --- CREAR SUBPLOTS (5 FILAS) ---
fig_combined = make_subplots(
    rows=5, 
    cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.02, 
    row_heights=[0.45, 0.13, 0.14, 0.14, 0.14],
)

# ----------------------------------------------------
# 1. GRÁFICO DE VELAS JAPONESAS (Fila 1)
# ----------------------------------------------------
hover_text_candles = [
    f"<b>{fecha}</b><br>Open: {o:.2f}<br>High: {h:.2f}<br>Low: {l:.2f}<br>Close: {c:.2f}"
    for fecha, o, h, l, c in zip(
        fechas_formateadas,
        spx_filtered['Open'],
        spx_filtered['High'],
        spx_filtered['Low'],
        spx_filtered['Close']
    )
]

fig_combined.add_trace(go.Candlestick(
    x=list(range(len(spx_filtered))),
    open=spx_filtered['Open'],
    high=spx_filtered['High'],
    low=spx_filtered['Low'],
    close=spx_filtered['Close'],
    name='S&P 500',
    text=hover_text_candles,
    hoverinfo='text',
    increasing=dict(line=dict(color='#00B06B')),
    decreasing=dict(line=dict(color='#F13A50'))
), row=1, col=1)

fig_combined.update_yaxes(title_text='Precio', row=1, col=1)
fig_combined.update_xaxes(showticklabels=False, row=1, col=1)

# ----------------------------------------------------
# 2. GRÁFICO DE VOLATILIDAD REALIZADA (RV_5d) (Fila 2)
# ----------------------------------------------------
for i in range(len(spx_filtered) - 1):
    color = '#00B06B' if is_up.iloc[i+1] else '#F13A50'
    fig_combined.add_trace(go.Scatter(
        x=[i, i+1], y=[spx_filtered['RV_5d_pct'].iloc[i], spx_filtered['RV_5d_pct'].iloc[i+1]],
        mode='lines', line=dict(color=color, width=2), showlegend=False, hoverinfo='skip'
    ), row=2, col=1)
fig_combined.add_trace(go.Scatter(
    x=list(range(len(spx_filtered))), y=spx_filtered['RV_5d_pct'], mode='markers',
    marker=dict(size=0.1, color='rgba(0,0,0,0)'), name='RV',
    customdata=[[fecha] for fecha in fechas_formateadas],
    hovertemplate='<b>%{customdata[0]}</b><br>RV: %{y:.2f}%<extra></extra>', showlegend=True
), row=2, col=1)
fig_combined.add_shape(type="line", x0=0, y0=UMBRAL_RV * 100, x1=len(spx_filtered) - 1, y1=UMBRAL_RV * 100,
                       line=dict(color="orange", width=2, dash="dot"), layer="below", row=2, col=1)
fig_combined.update_yaxes(title_text='RV (%)', row=2, col=1, tickformat=".2f")
fig_combined.update_xaxes(showticklabels=False, row=2, col=1) 
fig_combined.add_annotation(x=0, y=1.0, text=f'Umbral RV: {UMBRAL_RV*100:.2f}%', showarrow=False,
                            xref='x2', yref='y2 domain', xanchor='left', yanchor='top', 
                            font=dict(size=12, color="orange"), xshift=5, yshift=-5, row
