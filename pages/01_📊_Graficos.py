# pages/graficos.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Gráficos - HEDGEHOG", layout="wide")
st.title("📊 Gráficos de Análisis Técnico Combinados (K=2, K=3, NR/WR)")

# ==============================================================================
# VERIFICAR QUE EXISTEN LOS DATOS CALCULADOS
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

# Preparar las fechas formateadas para el hover
fechas_formateadas = spx_filtered.index.strftime('%Y-%m-%d').tolist()

# --- CREAR SUBPLOTS (5 FILAS) ---
fig_combined = make_subplots(
    rows=5, 
    cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.005, 
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

# **TRAZO FANTASMA (Para anclar el Spike)**
fig_combined.add_trace(go.Scatter(
    x=list(range(len(spx_filtered))),
    y=[0] * len(spx_filtered), 
    mode='lines',
    name='Spike Anchor',
    hoverinfo='skip',
    showlegend=False,
    line=dict(color='rgba(0,0,0,0)'),
    marker=dict(color='rgba(0,0,0,0)')
), row=1, col=1)

# ----------------------------------------------------
# 2. GRÁFICO DE VOLATILIDAD REALIZADA (RV_5d) (Fila 2)
# ----------------------------------------------------

for i in range(len(spx_filtered) - 1):
    color = '#00B06B' if is_up.iloc[i+1] else '#F13A50'
    
    fig_combined.add_trace(go.Scatter(
        x=[i, i+1],
        y=[spx_filtered['RV_5d_pct'].iloc[i], spx_filtered['RV_5d_pct'].iloc[i+1]],
        mode='lines',
        line=dict(color=color, width=2),
        showlegend=False,
        hoverinfo='skip'
    ), row=2, col=1)

fig_combined.add_trace(go.Scatter(
    x=list(range(len(spx_filtered))),
    y=spx_filtered['RV_5d_pct'],
    mode='markers',
    marker=dict(size=0.1, color='rgba(0,0,0,0)'),
    name='RV',
    customdata=[[fecha] for fecha in fechas_formateadas],
    hovertemplate='<b>%{customdata[0]}</b><br>RV: %{y:.2f}%<extra></extra>',
    showlegend=True
), row=2, col=1)

fig_combined.add_shape(
    type="line",
    x0=0, y0=UMBRAL_RV * 100,
    x1=len(spx_filtered) - 1, y1=UMBRAL_RV * 100,
    line=dict(color="orange", width=2, dash="dot"),
    layer="below",
    row=2, col=1
)

fig_combined.add_annotation(
    x=0, y=1.0, 
    text=f'Umbral RV: {UMBRAL_RV*100:.2f}%', 
    showarrow=False,
    xref='x2', yref='y2 domain', 
    xanchor='left', yanchor='top', 
    font=dict(size=12, color="orange"),
    xshift=5, yshift=-5, 
    row=2, col=1
)

fig_combined.update_yaxes(title_text='RV (%)', row=2, col=1, tickformat=".2f")
fig_combined.update_xaxes(showticklabels=False, row=2, col=1) 

# ----------------------------------------------------
# 3. GRÁFICO DE MARKOV K=2 (Fila 3)
# ----------------------------------------------------

fig_combined.add_trace(go.Scatter(
    x=list(range(len(spx_filtered))),
    y=prob_baja_serie_k2,
    mode='lines',
    name='Prob. K=2 (Baja Vol.)', 
    line=dict(color='#8A2BE2', width=2),
    fill='tozeroy', 
    fillcolor='rgba(138, 43, 226, 0.3)',
    customdata=[[fecha] for fecha in fechas_formateadas],
    hovertemplate='<b>%{customdata[0]}</b><br>Prob. Baja K=2: %{y:.4f}<extra></extra>',
    showlegend=True 
), row=3, col=1)

fig_combined.add_shape(
    type="line",
    x0=0, y0=UMBRAL_COMPRESION,
    x1=len(spx_filtered) - 1, y1=UMBRAL_COMPRESION,
    line=dict(color="#FFD700", width=2, dash="dash"), 
    layer="below",
    row=3, col=1
)

fig_combined.add_shape(
    type="line",
    x0=0, y0=UMBRAL_ALERTA,
    x1=len(spx_filtered) - 1, y1=UMBRAL_ALERTA,
    line=dict(color="#FFFFFF", width=1, dash="dot"),
    layer="below",
    row=3, col=1
)

fig_combined.add_annotation(
    x=0, 
    y=UMBRAL_COMPRESION, 
    text=f'Compresión Fuerte ({UMBRAL_COMPRESION*100:.0f}%)', 
    showarrow=False,
    xref='x3', yref='y3', 
    xanchor='left', 
    yanchor='bottom', 
    font=dict(size=12, color="#FFD700"),
    xshift=5, 
    yshift=5, 
    row=3, col=1
)

fig_combined.add_annotation(
    x=0, 
    y=UMBRAL_ALERTA, 
    text=f'Alerta ({UMBRAL_ALERTA*100:.0f}%)', 
    showarrow=False,
    xref='x3', yref='y3', 
    xanchor='left', 
    yanchor='bottom', 
    font=dict(size=12, color="#FFFFFF"), 
    xshift=5, 
    yshift=5,
    row=3, col=1
)

fig_combined.update_yaxes(title_text='Prob. K=2', row=3, col=1, tickformat=".2f", range=[0, 1])
fig_combined.update_xaxes(showticklabels=False, row=3, col=1) 

# ----------------------------------------------------
# 4. GRÁFICO DE MARKOV K=3 (Fila 4)
# ----------------------------------------------------

fig_combined.add_trace(go.Scatter(
    x=list(range(len(spx_filtered))),
    y=prob_k3_consolidada,
    mode='lines',
    name='Prob. K=3 (Baja+Media)', 
    line=dict(color='#00FF7F', width=2),
    fill='tozeroy', 
    fillcolor='rgba(0, 255, 127, 0.3)',
    customdata=[[fecha] for fecha in fechas_formateadas],
    hovertemplate='<b>%{customdata[0]}</b><br>Prob. Consolidada K=3: %{y:.4f}<extra></extra>',
    showlegend=True 
), row=4, col=1)

fig_combined.add_shape(
    type="line",
    x0=0, y0=UMBRAL_COMPRESION,
    x1=len(spx_filtered) - 1, y1=UMBRAL_COMPRESION,
    line=dict(color="#FFD700", width=2, dash="dash"), 
    layer="below",
    row=4, col=1
)

fig_combined.add_shape(
    type="line",
    x0=0, y0=UMBRAL_ALERTA,
    x1=len(spx_filtered) - 1, y1=UMBRAL_ALERTA,
    line=dict(color="#FFFFFF", width=1, dash="dot"),
    layer="below",
    row=4, col=1
)

fig_combined.add_annotation(
    x=0, 
    y=UMBRAL_COMPRESION, 
    text=f'Compresión Fuerte ({UMBRAL_COMPRESION*100:.0f}%)', 
    showarrow=False,
    xref='x4', yref='y4', 
    xanchor='left', 
    yanchor='bottom', 
    font=dict(size=12, color="#FFD700"),
    xshift=5, 
    yshift=5, 
    row=4, col=1
)

fig_combined.add_annotation(
    x=0, 
    y=UMBRAL_ALERTA, 
    text=f'Alerta ({UMBRAL_ALERTA*100:.0f}%)', 
    showarrow=False,
    xref='x4', yref='y4', 
    xanchor='left', 
    yanchor='bottom', 
    font=dict(size=12, color="#FFFFFF"), 
    xshift=5, 
    yshift=5,
    row=4, col=1
)

fig_combined.update_yaxes(title_text='Prob. K=3', row=4, col=1, tickformat=".2f", range=[0, 1])
fig_combined.update_xaxes(showticklabels=False, row=4, col=1)

# ----------------------------------------------------
# 5. GRÁFICO DE SEÑAL NR/WR (Fila 5)
# ----------------------------------------------------

fig_combined.add_trace(go.Bar(
    x=list(range(len(spx_filtered))),
    y=nr_wr_filtered,
    name='Señal NR/WR', 
    marker=dict(
        color='#FF6B35',
        line=dict(width=0)
    ),
    customdata=[[fecha, 'ACTIVA' if s > 0 else 'INACTIVA'] for fecha, s in zip(fechas_formateadas, nr_wr_filtered)],
    hovertemplate='<b>%{customdata[0]}</b><br>NR/WR: %{customdata[1]}<extra></extra>',
    showlegend=True,
    width=0.8
), row=5, col=1)

fig_combined.add_shape(
    type="line",
    x0=-0.5, y0=0.5,
    x1=len(spx_filtered) - 0.5, y1=0.5,
    line=dict(color="#AAAAAA", width=1, dash="dot"),
    layer="below",
    row=5, col=1
)

fig_combined.add_annotation(
    x=0, 
    y=0.9, 
    text='COMPRESIÓN ACTIVA', 
    showarrow=False,
    xref='x5', yref='y5', 
    xanchor='left', 
    yanchor='top', 
    font=dict(size=11, color="#FF6B35"),
    xshift=5, 
    yshift=-5,
    row=5, col=1
)

fig_combined.update_yaxes(title_text='NR/WR', row=5, col=1, range=[0, 1.05], tickvals=[0, 1], ticktext=['OFF', 'ON'])

# --- CONFIGURACIÓN FINAL DEL GRÁFICO COMBINADO ---
fig_combined.update_layout(
    template='plotly_dark',
    height=1100, 
    xaxis_rangeslider_visible=False,
    hovermode='x unified', 
    plot_bgcolor='#131722', 
    paper_bgcolor='#131722', 
    font=dict(color='#AAAAAA'),
    margin=dict(t=50, b=100, l=60, r=40),
    showlegend=True,
    legend=dict(
        orientation="v",
        yanchor="top",
        y=1, 
        xanchor="left",
        x=0.01, 
        bgcolor="rgba(0,0,0,0.5)",
        bordercolor="rgba(255,255,255,0.1)",
        borderwidth=1,
        font=dict(size=10)
    )
)

# ----------------------------------------------------------------------------------
# CONFIGURACIÓN DE SPIKE Y EJES (Solución más confiable)
# ----------------------------------------------------------------------------------

# Deshabilitar spikes en ejes Y (general)
fig_combined.update_yaxes(showspikes=False)

# Deshabilitar spikes en ejes X de las filas 2 a 5
for i in range(2, 6):
    fig_combined.update_xaxes(showspikes=False, row=i, col=1)

# Habilitar el spike ÚNICAMENTE en el eje principal (Fila 1), que tiene el anclaje fantasma.
fig_combined.update_xaxes(
    showspikes=True,
    spikemode='across+toaxis', 
    spikesnap='cursor',
    spikecolor='rgba(255, 255, 255, 0.4)',
    spikethickness=1.5,
    spikedash='dot',
    row=1, 
    col=1
)

# ----------------------------------------------------------------------------------
# CONFIGURACIONES DE EJE X (Estética)
# ----------------------------------------------------------------------------------

# Configurar el eje X compartido (solo las etiquetas del último plot)
fig_combined.update_xaxes(
    tickmode='array',
    tickvals=list(range(len(spx_filtered))),
    ticktext=date_labels,
    tickangle=-45,
    row=5, col=1, 
    showgrid=False
)

# Configuraciones adicionales para ejes en tema oscuro
fig_combined.update_xaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=1, col=1)
fig_combined.update_yaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=1, col=1)
fig_combined.update_xaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=2, col=1)
fig_combined.update_yaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=2, col=1)
fig_combined.update_xaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=3, col=1)
fig_combined.update_yaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=3, col=1)
fig_combined.update_xaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=4, col=1)
fig_combined.update_yaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=4, col=1)
fig_combined.update_xaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=5, col=1)
fig_combined.update_yaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=5, col=1)

st.plotly_chart(fig_combined, use_container_width=True)

# --- INFORMACIÓN ADICIONAL ---
st.markdown("---")
col1, col2, col3, col4, col5, col6 = st.columns(6) 

with col1:
    st.metric("Precio Actual", f"${spx_filtered['Close'].iloc[-1]:.2f}")
with col2:
    cambio = spx_filtered['Close'].iloc[-1] - spx_filtered['Close'].iloc[0]
    cambio_pct = (cambio / spx_filtered['Close'].iloc[0]) * 100
    st.metric(f"Cambio ({fecha_inicio} al {fecha_final})", f"${cambio:.2f}", f"{cambio_pct:.2f}%")
with col3:
    st.metric("Máximo", f"${spx_filtered['High'].max():.2f}")
with col4:
    st.metric("Mínimo", f"${spx_filtered['Low'].min():.2f}")
with col5:
    rv_latest = spx_filtered['RV_5d'].iloc[-1] * 100
    st.metric("RV_5d (Último)", f"{rv_latest:.2f}%")
with col6:
    nr_wr_status = "🟢 ACTIVA" if nr_wr_filtered.iloc[-1] > 0 else "⚪ INACTIVA"
    st.metric("Señal NR/WR", nr_wr_status)
