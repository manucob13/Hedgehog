import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots # Importamos subplots
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Gráficos - HEDGEHOG", layout="wide")
st.title("📊 Gráficos de Análisis Técnico Combinados")

# --- FUNCIÓN PARA CARGAR DATOS (Reutilizada de app.py) ---
@st.cache_data(ttl=86400)
def fetch_data():
    """Descarga datos históricos del ^GSPC (SPX) y ^VIX (VIX)."""
    start = "2010-01-01" 
    end = datetime.now()

    spx = yf.download("^GSPC", start=start, end=end, auto_adjust=False, multi_level_index=False, progress=False)
    vix = yf.download("^VIX", start=start, end=end, auto_adjust=False, multi_level_index=False, progress=False)

    spx.index = pd.to_datetime(spx.index)
    vix_series = vix['Close'].rename('VIX')
    vix_series.index = pd.to_datetime(vix_series.index)

    df_merged = spx.merge(vix_series, how='left', left_index=True, right_index=True)
    df_merged.dropna(subset=['VIX'], inplace=True)
    
    return df_merged

@st.cache_data(ttl=3600)
def calculate_indicators(df_raw: pd.DataFrame):
    """Calcula todos los indicadores técnicos necesarios."""
    spx = df_raw.copy()

    # 1. Volatilidad Realizada (RV_5d)
    spx['log_ret'] = np.log(spx['Close'] / spx['Close'].shift(1))
    # RV_5d: Desviación estándar de los retornos logarítmicos de 5 días, anualizada (multiplicado por sqrt(252))
    spx['RV_5d'] = spx['log_ret'].rolling(window=5).std() * np.sqrt(252)

    # 2. Average True Range (ATR_14)
    spx['tr1'] = spx['High'] - spx['Low']
    spx['tr2'] = (spx['High'] - spx['Close'].shift(1)).abs()
    spx['tr3'] = (spx['Low'] - spx['Close'].shift(1)).abs()
    spx['true_range'] = spx[['tr1', 'tr2', 'tr3']].max(axis=1)
    spx['ATR_14'] = spx['true_range'].rolling(window=14).mean()
    spx.drop(columns=['tr1', 'tr2', 'tr3', 'true_range'], inplace=True)

    # 3. Narrow Range (NR14)
    window = 14
    spx['nr14_threshold'] = spx['High'].rolling(window=window).max() - spx['Low'].rolling(window=window).min()
    spx['NR14'] = (spx['High'] - spx['Low'] < spx['nr14_threshold']).astype(int)
    spx.drop(columns=['nr14_threshold'], inplace=True)
    
    # 4. Ratio de volatilidad en el VIX
    spx['VIX_pct_change'] = spx['VIX'].pct_change()
    
    # Eliminamos la columna de retornos logarítmicos que ya no es necesaria
    spx.drop(columns=['log_ret'], inplace=True)
    
    return spx.dropna()

# --- CARGAR DATOS ---
with st.spinner("Cargando datos..."):
    df_raw = fetch_data()
    spx = calculate_indicators(df_raw)

st.success(f"✅ Datos cargados: {len(spx)} días disponibles")

# --- CONTROLES DE FECHA ---
st.sidebar.header("⚙️ Configuración del Gráfico")

# Fecha final (última disponible)
fecha_final = spx.index[-1].date()
st.sidebar.info(f"📅 Última fecha disponible: {fecha_final}")

# Fecha de inicio (por defecto 3 meses atrás)
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

# ELIMINAR fines de semana del DataFrame
spx_filtered = spx_filtered[spx_filtered.index.dayofweek < 5]  # 0=Lunes, 4=Viernes

st.markdown(f"**Período seleccionado:** {fecha_inicio} hasta {fecha_final} ({len(spx_filtered)} días)")

# --- PREPARACIÓN DE DATOS PARA GRÁFICO COMBINADO ---

# Crear etiquetas de fecha inteligentes para el eje x
date_labels = []
prev_year = None
prev_month = None

for d in spx_filtered.index:
    if prev_year is None or d.year != prev_year:
        date_labels.append(d.strftime('%b %y'))
        prev_year = d.year
        prev_month = d.month
    elif d.month != prev_month:
        date_labels.append(d.strftime('%b %d'))
        prev_month = d.month
    elif len(date_labels) % 5 == 0:
        date_labels.append(d.strftime('%d'))
    else:
        date_labels.append('')

# Convertir RV_5d a porcentaje
spx_filtered['RV_5d_pct'] = spx_filtered['RV_5d'] * 100

# Lógica de Color para RV
UMBRAL_RV = 0.10
spx_filtered['RV_change'] = spx_filtered['RV_5d_pct'].diff()
is_up = spx_filtered['RV_change'] >= 0

is_down = ~is_up
rv_green_mask = is_up | is_up.shift(-1).fillna(False)
rv_red_mask = is_down | is_down.shift(-1).fillna(False)

rv_green_plot = spx_filtered['RV_5d_pct'].where(rv_green_mask, other=np.nan)
rv_red_plot = spx_filtered['RV_5d_pct'].where(rv_red_mask, other=np.nan)

# --- CREAR SUBPLOTS ---
# 2 filas, 1 columna. Compartir el eje x
fig_combined = make_subplots(
    rows=2, 
    cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.02,
    row_heights=[0.7, 0.3], # SPX toma 70%, RV toma 30% del alto
    subplot_titles=("S&P 500 - Velas Japonesas", "") # El segundo título queda vacío
)

# ----------------------------------------------------
# 1. GRÁFICO DE VELAS JAPONESAS (Fila 1)
# ----------------------------------------------------
fig_combined.add_trace(go.Candlestick(
    x=list(range(len(spx_filtered))),
    open=spx_filtered['Open'],
    high=spx_filtered['High'],
    low=spx_filtered['Low'],
    close=spx_filtered['Close'],
    name='SPX',
    showlegend=False
), row=1, col=1)

# Configuraciones de la Fila 1
fig_combined.update_yaxes(title_text='Precio', row=1, col=1)
fig_combined.update_xaxes(showticklabels=False, row=1, col=1) # Ocultar etiquetas X en el gráfico superior

# ----------------------------------------------------
# 2. GRÁFICO DE VOLATILIDAD REALIZADA (RV_5d) (Fila 2)
# ----------------------------------------------------

# Traza de LÍNEA VERDE (Subida)
fig_combined.add_trace(go.Scatter(
    x=list(range(len(spx_filtered))),
    y=rv_green_plot,
    mode='lines',
    name='Subida', 
    line=dict(color='green', width=2),
    hoverinfo='text',
    text=[f"RV: {y:.2f}% ({'Sube' if u else 'Baja'})" for y, u in zip(spx_filtered['RV_5d_pct'], is_up)],
    showlegend=False # Ocultar leyenda
), row=2, col=1)

# Traza de LÍNEA ROJA (Bajada)
fig_combined.add_trace(go.Scatter(
    x=list(range(len(spx_filtered))),
    y=rv_red_plot,
    mode='lines',
    name='Bajada', 
    line=dict(color='red', width=2),
    hoverinfo='text',
    text=[f"RV: {y:.2f}% ({'Sube' if u else 'Baja'})" for y, u in zip(spx_filtered['RV_5d_pct'], is_up)],
    showlegend=False # Ocultar leyenda
), row=2, col=1)

# Traza de PUNTOS (Markers) en BLANCO
fig_combined.add_trace(go.Scatter(
    x=list(range(len(spx_filtered))),
    y=spx_filtered['RV_5d_pct'],
    mode='markers',
    name='Puntos',
    marker=dict(color='white', size=6, line=dict(width=1, color='blue')),
    showlegend=False,
    hoverinfo='skip'
), row=2, col=1)

# Añadir línea horizontal discontinua del umbral (Fila 2)
fig_combined.add_shape(
    type="line",
    x0=0, y0=UMBRAL_RV * 100,
    x1=len(spx_filtered) - 1, y1=UMBRAL_RV * 100,
    line=dict(color="orange", width=2, dash="dot"),
    layer="below",
    row=2, col=1
)

# Añadir etiqueta para el umbral (Fila 2)
fig_combined.add_annotation(
    x=len(spx_filtered) - 1, # Último punto del eje X
    y=UMBRAL_RV * 100,
    text=f'Umbral: {UMBRAL_RV*100:.2f}%', 
    showarrow=False,
    xanchor='left',
    yanchor='bottom', 
    font=dict(size=12, color="orange"),
    xshift=5,
    row=2, col=1
)

# Configuraciones de la Fila 2
fig_combined.update_yaxes(title_text='RV (%)', row=2, col=1, tickformat=".2f")


# --- CONFIGURACIÓN FINAL DEL GRÁFICO COMBINADO ---
fig_combined.update_layout(
    title=f'S&P 500 y Volatilidad Realizada RV_5d ({fecha_inicio} a {fecha_final})',
    template='plotly_white',
    height=800, # Aumento la altura para los dos gráficos
    xaxis_rangeslider_visible=False,
    hovermode='x unified',
)

# Configurar el eje X compartido (solo las etiquetas inferiores)
fig_combined.update_xaxes(
    tickmode='array',
    tickvals=list(range(len(spx_filtered))),
    ticktext=date_labels,
    tickangle=-45,
    row=2, col=1
)

st.plotly_chart(fig_combined, use_container_width=True)

# --- INFORMACIÓN ADICIONAL ---
st.markdown("---")
col1, col2, col3, col4, col5 = st.columns(5) 

with col1:
    st.metric("Precio Actual", f"${spx_filtered['Close'].iloc[-1]:.2f}")
with col2:
    cambio = spx_filtered['Close'].iloc[-1] - spx_filtered['Close'].iloc[0]
    cambio_pct = (cambio / spx_filtered['Close'].iloc[0]) * 100
    st.metric("Cambio en el Período", f"${cambio:.2f}", f"{cambio_pct:.2f}%")
with col3:
    st.metric("Máximo", f"${spx_filtered['High'].max():.2f}")
with col4:
    st.metric("Mínimo", f"${spx_filtered['Low'].min():.2f}")
with col5:
    rv_latest = spx_filtered['RV_5d'].iloc[-1] * 100
    st.metric("RV_5d (Último)", f"{rv_latest:.2f}%")

