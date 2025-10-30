import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Gráficos - HEDGEHOG", layout="wide")
st.title("📊 Gráficos de Análisis Técnico")

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

# --- GRÁFICO DE VELAS JAPONESAS ---
st.subheader("📈 S&P 500 - Velas Japonesas")

# Crear etiquetas de fecha inteligentes para el eje x del Candlestick
date_labels = []
prev_year = None
prev_month = None

for d in spx_filtered.index:
    # Si cambia el año, mostrar año
    if prev_year is None or d.year != prev_year:
        date_labels.append(d.strftime('%b %y'))
        prev_year = d.year
        prev_month = d.month
    # Si cambia el mes, mostrar mes y día
    elif d.month != prev_month:
        date_labels.append(d.strftime('%b %d'))
        prev_month = d.month
    # Mismo mes, solo mostrar el día cada 5 días aprox
    elif len(date_labels) % 5 == 0:
        date_labels.append(d.strftime('%d'))
    else:
        date_labels.append('')  # Sin etiqueta para no saturar

fig = go.Figure(data=[go.Candlestick(
    x=list(range(len(spx_filtered))),
    open=spx_filtered['Open'],
    high=spx_filtered['High'],
    low=spx_filtered['Low'],
    close=spx_filtered['Close'],
    name='SPX'
)])

fig.update_layout(
    title=f'S&P 500 - Velas Japonesas ({fecha_inicio} a {fecha_final})',
    yaxis_title='Precio',
    xaxis_title='',
    template='plotly_white',
    height=600,
    xaxis_rangeslider_visible=False,
    xaxis=dict(
        tickmode='array',
        tickvals=list(range(len(spx_filtered))),
        ticktext=date_labels,
        tickangle=-45
    ),
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

# --- GRÁFICO DE VOLATILIDAD REALIZADA (RV_5d) ---
st.subheader("📉 Volatilidad Realizada (RV\_5d)")

fig_rv = go.Figure(data=[
    go.Scatter(
        x=spx_filtered.index,
        y=spx_filtered['RV_5d'] * 100, # Multiplicar por 100 para mostrar en %
        mode='lines',
        name='RV 5 días (Anualizada)',
        line=dict(color='#E57373', width=2) # Color rojo/suave
    )
])

# Añadir línea de referencia del promedio para el período
rv_avg = spx_filtered['RV_5d'].mean() * 100

fig_rv.add_shape(
    type="line",
    x0=spx_filtered.index[0], y0=rv_avg,
    x1=spx_filtered.index[-1], y1=rv_avg,
    line=dict(color="gray", width=1, dash="dash"),
    # Añadir anotación para el promedio
    label=dict(
        text=f'Promedio: {rv_avg:.2f}%', 
        textangle=0,
        x=spx_filtered.index[-1], 
        y=rv_avg,
        xanchor='right',
        yanchor='bottom',
        font=dict(size=10, color="gray")
    )
)

fig_rv.update_layout(
    title='Volatilidad Realizada a 5 Días del S&P 500 (Anualizada)',
    yaxis_title='RV (%)',
    xaxis_title='Fecha',
    template='plotly_white',
    height=400,
    hovermode='x unified',
)

fig_rv.update_yaxes(tickformat=".2f") # Formato con 2 decimales
st.plotly_chart(fig_rv, use_container_width=True)

# --- INFORMACIÓN ADICIONAL ---
st.markdown("---")
# Hemos añadido una quinta columna para la Volatilidad Realizada
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
