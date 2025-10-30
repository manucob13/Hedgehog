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

st.markdown(f"**Período seleccionado:** {fecha_inicio} hasta {fecha_final} ({len(spx_filtered)} días)")

# --- GRÁFICO DE VELAS JAPONESAS ---
st.subheader("📈 S&P 500 - Velas Japonesas")

fig = go.Figure(data=[go.Candlestick(
    x=spx_filtered.index,
    open=spx_filtered['Open'],
    high=spx_filtered['High'],
    low=spx_filtered['Low'],
    close=spx_filtered['Close'],
    name='SPX'
)])

fig.update_layout(
    title=f'S&P 500 - Velas Japonesas ({fecha_inicio} a {fecha_final})',
    yaxis_title='',  # Sin título en eje Y
    xaxis_title='',  # Sin título en eje X
    template='plotly_white',
    height=600,
    xaxis_rangeslider_visible=False,
    # IMPORTANTE: Esto elimina los espacios de fines de semana
    xaxis=dict(
        rangebreaks=[
            dict(bounds=["sat", "sun"])  # Oculta sábados y domingos
        ],
        tickformat='%b %d',  # Solo Mes y Día (ej: Jul 31, Aug 15)
    ),
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

# --- INFORMACIÓN ADICIONAL ---
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

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
