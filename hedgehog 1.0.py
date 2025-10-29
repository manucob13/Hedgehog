import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

# --- Configuración de la app ---
st.set_page_config(page_title="SPX y VIX Simplificado", layout="wide")
st.title("📊 SPX y VIX Simplificado")

# --- Descarga de datos históricos ---
@st.cache_data(ttl=86400)  # caché de 1 día
def fetch_data():
    start = "2010-01-01"
    end = datetime.now().strftime("%Y-%m-%d")

    # Descarga SPX y VIX
    spx = yf.download("^GSPC", start=start, end=end, auto_adjust=False, multi_level_index=False)
    vix = yf.download("^VIX", start=start, end=end, auto_adjust=False, multi_level_index=False)

    # Procesamiento
    spx.index = pd.to_datetime(spx.index)
    vix_series = vix['Close'].rename('VIX')
    vix_series.index = pd.to_datetime(vix_series.index)
    vix_series = vix_series.asfreq('B').ffill()  # rellenar días hábiles faltantes

    # Fusionar SPX con VIX
    df_merged = spx.merge(vix_series, how='left', left_index=True, right_index=True)
    df_merged.dropna(subset=['VIX'], inplace=True)

    return df_merged

# Cargar datos
df = fetch_data()
st.success(f"✅ Datos cargados desde {df.index.min().date()} hasta {df.index.max().date()}")

# --- Selección de rango para graficar SPX ---
st.sidebar.header("Selecciona rango para graficar SPX")
min_date = df.index.min().date()
max_date = df.index.max().date()

start_plot = st.sidebar.date_input("Fecha inicio", min_value=min_date, max_value=max_date, value=min_date)
end_plot = st.sidebar.date_input("Fecha fin", min_value=start_plot, max_value=max_date, value=max_date)

# --- Filtrar DataFrame según rango ---
df_plot = df.loc[(df.index.date >= start_plot) & (df.index.date <= end_plot)]

# --- Gráfico SPX con velas japonesas ---
st.header(f"Gráfico de velas japonesas del S&P 500 ({start_plot} a {end_plot})")

if not df_plot.empty:
    fig = go.Figure(data=[go.Candlestick(
        x=df_plot.index,
        open=df_plot['Open'],
        high=df_plot['High'],
        low=df_plot['Low'],
        close=df_plot['Close'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])

    fig.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Precio",
        xaxis_rangeslider_visible=False,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No hay datos disponibles para el rango seleccionado.")
