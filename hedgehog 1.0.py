import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --- Configuración de la app ---
st.set_page_config(page_title="HEDGEHOG", layout="wide")
st.title("📊 HEDGEHOG 1.0")

# --- Descarga de datos históricos ---
@st.cache_data(ttl=86400)
def fetch_data():
    start = "2010-01-01"
    end = datetime.now()

    # Descarga SPX y VIX
    spx = yf.download("^GSPC", start=start, end=end, auto_adjust=False, multi_level_index=False)
    vix = yf.download("^VIX", start=start, end=end, auto_adjust=False, multi_level_index=False)

    # Convertir índices a datetime
    spx.index = pd.to_datetime(spx.index)
    vix_series = vix['Close'].rename('VIX')
    vix_series.index = pd.to_datetime(vix_series.index)

    # Fusionar SPX con VIX
    df_merged = spx.merge(vix_series, how='left', left_index=True, right_index=True)
    df_merged.dropna(subset=['VIX'], inplace=True)

    return df_merged

# Cargar datos
df = fetch_data()
st.success(f"✅ Descarga datos SPX - VIX desde {df.index.min().date()} - {df.index.max().date()}")

# --- Selección de rango para graficar SPX ---
st.sidebar.header("Rango grafico -  SPX")
min_date = df.index.min().date()
max_date = df.index.max().date()

# Valores por defecto: últimos 3 meses
default_end = max_date
default_start = (pd.Timestamp(default_end) - pd.DateOffset(months=3)).date()

# Selector de fechas
start_plot = st.sidebar.date_input("Fecha inicio", min_value=min_date, max_value=max_date, value=default_start)
end_plot = st.sidebar.date_input("Fecha fin", min_value=start_plot, max_value=max_date, value=default_end)

# --- Filtrar DataFrame según rango ---
df_plot = df.loc[(df.index.date >= start_plot) & (df.index.date <= end_plot)]

# --- Gráfico SPX con velas japonesas (sin huecos de fines de semana) ---
if not df_plot.empty:
    # Convertir índice a string para eje categórico
    df_plot_plotly = df_plot.copy()
    df_plot_plotly['Fecha_str'] = df_plot_plotly.index.strftime('%Y-%m-%d')

    fig = go.Figure(data=[go.Candlestick(
        x=df_plot_plotly['Fecha_str'],
        open=df_plot_plotly['Open'],
        high=df_plot_plotly['High'],
        low=df_plot_plotly['Low'],
        close=df_plot_plotly['Close'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])

    fig.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Precio",
        xaxis_rangeslider_visible=False,
        xaxis_type='category',  # elimina huecos de fines de semana
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No hay datos disponibles para el rango seleccionado.")
