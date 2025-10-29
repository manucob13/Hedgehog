import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

# --- Configuración ---
st.set_page_config(page_title="SPX Simplificado", layout="wide")
st.title("📊 SPX y VIX Simplificado")

# --- Descarga de datos desde 2010 ---
@st.cache_data(ttl=86400)
def fetch_data():
    start = "2010-01-01"
    end = datetime.now().strftime("%Y-%m-%d")

    spx = yf.download("^GSPC", start=start, end=end, auto_adjust=False, multi_level_index=False)
    vix = yf.download("^VIX", start=start, end=end, auto_adjust=False, multi_level_index=False)

    spx.index = pd.to_datetime(spx.index)
    vix_series = vix['Close'].rename('VIX')
    vix_series.index = pd.to_datetime(vix_series.index)
    vix_series = vix_series.asfreq('B').ffill()

    df_merged = spx.merge(vix_series, how='left', left_index=True, right_index=True)
    df_merged.dropna(subset=['VIX'], inplace=True)
    return df_merged

df = fetch_data()
st.success(f"✅ Datos cargados desde {df.index.min().date()} hasta {df.index.max().date()}")

# --- Selección de rango para graficar ---
st.sidebar.header("Rango para graficar SPX")
min_date = df.index.min().date()
max_date = df.index.max().date()

start_plot = st.sidebar.date_input("Fecha inicio", min_value=min_date, max_value=max_date, value=min_date)
end_plot = st.sidebar.date_input("Fecha fin", min_value=start_plot, max_value=max_date, value=max_date)

# Filtrar DataFrame para el rango seleccionado
df_plot = df.loc[(df.index.date >= start_plot) & (df.index.date <= end_plot)]

# --- Gráfico SPX ---
st.header(f"Gráfico del S&P 500 ({start_plot} a {end_plot})")
if not df_plot.empty:
    st.line_chart(df_plot['Close'].rename('SPX Close'))
else:
    st.warning("No hay datos disponibles para el rango seleccionado.")
