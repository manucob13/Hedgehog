import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

# --- Configuración de la Aplicación ---
st.set_page_config(
    page_title="Hedgehog version 1.0 31-October-2025",
    layout="wide"
)

st.title("📊 SPX y VIX Simplificado")

# --- Descarga de datos desde 2010 hasta hoy ---
@st.cache_data(ttl=86400)  # Se guarda en caché por 1 día
def fetch_data():
    start = "2010-01-01"
    end = datetime.now().strftime("%Y-%m-%d")

    spx = yf.download("^GSPC", start=start, end=end, auto_adjust=False, multi_level_index=False)
    vix = yf.download("^VIX", start=start, end=end, auto_adjust=False, multi_level_index=False)

    # Convertimos a datetime y preparamos VIX
    spx.index = pd.to_datetime(spx.index)
    vix_series = vix['Close'].rename('VIX')
    vix_series.index = pd.to_datetime(vix_series.index)
    vix_series = vix_series.asfreq('B').ffill()  # Rellenamos días hábiles faltantes

    # Fusionamos SPX con VIX
    df_merged = spx.merge(vix_series, how='left', left_index=True, right_index=True)
    df_merged.dropna(subset=['VIX'], inplace=True)
    
    return df_merged

# Cargamos los datos
df = fetch_data()
st.success(f"✅ Datos históricos cargados desde {df.index.min().date()} hasta {df.index.max().date()}")

# --- Selección de rango para graficar SPX ---
st.sidebar.header("Selecciona rango para graficar SPX")
min_date = df.index.min().date()
max_date = df.index.max().date()

start_plot = st.sidebar.date_input("Fecha Inicio", min_value=min_date, max_value=max_date, value=min_date)
end_plot = st.sidebar.date_input("Fecha Fin", min_value=start_plot, max_value=max_date, value=max_date)

# Filtramos el DataFrame según selección
df_plot = df.loc[start_plot:end_plot]

# --- Gráfico SPX ---
st.header(f"Gráfico del S&P 500 ({start_plot} a {end_plot})")
st.line_chart(df_plot['Close'].rename('SPX Close'))
st.caption("Evolución del precio de cierre del S&P 500")
