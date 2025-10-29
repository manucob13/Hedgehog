import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests

# --- Configuración de la Aplicación ---
st.set_page_config(
    page_title="Datos Históricos del SPX y VIX",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Títulos y descripción
st.title("📊 Análisis de Índices Históricos (SPX y VIX)")
st.markdown(
    """
    Esta aplicación descarga datos históricos del **S&P 500 (`^GSPC`)** y el **VIX (`^VIX`)** utilizando la librería `yfinance`. 
    Selecciona el rango de fechas en la barra lateral izquierda.
    """
)

# --- Controles de Fecha en la Barra Lateral ---
st.sidebar.header("Selección de Rango de Fechas")
today = datetime.now().date()
default_start = datetime(2010, 1, 1).date()

start_date = st.sidebar.date_input(
    "Fecha de Inicio", 
    value=default_start, 
    min_value=datetime(1990, 1, 1).date(), 
    max_value=today - timedelta(days=1)
)

end_date = st.sidebar.date_input(
    "Fecha de Fin", 
    value=today - timedelta(days=1), 
    min_value=start_date + timedelta(days=1), 
    max_value=today
)

# --- Función de Descarga de Datos (con Caché) ---

@st.cache_data(ttl=3600) # Los datos se guardan en caché por 1 hora
def fetch_raw_data(start, end):
    """
    Descarga los datos brutos del SPX y VIX por separado.
    
    Retorna una tupla: (DataFrame solo SPX, Series solo VIX procesada).
    """
    
    try:
        # Crea una sesión con simulación de navegador para evitar problemas de yfinance
              
        # 1. Descarga del S&P 500 (^GSPC). ESTE SERÁ EL DATAFRAME BASE.
        st.info(f"Descargando datos del ^GSPC desde {start} hasta {end}...")
        spx_data = yf.download(
            "^GSPC", 
            start=start, 
            end=end, 
            auto_adjust=False, 
            multi_level_index=False, 
            session=session
        )
        spx_data.index = pd.to_datetime(spx_data.index) # Asegurar índice de fecha y hora
        
        # 2. Descarga del VIX (^VIX)
        st.info("Descargando datos del ^VIX...")
        vol_data = yf.download(
            "^VIX", 
            start=start, 
            end=end, 
            auto_adjust=False, 
            multi_level_index=False, 
            session=session
        )
        
        # 3. Procesamiento y renombrado del VIX (se mantiene como una Serie separada)
        vix_series = vol_data['Close'].rename('VIX')
        vix_series.index = pd.to_datetime(vix_series.index)
        vix_series = vix_series.asfreq('B').ffill() # Frecuencia 'B' y rellenar faltantes
        
        # Retorna los componentes sin fusionar
        return spx_data, vix_series
    
    except Exception as e:
        st.error(f"Ocurrió un error al descargar los datos: {e}")
        return pd.DataFrame(), pd.Series(dtype='float64') # Retorna objetos vacíos en caso de error

# --- Lógica de la Aplicación Principal ---

# Asegurar que la fecha de inicio no es posterior a la de fin
if start_date >= end_date:
    st.error("⚠️ La fecha de inicio debe ser anterior a la fecha de fin.")
else:
    # Mostrar indicador de carga mientras se descargan los datos
    with st.spinner("Cargando y procesando datos financieros..."):
        # Descarga de datos sin fusionar
        spx, vix_series = fetch_raw_data(start_date, end_date)

    if not spx.empty:
        st.success("✅ Datos cargados y procesados con éxito.")
        
        # ---------------------------------------------
        # 1. GRAFICAR SPX (Usando el DataFrame spx original)
        # ---------------------------------------------
        
        st.header("Gráfico del S&P 500")
        st.line_chart(spx['Close'].rename('SPX Close'))
        st.caption("Evolución del precio de cierre del índice S&P 500.")
        
        st.markdown("---")
        
        # ---------------------------------------------
        # 2. AÑADIR VIX AL SPX (Merge según el código original)
        # ---------------------------------------------
        
        if not vix_series.empty:
            # Creamos el DataFrame fusionado (df_merged)
            df_merged = spx.merge(vix_series, how='left', left_index=True, right_index=True)
            
            # Limpieza final: eliminar filas donde VIX es NaN
            df_merged.dropna(subset=['VIX'], inplace=True)
            
            # ---------------------------------------------
            # 3. VISUALIZACIÓN DE TABLA Y GRÁFICO VIX
            # ---------------------------------------------
            
            st.header("Tabla de Datos Históricos (SPX y VIX Fusionados)")
            st.dataframe(df_merged.tail(20), use_container_width=True)
            st.caption(f"Mostrando las últimas 20 filas. Total de filas: {len(df_merged)}")
            
            st.markdown("---")
            
            # Gráfico del VIX (en una columna para el ejemplo)
            st.subheader("Índice de Volatilidad VIX")
            st.line_chart(df_merged['VIX'])
            st.caption("Evolución del VIX (Medida de la expectativa de volatilidad del mercado).")
            
        else:
            st.warning("No se pudieron cargar los datos del VIX. Solo se muestra el SPX.")
            
    else:
        st.warning("No se pudieron cargar los datos. Por favor, verifica el rango de fechas e inténtalo de nuevo.")
