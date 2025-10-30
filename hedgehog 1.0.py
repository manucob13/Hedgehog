import yfinance as yf
import pandas as pd
from datetime import datetime

# --- Configuración de la app ---
st.set_page_config(page_title="HEDGEHOG", layout="wide")
st.title("📊 HEDGEHOG 1.0")
@st.cache_data(ttl=86400)

# --- Descarga de datos históricos (Cacheado) ---
def fetch_data():
    """Descarga datos históricos del ^GSPC (SPX) y ^VIX (VIX)."""
    start = "2010-01-01"
    end = datetime.now()

    # Descarga SPX y VIX
    # progress=False se usa para evitar mucha salida en la consola durante la descarga.
    spx = yf.download("^GSPC", start=start, end=end, auto_adjust=False, multi_level_index=False, progress=False)
    vix = yf.download("^VIX", start=start, end=end, auto_adjust=False, multi_level_index=False, progress=False)

    # Preparación y fusión de datos
    spx.index = pd.to_datetime(spx.index)
    vix_series = vix['Close'].rename('VIX')
    vix_series.index = pd.to_datetime(vix_series.index)

    df_merged = spx.merge(vix_series, how='left', left_index=True, right_index=True)
    df_merged.dropna(subset=['VIX'], inplace=True)
    
    # El flush=True asegura que este mensaje se imprime inmediatamente
    print(f"DEBUG: Datos descargados desde {df_merged.index.min().date()} hasta {df_merged.index.max().date()}", flush=True)

    return df_merged

# --- EJECUCIÓN DE PRUEBA (SOLO DESCARGA) ---

print("--- INICIANDO PRUEBA DE MÓDULO DE ANÁLISIS (PASO 1: DESCARGA) ---", flush=True)
    
# 1. Cargar datos base
df_raw = fetch_data()

# 2. Mostrar un resumen de los datos descargados
print("\n--- DATOS RAW DESCARGADOS (ÚLTIMAS 5 FILAS) ---")
# to_string() es mejor que un simple print para asegurar el formato en la consola
print(df_raw.tail(5).to_string())
print(f"\nDataFrame cargado con éxito. Filas totales: {len(df_raw)}")
        
    
