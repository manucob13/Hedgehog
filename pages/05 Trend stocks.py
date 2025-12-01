# pages/Trend stocks.py
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import timedelta, datetime
from concurrent.futures import ThreadPoolExecutor
import os
import time
import requests
from typing import Optional, Tuple

# =========================================================================
# 0. CONFIGURACIÓN Y VARIABLES
# =========================================================================

st.set_page_config(page_title="Trend Stocks", layout="wide")

CONTRACT_SIZE = 100
CACHE_DIR = "data"
API_BASE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options"

# =========================================================================
# 1. PREPARACIÓN DE TICKERS (DESDE CSV)
# =========================================================================

@st.cache_resource(ttl=timedelta(hours=24), show_spinner=False)
def perform_initial_preparation():
    st.subheader("1. Preparación de Tickers")
    
    status_text = st.empty()
    
    # Leer tickers del archivo CSV sin encabezado
    csv_filename = 'Tickers.csv'
    if os.path.exists(csv_filename):
        try:
            df_tickers = pd.read_csv(csv_filename, header=None, names=['Ticker'])
            
            # Extraer tickers únicos
            tickers = df_tickers['Ticker'].astype(str).str.upper().str.strip().tolist()
            tickers = sorted(set(tickers))  # Eliminar duplicados y ordenar
            
            st.success(f"✅ '{csv_filename}' encontrado con {len(tickers)} tickers únicos.")
            
            # Mostrar información del dataset
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📊 Total Tickers", len(tickers))
            with col2:
                st.metric("📅 Columnas", 1)
            
            # Mostrar preview de datos
            with st.expander("👀 Vista previa del dataset"):
                st.dataframe(df_tickers.head(10), use_container_width=True)
            
            st.info("ℹ️ Los tickers se usan directamente sin validación adicional.")
            
            status_text.empty()
            st.divider()
            
            return tickers, df_tickers
            
        except Exception as e:
            st.error(f"❌ Error al leer '{csv_filename}': {e}")
            st.stop()
    else:
        st.error(f"❌ '{csv_filename}' no encontrado en el directorio raíz.")
        st.info(f"📝 Crea un archivo '{csv_filename}' con los tickers (sin encabezado)")
        st.stop()

# =========================================================================
# 2. FUNCIONES PARA OBTENER DATOS DE OPCIONES DESDE CBOE API
# =========================================================================

def ensure_cache_dir():
    """Crear directorio de caché si no existe"""
    os.makedirs(CACHE_DIR, exist_ok=True)

@st.cache_data(ttl=300)
def fetch_option_data(ticker: str) -> Optional[dict]:
    """Obtener datos de opciones desde CBOE API con caché"""
    ensure_cache_dir()
    
    urls = [
        f"{API_BASE_URL}/_{ticker}.json",
        f"{API_BASE_URL}/{ticker}.json"
    ]
    
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
        except:
            continue
    return None

def parse_option_data(raw_data: dict) -> Tuple[float, pd.DataFrame]:
    """Parsear datos crudos de opciones"""
    try:
        data = pd.DataFrame.from_dict(raw_data)
        spot_price = float(data.loc["current_price", "data"])
        option_data = pd.DataFrame(data.loc["options", "data"])
        return spot_price, option_data
    except Exception as e:
        return 0, pd.DataFrame()

def process_option_data_optimized(data: pd.DataFrame) -> pd.DataFrame:
    """Procesar y limpiar datos de opciones usando vectorización"""
    df = data.copy()
    
    df["type"] = df.option.str.extract(r'\d([CP])\d')
    df["strike_raw"] = df.option.str.extract(r'[CP](\d+)').astype(float)
    df["strike"] = df["strike_raw"] / 1000
    df["expiration_str"] = df.option.str.extract(r'[A-Z]+(\d{6})')
    df["expiration"] = pd.to_datetime(df["expiration_str"], format="%y%m%d", errors='coerce')
    
    numeric_cols = ['gamma', 'open_interest', 'volume', 'delta', 'vega', 'theta', 'iv']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    mask = (
        df['type'].notna() & 
        df['strike'].notna() & 
        df['expiration'].notna() & 
        df['gamma'].notna() & 
        df['open_interest'].notna() &
        (df['open_interest'] > 0) & 
        (df['gamma'] > 0)
    )
    
    return df[mask]

def obtener_datos_opciones_cboe(ticker):
    """
    Obtiene datos de opciones desde CBOE API:
    - Call/Put Ratio basado en open interest
    - Volumen total de opciones
    """
    try:
        raw_data = fetch_option_data(ticker)
        if not raw_data:
            return None, None
        
        spot_price, option_data = parse_option_data(raw_data)
        if option_data.empty:
            return None, None
        
        df = process_option_data_optimized(option_data)
        if df.empty:
            return None, None
        
        # Calcular volumen total y ratio
        calls = df[df['type'] == 'C']
        puts = df[df['type'] == 'P']
        
        total_call_volume = calls['volume'].fillna(0).sum()
        total_put_volume = puts['volume'].fillna(0).sum()
        total_call_oi = calls['open_interest'].fillna(0).sum()
        total_put_oi = puts['open_interest'].fillna(0).sum()
        
        options_volume = total_call_volume + total_put_volume
        
        # Calcular Call/Put Ratio basado en open interest (más estable que volumen)
        if total_put_oi > 0:
            call_put_ratio = total_call_oi / total_put_oi
        else:
            call_put_ratio = 0.0
        
        return options_volume, call_put_ratio
    
    except Exception:
        return None, None

def procesar_ticker_opciones(ticker):
    """Función helper para paralelizar obtención de datos de opciones"""
    options_volume, call_put_ratio = obtener_datos_opciones_cboe(ticker)
    
    return {
        'ticker': ticker,
        'options_volume': options_volume if options_volume else 0,
        'call_put_ratio': call_put_ratio if call_put_ratio else 0.0
    }

# =========================================================================
# 3. FILTROS Y PARÁMETROS
# =========================================================================

def seccion_filtros(df_original):
    """Sección de configuración de filtros"""
    st.subheader("2. Configuración de Filtros")
    
    with st.container():
        # Filtros de opciones
        st.markdown("#### 📊 Filtros de Opciones")
        col1, col2 = st.columns(2)
        
        with col1:
            volumen_min = st.number_input(
                "📊 Volumen mínimo de opciones",
                min_value=0,
                value=5000,
                step=1000,
                help="Volumen mínimo de contratos de opciones (calls + puts)"
            )
        
        with col2:
            ratio_min = st.number_input(
                "📈 Call/Put Ratio mínimo",
                min_value=0.0,
                value=0.5,
                step=0.1,
                format="%.2f",
                help="Ratio mínimo basado en Open Interest (>0.5 significa más calls que puts)"
            )
        
        st.markdown("---")
        
        # Información adicional
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("**Volumen > 5000**: Alta liquidez en opciones")
        with col2:
            st.info("**Ratio > 0.5**: Sesgo alcista (más calls)")
        with col3:
            st.info("**Ratio > 1.0**: Fuerte sesgo alcista")
    
    return {
        'volumen_min': volumen_min,
        'ratio_min': ratio_min
    }

# =========================================================================
# 4. ESCANEO DE OPCIONES (PARALELO CON CBOE API)
# =========================================================================

def ejecutar_escaneo_opciones(tickers, df_original, filtros):
    """Ejecuta el escaneo de opciones usando CBOE API EN PARALELO"""
    
    # Contenedores para mostrar progreso
    status_container = st.empty()
    progress_bar = st.progress(0)
    
    # Obtener datos de opciones (PARALELO)
    status_container.info(f"📊 Obteniendo datos de opciones para {len(tickers)} tickers desde CBOE API (paralelo)...")
    
    resultados = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(procesar_ticker_opciones, ticker) for ticker in tickers]
        for i, future in enumerate(futures):
            try:
                result = future.result()
                if result and result['options_volume'] > 0:
                    resultados.append(result)
            except Exception:
                pass
            progress_bar.progress((i + 1) / len(futures))
    
    if not resultados:
        status_container.error("❌ No se encontraron tickers con datos de opciones válidos")
        progress_bar.empty()
        return None
    
    df = pd.DataFrame(resultados)
    df.columns = ['Ticker', 'Options_Volume', 'Call_Put_Ratio']
    
    status_container.success(f"✅ Datos obtenidos: {len(df)} tickers con información de opciones")
    
    # Aplicar filtros de opciones
    df_filtrado = df[
        (df['Options_Volume'] > filtros['volumen_min']) & 
        (df['Call_Put_Ratio'] > filtros['ratio_min'])
    ].copy()
    
    if df_filtrado.empty:
        status_container.warning("⚠️ No hay tickers que cumplan los criterios de filtrado")
        progress_bar.empty()
        return None
    
    # Ordenar por volumen de opciones descendente
    df_filtrado = df_filtrado.sort_values('Options_Volume', ascending=False)
    df_filtrado.insert(0, 'Rank', range(1, len(df_filtrado) + 1))
    
    progress_bar.empty()
    status_container.success(f"🎉 Escaneo completado: {len(df_filtrado)} acciones encontradas")
    
    return df_filtrado

# =========================================================================
# 5. VISUALIZACIÓN DE RESULTADOS
# =========================================================================

def mostrar_resultados(df_resultados):
    """Muestra los resultados filtrados en tabla interactiva"""
    st.subheader("3. Resultados del Escaneo")
    
    if df_resultados is None or df_resultados.empty:
        st.warning("⚠️ No hay acciones que cumplan los criterios de filtrado")
        st.info("💡 Intenta reducir los valores mínimos de los filtros")
        return
    
    # Métricas de resumen (ANTES de formatear)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Acciones Filtradas", len(df_resultados))
    with col2:
        avg_ratio = df_resultados['Call_Put_Ratio'].mean()
        st.metric("📊 Ratio Promedio", f"{avg_ratio:.2f}")
    with col3:
        ratio_gt_1 = len(df_resultados[df_resultados['Call_Put_Ratio'] > 1.0])
        st.metric("🚀 Ratio > 1.0", ratio_gt_1)
    with col4:
        total_vol = df_resultados['Options_Volume'].sum()
        st.metric("📈 Vol. Total", f"{total_vol:,.0f}")
    
    st.markdown("---")
    
    # Seleccionar columnas clave para mostrar
    columnas_mostrar = ['Rank', 'Ticker', 'Options_Volume', 'Call_Put_Ratio']
    
    df_table = df_resultados[columnas_mostrar].copy()
    
    # Formatear Call_Put_Ratio para display
    df_table['Call_Put_Ratio'] = df_table['Call_Put_Ratio'].apply(lambda x: f"{x:.2f}")
    
    # Renombrar columnas para mejor presentación
    column_config = {
        "Rank": st.column_config.NumberColumn("🏅 Rank", width="small"),
        "Ticker": st.column_config.TextColumn("🎯 Ticker", width="small"),
        "Options_Volume": st.column_config.NumberColumn("📈 Vol. Opciones", width="medium", format="%d"),
        "Call_Put_Ratio": st.column_config.TextColumn("🔥 C/P Ratio", width="small"),
    }
    
    # Tabla de resultados
    st.markdown("#### 🏆 Acciones con Mayor Actividad en Opciones")
    st.dataframe(
        df_table,
        hide_index=True,
        use_container_width=True,
        column_config=column_config
    )
    
    # Gráficos de distribución
    st.markdown("---")
    st.markdown("#### 📊 Análisis Visual")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 10 por ratio
        top_10_ratio = df_resultados.nlargest(10, 'Call_Put_Ratio')
        st.bar_chart(
            top_10_ratio.set_index('Ticker')['Call_Put_Ratio'],
            use_container_width=True
        )
        st.caption("Top 10 acciones por Call/Put Ratio")
    
    with col2:
        # Top 10 por volumen
        top_10_vol = df_resultados.nlargest(10, 'Options_Volume')
        st.bar_chart(
            top_10_vol.set_index('Ticker')['Options_Volume'],
            use_container_width=True
        )
        st.caption("Top 10 acciones por Volumen de Opciones")
    
    # Botón de descarga
    st.markdown("---")
    df_csv = df_resultados.copy()
    df_csv['Call_Put_Ratio'] = df_csv['Call_Put_Ratio'].apply(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
    
    csv = df_csv.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar Resultados Completos (CSV)",
        data=csv,
        file_name=f"options_scanner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=False
    )

# =========================================================================
# 6. FUNCIÓN PRINCIPAL
# =========================================================================

def options_scanner_page():
    st.title("📊 Trend Stocks Scanner - Call/Put Ratio Analyzer")
    st.markdown("---")
    st.info("🔍 Este escáner identifica acciones con alta actividad en opciones y sesgo alcista usando CBOE API")
    
    # --- Punto 1: Preparación de Tickers ---
    col1, col2 = st.columns([1, 4])
    with col1:
        st.button("🔄 Recargar Datos", type="primary",
                  help="Borra la caché y recarga el archivo CSV",
                  on_click=perform_initial_preparation.clear)
    with col2:
        st.markdown("_(Los datos se cargan desde Tickers.csv sin encabezado.)_")
    
    st.divider()
    valid_tickers, df_original = perform_initial_preparation()
    
    # --- Punto 2: Filtros ---
    st.divider()
    filtros = seccion_filtros(df_original)
    
    # --- Punto 3: Escaneo ---
    st.divider()
    st.subheader("3. Escaneo de Opciones")
    
    st.info(f"📊 Tickers listos para escanear: **{len(valid_tickers)}** | 🚀 Modo: **Paralelo (10 hilos)**")
    st.warning("⚠️ El escaneo tardará 3-5 minutos. **No cambies de página durante el proceso.**")
    st.info("📊 **Fuente de datos**: CBOE API (Call/Put Ratio basado en Open Interest)")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        ejecutar_btn = st.button("🚀 Ejecutar Escaneo", type="primary", use_container_width=True)
    with col2:
        if 'df_resultados_options' in st.session_state and st.session_state.df_resultados_options is not None:
            st.success(f"✅ Último escaneo: {len(st.session_state.df_resultados_options)} resultados")
    with col3:
        if st.button("🗑️ Limpiar", use_container_width=True):
            if 'df_resultados_options' in st.session_state:
                del st.session_state.df_resultados_options
            st.rerun()
    
    if ejecutar_btn:
        start_time = time.time()
        with st.spinner("Ejecutando escaneo paralelo con CBOE API..."):
            try:
                df_resultados = ejecutar_escaneo_opciones(
                    valid_tickers,
                    df_original,
                    filtros
                )
                st.session_state.df_resultados_options = df_resultados
                
                elapsed_time = time.time() - start_time
                
                if df_resultados is not None and not df_resultados.empty:
                    st.balloons()
                    st.success(f"🎉 Escaneo completado en {elapsed_time:.1f} segundos - {len(df_resultados)} acciones encontradas")
                else:
                    st.warning("⚠️ No se encontraron acciones que cumplan los criterios")
            
            except Exception as e:
                st.error(f"❌ Error durante el escaneo: {str(e)}")
    
    # --- Punto 4: Resultados ---
    st.divider()
    if 'df_resultados_options' in st.session_state:
        mostrar_resultados(st.session_state.df_resultados_options)
    else:
        st.info("👆 Ejecuta el escaneo primero para ver los resultados aquí")
    
    # --- Estado final ---
    st.divider()
    st.success(f"🎯 Sistema listo con {len(valid_tickers)} tickers válidos usando CBOE API.")

# =========================================================================
# 7. PUNTO DE ENTRADA
# =========================================================================

if __name__ == "__main__":
    options_scanner_page()
