# pages/Trend stocks.py
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import timedelta, datetime
from concurrent.futures import ThreadPoolExecutor
import os
import time
from utils import check_password

# =========================================================================
# 0. CONFIGURACIÓN Y VARIABLES
# =========================================================================

st.set_page_config(page_title="Trend Stocks", layout="wide")

# =========================================================================
# 1. PREPARACIÓN DE TICKERS (DESDE CSV CON MÚLTIPLES COLUMNAS)
# =========================================================================

@st.cache_resource(ttl=timedelta(hours=24), show_spinner=False)
def perform_initial_preparation():
    st.subheader("1. Preparación de Tickers")
    
    status_text = st.empty()
    
    # Leer tickers del archivo CSV
    csv_filename = 'Explorer.csv'
    if os.path.exists(csv_filename):
        try:
            df_tickers = pd.read_csv(csv_filename)
            
            # Verificar que existe la columna 'Ticker'
            if 'Ticker' not in df_tickers.columns:
                st.error(f"❌ El archivo '{csv_filename}' no tiene una columna 'Ticker'")
                st.stop()
            
            # Extraer tickers únicos
            tickers = df_tickers['Ticker'].astype(str).str.upper().str.strip().tolist()
            tickers = sorted(set(tickers))  # Eliminar duplicados y ordenar
            
            st.success(f"✅ '{csv_filename}' encontrado con {len(tickers)} tickers únicos.")
            
            # Mostrar información del dataset
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Tickers", len(tickers))
            with col2:
                st.metric("📅 Columnas", len(df_tickers.columns))
            with col3:
                if 'Sector' in df_tickers.columns:
                    sectores = df_tickers['Sector'].nunique()
                    st.metric("🏢 Sectores", sectores)
            
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
        st.info(f"📝 Crea un archivo '{csv_filename}' con la columna 'Ticker' y otros datos")
        st.stop()

# =========================================================================
# 2. FUNCIONES PARA OBTENER DATOS DE OPCIONES DESDE YAHOO FINANCE
# =========================================================================

def obtener_datos_opciones_yfinance(ticker):
    """
    Obtiene datos de opciones desde Yahoo Finance:
    - Call/Put Ratio basado en open interest
    - Volumen total de opciones
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Obtener fechas de expiración disponibles
        exp_dates = stock.options
        
        if not exp_dates:
            return None, None
        
        total_call_volume = 0
        total_put_volume = 0
        total_call_oi = 0
        total_put_oi = 0
        
        # Iterar sobre todas las fechas de expiración (limitamos a las primeras 3 para velocidad)
        for exp_date in exp_dates[:3]:
            try:
                # Obtener cadena de opciones para esta fecha
                opt_chain = stock.option_chain(exp_date)
                
                calls = opt_chain.calls
                puts = opt_chain.puts
                
                # Sumar volúmenes y open interest
                if not calls.empty:
                    total_call_volume += calls['volume'].fillna(0).sum()
                    total_call_oi += calls['openInterest'].fillna(0).sum()
                
                if not puts.empty:
                    total_put_volume += puts['volume'].fillna(0).sum()
                    total_put_oi += puts['openInterest'].fillna(0).sum()
                    
            except Exception:
                continue
        
        # Calcular Call/Put Ratio basado en open interest (más estable que volumen)
        if total_put_oi > 0:
            call_put_ratio = total_call_oi / total_put_oi
        else:
            call_put_ratio = 0.0
        
        # Volumen total de opciones
        options_volume = total_call_volume + total_put_volume
        
        return options_volume, call_put_ratio
    
    except Exception as e:
        return None, None

def procesar_ticker_opciones(ticker):
    """Función helper para paralelizar obtención de datos de opciones"""
    options_volume, call_put_ratio = obtener_datos_opciones_yfinance(ticker)
    
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
        
        # Filtros adicionales del CSV original (si aplican)
        st.markdown("#### 🔍 Filtros Adicionales del Dataset")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filtrar_sector = st.checkbox("Filtrar por Sector", value=False)
            if filtrar_sector and 'Sector' in df_original.columns:
                sectores_disponibles = df_original['Sector'].unique().tolist()
                sectores_seleccionados = st.multiselect(
                    "Selecciona sectores",
                    options=sectores_disponibles,
                    default=sectores_disponibles
                )
            else:
                sectores_seleccionados = None
        
        with col2:
            filtrar_rsi = st.checkbox("Filtrar por RSI", value=False)
            if filtrar_rsi and 'RSI' in df_original.columns:
                rsi_min = st.slider("RSI mínimo", 0, 100, 40)
                rsi_max = st.slider("RSI máximo", 0, 100, 70)
            else:
                rsi_min, rsi_max = None, None
        
        with col3:
            filtrar_atlas = st.checkbox("Filtrar por Atlas", value=False)
            if filtrar_atlas and 'Atlas' in df_original.columns:
                atlas_value = st.selectbox("Valor Atlas", [0.0, 1.0], index=1)
            else:
                atlas_value = None
        
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
        'ratio_min': ratio_min,
        'sectores': sectores_seleccionados,
        'rsi_min': rsi_min,
        'rsi_max': rsi_max,
        'atlas': atlas_value
    }

# =========================================================================
# 4. ESCANEO DE OPCIONES (PARALELO CON YAHOO FINANCE)
# =========================================================================

def ejecutar_escaneo_opciones(tickers, df_original, filtros):
    """Ejecuta el escaneo de opciones usando Yahoo Finance EN PARALELO"""
    
    # Contenedores para mostrar progreso
    status_container = st.empty()
    progress_bar = st.progress(0)
    
    # Aplicar filtros del dataset original primero
    tickers_filtrados = tickers.copy()
    
    if filtros['sectores'] and 'Sector' in df_original.columns:
        df_temp = df_original[df_original['Sector'].isin(filtros['sectores'])]
        tickers_filtrados = df_temp['Ticker'].tolist()
        status_container.info(f"🔍 Filtro de sector aplicado: {len(tickers_filtrados)} tickers")
    
    if filtros['rsi_min'] is not None and 'RSI' in df_original.columns:
        df_temp = df_original[
            (df_original['RSI'] >= filtros['rsi_min']) & 
            (df_original['RSI'] <= filtros['rsi_max'])
        ]
        tickers_filtrados = [t for t in tickers_filtrados if t in df_temp['Ticker'].tolist()]
        status_container.info(f"🔍 Filtro RSI aplicado: {len(tickers_filtrados)} tickers")
    
    if filtros['atlas'] is not None and 'Atlas' in df_original.columns:
        df_temp = df_original[df_original['Atlas'] == filtros['atlas']]
        tickers_filtrados = [t for t in tickers_filtrados if t in df_temp['Ticker'].tolist()]
        status_container.info(f"🔍 Filtro Atlas aplicado: {len(tickers_filtrados)} tickers")
    
    # Obtener datos de opciones (PARALELO)
    status_container.info(f"📊 Obteniendo datos de opciones para {len(tickers_filtrados)} tickers desde Yahoo Finance (paralelo)...")
    
    resultados = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(procesar_ticker_opciones, ticker) for ticker in tickers_filtrados]
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
    
    # Merge con datos originales para tener toda la información
    df_filtrado = df_filtrado.merge(df_original, on='Ticker', how='left')
    
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
    
    # Añadir columnas adicionales si existen
    columnas_opcionales = ['Sector', 'Price', 'RSI', 'Atlas', 'Sharpe_ratio', 'ROC18', 'RSC']
    for col in columnas_opcionales:
        if col in df_resultados.columns:
            columnas_mostrar.append(col)
    
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
    
    # Añadir configs para columnas adicionales
    if 'Sector' in df_table.columns:
        column_config['Sector'] = st.column_config.TextColumn("🏢 Sector", width="medium")
    if 'Price' in df_table.columns:
        column_config['Price'] = st.column_config.NumberColumn("💲 Precio", width="small", format="%.2f")
    if 'RSI' in df_table.columns:
        column_config['RSI'] = st.column_config.NumberColumn("📊 RSI", width="small", format="%.1f")
    if 'Atlas' in df_table.columns:
        column_config['Atlas'] = st.column_config.NumberColumn("🗺️ Atlas", width="small", format="%.2f")
    
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
    
    # Distribución por sector si existe
    if 'Sector' in df_resultados.columns:
        st.markdown("---")
        st.markdown("#### 🏢 Distribución por Sector")
        sector_counts = df_resultados['Sector'].value_counts()
        st.bar_chart(sector_counts, use_container_width=True)
    
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
    st.info("🔍 Este escáner identifica acciones con alta actividad en opciones y sesgo alcista usando Yahoo Finance")
    
    # --- Punto 1: Preparación de Tickers ---
    col1, col2 = st.columns([1, 4])
    with col1:
        st.button("🔄 Recargar Datos", type="primary",
                  help="Borra la caché y recarga el archivo CSV",
                  on_click=perform_initial_preparation.clear)
    with col2:
        st.markdown("_(Los datos se cargan desde Tickers.csv con todas las columnas disponibles.)_")
    
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
    st.info("📊 **Fuente de datos**: Yahoo Finance (Call/Put Ratio basado en Open Interest)")
    
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
        with st.spinner("Ejecutando escaneo paralelo con Yahoo Finance..."):
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
    st.success(f"🎯 Sistema listo con {len(valid_tickers)} tickers válidos usando Yahoo Finance.")

# =========================================================================
# 7. PUNTO DE ENTRADA PROTEGIDO
# =========================================================================

if __name__ == "__main__":
    if check_password():
        options_scanner_page()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Introduce tus credenciales en el menú lateral para acceder.")
