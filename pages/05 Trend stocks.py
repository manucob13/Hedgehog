# pages/Trend stocks.py
import streamlit as st
import pandas as pd
import requests
import yfinance as yf
from datetime import timedelta, datetime
from io import StringIO
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
import time
from math import sqrt
import schwab
from schwab.auth import easy_client
from schwab.client import Client
from utils import check_password
import json

# =========================================================================
# 0. CONFIGURACIÓN Y VARIABLES
# =========================================================================

st.set_page_config(page_title="Trend Stocks", layout="wide")

# Cargar variables de Schwab desde secrets
try:
    api_key = st.secrets["schwab"]["api_key"]
    app_secret = st.secrets["schwab"]["app_secret"]
    redirect_uri = st.secrets["schwab"]["redirect_uri"]
except KeyError as e:
    st.error(f"❌ Falta configurar los secrets de Schwab. Clave faltante: {e}. Asegúrate de que tienes [schwab] en secrets.toml")
    st.stop()

# Ruta local del token
token_path = "schwab_token.json"

# =========================================================================
# 1. PREPARACIÓN DE TICKERS (DESDE CSV)
# =========================================================================

@st.cache_resource(ttl=timedelta(hours=24), show_spinner=False)
def perform_initial_preparation():
    st.subheader("1. Preparación de Tickers")
    
    status_text = st.empty()
    
    # Leer tickers del archivo CSV
    if os.path.exists('Tickers.csv'):
        try:
            df_tickers = pd.read_csv('Tickers.csv')
            tickers = df_tickers.iloc[:, 0].astype(str).str.upper().str.strip().tolist()
            tickers = sorted(set(tickers))  # Eliminar duplicados y ordenar
            
            st.success(f"✅ 'Tickers.csv' encontrado con {len(tickers)} tickers.")
            st.info("ℹ️ Los tickers se usan directamente sin validación adicional.")
            
            status_text.empty()
            st.divider()
            
            return tickers
            
        except Exception as e:
            st.error(f"❌ Error al leer 'Tickers.csv': {e}")
            st.stop()
    else:
        st.error("❌ 'Tickers.csv' no encontrado en el directorio raíz.")
        st.info("📝 Crea un archivo 'Tickers.csv' con una columna de tickers (uno por línea)")
        st.stop()

# =========================================================================
# 2. CONEXIÓN CON BROKER SCHWAB
# =========================================================================

def connect_to_schwab():
    """
    Usa el token existente si está disponible.
    No abre flujo OAuth ni usa puerto; solo valida token.json.
    """
    st.subheader("2. Conexión con Broker Schwab")
    
    if not os.path.exists(token_path):
        st.error("❌ No se encontró 'schwab_token.json'. Genera el token desde tu notebook local antes de usar esta página.")
        return None
    
    try:
        client = easy_client(
            api_key=api_key,
            app_secret=app_secret,
            callback_url=redirect_uri,
            token_path=token_path
        )
        
        # Verificar token
        test_response = client.get_quote("AAPL")
        if hasattr(test_response, "status_code") and test_response.status_code != 200:
            raise Exception(f"Respuesta inesperada: {test_response.status_code}")
        
        st.success("✅ Conexión con Schwab verificada (token activo).")
        return client
    
    except Exception as e:
        st.error(f"❌ Error al inicializar Schwab Client: {e}")
        st.warning("⚠️ Si el error persiste, elimina el archivo 'schwab_token.json' y vuelve a generarlo desde tu entorno local.")
        return None

# =========================================================================
# 3. FUNCIONES PARA OBTENER DATOS DE OPCIONES DESDE SCHWAB
# =========================================================================

def obtener_datos_opciones_schwab(client, ticker):
    """
    Obtiene datos de opciones desde Schwab API:
    - Call/Put Ratio
    - Volumen total de opciones
    """
    try:
        response = client.get_option_chain(ticker)
        if response.status_code != 200:
            return None, None
        
        opciones = response.json()
        
        # Obtener mapas de calls y puts
        call_map = opciones.get('callExpDateMap', {})
        put_map = opciones.get('putExpDateMap', {})
        
        # Calcular volúmenes totales
        total_call_volume = 0
        total_put_volume = 0
        
        # Sumar volumen de todas las calls
        for fecha, strikes in call_map.items():
            for strike, contratos in strikes.items():
                for contrato in contratos:
                    vol = contrato.get('totalVolume', 0)
                    if vol:
                        total_call_volume += vol
        
        # Sumar volumen de todas las puts
        for fecha, strikes in put_map.items():
            for strike, contratos in strikes.items():
                for contrato in contratos:
                    vol = contrato.get('totalVolume', 0)
                    if vol:
                        total_put_volume += vol
        
        # Calcular Call/Put Ratio
        if total_put_volume > 0:
            call_put_ratio = total_call_volume / total_put_volume
        else:
            call_put_ratio = 0.0
        
        # Volumen total de opciones
        options_volume = total_call_volume + total_put_volume
        
        return options_volume, call_put_ratio
    
    except Exception as e:
        return None, None

def procesar_ticker_opciones(args):
    """Función helper para paralelizar obtención de datos de opciones"""
    client, ticker = args
    options_volume, call_put_ratio = obtener_datos_opciones_schwab(client, ticker)
    
    return {
        'ticker': ticker,
        'options_volume': options_volume if options_volume else 0,
        'call_put_ratio': call_put_ratio if call_put_ratio else 0.0
    }

# =========================================================================
# 4. FILTROS Y PARÁMETROS
# =========================================================================

def seccion_filtros():
    """Sección de configuración de filtros"""
    st.subheader("3. Configuración de Filtros")
    
    with st.container():
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
                help="Ratio mínimo (>0.5 significa más calls que puts)"
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
    
    return volumen_min, ratio_min

# =========================================================================
# 5. ESCANEO DE OPCIONES (PARALELO CON SCHWAB API)
# =========================================================================

def ejecutar_escaneo_opciones(client, tickers, volumen_min, ratio_min):
    """Ejecuta el escaneo de opciones usando Schwab API EN PARALELO"""
    
    # Contenedores para mostrar progreso
    status_container = st.empty()
    progress_bar = st.progress(0)
    
    # Obtener datos de opciones (PARALELO)
    status_container.info("📊 Obteniendo datos de opciones desde Schwab API (paralelo)...")
    
    args_list = [(client, ticker) for ticker in tickers]
    
    resultados = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = [executor.submit(procesar_ticker_opciones, args) for args in args_list]
        for i, future in enumerate(futures):
            result = future.result()
            if result and result['options_volume'] > 0:
                resultados.append(result)
            progress_bar.progress((i + 1) / len(futures))
    
    if not resultados:
        status_container.error("❌ No se encontraron tickers con datos de opciones válidos")
        progress_bar.empty()
        return None
    
    df = pd.DataFrame(resultados)
    df.columns = ['Ticker', 'Options_Volume', 'Call_Put_Ratio']
    
    status_container.success(f"✅ Datos obtenidos: {len(df)} tickers con información de opciones")
    
    # Aplicar filtros
    df_filtrado = df[
        (df['Options_Volume'] > volumen_min) & 
        (df['Call_Put_Ratio'] > ratio_min)
    ].copy()
    
    if df_filtrado.empty:
        status_container.warning("⚠️ No hay tickers que cumplan los criterios de filtrado")
        progress_bar.empty()
        return None
    
    # Ordenar por volumen descendente
    df_filtrado = df_filtrado.sort_values('Options_Volume', ascending=False)
    df_filtrado.insert(0, 'Rank', range(1, len(df_filtrado) + 1))
    
    progress_bar.empty()
    status_container.success(f"🎉 Escaneo completado: {len(df_filtrado)} acciones encontradas")
    
    return df_filtrado

# =========================================================================
# 6. VISUALIZACIÓN DE RESULTADOS
# =========================================================================

def mostrar_resultados(df_resultados):
    """Muestra los resultados filtrados en tabla interactiva"""
    st.subheader("4. Resultados del Escaneo")
    
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
    
    # Crear copia para display y formatear
    df_display = df_resultados.copy()
    df_display['Call_Put_Ratio_Formatted'] = df_display['Call_Put_Ratio'].apply(lambda x: f"{x:.2f}")
    
    # Seleccionar columnas para la tabla
    df_table = df_display[['Rank', 'Ticker', 'Options_Volume', 'Call_Put_Ratio_Formatted']].copy()
    
    # Renombrar columnas para mejor presentación
    df_table.columns = ['🏅 Rank', '🎯 Ticker', '📈 Vol. Opciones', '🔥 Call/Put Ratio']
    
    # Tabla de resultados
    st.markdown("#### 🏆 Acciones con Mayor Actividad en Opciones")
    st.dataframe(
        df_table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "🏅 Rank": st.column_config.NumberColumn(width="small"),
            "🎯 Ticker": st.column_config.TextColumn(width="medium"),
            "📈 Vol. Opciones": st.column_config.NumberColumn(width="medium", format="%d"),
            "🔥 Call/Put Ratio": st.column_config.TextColumn(width="medium")
        }
    )
    
    # Gráfico de distribución (usar columnas numéricas originales)
    st.markdown("---")
    st.markdown("#### 📊 Distribución de Call/Put Ratio")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 10 por ratio (usar Call_Put_Ratio numérico original)
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
    
    # Botón de descarga (formatear para CSV legible)
    st.markdown("---")
    df_csv = df_resultados.copy()
    df_csv['Call_Put_Ratio'] = df_csv['Call_Put_Ratio'].apply(lambda x: f"{x:.2f}")
    
    csv = df_csv.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar Resultados (CSV)",
        data=csv,
        file_name=f"options_scanner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=False
    )

# =========================================================================
# 7. FUNCIÓN PRINCIPAL
# =========================================================================

def options_scanner_page():
    st.title("📊 Trend Stocks Scanner - Call/Put Ratio Analyzer")
    st.markdown("---")
    st.info("🔍 Este escáner identifica acciones con alta actividad en opciones y sesgo alcista usando datos REALES de Schwab API")
    
    # --- Punto 1: Preparación de Tickers ---
    col1, col2 = st.columns([1, 4])
    with col1:
        st.button("🔄 Recargar Tickers", type="primary",
                  help="Borra la caché y recarga Tickers.csv",
                  on_click=perform_initial_preparation.clear)
    with col2:
        st.markdown("_(Los tickers se cargan desde Tickers.csv sin validación.)_")
    
    st.divider()
    valid_tickers = perform_initial_preparation()
    
    # --- Punto 2: Conexión Schwab ---
    st.divider()
    
    # Guardar cliente en session_state si no existe
    if 'schwab_client_options' not in st.session_state:
        st.session_state.schwab_client_options = connect_to_schwab()
    else:
        st.subheader("2. Conexión con Broker Schwab")
        st.success("✅ Conexión con Schwab verificada (ya conectado en esta sesión).")
    
    schwab_client = st.session_state.schwab_client_options
    
    # --- Punto 3: Filtros ---
    st.divider()
    volumen_min, ratio_min = seccion_filtros()
    
    # --- Punto 4: Escaneo ---
    st.divider()
    st.subheader("4. Escaneo de Opciones")
    
    # Verificar si hay cliente válido
    if 'schwab_client_options' not in st.session_state or st.session_state.schwab_client_options is None:
        st.error("❌ Necesitas conectar con Schwab antes de ejecutar el escaneo")
        st.info("💡 Recarga la página para intentar reconectar")
    else:
        schwab_client = st.session_state.schwab_client_options
        st.info(f"📊 Tickers listos para escanear: **{len(valid_tickers)}** | 🚀 Modo: **Paralelo (15 hilos)**")
        st.warning("⚠️ El escaneo tardará 2-4 minutos. **No cambies de página durante el proceso.**")
        st.info("📊 **Datos REALES**: Call/Put Ratio y Volumen de opciones desde Schwab API")
        
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
            with st.spinner("Ejecutando escaneo paralelo con Schwab API..."):
                df_resultados = ejecutar_escaneo_opciones(
                    schwab_client,
                    valid_tickers,
                    volumen_min,
                    ratio_min
                )
                st.session_state.df_resultados_options = df_resultados
            
            elapsed_time = time.time() - start_time
            
            if df_resultados is not None and not df_resultados.empty:
                st.balloons()
                st.success(f"🎉 Escaneo completado en {elapsed_time:.1f} segundos - {len(df_resultados)} acciones encontradas")
            else:
                st.warning("⚠️ No se encontraron acciones que cumplan los criterios")
    
    # --- Punto 5: Resultados ---
    st.divider()
    if 'df_resultados_options' in st.session_state:
        mostrar_resultados(st.session_state.df_resultados_options)
    else:
        st.info("👆 Ejecuta el escaneo primero para ver los resultados aquí")
    
    # --- Estado final ---
    st.divider()
    if schwab_client:
        st.success(f"🎯 Sistema listo con {len(valid_tickers)} tickers válidos y conexión Schwab activa.")
    else:
        st.info("⏳ Conecta tu token Schwab para activar funciones de trading.")

# =========================================================================
# 8. PUNTO DE ENTRADA PROTEGIDO
# =========================================================================

if __name__ == "__main__":
    if check_password():
        options_scanner_page()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Introduce tus credenciales en el menú lateral para acceder.")
