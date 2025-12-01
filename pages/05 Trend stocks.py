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
# 6. ANÁLISIS DETALLADO DE UN TICKER ESPECÍFICO
# =========================================================================

def analizar_ticker_detallado(ticker):
    """Analiza un ticker específico mostrando volumen y call/put ratio por strike y expiración"""
    
    st.subheader(f"📊 Análisis Detallado de Opciones: {ticker.upper()}")
    
    with st.spinner(f"Obteniendo datos de {ticker} desde CBOE..."):
        raw_data = fetch_option_data(ticker)
        
        if not raw_data:
            st.error(f"❌ No se pudieron obtener datos de opciones para {ticker}")
            return
        
        spot_price, option_data = parse_option_data(raw_data)
        
        if option_data.empty:
            st.error(f"❌ No hay datos de opciones disponibles para {ticker}")
            return
        
        df = process_option_data_optimized(option_data)
        
        if df.empty:
            st.error(f"❌ No hay datos válidos después del procesamiento")
            return
    
    # Mostrar precio actual
    st.metric("💲 Precio Actual", f"${spot_price:.2f}")
    
    # Filtrar por fecha de expiración (60 días desde hoy)
    fecha_limite = datetime.now() + timedelta(days=60)
    df_filtrado = df[df['expiration'] <= fecha_limite].copy()
    
    if df_filtrado.empty:
        st.warning("⚠️ No hay opciones con expiración dentro de los próximos 60 días")
        return
    
    # Calcular rango de strikes (50 arriba y 50 abajo del precio actual)
    strike_min = spot_price - 50
    strike_max = spot_price + 50
    df_filtrado = df_filtrado[(df_filtrado['strike'] >= strike_min) & 
                               (df_filtrado['strike'] <= strike_max)].copy()
    
    if df_filtrado.empty:
        st.warning("⚠️ No hay opciones en el rango de strikes especificado")
        return
    
    st.success(f"✅ {len(df_filtrado)} contratos encontrados (±50 strikes, 60 días)")
    
    # Tabs para diferentes vistas
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Resumen General", "📈 Por Strike", "📅 Por Expiración", "🎯 Max Pain & Greeks", "🔍 Tabla Completa"])
    
    with tab1:
        st.markdown("#### 📊 Resumen General")
        
        calls = df_filtrado[df_filtrado['type'] == 'C']
        puts = df_filtrado[df_filtrado['type'] == 'P']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_call_vol = calls['volume'].sum()
            st.metric("📞 Volumen Calls", f"{total_call_vol:,.0f}")
        
        with col2:
            total_put_vol = puts['volume'].sum()
            st.metric("📉 Volumen Puts", f"{total_put_vol:,.0f}")
        
        with col3:
            total_vol = total_call_vol + total_put_vol
            st.metric("📊 Volumen Total", f"{total_vol:,.0f}")
        
        with col4:
            total_call_oi = calls['open_interest'].sum()
            total_put_oi = puts['open_interest'].sum()
            ratio = total_call_oi / total_put_oi if total_put_oi > 0 else 0
            
            # Determinar sentimiento
            if ratio > 1.5:
                sentiment = "🚀 Muy Alcista"
                color = "green"
            elif ratio > 1.0:
                sentiment = "📈 Alcista"
                color = "lightgreen"
            elif ratio > 0.7:
                sentiment = "⚖️ Neutral"
                color = "gray"
            elif ratio > 0.5:
                sentiment = "📉 Bajista"
                color = "orange"
            else:
                sentiment = "🔻 Muy Bajista"
                color = "red"
            
            st.metric("🔥 Call/Put Ratio", f"{ratio:.2f}", delta=sentiment)
        
        st.markdown("---")
        
        # Segunda fila de métricas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📞 Open Interest Calls", f"{total_call_oi:,.0f}")
        
        with col2:
            st.metric("📉 Open Interest Puts", f"{total_put_oi:,.0f}")
        
        with col3:
            # Put/Call Ratio (inverso)
            pc_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0
            st.metric("🔄 Put/Call Ratio", f"{pc_ratio:.2f}")
        
        with col4:
            # Volumen vs OI ratio
            vol_oi_ratio = total_vol / (total_call_oi + total_put_oi) if (total_call_oi + total_put_oi) > 0 else 0
            st.metric("⚡ Vol/OI Ratio", f"{vol_oi_ratio:.2f}")
        
        st.markdown("---")
        
        # Gráfico comparativo de volumen y OI
        st.markdown("#### 📊 Comparativa Calls vs Puts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Volumen**")
            vol_data = pd.DataFrame({
                'Calls': [total_call_vol],
                'Puts': [total_put_vol]
            })
            st.bar_chart(vol_data, use_container_width=True, height=200)
        
        with col2:
            st.markdown("**Open Interest**")
            oi_data = pd.DataFrame({
                'Calls': [total_call_oi],
                'Puts': [total_put_oi]
            })
            st.bar_chart(oi_data, use_container_width=True, height=200)
    
    with tab2:
        st.markdown("#### 📈 Análisis por Strike")
        
        # Agrupar por strike
        df_by_strike = df_filtrado.groupby(['strike', 'type']).agg({
            'volume': 'sum',
            'open_interest': 'sum'
        }).reset_index()
        
        # Pivotar para tener calls y puts en columnas
        df_pivot = df_by_strike.pivot(index='strike', columns='type', values=['volume', 'open_interest']).reset_index()
        df_pivot.columns = ['Strike', 'Call_OI', 'Put_OI', 'Call_Vol', 'Put_Vol']
        df_pivot = df_pivot.fillna(0)
        
        # Calcular métricas adicionales
        df_pivot['Call/Put_Ratio'] = df_pivot.apply(
            lambda row: row['Call_OI'] / row['Put_OI'] if row['Put_OI'] > 0 else 0, axis=1
        )
        df_pivot['Net_OI'] = df_pivot['Call_OI'] - df_pivot['Put_OI']
        df_pivot['Total_OI'] = df_pivot['Call_OI'] + df_pivot['Put_OI']
        df_pivot['OI_Skew'] = (df_pivot['Call_OI'] - df_pivot['Put_OI']) / df_pivot['Total_OI'] * 100
        
        # Identificar strike ATM (at the money)
        df_pivot['Distance_from_Price'] = abs(df_pivot['Strike'] - spot_price)
        atm_strike = df_pivot.loc[df_pivot['Distance_from_Price'].idxmin(), 'Strike']
        
        # Ordenar por strike
        df_pivot = df_pivot.sort_values('Strike')
        
        # Mostrar strike ATM
        st.info(f"💲 **Strike ATM (más cercano al precio)**: ${atm_strike:.2f}")
        
        # Tabla interactiva
        st.dataframe(
            df_pivot.style.format({
                'Strike': '${:.2f}',
                'Call_Vol': '{:.0f}',
                'Put_Vol': '{:.0f}',
                'Call_OI': '{:.0f}',
                'Put_OI': '{:.0f}',
                'Call/Put_Ratio': '{:.2f}',
                'Net_OI': '{:.0f}',
                'Total_OI': '{:.0f}',
                'OI_Skew': '{:.1f}%'
            }).background_gradient(subset=['OI_Skew'], cmap='RdYlGn', vmin=-100, vmax=100),
            use_container_width=True,
            height=400
        )
        
        st.markdown("---")
        
        # Gráficos mejorados
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📊 Open Interest por Strike")
            chart_data_oi = df_pivot.set_index('Strike')[['Call_OI', 'Put_OI']]
            st.bar_chart(chart_data_oi, use_container_width=True)
        
        with col2:
            st.markdown("##### 🎯 OI Skew (Sesgo de Open Interest)")
            chart_skew = df_pivot.set_index('Strike')['OI_Skew']
            st.bar_chart(chart_skew, use_container_width=True, color="#FF6B6B")
        
        st.markdown("---")
        
        # Call/Put Ratio por strike
        st.markdown("##### 🔥 Call/Put Ratio por Strike")
        chart_ratio = df_pivot.set_index('Strike')['Call/Put_Ratio']
        st.line_chart(chart_ratio, use_container_width=True)
    
    with tab3:
        st.markdown("#### 📅 Análisis por Fecha de Expiración")
        
        # Agrupar por expiración
        df_by_exp = df_filtrado.groupby(['expiration', 'type']).agg({
            'volume': 'sum',
            'open_interest': 'sum'
        }).reset_index()
        
        # Pivotar
        df_exp_pivot = df_by_exp.pivot(index='expiration', columns='type', values=['volume', 'open_interest']).reset_index()
        df_exp_pivot.columns = ['Expiration', 'Call_OI', 'Put_OI', 'Call_Vol', 'Put_Vol']
        df_exp_pivot = df_exp_pivot.fillna(0)
        
        # Calcular métricas
        df_exp_pivot['Call/Put_Ratio'] = df_exp_pivot.apply(
            lambda row: row['Call_OI'] / row['Put_OI'] if row['Put_OI'] > 0 else 0, axis=1
        )
        df_exp_pivot['Put/Call_Ratio'] = df_exp_pivot.apply(
            lambda row: row['Put_OI'] / row['Call_OI'] if row['Call_OI'] > 0 else 0, axis=1
        )
        df_exp_pivot['Days_to_Exp'] = df_exp_pivot['Expiration'].apply(
            lambda x: (x - datetime.now()).days
        )
        df_exp_pivot['Net_OI'] = df_exp_pivot['Call_OI'] - df_exp_pivot['Put_OI']
        df_exp_pivot['Total_Vol'] = df_exp_pivot['Call_Vol'] + df_exp_pivot['Put_Vol']
        
        # Ordenar por fecha
        df_exp_pivot = df_exp_pivot.sort_values('Expiration')
        
        # Calcular tendencia del ratio (si está subiendo o bajando)
        if len(df_exp_pivot) > 1:
            ratio_change = df_exp_pivot['Call/Put_Ratio'].iloc[-1] - df_exp_pivot['Call/Put_Ratio'].iloc[0]
            ratio_trend = "📈 Subiendo" if ratio_change > 0.1 else ("📉 Bajando" if ratio_change < -0.1 else "➡️ Estable")
            st.info(f"**Tendencia Call/Put Ratio**: {ratio_trend} ({ratio_change:+.2f} desde la primera expiración)")
        
        # Tabla
        st.dataframe(
            df_exp_pivot.style.format({
                'Expiration': lambda x: x.strftime('%Y-%m-%d'),
                'Call_Vol': '{:.0f}',
                'Put_Vol': '{:.0f}',
                'Call_OI': '{:.0f}',
                'Put_OI': '{:.0f}',
                'Call/Put_Ratio': '{:.2f}',
                'Put/Call_Ratio': '{:.2f}',
                'Days_to_Exp': '{:.0f}',
                'Net_OI': '{:+.0f}',
                'Total_Vol': '{:.0f}'
            }).background_gradient(subset=['Call/Put_Ratio'], cmap='RdYlGn', vmin=0.5, vmax=2.0),
            use_container_width=True,
            height=400
        )
        
        st.markdown("---")
        
        # Gráficos
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🔥 Call/Put Ratio por Expiración")
            chart_exp = df_exp_pivot.set_index('Expiration')['Call/Put_Ratio']
            st.line_chart(chart_exp, use_container_width=True)
        
        with col2:
            st.markdown("##### 📊 Net Open Interest (Call - Put)")
            chart_net = df_exp_pivot.set_index('Expiration')['Net_OI']
            st.bar_chart(chart_net, use_container_width=True)
        
        st.markdown("---")
        
        # Volumen total por expiración
        st.markdown("##### 📈 Volumen Total por Expiración")
        chart_vol = df_exp_pivot.set_index('Expiration')['Total_Vol']
        st.area_chart(chart_vol, use_container_width=True)
    
    with tab5:
        st.markdown("#### 🎯 Max Pain & Análisis de Greeks")
        
        # Calcular Max Pain (precio donde más opciones expiran sin valor)
        st.markdown("##### 💰 Max Pain Analysis")
        st.info("**Max Pain**: El precio al que la mayor cantidad de opciones (en valor $) expirarían sin valor, causando máxima pérdida a los compradores de opciones.")
        
        # Agrupar por strike para calcular max pain
        df_pain = df_filtrado.groupby('strike').agg({
            'open_interest': 'sum',
            'type': lambda x: list(x)
        }).reset_index()
        
        # Calcular pain por strike (simplificado)
        pain_scores = []
        for _, row in df_pain.iterrows():
            strike = row['strike']
            # Pain = suma del valor intrínseco de todas las opciones ITM
            call_pain = df_filtrado[(df_filtrado['type'] == 'C') & (df_filtrado['strike'] < strike)]['open_interest'].sum() * (strike - spot_price)
            put_pain = df_filtrado[(df_filtrado['type'] == 'P') & (df_filtrado['strike'] > strike)]['open_interest'].sum() * (spot_price - strike)
            total_pain = abs(call_pain) + abs(put_pain)
            pain_scores.append({'Strike': strike, 'Pain_Score': total_pain})
        
        df_pain_calc = pd.DataFrame(pain_scores)
        max_pain_strike = df_pain_calc.loc[df_pain_calc['Pain_Score'].idxmin(), 'Strike']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("💲 Precio Actual", f"${spot_price:.2f}")
        
        with col2:
            st.metric("🎯 Max Pain Strike", f"${max_pain_strike:.2f}")
        
        with col3:
            distance_to_pain = ((max_pain_strike - spot_price) / spot_price) * 100
            st.metric("📏 Distancia a Max Pain", f"{distance_to_pain:+.2f}%")
        
        st.markdown("---")
        
        # Greeks Analysis
        st.markdown("##### 🔬 Análisis de Greeks Agregados")
        
        if 'delta' in df_filtrado.columns and 'gamma' in df_filtrado.columns:
            # Calcular greeks totales
            total_call_delta = calls['delta'].fillna(0).sum()
            total_put_delta = puts['delta'].fillna(0).sum()
            net_delta = total_call_delta + total_put_delta  # Put delta es negativo
            
            total_gamma = df_filtrado['gamma'].fillna(0).sum()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Net Delta", f"{net_delta:,.0f}")
            
            with col2:
                st.metric("🔺 Total Gamma", f"{total_gamma:,.2f}")
            
            with col3:
                if 'vega' in df_filtrado.columns:
                    total_vega = df_filtrado['vega'].fillna(0).sum()
                    st.metric("📈 Total Vega", f"{total_vega:,.0f}")
            
            with col4:
                if 'theta' in df_filtrado.columns:
                    total_theta = df_filtrado['theta'].fillna(0).sum()
                    st.metric("⏰ Total Theta", f"{total_theta:,.0f}")
            
            st.markdown("---")
            
            # Gamma Exposure por strike
            st.markdown("##### ⚡ Gamma Exposure por Strike")
            
            gamma_by_strike = df_filtrado.groupby('strike')['gamma'].sum().reset_index()
            gamma_by_strike = gamma_by_strike.sort_values('strike')
            
            chart_gamma = gamma_by_strike.set_index('strike')['gamma']
            st.bar_chart(chart_gamma, use_container_width=True)
            
            st.info("💡 **Gamma Exposure**: Zonas de alta gamma indican niveles de precio donde los market makers tendrán que cubrir agresivamente sus posiciones.")
        
        st.markdown("---")
        
        # Implied Volatility Analysis
        if 'iv' in df_filtrado.columns:
            st.markdown("##### 📊 Análisis de Volatilidad Implícita (IV)")
            
            avg_call_iv = calls['iv'].mean() * 100 if not calls.empty else 0
            avg_put_iv = puts['iv'].mean() * 100 if not puts.empty else 0
            iv_skew = avg_put_iv - avg_call_iv
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📞 IV Promedio Calls", f"{avg_call_iv:.2f}%")
            
            with col2:
                st.metric("📉 IV Promedio Puts", f"{avg_put_iv:.2f}%")
            
            with col3:
                skew_sentiment = "Bearish" if iv_skew > 5 else ("Bullish" if iv_skew < -5 else "Neutral")
                st.metric("🎭 IV Skew", f"{iv_skew:+.2f}%", delta=skew_sentiment)
            
            # IV por strike
            iv_by_strike = df_filtrado.groupby(['strike', 'type'])['iv'].mean().reset_index()
            iv_pivot = iv_by_strike.pivot(index='strike', columns='type', values='iv')
            iv_pivot = iv_pivot.sort_index()
            
            st.markdown("##### 📈 IV por Strike (Calls vs Puts)")
            st.line_chart(iv_pivot * 100, use_container_width=True)
        st.markdown("#### 🔍 Tabla Completa de Opciones")
        
        # Preparar tabla completa
        df_display = df_filtrado[['option', 'type', 'strike', 'expiration', 'volume', 
                                   'open_interest', 'iv', 'delta', 'gamma']].copy()
        
        df_display['days_to_exp'] = df_display['expiration'].apply(
            lambda x: (x - datetime.now()).days
        )
        
        df_display = df_display.sort_values(['expiration', 'strike'])
        
        st.dataframe(
            df_display.style.format({
                'strike': '${:.2f}',
                'expiration': lambda x: x.strftime('%Y-%m-%d'),
                'volume': '{:.0f}',
                'open_interest': '{:.0f}',
                'iv': '{:.2%}',
                'delta': '{:.3f}',
                'gamma': '{:.4f}',
                'days_to_exp': '{:.0f}'
            }),
            use_container_width=True,
            height=600
        )
        
        # Botón de descarga
        csv = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar Datos Completos (CSV)",
            data=csv,
            file_name=f"{ticker}_options_detail_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# =========================================================================
# 7. VISUALIZACIÓN DE RESULTADOS
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
# 8. FUNCIÓN PRINCIPAL
# =========================================================================

def options_scanner_page():
    st.title("📊 Trend Stocks Scanner - Call/Put Ratio Analyzer")
    st.markdown("---")
    
    # TABS PRINCIPALES
    tab_scanner, tab_analisis = st.tabs(["🔍 Scanner de Opciones", "📊 Análisis Detallado de Ticker"])
    
    # =====================================================================
    # TAB 1: SCANNER DE OPCIONES (CÓDIGO ORIGINAL)
    # =====================================================================
    with tab_scanner:
        st.info("🔍 Este escáner identifica acciones con alta actividad en opciones y sesgo alcista usando CBOE API")
        
        # --- Punto 1: Preparación de Tickers ---
        col1, col2 = st.columns([1, 4])
        with col1:
            st.button("🔄 Recargar Datos", type="primary",
                      help="Borra la caché y recarga el archivo CSV",
                      on_click=perform_initial_preparation.clear,
                      key="reload_scanner")
        with col2:
            st.markdown("_(Los datos se cargan desde Explorer.csv con todas las columnas disponibles.)_")
        
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
            if st.button("🗑️ Limpiar", use_container_width=True, key="limpiar_scanner"):
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
    
    # =====================================================================
    # TAB 2: ANÁLISIS DETALLADO DE TICKER
    # =====================================================================
    with tab_analisis:
        st.info("📊 Analiza un ticker específico mostrando volumen y call/put ratio para ±50 strikes y 60 días")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            ticker_input = st.text_input(
                "🎯 Introduce el ticker a analizar",
                placeholder="Ej: AAPL, TSLA, SPY...",
                max_chars=10,
                key="ticker_analisis"
            ).strip().upper()
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            analizar_btn = st.button("🔍 Analizar", type="primary", use_container_width=True)
        
        st.markdown("---")
        
        if analizar_btn:
            if ticker_input:
                analizar_ticker_detallado(ticker_input)
            else:
                st.warning("⚠️ Por favor introduce un ticker válido")
        else:
            st.info("👆 Introduce un ticker y presiona 'Analizar' para ver los datos de opciones")

# =========================================================================
# 9. PUNTO DE ENTRADA
# =========================================================================

if __name__ == "__main__":
    options_scanner_page()
