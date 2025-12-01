# pages/Trend_stocks.py
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import timedelta, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
import requests
from typing import Optional, Tuple

# =========================================================================
# 0. CONFIGURACIÓN Y CONSTANTES
# =========================================================================

st.set_page_config(page_title="Trend Stocks Scanner", layout="wide")

CONTRACT_SIZE = 100
CACHE_DIR = "data"
API_BASE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options"

# =========================================================================
# 1. FUNCIONES AUXILIARES PARA OPCIONES (CBOE API)
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
    """Obtiene volumen y call/put ratio desde CBOE"""
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
        
        if total_put_oi > 0:
            call_put_ratio = total_call_oi / total_put_oi
        else:
            call_put_ratio = 0.0
        
        return options_volume, call_put_ratio
    
    except Exception:
        return None, None

# =========================================================================
# 2. CARGA DE TICKERS
# =========================================================================

@st.cache_resource(ttl=timedelta(hours=24))
def cargar_tickers():
    """Cargar tickers desde CSV"""
    csv_filename = 'Tickers.csv'
    if os.path.exists(csv_filename):
        try:
            df_tickers = pd.read_csv(csv_filename)
            if 'Ticker' not in df_tickers.columns:
                st.error(f"❌ El archivo '{csv_filename}' no tiene columna 'Ticker'")
                st.stop()
            
            tickers = df_tickers['Ticker'].astype(str).str.upper().str.strip().tolist()
            tickers = sorted(set(tickers))
            
            return tickers
        except Exception as e:
            st.error(f"❌ Error al leer '{csv_filename}': {e}")
            st.stop()
    else:
        st.error(f"❌ '{csv_filename}' no encontrado")
        st.stop()

# =========================================================================
# 3. FUNCIONES DE CÁLCULO DE INDICADORES
# =========================================================================

def calcular_rsc_mansfield(prices, benchmark_prices, periodo):
    """Calcula RSC Mansfield"""
    try:
        cociente = prices / benchmark_prices
        count_rsc = cociente.rolling(window=periodo).sum()
        base_price = count_rsc / periodo
        rsc = ((cociente / base_price) - 1) * 10
        return rsc.iloc[-1] if not rsc.empty else None
    except:
        return None

def calcular_liquidez(prices, volumes, periodo):
    """Calcula liquidez promedio"""
    try:
        liquidity = (prices * volumes).rolling(window=periodo).mean()
        return liquidity.iloc[-1] if not liquidity.empty else None
    except:
        return None

def calcular_dist_max(prices, periodos):
    """Calcula distancia al máximo"""
    try:
        maximo = prices.rolling(window=periodos).max().iloc[-1]
        precio_actual = prices.iloc[-1]
        dist = abs(precio_actual - maximo) / precio_actual * 100
        return dist
    except:
        return None

def calcular_wma(prices, periodo=30):
    """Calcula WMA (Weighted Moving Average)"""
    try:
        weights = np.arange(1, periodo + 1)
        wma = prices.rolling(window=periodo).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
        return wma.iloc[-1] if not wma.empty else None
    except:
        return None

def calcular_atlas(prices):
    """Calcula indicador Atlas"""
    try:
        # Bollinger Bands
        bb_period = 20
        rolling_mean = prices.rolling(window=bb_period).mean()
        rolling_std = prices.rolling(window=bb_period).std()
        bb_top = rolling_mean + (2 * rolling_std)
        bb_bot = rolling_mean - (2 * rolling_std)
        
        dbb = np.sqrt((bb_top - bb_bot) / bb_top) * 20
        dbbmed = dbb.ewm(span=120).mean()
        factor = dbbmed * 4 / 5
        atl = dbb - factor
        al1 = np.where(atl > 0, 0, 1)
        
        return al1[-1] if len(al1) > 0 else 0
    except:
        return 0

def calcular_mic_value(prices):
    """Calcula MIC Value (ROC/ATR)"""
    try:
        roc18 = ((prices.iloc[-1] / prices.iloc[-19]) - 1) * 100 if len(prices) > 19 else 0
        roc50 = ((prices.iloc[-1] / prices.iloc[-51]) - 1) * 100 if len(prices) > 51 else 0
        
        # ATR simplificado (High-Low range)
        atr18 = prices.rolling(window=18).std().iloc[-1] / prices.iloc[-1] * 100
        atr50 = prices.rolling(window=50).std().iloc[-1] / prices.iloc[-1] * 100
        
        if atr18 > 0 and atr50 > 0:
            mic_value = (roc18 / atr18) * 0.6 + (roc50 / atr50) * 0.4
        else:
            mic_value = 0
        
        return mic_value
    except:
        return 0

def calcular_linear_regression(prices, periods=30):
    """Calcula regresión lineal: Slope, R2, Normalized Distance"""
    try:
        if len(prices) < periods:
            return None, None, None
        
        y = prices.iloc[-periods:].values
        x = np.arange(len(y))
        
        # Regresión lineal
        A = np.vstack([x, np.ones(len(x))]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Normalized distance
        se_regression = np.sqrt(ss_res / (len(y) - 2))
        distance = abs(y[-1] - y_pred[-1])
        normalized_distance = distance / se_regression if se_regression > 0 else 0
        
        return slope, r_squared, normalized_distance
    except:
        return None, None, None

def calcular_sharpe_ratio(prices, periodos=50):
    """Calcula Sharpe Ratio"""
    try:
        returns = prices.pct_change().dropna()
        if len(returns) < periodos:
            return None
        
        mean_return = returns.iloc[-periodos:].mean()
        std_return = returns.iloc[-periodos:].std()
        
        if std_return > 0:
            sharpe = (mean_return / std_return) * np.sqrt(periodos)
        else:
            sharpe = 0
        
        return sharpe
    except:
        return None

def calcular_macdv(prices):
    """Calcula MACD-V"""
    try:
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        
        # ATR simplificado
        atr = prices.rolling(window=26).std()
        
        lineamacd = ((ema12 - ema26) / atr) * 100
        return lineamacd.iloc[-1] if not lineamacd.empty else None
    except:
        return None

def calcular_rsi(prices, periodo=14):
    """Calcula RSI"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periodo).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not rsi.empty else None
    except:
        return None

# =========================================================================
# 4. PROCESAMIENTO DE UN TICKER
# =========================================================================

def procesar_ticker(ticker, config, benchmark_data, mercado):
    """Procesa un ticker completo con todos los filtros"""
    try:
        # Descargar datos históricos
        periodo_descarga = '3y' if config['timeframe'] == 'Semanal' else '2y'
        interval = '1wk' if config['timeframe'] == 'Semanal' else '1d'
        
        stock = yf.Ticker(ticker)
        hist = stock.history(period=periodo_descarga, interval=interval)
        
        if hist.empty or len(hist) < 100:
            return None
        
        prices = hist['Close']
        volumes = hist['Volume']
        
        # Obtener sector
        try:
            sector = stock.info.get('sector', 'N/A')
        except:
            sector = 'N/A'
        
        # Calcular indicadores según configuración
        periodo_rsc = config['periodo_rsc']
        
        # RSC Mansfield
        rsc = calcular_rsc_mansfield(prices, benchmark_data, periodo_rsc)
        if rsc is None or rsc <= 0:
            return None
        
        # Liquidez
        liquidity = calcular_liquidez(prices, volumes, periodo_rsc)
        volumen_min = config['volumen_min']
        
        if config['inversion'] > 0:
            precio_actual = prices.iloc[-1]
            if liquidity is None or liquidity <= volumen_min:
                return None
            if precio_actual < 3 or precio_actual > (config['inversion'] / 100):
                return None
        else:
            if liquidity is None or liquidity <= volumen_min:
                return None
        
        # Distancia a máximos
        periodos_max = 52 * config['max_years'] if config['timeframe'] == 'Semanal' else 250 * config['max_years']
        dist_max = calcular_dist_max(prices, periodos_max)
        if dist_max is None or dist_max > config['pct_a_max']:
            return None
        
        # WMA
        wma = calcular_wma(prices, 30)
        precio_actual = prices.iloc[-1]
        if wma is None or precio_actual <= wma:
            return None
        
        dist_wma = abs(precio_actual - wma) / precio_actual * 100
        if dist_wma > config['dist_wma']:
            return None
        
        # RSC Sectorial (simplificado, usamos RSC general como aproximación)
        rsc_s = rsc * 0.8  # Aproximación
        if rsc_s <= 0:
            return None
        
        resultado = {
            'Ticker': ticker,
            'Price': precio_actual,
            'Liquidity': liquidity,
            'Sector': sector,
            'RSC': rsc,
            'RSC_S': rsc_s,
            'dist_a_max': dist_max,
            'dist_a_WMA30': dist_wma,
        }
        
        # Filtros opcionales
        if config['atlas_enabled']:
            atlas = calcular_atlas(prices)
            resultado['Atlas'] = atlas
            if atlas <= 0:
                return None
        
        if config['mic_enabled']:
            mic_value = calcular_mic_value(prices)
            resultado['MICValue'] = mic_value
            if mic_value <= 5:
                return None
        
        if config['lr_enabled']:
            slope, r2, norm_dist = calcular_linear_regression(prices, 30)
            resultado['Slope'] = slope
            resultado['R2'] = r2
            resultado['Norm_Dist'] = norm_dist
            if r2 is None or r2 < 0.7 or norm_dist is None or norm_dist > 1.5:
                return None
        
        if config['sr_enabled']:
            sharpe = calcular_sharpe_ratio(prices, 50)
            resultado['Sharpe_ratio'] = sharpe
            if sharpe is None or sharpe < 1.5:
                return None
        
        if config['macdv_enabled']:
            macdv = calcular_macdv(prices)
            resultado['MACD_v'] = macdv
            if macdv is None or macdv < 50:
                return None
        
        if config['rsi_enabled']:
            rsi = calcular_rsi(prices, config['rsi_periodo'])
            resultado['RSI'] = rsi
            if rsi is None or rsi < config['rsi_min'] or rsi > config['rsi_max']:
                return None
        
        return resultado
    
    except Exception as e:
        return None

# =========================================================================
# 5. ESCANEO PRINCIPAL
# =========================================================================

def ejecutar_escaneo(tickers, config, mercado):
    """Ejecuta escaneo en paralelo"""
    
    # Descargar benchmark
    st.info(f"📊 Descargando datos del benchmark {config['benchmark']}...")
    periodo_descarga = '3y' if config['timeframe'] == 'Semanal' else '2y'
    interval = '1wk' if config['timeframe'] == 'Semanal' else '1d'
    
    benchmark = yf.Ticker(config['benchmark'])
    benchmark_hist = benchmark.history(period=periodo_descarga, interval=interval)
    
    if benchmark_hist.empty:
        st.error("❌ No se pudo descargar datos del benchmark")
        return None
    
    benchmark_data = benchmark_hist['Close']
    
    # Procesar tickers en paralelo
    resultados = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(procesar_ticker, ticker, config, benchmark_data, mercado): ticker 
                   for ticker in tickers}
        
        completed = 0
        total = len(futures)
        
        for future in as_completed(futures):
            completed += 1
            progress_bar.progress(completed / total)
            status_text.text(f"Procesando: {completed}/{total} tickers...")
            
            try:
                result = future.result()
                if result:
                    resultados.append(result)
            except Exception:
                pass
    
    progress_bar.empty()
    status_text.empty()
    
    if not resultados:
        return None
    
    df = pd.DataFrame(resultados)
    return df

# =========================================================================
# 6. ENRIQUECIMIENTO CON DATOS DE OPCIONES
# =========================================================================

def enriquecer_con_opciones(df, filtros_opciones):
    """Añade datos de opciones desde CBOE"""
    
    st.info(f"📊 Obteniendo datos de opciones para {len(df)} tickers desde CBOE...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    options_data = []
    
    for idx, ticker in enumerate(df['Ticker']):
        progress_bar.progress((idx + 1) / len(df))
        status_text.text(f"Opciones: {idx + 1}/{len(df)} - {ticker}")
        
        vol, ratio = obtener_datos_opciones_cboe(ticker)
        
        options_data.append({
            'Ticker': ticker,
            'Options_Volume': vol if vol else 0,
            'Call_Put_Ratio': ratio if ratio else 0.0
        })
    
    progress_bar.empty()
    status_text.empty()
    
    df_options = pd.DataFrame(options_data)
    df_final = df.merge(df_options, on='Ticker', how='left')
    
    # Aplicar filtros de opciones
    if filtros_opciones['aplicar_filtros']:
        df_final = df_final[
            (df_final['Options_Volume'] >= filtros_opciones['volumen_min']) &
            (df_final['Call_Put_Ratio'] >= filtros_opciones['ratio_min'])
        ]
    
    return df_final

# =========================================================================
# 7. INTERFAZ STREAMLIT
# =========================================================================

def main():
    st.title("📊 Trend Stocks Scanner - Diario y Semanal")
    st.markdown("---")
    
    # Cargar tickers
    with st.spinner("Cargando tickers..."):
        tickers = cargar_tickers()
    
    st.success(f"✅ {len(tickers)} tickers cargados")
    
    # CONFIGURACIÓN
    st.header("⚙️ Configuración del Scanner")
    
    tab1, tab2 = st.tabs(["📅 Timeframe y Filtros Base", "🔧 Filtros Opcionales"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌍 Mercado y Timeframe")
            mercado = st.selectbox("Mercado", ["USA", "AUS"])
            timeframe = st.selectbox("Timeframe", ["Diario", "Semanal"])
            
            # Benchmark automático
            if mercado == "USA":
                benchmark = "SPY"
            else:
                benchmark = "^AXJO"
            
            st.info(f"📊 Benchmark: {benchmark}")
        
        with col2:
            st.subheader("📊 Períodos RSC")
            if timeframe == "Semanal":
                periodo_rsc = st.number_input("Período RSC", min_value=1, max_value=500, value=52)
            else:
                periodo_rsc = st.number_input("Período RSC", min_value=1, max_value=500, value=250)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("💰 Liquidez")
            if timeframe == "Semanal":
                volumen_default = 5000000
            else:
                volumen_default = 500000
            
            volumen_min = st.number_input(
                "Volumen mínimo", 
                min_value=500000, 
                max_value=10000000, 
                value=volumen_default,
                step=100000
            )
            
            inversion = st.number_input(
                "Inversión (0=sin filtro precio)",
                min_value=0,
                max_value=10000,
                value=0,
                step=500
            )
        
        with col2:
            st.subheader("📈 Máximos")
            if timeframe == "Semanal":
                pct_default = 5
            else:
                pct_default = 10
            
            pct_a_max = st.slider("% a Máximo", 0, 10, pct_default)
            max_years = st.slider("Años Máximo", 1, 5, 3 if timeframe == "Semanal" else 1)
        
        with col3:
            st.subheader("📉 WMA")
            if timeframe == "Semanal":
                dist_default = 9
            else:
                dist_default = 10
            
            dist_wma = st.slider("% a WMA30", 0, 15, dist_default)
    
    with tab2:
        st.subheader("🔧 Filtros Opcionales")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            atlas_enabled = st.checkbox("Atlas", value=False)
            mic_enabled = st.checkbox("MIC Value > 5", value=True)
            lr_enabled = st.checkbox("Linear Regression (R²≥0.7)", value=True)
        
        with col2:
            sr_enabled = st.checkbox("Sharpe Ratio ≥ 1.5", value=True)
            macdv_enabled = st.checkbox("MACD-v ≥ 50", value=True)
            rsi_enabled = st.checkbox("Filtro RSI", value=True if timeframe == "Diario" else False)
        
        with col3:
            if rsi_enabled:
                rsi_periodo = st.number_input("RSI Período", 1, 50, 14)
                rsi_min = st.slider("RSI Mínimo", 0, 100, 40)
                rsi_max = st.slider("RSI Máximo", 0, 100, 60)
            else:
                rsi_periodo = 14
                rsi_min = 40
                rsi_max = 60
    
    # FILTROS DE OPCIONES
    st.markdown("---")
    st.header("📊 Filtros de Opciones (CBOE)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        aplicar_filtros_opciones = st.checkbox("Aplicar filtros de opciones", value=True)
    
    with col2:
        volumen_opciones_min = st.number_input(
            "Volumen mínimo opciones",
            min_value=0,
            value=5000,
            step=1000
        )
    
    with col3:
        ratio_min = st.number_input(
            "Call/Put Ratio mínimo",
            min_value=0.0,
            value=0.5,
            step=0.1,
            format="%.2f"
        )
    
    # BOTÓN DE ESCANEO
    st.markdown("---")
    
    if st.button("🚀 EJECUTAR SCANNER", type="primary", use_container_width=True):
        
        config = {
            'timeframe': timeframe,
            'benchmark': benchmark,
            'periodo_rsc': periodo_rsc,
            'volumen_min': volumen_min,
            'inversion': inversion,
            'pct_a_max': pct_a_max,
            'max_years': max_years,
            'dist_wma': dist_wma,
            'atlas_enabled': atlas_enabled,
            'mic_enabled': mic_enabled,
            'lr_enabled': lr_enabled,
            'sr_enabled': sr_enabled,
            'macdv_enabled': macdv_enabled,
            'rsi_enabled': rsi_enabled,
            'rsi_periodo': rsi_periodo,
            'rsi_min': rsi_min,
            'rsi_max': rsi_max
        }
        
        filtros_opciones = {
            'aplicar_filtros': aplicar_filtros_opciones,
            'volumen_min': volumen_opciones_min,
            'ratio_min': ratio_min
        }
        
        start_time = time.time()
        
        with st.spinner("🔍 Escaneando tickers..."):
            df_resultados = ejecutar_escaneo(tickers, config, mercado)
        
        if df_resultados is None or df_resultados.empty:
            st.warning("⚠️ No se encontraron acciones que cumplan los criterios")
            st.stop()
        
        st.success(f"✅ Primera fase completada: {len(df_resultados)} acciones encontradas")
        
        # Enriquecer con opciones
        with st.spinner("📊 Obteniendo datos de opciones..."):
            df_final = enriquecer_con_opciones(df_resultados, filtros_opciones)
        
        elapsed = time.time() - start_time
        
        if df_final.empty:
            st.warning("⚠️ Ninguna acción pasó los filtros de opciones")
            st.stop()
        
        st.balloons()
        st.success(f"🎉 Escaneo completado en {elapsed:.1f}s - {len(df_final)} acciones finales")
        
        # MOSTRAR RESULTADOS
        st.markdown("---")
        st.header("📊 Resultados")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Total Acciones", len(df_final))
        with col2:
            avg_ratio = df_final['Call_Put_Ratio'].mean()
            st.metric("📊 Ratio Promedio", f"{avg_ratio:.2f}")
        with col3:
            ratio_gt_1 = len(df_final[df_final['Call_Put_Ratio'] > 1.0])
            st.metric("🚀 Ratio > 1.0", ratio_gt_1)
        with col4:
            total_vol = df_final['Options_Volume'].sum()
            st.metric("📈 Vol. Total", f"{total_vol:,.0f}")
        
        # Tabla de resultados
        st.dataframe(df_final, use_container_width=True)
        
        # Descarga
        csv = df_final.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar CSV",
            data=csv,
            file_name=f"scanner_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
