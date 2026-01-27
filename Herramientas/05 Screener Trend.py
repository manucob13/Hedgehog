import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import random
from utils.utils import check_password
from utils.tickers import create_tickers_universe

warnings.filterwarnings('ignore')

# Lock global para sincronizar descargas de yfinance
_yfinance_lock = Lock()

# ============= MAPEO DE SECTORES A ETFs =============
SECTOR_ETFS = {
    'Technology': 'XLK',
    'Healthcare': 'XLV',
    'Financial Services': 'XLF',
    'Financials': 'XLF',
    'Consumer Cyclical': 'XLY',
    'Consumer Defensive': 'XLP',
    'Industrials': 'XLI',
    'Energy': 'XLE',
    'Utilities': 'XLU',
    'Real Estate': 'XLRE',
    'Basic Materials': 'XLB',
    'Materials': 'XLB',
    'Communication Services': 'XLC',
    'Consumer Discretionary': 'XLY'
}

# ============= FUNCIONES DE CÁLCULO =============

def download_weekly_data(ticker, period="5y"):
    """Descarga datos semanales para un ticker"""
    try:
        with _yfinance_lock:
            data = yf.download(ticker, period=period, interval="1wk", progress=False, show_errors=False)
        
        if data.empty or len(data) < 52:
            return None
        
        # Asegurar que tenemos las columnas necesarias
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        return data
    except Exception as e:
        return None

def get_sector_info(ticker):
    """Obtiene información del sector del ticker"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        sector = info.get('sector', None)
        
        if sector and sector in SECTOR_ETFS:
            return sector, SECTOR_ETFS[sector]
        return None, None
    except:
        return None, None

def calculate_rsc_mansfield(prices, benchmark_prices, period=52):
    """
    Calcula el RSC Mansfield
    RSC = ((Ratio / Media_Ratio) - 1) * 10
    """
    try:
        if len(prices) < period or len(benchmark_prices) < period:
            return None
        
        # Alinear índices
        common_index = prices.index.intersection(benchmark_prices.index)
        if len(common_index) < period:
            return None
        
        prices_aligned = prices.loc[common_index]
        benchmark_aligned = benchmark_prices.loc[common_index]
        
        ratio = prices_aligned / benchmark_aligned
        avg_ratio = ratio.rolling(window=period).mean()
        rsc = ((ratio / avg_ratio) - 1) * 10
        
        return rsc.iloc[-1] if not pd.isna(rsc.iloc[-1]) else None
    except:
        return None

def calculate_wma(prices, period=30):
    """Calcula la media móvil ponderada"""
    try:
        if len(prices) < period:
            return None
        
        weights = np.arange(1, period + 1)
        wma = prices.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
        return wma.iloc[-1] if not pd.isna(wma.iloc[-1]) else None
    except:
        return None

def calculate_linear_regression(prices, period=30):
    """
    Calcula regresión lineal y sus métricas
    Returns: slope, r_squared, normalized_distance
    """
    try:
        if len(prices) < period:
            return None, None, None
        
        y = prices.values[-period:]
        x = np.arange(len(y))
        
        # Regresión lineal
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]
        
        # Línea de regresión
        lin_reg = np.polyval(coeffs, x)
        
        # R-squared
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        ss_res = np.sum((y - lin_reg) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Error estándar
        se_regression = np.sqrt(ss_res / (period - 2)) if period > 2 else 0
        
        # Distancia normalizada
        distance = abs(y[-1] - lin_reg[-1])
        normalized_distance = distance / se_regression if se_regression > 0 else 0
        
        return slope, r_squared, normalized_distance
    except:
        return None, None, None

def calculate_atlas(prices, period_bb=20, period_ema=120):
    """Calcula el indicador Atlas"""
    try:
        if len(prices) < max(period_bb, period_ema):
            return 0
        
        # Bandas de Bollinger
        sma = prices.rolling(window=period_bb).mean()
        std = prices.rolling(window=period_bb).std()
        bb_top = sma + (2 * std)
        bb_bot = sma - (2 * std)
        
        # DBB
        dbb = np.sqrt((bb_top - bb_bot) / bb_top) * 20
        dbb_med = dbb.ewm(span=period_ema, adjust=False).mean()
        
        factor = dbb_med * 4 / 5
        atl = dbb - factor
        al1 = np.where(atl > 0, 0, 1)
        
        return al1[-1] if len(al1) > 0 else 0
    except:
        return 0

def calculate_mic_value(prices):
    """Calcula el MIC Value: (ROC18/NATR18)*0.6 + (ROC50/NATR50)*0.4"""
    try:
        if len(prices) < 50:
            return None
        
        close = prices
        
        # ROC
        roc18 = ((close.iloc[-1] - close.iloc[-19]) / close.iloc[-19]) * 100 if len(close) > 18 else 0
        roc50 = ((close.iloc[-1] - close.iloc[-51]) / close.iloc[-51]) * 100 if len(close) > 50 else 0
        
        # ATR
        high = close  # Aproximación con close
        low = close
        
        tr1 = high - low
        tr = tr1.rolling(window=14).mean()
        
        natr18 = (tr.rolling(window=18).mean().iloc[-1] / close.iloc[-1]) * 100 if len(tr) > 18 else 1
        natr50 = (tr.rolling(window=50).mean().iloc[-1] / close.iloc[-1]) * 100 if len(tr) > 50 else 1
        
        if natr18 == 0:
            natr18 = 1
        if natr50 == 0:
            natr50 = 1
        
        mic_value = (roc18 / natr18) * 0.6 + (roc50 / natr50) * 0.4
        
        return mic_value
    except:
        return None

def calculate_sharpe_ratio(prices, period=50):
    """Calcula el Sharpe Ratio anualizado"""
    try:
        if len(prices) < period + 1:
            return None
        
        returns = prices.pct_change().dropna()
        
        if len(returns) < period:
            return None
        
        mean_return = returns.rolling(window=period).mean().iloc[-1]
        std_return = returns.rolling(window=period).std().iloc[-1]
        
        if std_return == 0 or pd.isna(std_return):
            return None
        
        sharpe = (mean_return / std_return) * np.sqrt(50)  # Anualizado (52 semanas)
        
        return sharpe
    except:
        return None

def calculate_macd_v(prices):
    """Calcula MACD-V normalizado por ATR"""
    try:
        if len(prices) < 26:
            return None
        
        ema_12 = prices.ewm(span=12, adjust=False).mean()
        ema_26 = prices.ewm(span=26, adjust=False).mean()
        
        # ATR aproximado
        high = prices
        low = prices
        tr = (high - low).rolling(window=14).mean()
        atr = tr.rolling(window=26).mean()
        
        if atr.iloc[-1] == 0 or pd.isna(atr.iloc[-1]):
            return None
        
        macd_line = ((ema_12.iloc[-1] - ema_26.iloc[-1]) / atr.iloc[-1]) * 100
        
        return macd_line
    except:
        return None

def analyze_ticker(ticker, params, benchmark_data, sector_etfs_data):
    """Analiza un ticker individual con todos los filtros"""
    try:
        # Descargar datos
        data = download_weekly_data(ticker, period="5y")
        if data is None or len(data) < 156:  # Mínimo 3 años
            return None
        
        close = data['Close']
        current_price = close.iloc[-1]
        
        # Obtener sector
        sector, sector_etf = get_sector_info(ticker)
        
        # ========== FILTRO 1: RSC Mansfield vs SPY > 0 ==========
        rsc = calculate_rsc_mansfield(close, benchmark_data, period=52)
        if rsc is None or rsc <= 0:
            return None
        
        # ========== FILTRO 2: % a Máximos ==========
        max_years = params['max_years']
        periods = min(52 * max_years, len(close))
        max_price = close.iloc[-periods:].max()
        dist_to_max = abs(current_price - max_price) / current_price * 100
        
        if dist_to_max > params['dist_to_max']:
            return None
        
        # ========== FILTRO 3: Close > WMA30 ==========
        wma30 = calculate_wma(close, period=30)
        if wma30 is None or current_price <= wma30:
            return None
        
        # ========== FILTRO 4: Distancia a WMA30 <= % ==========
        if params['apply_wma_dist']:
            dist_to_wma = abs(current_price - wma30) / current_price * 100
            if dist_to_wma > params['dist_to_wma']:
                return None
        else:
            dist_to_wma = abs(current_price - wma30) / current_price * 100
        
        # ========== FILTRO 5: RSC Sectorial > 0 (OBLIGATORIO si hay sector) ==========
        rsc_sector = None
        if sector_etf and sector_etf in sector_etfs_data:
            sector_data = sector_etfs_data[sector_etf]
            rsc_sector = calculate_rsc_mansfield(sector_data, benchmark_data, period=52)
            
            # Si hay sector, es OBLIGATORIO que RSC_Sector > 0
            if rsc_sector is not None and rsc_sector <= 0:
                return None
        
        # ========== FILTROS OPCIONALES ==========
        
        # Regresión Lineal
        slope, r_squared, normalized_dist = calculate_linear_regression(close, period=30)
        if params['apply_lr']:
            if r_squared is None or r_squared < 0.7 or normalized_dist is None or normalized_dist > 1.5:
                return None
        
        # Atlas
        atlas_value = calculate_atlas(close)
        if params['apply_atlas']:
            if atlas_value == 0:
                return None
        
        # MIC Ranking
        mic_value = calculate_mic_value(close)
        if params['apply_mic']:
            if mic_value is None or mic_value < 5:
                return None
        
        # Sharpe Ratio
        sharpe = calculate_sharpe_ratio(close)
        if params['apply_sharpe']:
            if sharpe is None or sharpe < 1.5:
                return None
        
        # MACD-V
        macd_v = calculate_macd_v(close)
        if params['apply_macd']:
            if macd_v is None or macd_v < 50:
                return None
        
        # ========== RESULTADO ==========
        result = {
            'Ticker': ticker,
            'Price': round(current_price, 2),
            'Sector': sector if sector else 'N/A',
            'RSC': round(rsc, 2) if rsc else None,
            'RSC_Sector': round(rsc_sector, 2) if rsc_sector else None,
            'Dist_Max_%': round(dist_to_max, 2),
            'WMA30': round(wma30, 2) if wma30 else None,
            'Dist_WMA_%': round(dist_to_wma, 2),
            'Slope': round(slope, 2) if slope else None,
            'R2': round(r_squared, 3) if r_squared else None,
            'Norm_Dist': round(normalized_dist, 2) if normalized_dist else None,
            'Atlas': int(atlas_value),
            'MIC_Value': round(mic_value, 2) if mic_value else None,
            'Sharpe': round(sharpe, 2) if sharpe else None,
            'MACD_V': round(macd_v, 2) if macd_v else None
        }
        
        return result
        
    except Exception as e:
        return None

def run_screener(tickers, params, progress_bar, status_text):
    """Ejecuta el screener en paralelo"""
    
    # Descargar benchmark (SPY)
    status_text.text("📊 Descargando datos del benchmark (^GSPC)...")
    benchmark_data_full = download_weekly_data("^GSPC", period="5y")
    
    if benchmark_data_full is None:
        st.error("❌ Error descargando datos del benchmark ^GSPC")
        return pd.DataFrame()
    
    benchmark_data = benchmark_data_full['Close']
    
    # Descargar ETFs sectoriales
    status_text.text("📊 Descargando ETFs sectoriales...")
    sector_etfs_data = {}
    unique_sector_etfs = list(set(SECTOR_ETFS.values()))
    
    for etf in unique_sector_etfs:
        data = download_weekly_data(etf, period="5y")
        if data is not None:
            sector_etfs_data[etf] = data['Close']
    
    # Procesar tickers en paralelo
    results = []
    total = len(tickers)
    
    status_text.text(f"🔍 Analizando {total} tickers...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(analyze_ticker, ticker, params, benchmark_data, sector_etfs_data): ticker 
            for ticker in tickers
        }
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            progress_bar.progress(completed / total)
            
            result = future.result()
            if result is not None:
                results.append(result)
            
            if completed % 50 == 0:
                status_text.text(f"🔍 Procesados: {completed}/{total} | Encontrados: {len(results)}")
    
    status_text.text(f"✅ Análisis completado: {len(results)} acciones encontradas")
    
    if len(results) == 0:
        return pd.DataFrame()
    
    df_results = pd.DataFrame(results)
    
    # Ordenar por RSC descendente
    df_results = df_results.sort_values('RSC', ascending=False).reset_index(drop=True)
    
    return df_results

# ============= INTERFAZ PRINCIPAL =============
def main():
    st.set_page_config(
        page_title="Trend Stocks Screener",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Trend Stocks Screener")
    st.markdown("**Detecta acciones con tendencias fuertes y sostenibles (Timeframe Semanal)**")
    st.markdown("---")
    
    # ============= PASO 1: UNIVERSO DE TICKERS =============
    st.markdown("### 📂 PASO 1: Universo de Tickers")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if 'tickers_universe' not in st.session_state or len(st.session_state.get('tickers_universe', [])) == 0:
            st.info("🔄 Cargando universo de tickers...")
            try:
                df_tickers = create_tickers_universe()
                if isinstance(df_tickers, pd.DataFrame):
                    tickers_list = df_tickers['Ticker'].astype(str).tolist()
                else:
                    tickers_list = list(df_tickers)
                st.session_state['tickers_universe'] = tickers_list
                st.session_state['random_seed'] = random.randint(1, 10000)
                st.success(f"✅ {len(tickers_list):,} tickers cargados")
            except Exception as e:
                st.error(f"❌ Error cargando tickers: {e}")
                st.session_state['tickers_universe'] = []
        else:
            st.success(f"✅ {len(st.session_state['tickers_universe']):,} tickers disponibles")
    
    with col2:
        if st.button("🔄 Recargar Tickers", use_container_width=True):
            if 'tickers_universe' in st.session_state:
                del st.session_state['tickers_universe']
            if 'random_seed' in st.session_state:
                del st.session_state['random_seed']
            st.rerun()
    
    st.markdown("---")
    
    # ============= PASO 2: CONFIGURACIÓN =============
    st.markdown("### ⚙️ PASO 2: Configuración de Parámetros")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Filtros Principales**")
        dist_to_max = st.slider(
            "% Distancia a Máximos",
            min_value=0,
            max_value=15,
            value=5,
            step=1,
            help="Porcentaje máximo de distancia al precio máximo"
        )
        
        max_years = st.selectbox(
            "Años para calcular Máximo",
            options=[1, 2, 3, 4, 5],
            index=2,
            help="Años históricos para buscar el máximo"
        )
        
        apply_wma_dist = st.checkbox(
            "Aplicar filtro distancia WMA30",
            value=True,
            help="Filtrar por distancia máxima a WMA30"
        )
        
        dist_to_wma = st.slider(
            "% Distancia a WMA30",
            min_value=0,
            max_value=20,
            value=9,
            step=1,
            help="Porcentaje máximo de distancia a WMA30",
            disabled=not apply_wma_dist
        )
    
    with col2:
        st.markdown("**🔧 Filtros Técnicos**")
        
        apply_lr = st.checkbox(
            "Regresión Lineal (R²≥0.7, Dist≤1.5)",
            value=True,
            help="Filtro de regresión lineal"
        )
        
        apply_mic = st.checkbox(
            "MIC Ranking (>5)",
            value=True,
            help="MIC = (ROC18/NATR18)*0.6 + (ROC50/NATR50)*0.4"
        )
        
        apply_sharpe = st.checkbox(
            "Sharpe Ratio (≥1.5)",
            value=True,
            help="Sharpe ratio anualizado"
        )
        
        apply_macd = st.checkbox(
            "MACD-V (≥50)",
            value=True,
            help="MACD normalizado por ATR"
        )
    
    with col3:
        st.markdown("**⚡ Filtros Especiales**")
        
        apply_atlas = st.checkbox(
            "Atlas Encendido",
            value=False,
            help="Indicador Atlas basado en Bandas de Bollinger"
        )
        
        st.markdown("---")
        st.markdown("**📋 Resumen Filtros Activos:**")
        
        filters_count = sum([
            True,  # RSC > 0 (siempre)
            True,  # Distancia a máximos (siempre)
            True,  # Close > WMA30 (siempre)
            apply_wma_dist,
            True,  # RSC Sectorial (siempre si aplica)
            apply_lr,
            apply_mic,
            apply_sharpe,
            apply_macd,
            apply_atlas
        ])
        
        st.info(f"**{filters_count}** filtros activos")
    
    st.markdown("---")
    
    # ============= PASO 3: ESCANEO =============
    st.markdown("### 🚀 PASO 3: Ejecutar Escaneo")
    
    scan_button = st.button(
        "🚀 INICIAR ESCANEO", 
        type="primary", 
        use_container_width=True,
        disabled=len(st.session_state.get('tickers_universe', [])) == 0
    )
    
    if scan_button:
        params = {
            'dist_to_max': dist_to_max,
            'max_years': max_years,
            'dist_to_wma': dist_to_wma,
            'apply_wma_dist': apply_wma_dist,
            'apply_lr': apply_lr,
            'apply_mic': apply_mic,
            'apply_sharpe': apply_sharpe,
            'apply_macd': apply_macd,
            'apply_atlas': apply_atlas
        }
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        tickers = st.session_state['tickers_universe']
        
        df_results = run_screener(tickers, params, progress_bar, status_text)
        
        progress_bar.empty()
        
        if len(df_results) > 0:
            st.session_state['scan_results'] = df_results
            st.session_state['scan_timestamp'] = datetime.now()
            st.success(f"✅ Escaneo completado: **{len(df_results)}** acciones encontradas")
        else:
            st.warning("⚠️ No se encontraron acciones que cumplan todos los criterios")
            if 'scan_results' in st.session_state:
                del st.session_state['scan_results']
    
    st.markdown("---")
    
    # ============= PASO 4: RESULTADOS =============
    st.markdown("### 📈 PASO 4: Resultados del Escaneo")
    
    if 'scan_results' in st.session_state and len(st.session_state['scan_results']) > 0:
        df_display = st.session_state['scan_results']
        
        # Métricas resumen
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Acciones Encontradas", len(df_display))
        with col2:
            avg_rsc = df_display['RSC'].mean()
            st.metric("📈 RSC Promedio", f"{avg_rsc:.2f}")
        with col3:
            avg_sharpe = df_display['Sharpe'].mean() if 'Sharpe' in df_display.columns else 0
            st.metric("⚡ Sharpe Promedio", f"{avg_sharpe:.2f}")
        with col4:
            timestamp = st.session_state.get('scan_timestamp', datetime.now())
            st.metric("🕐 Último Escaneo", timestamp.strftime("%H:%M:%S"))
        
        st.markdown("---")
        
        # Tabla de resultados
        st.dataframe(
            df_display,
            use_container_width=True,
            height=600,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                "Price": st.column_config.NumberColumn("Precio", format="$%.2f"),
                "Sector": st.column_config.TextColumn("Sector", width="medium"),
                "RSC": st.column_config.NumberColumn("RSC", format="%.2f"),
                "RSC_Sector": st.column_config.NumberColumn("RSC Sector", format="%.2f"),
                "Dist_Max_%": st.column_config.NumberColumn("Dist Max %", format="%.2f%%"),
                "WMA30": st.column_config.NumberColumn("WMA30", format="%.2f"),
                "Dist_WMA_%": st.column_config.NumberColumn("Dist WMA %", format="%.2f%%"),
                "Slope": st.column_config.NumberColumn("Slope", format="%.2f"),
                "R2": st.column_config.NumberColumn("R²", format="%.3f"),
                "Norm_Dist": st.column_config.NumberColumn("Norm Dist", format="%.2f"),
                "Atlas": st.column_config.NumberColumn("Atlas", format="%d"),
                "MIC_Value": st.column_config.NumberColumn("MIC Value", format="%.2f"),
                "Sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                "MACD_V": st.column_config.NumberColumn("MACD-V", format="%.2f")
            }
        )
        
        # Botón de descarga
        csv = df_display.to_csv(index=False)
        st.download_button(
            label="📥 Descargar Resultados (CSV)",
            data=csv,
            file_name=f"trend_stocks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    else:
        st.info("🚧 Los resultados aparecerán aquí una vez completado el escaneo")

if __name__ == "__main__":
    if check_password():
        main()
    else:
        st.markdown(
            """
            <div style='text-align: center; padding: 60px 20px;'>
                <h1 style='color: #FF6B6B; font-size: 48px;'>🔒 Acceso Restringido</h1>
                <p style='color: #B0B0B0; font-size: 20px; margin-top: 20px;'>
                    Introduce tus credenciales en el menú lateral.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
