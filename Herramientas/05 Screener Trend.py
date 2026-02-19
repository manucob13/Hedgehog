import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta, date
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import random
import pandas_ta as ta
import plotly.graph_objects as go
from utils.utils import check_password
from utils.tickers import (
    create_tickers_universe,
    get_all_index_names,
    get_all_etf_names,
    get_index_name,
    get_etf_name
)

warnings.filterwarnings('ignore')

# Lock global para sincronizar descargas de yfinance
_yfinance_lock = Lock()

# ============= FUNCIONES DE CÁLCULO =============

def get_ticker_name_and_marketcap(ticker):
    """
    Obtiene el nombre y market cap del ticker
    Prioridad: 1) Índices, 2) ETFs, 3) Acciones (yfinance)
    """
    try:
        # 1. Verificar si es un ÍNDICE
        index_name = get_index_name(ticker)
        if index_name is not None:
            return index_name, None  # Los índices no tienen market cap
        
        # 2. Verificar si es un ETF
        etf_name = get_etf_name(ticker)
        if etf_name is not None:
            # Para ETFs, intentar obtener market cap de yfinance
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                market_cap = info.get('marketCap', None)
                return etf_name, market_cap
            except:
                return etf_name, None
        
        # 3. Si no es índice ni ETF, es una ACCIÓN - usar yfinance
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Intentar múltiples campos para el nombre
        name = (info.get('longName') or 
                info.get('shortName') or 
                info.get('name') or
                info.get('quoteType', '') + ' - ' + ticker if info.get('quoteType') else ticker)
        
        # Si aún no hay nombre, intentar con el ticker directamente
        if name == ticker or not name:
            name = info.get('shortName', ticker)
        
        market_cap = info.get('marketCap', None)
        
        return name, market_cap
        
    except:
        return ticker, None

def get_next_dividend_date(ticker):
    """
    Obtiene la fecha del próximo dividendo para un ticker.
    Retorna: fecha del próximo dividendo (str) o None
    """
    try:
        stock = yf.Ticker(ticker)
        
        calendar = stock.calendar
        
        if calendar is not None and 'Ex-Dividend Date' in calendar:
            ex_div_date = calendar['Ex-Dividend Date']
            
            if pd.notna(ex_div_date):
                if isinstance(ex_div_date, str):
                    ex_div_date = pd.to_datetime(ex_div_date)
                
                if ex_div_date > pd.Timestamp.now():
                    return ex_div_date.strftime('%Y-%m-%d')
        
        dividends = stock.dividends
        
        if dividends is not None and not dividends.empty:
            recent_divs = dividends.tail(4)
            
            if len(recent_divs) >= 2:
                dates = recent_divs.index
                intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
                avg_interval = np.mean(intervals)
                
                last_div_date = dividends.index[-1]
                next_div_estimate = last_div_date + pd.Timedelta(days=avg_interval)
                
                if next_div_estimate > pd.Timestamp.now():
                    return next_div_estimate.strftime('%Y-%m-%d') + ' (Est.)'
        
        return "N/A"
        
    except Exception as e:
        return "N/A"

def get_dividend_yield(ticker):
    """
    Obtiene el dividend yield de un ticker
    Retorna: yield en porcentaje (float) o None
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        div_yield = info.get('dividendYield', None)
        
        if div_yield is not None and div_yield > 0:
            return round(div_yield * 100, 2)
        
        return None
        
    except:
        return None

def download_weekly_data(ticker, period="5y", use_lock=True):
    """Descarga datos semanales para un ticker"""
    try:
        end = datetime.now() + timedelta(days=1)
        
        period_days = {
            '1y': 365,
            '2y': 730,
            '3y': 1095,
            '5y': 1825,
            '10y': 3650
        }
        
        days = period_days.get(period, 1825)
        start = end - timedelta(days=days)
        
        if use_lock:
            with _yfinance_lock:
                data = yf.download(
                    ticker, 
                    start=start, 
                    end=end, 
                    interval="1wk", 
                    auto_adjust=False, 
                    multi_level_index=False, 
                    progress=False
                )
        else:
            data = yf.download(
                ticker, 
                start=start, 
                end=end, 
                interval="1wk", 
                auto_adjust=False, 
                multi_level_index=False, 
                progress=False
            )
        
        if data is None or data.empty or len(data) < 52:
            return None
        
        data.index = pd.to_datetime(data.index)
        return data
        
    except:
        return None

def calculate_rsc_mansfield(prices, benchmark_prices, period=52):
    """
    Calcula el RSC Mansfield
    RSC = ((Ratio / Media_Ratio) - 1) * 10
    """
    try:
        if len(prices) < period or len(benchmark_prices) < period:
            return None
        
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
    """Calcula la media móvil ponderada usando pandas_ta"""
    try:
        if len(prices) < period:
            return None
        
        wma = ta.wma(prices, length=period)
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
        
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]
        
        lin_reg = np.polyval(coeffs, x)
        
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        ss_res = np.sum((y - lin_reg) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        se_regression = np.sqrt(ss_res / (period - 2)) if period > 2 else 0
        
        distance = abs(y[-1] - lin_reg[-1])
        normalized_distance = distance / se_regression if se_regression > 0 else 0
        
        return slope, r_squared, normalized_distance
    except:
        return None, None, None

def calculate_atlas(prices, period_bb=20, period_ema=120):
    """Calcula el indicador Atlas usando pandas_ta"""
    try:
        if len(prices) < max(period_bb, period_ema):
            return 0
        
        bbands = ta.bbands(prices, length=period_bb, std=2)
        
        if bbands is None or bbands.empty:
            return 0
        
        bb_top = bbands[f'BBU_{period_bb}_2.0']
        bb_bot = bbands[f'BBL_{period_bb}_2.0']
        
        dbb = np.sqrt((bb_top - bb_bot) / bb_top) * 20
        dbb_med = dbb.ewm(span=period_ema, adjust=False).mean()
        
        factor = dbb_med * 4 / 5
        atl = dbb - factor
        al1 = np.where(atl > 0, 0, 1)
        
        return al1[-1] if len(al1) > 0 else 0
    except:
        return 0

def calculate_mic_value(data):
    """Calcula el MIC Value: (ROC18/NATR18)*0.6 + (ROC50/NATR50)*0.4 usando pandas_ta"""
    try:
        if len(data) < 50:
            return None
        
        close = data['Close']
        high = data['High']
        low = data['Low']
        
        roc18 = ta.roc(close, length=18)
        roc50 = ta.roc(close, length=50)
        
        if roc18 is None or roc50 is None:
            return None
        
        roc18_val = roc18.iloc[-1] if not pd.isna(roc18.iloc[-1]) else 0
        roc50_val = roc50.iloc[-1] if not pd.isna(roc50.iloc[-1]) else 0
        
        natr18 = ta.natr(high=high, low=low, close=close, length=18)
        natr50 = ta.natr(high=high, low=low, close=close, length=50)
        
        if natr18 is None or natr50 is None:
            return None
        
        natr18_val = natr18.iloc[-1] if not pd.isna(natr18.iloc[-1]) else 1
        natr50_val = natr50.iloc[-1] if not pd.isna(natr50.iloc[-1]) else 1
        
        if natr18_val == 0:
            natr18_val = 1
        if natr50_val == 0:
            natr50_val = 1
        
        mic_value = (roc18_val / natr18_val) * 0.6 + (roc50_val / natr50_val) * 0.4
        
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
        
        sharpe = (mean_return / std_return) * np.sqrt(50)
        
        return sharpe
    except:
        return None

def calculate_macd_v(data):
    """Calcula MACD-V normalizado por ATR usando pandas_ta"""
    try:
        if len(data) < 26:
            return None
        
        close = data['Close']
        high = data['High']
        low = data['Low']
        
        atr = ta.atr(high=high, low=low, close=close, length=26)
        
        if atr is None or pd.isna(atr.iloc[-1]) or atr.iloc[-1] == 0:
            return None
        
        macd = ta.macd(close, fast=12, slow=26, signal=9)
        
        if macd is None or macd.empty:
            return None
        
        macd_line = macd['MACD_12_26_9']
        
        if pd.isna(macd_line.iloc[-1]):
            return None
        
        macd_v = (macd_line.iloc[-1] / atr.iloc[-1]) * 100
        
        return macd_v
    except:
        return None


# ============= FUNCIONES DE OPCIONES (MEJORADAS) =============

def is_monthly_expiration(date_str):
    """
    Determina si una fecha de vencimiento es un vencimiento MENSUAL estándar.
    Los vencimientos mensuales son el 3er viernes de cada mes.
    
    Args:
        date_str (str): Fecha en formato 'YYYY-MM-DD'
    
    Returns:
        bool: True si es vencimiento mensual
    """
    try:
        exp_date = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Verificar que sea viernes (weekday() == 4)
        if exp_date.weekday() != 4:
            return False
        
        # Verificar que sea el 3er viernes del mes
        # El 3er viernes cae entre el día 15 y el 21
        if 15 <= exp_date.day <= 21:
            return True
        
        return False
    except:
        return False


def get_target_monthly_expiration(expirations, min_dte=25):
    """
    Encuentra el vencimiento mensual más cercano con al menos min_dte días.
    
    Lógica:
    1. Filtrar solo vencimientos mensuales (3er viernes del mes)
    2. De esos, quedarse con el más cercano que tenga >= min_dte días
    3. Si ninguno cumple min_dte, devolver el primer mensual futuro disponible
    
    Args:
        expirations (tuple): Tupla de strings con fechas de vencimiento 'YYYY-MM-DD'
        min_dte (int): Días mínimos hasta el vencimiento (default: 25)
    
    Returns:
        str o None: Fecha del vencimiento objetivo, o None si no hay
    """
    today = datetime.now().date()
    
    monthly_expirations = []
    
    for exp_str in expirations:
        try:
            exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
            
            # Solo fechas futuras
            if exp_date <= today:
                continue
            
            dte = (exp_date - today).days
            
            # Verificar si es vencimiento mensual
            if is_monthly_expiration(exp_str):
                monthly_expirations.append({
                    'date': exp_str,
                    'dte': dte,
                    'exp_date': exp_date
                })
        except:
            continue
    
    if not monthly_expirations:
        return None
    
    # Ordenar por DTE ascendente
    monthly_expirations.sort(key=lambda x: x['dte'])
    
    # Buscar el más cercano con >= min_dte días
    for exp in monthly_expirations:
        if exp['dte'] >= min_dte:
            return exp['date']
    
    # Si ninguno cumple min_dte, devolver el primero disponible (fallback)
    return monthly_expirations[0]['date'] if monthly_expirations else None


def get_options_metrics_yf(ticker, current_price, price_range_pct=10, min_dte=25):
    """
    Obtiene métricas de volumen y Open Interest de opciones desde Yahoo Finance.
    
    Usa el vencimiento mensual más cercano con >= min_dte días para capturar
    el flujo institucional real (Smart Money), evitando el ruido de weeklies.
    
    Args:
        ticker (str): Símbolo del ticker
        current_price (float): Precio actual de la acción
        price_range_pct (int): Porcentaje de rango de strikes alrededor del precio (default: 10%)
        min_dte (int): DTE mínimo para el vencimiento mensual (default: 25)
    
    Returns:
        dict con métricas o None si no hay datos
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Verificar si tiene opciones disponibles
        expirations = stock.options
        
        if not expirations or len(expirations) == 0:
            return None
        
        # ---- NUEVO: Buscar el vencimiento mensual objetivo ----
        target_expiration = get_target_monthly_expiration(expirations, min_dte=min_dte)
        
        if target_expiration is None:
            return None
        
        # Calcular DTE para info
        today = datetime.now().date()
        exp_date = datetime.strptime(target_expiration, '%Y-%m-%d').date()
        dte = (exp_date - today).days
        
        # Calcular rango de strikes (±price_range_pct%)
        lower_bound = current_price * (1 - price_range_pct / 100)
        upper_bound = current_price * (1 + price_range_pct / 100)
        
        # ---- Obtener la cadena de opciones para el vencimiento mensual ----
        try:
            chain = stock.option_chain(target_expiration)
        except Exception:
            return None
        
        # ----- CALLS -----
        calls = chain.calls
        calls_in_range = calls[
            (calls['strike'] >= lower_bound) &
            (calls['strike'] <= upper_bound)
        ]
        
        call_volume = calls_in_range['volume'].fillna(0).sum()
        call_oi     = calls_in_range['openInterest'].fillna(0).sum()
        
        # ----- PUTS -----
        puts = chain.puts
        puts_in_range = puts[
            (puts['strike'] >= lower_bound) &
            (puts['strike'] <= upper_bound)
        ]
        
        put_volume = puts_in_range['volume'].fillna(0).sum()
        put_oi     = puts_in_range['openInterest'].fillna(0).sum()
        
        # ----- Totales -----
        total_volume = call_volume + put_volume
        total_oi     = call_oi + put_oi
        
        # Si no hay datos significativos, retornar None
        if total_volume == 0 and total_oi == 0:
            return None
        
        # ----- Put/Call Ratio (Volumen) -----
        if call_volume == 0:
            pc_ratio_vol = None  # No divisible
        else:
            pc_ratio_vol = round(put_volume / call_volume, 2)
        
        # ----- Put/Call Ratio (Open Interest) -----
        if call_oi == 0:
            pc_ratio_oi = None
        else:
            pc_ratio_oi = round(put_oi / call_oi, 2)
        
        # ----- Sentiment basado en OI (más fiable) -----
        ratio_for_sentiment = pc_ratio_oi if pc_ratio_oi is not None else pc_ratio_vol
        
        if ratio_for_sentiment is None:
            sentiment = "⚪ Sin datos"
        elif ratio_for_sentiment < 0.7:
            sentiment = "🟢 Muy Alcista"
        elif ratio_for_sentiment < 1.0:
            sentiment = "🟢 Alcista"
        elif ratio_for_sentiment < 1.3:
            sentiment = "🟡 Neutral"
        else:
            sentiment = "🔴 Bajista"
        
        return {
            # Vencimiento usado
            'Exp_Date':     target_expiration,
            'DTE':          dte,
            # Volumen
            'Call_Vol':     int(call_volume),
            'Put_Vol':      int(put_volume),
            'Total_Vol':    int(total_volume),
            'PC_Ratio_Vol': pc_ratio_vol,
            # Open Interest
            'Call_OI':      int(call_oi),
            'Put_OI':       int(put_oi),
            'Total_OI':     int(total_oi),
            'PC_Ratio_OI':  pc_ratio_oi,
            # Sentiment
            'Sentiment':    sentiment
        }
        
    except Exception:
        return None


def format_market_cap(value):
    """Formatea el market cap de forma legible"""
    if value is None or pd.isna(value):
        return "N/A"
    
    if value >= 1e12:
        return f"${value/1e12:.2f}T"
    elif value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.2f}M"
    else:
        return f"${value:,.0f}"

def analyze_ticker(ticker, params, benchmark_data):
    """Analiza un ticker individual con todos los filtros"""
    try:
        data = download_weekly_data(ticker, period="5y")
        if data is None or len(data) < 156:
            return None
        
        close = data['Close']
        current_price = close.iloc[-1]
        
        # ========== FILTRO 0: Precio máximo ==========
        if current_price > params['max_price']:
            return None
        
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
        dist_to_wma = abs(current_price - wma30) / current_price * 100
        if params['apply_wma_dist'] and dist_to_wma > params['dist_to_wma']:
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
        mic_value = calculate_mic_value(data)
        if params['apply_mic']:
            if mic_value is None or mic_value < 5:
                return None
        
        # Sharpe Ratio
        sharpe = calculate_sharpe_ratio(close)
        if params['apply_sharpe']:
            if sharpe is None or sharpe < 1.5:
                return None
        
        # MACD-V
        macd_v = calculate_macd_v(data)
        macd_filter = params['macd_filter']
        
        if macd_filter == "MACD-V ≥ 50":
            if macd_v is None or macd_v < 50:
                return None
        elif macd_filter == "MACD-V entre 50-150":
            if macd_v is None or macd_v < 50 or macd_v >= 150:
                return None
        elif macd_filter == "MACD-V ≥ 150":
            if macd_v is None or macd_v < 150:
                return None
        
        # ========== TODOS LOS FILTROS PASADOS ==========
        name, market_cap = get_ticker_name_and_marketcap(ticker)
        next_dividend = get_next_dividend_date(ticker)
        div_yield = get_dividend_yield(ticker)
        
        result = {
            'Ticker':        ticker,
            'Name':          name,
            'Price':         round(current_price, 2),
            'Market_Cap':    market_cap,
            'Next_Dividend': next_dividend,
            'Div_Yield_%':   div_yield,
            'RSC':           round(rsc, 2) if rsc else None,
            'Dist_Max_%':    round(dist_to_max, 2),
            'WMA30':         round(wma30, 2) if wma30 else None,
            'Dist_WMA_%':    round(dist_to_wma, 2),
            'Slope':         round(slope, 2) if slope else None,
            'R2':            round(r_squared, 3) if r_squared else None,
            'Norm_Dist':     round(normalized_dist, 2) if normalized_dist else None,
            'Atlas':         int(atlas_value),
            'MIC_Value':     round(mic_value, 2) if mic_value else None,
            'Sharpe':        round(sharpe, 2) if sharpe else None,
            'MACD_V':        round(macd_v, 2) if macd_v else None
        }
        
        return result
        
    except:
        return None

def run_screener(tickers, params, progress_bar, status_text):
    """Ejecuta el screener en paralelo"""
    
    status_text.text("📊 Descargando datos del benchmark (SPY)...")
    benchmark_data_full = download_weekly_data("SPY", period="5y", use_lock=False)
    
    if benchmark_data_full is None:
        st.error("❌ Error descargando datos del benchmark SPY. Verifica tu conexión a internet.")
        return pd.DataFrame()
    
    benchmark_data = benchmark_data_full['Close']
    
    results = []
    total = len(tickers)
    
    status_text.text(f"🔍 Analizando {total} tickers...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(analyze_ticker, ticker, params, benchmark_data): ticker 
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
    df_results = df_results.sort_values('RSC', ascending=False).reset_index(drop=True)
    
    return df_results

def plot_candlestick_chart(ticker, ticker_name, period="1y"):
    """Crea un gráfico de velas para un ticker"""
    try:
        data = download_weekly_data(ticker, period=period, use_lock=False)
        
        if data is None or data.empty:
            st.error(f"No se pudieron cargar datos para {ticker}")
            return
        
        wma30 = ta.wma(data['Close'], length=30)
        
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=ticker,
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ))
        
        if wma30 is not None:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=wma30,
                mode='lines',
                name='WMA30',
                line=dict(color='#FFA726', width=2)
            ))
        
        fig.update_layout(
            title=f'{ticker} - {ticker_name} - Gráfico Semanal',
            yaxis_title='Precio ($)',
            xaxis_title='Fecha',
            template='plotly_dark',
            height=500,
            xaxis_rangeslider_visible=False,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error al crear gráfico para {ticker}: {str(e)}")

# ============= INTERFAZ PRINCIPAL =============
def main():
    st.set_page_config(
        page_title="Trend Stocks Screener",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("📈 Trend Stocks Screener")
    st.markdown("**Detecta acciones con tendencias fuertes y sostenibles (Timeframe Semanal vs SPY)**")
    
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
        
        max_price = st.slider(
            "Precio Máximo ($)",
            min_value=25,
            max_value=1000,
            value=500,
            step=25,
            help="Precio máximo de la acción en múltiplos de $25"
        )
        
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
            index=1,
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
        
        st.markdown("**📊 Filtros MACD-V**")
        
        macd_filter = st.radio(
            "Selecciona filtro MACD-V:",
            options=["Sin filtro", "MACD-V ≥ 50", "MACD-V entre 50-150", "MACD-V ≥ 150"],
            index=2,
            help="Filtra por valores de MACD-V normalizado por ATR"
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
            True,  # Precio máximo (siempre)
            True,  # RSC > 0 (siempre)
            True,  # Distancia a máximos (siempre)
            True,  # Close > WMA30 (siempre)
            apply_wma_dist,
            apply_lr,
            apply_mic,
            apply_sharpe,
            1 if macd_filter != "Sin filtro" else 0,
            apply_atlas
        ])
        
        st.info(f"**{filters_count}** filtros activos")
        st.success(f"Precio máx: **${max_price}**")
        if macd_filter != "Sin filtro":
            st.success(f"MACD-V: {macd_filter.replace('MACD-V ', '')}")
    
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
            'max_price':     max_price,
            'dist_to_max':   dist_to_max,
            'max_years':     max_years,
            'dist_to_wma':   dist_to_wma,
            'apply_wma_dist': apply_wma_dist,
            'apply_lr':      apply_lr,
            'apply_mic':     apply_mic,
            'apply_sharpe':  apply_sharpe,
            'macd_filter':   macd_filter,
            'apply_atlas':   apply_atlas
        }
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        tickers = st.session_state['tickers_universe']
        
        df_results = run_screener(tickers, params, progress_bar, status_text)
        
        progress_bar.empty()
        
        if len(df_results) > 0:
            st.session_state['scan_results'] = df_results
            st.session_state['scan_timestamp'] = datetime.now()
            if 'options_calculated' in st.session_state:
                del st.session_state['options_calculated']
            st.success(f"✅ Escaneo completado: **{len(df_results)}** acciones encontradas")
        else:
            st.warning("⚠️ No se encontraron acciones que cumplan todos los criterios")
            if 'scan_results' in st.session_state:
                del st.session_state['scan_results']
    
    st.markdown("---")
    
    # ============= PASO 4: RESULTADOS =============
    st.markdown("### 📈 PASO 4: Resultados del Escaneo")
    
    if 'scan_results' in st.session_state and len(st.session_state['scan_results']) > 0:
        df_display = st.session_state['scan_results'].copy()
        
        # Formatear Market Cap para display
        df_display['Market_Cap_Formatted'] = df_display['Market_Cap'].apply(format_market_cap)
        
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
        
        # ============= SECCIÓN DE OPCIONES =============
        st.markdown("### 📊 Volumen de Opciones (Smart Money)")
        
        st.info(
            "💡 **Metodología:** Se usa el **vencimiento mensual más cercano con ≥25 DTE** (3er viernes del mes) "
            "para capturar flujo institucional real. Strikes filtrados a ±10% del precio actual.\n\n"
            "**Put/Call Ratio (OI):** < 0.7 = 🟢 Muy Alcista | 0.7-1.0 = 🟢 Alcista | "
            "1.0-1.3 = 🟡 Neutral | > 1.3 = 🔴 Bajista"
        )
        
        col_opt1, col_opt2 = st.columns([3, 1])
        
        with col_opt1:
            if 'options_calculated' not in st.session_state or not st.session_state['options_calculated']:
                st.warning("⚠️ Aún no se ha calculado el volumen de opciones para estos tickers")
            else:
                st.success(f"✅ Opciones calculadas — vencimientos mensuales con ≥25 DTE")
        
        with col_opt2:
            calculate_options_btn = st.button(
                "📊 Calcular Opciones",
                type="primary",
                use_container_width=True,
                disabled='options_calculated' in st.session_state and st.session_state['options_calculated']
            )
        
        if calculate_options_btn:
            st.markdown("---")
            st.info("🔄 Calculando opciones... Esto puede tomar algunos minutos.")
            
            progress_opt = st.progress(0)
            status_opt = st.empty()
            
            total_tickers = len(df_display)
            options_data = []
            
            for idx, row in enumerate(df_display.itertuples(), 1):
                ticker        = row.Ticker
                current_price = row.Price
                
                progress_opt.progress(idx / total_tickers)
                status_opt.text(f"🔍 Analizando opciones: {ticker} ({idx}/{total_tickers})")
                
                metrics = get_options_metrics_yf(
                    ticker,
                    current_price,
                    price_range_pct=10,
                    min_dte=25
                )
                
                if metrics:
                    options_data.append({
                        'Ticker':       ticker,
                        'Exp_Date':     metrics['Exp_Date'],
                        'DTE':          metrics['DTE'],
                        # Volumen
                        'Call_Vol':     metrics['Call_Vol'],
                        'Put_Vol':      metrics['Put_Vol'],
                        'Total_Vol':    metrics['Total_Vol'],
                        'PC_Ratio_Vol': metrics['PC_Ratio_Vol'],
                        # Open Interest
                        'Call_OI':      metrics['Call_OI'],
                        'Put_OI':       metrics['Put_OI'],
                        'Total_OI':     metrics['Total_OI'],
                        'PC_Ratio_OI':  metrics['PC_Ratio_OI'],
                        # Sentiment
                        'Sentiment':    metrics['Sentiment']
                    })
                else:
                    options_data.append({
                        'Ticker':       ticker,
                        'Exp_Date':     'N/A',
                        'DTE':          None,
                        'Call_Vol':     None,
                        'Put_Vol':      None,
                        'Total_Vol':    None,
                        'PC_Ratio_Vol': None,
                        'Call_OI':      None,
                        'Put_OI':       None,
                        'Total_OI':     None,
                        'PC_Ratio_OI':  None,
                        'Sentiment':    'N/A'
                    })
            
            progress_opt.empty()
            status_opt.empty()
            
            df_options = pd.DataFrame(options_data)
            df_display = df_display.merge(df_options, on='Ticker', how='left')
            
            st.session_state['scan_results'] = df_display
            st.session_state['options_calculated'] = True
            
            st.success("✅ Opciones calculadas exitosamente")
            st.rerun()
        
        st.markdown("---")
        
        # ============= TABLA DE RESULTADOS =============
        has_options = 'Options_Vol' in df_display.columns or 'Total_Vol' in df_display.columns
        
        if 'Total_Vol' in df_display.columns:
            # Con datos de opciones completos
            df_table = df_display[[
                'Ticker', 'Name', 'Price', 'Market_Cap_Formatted', 'Next_Dividend', 'Div_Yield_%',
                'Exp_Date', 'DTE',
                'Call_Vol', 'Put_Vol', 'Total_Vol', 'PC_Ratio_Vol',
                'Call_OI',  'Put_OI',  'Total_OI',  'PC_Ratio_OI',
                'Sentiment',
                'RSC', 'Dist_Max_%', 'WMA30', 'Dist_WMA_%',
                'Slope', 'R2', 'Norm_Dist', 'Atlas', 'MIC_Value', 'Sharpe', 'MACD_V'
            ]]
            
            column_config = {
                "Ticker":           st.column_config.TextColumn("Ticker", width="small"),
                "Name":             st.column_config.TextColumn("Nombre", width="medium"),
                "Price":            st.column_config.NumberColumn("Precio", format="$%.2f"),
                "Market_Cap_Formatted": st.column_config.TextColumn("Market Cap", width="small"),
                "Next_Dividend":    st.column_config.TextColumn("Próx. Dividendo", width="medium"),
                "Div_Yield_%":      st.column_config.NumberColumn("Yield %", format="%.2f%%"),
                # Vencimiento
                "Exp_Date":         st.column_config.TextColumn("Venc. Mensual", width="small",
                                        help="Vencimiento mensual usado (3er viernes, ≥25 DTE)"),
                "DTE":              st.column_config.NumberColumn("DTE", format="%d",
                                        help="Días hasta el vencimiento"),
                # Volumen
                "Call_Vol":         st.column_config.NumberColumn("Call Vol", format="%d"),
                "Put_Vol":          st.column_config.NumberColumn("Put Vol", format="%d"),
                "Total_Vol":        st.column_config.NumberColumn("Total Vol", format="%d"),
                "PC_Ratio_Vol":     st.column_config.NumberColumn("P/C Vol", format="%.2f",
                                        help="Put/Call Ratio por Volumen"),
                # Open Interest
                "Call_OI":          st.column_config.NumberColumn("Call OI", format="%d"),
                "Put_OI":           st.column_config.NumberColumn("Put OI", format="%d"),
                "Total_OI":         st.column_config.NumberColumn("Total OI", format="%d"),
                "PC_Ratio_OI":      st.column_config.NumberColumn("P/C OI", format="%.2f",
                                        help="Put/Call Ratio por Open Interest (más fiable)"),
                # Sentiment
                "Sentiment":        st.column_config.TextColumn("Sentiment", width="medium"),
                # Técnicos
                "RSC":              st.column_config.NumberColumn("RSC", format="%.2f"),
                "Dist_Max_%":       st.column_config.NumberColumn("Dist Max %", format="%.2f%%"),
                "WMA30":            st.column_config.NumberColumn("WMA30", format="%.2f"),
                "Dist_WMA_%":       st.column_config.NumberColumn("Dist WMA %", format="%.2f%%"),
                "Slope":            st.column_config.NumberColumn("Slope", format="%.2f"),
                "R2":               st.column_config.NumberColumn("R²", format="%.3f"),
                "Norm_Dist":        st.column_config.NumberColumn("Norm Dist", format="%.2f"),
                "Atlas":            st.column_config.NumberColumn("Atlas", format="%d"),
                "MIC_Value":        st.column_config.NumberColumn("MIC Value", format="%.2f"),
                "Sharpe":           st.column_config.NumberColumn("Sharpe", format="%.2f"),
                "MACD_V":           st.column_config.NumberColumn("MACD-V", format="%.2f")
            }
        else:
            # Sin datos de opciones
            df_table = df_display[[
                'Ticker', 'Name', 'Price', 'Market_Cap_Formatted', 'Next_Dividend', 'Div_Yield_%',
                'RSC', 'Dist_Max_%', 'WMA30', 'Dist_WMA_%',
                'Slope', 'R2', 'Norm_Dist', 'Atlas', 'MIC_Value', 'Sharpe', 'MACD_V'
            ]]
            
            column_config = {
                "Ticker":           st.column_config.TextColumn("Ticker", width="small"),
                "Name":             st.column_config.TextColumn("Nombre", width="medium"),
                "Price":            st.column_config.NumberColumn("Precio", format="$%.2f"),
                "Market_Cap_Formatted": st.column_config.TextColumn("Market Cap", width="small"),
                "Next_Dividend":    st.column_config.TextColumn("Próx. Dividendo", width="medium"),
                "Div_Yield_%":      st.column_config.NumberColumn("Yield %", format="%.2f%%"),
                "RSC":              st.column_config.NumberColumn("RSC", format="%.2f"),
                "Dist_Max_%":       st.column_config.NumberColumn("Dist Max %", format="%.2f%%"),
                "WMA30":            st.column_config.NumberColumn("WMA30", format="%.2f"),
                "Dist_WMA_%":       st.column_config.NumberColumn("Dist WMA %", format="%.2f%%"),
                "Slope":            st.column_config.NumberColumn("Slope", format="%.2f"),
                "R2":               st.column_config.NumberColumn("R²", format="%.3f"),
                "Norm_Dist":        st.column_config.NumberColumn("Norm Dist", format="%.2f"),
                "Atlas":            st.column_config.NumberColumn("Atlas", format="%d"),
                "MIC_Value":        st.column_config.NumberColumn("MIC Value", format="%.2f"),
                "Sharpe":           st.column_config.NumberColumn("Sharpe", format="%.2f"),
                "MACD_V":           st.column_config.NumberColumn("MACD-V", format="%.2f")
            }
        
        st.dataframe(
            df_table,
            use_container_width=True,
            height=400,
            column_config=column_config
        )
        
        # Botón de descarga
        csv_df = df_display.drop('Market_Cap_Formatted', axis=1)
        csv = csv_df.to_csv(index=False)
        st.download_button(
            label="📥 Descargar Resultados (CSV)",
            data=csv,
            file_name=f"trend_stocks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.markdown("---")
        
        # ============= PASO 5: GRÁFICOS =============
        st.markdown("### 📊 PASO 5: Gráficos de Velas")
        
        st.info("📌 Selecciona el período de tiempo y los tickers que deseas visualizar")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            chart_period = st.selectbox(
                "Período del gráfico:",
                options=['1y', '2y', '3y', '5y'],
                index=0,
                format_func=lambda x: {
                    '1y': '1 Año',
                    '2y': '2 Años',
                    '3y': '3 Años',
                    '5y': '5 Años'
                }[x]
            )
            
            st.markdown("---")
            
            ticker_options = [f"{row['Ticker']} - {row['Name']}" for _, row in df_display.iterrows()]
            ticker_map = {f"{row['Ticker']} - {row['Name']}": row['Ticker'] for _, row in df_display.iterrows()}
            name_map = {row['Ticker']: row['Name'] for _, row in df_display.iterrows()}
            
            selected_ticker_option = st.selectbox(
                "Selecciona un ticker:",
                options=["Ninguno"] + ticker_options,
                index=0
            )
            
            if selected_ticker_option != "Ninguno":
                selected_ticker = ticker_map[selected_ticker_option]
            else:
                selected_ticker = None
            
            st.markdown("---")
            
            show_all = st.checkbox("Mostrar todos los gráficos", value=False)
            
            if not show_all and selected_ticker is None:
                num_charts = st.slider(
                    "Número de gráficos:",
                    min_value=1,
                    max_value=min(20, len(df_display)),
                    value=min(5, len(df_display)),
                    step=1
                )
        
        with col2:
            if selected_ticker is not None:
                tickers_to_plot = [selected_ticker]
                st.info(f"📊 Mostrando gráfico para: **{selected_ticker} - {name_map[selected_ticker]}**")
            elif show_all:
                tickers_to_plot = df_display['Ticker'].tolist()
                st.warning(f"⚠️ Se mostrarán {len(tickers_to_plot)} gráficos. Esto puede tardar un momento...")
            else:
                tickers_to_plot = df_display['Ticker'].head(num_charts).tolist()
            
            if st.button("📊 GENERAR GRÁFICOS", type="primary", use_container_width=True):
                st.markdown("---")
                
                for i, ticker in enumerate(tickers_to_plot, 1):
                    ticker_name = name_map.get(ticker, ticker)
                    st.markdown(f"#### {i}. {ticker} - {ticker_name}")
                    
                    ticker_data = df_display[df_display['Ticker'] == ticker].iloc[0]
                    
                    # Mostrar métricas del ticker
                    has_options_data = 'Total_Vol' in df_display.columns and pd.notna(ticker_data.get('Total_Vol'))
                    
                    if has_options_data:
                        col_a, col_b, col_c, col_d, col_e, col_f, col_g = st.columns(7)
                        with col_a:
                            st.metric("Precio", f"${ticker_data['Price']:.2f}")
                        with col_b:
                            st.metric("RSC", f"{ticker_data['RSC']:.2f}")
                        with col_c:
                            st.metric("R²", f"{ticker_data['R2']:.3f}" if pd.notna(ticker_data['R2']) else "N/A")
                        with col_d:
                            st.metric("Sharpe", f"{ticker_data['Sharpe']:.2f}" if pd.notna(ticker_data['Sharpe']) else "N/A")
                        with col_e:
                            st.metric("MACD-V", f"{ticker_data['MACD_V']:.2f}" if pd.notna(ticker_data['MACD_V']) else "N/A")
                        with col_f:
                            pc_oi = ticker_data.get('PC_Ratio_OI')
                            st.metric("P/C OI", f"{pc_oi:.2f}" if pd.notna(pc_oi) else "N/A",
                                      help="Put/Call Ratio por Open Interest")
                        with col_g:
                            st.metric("Sentiment", ticker_data.get('Sentiment', 'N/A'))
                    else:
                        col_a, col_b, col_c, col_d, col_e = st.columns(5)
                        with col_a:
                            st.metric("Precio", f"${ticker_data['Price']:.2f}")
                        with col_b:
                            st.metric("RSC", f"{ticker_data['RSC']:.2f}")
                        with col_c:
                            st.metric("R²", f"{ticker_data['R2']:.3f}" if pd.notna(ticker_data['R2']) else "N/A")
                        with col_d:
                            st.metric("Sharpe", f"{ticker_data['Sharpe']:.2f}" if pd.notna(ticker_data['Sharpe']) else "N/A")
                        with col_e:
                            st.metric("MACD-V", f"{ticker_data['MACD_V']:.2f}" if pd.notna(ticker_data['MACD_V']) else "N/A")
                    
                    plot_candlestick_chart(ticker, ticker_name, period=chart_period)
                    
                    st.markdown("---")
                
                st.success(f"✅ {len(tickers_to_plot)} gráfico(s) generado(s) correctamente")
        
    else:
        st.info("🚧 Los resultados aparecerán aquí una vez completado el escaneo")

if __name__ == "__main__":

    if check_password():
        main()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder.")
