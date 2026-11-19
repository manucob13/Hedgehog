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


def calculate_k_ratio(prices, period=52):
    """
    Calcula el K-Ratio = slope / (SE_slope × √n) sobre log-precios semanales.

    SE_slope = SE_residuos / √Sxx   donde Sxx = Σ(x - x̄)²
    Esto es equivalente al estadístico-t del slope dividido entre √n.

    Timeframe: Semanal · Período: 52 semanas (1 año)

    Umbrales orientativos para n=52:
        K < 0.5   → Tendencia débil o ruido
        K 0.5–0.9 → Tendencia moderada
        K 0.9–1.5 → Tendencia sólida       ✅ (filtro "Sólida")
        K ≥ 1.5   → Tendencia muy consistente ✅ (filtro "Muy consistente")

    Con R²=0.86 y tendencia del 30% anual → K ≈ 2.5–3.0
    Con R²=0.50 y tendencia débil         → K ≈ 0.5–0.8
    """
    try:
        if len(prices) < period:
            return None

        y = np.log(prices.values[-period:].astype(float))
        x = np.arange(period)

        # Regresión lineal sobre log-precios
        coeffs = np.polyfit(x, y, 1)
        slope  = coeffs[0]

        # Residuos → SE de los residuos (ddof=2: pendiente + intercepto)
        y_pred       = np.polyval(coeffs, x)
        residuals    = y - y_pred
        se_residuals = np.sqrt(np.sum(residuals ** 2) / (period - 2))

        if se_residuals == 0:
            return None

        # Sxx = Σ(x - x̄)²  →  SE del slope
        x_mean   = np.mean(x)
        sxx      = np.sum((x - x_mean) ** 2)
        se_slope = se_residuals / np.sqrt(sxx)

        if se_slope == 0:
            return None

        # K-Ratio final
        k_ratio = slope / (se_slope * np.sqrt(period))
        return round(k_ratio, 3)

    except:
        return None



def calculate_hurst_exponent(prices, min_window=10, max_window=None):
    """
    Calcula el Exponente de Hurst usando DFA (Detrended Fluctuation Analysis).

    DFA es el método preferido frente al R/S clásico porque:
    - Tiene mucho menos sesgo con muestras pequeñas (~260 barras semanales)
    - R/S clásico sobreestima H en ~0.12 con estos tamaños de muestra
    - DFA distingue mejor entre tendencia real y ruido

    Interpretación calibrada para datos semanales (5 años, ~260 barras):
        H < 0.50  → Serie antipersistente (reversión a la media)
        H ≈ 0.51  → Ruido blanco / camino aleatorio
        H > 0.55  → Tendencia persistente (umbral recomendado)
        H > 0.60  → Tendencia fuerte
        H > 0.65  → Tendencia muy fuerte (solo ~20% de trending stocks)

    NOTA: Con datos semanales, H > 0.70 es extremadamente raro incluso
    en acciones con tendencias claras. No usar umbrales superiores a 0.65.

    Algoritmo DFA:
        1. Calcular log-retornos y su perfil acumulado centrado
        2. Dividir el perfil en ventanas de tamaño s
        3. Detrend lineal en cada ventana → RMS de residuos F(s)
        4. Regresión log(F) ~ alpha * log(s)  →  alpha ≈ H

    Args:
        prices: Serie de precios de cierre (pd.Series o similar)
        min_window: Ventana mínima (default 10 semanas)
        max_window: Ventana máxima (default: n/4)

    Returns:
        float: Exponente de Hurst DFA (0-1) o None si datos insuficientes
    """
    try:
        if len(prices) < 52:
            return None

        # Paso 1: log-retornos
        log_ret = np.diff(np.log(prices.values.astype(float)))
        n = len(log_ret)

        if n < 40:
            return None

        # Paso 2: perfil = cumsum de retornos centrados en su media
        profile = np.cumsum(log_ret - np.mean(log_ret))

        if max_window is None:
            max_window = n // 4

        # Paso 3: escala logarítmica de ventanas
        windows = []
        w = min_window
        while w <= max_window:
            windows.append(w)
            w = int(w * 1.5)
            if w == windows[-1]:
                w += 1

        if len(windows) < 3:
            return None

        fluctuations = []
        valid_windows = []

        for w in windows:
            n_segments = n // w
            if n_segments < 4:      # mínimo 4 segmentos para fiabilidad
                continue

            rms_list = []
            for seg in range(n_segments):
                seg_data = profile[seg * w:(seg + 1) * w]
                x = np.arange(w)
                # Detrend lineal dentro de cada segmento
                coeffs = np.polyfit(x, seg_data, 1)
                trend = np.polyval(coeffs, x)
                residuals = seg_data - trend
                rms_list.append(np.sqrt(np.mean(residuals ** 2)))

            if rms_list:
                fluctuations.append(np.mean(rms_list))
                valid_windows.append(w)

        if len(valid_windows) < 3:
            return None

        # Paso 4: regresión log-log → pendiente = alpha (≈ H)
        log_w = np.log(valid_windows)
        log_f = np.log(fluctuations)
        coeffs = np.polyfit(log_w, log_f, 1)
        alpha = coeffs[0]

        # Limitar al rango válido
        alpha = max(0.0, min(1.0, alpha))
        return round(alpha, 3)

    except:
        return None

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


def get_monthly_expirations(expirations, days_threshold=7):
    """
    Filtra las expiraciones mensuales (3er viernes de cada mes)
    y aplica la regla de proximidad: si el próximo mensual está a
    menos de `days_threshold` días, incluye también el siguiente mensual.

    Args:
        expirations: tuple/list de strings 'YYYY-MM-DD' de yfinance
        days_threshold: días mínimos para considerar que un vencimiento
                        está "demasiado cerca" (default 7)

    Returns:
        list de strings con las expiraciones mensuales seleccionadas (1 o 2)
    """
    today = datetime.now().date()

    def is_third_friday(d):
        """Comprueba si una fecha es el 3er viernes del mes."""
        if d.weekday() != 4:   # 4 = viernes
            return False
        # El 3er viernes cae siempre entre el día 15 y el 21
        return 15 <= d.day <= 21

    # Filtrar solo vencimientos mensuales futuros
    monthly = []
    for exp_str in expirations:
        try:
            exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
            if exp_date > today and is_third_friday(exp_date):
                monthly.append((exp_date, exp_str))
        except:
            continue

    if not monthly:
        return []

    # Ordenar por fecha
    monthly.sort(key=lambda x: x[0])

    # Regla de proximidad: si el primer mensual está a < threshold días, coger también el segundo
    first_date, first_str = monthly[0]
    days_to_first = (first_date - today).days

    if days_to_first < days_threshold and len(monthly) >= 2:
        # Primer mensual muy cercano → usar primero + segundo
        return [first_str, monthly[1][1]]
    else:
        # Usar solo el primer mensual
        return [first_str]


def get_options_metrics_yf(ticker, current_price, price_range_pct=10, days_threshold=7):
    """
    Obtiene métricas de volumen de opciones desde Yahoo Finance.

    Lógica de vencimientos:
        - Usa únicamente vencimientos MENSUALES (3er viernes del mes).
        - Si el próximo mensual está a menos de `days_threshold` días,
          añade también el siguiente mensual.

    Lógica de actividad (volumen + open interest):
        - Para cada strike en rango ±price_range_pct%:
            · Si volume > 0  → usar volume  (actividad intradía)
            · Si volume == 0 → usar openInterest como fallback
        - El P/C Ratio se calcula sobre la suma total de ambas métricas.

    Args:
        ticker (str): Símbolo del ticker
        current_price (float): Precio actual de la acción
        price_range_pct (int): Rango de strikes alrededor del precio (default 10%)
        days_threshold (int): Días mínimos al primer mensual (default 7)

    Returns:
        dict con 'Options_Vol', 'PC_Ratio', 'Sentiment', 'Exp_Used'  o  None
    """
    try:
        stock = yf.Ticker(ticker)

        # Verificar si tiene opciones disponibles
        expirations = stock.options
        if not expirations or len(expirations) == 0:
            return None

        # Seleccionar vencimientos mensuales
        selected_exps = get_monthly_expirations(expirations, days_threshold=days_threshold)

        # Fallback: si no hay mensuales disponibles, usar las 2 primeras
        if not selected_exps:
            selected_exps = list(expirations[:min(2, len(expirations))])

        # Calcular rango de strikes (±price_range_pct%)
        lower_bound = current_price * (1 - price_range_pct / 100)
        upper_bound = current_price * (1 + price_range_pct / 100)

        total_call_activity = 0
        total_put_activity  = 0

        for exp_date in selected_exps:
            try:
                chain = stock.option_chain(exp_date)

                # ---- CALLS ----
                calls = chain.calls
                calls_in_range = calls[
                    (calls['strike'] >= lower_bound) &
                    (calls['strike'] <= upper_bound)
                ].copy()

                if not calls_in_range.empty:
                    calls_in_range['volume']       = calls_in_range['volume'].fillna(0)
                    calls_in_range['openInterest'] = calls_in_range['openInterest'].fillna(0) \
                        if 'openInterest' in calls_in_range.columns else 0

                    call_activity = calls_in_range.apply(
                        lambda row: row['volume'] if row['volume'] > 0 else row['openInterest'],
                        axis=1
                    ).sum()
                    total_call_activity += call_activity

                # ---- PUTS ----
                puts = chain.puts
                puts_in_range = puts[
                    (puts['strike'] >= lower_bound) &
                    (puts['strike'] <= upper_bound)
                ].copy()

                if not puts_in_range.empty:
                    puts_in_range['volume']       = puts_in_range['volume'].fillna(0)
                    puts_in_range['openInterest'] = puts_in_range['openInterest'].fillna(0) \
                        if 'openInterest' in puts_in_range.columns else 0

                    put_activity = puts_in_range.apply(
                        lambda row: row['volume'] if row['volume'] > 0 else row['openInterest'],
                        axis=1
                    ).sum()
                    total_put_activity += put_activity

            except Exception:
                continue

        total_activity = total_call_activity + total_put_activity

        if total_activity == 0:
            return None

        if total_call_activity == 0:
            pc_ratio = float('inf')
        else:
            pc_ratio = total_put_activity / total_call_activity

        if pc_ratio == float('inf'):
            sentiment = "🔴 Muy Bajista"
        elif pc_ratio < 0.7:
            sentiment = "🟢 Muy Alcista"
        elif pc_ratio < 1.0:
            sentiment = "🟢 Alcista"
        elif pc_ratio < 1.3:
            sentiment = "🟡 Neutral"
        else:
            sentiment = "🔴 Bajista"

        exp_used = ", ".join(selected_exps)

        return {
            'Options_Vol': int(total_activity),
            'PC_Ratio':    round(pc_ratio, 2) if pc_ratio != float('inf') else None,
            'Sentiment':   sentiment,
            'Exp_Used':    exp_used
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
        
        # MACD-V - Filtro configurable con radio button
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

        # ========== K-RATIO (52 semanas) ==========
        # Siempre se calcula; solo filtra si el usuario lo activa
        k_ratio = calculate_k_ratio(close, period=52)

        k_ratio_filter = params['k_ratio_filter']
        if k_ratio_filter == "K-Ratio ≥ 1.5 (Tendencia fuerte)":
            if k_ratio is None or k_ratio < 1.5:
                return None
        elif k_ratio_filter == "K-Ratio ≥ 2.0 (Muy consistente)":
            if k_ratio is None or k_ratio < 2.0:
                return None
        # "Sin filtro" → no elimina nada, solo calcula

        # ========== TODOS LOS FILTROS PASADOS ==========
        name, market_cap = get_ticker_name_and_marketcap(ticker)
        next_dividend = get_next_dividend_date(ticker)
        div_yield = get_dividend_yield(ticker)
        
        # ========== RESULTADO ==========
        result = {
            'Ticker':      ticker,
            'Name':        name,
            'Price':       round(current_price, 2),
            'Market_Cap':  market_cap,
            'Next_Dividend': next_dividend,
            'Div_Yield_%': div_yield,
            'RSC':         round(rsc, 2) if rsc is not None else None,
            'Dist_Max_%':  round(dist_to_max, 2),
            'WMA30':       round(wma30, 2) if wma30 is not None else None,
            'Dist_WMA_%':  round(dist_to_wma, 2),
            'Slope':       round(slope, 2) if slope is not None else None,
            'R2':          round(r_squared, 3) if r_squared is not None else None,
            'Norm_Dist':   round(normalized_dist, 2) if normalized_dist is not None else None,
            'K_Ratio':     k_ratio,
            'Atlas':       int(atlas_value),
            'MIC_Value':   round(mic_value, 2) if mic_value is not None else None,
            'Sharpe':      round(sharpe, 2) if sharpe is not None else None,
            'MACD_V':      round(macd_v, 2) if macd_v is not None else None
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
        
        # Calcular WMA30
        wma30 = ta.wma(data['Close'], length=30)
        
        # Crear gráfico
        fig = go.Figure()
        
        # Velas japonesas
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
        
        # WMA30
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

        st.markdown("**📐 K-Ratio (52 sem. · 1 año)**")

        k_ratio_filter = st.radio(
            "Selecciona filtro K-Ratio:",
            options=[
                "Sin filtro (solo calcular)",
                "K-Ratio ≥ 1.5 (Tendencia fuerte)",
                "K-Ratio ≥ 2.0 (Muy consistente)",
            ],
            index=0,
            help=(
                "K-Ratio = slope / (SE_residuos × √52) sobre log-precios semanales.\n"
                "Mide la consistencia de la tendencia en el último año.\n\n"
                "Sin filtro → aparece en resultados pero no elimina tickers\n"
                "≥ 1.5 → tendencia fuerte (1 año de subida consistente)\n"
                "≥ 2.0 → tendencia muy consistente (pocos candidatos)"
            )
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
            True,   # Precio máximo (siempre)
            True,   # RSC > 0 (siempre)
            True,   # Distancia a máximos (siempre)
            True,   # Close > WMA30 (siempre)
            apply_wma_dist,
            apply_lr,
            apply_mic,
            apply_sharpe,
            1 if macd_filter != "Sin filtro" else 0,
            1 if k_ratio_filter != "Sin filtro (solo calcular)" else 0,
            apply_atlas,
        ])
        
        st.info(f"**{filters_count}** filtros activos")
        st.success(f"Precio máx: **${max_price}**")
        if macd_filter != "Sin filtro":
            st.success(f"MACD-V: {macd_filter.replace('MACD-V ', '')}")
        if k_ratio_filter != "Sin filtro (solo calcular)":
            st.success(f"K-Ratio: {k_ratio_filter.replace('K-Ratio ', '')}")
        else:
            st.info("K-Ratio: solo cálculo")
    
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
            'max_price':      max_price,
            'dist_to_max':    dist_to_max,
            'max_years':      max_years,
            'dist_to_wma':    dist_to_wma,
            'apply_wma_dist': apply_wma_dist,
            'apply_lr':       apply_lr,
            'apply_mic':      apply_mic,
            'apply_sharpe':   apply_sharpe,
            'macd_filter':    macd_filter,
            'k_ratio_filter': k_ratio_filter,
            'apply_atlas':    apply_atlas,
        }
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        tickers = st.session_state['tickers_universe']
        
        df_results = run_screener(tickers, params, progress_bar, status_text)
        
        progress_bar.empty()
        
        if len(df_results) > 0:
            st.session_state['scan_results'] = df_results
            st.session_state['scan_timestamp'] = datetime.now()
            # Resetear datos de opciones al hacer nuevo scan
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
        
        # ============= BOTÓN PARA CALCULAR OPCIONES + HURST =============
        st.markdown("### 📊 Opciones & Hurst (DFA)")

        st.info(
            "💡 **P/C Ratio:** < 0.7 = 🟢 Muy Alcista | 0.7–1.0 = 🟢 Alcista | "
            "1.0–1.3 = 🟡 Neutral | > 1.3 = 🔴 Bajista  ·  "
            "📅 Vencimiento mensual (3er viernes). Si está a < 7 días se añade el siguiente.  ·  "
            "📊 Actividad: volumen del día; si es 0 se usa Open Interest.  |  "
            "📈 **Hurst DFA:** < 0.50 = 🔴 Antipersistente | 0.50–0.54 = ⚪ Aleatorio | "
            "0.55–0.59 = 🟡 Persistente | 0.60–0.64 = 🟠 Tendencia fuerte | ≥ 0.65 = 🟢 Muy fuerte"
        )

        col_opt1, col_opt2 = st.columns([3, 1])

        with col_opt1:
            if 'options_calculated' not in st.session_state or not st.session_state['options_calculated']:
                st.warning("⚠️ Aún no se ha calculado el volumen de opciones ni el Hurst DFA")
            else:
                st.success(f"✅ Opciones y Hurst DFA calculados para {len(df_display)} tickers")

        with col_opt2:
            calculate_options_btn = st.button(
                "📊 Calcular Opciones + Hurst",
                type="primary",
                use_container_width=True,
                disabled='options_calculated' in st.session_state and st.session_state['options_calculated']
            )

        if calculate_options_btn:
            st.markdown("---")
            st.info("🔄 Calculando opciones y Hurst DFA... Esto puede tomar algunos minutos.")

            progress_opt = st.progress(0)
            status_opt = st.empty()

            total_tickers = len(df_display)
            enriched_data = []

            for idx, (_, row) in enumerate(df_display.iterrows()):
                ticker = row['Ticker']
                current_price = row['Price']

                progress_opt.progress((idx + 1) / total_tickers)
                status_opt.text(f"🔍 Procesando: {ticker} ({idx + 1}/{total_tickers})")

                # ---- Opciones ----
                metrics = get_options_metrics_yf(
                    ticker,
                    current_price,
                    price_range_pct=10,
                    days_threshold=7
                )

                # ---- Hurst DFA (necesita datos semanales) ----
                weekly_data = download_weekly_data(ticker, period="5y", use_lock=False)
                hurst_val = None
                if weekly_data is not None and len(weekly_data) >= 52:
                    hurst_val = calculate_hurst_exponent(weekly_data['Close'])

                if hurst_val is None:
                    hurst_label = "N/A"
                elif hurst_val < 0.50:
                    hurst_label = "🔴 Antipersistente"
                elif hurst_val < 0.55:
                    hurst_label = "⚪ Aleatorio"
                elif hurst_val < 0.60:
                    hurst_label = "🟡 Persistente"
                elif hurst_val < 0.65:
                    hurst_label = "🟠 Tendencia fuerte"
                else:
                    hurst_label = "🟢 Muy fuerte"

                record = {
                    'Ticker':      ticker,
                    'Options_Vol': metrics['Options_Vol'] if metrics else None,
                    'PC_Ratio':    metrics['PC_Ratio']    if metrics else None,
                    'Sentiment':   metrics['Sentiment']   if metrics else 'N/A',
                    'Exp_Used':    metrics.get('Exp_Used', 'N/A') if metrics else 'N/A',
                    'Hurst':       hurst_val,
                    'Hurst_Label': hurst_label,
                }
                enriched_data.append(record)

            progress_opt.empty()
            status_opt.empty()

            df_enriched = pd.DataFrame(enriched_data)
            df_display = df_display.merge(df_enriched, on='Ticker', how='left')

            st.session_state['scan_results'] = df_display
            st.session_state['options_calculated'] = True

            st.success(f"✅ Opciones y Hurst DFA calculados para {len(df_display)} tickers")
            st.rerun()
        
        st.markdown("---")

        # Preparar dataframe para mostrar
        has_options = 'Options_Vol' in df_display.columns
        has_hurst   = 'Hurst' in df_display.columns

        # Columnas base (siempre presentes)
        base_cols = ['Ticker', 'Name', 'Price', 'Market_Cap_Formatted',
                     'Next_Dividend', 'Div_Yield_%']

        # Opciones + Hurst (aparecen juntos tras pulsar el botón)
        options_cols = ['Options_Vol', 'PC_Ratio', 'Sentiment', 'Exp_Used'] if has_options else []
        hurst_cols   = ['Hurst', 'Hurst_Label'] if has_hurst else []

        # Columnas técnicas del screener — K_Ratio incluido siempre
        tech_cols  = ['RSC', 'Dist_Max_%', 'WMA30', 'Dist_WMA_%',
                      'Slope', 'R2', 'Norm_Dist', 'K_Ratio']
        extra_cols = ['Atlas', 'MIC_Value', 'Sharpe', 'MACD_V']

        all_display_cols = base_cols + options_cols + hurst_cols + tech_cols + extra_cols
        all_display_cols = [c for c in all_display_cols if c in df_display.columns]

        df_table = df_display[all_display_cols]

        column_config = {
            "Ticker":               st.column_config.TextColumn("Ticker", width="small"),
            "Name":                 st.column_config.TextColumn("Nombre", width="medium"),
            "Price":                st.column_config.NumberColumn("Precio", format="$%.2f"),
            "Market_Cap_Formatted": st.column_config.TextColumn("Market Cap", width="small"),
            "Next_Dividend":        st.column_config.TextColumn("Próx. Dividendo", width="medium"),
            "Div_Yield_%":          st.column_config.NumberColumn("Yield %", format="%.2f%%"),
            "Options_Vol":          st.column_config.NumberColumn(
                "Act. Opciones",
                format="%d",
                help="Actividad total de opciones (±10% precio, venc. mensual). "
                     "Volumen del día; si es 0, se usa Open Interest."
            ),
            "PC_Ratio":             st.column_config.NumberColumn(
                "P/C Ratio",
                format="%.2f",
                help="Put/Call Ratio sobre vencimiento mensual (±10% strikes)"
            ),
            "Sentiment":            st.column_config.TextColumn("Sentiment", width="medium"),
            "Exp_Used":             st.column_config.TextColumn(
                "Vencimiento(s)",
                width="medium",
                help="Vencimiento(s) mensual(es) utilizados para el cálculo"
            ),
            "Hurst":                st.column_config.NumberColumn(
                "Hurst (DFA)",
                format="%.3f",
                help=(
                    "Exponente de Hurst por DFA (datos semanales, 5 años). "
                    "< 0.50 Antipersistente · 0.50–0.54 Aleatorio · "
                    "0.55–0.59 Persistente · 0.60–0.64 Fuerte · ≥ 0.65 Muy fuerte"
                )
            ),
            "Hurst_Label":          st.column_config.TextColumn(
                "Hurst señal",
                width="medium",
                help="Interpretación del Exponente de Hurst DFA"
            ),
            "RSC":                  st.column_config.NumberColumn("RSC", format="%.2f"),
            "Dist_Max_%":           st.column_config.NumberColumn("Dist Max %", format="%.2f%%"),
            "WMA30":                st.column_config.NumberColumn("WMA30", format="%.2f"),
            "Dist_WMA_%":           st.column_config.NumberColumn("Dist WMA %", format="%.2f%%"),
            "Slope":                st.column_config.NumberColumn("Slope", format="%.2f"),
            "R2":                   st.column_config.NumberColumn("R²", format="%.3f"),
            "Norm_Dist":            st.column_config.NumberColumn("Norm Dist", format="%.2f"),
            "K_Ratio":              st.column_config.NumberColumn(
                "K-Ratio (52s)",
                format="%.3f",
                help=(
                    "K-Ratio semanal (n=52, 1 año): slope / (SE × √52) sobre log-precios.\n"
                    "< 0.5 débil · 0.5–1.0 moderada · 1.0–1.5 sólida · 1.5–2.0 fuerte · ≥ 2.0 muy consistente"
                )
            ),
            "Atlas":                st.column_config.NumberColumn("Atlas", format="%d"),
            "MIC_Value":            st.column_config.NumberColumn("MIC Value", format="%.2f"),
            "Sharpe":               st.column_config.NumberColumn("Sharpe", format="%.2f"),
            "MACD_V":               st.column_config.NumberColumn("MACD-V", format="%.2f")
        }
        
        # Tabla de resultados
        st.dataframe(
            df_table,
            use_container_width=True,
            height=400,
            column_config=column_config
        )
        
        # Botón de descarga
        csv_df = df_display.drop('Market_Cap_Formatted', axis=1, errors='ignore')
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
                    
                    # Métricas: adaptar según datos disponibles
                    metric_cols_data = [
                        ("Precio",   f"${ticker_data['Price']:.2f}"),
                        ("RSC",      f"{ticker_data['RSC']:.2f}"    if pd.notna(ticker_data.get('RSC'))     else "N/A"),
                        ("R²",       f"{ticker_data['R2']:.3f}"     if pd.notna(ticker_data.get('R2'))      else "N/A"),
                        ("K-Ratio",  f"{ticker_data['K_Ratio']:.3f}" if pd.notna(ticker_data.get('K_Ratio')) else "N/A"),
                        ("Sharpe",   f"{ticker_data['Sharpe']:.2f}" if pd.notna(ticker_data.get('Sharpe'))  else "N/A"),
                        ("MACD-V",   f"{ticker_data['MACD_V']:.2f}" if pd.notna(ticker_data.get('MACD_V'))  else "N/A"),
                    ]

                    if 'Hurst' in ticker_data and pd.notna(ticker_data.get('Hurst')):
                        metric_cols_data.append(
                            ("Hurst DFA", ticker_data.get('Hurst_Label', f"{ticker_data['Hurst']:.3f}"))
                        )

                    if has_options and pd.notna(ticker_data.get('Options_Vol')):
                        metric_cols_data.append(
                            ("Sentiment", ticker_data.get('Sentiment', 'N/A'))
                        )

                    metric_cols = st.columns(len(metric_cols_data))
                    for col_obj, (label, value) in zip(metric_cols, metric_cols_data):
                        with col_obj:
                            st.metric(label, value)
                    
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
