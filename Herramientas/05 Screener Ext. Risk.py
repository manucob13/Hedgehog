import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import random
import requests
from utils.utils import check_password
from utils.tickers import create_tickers_universe
from utils.utils_schwab import connect_to_schwab, get_iv_rank_schwab

warnings.filterwarnings('ignore')

# Lock global para sincronizar descargas de yfinance
_yfinance_lock = Lock()

# ============= CONSTANTES PARA OPCIONES =============
CONTRACT_SIZE = 100
API_BASE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options"

# ============= FUNCIONES TÉCNICAS =============
def calculate_ema(data, period):
    """Calcula EMA"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_sma(data, period):
    """Calcula SMA"""
    return data.rolling(window=period).mean()

def calculate_bollinger_bands(df, period=20, std_dev=2.5):
    """Calcula Bollinger Bands y %B"""
    close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
    
    sma = calculate_sma(close, period)
    std = close.rolling(window=period).std()
    
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    
    percent_b = (close - lower_band) / (upper_band - lower_band)
    
    return sma.squeeze(), upper_band.squeeze(), lower_band.squeeze(), percent_b.squeeze()

def calculate_zscore(df, period=20):
    """Calcula Z-Score: (Precio - Media) / Desviación Estándar"""
    close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
    
    sma = calculate_sma(close, period)
    std = close.rolling(window=period).std()
    
    zscore = (close - sma) / std
    
    return zscore.squeeze()

def calculate_macd_v(df, fast_len=12, slow_len=26, signal_len=9, atr_len=26):
    """Calcula MACD-V (MACD normalizado por ATR)"""
    try:
        fast_ema = calculate_ema(df['Close'], fast_len)
        slow_ema = calculate_ema(df['Close'], slow_len)
        
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=atr_len).mean()
        
        macd = ((fast_ema - slow_ema) / atr) * 100
        signal = calculate_ema(macd, signal_len)
        
        return macd, signal
    except Exception:
        return None, None

def safe_extract_value(series_or_df, index=-1):
    """Extrae un valor escalar de una Serie o DataFrame de pandas de forma segura"""
    try:
        if series_or_df is None:
            return None
        
        if np.isscalar(series_or_df):
            return float(series_or_df) if not pd.isna(series_or_df) else None
        
        if isinstance(series_or_df, pd.DataFrame):
            if series_or_df.empty:
                return None
            if isinstance(series_or_df.columns, pd.MultiIndex):
                series_or_df = series_or_df.iloc[:, 0]
            else:
                series_or_df = series_or_df.squeeze()
        
        if isinstance(series_or_df, pd.Series):
            if len(series_or_df) == 0:
                return None
            value = series_or_df.iloc[index]
        else:
            arr = np.asarray(series_or_df)
            if arr.size == 0:
                return None
            value = arr.flat[index]
        
        if pd.isna(value):
            return None
        
        return float(value)
        
    except Exception as e:
        print(f"Error extrayendo valor: {type(e).__name__}: {str(e)}")
        return None

# ============= FUNCIONES OPCIONES (CBOE + SCHWAB) =============

@st.cache_data(ttl=300)
def fetch_option_data_cboe(ticker: str):
    """Obtener datos de opciones desde CBOE API"""
    urls = [
        f"{API_BASE_URL}/_{ticker}.json",
        f"{API_BASE_URL}/{ticker}.json"
    ]
    
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception:
            continue
    return None

def parse_option_data_cboe(raw_data: dict):
    """Parsear datos de opciones de CBOE"""
    try:
        data = pd.DataFrame.from_dict(raw_data)
        spot_price = float(data.loc["current_price", "data"])
        option_data = pd.DataFrame(data.loc["options", "data"])
        return spot_price, option_data
    except Exception:
        return 0, pd.DataFrame()

def process_option_data_cboe(data: pd.DataFrame) -> pd.DataFrame:
    """Procesar datos de opciones de CBOE"""
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
        df['open_interest'].notna() &
        (df['open_interest'] > 0)
    )
    
    return df[mask]

def get_options_volume_schwab(client, ticker):
    """Obtiene volumen total de opciones usando Schwab API como alternativa"""
    try:
        if client is None:
            return None
        
        from utils.utils_schwab import normalize_ticker
        
        symbol = normalize_ticker(ticker)
        cutoff_date = datetime.now() + timedelta(days=60)
        from_date = datetime.now().date()
        to_date = cutoff_date.date()
        
        response = client.get_option_chain(
            symbol,
            from_date=from_date,
            to_date=to_date
        )
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        total_volume = 0
        
        # Sumar volumen de calls
        call_map = data.get('callExpDateMap', {})
        for exp_date, strikes in call_map.items():
            for strike, contracts in strikes.items():
                for contract in contracts:
                    vol = contract.get('totalVolume', 0)
                    if vol and vol > 0:
                        total_volume += int(vol)
        
        # Sumar volumen de puts
        put_map = data.get('putExpDateMap', {})
        for exp_date, strikes in put_map.items():
            for strike, contracts in strikes.items():
                for contract in contracts:
                    vol = contract.get('totalVolume', 0)
                    if vol and vol > 0:
                        total_volume += int(vol)
        
        return total_volume if total_volume > 0 else None
        
    except Exception as e:
        return None

def get_options_volume_yahoo(ticker, days=7):
    """Volumen total de opciones (aprox última semana) con Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        exp_dates = stock.options
        if not exp_dates:
            return None
        
        cutoff_date = datetime.now() + timedelta(days=60)
        valid_expirations = [
            exp for exp in exp_dates 
            if datetime.strptime(exp, '%Y-%m-%d') <= cutoff_date
        ]
        if not valid_expirations:
            return None
        
        total_volume = 0
        for exp_date in valid_expirations[:3]:
            try:
                opt_chain = stock.option_chain(exp_date)
                if 'volume' in opt_chain.calls.columns:
                    total_volume += opt_chain.calls['volume'].fillna(0).sum()
                if 'volume' in opt_chain.puts.columns:
                    total_volume += opt_chain.puts['volume'].fillna(0).sum()
            except Exception:
                continue
        
        return int(total_volume) if total_volume > 0 else None
    except Exception:
        return None

def get_call_put_ratio_cboe(ticker):
    """Call/Put Ratio basado en Open Interest desde CBOE"""
    try:
        raw_data = fetch_option_data_cboe(ticker)
        if not raw_data:
            return None
        
        _, option_data = parse_option_data_cboe(raw_data)
        if option_data.empty:
            return None
        
        df = process_option_data_cboe(option_data)
        if df.empty:
            return None
        
        calls = df[df['type'] == 'C']
        puts = df[df['type'] == 'P']
        
        total_call_oi = calls['open_interest'].fillna(0).sum()
        total_put_oi = puts['open_interest'].fillna(0).sum()
        
        if total_put_oi > 0:
            ratio = total_call_oi / total_put_oi
            return round(ratio, 2)
        return None
    except Exception:
        return None

def calculate_put_wall_gex(ticker, spot_price=None):
    """Put Wall = strike con mayor GEX negativo (puts) en los próximos 60 días"""
    try:
        raw_data = fetch_option_data_cboe(ticker)
        if not raw_data:
            return None
        
        current_spot, option_data = parse_option_data_cboe(raw_data)
        if option_data.empty:
            return None
        
        spot = spot_price if spot_price else current_spot
        
        df = process_option_data_cboe(option_data)
        if df.empty:
            return None
        
        fecha_limite = datetime.now() + timedelta(days=60)
        df = df[df['expiration'] <= fecha_limite].copy()
        if df.empty:
            return None
        
        puts = df[df['type'] == 'P'].copy()
        if puts.empty:
            return None
        
        if 'gamma' in puts.columns:
            puts['GEX'] = puts['gamma'] * puts['open_interest'] * CONTRACT_SIZE * (spot ** 2) * 0.01
            puts['GEX'] = -puts['GEX']
        else:
            puts['GEX'] = -puts['open_interest'] * CONTRACT_SIZE
        
        gex_by_strike = puts.groupby('strike')['GEX'].sum()
        if len(gex_by_strike) > 0:
            return float(gex_by_strike.idxmin())
        return None
    except Exception:
        return None

def get_options_metrics(ticker, current_price=None, schwab_client=None):
    """Wrapper para obtener Vol Opciones, C/P ratio, Put Wall e IV Rank"""
    metrics = {
        'options_volume': None,
        'call_put_ratio': None,
        'put_wall': None,
        'iv_rank': None
    }
    try:
        # Intentar volumen con Schwab primero, luego Yahoo
        if schwab_client:
            metrics['options_volume'] = get_options_volume_schwab(schwab_client, ticker)
        
        if metrics['options_volume'] is None:
            metrics['options_volume'] = get_options_volume_yahoo(ticker, days=7)
        
        metrics['call_put_ratio'] = get_call_put_ratio_cboe(ticker)
        metrics['put_wall'] = calculate_put_wall_gex(ticker, spot_price=current_price)
        
        # IV Rank desde Schwab
        if schwab_client:
            metrics['iv_rank'] = get_iv_rank_schwab(schwab_client, ticker)
    except Exception:
        pass
    return metrics

# ============= ANALISIS DE TICKER (MODIFICADO) =============

def analyze_ticker(ticker, period="2y", interval="1d", bb_period=20, zscore_period=20, 
                   zscore_threshold_pos=2.5, zscore_threshold_neg=-2.5,
                   bb_threshold_upper=1.1, bb_threshold_lower=-0.1, 
                   macdv_threshold=50, filter_sma200=False, 
                   filter_zscore_positive=True, filter_zscore_negative=True, 
                   use_lock=True, schwab_client=None):
    """Analiza un ticker buscando sobreextensión estadística + métricas de opciones + IV Rank"""
    try:
        if use_lock:
            with _yfinance_lock:
                data = yf.download(
                    ticker, 
                    period=period, 
                    interval=interval, 
                    progress=False,
                    auto_adjust=True, 
                    actions=False,
                    prepost=False,
                    threads=False
                )
                if not data.empty:
                    data = data.copy(deep=True)
        else:
            data = yf.download(
                ticker, 
                period=period, 
                interval=interval, 
                progress=False,
                auto_adjust=True, 
                actions=False,
                prepost=False,
                threads=False
            )
            if not data.empty:
                data = data.copy(deep=True)
        
        if data.empty or len(data) < 50:
            return None
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            return None
        
        for col in required_cols:
            if isinstance(data[col], pd.DataFrame):
                data[col] = data[col].squeeze()
        
        try:
            if use_lock:
                with _yfinance_lock:
                    ticker_info = yf.Ticker(ticker)
                    info = ticker_info.info
                    company_name = info.get('longName') or info.get('shortName') or ticker
            else:
                ticker_info = yf.Ticker(ticker)
                info = ticker_info.info
                company_name = info.get('longName') or info.get('shortName') or ticker
        except Exception:
            company_name = ticker
        
        df_copy = data.copy(deep=True)
        
        bb_sma, bb_upper, bb_lower, bb_percent = calculate_bollinger_bands(
            df_copy, period=bb_period, std_dev=2.5
        )
        zscore = calculate_zscore(df_copy, period=zscore_period)
        macd_v, macd_v_signal = calculate_macd_v(df_copy)
        sma_200 = calculate_sma(df_copy['Close'], 200)
        
        if zscore is None or len(zscore) == 0:
            return None
        
        bb_sma = bb_sma.copy(deep=True) if bb_sma is not None else None
        bb_upper = bb_upper.copy(deep=True) if bb_upper is not None else None
        bb_lower = bb_lower.copy(deep=True) if bb_lower is not None else None
        bb_percent = bb_percent.copy(deep=True) if bb_percent is not None else None
        zscore = zscore.copy(deep=True) if zscore is not None else None
        macd_v = macd_v.copy(deep=True) if macd_v is not None else None
        macd_v_signal = macd_v_signal.copy(deep=True) if macd_v_signal is not None else None
        sma_200 = sma_200.copy(deep=True) if sma_200 is not None else None
        
        current_price = safe_extract_value(data['Close'])
        current_zscore = safe_extract_value(zscore)
        current_bb_percent = safe_extract_value(bb_percent)
        current_macdv = safe_extract_value(macd_v) or 0.0
        current_signal = safe_extract_value(macd_v_signal) or 0.0
        current_sma200 = safe_extract_value(sma_200)
        
        if current_price is None or current_zscore is None or current_bb_percent is None:
            return None
        
        if current_price <= 0:
            return None
        
        if filter_sma200 and current_sma200 is not None:
            if current_price < current_sma200:
                return None
        
        zscore_check = False
        if filter_zscore_positive and current_zscore >= zscore_threshold_pos:
            zscore_check = True
        if filter_zscore_negative and current_zscore <= zscore_threshold_neg:
            zscore_check = True
        
        if not zscore_check:
            return None
        
        bb_check = current_bb_percent >= bb_threshold_upper or current_bb_percent <= bb_threshold_lower
        if not bb_check:
            return None
        
        if abs(current_macdv) < macdv_threshold:
            return None
        
        if current_zscore >= zscore_threshold_pos and current_bb_percent >= bb_threshold_upper:
            signal_type = 'SOBRECOMPRA'
            signal_strength = min(current_zscore, 5.0)
        elif current_zscore <= zscore_threshold_neg and current_bb_percent <= bb_threshold_lower:
            signal_type = 'SOBREVENTA'
            signal_strength = min(abs(current_zscore), 5.0)
        else:
            return None
        
        macdv_confirms = False
        if signal_type == 'SOBRECOMPRA' and current_macdv > 150:
            macdv_confirms = True
        elif signal_type == 'SOBREVENTA' and current_macdv < -150:
            macdv_confirms = True
        
        options_metrics = get_options_metrics(ticker, current_price, schwab_client)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        
        return {
            'Ticker': str(ticker),
            'Company': str(company_name),
            'Price': current_price,
            'SMA_200': current_sma200,
            'Z-Score': current_zscore,
            'BB_%B': current_bb_percent,
            'MACD_V': current_macdv,
            'Signal': current_signal,
            'Type': str(signal_type),
            'Strength': signal_strength,
            'MACDV_Confirm': bool(macdv_confirms),
            'Date': data.index[-1],
            'Options_Volume': options_metrics['options_volume'],
            'Call_Put_Ratio': options_metrics['call_put_ratio'],
            'Put_Wall': options_metrics['put_wall'],
            'IV_Rank': options_metrics['iv_rank'],
            'Data': data.copy(deep=True),
            'BB_SMA': bb_sma,
            'BB_Upper': bb_upper,
            'BB_Lower': bb_lower,
            'ZScore_Series': zscore,
            'MACD_V_Series': macd_v,
            'Signal_Series': macd_v_signal,
            'SMA_200_Series': sma_200,
            'ID': f"{ticker}_{timestamp}"
        }
    except Exception as e:
        print(f"Error analyzing {ticker}: {type(e).__name__}: {str(e)}")
        return None

# ============= ESCANEO EN PARALELO =============

def scan_tickers(tickers_list, max_workers=5, use_lock=True, schwab_client=None, **kwargs):
    """Escanea múltiples tickers en paralelo con sincronización"""
    results = []
    errors = []
    filtered_out = []
    download_errors = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(tickers_list)
    completed = 0
    
    effective_workers = min(max_workers, 5) if use_lock else max_workers
    
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {executor.submit(analyze_ticker, ticker, use_lock=use_lock, 
                                   schwab_client=schwab_client, **kwargs): ticker 
                   for ticker in tickers_list}
        
        for future in as_completed(futures):
            ticker = futures[future]
            completed += 1
            progress = completed / total
            progress_bar.progress(progress)
            status_text.text(f"Analizando: {completed}/{total} tickers ({progress*100:.1f}%)")
            
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                else:
                    filtered_out.append(ticker)
            except Exception as e:
                error_msg = str(e)
                if "download" in error_msg.lower() or "fetch" in error_msg.lower():
                    download_errors.append(f"{ticker}: {error_msg}")
                else:
                    errors.append(f"{ticker}: {error_msg}")
    
    progress_bar.empty()
    status_text.empty()
    
    st.info(f"""
    📊 **Estadísticas del Escaneo:**
    - Total analizado: {total}
    - ✅ Detectados: {len(results)}
    - 🔍 Filtrados (no cumplen): {len(filtered_out)}
    - ❌ Errores de descarga: {len(download_errors)}
    - ⚠️ Otros errores: {len(errors)}
    """)
    
    if download_errors and len(download_errors) > 0:
        with st.expander(f"❌ Errores de descarga ({len(download_errors)} tickers)"):
            for error in download_errors[:20]:
                st.caption(error)
    
    if errors and len(errors) > 0:
        with st.expander(f"⚠️ Otros errores ({len(errors)} tickers)"):
            for error in errors[:20]:
                st.caption(error)
    
    return results

# ============= UTILIDADES PLOT =============

def ensure_series(data_series):
    """Asegura que los datos sean una Serie unidimensional válida"""
    if data_series is None:
        return None
    
    if np.isscalar(data_series):
        return None
    
    if isinstance(data_series, pd.DataFrame):
        if data_series.empty:
            return None
        data_series = data_series.iloc[:, 0]
    
    if isinstance(data_series, pd.Series):
        return data_series.squeeze()
    
    if isinstance(data_series, np.ndarray):
        if data_series.size == 0:
            return None
        return pd.Series(data_series)
    
    return data_series

def plot_ticker_analysis(ticker_data, chart_period='3mo'):
    """
    Genera gráfico con 3 paneles (sin volumen):
    - Precio + BB + SMA200 + Put Wall
    - Z-Score
    - MACD-V
    
    Args:
        ticker_data: Dict con datos del ticker
        chart_period: Periodo a mostrar ('3mo', '6mo', '1y', '2y')
    """
    try:
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(18, 11), facecolor='#0E1117')
        gs = fig.add_gridspec(3, 1, height_ratios=[2.5, 1, 1], hspace=0.3)
        
        data = ticker_data['Data']
        ticker = ticker_data['Ticker']
        put_wall = ticker_data.get('Put_Wall')
        
        if data is None or len(data) == 0:
            st.error(f"No hay datos disponibles para {ticker}")
            return None
        
        period_map = {
            '3mo': 90,
            '6mo': 180,
            '1y': 365,
            '2y': 730
        }
        days = period_map.get(chart_period, 90)
        cutoff_date = datetime.now() - timedelta(days=days)
        data_filtered = data[data.index >= cutoff_date].copy()
        
        if data_filtered.empty:
            data_filtered = data.copy()
        
        if isinstance(data_filtered.columns, pd.MultiIndex):
            data_filtered.columns = data_filtered.columns.get_level_values(0)
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in data_filtered.columns:
                data_filtered[col] = ensure_series(data_filtered[col])
        
        bb_lower = ensure_series(ticker_data.get('BB_Lower'))
        bb_upper = ensure_series(ticker_data.get('BB_Upper'))
        bb_sma = ensure_series(ticker_data.get('BB_SMA'))
        sma_200 = ensure_series(ticker_data.get('SMA_200_Series'))
        
        if bb_lower is not None:
            bb_lower = bb_lower[bb_lower.index >= cutoff_date]
        if bb_upper is not None:
            bb_upper = bb_upper[bb_upper.index >= cutoff_date]
        if bb_sma is not None:
            bb_sma = bb_sma[bb_sma.index >= cutoff_date]
        if sma_200 is not None:
            sma_200 = sma_200[sma_200.index >= cutoff_date]
        
        # PANEL 1: PRECIO + BB + SMA200 + PUT WALL
        ax1 = fig.add_subplot(gs[0])
        ax1.set_facecolor('#1A1D29')
        
        if bb_lower is not None and bb_upper is not None and len(bb_lower) > 0 and len(bb_upper) > 0:
            ax1.fill_between(data_filtered.index, bb_lower, bb_upper,
                             color='#FFB86C', alpha=0.1, label='Bollinger Bands (2.5σ)', zorder=1)
            ax1.plot(data_filtered.index, bb_upper, color='#FFB86C', 
                     linewidth=1.2, linestyle='--', alpha=0.7, zorder=2)
            if bb_sma is not None and len(bb_sma) > 0:
                ax1.plot(data_filtered.index, bb_sma, color='#BD93F9', 
                         linewidth=1.5, label='SMA(20)', zorder=2)
            ax1.plot(data_filtered.index, bb_lower, color='#FFB86C', 
                     linewidth=1.2, linestyle='--', alpha=0.7, zorder=2)
        
        if sma_200 is not None and len(sma_200) > 0:
            valid_sma = sma_200.dropna()
            if len(valid_sma) > 0:
                ax1.plot(valid_sma.index, valid_sma.values, color='#00D9FF', 
                         linewidth=1.8, label='SMA(200)', zorder=2, alpha=0.8)
        
        close_series = ensure_series(data_filtered['Close'])
        if close_series is not None and len(close_series) > 0:
            ax1.plot(data_filtered.index, close_series, color='#FFFFFF', linewidth=1.8, 
                     label='Precio', zorder=3)
            current_color = '#FF6B6B' if ticker_data['Type'] == 'SOBRECOMPRA' else '#4ECDC4'
            ax1.scatter(data_filtered.index[-1], close_series.iloc[-1], 
                        color=current_color, s=150, edgecolors='white', 
                        linewidth=2, zorder=10, label='Actual')
        
        if put_wall is not None and not pd.isna(put_wall):
            ax1.axhline(y=put_wall, color='#FE53BB', linestyle=':', linewidth=2, 
                        alpha=0.9, label=f'Put Wall: ${put_wall:.2f}', zorder=4)
            ax1.annotate(
                f'PUT WALL\n${put_wall:.2f}', 
                xy=(data_filtered.index[-1], put_wall),
                xytext=(data_filtered.index[int(len(data_filtered)*0.85)], put_wall),
                fontsize=10, fontweight='bold', color='#FE53BB',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#FE53BB', 
                          alpha=0.3, edgecolor='#FE53BB', linewidth=1.5),
                ha='right', va='center'
            )
        
        # Rango dinámico
        if close_series is not None and len(close_series) > 0:
            all_values = [close_series]
            if bb_lower is not None and len(bb_lower) > 0:
                all_values.append(bb_lower.dropna())
            if bb_upper is not None and len(bb_upper) > 0:
                all_values.append(bb_upper.dropna())
            if sma_200 is not None and len(sma_200) > 0:
                valid_sma = sma_200.dropna()
                if len(valid_sma) > 0:
                    all_values.append(valid_sma)
            combined = pd.concat(all_values, axis=1)
            y_min = combined.min().min()
            y_max = combined.max().max()
            if put_wall is not None and not pd.isna(put_wall):
                y_min = min(y_min, put_wall)
                y_max = max(y_max, put_wall)
            margin = (y_max - y_min) * 0.08
            ax1.set_ylim(y_min - margin, y_max + margin)
        
        ax1.set_ylabel('Precio ($)', fontsize=12, fontweight='bold', color='#FFFFFF')
        ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax1.grid(True, alpha=0.1, linestyle='-', linewidth=0.6)
        ax1.tick_params(labelsize=9, colors='#B0B0B0', labelbottom=False)
        for spine in ax1.spines.values():
            spine.set_color('#2D3142')
            spine.set_linewidth(1.2)
        
        # PANEL 2: Z-SCORE
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.set_facecolor('#1A1D29')
        
        zscore = ensure_series(ticker_data.get('ZScore_Series'))
        if zscore is not None:
            zscore = zscore[zscore.index >= cutoff_date]
        
        if zscore is not None and len(zscore) > 0:
            for i in range(1, len(zscore)):
                if pd.notna(zscore.iloc[i-1]) and pd.notna(zscore.iloc[i]):
                    y2 = float(zscore.iloc[i])
                    if abs(y2) > 3:
                        color, width = '#FF6B6B', 2.2
                    elif abs(y2) > 2.5:
                        color, width = '#FFB86C', 1.8
                    elif abs(y2) > 2:
                        color, width = '#FFD93D', 1.5
                    else:
                        color, width = '#95A5A6', 1.2
                    ax2.plot([zscore.index[i-1], zscore.index[i]], 
                             [float(zscore.iloc[i-1]), y2], 
                             color=color, linewidth=width, alpha=0.95, zorder=5)
            ax2.axhline(y=3, color='#FF6B6B', linestyle='--', linewidth=1.5, alpha=0.8, label='±3σ')
            ax2.axhline(y=2.5, color='#FFB86C', linestyle='--', linewidth=1.5, alpha=0.8, label='±2.5σ')
            ax2.axhline(y=0, color='#8E93A1', linestyle='-', linewidth=1.2, alpha=0.7)
            ax2.axhline(y=-2.5, color='#FFB86C', linestyle='--', linewidth=1.5, alpha=0.8)
            ax2.axhline(y=-3, color='#FF6B6B', linestyle='--', linewidth=1.5, alpha=0.8)
            ax2.fill_between(zscore.index, 2.5, 5, alpha=0.12, color='#FFB86C', zorder=0)
            ax2.fill_between(zscore.index, -5, -2.5, alpha=0.12, color='#FFB86C', zorder=0)
            ax2.fill_between(zscore.index, 3, 5, alpha=0.15, color='#FF6B6B', zorder=0)
            ax2.fill_between(zscore.index, -5, -3, alpha=0.15, color='#FF6B6B', zorder=0)
            current_zscore = ticker_data['Z-Score']
            zscore_color = '#FF6B6B' if abs(current_zscore) > 3 else '#FFB86C' if abs(current_zscore) > 2.5 else '#FFD93D'
            ax2.text(0.02, 0.95, f'Z-Score: {current_zscore:.2f}σ', 
                     transform=ax2.transAxes, fontsize=11, fontweight='bold', 
                     color='white', verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor=zscore_color, 
                               alpha=0.95, edgecolor='white', linewidth=1.5))
        ax2.set_ylabel('Z-Score (σ)', fontsize=11, fontweight='bold', color='#FFFFFF')
        ax2.set_ylim([-5, 5])
        ax2.legend(loc='upper right', fontsize=8, framealpha=0.9, ncol=2)
        ax2.grid(True, alpha=0.1, linestyle=':', linewidth=0.8)
        ax2.tick_params(labelsize=9, colors='#B0B0B0', labelbottom=False)
        for spine in ax2.spines.values():
            spine.set_color('#2D3142')
            spine.set_linewidth(1.2)
        
        # PANEL 3: MACD-V
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.set_facecolor('#1A1D29')
        
        macd_v = ensure_series(ticker_data.get('MACD_V_Series'))
        signal = ensure_series(ticker_data.get('Signal_Series'))
        
        if macd_v is not None:
            macd_v = macd_v[macd_v.index >= cutoff_date]
        if signal is not None:
            signal = signal[signal.index >= cutoff_date]
        
        if macd_v is not None and len(macd_v) > 0 and not macd_v.isna().all():
            for i in range(1, len(macd_v)):
                if pd.notna(macd_v.iloc[i-1]) and pd.notna(macd_v.iloc[i]):
                    y2 = float(macd_v.iloc[i])
                    if y2 > 150:
                        color, width = '#FF6B6B', 2.2
                    elif y2 > 50:
                        color, width = '#4ECDC4', 1.8
                    elif y2 > -50:
                        color, width = '#95A5A6', 1.5
                    elif y2 > -150:
                        color, width = '#EE5A6F', 1.8
                    else:
                        color, width = '#FF6B6B', 2.2
                    ax3.plot([macd_v.index[i-1], macd_v.index[i]], 
                             [float(macd_v.iloc[i-1]), y2], 
                             color=color, linewidth=width, alpha=0.95, zorder=5)
            if signal is not None and len(signal) > 0:
                ax3.plot(signal.index, signal, color='#FFB86C', linewidth=1.2, 
                         alpha=0.5, linestyle='--', zorder=3)
            ax3.axhline(y=150, color='#FF6B6B', linestyle='--', linewidth=1.5, alpha=0.8)
            ax3.axhline(y=50, color='#4ECDC4', linestyle='--', linewidth=1.5, alpha=0.8, label='±50')
            ax3.axhline(y=0, color='#8E93A1', linestyle='-', linewidth=1.2, alpha=0.7)
            ax3.axhline(y=-50, color='#EE5A6F', linestyle='--', linewidth=1.5, alpha=0.8)
            ax3.axhline(y=-150, color='#FF6B6B', linestyle='--', linewidth=1.5, alpha=0.8)
            current_macdv = ticker_data['MACD_V']
            macd_color = '#FF6B6B' if abs(current_macdv) > 150 else '#4ECDC4' if current_macdv > 50 else '#EE5A6F' if current_macdv < -50 else '#95A5A6'
            ax3.text(0.02, 0.95, f'MACD-V: {current_macdv:.1f}', 
                     transform=ax3.transAxes, fontsize=11, fontweight='bold', 
                     color='white', verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor=macd_color, 
                               alpha=0.95, edgecolor='white', linewidth=1.5))
        ax3.set_ylabel('MACD-V', fontsize=11, fontweight='bold', color='#FFFFFF')
        ax3.set_xlabel('Fecha', fontsize=12, fontweight='bold', color='#FFFFFF')
        ax3.legend(loc='upper right', fontsize=8, framealpha=0.9)
        ax3.grid(True, alpha=0.1, linestyle=':', linewidth=0.8)
        ax3.tick_params(labelsize=9, colors='#B0B0B0')
        for spine in ax3.spines.values():
            spine.set_color('#2D3142')
            spine.set_linewidth(1.2)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error generando gráfico: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

# ============= INTERFAZ PRINCIPAL =============

def main():
    st.set_page_config(
        page_title="Statistical Screener",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Statistical Overextension Screener")
    st.markdown("**Detecta sobrecompra/sobreventa estadística con Z-Score, Bollinger Bands, MACD-V e IV Rank**")
    st.markdown("---")
    
    # Conectar a Schwab al inicio
    if 'schwab_client' not in st.session_state:
        with st.spinner("🔐 Conectando con Schwab API..."):
            st.session_state['schwab_client'] = connect_to_schwab()
            if st.session_state['schwab_client']:
                st.success("✅ Conectado a Schwab API")
            else:
                st.warning("⚠️ No se pudo conectar a Schwab. Usando solo Yahoo Finance.")
    
    schwab_client = st.session_state.get('schwab_client')
    
    # PASO 1: UNIVERSO DE TICKERS
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
    
    # PASO 2: CONFIGURACIÓN
    st.markdown("### ⚙️ PASO 2: Configuración de Parámetros")
    
    with st.expander("📊 Configuración de Análisis", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📅 Periodo de Datos**")
            period = st.selectbox("Periodo histórico", 
                                  ["1mo", "3mo", "6mo", "1y", "2y"],
                                  index=4)
            interval = st.selectbox("Intervalo", 
                                    ["1d", "1wk"],
                                    index=0)
            st.markdown("**📈 Periodos Indicadores**")
            bb_period = st.number_input("Periodo BB", 10, 100, 20)
            zscore_period = st.number_input("Periodo Z-Score", 10, 100, 20)
        
        with col2:
            st.markdown("**📈 Filtros Estadísticos**")
            col2a, col2b = st.columns(2)
            with col2a:
                filter_zscore_positive = st.checkbox("🔴 Sobrecompra", value=True)
            with col2b:
                filter_zscore_negative = st.checkbox("🟢 Sobreventa", value=True)
            
            zscore_threshold_pos = st.number_input(
                "Z-Score Positivo (≥)", 
                min_value=1.0, max_value=5.0, value=2.5, step=0.1,
                disabled=not filter_zscore_positive
            )
            zscore_threshold_neg = st.number_input(
                "Z-Score Negativo (≤)", 
                min_value=-5.0, max_value=-1.0, value=-2.5, step=0.1,
                disabled=not filter_zscore_negative
            )
            
            bb_threshold = st.slider("BB %B Threshold", 0.0, 0.5, 0.1, 0.05)
            bb_threshold_upper = 1.0 + bb_threshold
            bb_threshold_lower = 0.0 - bb_threshold
            st.caption(f"📊 Superior: %B ≥ {bb_threshold_upper:.2f} | Inferior: %B ≤ {bb_threshold_lower:.2f}")
        
        with col3:
            st.markdown("**🎯 Filtros Adicionales**")
            macdv_threshold = st.number_input(
                "MACD-V Mínimo (|valor|)", 
                min_value=0, max_value=200, value=50, step=10
            )
            filter_sma200 = st.checkbox("Filtrar por SMA 200", value=False)
            st.markdown("**⚡ Paralelización**")
            max_workers = st.slider("Threads paralelos", 1, 10, 5)
            st.markdown("**🎲 Subconjunto Aleatorio**")
            use_random_subset = st.checkbox("Usar subconjunto aleatorio", value=False)
            random_count = st.number_input(
                "Cantidad de tickers", 10, 2000, 100, 10,
                disabled=not use_random_subset
            )
    
    st.markdown("---")
    
    # PASO 3: ESCANEO
    st.markdown("### 🚀 PASO 3: Ejecutar Escaneo")
    
    if not filter_zscore_positive and not filter_zscore_negative:
        st.error("⚠️ Debes seleccionar al menos un tipo de señal (Sobrecompra o Sobreventa)")
    
    scan_button = st.button(
        "🚀 INICIAR ESCANEO", 
        type="primary", 
        use_container_width=True,
        disabled=not filter_zscore_positive and not filter_zscore_negative
    )
    
    if scan_button:
        if 'tickers_universe' not in st.session_state or len(st.session_state.get('tickers_universe', [])) == 0:
            st.error("❌ No hay tickers disponibles para escanear")
            st.warning("⚠️ Por favor, recarga los tickers usando el botón 'Recargar Tickers'")
            st.stop()
        
        all_tickers = st.session_state['tickers_universe']
        if use_random_subset and random_count > 0:
            seed = st.session_state.get('random_seed', 42)
            random.seed(seed)
            tickers_list = random.sample(all_tickers, min(int(random_count), len(all_tickers)))
            st.info(f"🎲 Usando subconjunto aleatorio de {len(tickers_list)} tickers (seed: {seed})")
        else:
            tickers_list = all_tickers
        
        st.markdown("### 🔄 Escaneando tickers...")
        st.info(f"📊 Analizando {len(tickers_list):,} tickers...")
        
        with st.spinner("Analizando..."):
            results = scan_tickers(
                tickers_list, 
                max_workers=max_workers,
                use_lock=True,
                schwab_client=schwab_client,
                period=period,
                interval=interval,
                bb_period=bb_period,
                zscore_period=zscore_period,
                zscore_threshold_pos=zscore_threshold_pos,
                zscore_threshold_neg=zscore_threshold_neg,
                bb_threshold_upper=bb_threshold_upper,
                bb_threshold_lower=bb_threshold_lower,
                macdv_threshold=macdv_threshold,
                filter_sma200=filter_sma200,
                filter_zscore_positive=filter_zscore_positive,
                filter_zscore_negative=filter_zscore_negative
            )
        
        if results:
            st.session_state['scan_results'] = results
            st.success(f"✅ {len(results)} tickers detectados con sobreextensión estadística")
            st.rerun()
        else:
            st.warning("⚠️ No se encontraron tickers con las condiciones especificadas")
    
    st.markdown("---")
    
    # PASO 4: RESULTADOS
    if 'scan_results' in st.session_state:
        results = st.session_state['scan_results']
        st.markdown("### 📈 PASO 4: Resultados del Escaneo")
        
        df_results = pd.DataFrame([
            {
                'Ticker': r['Ticker'],
                'Compañía': r.get('Company', r['Ticker'])[:40] + '...' if len(r.get('Company', r['Ticker'])) > 40 else r.get('Company', r['Ticker']),
                'Tipo': r['Type'],
                'Fuerza (σ)': round(r['Strength'], 2),
                'Z-Score': round(r['Z-Score'], 2),
                'BB %B': round(r['BB_%B'], 2),
                'MACD-V': round(r['MACD_V'], 1),
                'Precio': round(r['Price'], 2),
                'SMA 200': f"${round(r['SMA_200'], 2)}" if r.get('SMA_200') is not None and not pd.isna(r['SMA_200']) else '-',
                'Vol Opciones': f"{r.get('Options_Volume', 0):,.0f}" if r.get('Options_Volume') else '-',
                'C/P Ratio': f"{r.get('Call_Put_Ratio', 0):.2f}" if r.get('Call_Put_Ratio') else '-',
                'Put Wall': f"${r.get('Put_Wall', 0):.2f}" if r.get('Put_Wall') else '-',
                'IV Rank (%)': f"{r.get('IV_Rank', 0):.1f}%" if r.get('IV_Rank') is not None else '-',
                'ID': r['ID']
            }
            for r in results
        ])
        
        df_results = df_results.sort_values('Fuerza (σ)', ascending=False).reset_index(drop=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Detectados", len(results))
        with col2:
            sobrecompra = len([r for r in results if r['Type'] == 'SOBRECOMPRA'])
            st.metric("🔴 Sobrecompra", sobrecompra)
        with col3:
            sobreventa = len([r for r in results if r['Type'] == 'SOBREVENTA'])
            st.metric("🟢 Sobreventa", sobreventa)
        with col4:
            macd_confirms = len([r for r in results if r['MACDV_Confirm']])
            st.metric("✅ MACD Confirmados", macd_confirms)
        
        tab1, tab2 = st.tabs(["📊 Tabla de Resultados", "📈 Análisis Individual"])
        
        with tab1:
            # Altura dinámica según cantidad de resultados
            table_height = min(max(len(results) * 40 + 50, 200), 600)
            
            st.dataframe(
                df_results.drop('ID', axis=1),
                use_container_width=True,
                height=table_height
            )
        
        with tab2:
            st.markdown("#### Selecciona un ticker para análisis detallado")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                ticker_options = {f"{r['Ticker']} - {r['Company'][:30]}": r['ID'] for r in results}
                selected_display = st.selectbox("Ticker", options=list(ticker_options.keys()))
            
            with col2:
                chart_period = st.selectbox(
                    "Periodo del gráfico",
                    options=['3mo', '6mo', '1y', '2y'],
                    index=0,
                    help="Selecciona el periodo a mostrar en el gráfico"
                )
            
            if selected_display:
                selected_id = ticker_options[selected_display]
                selected_result = next((r for r in results if r['ID'] == selected_id), None)
                if selected_result:
                    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
                    with col1:
                        st.metric("Precio", f"${selected_result['Price']:.2f}")
                    with col2:
                        st.metric("Z-Score", f"{selected_result['Z-Score']:.2f}σ")
                    with col3:
                        st.metric("BB %B", f"{selected_result['BB_%B']:.2f}")
                    with col4:
                        vol_opt = selected_result.get('Options_Volume')
                        st.metric("Vol Opciones", f"{vol_opt:,.0f}" if vol_opt else 'N/A')
                    with col5:
                        cp_ratio = selected_result.get('Call_Put_Ratio')
                        st.metric("C/P Ratio", f"{cp_ratio:.2f}" if cp_ratio else 'N/A')
                    with col6:
                        put_wall = selected_result.get('Put_Wall')
                        st.metric("Put Wall", f"${put_wall:.2f}" if put_wall else 'N/A')
                    with col7:
                        iv_rank = selected_result.get('IV_Rank')
                        st.metric("IV Rank", f"{iv_rank:.1f}%" if iv_rank is not None else 'N/A')
                    st.markdown("---")
                    fig = plot_ticker_analysis(selected_result, chart_period=chart_period)
                    if fig:
                        st.pyplot(fig)
                        plt.close(fig)

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
