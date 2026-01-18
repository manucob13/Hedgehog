import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import random
from utils.utils import check_password
from utils.tickers import create_tickers_universe

warnings.filterwarnings('ignore')

# Lock global para sincronizar descargas de yfinance
_yfinance_lock = Lock()

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
    except Exception as e:
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

def analyze_ticker(ticker, period="6mo", interval="1d", bb_period=20, zscore_period=20, 
                   zscore_threshold_pos=2.5, zscore_threshold_neg=-2.5,
                   bb_threshold_upper=1.1, bb_threshold_lower=-0.1, 
                   macdv_threshold=50, filter_sma200=False, 
                   filter_zscore_positive=True, filter_zscore_negative=True, use_lock=True):
    """Analiza un ticker buscando sobreextensión estadística"""
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
        except:
            company_name = ticker
        
        df_copy = data.copy(deep=True)
        
        # Calcular SMA 200
        sma_200 = calculate_sma(df_copy['Close'], 200)
        
        bb_sma, bb_upper, bb_lower, bb_percent = calculate_bollinger_bands(
            df_copy, period=bb_period, std_dev=2.5
        )
        zscore = calculate_zscore(df_copy, period=zscore_period)
        macd_v, macd_v_signal = calculate_macd_v(df_copy)
        
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
        
        # Filtro SMA 200
        if filter_sma200 and current_sma200 is not None:
            if current_price < current_sma200:
                return None
        
        # Filtro Z-Score (positivo o negativo según selección)
        zscore_check = False
        if filter_zscore_positive and current_zscore >= zscore_threshold_pos:
            zscore_check = True
        if filter_zscore_negative and current_zscore <= zscore_threshold_neg:
            zscore_check = True
        
        if not zscore_check:
            return None
        
        # Filtro Bollinger Bands
        bb_check = current_bb_percent >= bb_threshold_upper or current_bb_percent <= bb_threshold_lower
        if not bb_check:
            return None
        
        # Filtro MACD-V
        if abs(current_macdv) < macdv_threshold:
            return None
        
        # Determinar tipo de señal
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

def scan_tickers(tickers_list, max_workers=5, use_lock=True, **kwargs):
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
        futures = {executor.submit(analyze_ticker, ticker, use_lock=use_lock, **kwargs): ticker 
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

def ensure_series(data_series):
    """Asegura que los datos sean una Serie unidimensional válida"""
    if data_series is None:
        return None
    
    # Si ya es un escalar, devolver None (no es una serie)
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

def plot_ticker_analysis(ticker_data):
    """Genera gráfico completo con 4 paneles: Precio + BB + SMA200, Z-Score, MACD-V, Volumen"""
    try:
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(18, 14), facecolor='#0E1117')
        gs = fig.add_gridspec(4, 1, height_ratios=[2.5, 1, 1, 1], hspace=0.3)
        
        data = ticker_data['Data']
        ticker = ticker_data['Ticker']
        
        if data is None or len(data) == 0:
            st.error(f"No hay datos disponibles para {ticker}")
            return None
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in data.columns:
                data[col] = ensure_series(data[col])
        
        # PANEL 1: PRECIO + BOLLINGER BANDS + SMA 200
        ax1 = fig.add_subplot(gs[0])
        ax1.set_facecolor('#1A1D29')
        
        bb_lower = ensure_series(ticker_data.get('BB_Lower'))
        bb_upper = ensure_series(ticker_data.get('BB_Upper'))
        bb_sma = ensure_series(ticker_data.get('BB_SMA'))
        sma_200 = ensure_series(ticker_data.get('SMA_200_Series'))
        
        if bb_lower is not None and bb_upper is not None and len(bb_lower) > 0 and len(bb_upper) > 0:
            ax1.fill_between(data.index, bb_lower, bb_upper,
                             color='#FFB86C', alpha=0.1, label='Bollinger Bands (2.5σ)', zorder=1)
            ax1.plot(data.index, bb_upper, color='#FFB86C', 
                    linewidth=2, linestyle='--', alpha=0.7, zorder=2)
            if bb_sma is not None and len(bb_sma) > 0:
                ax1.plot(data.index, bb_sma, color='#BD93F9', 
                        linewidth=2.5, label='SMA(20)', zorder=2)
            ax1.plot(data.index, bb_lower, color='#FFB86C', 
                    linewidth=2, linestyle='--', alpha=0.7, zorder=2)
        
        # Añadir SMA 200
        if sma_200 is not None and len(sma_200) > 0:
            ax1.plot(data.index, sma_200, color='#00D9FF', 
                    linewidth=3, label='SMA(200)', zorder=2, alpha=0.8)
        
        close_series = ensure_series(data['Close'])
        if close_series is not None and len(close_series) > 0:
            ax1.plot(data.index, close_series, color='#FFFFFF', linewidth=2.5, 
                    label='Precio', zorder=3)
            
            current_color = '#FF6B6B' if ticker_data['Type'] == 'SOBRECOMPRA' else '#4ECDC4'
            ax1.scatter(data.index[-1], close_series.iloc[-1], 
                       color=current_color, s=200, edgecolors='white', 
                       linewidth=3, zorder=10, label='Actual')
        
        ax1.set_ylabel('Precio ($)', fontsize=13, fontweight='bold', color='#FFFFFF')
        ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
        ax1.grid(True, alpha=0.1, linestyle='-', linewidth=0.8)
        ax1.tick_params(labelsize=10, colors='#B0B0B0', labelbottom=False)
        
        for spine in ax1.spines.values():
            spine.set_color('#2D3142')
            spine.set_linewidth(1.5)
        
        # PANEL 2: Z-SCORE
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.set_facecolor('#1A1D29')
        
        zscore = ensure_series(ticker_data.get('ZScore_Series'))
        
        if zscore is not None and len(zscore) > 0:
            if len(zscore) != len(data.index):
                zscore = zscore.iloc[:len(data.index)]
            
            for i in range(1, len(zscore)):
                if pd.notna(zscore.iloc[i-1]) and pd.notna(zscore.iloc[i]):
                    y2 = float(zscore.iloc[i])
                    if abs(y2) > 3:
                        color, width = '#FF6B6B', 3.5
                    elif abs(y2) > 2.5:
                        color, width = '#FFB86C', 3
                    elif abs(y2) > 2:
                        color, width = '#FFD93D', 2.5
                    else:
                        color, width = '#95A5A6', 2
                    
                    ax2.plot([data.index[i-1], data.index[i]], 
                            [float(zscore.iloc[i-1]), y2], 
                            color=color, linewidth=width, alpha=0.95, zorder=5)
            
            ax2.axhline(y=3, color='#FF6B6B', linestyle='--', linewidth=2, alpha=0.8, label='±3σ')
            ax2.axhline(y=2.5, color='#FFB86C', linestyle='--', linewidth=2, alpha=0.8, label='±2.5σ')
            ax2.axhline(y=0, color='#8E93A1', linestyle='-', linewidth=1.5, alpha=0.7)
            ax2.axhline(y=-2.5, color='#FFB86C', linestyle='--', linewidth=2, alpha=0.8)
            ax2.axhline(y=-3, color='#FF6B6B', linestyle='--', linewidth=2, alpha=0.8)
            
            ax2.fill_between(data.index, 2.5, 5, alpha=0.12, color='#FFB86C', zorder=0)
            ax2.fill_between(data.index, -5, -2.5, alpha=0.12, color='#FFB86C', zorder=0)
            ax2.fill_between(data.index, 3, 5, alpha=0.15, color='#FF6B6B', zorder=0)
            ax2.fill_between(data.index, -5, -3, alpha=0.15, color='#FF6B6B', zorder=0)
            
            current_zscore = ticker_data['Z-Score']
            zscore_color = '#FF6B6B' if abs(current_zscore) > 3 else '#FFB86C' if abs(current_zscore) > 2.5 else '#FFD93D'
            ax2.text(0.02, 0.95, f'Z-Score: {current_zscore:.2f}σ', 
                    transform=ax2.transAxes, fontsize=13, fontweight='bold', 
                    color='white', verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.7', facecolor=zscore_color, 
                             alpha=0.95, edgecolor='white', linewidth=2))
        
        ax2.set_ylabel('Z-Score (σ)', fontsize=12, fontweight='bold', color='#FFFFFF')
        ax2.set_ylim([-5, 5])
        ax2.legend(loc='upper right', fontsize=9, framealpha=0.9, ncol=2)
        ax2.grid(True, alpha=0.1, linestyle=':', linewidth=1)
        ax2.tick_params(labelsize=10, colors='#B0B0B0', labelbottom=False)
        
        for spine in ax2.spines.values():
            spine.set_color('#2D3142')
            spine.set_linewidth(1.5)
        
        # PANEL 3: MACD-V
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.set_facecolor('#1A1D29')
        
        macd_v = ensure_series(ticker_data.get('MACD_V_Series'))
        signal = ensure_series(ticker_data.get('Signal_Series'))
        
        if macd_v is not None and len(macd_v) > 0 and not macd_v.isna().all():
            if len(macd_v) != len(data.index):
                macd_v = macd_v.iloc[:len(data.index)]
            
            for i in range(1, len(macd_v)):
                if pd.notna(macd_v.iloc[i-1]) and pd.notna(macd_v.iloc[i]):
                    y2 = float(macd_v.iloc[i])
                    if y2 > 150:
                        color, width = '#FF6B6B', 3.5
                    elif y2 > 50:
                        color, width = '#4ECDC4', 3
                    elif y2 > -50:
                        color, width = '#95A5A6', 2.5
                    elif y2 > -150:
                        color, width = '#EE5A6F', 3
                    else:
                        color, width = '#FF6B6B', 3.5
                    
                    ax3.plot([data.index[i-1], data.index[i]], 
                            [float(macd_v.iloc[i-1]), y2], 
                            color=color, linewidth=width, alpha=0.95, zorder=5)
            
            if signal is not None and len(signal) > 0:
                if len(signal) != len(data.index):
                    signal = signal.iloc[:len(data.index)]
                ax3.plot(data.index, signal, color='#FFB86C', linewidth=1.5, 
                        alpha=0.5, linestyle='--', zorder=3)
            
            ax3.axhline(y=150, color='#FF6B6B', linestyle='--', linewidth=2, alpha=0.8)
            ax3.axhline(y=50, color='#4ECDC4', linestyle='--', linewidth=2, alpha=0.8, label='±50')
            ax3.axhline(y=0, color='#8E93A1', linestyle='-', linewidth=1.5, alpha=0.7)
            ax3.axhline(y=-50, color='#EE5A6F', linestyle='--', linewidth=2, alpha=0.8)
            ax3.axhline(y=-150, color='#FF6B6B', linestyle='--', linewidth=2, alpha=0.8)
            
            current_macdv = ticker_data['MACD_V']
            macd_color = '#FF6B6B' if abs(current_macdv) > 150 else '#4ECDC4' if current_macdv > 50 else '#EE5A6F' if current_macdv < -50 else '#95A5A6'
            confirm_text = " ✓" if ticker_data['MACDV_Confirm'] else ""
            ax3.text(0.02, 0.95, f'MACD-V: {current_macdv:.1f}{confirm_text}', 
                    transform=ax3.transAxes, fontsize=13, fontweight='bold', 
                    color='white', verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.7', facecolor=macd_color, 
                             alpha=0.95, edgecolor='white', linewidth=2))
        
        ax3.set_ylabel('MACD-V', fontsize=12, fontweight='bold', color='#FFFFFF')
        ax3.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax3.grid(True, alpha=0.1, linestyle=':', linewidth=1)
        ax3.tick_params(labelsize=10, colors='#B0B0B0', labelbottom=False)
        
        for spine in ax3.spines.values():
            spine.set_color('#2D3142')
            spine.set_linewidth(1.5)
        
        # PANEL 4: VOLUMEN
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        ax4.set_facecolor('#1A1D29')
        
        volume = ensure_series(data['Volume'])
        if volume is not None and len(volume) > 0:
            colors = ['#4ECDC4' if close_series.iloc[i] >= close_series.iloc[i-1] else '#FF6B6B' 
                      for i in range(1, len(close_series))]
            colors.insert(0, '#95A5A6')
            
            ax4.bar(data.index, volume, color=colors, alpha=0.6, width=0.8)
            
            vol_avg = volume.rolling(window=20).mean()
            ax4.plot(data.index, vol_avg, color='#FFB86C', linewidth=2, 
                    label='Vol Avg (20)', alpha=0.8)
        
        ax4.set_ylabel('Volumen', fontsize=12, fontweight='bold', color='#FFFFFF')
        ax4.set_xlabel('Fecha', fontsize=13, fontweight='bold', color='#FFFFFF')
        ax4.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax4.grid(True, alpha=0.1, linestyle=':', linewidth=1)
        ax4.tick_params(labelsize=10, colors='#B0B0B0')
        
        for spine in ax4.spines.values():
            spine.set_color('#2D3142')
            spine.set_linewidth(1.5)
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        st.error(f"Error generando gráfico: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

def main_app():
    """Aplicación principal del screener"""
    st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stButton>button { 
        background: linear-gradient(135deg, #4ECDC4 0%, #00D9FF 100%); 
        color: white; 
        font-weight: 700; 
        border: none; 
        padding: 10px 20px; 
        border-radius: 8px; 
    }
    h1, h2, h3 { color: #FFFFFF !important; font-weight: 800 !important; }
    .stNumberInput > div > div > input { background-color: #1A1D29; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎯 Mean Reversion Screener - Bollinger + Z-Score")
    st.markdown("**Detecta valores sobreextendidos estadísticamente**")
    st.markdown("---")
    
    # ============= PASO 1: OBTENER TICKERS AUTOMÁTICAMENTE =============
    st.markdown("### 📊 PASO 1: Universo de Tickers")
    
    # Inicializar el universo de tickers si no existe
    if 'tickers_universe' not in st.session_state:
        with st.spinner("📥 Descargando universo de tickers (Russell 1000 + ETFs + Índices)..."):
            try:
                df_tickers = create_tickers_universe()
                
                if df_tickers is not None and len(df_tickers) > 0:
                    st.session_state['tickers_universe'] = df_tickers['Ticker'].tolist()
                    st.session_state['tickers_info'] = {
                        'total': len(df_tickers),
                        'stocks': len(df_tickers[df_tickers['Type'] == 'Stock']) if 'Type' in df_tickers.columns else 0,
                        'etfs': len(df_tickers[df_tickers['Type'] == 'ETF']) if 'Type' in df_tickers.columns else 0,
                        'indices': len(df_tickers[df_tickers['Type'] == 'Index']) if 'Type' in df_tickers.columns else 0,
                        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    st.success(f"✅ {len(df_tickers)} tickers cargados correctamente!")
                else:
                    st.error("❌ Error al descargar los tickers")
                    st.session_state['tickers_universe'] = []
            except Exception as e:
                st.error(f"❌ Error descargando tickers: {str(e)}")
                st.session_state['tickers_universe'] = []
    
    # Mostrar información del universo de tickers
    if 'tickers_info' in st.session_state:
        info = st.session_state['tickers_info']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tickers", f"{info['total']:,}")
        with col2:
            st.metric("Stocks", f"{info['stocks']:,}")
        with col3:
            st.metric("ETFs", f"{info['etfs']:,}")
        with col4:
            st.metric("Índices", f"{info['indices']:,}")
        
        st.caption(f"📅 Última actualización: {info['date']}")
    
    # NUEVA SECCIÓN: Selección de subconjunto aleatorio
    st.markdown("---")
    st.markdown("#### 🎲 Subconjunto Aleatorio (Opcional)")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        use_random_subset = st.checkbox(
            "Usar subconjunto aleatorio",
            value=False,
            help="Seleccionar un número aleatorio de tickers para un escaneo más rápido"
        )
    
    with col2:
        if use_random_subset and 'tickers_universe' in st.session_state:
            total_tickers = len(st.session_state['tickers_universe'])
            random_count = st.slider(
                "Número de tickers a escanear",
                min_value=1,
                max_value=total_tickers,
                value=min(100, total_tickers),
                step=1,
                help="Selecciona cuántos tickers aleatorios quieres analizar"
            )
        else:
            random_count = 0
    
    with col3:
        if use_random_subset and st.button("🔀 Aleatorizar", help="Generar nueva selección aleatoria"):
            if 'random_seed' in st.session_state:
                st.session_state['random_seed'] += 1
            else:
                st.session_state['random_seed'] = 1
            st.rerun()
    
    # Botón para recargar el universo de tickers
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Recargar Tickers", help="Volver a descargar el universo de tickers"):
            if 'tickers_universe' in st.session_state:
                del st.session_state['tickers_universe']
            if 'tickers_info' in st.session_state:
                del st.session_state['tickers_info']
            if 'random_seed' in st.session_state:
                del st.session_state['random_seed']
            st.rerun()
    
    st.markdown("---")
    
    # ============= PASO 2: CONFIGURACIÓN DEL ESCANEO =============
    st.markdown("### ⚙️ PASO 2: Configuración del Escaneo")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📅 Datos Históricos**")
        period = st.selectbox("Periodo", ["3mo", "6mo", "1y", "2y"], index=1)
        interval = st.selectbox("Intervalo", ["1d", "1wk"], index=0)
        
        st.markdown("**📊 Periodos de Cálculo**")
        bb_period = st.number_input("Periodo Bollinger", min_value=5, max_value=50, value=20, step=1)
        zscore_period = st.number_input("Periodo Z-Score", min_value=5, max_value=50, value=20, step=1)
    
    with col2:
        st.markdown("**📈 Filtros Estadísticos**")
        
        # NUEVO: Checkboxes para seleccionar tipo de señal
        col2a, col2b = st.columns(2)
        with col2a:
            filter_zscore_positive = st.checkbox(
                "🔴 Sobrecompra",
                value=True,
                help="Buscar señales de SOBRECOMPRA (Z-Score positivo alto)"
            )
        with col2b:
            filter_zscore_negative = st.checkbox(
                "🟢 Sobreventa",
                value=True,
                help="Buscar señales de SOBREVENTA (Z-Score negativo bajo)"
            )
        
        zscore_threshold_pos = st.number_input(
            "Z-Score Positivo (≥)", 
            min_value=1.0, 
            max_value=5.0, 
            value=2.5, 
            step=0.1,
            help="Mínimo Z-Score positivo para SOBRECOMPRA",
            disabled=not filter_zscore_positive
        )
        
        zscore_threshold_neg = st.number_input(
            "Z-Score Negativo (≤)", 
            min_value=-5.0, 
            max_value=-1.0, 
            value=-2.5, 
            step=0.1,
            help="Máximo Z-Score negativo para SOBREVENTA",
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
            min_value=0, 
            max_value=200, 
            value=50, 
            step=10,
            help="Valor absoluto mínimo de MACD-V"
        )
        
        filter_sma200 = st.checkbox(
            "Filtrar por encima SMA 200",
            value=False,
            help="Solo mostrar valores cuyo precio esté por encima de la SMA de 200 períodos"
        )
        
        st.markdown("**⚡ Rendimiento**")
        max_workers = st.slider("Threads paralelos", 3, 10, 5, 1,
                               help="Recomendado: 3-5 threads")
    
    st.markdown("---")
    
    # ============= PASO 3: ESCANEAR =============
    st.markdown("### 🚀 PASO 3: Escanear")
    
    # Validación: al menos un tipo de señal debe estar activo
    if not filter_zscore_positive and not filter_zscore_negative:
        st.warning("⚠️ Debes seleccionar al menos un tipo de señal (Sobrecompra o Sobreventa)")
    
    scan_button = st.button(
        "🚀 INICIAR ESCANEO", 
        type="primary", 
        use_container_width=True,
        disabled=not filter_zscore_positive and not filter_zscore_negative
    )
    
    if scan_button:
        # Verificar que tengamos el universo de tickers
        if 'tickers_universe' not in st.session_state or not st.session_state['tickers_universe']:
            st.error("❌ No hay tickers disponibles para escanear")
            st.warning("⚠️ Por favor, recarga los tickers usando el botón 'Recargar Tickers'")
            st.stop()
        
        # Seleccionar tickers (todos o subconjunto aleatorio)
        all_tickers = st.session_state['tickers_universe']
        
        if use_random_subset and random_count > 0:
            # Usar seed para reproducibilidad durante la misma sesión
            seed = st.session_state.get('random_seed', 42)
            random.seed(seed)
            tickers_list = random.sample(all_tickers, min(random_count, len(all_tickers)))
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
            
            signal_types = []
            if filter_zscore_positive:
                signal_types.append("Sobrecompra")
            if filter_zscore_negative:
                signal_types.append("Sobreventa")
            
            st.info(f"""
            **💡 Sugerencias para encontrar más resultados:**
            
            **Señales activas:** {', '.join(signal_types)}
            
            1️⃣ **Relajar Z-Score Positivo:** Actual = {zscore_threshold_pos}σ → Prueba con 2.0σ
            
            2️⃣ **Relajar Z-Score Negativo:** Actual = {zscore_threshold_neg}σ → Prueba con -2.0σ
            
            3️⃣ **Relajar MACD-V:** Actual = {macdv_threshold} → Prueba con 30 o 0
            
            4️⃣ **Desactivar filtro SMA 200:** {'Activo' if filter_sma200 else 'Inactivo'}
            
            5️⃣ **Relajar BB %B:** Actual = {bb_threshold:.2f} → Prueba con 0.05 o 0.0
            
            6️⃣ **Cambiar periodo:** Actual = {period} → Prueba con "3mo" o "1y"
            
            7️⃣ **Activar ambos tipos de señal** si solo tienes uno activo
            
            📊 Recuerda: Los filtros detectan sobreextensión **estadísticamente significativa**
            """)
    
    st.markdown("---")
    
    # ============= PASO 4: RESULTADOS =============
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
                'SMA 200': round(r['SMA_200'], 2) if r.get('SMA_200') else None,
                'Confirm': '✓' if r['MACDV_Confirm'] else ''
            }
            for r in results
        ])
        
        df_results = df_results.sort_values('Fuerza (σ)', ascending=False).reset_index(drop=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Detectados", len(results))
        with col2:
            sobrecompra = len([r for r in results if r['Type'] == 'SOBRECOMPRA'])
            st.metric("Sobrecompra", sobrecompra, delta_color="inverse")
        with col3:
            sobreventa = len([r for r in results if r['Type'] == 'SOBREVENTA'])
            st.metric("Sobreventa", sobreventa, delta_color="normal")
        
        st.markdown("---")
        
        st.markdown("#### 📋 Click en un ticker para ver el gráfico")
        
        event = st.dataframe(
            df_results,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                "Compañía": st.column_config.TextColumn("Compañía", width="large"),
                "Tipo": st.column_config.TextColumn("Tipo", width="medium"),
                "Fuerza (σ)": st.column_config.NumberColumn("Fuerza (σ)", format="%.2f"),
                "Z-Score": st.column_config.NumberColumn("Z-Score", format="%.2f"),
                "BB %B": st.column_config.NumberColumn("BB %B", format="%.2f"),
                "MACD-V": st.column_config.NumberColumn("MACD-V", format="%.1f"),
                "Precio": st.column_config.NumberColumn("Precio", format="$%.2f"),
                "SMA 200": st.column_config.NumberColumn("SMA 200", format="$%.2f"),
                "Confirm": st.column_config.TextColumn("Confirm", width="small"),
            }
        )
        
        st.markdown("---")
        
        if len(event.selection.rows) > 0:
            selected_idx = event.selection.rows[0]
            selected_ticker = df_results.iloc[selected_idx]['Ticker']
            ticker_data = next((r for r in results if r['Ticker'] == selected_ticker), None)
            
            if ticker_data:
                st.markdown(f"### 📈 Análisis Detallado: {selected_ticker}")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Precio", f"${ticker_data['Price']:.2f}")
                with col2:
                    st.metric("Z-Score", f"{ticker_data['Z-Score']:.2f}σ")
                with col3:
                    st.metric("BB %B", f"{ticker_data['BB_%B']:.2f}")
                with col4:
                    confirm_icon = "✅" if ticker_data['MACDV_Confirm'] else "⚪"
                    st.metric("MACD-V", f"{ticker_data['MACD_V']:.0f} {confirm_icon}")
                with col5:
                    if ticker_data.get('SMA_200'):
                        st.metric("SMA 200", f"${ticker_data['SMA_200']:.2f}")
                    else:
                        st.metric("Tipo", ticker_data['Type'])
                
                with st.spinner(f"Generando gráfico para {selected_ticker}..."):
                    try:
                        fig = plot_ticker_analysis(ticker_data)
                        if fig:
                            st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error al generar gráfico: {str(e)}")
        else:
            st.info("👆 Selecciona un ticker en la tabla para ver su gráfico detallado")
    
    else:
        st.markdown("""
        <div style='text-align: center; padding: 60px 20px;
                    background: linear-gradient(135deg, #1A1D29 0%, #2D3142 100%);
                    border-radius: 20px; border: 3px solid #4ECDC4;
                    margin-top: 40px;'>
            <h2 style='color: #4ECDC4; margin: 0;'>
                🎯 Mean Reversion Screener
            </h2>
            <p style='color: #B0B0B0; font-size: 18px; margin: 20px 0;'>
                Sigue los 3 pasos para comenzar
            </p>
            <p style='color: #8E93A1; font-size: 14px;'>
                1️⃣ Carga automática de tickers (Russell 1000 + ETFs + Índices)<br>
                   🎲 Opción de subconjunto aleatorio para análisis rápido<br>
                2️⃣ Configura los parámetros del escaneo<br>
                   🔴 Selecciona Sobrecompra, 🟢 Sobreventa, o ambos<br>
                3️⃣ Inicia el escaneo y revisa resultados
            </p>
            <p style='color: #8E93A1; font-size: 14px; margin-top: 20px;'>
                ✨ Bollinger Bands (2.5σ)<br>
                ✨ Z-Score estadístico configurable<br>
                ✨ MACD-V confirmación ajustable<br>
                ✨ Filtro SMA 200 opcional<br>
                ✨ Alta probabilidad de reversión
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    if check_password():
        main_app()
    else:
        st.markdown("""
        <div style='text-align: center; padding: 60px 20px;'>
            <h1 style='color: #FF6B6B; font-size: 48px;'>🔒 Acceso Restringido</h1>
            <p style='color: #B0B0B0; font-size: 20px; margin-top: 20px;'>
                Introduce tus credenciales en el menú lateral.
            </p>
        </div>
        """, unsafe_allow_html=True)
