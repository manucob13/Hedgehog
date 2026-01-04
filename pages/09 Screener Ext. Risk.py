import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from utils import check_password

warnings.filterwarnings('ignore')

# Lock global para sincronizar descargas de yfinance
_yfinance_lock = Lock()

st.set_page_config(
    page_title="Mean Reversion Screener",
    page_icon="🎯",
    layout="wide"
)

# ============= FUNCIONES TÉCNICAS =============
def calculate_ema(data, period):
    """Calcula EMA"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_sma(data, period):
    """Calcula SMA"""
    return data.rolling(window=period).mean()

def calculate_bollinger_bands(df, period=20, std_dev=2.5):
    """Calcula Bollinger Bands y %B"""
    sma = calculate_sma(df['Close'], period)
    std = df['Close'].rolling(window=period).std()
    
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    
    # %B: Posición relativa dentro de las bandas
    percent_b = (df['Close'] - lower_band) / (upper_band - lower_band)
    
    return sma, upper_band, lower_band, percent_b

def calculate_zscore(df, period=20):
    """Calcula Z-Score: (Precio - Media) / Desviación Estándar"""
    sma = calculate_sma(df['Close'], period)
    std = df['Close'].rolling(window=period).std()
    
    zscore = (df['Close'] - sma) / std
    
    return zscore

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

def analyze_ticker(ticker, period="6mo", interval="1d", bb_period=20, zscore_period=20, 
                   zscore_threshold=2.5, bb_threshold_upper=1.1, bb_threshold_lower=-0.1, 
                   use_lock=True):
    """Analiza un ticker buscando sobreextensión estadística"""
    try:
        # SINCRONIZAR descarga para evitar race conditions en yfinance (si use_lock=True)
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
            # Sin lock (más rápido pero puede tener problemas)
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
        
        if data.empty:
            return None
            
        if len(data) < 50:
            return None
        
        # Asegurar que tenemos las columnas necesarias
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            return None
        
        # Obtener nombre de la compañía (también con lock)
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
        
        # Calcular indicadores (trabajar sobre copias)
        df_copy = data.copy(deep=True)
        bb_sma, bb_upper, bb_lower, bb_percent = calculate_bollinger_bands(
            df_copy, period=bb_period, std_dev=2.5
        )
        zscore = calculate_zscore(df_copy, period=zscore_period)
        macd_v, macd_v_signal = calculate_macd_v(df_copy)
        
        # FIX: Separar las verificaciones de None y .isna().all()
        if zscore is None:
            return None
        if len(zscore) == 0:
            return None
        # Convertir explícitamente a bool para evitar ambigüedad
        if bool(zscore.isna().all()):
            return None
        
        # Hacer copias independientes de las series antes de extraer valores
        bb_sma = bb_sma.copy(deep=True)
        bb_upper = bb_upper.copy(deep=True)
        bb_lower = bb_lower.copy(deep=True)
        bb_percent = bb_percent.copy(deep=True)
        zscore = zscore.copy(deep=True)
        if macd_v is not None:
            macd_v = macd_v.copy(deep=True)
        if macd_v_signal is not None:
            macd_v_signal = macd_v_signal.copy(deep=True)
        
        # Valores actuales - convertir a tipos nativos de Python (no numpy)
        current_price = float(data['Close'].iloc[-1])
        current_zscore = float(zscore.iloc[-1])
        current_bb_percent = float(bb_percent.iloc[-1])
        current_macdv = float(macd_v.iloc[-1]) if macd_v is not None and not pd.isna(macd_v.iloc[-1]) else 0.0
        current_signal = float(macd_v_signal.iloc[-1]) if macd_v_signal is not None and not pd.isna(macd_v_signal.iloc[-1]) else 0.0
        
        # Validar que los valores son numéricos y razonables
        if not all(pd.notna([current_price, current_zscore, current_bb_percent])):
            return None
        
        if current_price <= 0:
            return None
        
        # FILTRO PRINCIPAL: Z-Score >= 2.5 o <= -2.5
        zscore_check = abs(current_zscore) >= zscore_threshold
        if not zscore_check:
            return None
        
        # FILTRO SECUNDARIO: Bollinger %B fuera de bandas
        bb_check = current_bb_percent >= bb_threshold_upper or current_bb_percent <= bb_threshold_lower
        if not bb_check:
            return None
        
        # Determinar tipo de señal
        if current_zscore > zscore_threshold and current_bb_percent > bb_threshold_upper:
            signal_type = 'SOBRECOMPRA'
            signal_strength = min(current_zscore, 5.0)
        elif current_zscore < -zscore_threshold and current_bb_percent < bb_threshold_lower:
            signal_type = 'SOBREVENTA'
            signal_strength = min(abs(current_zscore), 5.0)
        else:
            return None
        
        # Confirmación MACD-V
        macdv_confirms = False
        if signal_type == 'SOBRECOMPRA' and current_macdv > 150:
            macdv_confirms = True
        elif signal_type == 'SOBREVENTA' and current_macdv < -150:
            macdv_confirms = True
        
        # Timestamp único para identificar este análisis específico
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        
        return {
            'Ticker': str(ticker),
            'Company': str(company_name),
            'Price': current_price,
            'Z-Score': current_zscore,
            'BB_%B': current_bb_percent,
            'MACD_V': current_macdv,
            'Signal': current_signal,
            'Type': str(signal_type),
            'Strength': signal_strength,
            'MACDV_Confirm': bool(macdv_confirms),
            'Date': data.index[-1],
            'Data': data.copy(deep=True),
            'BB_SMA': bb_sma.copy(deep=True),
            'BB_Upper': bb_upper.copy(deep=True),
            'BB_Lower': bb_lower.copy(deep=True),
            'ZScore_Series': zscore.copy(deep=True),
            'MACD_V_Series': macd_v.copy(deep=True) if macd_v is not None else None,
            'Signal_Series': macd_v_signal.copy(deep=True) if macd_v_signal is not None else None,
            'ID': f"{ticker}_{timestamp}"
        }
    except Exception as e:
        # Log error con más detalle
        print(f"Error analyzing {ticker}: {type(e).__name__}: {str(e)}")
        raise

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
    
    if st.session_state.get('debug_mode', False) and len(filtered_out) > 0:
        with st.expander(f"🔍 Tickers filtrados - Muestra ({min(10, len(filtered_out))} de {len(filtered_out)})"):
            st.write("Estos tickers se descargaron correctamente pero no cumplieron los criterios:")
            for ticker in filtered_out[:10]:
                st.caption(f"- {ticker}")
    
    return results

def plot_ticker_analysis(ticker_data):
    """Genera gráfico completo con 4 paneles"""
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 14), facecolor='#0E1117')
    gs = fig.add_gridspec(4, 1, height_ratios=[2.5, 1, 1, 1], hspace=0.3)
    
    data = ticker_data['Data']
    ticker = ticker_data['Ticker']
    company_name = ticker_data.get('Company', ticker)
    
    if data is None or len(data) == 0:
        st.error(f"No hay datos disponibles para {ticker}")
        return None
    
    # PANEL 1: PRECIO + BOLLINGER BANDS
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#1A1D29')
    
    bb_lower = ticker_data.get('BB_Lower')
    bb_upper = ticker_data.get('BB_Upper')
    bb_sma = ticker_data.get('BB_SMA')
    
    if bb_lower is not None and bb_upper is not None:
        ax1.fill_between(data.index, bb_lower, bb_upper,
                         color='#FFB86C', alpha=0.1, label='Bollinger Bands (2.5σ)', zorder=1)
        ax1.plot(data.index, bb_upper, color='#FFB86C', 
                linewidth=2, linestyle='--', alpha=0.7, zorder=2)
        ax1.plot(data.index, bb_sma, color='#BD93F9', 
                linewidth=2.5, label='SMA(20)', zorder=2)
        ax1.plot(data.index, bb_lower, color='#FFB86C', 
                linewidth=2, linestyle='--', alpha=0.7, zorder=2)
    
    ax1.plot(data.index, data['Close'], color='#FFFFFF', linewidth=2.5, 
            label='Precio', zorder=3)
    
    current_color = '#FF6B6B' if ticker_data['Type'] == 'SOBRECOMPRA' else '#4ECDC4'
    ax1.scatter(data.index[-1], data['Close'].iloc[-1], 
               color=current_color, s=200, edgecolors='white', 
               linewidth=3, zorder=10, label='Actual')
    
    ax1.set_title(f'{ticker} - {company_name}\n${ticker_data["Price"]:.2f} | {ticker_data["Type"]} ({ticker_data["Strength"]:.1f}σ)', 
                  fontsize=18, fontweight='bold', color='#FFFFFF', pad=15)
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
    
    zscore = ticker_data.get('ZScore_Series')
    
    if zscore is not None and len(zscore) > 0:
        for i in range(1, len(zscore)):
            if pd.notna(zscore.iloc[i-1]) and pd.notna(zscore.iloc[i]):
                y2 = zscore.iloc[i]
                if abs(y2) > 3:
                    color, width = '#FF6B6B', 3.5
                elif abs(y2) > 2.5:
                    color, width = '#FFB86C', 3
                elif abs(y2) > 2:
                    color, width = '#FFD93D', 2.5
                else:
                    color, width = '#95A5A6', 2
                
                ax2.plot([data.index[i-1], data.index[i]], 
                        [zscore.iloc[i-1], y2], 
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
    
    macd_v = ticker_data.get('MACD_V_Series')
    signal = ticker_data.get('Signal_Series')
    
    if macd_v is not None and len(macd_v) > 0 and not macd_v.isna().all():
        for i in range(1, len(macd_v)):
            if pd.notna(macd_v.iloc[i-1]) and pd.notna(macd_v.iloc[i]):
                y2 = macd_v.iloc[i]
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
                        [macd_v.iloc[i-1], y2], 
                        color=color, linewidth=width, alpha=0.95, zorder=5)
        
        if signal is not None and len(signal) > 0:
            ax3.plot(data.index, signal, color='#FFB86C', linewidth=1.5, 
                    alpha=0.5, linestyle='--', zorder=3)
        
        ax3.axhline(y=150, color='#FF6B6B', linestyle='--', linewidth=2, alpha=0.8)
        ax3.axhline(y=0, color='#8E93A1', linestyle='-', linewidth=1.5, alpha=0.7)
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
    ax3.grid(True, alpha=0.1, linestyle=':', linewidth=1)
    ax3.tick_params(labelsize=10, colors='#B0B0B0', labelbottom=False)
    
    for spine in ax3.spines.values():
        spine.set_color('#2D3142')
        spine.set_linewidth(1.5)
    
    # PANEL 4: BOLLINGER %B
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.set_facecolor('#1A1D29')
    
    bb_percent = ticker_data.get('BB_%B')
    
    if bb_percent is not None and hasattr(bb_percent, '__len__') and len(bb_percent) > 0:
        for i in range(1, len(bb_percent)):
            if pd.notna(bb_percent.iloc[i-1]) and pd.notna(bb_percent.iloc[i]):
                y2 = bb_percent.iloc[i]
                if y2 > 1.2:
                    color, width = '#FF6B6B', 3.5
                elif y2 > 1.0:
                    color, width = '#FFB86C', 3
                elif y2 < -0.2:
                    color, width = '#FF6B6B', 3.5
                elif y2 < 0:
                    color, width = '#4ECDC4', 3
                else:
                    color, width = '#95A5A6', 2
                
                ax4.plot([data.index[i-1], data.index[i]], 
                        [bb_percent.iloc[i-1], y2], 
                        color=color, linewidth=width, alpha=0.95, zorder=5)
        
        ax4.axhline(y=1.0, color='#FFB86C', linestyle='--', linewidth=2, alpha=0.8, label='Banda Superior')
        ax4.axhline(y=0.5, color='#BD93F9', linestyle='-', linewidth=1.5, alpha=0.7, label='Media')
        ax4.axhline(y=0.0, color='#4ECDC4', linestyle='--', linewidth=2, alpha=0.8, label='Banda Inferior')
        
        ax4.fill_between(data.index, 1.0, 1.5, alpha=0.12, color='#FFB86C', zorder=0)
        ax4.fill_between(data.index, -0.5, 0.0, alpha=0.12, color='#4ECDC4', zorder=0)
        
        current_bb = ticker_data.get('BB_%B', 0)
        bb_color = '#FF6B6B' if current_bb > 1.1 or current_bb < -0.1 else '#FFB86C' if current_bb > 1.0 else '#4ECDC4' if current_bb < 0 else '#95A5A6'
        ax4.text(0.02, 0.95, f'%B: {current_bb:.2f}', 
                transform=ax4.transAxes, fontsize=13, fontweight='bold', 
                color='white', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.7', facecolor=bb_color, 
                         alpha=0.95, edgecolor='white', linewidth=2))
    
    ax4.set_ylabel('Bollinger %B', fontsize=12, fontweight='bold', color='#FFFFFF')
    ax4.set_xlabel('Fecha', fontsize=13, fontweight='bold', color='#FFFFFF')
    ax4.set_ylim([-0.5, 1.5])
    ax4.legend(loc='upper right', fontsize=9, framealpha=0.9, ncol=3)
    ax4.grid(True, alpha=0.1, linestyle=':', linewidth=1)
    ax4.tick_params(labelsize=10, colors='#B0B0B0')
    
    for spine in ax4.spines.values():
        spine.set_color('#2D3142')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    return fig

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
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎯 Mean Reversion Screener - Bollinger + Z-Score")
    st.markdown("**Detecta valores sobreextendidos estadísticamente (2.5σ)**")
    st.markdown("---")
    
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 20px; 
                    background: linear-gradient(135deg, #4ECDC4 0%, #00D9FF 100%); 
                    border-radius: 15px; margin-bottom: 20px;'>
            <h2 style='color: white; margin: 0;'>⚙️ Configuración</h2>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 RESET", type="secondary", use_container_width=True):
            for key in ['scan_results', 'selected_ticker']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        st.markdown("---")
        
        period = st.selectbox("Periodo Histórico", ["3mo", "6mo", "1y", "2y"], index=1)
        interval = st.selectbox("Intervalo", ["1d", "1wk"], index=0)
        
        st.markdown("---")
        st.subheader("📊 Filtros Estadísticos")
        
        zscore_threshold = st.slider("Z-Score Mínimo (σ)", 2.0, 4.0, 2.5, 0.1)
        bb_threshold = st.slider("BB %B Threshold", 0.0, 0.5, 0.1, 0.05)
        
        bb_threshold_upper = 1.0 + bb_threshold
        bb_threshold_lower = 0.0 - bb_threshold
        
        st.caption(f"📊 Superior: %B > {bb_threshold_upper:.2f} | Inferior: %B < {bb_threshold_lower:.2f}")
        
        st.markdown("---")
        
        max_workers = st.slider("Threads paralelos", 3, 10, 5, 1,
                               help="Recomendado: 3-5 threads (con sincronización de descargas)")
        
        st.markdown("---")
        
        debug_mode = st.checkbox("🐛 Modo Debug", value=False, 
                                help="Muestra información adicional para diagnóstico")
        
        if debug_mode:
            disable_lock = st.checkbox("⚠️ Deshabilitar Lock (experimental)", value=False,
                                      help="Puede causar duplicados pero es más rápido")
            if disable_lock:
                st.session_state['disable_lock'] = True
            elif 'disable_lock' in st.session_state:
                del st.session_state['disable_lock']
        
        if debug_mode:
            st.session_state['debug_mode'] = True
        elif 'debug_mode' in st.session_state:
            del st.session_state['debug_mode']
        
        st.markdown("---")
        
        if debug_mode:
            st.markdown("#### 🔬 Test Individual")
            test_ticker = st.text_input("Ticker a probar:", "AAPL")
            if st.button("🧪 Probar Ticker", use_container_width=True):
                with st.spinner(f"Analizando {test_ticker}..."):
                    result = analyze_ticker(
                        test_ticker,
                        period=period,
                        interval=interval,
                        zscore_threshold=zscore_threshold,
                        bb_threshold_upper=bb_threshold_upper,
                        bb_threshold_lower=bb_threshold_lower,
                        use_lock=False
                    )
                    
                    if result:
                        st.success(f"✅ {test_ticker} PASA los filtros!")
                        st.json({
                            'Z-Score': result['Z-Score'],
                            'BB_%B': result['BB_%B'],
                            'MACD_V': result['MACD_V'],
                            'Type': result['Type'],
                            'Strength': result['Strength']
                        })
                    else:
                        st.warning(f"⚠️ {test_ticker} NO pasa los filtros")
                        st.info("Probando con filtros más relajados...")
                        
                        try:
                            data = yf.download(test_ticker, period=period, interval=interval, 
                                             progress=False, auto_adjust=True, actions=False)
                            if not data.empty and len(data) >= 50:
                                bb_sma, bb_upper, bb_lower, bb_percent = calculate_bollinger_bands(data, period=20, std_dev=2.5)
                                zscore = calculate_zscore(data, period=20)
                                macd_v, _ = calculate_macd_v(data)
                                
                                current_zscore = float(zscore.iloc[-1])
                                current_bb = float(bb_percent.iloc[-1])
                                current_macdv = float(macd_v.iloc[-1]) if macd_v is not None else 0
                                
                                st.info(f"""
                                **Valores actuales:**
                                - Z-Score: {current_zscore:.2f} (necesita ≥{zscore_threshold} o ≤{-zscore_threshold})
                                - BB %B: {current_bb:.2f} (necesita >{bb_threshold_upper} o <{bb_threshold_lower})
                                - MACD-V: {current_macdv:.1f}
                                
                                **¿Por qué NO pasa?**
                                {'❌ Z-Score insuficiente' if abs(current_zscore) < zscore_threshold else '✅ Z-Score OK'}
                                {'❌ BB %B dentro de bandas' if bb_threshold_lower < current_bb < bb_threshold_upper else '✅ BB %B OK'}
                                """)
                            else:
                                st.error("No se pudieron descargar suficientes datos")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        
        st.markdown("---")
        
        scan_button = st.button("🚀 ESCANEAR", type="primary", use_container_width=True)
    
    if scan_button:
        st.markdown("### 🔄 Escaneando tickers...")
        
        import os
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.dirname(current_dir)
            csv_path = os.path.join(root_dir, "Tickers.csv")
            tickers_df = pd.read_csv(csv_path)
        except Exception as e:
            st.error(f"❌ Error cargando Tickers.csv: {str(e)}")
            st.stop()
        
        tickers_list = tickers_df['Ticker'].tolist()
        st.info(f"📊 Analizando {len(tickers_list)} tickers...")
        
        debug_mode = st.session_state.get('debug_mode', False)
        disable_lock = st.session_state.get('disable_lock', False)
        
        if disable_lock:
            st.warning("⚠️ Lock deshabilitado - puede haber duplicados pero será más rápido")
        
        with st.spinner("Analizando..."):
            results = scan_tickers(
                tickers_list, 
                max_workers=max_workers,
                use_lock=not disable_lock,
                period=period,
                interval=interval,
                zscore_threshold=zscore_threshold,
                bb_threshold_upper=bb_threshold_upper,
                bb_threshold_lower=bb_threshold_lower
            )
        
        if results:
            st.session_state['scan_results'] = results
            st.success(f"✅ {len(results)} tickers detectados con sobreextensión estadística")
            
            if debug_mode:
                with st.expander("🐛 Información de Debug"):
                    st.write(f"**Total tickers analizados:** {len(tickers_list)}")
                    st.write(f"**Tickers detectados:** {len(results)}")
                    st.write(f"**Tasa de detección:** {len(results)/len(tickers_list)*100:.1f}%")
                    
                    ids = [r['ID'] for r in results]
                    unique_ids = len(set(ids))
                    st.write(f"**IDs únicos:** {unique_ids} de {len(ids)}")
                    if unique_ids != len(ids):
                        st.error("⚠️ ¡ADVERTENCIA! Hay IDs duplicados (problema de concurrencia)")
                    else:
                        st.success("✅ Todos los IDs son únicos")
                    
                    ticker_prices = {}
                    for r in results:
                        tick = r['Ticker']
                        price = r['Price']
                        if tick in ticker_prices:
                            if ticker_prices[tick] == price:
                                st.warning(f"⚠️ {tick} aparece con el mismo precio: ${price}")
                        else:
                            ticker_prices[tick] = price
                    
                    st.write("**Muestra de datos (primer resultado):**")
                    if len(results) > 0:
                        first = results[0]
                        st.json({
                            'ID': first['ID'],
                            'Ticker': first['Ticker'],
                            'Company': first.get('Company', 'N/A'),
                            'Price': first['Price'],
                            'Z-Score': first['Z-Score'],
                            'BB_%B': first['BB_%B'],
                            'MACD_V': first['MACD_V'],
                            'Data_Shape': str(first['Data'].shape),
                            'Date': str(first['Date'])
                        })
        else:
            st.warning("⚠️ No se encontraron tickers con las condiciones especificadas")
            
            st.info(f"""
            **💡 Sugerencias para encontrar más resultados:**
            
            1️⃣ **Relajar Z-Score:** Actual = {zscore_threshold}σ → Prueba con 2.0σ o 2.2σ
            
            2️⃣ **Relajar BB %B:** Actual = {bb_threshold:.2f} → Prueba con 0.05 o 0.0
            
            3️⃣ **Cambiar periodo:** Actual = {period} → Prueba con "3mo" o "1y"
            
            4️⃣ **Usar modo Debug:** Activa el checkbox "🐛 Modo Debug" para probar tickers individuales
            
            5️⃣ **Verificar datos:** Es posible que el mercado no tenga valores muy sobreextendidos en este momento
            
            📊 Recuerda: Los filtros detectan sobreextensión **estadísticamente significativa** (2.5σ = eventos raros)
            """)
            
            if st.button("🔄 Intentar con filtros más relajados (Z-Score 2.0σ, BB 0.05)", type="secondary"):
                st.info("Re-ejecutando con filtros más relajados...")
                disable_lock_relaxed = st.session_state.get('disable_lock', False)
                with st.spinner("Analizando..."):
                    results_relaxed = scan_tickers(
                        tickers_list, 
                        max_workers=max_workers,
                        use_lock=not disable_lock_relaxed,
                        period=period,
                        interval=interval,
                        zscore_threshold=2.0,
                        bb_threshold_upper=1.05,
                        bb_threshold_lower=-0.05
                    )
                
                if results_relaxed:
                    st.session_state['scan_results'] = results_relaxed
                    st.success(f"✅ {len(results_relaxed)} tickers encontrados con filtros relajados")
                    st.rerun()
                else:
                    st.error("❌ Incluso con filtros relajados no se encontraron resultados")
    
    if 'scan_results' in st.session_state:
        results = st.session_state['scan_results']
        
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
                'Confirm': '✓' if r['MACDV_Confirm'] else ''
            }
            for r in results
        ])
        
        df_results = df_results.sort_values('Fuerza (σ)', ascending=False).reset_index(drop=True)
        
        if st.session_state.get('debug_mode', False):
            duplicates = df_results[df_results.duplicated(subset=['Precio', 'Z-Score', 'BB %B'], keep=False)]
            if len(duplicates) > 0:
                st.warning(f"⚠️ Se detectaron {len(duplicates)} posibles duplicados en los datos")
                with st.expander("Ver duplicados detectados"):
                    st.dataframe(duplicates)
        
        st.markdown("---")
        st.markdown("### 📊 Resultados del Escaneo")
        
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
                "Confirm": st.column_config.TextColumn("Confirm", width="small"),
            }
        )
        
        csv = df_results.to_csv(index=False)
        st.download_button(
            "📥 Descargar CSV",
            data=csv,
            file_name=f"mean_reversion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        
        if len(event.selection.rows) > 0:
            selected_idx = event.selection.rows[0]
            selected_ticker = df_results.iloc[selected_idx]['Ticker']
            ticker_data = next((r for r in results if r['Ticker'] == selected_ticker), None)
            
            if ticker_data:
                st.markdown(f"### 📈 Análisis Detallado: {selected_ticker}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Precio", f"${ticker_data['Price']:.2f}")
                with col2:
                    st.metric("Z-Score", f"{ticker_data['Z-Score']:.2f}σ")
                with col3:
                    st.metric("BB %B", f"{ticker_data['BB_%B']:.2f}")
                with col4:
                    confirm_icon = "✅" if ticker_data['MACDV_Confirm'] else "⚪"
                    st.metric("MACD-V", f"{ticker_data['MACD_V']:.0f} {confirm_icon}")
                
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
                Presiona "🚀 ESCANEAR" para comenzar
            </p>
            <p style='color: #8E93A1; font-size: 14px;'>
                ✨ Bollinger Bands (2.5σ)<br>
                ✨ Z-Score estadístico<br>
                ✨ MACD-V confirmación<br>
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
