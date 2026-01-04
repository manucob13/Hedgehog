import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import check_password

warnings.filterwarnings('ignore')

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
    # %B > 1 = precio sobre banda superior
    # %B < 0 = precio bajo banda inferior
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
                   zscore_threshold=2.5, bb_threshold_upper=1.1, bb_threshold_lower=-0.1):
    """Analiza un ticker buscando sobreextensión estadística"""
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if data.empty or len(data) < 50:
            return None
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Calcular indicadores
        bb_sma, bb_upper, bb_lower, bb_percent = calculate_bollinger_bands(
            data, period=bb_period, std_dev=2.5
        )
        zscore = calculate_zscore(data, period=zscore_period)
        macd_v, macd_v_signal = calculate_macd_v(data)
        
        if zscore is None or zscore.isna().all():
            return None
        
        # Valores actuales
        current_price = data['Close'].iloc[-1]
        current_zscore = zscore.iloc[-1]
        current_bb_percent = bb_percent.iloc[-1]
        current_macdv = macd_v.iloc[-1] if macd_v is not None else 0
        current_signal = macd_v_signal.iloc[-1] if macd_v_signal is not None else 0
        
        # FILTRO PRINCIPAL: Z-Score >= 2.5 o <= -2.5
        if abs(current_zscore) < zscore_threshold:
            return None
        
        # FILTRO SECUNDARIO: Bollinger %B fuera de bandas
        if not (current_bb_percent >= bb_threshold_upper or current_bb_percent <= bb_threshold_lower):
            return None
        
        # Determinar tipo de señal
        if current_zscore > zscore_threshold and current_bb_percent > bb_threshold_upper:
            signal_type = 'SOBRECOMPRA'
            signal_strength = min(current_zscore, 5.0)  # Cap at 5σ
        elif current_zscore < -zscore_threshold and current_bb_percent < bb_threshold_lower:
            signal_type = 'SOBREVENTA'
            signal_strength = min(abs(current_zscore), 5.0)
        else:
            return None
        
        # Confirmación MACD-V (opcional pero suma puntos)
        macdv_confirms = False
        if signal_type == 'SOBRECOMPRA' and current_macdv > 150:
            macdv_confirms = True
        elif signal_type == 'SOBREVENTA' and current_macdv < -150:
            macdv_confirms = True
        
        return {
            'Ticker': ticker,
            'Price': current_price,
            'Z-Score': current_zscore,
            'BB_%B': current_bb_percent,
            'MACD_V': current_macdv,
            'Signal': current_signal,
            'Type': signal_type,
            'Strength': signal_strength,
            'MACDV_Confirm': macdv_confirms,
            'Date': data.index[-1],
            'Data': data,
            'BB_SMA': bb_sma,
            'BB_Upper': bb_upper,
            'BB_Lower': bb_lower,
            'ZScore_Series': zscore,
            'MACD_V_Series': macd_v,
            'Signal_Series': macd_v_signal
        }
    except Exception as e:
        return None

def scan_tickers(tickers_list, max_workers=10, **kwargs):
    """Escanea múltiples tickers en paralelo"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(tickers_list)
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_ticker, ticker, **kwargs): ticker 
                   for ticker in tickers_list}
        
        for future in as_completed(futures):
            completed += 1
            progress = completed / total
            progress_bar.progress(progress)
            status_text.text(f"Analizando: {completed}/{total} tickers ({progress*100:.1f}%)")
            
            result = future.result()
            if result is not None:
                results.append(result)
    
    progress_bar.empty()
    status_text.empty()
    
    return results

def plot_ticker_analysis(ticker_data):
    """Genera gráfico completo con 4 paneles: Precio+BB, Z-Score, MACD-V, BB %B"""
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 14), facecolor='#0E1117')
    gs = fig.add_gridspec(4, 1, height_ratios=[2.5, 1, 1, 1], hspace=0.3)
    
    data = ticker_data['Data']
    ticker = ticker_data['Ticker']
    
    # ============= PANEL 1: PRECIO + BOLLINGER BANDS =============
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#1A1D29')
    
    # Bandas Bollinger
    ax1.fill_between(data.index, ticker_data['BB_Lower'], ticker_data['BB_Upper'],
                     color='#FFB86C', alpha=0.1, label='Bollinger Bands (2.5σ)', zorder=1)
    ax1.plot(data.index, ticker_data['BB_Upper'], color='#FFB86C', 
            linewidth=2, linestyle='--', alpha=0.7, zorder=2)
    ax1.plot(data.index, ticker_data['BB_SMA'], color='#BD93F9', 
            linewidth=2.5, label='SMA(20)', zorder=2)
    ax1.plot(data.index, ticker_data['BB_Lower'], color='#FFB86C', 
            linewidth=2, linestyle='--', alpha=0.7, zorder=2)
    
    # Precio
    ax1.plot(data.index, data['Close'], color='#FFFFFF', linewidth=2.5, 
            label='Precio', zorder=3)
    
    # Marcar punto actual
    current_color = '#FF6B6B' if ticker_data['Type'] == 'SOBRECOMPRA' else '#4ECDC4'
    ax1.scatter(data.index[-1], data['Close'].iloc[-1], 
               color=current_color, s=200, edgecolors='white', 
               linewidth=3, zorder=10, label='Actual')
    
    ax1.set_title(f'{ticker} - ${ticker_data["Price"]:.2f} | {ticker_data["Type"]} ({ticker_data["Strength"]:.1f}σ)', 
                  fontsize=18, fontweight='bold', color='#FFFFFF', pad=15)
    ax1.set_ylabel('Precio ($)', fontsize=13, fontweight='bold', color='#FFFFFF')
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.1, linestyle='-', linewidth=0.8)
    ax1.tick_params(labelsize=10, colors='#B0B0B0', labelbottom=False)
    
    for spine in ax1.spines.values():
        spine.set_color('#2D3142')
        spine.set_linewidth(1.5)
    
    # ============= PANEL 2: Z-SCORE =============
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_facecolor('#1A1D29')
    
    zscore = ticker_data['ZScore_Series']
    
    # Colorear línea según nivel
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
    
    # Niveles de referencia
    ax2.axhline(y=3, color='#FF6B6B', linestyle='--', linewidth=2, alpha=0.8, label='±3σ')
    ax2.axhline(y=2.5, color='#FFB86C', linestyle='--', linewidth=2, alpha=0.8, label='±2.5σ')
    ax2.axhline(y=0, color='#8E93A1', linestyle='-', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=-2.5, color='#FFB86C', linestyle='--', linewidth=2, alpha=0.8)
    ax2.axhline(y=-3, color='#FF6B6B', linestyle='--', linewidth=2, alpha=0.8)
    
    # Áreas de fondo
    ax2.fill_between(data.index, 2.5, 5, alpha=0.12, color='#FFB86C', zorder=0)
    ax2.fill_between(data.index, -5, -2.5, alpha=0.12, color='#FFB86C', zorder=0)
    ax2.fill_between(data.index, 3, 5, alpha=0.15, color='#FF6B6B', zorder=0)
    ax2.fill_between(data.index, -5, -3, alpha=0.15, color='#FF6B6B', zorder=0)
    
    # Valor actual
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
    
    # ============= PANEL 3: MACD-V =============
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.set_facecolor('#1A1D29')
    
    macd_v = ticker_data['MACD_V_Series']
    signal = ticker_data['Signal_Series']
    
    if macd_v is not None and not macd_v.isna().all():
        # Colorear línea MACD-V
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
        
        # Línea de señal
        if signal is not None:
            ax3.plot(data.index, signal, color='#FFB86C', linewidth=1.5, 
                    alpha=0.5, linestyle='--', zorder=3)
        
        # Niveles de referencia
        ax3.axhline(y=150, color='#FF6B6B', linestyle='--', linewidth=2, alpha=0.8)
        ax3.axhline(y=50, color='#4ECDC4', linestyle=':', linewidth=1.5, alpha=0.7)
        ax3.axhline(y=0, color='#8E93A1', linestyle='-', linewidth=1.5, alpha=0.7)
        ax3.axhline(y=-50, color='#EE5A6F', linestyle=':', linewidth=1.5, alpha=0.7)
        ax3.axhline(y=-150, color='#FF6B6B', linestyle='--', linewidth=2, alpha=0.8)
        
        # Áreas de fondo
        y_max = ax3.get_ylim()[1]
        y_min = ax3.get_ylim()[0]
        ax3.fill_between(data.index, 150, y_max, alpha=0.12, color='#FF6B6B', zorder=0)
        ax3.fill_between(data.index, y_min, -150, alpha=0.12, color='#FF6B6B', zorder=0)
        
        # Valor actual
        current_macdv = ticker_data['MACD_V']
        macd_color = '#FF6B6B' if abs(current_macdv) > 150 else '#4ECDC4' if current_macdv > 50 else '#EE5A6F' if current_macdv < -50 else '#95A5A6'
        confirm_text = " ✓" if ticker_data['MACDV_Confirm'] else ""
        ax3.text(0.02, 0.95, f'MACD-V: {current_macdv:.1f}{confirm_text}', 
                transform=ax3.transAxes, fontsize=13, fontweight='bold', 
                color='white', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.7', facecolor=macd_color, 
                         alpha=0.95, edgecolor='white', linewidth=2))
        
        # Leyenda Signal
        ax3.text(0.98, 0.95, 'Signal', 
                transform=ax3.transAxes, fontsize=10, 
                color='#FFB86C', verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#1A1D29', 
                         alpha=0.8, edgecolor='#FFB86C', linewidth=1))
    
    ax3.set_ylabel('MACD-V', fontsize=12, fontweight='bold', color='#FFFFFF')
    ax3.grid(True, alpha=0.1, linestyle=':', linewidth=1)
    ax3.tick_params(labelsize=10, colors='#B0B0B0', labelbottom=False)
    
    for spine in ax3.spines.values():
        spine.set_color('#2D3142')
        spine.set_linewidth(1.5)
    
    # ============= PANEL 4: BOLLINGER %B =============
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.set_facecolor('#1A1D29')
    
    bb_percent = ticker_data['BB_%B']
    
    # Colorear línea según posición
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
    
    # Niveles de referencia
    ax4.axhline(y=1.0, color='#FFB86C', linestyle='--', linewidth=2, alpha=0.8, label='Banda Superior')
    ax4.axhline(y=0.5, color='#BD93F9', linestyle='-', linewidth=1.5, alpha=0.7, label='Media')
    ax4.axhline(y=0.0, color='#4ECDC4', linestyle='--', linewidth=2, alpha=0.8, label='Banda Inferior')
    
    # Áreas de fondo
    ax4.fill_between(data.index, 1.0, 1.5, alpha=0.12, color='#FFB86C', zorder=0)
    ax4.fill_between(data.index, -0.5, 0.0, alpha=0.12, color='#4ECDC4', zorder=0)
    
    # Valor actual
    current_bb = ticker_data['BB_%B']
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

# ============= INTERFAZ STREAMLIT =============
def main_app():
    """Aplicación principal del screener"""
    st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .ticker-card {
        background: linear-gradient(135deg, #1A1D29 0%, #2D3142 100%);
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid;
        margin: 8px 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .ticker-card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 15px rgba(78, 205, 196, 0.3);
    }
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
    
    # Sidebar
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
        
        zscore_threshold = st.slider("Z-Score Mínimo (σ)", 2.0, 4.0, 2.5, 0.1,
                                    help="Desviaciones estándar del precio vs media")
        
        bb_threshold_upper = st.slider("BB %B Superior", 1.0, 1.5, 1.1, 0.05,
                                      help="Umbral para precio sobre banda superior")
        
        bb_threshold_lower = st.slider("BB %B Inferior", -0.5, 0.0, -0.1, 0.05,
                                      help="Umbral para precio bajo banda inferior")
        
        st.markdown("---")
        
        max_workers = st.slider("Threads paralelos", 5, 20, 10, 1)
        
        st.markdown("---")
        
        scan_button = st.button("🚀 ESCANEAR", type="primary", use_container_width=True)
        
        st.markdown("---")
        
        with st.expander("ℹ️ Metodología"):
            st.markdown(f"""
            **Filtros Principales:**
            
            1️⃣ **Z-Score ≥ {zscore_threshold}σ**
            - Mide desviaciones estándar del precio
            - {zscore_threshold}σ = ~{(1 - 2*(1-0.9938))*100:.1f}% de confianza estadística
            
            2️⃣ **Bollinger %B fuera de bandas**
            - %B > {bb_threshold_upper}: Sobre banda superior
            - %B < {bb_threshold_lower}: Bajo banda inferior
            - Bandas a 2.5 desviaciones estándar
            
            3️⃣ **MACD-V (Confirmación)**
            - > 150 o < -150: Confirma sobreextensión
            - Opcional pero suma fuerza a la señal
            
            **Objetivo:** Encontrar valores con alta probabilidad de reversión a la media.
            """)
    
    # Escaneo
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
        
        with st.spinner("Analizando..."):
            results = scan_tickers(
                tickers_list, 
                max_workers=max_workers,
                period=period,
                interval=interval,
                zscore_threshold=zscore_threshold,
                bb_threshold_upper=bb_threshold_upper,
                bb_threshold_lower=bb_threshold_lower
            )
        
        if results:
            st.session_state['scan_results'] = results
            st.success(f"✅ {len(results)} tickers detectados con sobreextensión estadística")
        else:
            st.warning("⚠️ No se encontraron tickers con las condiciones especificadas")
    
    # Resultados
    if 'scan_results' in st.session_state:
        results = st.session_state['scan_results']
        
        # DataFrame
        df_results = pd.DataFrame([
            {
                'Ticker': r['Ticker'],
                'Tipo': r['Type'],
                'Z-Score': r['Z-Score'],
                'BB_%B': r['BB_%B'],
                'MACD_V': r['MACD_V'],
                'Precio': r['Price'],
                'Fuerza': r['Strength'],
                'Confirm': '✓' if r['MACDV_Confirm'] else ''
            }
            for r in results
        ])
        
        df_results = df_results.sort_values('Fuerza', ascending=False)
        
        st.markdown("---")
        
        # Layout
        col_list, col_chart = st.columns([1, 3])
        
        with col_list:
            st.markdown("### 📋 Tickers Detectados")
            
            # Resumen
            sobrecompra = len([r for r in results if r['Type'] == 'SOBRECOMPRA'])
            sobreventa = len([r for r in results if r['Type'] == 'SOBREVENTA'])
            
            st.markdown(f"""
            <div style='background: #1A1D29; padding: 15px; border-radius: 10px; margin-bottom: 15px;'>
                <div style='display: flex; justify-content: space-between;'>
                    <div style='text-align: center;'>
                        <div style='color: #FF6B6B; font-size: 24px; font-weight: bold;'>{sobrecompra}</div>
                        <div style='color: #B0B0B0; font-size: 12px;'>Sobrecompra</div>
                    </div>
                    <div style='text-align: center;'>
                        <div style='color: #4ECDC4; font-size: 24px; font-weight: bold;'>{sobreventa}</div>
                        <div style='color: #B0B0B0; font-size: 12px;'>Sobreventa</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Lista clickeable
            for idx, row in df_results.iterrows():
                ticker_data = next((r for r in results if r['Ticker'] == row['Ticker']), None)
                if ticker_data:
                    color = '#FF6B6B' if row['Tipo'] == 'SOBRECOMPRA' else '#4ECDC4'
                    
                    if st.button(
                        f"{row['Ticker']} • {row['Z-Score']:.1f}σ",
                        key=f"btn_{row['Ticker']}",
                        use_container_width=True
                    ):
                        st.session_state['selected_ticker'] = ticker_data
                    
                    confirm_badge = "✓" if row['Confirm'] else ""
                    st.markdown(f"""
                    <div class='ticker-card' style='border-left-color: {color};'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <div style='font-size: 16px; font-weight: bold; color: white;'>{row['Ticker']} {confirm_badge}</div>
                                <div style='font-size: 11px; color: #8E93A1;'>${row['Precio']:.2f}</div>
                            </div>
                            <div style='text-align: right;'>
                                <div style='font-size: 18px; font-weight: bold; color: {color};'>{row['Z-Score']:.1f}σ</div>
                                <div style='font-size: 10px; color: #B0B0B0;'>%B: {row['BB_%B']:.2f}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            csv = df_results.to_csv(index=False)
            st.download_button(
                "📥 Descargar CSV",
                data=csv,
                file_name=f"mean_reversion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_chart:
            if 'selected_ticker' in st.session_state:
                ticker_data = st.session_state['selected_ticker']
                
                st.markdown(f"### 📈 {ticker_data['Ticker']}")
                
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
                
                fig = plot_ticker_analysis(ticker_data)
                st.pyplot(fig)
            else:
                st.markdown("""
                <div style='text-align: center; padding: 80px 20px;
                            background: linear-gradient(135deg, #1A1D29 0%, #2D3142 100%);
                            border-radius: 15px; border: 2px dashed #4ECDC4;'>
                    <h2 style='color: #4ECDC4; margin: 0;'>👈 Selecciona un ticker</h2>
                    <p style='color: #8E93A1; margin-top: 15px;'>
                        Click en la lista para ver análisis completo
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
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
