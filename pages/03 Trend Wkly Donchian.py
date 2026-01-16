# pages/Market_Regime_Weekly.py
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
from utils.utils import check_password

warnings.filterwarnings('ignore')

# Configuración de página
st.set_page_config(
    page_title="Trend Analyzer",
    page_icon="📈",
    layout="wide"
)

# Colores para regímenes semanales (con dos tipos de riesgo)
REGIME_COLORS_WEEKLY = {
    'RIESGO_DONCHIAN': '#FF6B6B',  # Rojo intenso
    'RIESGO_MACDV': '#FF9F40',      # Naranja
    'BAJISTA': '#EE5A6F',
    'RANGO': '#FFD93D',
    'ALCISTA': '#4ECDC4'
}

def calculate_atr(df, period=26):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def calculate_sma(data, period):
    return data.rolling(window=period).mean()

def calculate_donchian(df, period=20):
    upper = df['High'].rolling(window=period).max()
    lower = df['Low'].rolling(window=period).min()
    middle = (upper + lower) / 2
    return upper, middle, lower

def calculate_choppiness(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_sum = true_range.rolling(window=period).sum()
    high_max = df['High'].rolling(window=period).max()
    low_min = df['Low'].rolling(window=period).min()
    chop = 100 * np.log10(atr_sum / (high_max - low_min)) / np.log10(period)
    return chop

def calculate_macd_v(df, fast_len=12, slow_len=26, signal_len=9, atr_len=26):
    fast_ema = calculate_ema(df['Close'], fast_len)
    slow_ema = calculate_ema(df['Close'], slow_len)
    atr = calculate_atr(df, atr_len)
    macd = ((fast_ema - slow_ema) / atr) * 100
    signal = calculate_ema(macd, signal_len)
    return macd, signal

@st.cache_data(ttl=timedelta(hours=1))
def download_weekly_data(ticker, start_date='2018-01-01'):
    try:
        # Añadir parámetros adicionales para mejorar la descarga
        data = yf.download(
            ticker, 
            start=start_date, 
            interval='1wk', 
            progress=False,
            auto_adjust=True,
            actions=False,
            timeout=10
        )
        
        # Validación exhaustiva
        if data is None or data.empty:
            st.error(f"⚠️ No se encontraron datos para el ticker '{ticker}'. Verifica que el símbolo sea correcto.")
            st.info("💡 Intenta con: AAPL, MSFT, GOOGL, TSLA, SPY, QQQ, etc.")
            return None
        
        # Verificar que tengamos suficientes datos
        if len(data) < 50:
            st.warning(f"⚠️ Datos insuficientes para {ticker} ({len(data)} semanas). Se requieren al menos 50 períodos.")
            return None
        
        # Manejar MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Construir DataFrame con validación
        required_columns = ['Close', 'Open', 'High', 'Low', 'Volume']
        missing_cols = [col for col in required_columns if col not in data.columns]
        
        if missing_cols:
            st.error(f"❌ Faltan columnas requeridas: {missing_cols}")
            return None
        
        df = pd.DataFrame({
            'Close': data['Close'].squeeze(),
            'Open': data['Open'].squeeze(),
            'High': data['High'].squeeze(),
            'Low': data['Low'].squeeze(),
            'Volume': data['Volume'].squeeze()
        }, index=data.index)
        
        # Eliminar filas con NaN en columnas críticas
        df = df.dropna(subset=['Close', 'High', 'Low'])
        
        if len(df) < 50:
            st.warning(f"⚠️ Después de limpiar datos, quedan {len(df)} períodos (mínimo: 50)")
            return None
        
        # Calcular indicadores
        df['SMA_20'] = calculate_sma(df['Close'], 20)
        df['SMA_50'] = calculate_sma(df['Close'], 50)
        df['Donchian_Upper'], df['Donchian_Middle'], df['Donchian_Lower'] = calculate_donchian(df, period=20)
        df['Choppiness'] = calculate_choppiness(df, period=14)
        df['MACD_V'], df['MACD_V_Signal'] = calculate_macd_v(df, fast_len=12, slow_len=26, signal_len=9, atr_len=26)
        
        # Eliminar NaN resultantes de cálculos
        df = df.dropna()
        
        if len(df) < 30:
            st.warning(f"⚠️ Después de calcular indicadores, quedan {len(df)} períodos (mínimo recomendado: 30)")
            return None
        
        return df
        
    except Exception as e:
        error_msg = str(e)
        st.error(f"❌ Error descargando datos para '{ticker}'")
        
        # Mensajes de error más específicos
        if "404" in error_msg or "Not Found" in error_msg:
            st.error(f"🔍 Ticker '{ticker}' no encontrado. Verifica el símbolo.")
            st.info("💡 Ejemplos válidos: AAPL, MSFT, GOOGL, AMZN, TSLA, SPY, QQQ")
        elif "timeout" in error_msg.lower():
            st.error("⏱️ Timeout de conexión. Intenta nuevamente.")
        elif "No data found" in error_msg:
            st.error(f"📊 No hay datos disponibles para '{ticker}' en el período seleccionado.")
        else:
            st.error(f"⚠️ Error técnico: {error_msg}")
        
        return None

def classify_regime_weekly(df):
    """
    Clasificación SEMANAL mejorada con DOS tipos de RIESGO
    
    Jerarquía clara:
    1. RIESGO_DONCHIAN: Precio fuera del canal Donchian (breakout extremo)
    2. RIESGO_MACDV: MACD-V > 150 o < -150 (momentum extremo)
    3. RANGO: Choppiness > 61.8 Y momentum débil (-50 < MACD-V < 50)
    4. ALCISTA: MACD-V > 50 Y Precio >= Donchian_Middle
    5. BAJISTA: MACD-V < -50 Y Precio <= Donchian_Middle
    6. RANGO (fallback): Todo lo demás
    """
    regimes = []
    
    for idx, row in df.iterrows():
        macd_v = row['MACD_V']
        chop = row['Choppiness']
        price = row['Close']
        donchian_upper = row['Donchian_Upper']
        donchian_lower = row['Donchian_Lower']
        donchian_middle = row['Donchian_Middle']
        
        # PRIORIDAD 1: RIESGO_DONCHIAN - Precio fuera del canal
        # Esto indica breakout/breakdown extremo
        if price > donchian_upper or price < donchian_lower:
            regime = 'RIESGO_DONCHIAN'
        
        # PRIORIDAD 2: RIESGO_MACDV - Momentum extremo
        # Sobreextensión en el indicador de momentum
        elif macd_v > 150 or macd_v < -150:
            regime = 'RIESGO_MACDV'
        
        # PRIORIDAD 3: RANGO - Mercado choppy CON momentum débil
        elif chop > 61.8 and -50 <= macd_v <= 50:
            regime = 'RANGO'
        
        # PRIORIDAD 4: ALCISTA - Momentum alcista + precio en zona superior
        elif macd_v > 50 and price >= donchian_middle:
            regime = 'ALCISTA'
        
        # PRIORIDAD 5: BAJISTA - Momentum bajista + precio en zona inferior
        elif macd_v < -50 and price <= donchian_middle:
            regime = 'BAJISTA'
        
        # PRIORIDAD 6: Casos intermedios
        elif macd_v > 50:
            regime = 'ALCISTA'
        elif macd_v < -50:
            regime = 'BAJISTA'
        
        # PRIORIDAD 7: RANGO por defecto
        else:
            regime = 'RANGO'
        
        regimes.append(regime)
    
    return regimes

def plot_weekly_dashboard(df_recent, ticker):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(24, 14), facecolor='#0E1117')
    gs = fig.add_gridspec(4, 1, height_ratios=[3.5, 1, 1, 1], hspace=0.4)
    
    # PANEL 1: Precio + Donchian + Regímenes
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#1A1D29')
    ax1.fill_between(df_recent.index, df_recent['Donchian_Upper'], df_recent['Donchian_Lower'], 
                     alpha=0.08, color='#00D9FF', zorder=0)
    ax1.plot(df_recent.index, df_recent['Donchian_Upper'], color='#00D9FF', alpha=0.6, 
            linewidth=2, linestyle='--', label='Donchian Upper', zorder=2)
    ax1.plot(df_recent.index, df_recent['Donchian_Middle'], color='#00D9FF', alpha=0.8, 
            linewidth=2.5, linestyle='-', label='Donchian Middle', zorder=2)
    ax1.plot(df_recent.index, df_recent['Donchian_Lower'], color='#00D9FF', alpha=0.6, 
            linewidth=2, linestyle='--', label='Donchian Lower', zorder=2)
    
    ax1.plot(df_recent.index, df_recent['Close'], color='#FFFFFF', alpha=0.15, linewidth=6, zorder=1)
    ax1.plot(df_recent.index, df_recent['Close'], color='#E0E0E0', alpha=0.4, linewidth=3, zorder=2)
    ax1.plot(df_recent.index, df_recent['Close'], color='#FFFFFF', linewidth=1.5, zorder=3)
    
    for regime_name, color in REGIME_COLORS_WEEKLY.items():
        mask = df_recent['Regime_Weekly'] == regime_name
        if mask.sum() > 0:
            ax1.scatter(df_recent[mask].index, df_recent[mask]['Close'], c=color, alpha=0.15, 
                       s=220, edgecolors='none', zorder=4)
            
            # Etiquetas más descriptivas
            label_map = {
                'RIESGO_DONCHIAN': 'RIESGO (Donchian)',
                'RIESGO_MACDV': 'RIESGO (MACD-V)',
                'ALCISTA': 'ALCISTA',
                'BAJISTA': 'BAJISTA',
                'RANGO': 'RANGO'
            }
            ax1.scatter(df_recent[mask].index, df_recent[mask]['Close'], c=color, 
                       label=label_map[regime_name], alpha=0.95, s=85, edgecolors='white', 
                       linewidth=1.5, zorder=5)
    
    ax1.plot(df_recent.index, df_recent['SMA_20'], color='#50FA7B', alpha=0.9, linewidth=2.5, 
            label='SMA(20)', zorder=3)
    ax1.plot(df_recent.index, df_recent['SMA_50'], color='#BD93F9', alpha=0.9, linewidth=2.5, 
            label='SMA(50)', zorder=3)
    
    current = df_recent.iloc[-1]
    ax1.scatter(current.name, current['Close'], facecolors='none', 
               edgecolors=REGIME_COLORS_WEEKLY[current['Regime_Weekly']], s=450, 
               linewidth=5, alpha=0.4, marker='o', zorder=9)
    ax1.scatter(current.name, current['Close'], facecolors=REGIME_COLORS_WEEKLY[current['Regime_Weekly']], 
               edgecolors='white', s=280, linewidth=4, marker='o', 
               label=f'Actual: {current["Regime_Weekly"]}', zorder=10)
    
    bbox_color = REGIME_COLORS_WEEKLY[current['Regime_Weekly']]
    regime_display = current['Regime_Weekly'].replace('RIESGO_DONCHIAN', 'RIESGO (D)').replace('RIESGO_MACDV', 'RIESGO (M)')
    ax1.annotate(f'{regime_display}\n${current["Close"]:.2f}', 
                xy=(current.name, current['Close']), xytext=(25, 40), 
                textcoords='offset points', fontsize=14, fontweight='bold', color='white', 
                bbox=dict(boxstyle='round,pad=1', facecolor=bbox_color, alpha=0.95, 
                         edgecolor='white', linewidth=3), 
                arrowprops=dict(arrowstyle='->', lw=3, color=bbox_color, 
                               connectionstyle='arc3,rad=0.3'), zorder=11)
    
    ax1.text(0.5, 1.10, f'{ticker}', transform=ax1.transAxes, fontsize=28, 
            fontweight='bold', ha='center', color='#FFFFFF')
    ax1.text(0.5, 1.05, 'Weekly Market Regime Analysis - Dual Risk Detection', transform=ax1.transAxes, 
            fontsize=13, style='italic', ha='center', color='#8E93A1')
    ax1.set_ylabel('Price ($)', fontsize=15, fontweight='700', color='#FFFFFF', labelpad=12)
    legend = ax1.legend(loc='upper left', fontsize=9, framealpha=0.95, ncol=4, 
                       edgecolor='#00D9FF', fancybox=True, borderpad=1.2, 
                       labelspacing=1, columnspacing=1.5)
    legend.get_frame().set_facecolor('#1A1D29')
    legend.get_frame().set_linewidth(2)
    ax1.grid(True, alpha=0.08, linestyle='-', linewidth=1, color='#FFFFFF')
    ax1.tick_params(labelsize=11, colors='#B0B0B0', width=1.5)
    for spine in ax1.spines.values():
        spine.set_color('#2D3142')
        spine.set_linewidth(2)
    
    # PANEL 2: Choppiness Index
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_facecolor('#1A1D29')
    ax2.plot(df_recent.index, df_recent['Choppiness'], color='#FFD93D', linewidth=2.5, 
            label='Choppiness Index', zorder=3)
    ax2.fill_between(df_recent.index, df_recent['Choppiness'], 50, 
                     where=(df_recent['Choppiness'] >= 61.8), color='#FFD93D', alpha=0.2, zorder=1)
    ax2.fill_between(df_recent.index, df_recent['Choppiness'], 50, 
                     where=(df_recent['Choppiness'] <= 38.2), color='#4ECDC4', alpha=0.2, zorder=1)
    ax2.axhline(y=61.8, color='#FFD93D', linestyle='--', linewidth=2, alpha=0.8, 
               label='Choppy > 61.8', zorder=2)
    ax2.axhline(y=50, color='#8E93A1', linestyle='-', linewidth=1.5, alpha=0.7, zorder=2)
    ax2.axhline(y=38.2, color='#4ECDC4', linestyle='--', linewidth=2, alpha=0.8, 
               label='Trending < 38.2', zorder=2)
    ax2.fill_between(df_recent.index, 61.8, 100, alpha=0.12, color='#FFD93D', zorder=0)
    ax2.fill_between(df_recent.index, 0, 38.2, alpha=0.12, color='#4ECDC4', zorder=0)
    ax2.scatter(current.name, current['Choppiness'], facecolors='#FFD93D', edgecolors='white', 
               s=220, linewidth=4, marker='o', zorder=10)
    
    chop_color = '#FFD93D' if current['Choppiness'] > 61.8 else '#4ECDC4' if current['Choppiness'] < 38.2 else '#8E93A1'
    chop_status = 'Choppy' if current['Choppiness'] > 61.8 else 'Trending' if current['Choppiness'] < 38.2 else 'Neutral'
    ax2.text(0.02, 0.88, f'Chop: {current["Choppiness"]:.1f} ({chop_status})', 
            transform=ax2.transAxes, fontsize=13, fontweight='bold', color='white', 
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.7', facecolor=chop_color, 
                                              alpha=0.95, edgecolor='white', linewidth=2.5))
    ax2.set_ylabel('Choppiness', fontsize=14, fontweight='700', color='#FFFFFF', labelpad=12)
    ax2.set_ylim([0, 100])
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax2.grid(True, alpha=0.08, linestyle='-', linewidth=1, color='#FFFFFF')
    ax2.tick_params(labelsize=10.5, colors='#B0B0B0', width=1.5)
    for spine in ax2.spines.values():
        spine.set_color('#2D3142')
        spine.set_linewidth(2)
    
    # PANEL 3: MACD-V (Escala adaptativa)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.set_facecolor('#1A1D29')
    
    # Calcular límites inteligentes
    macd_min = df_recent['MACD_V'].min()
    macd_max = df_recent['MACD_V'].max()
    y_range = macd_max - macd_min
    y_min = macd_min - (y_range * 0.2)
    y_max = macd_max + (y_range * 0.2)
    
    if macd_max > 100:
        y_max = max(y_max, 180)
    if macd_min < -100:
        y_min = min(y_min, -180)
    
    # Plot MACD-V con colores por segmento
    for i in range(1, len(df_recent)):
        if pd.notna(df_recent['MACD_V'].iloc[i-1]) and pd.notna(df_recent['MACD_V'].iloc[i]):
            x1, x2 = df_recent.index[i-1], df_recent.index[i]
            y1, y2 = df_recent['MACD_V'].iloc[i-1], df_recent['MACD_V'].iloc[i]
            
            if y2 > 150 or y2 < -150:
                color, width = '#FF9F40', 3.5  # Naranja para RIESGO_MACDV
            elif y2 > 50:
                color, width = '#4ECDC4', 3
            elif y2 > -50:
                color, width = '#95A5A6', 2.5
            else:
                color, width = '#EE5A6F', 3
            
            ax3.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=0.95, zorder=5)
    
    # Líneas de referencia
    if y_max > 150:
        ax3.axhline(y=150, color='#FF9F40', linestyle='--', linewidth=2, alpha=0.8, 
                   label='Riesgo MACD-V > 150', zorder=2)
        ax3.fill_between(df_recent.index, 150, y_max, alpha=0.12, color='#FF9F40', zorder=0)
    
    if y_max > 50:
        ax3.axhline(y=50, color='#4ECDC4', linestyle=':', linewidth=1.5, alpha=0.7, 
                   label='Alcista > 50', zorder=2)
    
    ax3.axhline(y=0, color='#8E93A1', linestyle='-', linewidth=1.5, alpha=0.7, zorder=2)
    
    if y_min < -50:
        ax3.axhline(y=-50, color='#EE5A6F', linestyle=':', linewidth=1.5, alpha=0.7, 
                   label='Bajista < -50', zorder=2)
    
    if y_min < -150:
        ax3.axhline(y=-150, color='#FF9F40', linestyle='--', linewidth=2, alpha=0.8, 
                   label='Riesgo MACD-V < -150', zorder=2)
        ax3.fill_between(df_recent.index, y_min, -150, alpha=0.12, color='#FF9F40', zorder=0)
    
    neutral_top = min(50, y_max)
    neutral_bottom = max(-50, y_min)
    ax3.fill_between(df_recent.index, neutral_bottom, neutral_top, alpha=0.08, 
                     color='#95A5A6', zorder=0)
    
    if 'MACD_V_Signal' in df_recent.columns:
        ax3.plot(df_recent.index, df_recent['MACD_V_Signal'], color='#FFB86C', 
                linewidth=1.2, alpha=0.4, linestyle='--', label='Signal', zorder=3)
    
    ax3.scatter(current.name, current['MACD_V'], facecolors='white', edgecolors='#00D9FF', 
               s=220, linewidth=4, marker='o', zorder=10)
    
    macd_color = '#FF9F40' if abs(current['MACD_V']) > 150 else '#4ECDC4' if current['MACD_V'] > 50 else '#EE5A6F' if current['MACD_V'] < -50 else '#95A5A6'
    ax3.text(0.02, 0.88, f'MACD-V: {current["MACD_V"]:.1f}', transform=ax3.transAxes, 
            fontsize=13, fontweight='bold', color='white', verticalalignment='top', 
            bbox=dict(boxstyle='round,pad=0.7', facecolor=macd_color, alpha=0.95, 
                     edgecolor='white', linewidth=2.5))
    
    ax3.set_ylabel('MACD-V', fontsize=14, fontweight='700', color='#FFFFFF', labelpad=12)
    ax3.set_ylim([y_min, y_max])
    ax3.legend(loc='upper right', fontsize=9, framealpha=0.95, ncol=2)
    ax3.grid(True, alpha=0.08, linestyle=':', linewidth=1, color='#FFFFFF')
    ax3.tick_params(labelsize=10.5, colors='#B0B0B0', width=1.5)
    for spine in ax3.spines.values():
        spine.set_color('#2D3142')
        spine.set_linewidth(2)
    
    # PANEL 4: Timeline de Regímenes
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.set_facecolor('#1A1D29')
    regime_order = ['BAJISTA', 'RANGO', 'ALCISTA', 'RIESGO_MACDV', 'RIESGO_DONCHIAN']
    regime_labels = ['BAJISTA', 'RANGO', 'ALCISTA', 'RIESGO (M)', 'RIESGO (D)']
    
    for regime_name in regime_order:
        mask = df_recent['Regime_Weekly'] == regime_name
        if mask.sum() > 0:
            color = REGIME_COLORS_WEEKLY[regime_name]
            regime_num = regime_order.index(regime_name)
            ax4.scatter(df_recent[mask].index, [regime_num] * mask.sum(), c=color, 
                       alpha=0.95, s=110, edgecolors='white', linewidth=1.8, zorder=4)
    
    ax4.set_ylabel('Regime', fontsize=14, fontweight='700', color='#FFFFFF', labelpad=12)
    ax4.set_yticks(range(5))
    ax4.set_yticklabels(regime_labels, fontsize=11, fontweight='700', color='#E0E0E0')
    ax4.set_xlabel('Date', fontsize=15, fontweight='700', color='#FFFFFF', labelpad=12)
    ax4.grid(True, alpha=0.08, linestyle='-', linewidth=1, color='#FFFFFF', axis='x')
    ax4.tick_params(labelsize=11, colors='#B0B0B0', width=1.5)
    for spine in ax4.spines.values():
        spine.set_color('#2D3142')
        spine.set_linewidth(2)
    
    plt.tight_layout()
    return fig

def market_regime_page():
    st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stMetric { 
        background: linear-gradient(135deg, #1A1D29 0%, #2D3142 100%); 
        padding: 20px; 
        border-radius: 15px; 
        border: 2px solid #00D9FF; 
        box-shadow: 0 4px 15px rgba(0, 217, 255, 0.2); 
    }
    .stMetric label { 
        color: #00D9FF !important; 
        font-weight: 700 !important; 
        font-size: 14px !important; 
    }
    .stMetric [data-testid="stMetricValue"] { 
        color: #FFFFFF !important; 
        font-size: 24px !important; 
        font-weight: 800 !important; 
    }
    .stButton>button { 
        background: linear-gradient(135deg, #4ECDC4 0%, #00D9FF 100%); 
        color: white; 
        font-weight: 700; 
        border: none; 
        padding: 12px 24px; 
        border-radius: 10px; 
        box-shadow: 0 4px 15px rgba(78, 205, 196, 0.4); 
        transition: all 0.3s ease; 
    }
    .stButton>button:hover { 
        transform: translateY(-2px); 
        box-shadow: 0 6px 20px rgba(78, 205, 196, 0.6); 
    }
    h1, h2, h3 { 
        color: #FFFFFF !important; 
        font-weight: 800 !important; 
    }
    .stAlert { 
        border-radius: 12px; 
        border-left: 5px solid #00D9FF; 
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("📈 Trend Analyzer - Dual Risk Detection")
    st.markdown("---")
    
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #4ECDC4 0%, #00D9FF 100%); 
                    border-radius: 15px; margin-bottom: 20px;'>
            <h2 style='color: white; margin: 0;'>⚙️ Settings</h2>
        </div>
        """, unsafe_allow_html=True)
        
        ticker = st.text_input("Ticker Symbol", value="AAPL", help="Ingresa el símbolo del ticker").upper()
        
        # Validación del ticker
        if ticker:
            if len(ticker) > 10:
                st.warning("⚠️ El ticker parece demasiado largo")
            elif not ticker.replace('.', '').replace('-', '').isalnum():
                st.warning("⚠️ El ticker contiene caracteres inválidos")
        
        # Sugerencias de tickers populares
        st.markdown("**Tickers populares:**")
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            if st.button("📱 AAPL", key="aapl", use_container_width=True):
                st.session_state['ticker_input'] = "AAPL"
                st.rerun()
            if st.button("🚗 TSLA", key="tsla", use_container_width=True):
                st.session_state['ticker_input'] = "TSLA"
                st.rerun()
        with col_t2:
            if st.button("💻 MSFT", key="msft", use_container_width=True):
                st.session_state['ticker_input'] = "MSFT"
                st.rerun()
            if st.button("📊 SPY", key="spy", use_container_width=True):
                st.session_state['ticker_input'] = "SPY"
                st.rerun()
        with col_t3:
            if st.button("🔍 GOOGL", key="googl", use_container_width=True):
                st.session_state['ticker_input'] = "GOOGL"
                st.rerun()
            if st.button("💰 QQQ", key="qqq", use_container_width=True):
                st.session_state['ticker_input'] = "QQQ"
                st.rerun()
        
        # Actualizar ticker si se seleccionó uno
        if 'ticker_input' in st.session_state:
            ticker = st.session_state['ticker_input']
        
        st.markdown("---")
        
        lookback_months = st.slider(
            "📅 Meses a Visualizar", 
            min_value=1, 
            max_value=24, 
            value=3,
            step=1,
            help="Selecciona cuántos meses de datos históricos visualizar"
        )
        lookback_weeks = int(lookback_months * 4.33)
        
        st.markdown("---")
        
        start_date = st.date_input("📆 Fecha Inicio de Datos", value=datetime(2018, 1, 1))
        
        st.markdown("---")
        
        analizar_btn = st.button("🚀 ANALIZAR MERCADO", type="primary", use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1A1D29 0%, #2D3142 100%); 
                    padding: 15px; border-radius: 12px; border: 2px solid #BD93F9;'>
            <h3 style='color: #BD93F9; margin-top: 0;'>📖 Metodología</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 🔍 ANÁLISIS SEMANAL
        
        **Sistema de Doble Riesgo:**
        
        🔴 **RIESGO DONCHIAN**
        - Precio > Canal Superior
        - Precio < Canal Inferior
        - *Breakout extremo del rango*
        
        🟠 **RIESGO MACD-V**
        - MACD-V > 150 o < -150
        - *Momentum sobreextendido*
        
        ---
        
        **Otros Regímenes:**
        
        🟢 **ALCISTA**
        - MACD-V > 50
        - Precio ≥ Donchian Middle
        
        🔴 **BAJISTA**
        - MACD-V < -50
        - Precio ≤ Donchian Middle
        
        🟡 **RANGO**
        - Choppiness > 61.8
        - -50 < MACD-V < 50
        
        ---
        
        **Indicadores:**
        - **MACD-V**: Momentum normalizado
        - **Choppiness**: 61.8/38.2 thresholds
        - **Donchian**: Canal 20 períodos
        - **SMAs**: 20 y 50 períodos
        """)
        
        st.markdown("---")
        
        st.markdown("""
        <div style='text-align: center; padding: 10px; background: #1A1D29; 
                    border-radius: 10px; border: 1px solid #FF6B6B;'>
            <p style='color: #FF6B6B; font-size: 11px; margin: 0; font-weight: bold;'>
                ⚠️ RIESGO DONCHIAN = Precio fuera del canal
            </p>
            <p style='color: #FF9F40; font-size: 11px; margin: 5px 0 0 0; font-weight: bold;'>
                ⚠️ RIESGO MACD-V = Momentum extremo
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    if analizar_btn:
        with st.spinner(f"⏳ Descargando datos para {ticker}..."):
            # Validación previa
            if not ticker or ticker.strip() == "":
                st.error("❌ Por favor ingresa un ticker válido")
                st.stop()
            
            if len(ticker) < 1 or len(ticker) > 10:
                st.error("❌ El ticker debe tener entre 1 y 10 caracteres")
                st.stop()
            
            # Intentar descarga
            df_weekly = download_weekly_data(ticker, start_date.strftime('%Y-%m-%d'))
            
            if df_weekly is None:
                st.error(f"❌ No se pudieron obtener datos para {ticker}")
                
                # Sugerencias adicionales
                st.markdown("""
                ---
                ### 🔍 Sugerencias para resolver el problema:
                
                1. **Verifica el símbolo del ticker**
                   - ¿Es el símbolo correcto? (Ej: AAPL para Apple)
                   - ¿Está en el mercado correcto? (US, etc.)
                
                2. **Prueba con tickers conocidos:**
                   - **Acciones**: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META
                   - **ETFs**: SPY, QQQ, DIA, IWM, VTI
                   - **Índices**: ^GSPC, ^DJI, ^IXIC
                
                3. **Verifica la fecha de inicio**
                   - Algunos tickers nuevos no tienen datos desde 2018
                   - Intenta con una fecha más reciente
                
                4. **Problemas comunes:**
                   - Ticker delisted (dejó de cotizar)
                   - Símbolo incorrecto o cambió de nombre
                   - Problemas temporales con Yahoo Finance
                """)
                st.stop()
            
            # Clasificar regímenes
            df_weekly['Regime_Weekly'] = classify_regime_weekly(df_weekly)
            
            # Guardar en session state
            st.session_state['ticker'] = ticker
            st.session_state['df_weekly'] = df_weekly
            st.session_state['lookback_months'] = lookback_months
            
            # Mostrar información de éxito
            st.success(f"✅ Datos cargados exitosamente para {ticker}")
            st.info(f"📊 {len(df_weekly)} semanas de datos | Desde {df_weekly.index[0].strftime('%Y-%m-%d')} hasta {df_weekly.index[-1].strftime('%Y-%m-%d')}")
    
    if 'df_weekly' in st.session_state:
        df_weekly = st.session_state['df_weekly']
        ticker = st.session_state['ticker']
        
        df_weekly_recent = df_weekly.tail(lookback_weeks)
        current_weekly = df_weekly.iloc[-1]
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; margin-bottom: 20px;'>
            <h2 style='color: #4ECDC4; font-size: 32px; margin: 0;'>📈 ANÁLISIS SEMANAL</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Métricas principales
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            regime_emoji = {
                'ALCISTA': '🟢',
                'BAJISTA': '🔴',
                'RANGO': '🟡',
                'RIESGO_DONCHIAN': '🔴',
                'RIESGO_MACDV': '🟠'
            }[current_weekly['Regime_Weekly']]
            
            regime_display = current_weekly['Regime_Weekly'].replace('RIESGO_DONCHIAN', 'RIESGO (D)').replace('RIESGO_MACDV', 'RIESGO (M)')
            st.metric("RÉGIMEN", f"{regime_emoji} {regime_display}")
        
        with col2:
            price_change_pct = ((current_weekly['Close'] - df_weekly['Close'].iloc[-5]) / 
                               df_weekly['Close'].iloc[-5] * 100)
            st.metric("PRECIO", f"${current_weekly['Close']:.2f}", f"{price_change_pct:+.2f}%")
        
        with col3:
            chop_status = ("Choppy" if current_weekly['Choppiness'] > 61.8 else 
                          "Trending" if current_weekly['Choppiness'] < 38.2 else "Neutral")
            st.metric("CHOPPINESS", f"{current_weekly['Choppiness']:.1f}", chop_status)
        
        with col4:
            macd_status = ("Extremo" if abs(current_weekly['MACD_V']) > 150 else 
                          "Alcista" if current_weekly['MACD_V'] > 50 else 
                          "Bajista" if current_weekly['MACD_V'] < -50 else "Neutral")
            st.metric("MACD-V", f"{current_weekly['MACD_V']:.1f}", macd_status)
        
        with col5:
            donch_pos = ((current_weekly['Close'] - current_weekly['Donchian_Lower']) / 
                        (current_weekly['Donchian_Upper'] - current_weekly['Donchian_Lower']) * 100)
            
            # Detectar si está fuera del canal
            if current_weekly['Close'] > current_weekly['Donchian_Upper']:
                donch_status = "⚠️ ARRIBA"
            elif current_weekly['Close'] < current_weekly['Donchian_Lower']:
                donch_status = "⚠️ ABAJO"
            else:
                donch_status = f"{donch_pos:.0f}%"
            
            st.metric("DONCHIAN", donch_status)
        
        with col6:
            st.metric("FECHA", current_weekly.name.strftime('%Y-%m-%d'), 
                     current_weekly.name.strftime('%b'))
        
        st.markdown("---")
        
        # Estadísticas de régimen
        regime_counts = df_weekly_recent['Regime_Weekly'].value_counts()
        regime_pcts = (regime_counts / len(df_weekly_recent) * 100).round(1)
        
        st.markdown("""
        <div style='text-align: center; margin: 20px 0;'>
            <h3 style='color: #BD93F9;'>📊 Distribución de Regímenes (Período Seleccionado)</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            alcista_count = regime_counts.get('ALCISTA', 0)
            alcista_pct = regime_pcts.get('ALCISTA', 0)
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #4ECDC4 0%, #3DBCB4 100%); 
                        padding: 20px; border-radius: 12px; text-align: center;'>
                <h2 style='color: white; margin: 0; font-size: 36px;'>{alcista_count}</h2>
                <p style='color: white; margin: 5px 0 0 0; font-size: 13px;'>
                    🟢 ALCISTA<br>({alcista_pct}%)
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            bajista_count = regime_counts.get('BAJISTA', 0)
            bajista_pct = regime_pcts.get('BAJISTA', 0)
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #EE5A6F 0%, #DC4A5F 100%); 
                        padding: 20px; border-radius: 12px; text-align: center;'>
                <h2 style='color: white; margin: 0; font-size: 36px;'>{bajista_count}</h2>
                <p style='color: white; margin: 5px 0 0 0; font-size: 13px;'>
                    🔴 BAJISTA<br>({bajista_pct}%)
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            rango_count = regime_counts.get('RANGO', 0)
            rango_pct = regime_pcts.get('RANGO', 0)
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #FFD93D 0%, #F0C929 100%); 
                        padding: 20px; border-radius: 12px; text-align: center;'>
                <h2 style='color: white; margin: 0; font-size: 36px;'>{rango_count}</h2>
                <p style='color: white; margin: 5px 0 0 0; font-size: 13px;'>
                    🟡 RANGO<br>({rango_pct}%)
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            riesgo_d_count = regime_counts.get('RIESGO_DONCHIAN', 0)
            riesgo_d_pct = regime_pcts.get('RIESGO_DONCHIAN', 0)
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #FF6B6B 0%, #EE5555 100%); 
                        padding: 20px; border-radius: 12px; text-align: center;'>
                <h2 style='color: white; margin: 0; font-size: 36px;'>{riesgo_d_count}</h2>
                <p style='color: white; margin: 5px 0 0 0; font-size: 13px;'>
                    🔴 RIESGO (D)<br>({riesgo_d_pct}%)
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            riesgo_m_count = regime_counts.get('RIESGO_MACDV', 0)
            riesgo_m_pct = regime_pcts.get('RIESGO_MACDV', 0)
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #FF9F40 0%, #F08B28 100%); 
                        padding: 20px; border-radius: 12px; text-align: center;'>
                <h2 style='color: white; margin: 0; font-size: 36px;'>{riesgo_m_count}</h2>
                <p style='color: white; margin: 5px 0 0 0; font-size: 13px;'>
                    🟠 RIESGO (M)<br>({riesgo_m_pct}%)
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Dashboard principal
        st.markdown(f"""
        <div style='text-align: center; margin-bottom: 20px;'>
            <h2 style='color: #BD93F9; font-size: 28px;'>📉 Dashboard Técnico Semanal</h2>
            <p style='color: #8E93A1; font-size: 16px;'>
                Últimos {lookback_months} meses - Timeframe: Weekly (1w)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        fig_weekly = plot_weekly_dashboard(df_weekly_recent, ticker)
        st.pyplot(fig_weekly)
        
        st.markdown("---")
        
        # Análisis detallado
        st.markdown("""
        <div style='text-align: center; margin: 20px 0;'>
            <h3 style='color: #00D9FF;'>💡 Análisis Técnico Detallado</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Indicadores Técnicos Actuales")
            
            # MACD-V Analysis
            if current_weekly['MACD_V'] > 150:
                macd_interpretation = "⚠️ **Sobreextensión alcista extrema** (RIESGO MACD-V) - Alta probabilidad de corrección"
            elif current_weekly['MACD_V'] > 50:
                macd_interpretation = "✅ **Momentum alcista fuerte** - Tendencia positiva confirmada"
            elif current_weekly['MACD_V'] > -50:
                macd_interpretation = "⚪ **Momentum neutral** - Sin dirección clara definida"
            elif current_weekly['MACD_V'] > -150:
                macd_interpretation = "⚠️ **Momentum bajista** - Presión vendedora presente"
            else:
                macd_interpretation = "🚨 **Sobreextensión bajista extrema** (RIESGO MACD-V) - Posible rebote técnico"
            
            st.markdown(f"**MACD-V:** {macd_interpretation}")
            
            # Donchian Position Analysis
            if current_weekly['Close'] > current_weekly['Donchian_Upper']:
                donch_interpretation = "🚨 **RIESGO DONCHIAN** - Precio por ENCIMA del canal (breakout extremo)"
            elif current_weekly['Close'] < current_weekly['Donchian_Lower']:
                donch_interpretation = "🚨 **RIESGO DONCHIAN** - Precio por DEBAJO del canal (breakdown extremo)"
            elif donch_pos > 80:
                donch_interpretation = "🔝 **Zona superior del canal** - Cerca de resistencia"
            elif donch_pos > 50:
                donch_interpretation = "📈 **Zona alcista** - Por encima del medio del canal"
            elif donch_pos > 20:
                donch_interpretation = "📉 **Zona bajista** - Por debajo del medio del canal"
            else:
                donch_interpretation = "🔻 **Zona inferior del canal** - Cerca de soporte"
            
            st.markdown(f"**Posición Donchian:** {donch_interpretation}")
            
            # Choppiness Analysis
            if current_weekly['Choppiness'] > 61.8:
                chop_interpretation = "🔄 **Mercado Choppy** - Evitar estrategias de seguimiento de tendencia"
            elif current_weekly['Choppiness'] < 38.2:
                chop_interpretation = "🎯 **Mercado Trending** - Favorable para operar tendencias"
            else:
                chop_interpretation = "⚖️ **Mercado Neutral** - Transición entre estados"
            
            st.markdown(f"**Choppiness:** {chop_interpretation}")
        
        with col2:
            st.markdown("#### 🎯 Contexto de Mercado")
            
            # SMA Trend
            if current_weekly['Close'] > current_weekly['SMA_20'] > current_weekly['SMA_50']:
                sma_trend = "📈 **Tendencia alcista confirmada** - Precio > SMA20 > SMA50"
            elif current_weekly['Close'] < current_weekly['SMA_20'] < current_weekly['SMA_50']:
                sma_trend = "📉 **Tendencia bajista confirmada** - Precio < SMA20 < SMA50"
            else:
                sma_trend = "🔀 **Tendencia mixta** - SMAs entrelazadas, sin dirección clara"
            
            st.markdown(f"**SMAs:** {sma_trend}")
            
            # Regime interpretation
            regime_interpretation = {
                'ALCISTA': "🟢 **Régimen Alcista** - Momentum positivo, favorable para posiciones largas",
                'BAJISTA': "🔴 **Régimen Bajista** - Momentum negativo, precaución con posiciones largas",
                'RANGO': "🟡 **Régimen de Rango** - Mercado lateral, operar dentro del rango",
                'RIESGO_DONCHIAN': "🔴 **Régimen de Riesgo (Donchian)** - Precio fuera del canal, alta probabilidad de reversión",
                'RIESGO_MACDV': "🟠 **Régimen de Riesgo (MACD-V)** - Momentum sobreextendido, precaución extrema"
            }
            
            st.markdown(f"**Régimen Actual:** {regime_interpretation[current_weekly['Regime_Weekly']]}")
            
            # Volatility context
            recent_volatility = df_weekly_recent['Close'].pct_change().std() * 100
            if recent_volatility > 5:
                vol_context = f"⚡ **Alta volatilidad** ({recent_volatility:.1f}%) - Mayor riesgo en operaciones"
            elif recent_volatility > 2:
                vol_context = f"📊 **Volatilidad moderada** ({recent_volatility:.1f}%) - Riesgo normal"
            else:
                vol_context = f"😌 **Baja volatilidad** ({recent_volatility:.1f}%) - Mercado tranquilo"
            
            st.markdown(f"**Volatilidad:** {vol_context}")
        
        st.markdown("---")
        
        # Botones de descarga
        col1, col2 = st.columns(2)
        
        with col1:
            export_cols = ['Close', 'Regime_Weekly', 'Choppiness', 'MACD_V', 'MACD_V_Signal', 
                          'Donchian_Upper', 'Donchian_Middle', 'Donchian_Lower', 
                          'SMA_20', 'SMA_50']
            csv_data = df_weekly[export_cols].to_csv()
            st.download_button(
                "📥 Descargar Datos Completos (CSV)",
                data=csv_data,
                file_name=f"regime_weekly_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            csv_recent = df_weekly_recent[export_cols].to_csv()
            st.download_button(
                "📥 Descargar Período Visible (CSV)",
                data=csv_recent,
                file_name=f"regime_weekly_{ticker}_recent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    else:
        st.markdown("""
        <div style='text-align: center; padding: 40px; 
                    background: linear-gradient(135deg, #1A1D29 0%, #2D3142 100%); 
                    border-radius: 20px; border: 3px solid #4ECDC4; margin-top: 30px;
                    box-shadow: 0 8px 30px rgba(78, 205, 196, 0.3);'>
            <h2 style='color: #4ECDC4; margin: 0;'>
                👋 Bienvenido al Market Regime Analyzer
            </h2>
            <p style='color: #B0B0B0; font-size: 18px; margin: 20px 0;'>
                Configura los parámetros en el panel lateral y presiona 
                <strong style='color: #00D9FF;'>🚀 ANALIZAR MERCADO</strong> 
                para comenzar el análisis técnico semanal avanzado.
            </p>
            <p style='color: #8E93A1; font-size: 14px; margin: 10px 0 0 0;'>
                🔴 RIESGO DONCHIAN: Precio fuera del canal<br>
                🟠 RIESGO MACD-V: Momentum extremo
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    if check_password():
        market_regime_page()
    else:
        st.markdown("""
        <div style='text-align: center; padding: 60px 20px;'>
            <h1 style='color: #FF6B6B; font-size: 48px;'>🔒 Acceso Restringido</h1>
            <p style='color: #B0B0B0; font-size: 20px; margin-top: 20px;'>
                Introduce tus credenciales en el menú lateral para acceder al análisis.
            </p>
        </div>
        """, unsafe_allow_html=True)
