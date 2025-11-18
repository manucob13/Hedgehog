# pages/Market_Regime_ADX.py
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
import warnings
from utils import check_password

warnings.filterwarnings('ignore')

# =========================================================================
# CONFIGURACIÓN
# =========================================================================

st.set_page_config(page_title="Market Regime ADX", layout="wide")

# =========================================================================
# FUNCIONES TÉCNICAS ADX
# =========================================================================

def calculate_adx(df, period=14):
    """Calcula ADX (Average Directional Index)"""
    high_diff = df['High'].diff()
    low_diff = -df['Low'].diff()
    
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    atr = true_range.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(span=period, adjust=False).mean()
    
    return adx, plus_di, minus_di

def calculate_rsi(df, period=14):
    """Calcula RSI"""
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_sma(data, period):
    """Calcula SMA"""
    return data.rolling(window=period).mean()

def calculate_ema(data, period):
    """Calcula EMA (Exponential Moving Average)"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_atr(df, period=26):
    """Calcula ATR (Average True Range)"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def calculate_macd_v(df, fast_len=12, slow_len=26, signal_len=20, atr_len=26):
    """Calcula MACD-V (MACD normalizado por volatilidad) con parámetros personalizados"""
    fast_ema = calculate_ema(df['Close'], fast_len)
    slow_ema = calculate_ema(df['Close'], slow_len)
    atr = calculate_atr(df, atr_len)
    
    macd = ((fast_ema - slow_ema) / atr) * 100
    signal = calculate_ema(macd, signal_len)
    
    return macd, signal

# =========================================================================
# PREPARACIÓN DE DATOS
# =========================================================================

@st.cache_data(ttl=timedelta(hours=1))
def prepare_data(ticker, start_date='2018-01-01'):
    """Descarga datos DIARIOS y calcula indicadores"""
    try:
        data = yf.download(ticker, start=start_date, interval='1d', progress=False)
        
        if data.empty:
            return None
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        df = pd.DataFrame({
            'Close': data['Close'].squeeze(),
            'Open': data['Open'].squeeze(),
            'High': data['High'].squeeze(),
            'Low': data['Low'].squeeze(),
            'Volume': data['Volume'].squeeze()
        }, index=data.index)
        
        df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['ADX'], df['Plus_DI'], df['Minus_DI'] = calculate_adx(df, period=14)
        df['RSI'] = calculate_rsi(df, period=14)
        df['SMA_20'] = calculate_sma(df['Close'], 20)
        df['SMA_50'] = calculate_sma(df['Close'], 50)
        
        df['MACD_V'], df['MACD_V_Signal'] = calculate_macd_v(
            df, 
            fast_len=12, 
            slow_len=26, 
            signal_len=20, 
            atr_len=26
        )
        
        df = df.dropna()
        
        return df
    
    except Exception as e:
        st.error(f"Error descargando datos para {ticker}: {str(e)}")
        return None

# =========================================================================
# CLASIFICACIÓN DE REGÍMENES
# =========================================================================

def classify_regime(df):
    """Clasifica régimen usando ADX + RSI + SMAs"""
    regimes = []
    states = []
    
    for idx, row in df.iterrows():
        adx = row['ADX']
        rsi = row['RSI']
        plus_di = row['Plus_DI']
        minus_di = row['Minus_DI']
        price = row['Close']
        sma_20 = row['SMA_20']
        sma_50 = row['SMA_50']
        
        if rsi >= 75 and adx > 25 and plus_di > minus_di:
            regime = 'RIESGO'
            state = 'Overbought'
        elif rsi <= 25 and adx > 25 and minus_di > plus_di:
            regime = 'RIESGO'
            state = 'Oversold'
        elif (adx > 25 and plus_di > minus_di and 
              price > sma_20 and sma_20 > sma_50):
            regime = 'ALCISTA'
            state = 'Uptrend'
        elif (adx > 25 and minus_di > plus_di and 
              price < sma_20 and sma_20 < sma_50):
            regime = 'BAJISTA'
            state = 'Downtrend'
        else:
            regime = 'RANGO'
            state = 'Ranging'
        
        regimes.append(regime)
        states.append(state)
    
    return regimes, states

# =========================================================================
# ANÁLISIS
# =========================================================================

def analyze_regime(ticker, start_date, lookback_days):
    """Analiza regímenes"""
    df = prepare_data(ticker, start_date)
    
    if df is None or df.empty:
        return None, None
    
    df['Regime_Name'], df['State'] = classify_regime(df)
    df_recent = df.tail(lookback_days)
    
    return df, df_recent

# =========================================================================
# VISUALIZACIÓN MEJORADA Y MODERNA
# =========================================================================

def plot_regime_dashboard(df_recent, ticker):
    """Dashboard moderno con gradientes y diseño premium"""
    
    required_cols = ['Close', 'Regime_Name', 'SMA_20', 'SMA_50', 'RSI', 'ADX', 
                     'Plus_DI', 'Minus_DI', 'MACD_V_Signal']
    
    missing_cols = [col for col in required_cols if col not in df_recent.columns]
    if missing_cols:
        st.error(f"❌ Columnas faltantes: {missing_cols}")
        return None
    
    # Paleta de colores moderna y vibrante
    regime_colors = {
        'RIESGO': '#FF6B6B',      # Rojo coral vibrante
        'BAJISTA': '#EE5A6F',     # Rosa rojizo
        'RANGO': '#95A5A6',       # Gris neutro
        'ALCISTA': '#4ECDC4'      # Turquesa brillante
    }
    
    # Configuración de figura con fondo oscuro elegante
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(24, 16), facecolor='#0E1117')
    gs = fig.add_gridspec(5, 1, height_ratios=[3.8, 1, 1, 1, 1.2], hspace=0.4)
    
    # =====================================================================
    # GRÁFICO 1: PRECIO CON REGÍMENES - DISEÑO PREMIUM
    # =====================================================================
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#1A1D29')
    
    # Gradiente de fondo
    ax1.fill_between(df_recent.index, 
                     df_recent['Close'].min() * 0.97, 
                     df_recent['Close'].max() * 1.03,
                     color='#0E1117', alpha=0.5, zorder=0)
    
    # Línea de precio con efecto glow
    ax1.plot(df_recent.index, df_recent['Close'], 
             color='#FFFFFF', alpha=0.15, linewidth=6, zorder=1)
    ax1.plot(df_recent.index, df_recent['Close'], 
             color='#E0E0E0', alpha=0.4, linewidth=3, zorder=2)
    ax1.plot(df_recent.index, df_recent['Close'], 
             color='#FFFFFF', linewidth=1.5, zorder=3)
    
    # Puntos de régimen con efecto glow
    for regime_name, color in regime_colors.items():
        mask = df_recent['Regime_Name'] == regime_name
        if mask.sum() > 0:
            # Halo exterior
            ax1.scatter(df_recent[mask].index, 
                       df_recent[mask]['Close'],
                       c=color, 
                       alpha=0.15, 
                       s=220,
                       edgecolors='none',
                       zorder=4)
            # Punto principal
            ax1.scatter(df_recent[mask].index, 
                       df_recent[mask]['Close'],
                       c=color, 
                       label=regime_name,
                       alpha=0.95, 
                       s=85,
                       edgecolors='white', 
                       linewidth=1.5,
                       zorder=5)
    
    # SMAs con colores vibrantes
    ax1.plot(df_recent.index, df_recent['SMA_20'], 
             color='#00D9FF', alpha=0.9, linewidth=2.5, linestyle='-',
             label='SMA(20)', zorder=3)
    ax1.plot(df_recent.index, df_recent['SMA_50'], 
             color='#BD93F9', alpha=0.9, linewidth=2.5, linestyle='-',
             label='SMA(50)', zorder=3)
    
    # Punto actual destacado
    current = df_recent.iloc[-1]
    ax1.scatter(current.name, current['Close'], 
               facecolors='none',
               edgecolors=regime_colors[current['Regime_Name']], 
               s=450,
               linewidth=5,
               alpha=0.4,
               marker='o', 
               zorder=9)
    ax1.scatter(current.name, current['Close'], 
               facecolors=regime_colors[current['Regime_Name']],
               edgecolors='white', 
               s=280,
               linewidth=4,
               marker='o', 
               label=f'📍 Actual: {current["Regime_Name"]}', 
               zorder=10)
    
    # Anotación moderna
    bbox_color = regime_colors[current['Regime_Name']]
    ax1.annotate(f'{current["Regime_Name"]}\n${current["Close"]:.2f}',
                xy=(current.name, current['Close']),
                xytext=(25, 40), 
                textcoords='offset points',
                fontsize=14,
                fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=1', 
                         facecolor=bbox_color, 
                         alpha=0.95, 
                         edgecolor='white', 
                         linewidth=3),
                arrowprops=dict(arrowstyle='->', lw=3, 
                               color=bbox_color, 
                               connectionstyle='arc3,rad=0.3'),
                zorder=11)
    
    # Título elegante
    ax1.text(0.5, 1.10, f'{ticker}', 
            transform=ax1.transAxes,
            fontsize=28, fontweight='bold', 
            ha='center', color='#FFFFFF')
    ax1.text(0.5, 1.05, 'Market Regime Analysis • Enhanced Daily View', 
            transform=ax1.transAxes,
            fontsize=13, style='italic',
            ha='center', color='#8E93A1')
    
    ax1.set_ylabel('Price ($)', fontsize=15, fontweight='700', color='#FFFFFF', labelpad=12)
    
    legend = ax1.legend(loc='upper left', fontsize=11, framealpha=0.95, 
                       ncol=3, edgecolor='#00D9FF', fancybox=True,
                       borderpad=1.2, labelspacing=1, columnspacing=2)
    legend.get_frame().set_facecolor('#1A1D29')
    legend.get_frame().set_linewidth(2)
    
    ax1.grid(True, alpha=0.08, linestyle='-', linewidth=1, color='#FFFFFF')
    ax1.tick_params(labelsize=11, colors='#B0B0B0', width=1.5)
    for spine in ax1.spines.values():
        spine.set_color('#2D3142')
        spine.set_linewidth(2)
    
    # =====================================================================
    # GRÁFICO 2: RSI CON DISEÑO MODERNO
    # =====================================================================
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_facecolor('#1A1D29')
    
    # Línea RSI con gradiente
    ax2.plot(df_recent.index, df_recent['RSI'], 
             color='#FF79C6', linewidth=2.5, label='RSI', zorder=3)
    
    # Zonas de color
    ax2.fill_between(df_recent.index, df_recent['RSI'], 50,
                     where=(df_recent['RSI'] >= 50), 
                     color='#4ECDC4', alpha=0.2, zorder=1)
    ax2.fill_between(df_recent.index, df_recent['RSI'], 50,
                     where=(df_recent['RSI'] < 50), 
                     color='#FF6B6B', alpha=0.2, zorder=1)
    
    # Líneas de referencia
    ax2.axhline(y=75, color='#FF6B6B', linestyle='--', linewidth=2, alpha=0.8, zorder=2)
    ax2.axhline(y=70, color='#FFB86C', linestyle=':', linewidth=1.5, alpha=0.7, zorder=2)
    ax2.axhline(y=50, color='#8E93A1', linestyle='-', linewidth=1.5, alpha=0.7, zorder=2)
    ax2.axhline(y=30, color='#FFB86C', linestyle=':', linewidth=1.5, alpha=0.7, zorder=2)
    ax2.axhline(y=25, color='#4ECDC4', linestyle='--', linewidth=2, alpha=0.8, zorder=2)
    
    # Zonas extremas
    ax2.fill_between(df_recent.index, 75, 100, alpha=0.12, color='#FF6B6B', zorder=0)
    ax2.fill_between(df_recent.index, 0, 25, alpha=0.12, color='#4ECDC4', zorder=0)
    
    # Punto actual
    ax2.scatter(current.name, current['RSI'], 
               facecolors='#FF79C6',
               edgecolors='white', 
               s=220,
               linewidth=4,
               marker='o',
               zorder=10)
    
    rsi_color = '#FF6B6B' if current['RSI'] > 70 else '#4ECDC4' if current['RSI'] < 30 else '#8E93A1'
    rsi_status = 'Overbought' if current['RSI'] > 70 else 'Oversold' if current['RSI'] < 30 else 'Neutral'
    
    ax2.text(0.02, 0.88, f'RSI: {current["RSI"]:.1f}', 
            transform=ax2.transAxes, 
            fontsize=13, 
            fontweight='bold',
            color='white',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.7', 
                     facecolor=rsi_color, 
                     alpha=0.95, 
                     edgecolor='white', 
                     linewidth=2.5))
    ax2.text(0.14, 0.88, rsi_status, 
            transform=ax2.transAxes, 
            fontsize=11, 
            style='italic',
            fontweight='600',
            color=rsi_color,
            verticalalignment='top')
    
    ax2.set_ylabel('RSI', fontsize=14, fontweight='700', color='#FFFFFF', labelpad=12)
    ax2.set_ylim([0, 100])
    ax2.grid(True, alpha=0.08, linestyle='-', linewidth=1, color='#FFFFFF')
    ax2.tick_params(labelsize=10.5, colors='#B0B0B0', width=1.5)
    for spine in ax2.spines.values():
        spine.set_color('#2D3142')
        spine.set_linewidth(2)
    
    # =====================================================================
    # GRÁFICO 3: ADX CON DISEÑO PREMIUM
    # =====================================================================
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.set_facecolor('#1A1D29')
    
    ax3.plot(df_recent.index, df_recent['ADX'], 
             color='#F8F8F2', linewidth=2.5, label='ADX', zorder=3)
    ax3.plot(df_recent.index, df_recent['Plus_DI'], 
             color='#4ECDC4', linewidth=2.2, alpha=0.9, label='+DI', zorder=2)
    ax3.plot(df_recent.index, df_recent['Minus_DI'], 
             color='#FF6B6B', linewidth=2.2, alpha=0.9, label='-DI', zorder=2)
    
    ax3.axhline(y=25, color='#4ECDC4', linestyle='--', linewidth=2, alpha=0.8)
    ax3.axhline(y=20, color='#FFB86C', linestyle=':', linewidth=1.5, alpha=0.7)
    
    ax3.fill_between(df_recent.index, 0, 20, alpha=0.12, color='#95A5A6')
    ax3.fill_between(df_recent.index, 25, 70, alpha=0.12, color='#4ECDC4')
    
    ax3.scatter(current.name, current['ADX'], 
               facecolors='#F8F8F2',
               edgecolors='white', 
               s=220,
               linewidth=4,
               marker='o',
               zorder=10)
    
    adx_color = '#4ECDC4' if current['ADX'] > 25 else '#FFB86C' if current['ADX'] > 20 else '#95A5A6'
    trend_strength = 'Strong' if current['ADX'] > 25 else 'Moderate' if current['ADX'] > 20 else 'Weak'
    
    ax3.text(0.02, 0.82, f'ADX: {current["ADX"]:.1f}', 
            transform=ax3.transAxes, 
            fontsize=12,
            fontweight='bold',
            color='white',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.6',
                     facecolor=adx_color, 
                     alpha=0.95, 
                     edgecolor='white', 
                     linewidth=2))
    ax3.text(0.12, 0.82, trend_strength, 
            transform=ax3.transAxes, 
            fontsize=10,
            style='italic',
            fontweight='600',
            color=adx_color,
            verticalalignment='top')
    
    ax3.set_ylabel('ADX / DI', fontsize=14, fontweight='700', color='#FFFFFF', labelpad=12)
    ax3.set_ylim([0, 70])
    legend = ax3.legend(loc='upper left', fontsize=10, framealpha=0.95, 
                       ncol=3, edgecolor='#00D9FF', fancybox=True)
    legend.get_frame().set_facecolor('#1A1D29')
    legend.get_frame().set_linewidth(2)
    ax3.grid(True, alpha=0.08, linestyle='-', linewidth=1, color='#FFFFFF')
    ax3.tick_params(labelsize=10.5, colors='#B0B0B0', width=1.5)
    for spine in ax3.spines.values():
        spine.set_color('#2D3142')
        spine.set_linewidth(2)
    
    # =====================================================================
    # GRÁFICO 4: MACD-V CON BANDAS DE COLOR MEJORADAS
    # =====================================================================
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.set_facecolor('#1A1D29')
    
    if 'MACD_V_Signal' in df_recent.columns and not df_recent['MACD_V_Signal'].isna().all():
        macd_signal = df_recent['MACD_V_Signal'].dropna()
        
        if len(macd_signal) > 0:
            macd_max = macd_signal.max()
            macd_min = macd_signal.min()
            macd_range = macd_max - macd_min
            
            y_max = macd_max + (macd_range * 0.25)
            y_min = macd_min - (macd_range * 0.25)
            
            # Bandas de régimen con colores vibrantes
            ax4.fill_between(df_recent.index, 150, y_max, 
                             color='#FF6B6B', alpha=0.20, zorder=0)
            ax4.fill_between(df_recent.index, 50, 150, 
                             color='#4ECDC4', alpha=0.25, zorder=0)
            ax4.fill_between(df_recent.index, -50, 50, 
                             color='#95A5A6', alpha=0.20, zorder=0)
            ax4.fill_between(df_recent.index, -150, -50, 
                             color='#EE5A6F', alpha=0.25, zorder=0)
            ax4.fill_between(df_recent.index, y_min, -150, 
                             color='#FF6B6B', alpha=0.20, zorder=0)
            
            # Líneas de referencia con glow
            for y_val, color, width in [(150, '#FF6B6B', 2.2), (50, '#4ECDC4', 1.8), 
                                         (0, '#FFFFFF', 2), (-50, '#EE5A6F', 1.8), 
                                         (-150, '#FF6B6B', 2.2)]:
                ax4.axhline(y=y_val, color=color, linestyle='--', linewidth=width, 
                           alpha=0.9, zorder=2)
            
            # Línea MACD-V con degradado
            for i in range(1, len(df_recent)):
                if pd.notna(df_recent['MACD_V_Signal'].iloc[i-1]) and pd.notna(df_recent['MACD_V_Signal'].iloc[i]):
                    x1, x2 = df_recent.index[i-1], df_recent.index[i]
                    y1, y2 = df_recent['MACD_V_Signal'].iloc[i-1], df_recent['MACD_V_Signal'].iloc[i]
                    
                    if y2 > 150:
                        color, width = '#FF6B6B', 3
                    elif y2 > 50:
                        color, width = '#4ECDC4', 2.8
                    elif y2 > -50:
                        color, width = '#95A5A6', 2.5
                    elif y2 > -150:
                        color, width = '#EE5A6F', 2.8
                    else:
                        color, width = '#FF6B6B', 3
                    
                    ax4.plot([x1, x2], [y1, y2], color=color, linewidth=width, 
                            alpha=0.95, zorder=5)
            
            # Punto actual destacado
            if pd.notna(current['MACD_V_Signal']):
                if current['MACD_V_Signal'] > 150:
                    current_color = '#FF6B6B'
                    regime_label = 'RIESGO ALCISTA'
                elif current['MACD_V_Signal'] > 50:
                    current_color = '#4ECDC4'
                    regime_label = 'ALCISTA'
                elif current['MACD_V_Signal'] > -50:
                    current_color = '#95A5A6'
                    regime_label = 'RANGO'
                elif current['MACD_V_Signal'] > -150:
                    current_color = '#EE5A6F'
                    regime_label = 'BAJISTA'
                else:
                    current_color = '#FF6B6B'
                    regime_label = 'RIESGO BAJISTA'
                
                # Glow effect
                ax4.scatter(current.name, current['MACD_V_Signal'], 
                           facecolors=current_color,
                           edgecolors='none', 
                           s=400,
                           alpha=0.3,
                           marker='o',
                           zorder=9)
                
                ax4.scatter(current.name, current['MACD_V_Signal'], 
                           facecolors=current_color,
                           edgecolors='white', 
                           s=220,
                           linewidth=4,
                           marker='o',
                           alpha=0.95,
                           zorder=10)
                
                if len(df_recent) >= 2 and pd.notna(df_recent['MACD_V_Signal'].iloc[-2]):
                    signal_diff = current['MACD_V_Signal'] - df_recent['MACD_V_Signal'].iloc[-2]
                    signal_direction = '▲' if signal_diff > 0 else '▼'
                    direction_color = '#4ECDC4' if signal_diff > 0 else '#FF6B6B'
                    
                    ax4.text(0.02, 0.22, f'MACD-V: {current["MACD_V_Signal"]:.1f}', 
                            transform=ax4.transAxes, 
                            fontsize=12,
                            fontweight='bold',
                            color='white',
                            verticalalignment='top',
                            bbox=dict(boxstyle='round,pad=0.6',
                                     facecolor=current_color, 
                                     alpha=0.95, 
                                     edgecolor='white', 
                                     linewidth=2))
                    
                    ax4.text(0.02, 0.08, f'{signal_direction} {abs(signal_diff):.1f}', 
                            transform=ax4.transAxes, 
                            fontsize=10,
                            fontweight='bold',
                            color=direction_color,
                            verticalalignment='top')
                    
                    ax4.text(0.98, 0.12, regime_label, 
                            transform=ax4.transAxes, 
                            fontsize=11,
                            fontweight='bold',
                            color='white',
                            verticalalignment='top',
                            horizontalalignment='right',
                            bbox=dict(boxstyle='round,pad=0.5',
                                     facecolor=current_color, 
                                     alpha=0.95, 
                                     edgecolor='white', 
                                     linewidth=2))
            
            ax4.set_ylim([y_min, y_max])
    
    ax4.set_ylabel('MACD-V', fontsize=14, fontweight='700', color='#FFFFFF', labelpad=12)
    ax4.grid(True, alpha=0.08, linestyle=':', linewidth=1, color='#FFFFFF')
    ax4.tick_params(labelsize=10.5, colors='#B0B0B0', width=1.5)
    for spine in ax4.spines.values():
        spine.set_color('#2D3142')
        spine.set_linewidth(2)
    
    # =====================================================================
    # GRÁFICO 5: TIMELINE DE REGÍMENES MEJORADO
    # =====================================================================
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    ax5.set_facecolor('#1A1D29')
    
    regime_order = ['BAJISTA', 'RANGO', 'ALCISTA', 'RIESGO']
    
    # Línea conectora sutil
    regime_nums = [regime_order.index(r) for r in df_recent['Regime_Name']]
    ax5.plot(df_recent.index, regime_nums, 
            color='#2D3142', linewidth=2, alpha=0.6, zorder=1, linestyle='-')
    
    # Puntos de régimen con glow effect
    for regime_name in regime_order:
        mask = df_recent['Regime_Name'] == regime_name
        if mask.sum() > 0:
            color = regime_colors[regime_name]
            regime_num = regime_order.index(regime_name)
            
            # Halo exterior
            ax5.scatter(df_recent[mask].index, 
                       [regime_num] * mask.sum(),
                       c=color, 
                       alpha=0.25, 
                       s=200,
                       edgecolors='none',
                       zorder=3)
            # Punto principal
            ax5.scatter(df_recent[mask].index, 
                       [regime_num] * mask.sum(),
                       c=color, 
                       alpha=0.95, 
                       s=110,
                       edgecolors='white', 
                       linewidth=1.8,
                       zorder=4)
    
    # Punto actual destacado
    current_regime_num = regime_order.index(current['Regime_Name'])
    ax5.scatter(current.name, current_regime_num, 
               facecolors=regime_colors[current['Regime_Name']],
               edgecolors='white', 
               s=300,
               linewidth=4.5,
               marker='o',
               zorder=10)
    
    ax5.set_ylabel('Regime', fontsize=14, fontweight='700', color='#FFFFFF', labelpad=12)
    ax5.set_yticks(range(4))
    ax5.set_yticklabels(regime_order, fontsize=12, fontweight='700', color='#E0E0E0')
    ax5.set_xlabel('Date', fontsize=15, fontweight='700', color='#FFFFFF', labelpad=12)
    ax5.grid(True, alpha=0.08, linestyle='-', linewidth=1, color='#FFFFFF', axis='x')
    ax5.tick_params(labelsize=11, colors='#B0B0B0', width=1.5)
    for spine in ax5.spines.values():
        spine.set_color('#2D3142')
        spine.set_linewidth(2)
    
    plt.tight_layout()
    
    return fig

# =========================================================================
# PÁGINA PRINCIPAL CON UI MEJORADA
# =========================================================================

def market_regime_page():
    # CSS personalizado para mejorar la UI
    st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
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
    
    st.title("📊 Market Regime Analyzer Pro")
    st.markdown("---")
    
    # Header con diseño mejorado
    col_header1, col_header2 = st.columns([3, 1])
    with col_header1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1A1D29 0%, #2D3142 100%); 
                    padding: 20px; border-radius: 15px; border: 2px solid #4ECDC4;
                    box-shadow: 0 4px 15px rgba(78, 205, 196, 0.3);'>
            <h3 style='color: #4ECDC4; margin: 0;'>🔍 Advanced Technical Analysis</h3>
            <p style='color: #B0B0B0; margin: 5px 0 0 0;'>
                Análisis de regímenes usando ADX + RSI + SMAs + MACD-V (Daily Timeframe)
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #4ECDC4 0%, #00D9FF 100%); 
                    border-radius: 15px; margin-bottom: 20px;'>
            <h2 style='color: white; margin: 0;'>⚙️ Settings</h2>
        </div>
        """, unsafe_allow_html=True)
        
        ticker = st.text_input(
            "🎯 Ticker Symbol",
            value="AAPL",
            help="Ingresa el símbolo del ticker (ej: AAPL, MSFT, SPY, TSLA)"
        ).upper()
        
        st.markdown("---")
        
        lookback_months = st.slider(
            "📅 Meses a Visualizar",
            min_value=1,
            max_value=12,
            value=3,
            step=1,
            help="Selecciona el número de meses de historia a mostrar en los gráficos"
        )
        
        lookback_days = int(lookback_months * 21)
        
        st.markdown("---")
        
        start_date = st.date_input(
            "📆 Fecha Inicio de Datos",
            value=datetime(2018, 1, 1),
            help="Fecha de inicio para la descarga de datos históricos completos"
        )
        
        st.markdown("---")
        
        analizar_btn = st.button(
            "🚀 ANALIZAR RÉGIMEN",
            type="primary",
            use_container_width=True,
            key="sidebar_analyze"
        )
        
        st.markdown("---")
        
        # Metodología con diseño mejorado
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1A1D29 0%, #2D3142 100%); 
                    padding: 15px; border-radius: 12px; border: 2px solid #BD93F9;'>
            <h3 style='color: #BD93F9; margin-top: 0;'>📖 Metodología</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **⏱️ Timeframe: Diario (1d)**
        
        **📊 ADX (Fuerza de Tendencia):**
        - ✅ ADX > 25: Tendencia fuerte
        - ⚠️ ADX 20-25: Tendencia moderada
        - ❌ ADX < 20: Sin tendencia / Rango
        
        **📈 RSI (Momentum):**
        - 🔴 RSI > 75: Overbought (riesgo)
        - 🟢 RSI > 70: Zona de precaución
        - ⚪ RSI 30-70: Zona neutral
        - 🟢 RSI < 30: Zona de oportunidad
        - 🔵 RSI < 25: Oversold (riesgo)
        
        **🎯 MACD-V (Volatility-Adjusted):**
        - Fast: 12 | Slow: 26
        - Signal: 20 | ATR: 26
        - **>150**: 🟠 Riesgo Alcista
        - **50-150**: 🟢 Alcista
        - **-50 a 50**: ⚫ Rango
        - **-150 a -50**: 🔴 Bajista
        - **<-150**: 🟠 Riesgo Bajista
        
        **🎨 Regímenes de Mercado:**
        - 🟢 **ALCISTA**: Tendencia fuerte alcista confirmada
        - 🔴 **BAJISTA**: Tendencia fuerte bajista confirmada
        - ⚫ **RANGO**: Sin dirección clara, lateralización
        - 🟠 **RIESGO**: Zonas extremas, posible reversión
        """)
    
    if analizar_btn:
        with st.spinner(f"🔄 Descargando y procesando datos para {ticker}..."):
            df, df_recent = analyze_regime(
                ticker,
                start_date.strftime('%Y-%m-%d'),
                lookback_days
            )
            
            if df is None or df_recent is None:
                st.error(f"❌ No se pudieron obtener datos para {ticker}. Verifica el símbolo.")
                st.stop()
            
            st.session_state.last_ticker = ticker
            st.session_state.df = df
            st.session_state.df_recent = df_recent
            st.session_state.lookback_months = lookback_months
            st.success(f"✅ Datos cargados exitosamente para {ticker}")
    
    if 'df' in st.session_state and 'df_recent' in st.session_state:
        df = st.session_state.df
        df_recent = st.session_state.df_recent
        current = df.iloc[-1]
        
        if 'last_ticker' not in st.session_state or st.session_state.last_ticker != ticker:
            st.warning("⚠️ El ticker ha cambiado. Presiona '🚀 ANALIZAR RÉGIMEN' para actualizar.")
        
        st.markdown("---")
        
        # Métricas principales con diseño mejorado
        st.markdown("""
        <div style='text-align: center; margin-bottom: 20px;'>
            <h2 style='color: #00D9FF; font-size: 32px; margin: 0;'>🎯 Estado Actual del Mercado</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            regime_emoji = {
                'ALCISTA': '🟢',
                'BAJISTA': '🔴',
                'RANGO': '⚫',
                'RIESGO': '🟠'
            }[current['Regime_Name']]
            st.metric(
                "RÉGIMEN",
                f"{regime_emoji} {current['Regime_Name']}"
            )
        
        with col2:
            price_change = ((current['Close'] - df['Close'].iloc[-5]) / df['Close'].iloc[-5] * 100)
            st.metric(
                "PRECIO",
                f"${current['Close']:.2f}",
                f"{price_change:+.2f}%"
            )
        
        with col3:
            adx_status = "🔥 Fuerte" if current['ADX'] > 25 else "⚡ Moderado" if current['ADX'] > 20 else "💤 Débil"
            st.metric(
                "ADX",
                f"{current['ADX']:.1f}",
                adx_status
            )
        
        with col4:
            rsi_status = "🔴 OB" if current['RSI'] > 70 else "🔵 OS" if current['RSI'] < 30 else "⚪ Neutral"
            st.metric(
                "RSI",
                f"{current['RSI']:.1f}",
                rsi_status
            )
        
        with col5:
            try:
                if 'MACD_V_Signal' in df.columns and len(df) >= 2:
                    current_macd = current.get('MACD_V_Signal')
                    previous_macd = df['MACD_V_Signal'].iloc[-2]
                    
                    if pd.notna(current_macd) and pd.notna(previous_macd):
                        macd_direction = "📈 ↑" if current_macd > previous_macd else "📉 ↓"
                        macd_delta = f"{current_macd - previous_macd:+.1f}"
                        st.metric(
                            "MACD-V",
                            f"{current_macd:.1f}",
                            f"{macd_direction} {macd_delta}"
                        )
                    else:
                        st.metric("MACD-V", "N/A", "Sin datos")
                else:
                    st.metric("MACD-V", "N/A", "Insuficiente")
            except (KeyError, IndexError, TypeError):
                st.metric("MACD-V", "Error", "")
        
        with col6:
            st.metric(
                "FECHA",
                current.name.strftime('%Y-%m-%d'),
                current.name.strftime('%A')[:3]
            )
        
        st.markdown("---")
        
        # Recomendación con diseño mejorado
        st.markdown("""
        <div style='text-align: center; margin-bottom: 15px;'>
            <h3 style='color: #FFB86C; font-size: 24px;'>💡 Recomendación de Trading</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if current['Regime_Name'] == 'RIESGO':
            if current['RSI'] > 70:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #FF6B6B 0%, #EE5A6F 100%); 
                            padding: 20px; border-radius: 15px; border: 3px solid #FFFFFF;
                            box-shadow: 0 4px 20px rgba(255, 107, 107, 0.5);'>
                    <h4 style='color: white; margin: 0;'>⚠️ PRECAUCIÓN EXTREMA</h4>
                    <p style='color: white; margin: 10px 0 0 0; font-size: 16px;'>
                        Zona de <strong>SOBRECOMPRA</strong> crítica. Considera tomar ganancias parciales 
                        o totales. Alto riesgo de corrección inminente.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #4ECDC4 0%, #00D9FF 100%); 
                            padding: 20px; border-radius: 15px; border: 3px solid #FFFFFF;
                            box-shadow: 0 4px 20px rgba(78, 205, 196, 0.5);'>
                    <h4 style='color: white; margin: 0;'>💡 OPORTUNIDAD POTENCIAL</h4>
                    <p style='color: white; margin: 10px 0 0 0; font-size: 16px;'>
                        Zona de <strong>SOBREVENTA</strong>. Posible rebote técnico. 
                        Espera confirmación con ADX > 25 y RSI > 30 antes de entrar.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        elif current['Regime_Name'] == 'ALCISTA':
            st.markdown("""
            <div style='background: linear-gradient(135deg, #4ECDC4 0%, #00D9FF 100%); 
                        padding: 20px; border-radius: 15px; border: 3px solid #FFFFFF;
                        box-shadow: 0 4px 20px rgba(78, 205, 196, 0.5);'>
                <h4 style='color: white; margin: 0;'>✅ SEÑAL POSITIVA</h4>
                <p style='color: white; margin: 10px 0 0 0; font-size: 16px;'>
                    Tendencia <strong>ALCISTA</strong> fuerte confirmada. Mantén posiciones largas 
                    o busca puntos de entrada en retrocesos hacia SMA(20).
                </p>
            </div>
            """, unsafe_allow_html=True)
        elif current['Regime_Name'] == 'BAJISTA':
            st.markdown("""
            <div style='background: linear-gradient(135deg, #EE5A6F 0%, #FF6B6B 100%); 
                        padding: 20px; border-radius: 15px; border: 3px solid #FFFFFF;
                        box-shadow: 0 4px 20px rgba(238, 90, 111, 0.5);'>
                <h4 style='color: white; margin: 0;'>🔴 SEÑAL NEGATIVA</h4>
                <p style='color: white; margin: 10px 0 0 0; font-size: 16px;'>
                    Tendencia <strong>BAJISTA</strong> confirmada. Evita posiciones largas. 
                    Considera posiciones cortas o espera ADX < 20 para entrar.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #95A5A6 0%, #7F8C8D 100%); 
                        padding: 20px; border-radius: 15px; border: 3px solid #FFFFFF;
                        box-shadow: 0 4px 20px rgba(149, 165, 166, 0.5);'>
                <h4 style='color: white; margin: 0;'>⚖️ MERCADO LATERAL</h4>
                <p style='color: white; margin: 10px 0 0 0; font-size: 16px;'>
                    Sin dirección clara. Estrategia <strong>MEAN REVERSION</strong> recomendada. 
                    Opera en los extremos del rango con stops ajustados.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Gráfico principal
        lookback_display = st.session_state.get('lookback_months', lookback_months)
        st.markdown(f"""
        <div style='text-align: center; margin-bottom: 20px;'>
            <h2 style='color: #BD93F9; font-size: 28px;'>📈 Análisis Técnico Visual</h2>
            <p style='color: #8E93A1; font-size: 16px;'>
                Últimos {lookback_display} meses • Timeframe Diario (1d)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        fig = plot_regime_dashboard(df_recent, st.session_state.last_ticker)
        
        if fig is not None:
            st.pyplot(fig)
        else:
            st.error("❌ Error al generar el gráfico. Verifica la integridad de los datos.")
        
        st.markdown("---")
        
        # Estadísticas adicionales
        col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
        
        with col_stats1:
            regime_counts = df_recent['Regime_Name'].value_counts()
            st.markdown("""
            <div style='background: linear-gradient(135deg, #1A1D29 0%, #2D3142 100%); 
                        padding: 15px; border-radius: 12px; border: 2px solid #4ECDC4;'>
                <h4 style='color: #4ECDC4; margin: 0;'>📊 Distribución</h4>
            </div>
            """, unsafe_allow_html=True)
            for regime, count in regime_counts.items():
                percentage = (count / len(df_recent)) * 100
                st.write(f"{regime}: {percentage:.1f}%")
        
        with col_stats2:
            volatility = df_recent['Returns'].std() * np.sqrt(252) * 100
            st.markdown("""
            <div style='background: linear-gradient(135deg, #1A1D29 0%, #2D3142 100%); 
                        padding: 15px; border-radius: 12px; border: 2px solid #FFB86C;'>
                <h4 style='color: #FFB86C; margin: 0;'>📉 Volatilidad</h4>
            </div>
            """, unsafe_allow_html=True)
            st.metric("Anualizada", f"{volatility:.2f}%")
        
        with col_stats3:
            total_return = ((df_recent['Close'].iloc[-1] / df_recent['Close'].iloc[0]) - 1) * 100
            st.markdown("""
            <div style='background: linear-gradient(135deg, #1A1D29 0%, #2D3142 100%); 
                        padding: 15px; border-radius: 12px; border: 2px solid #BD93F9;'>
                <h4 style='color: #BD93F9; margin: 0;'>💰 Retorno</h4>
            </div>
            """, unsafe_allow_html=True)
            st.metric("Período", f"{total_return:+.2f}%")
        
        with col_stats4:
            max_dd = ((df_recent['Close'] / df_recent['Close'].cummax()) - 1).min() * 100
            st.markdown("""
            <div style='background: linear-gradient(135deg, #1A1D29 0%, #2D3142 100%); 
                        padding: 15px; border-radius: 12px; border: 2px solid #FF6B6B;'>
                <h4 style='color: #FF6B6B; margin: 0;'>📊 Drawdown</h4>
            </div>
            """, unsafe_allow_html=True)
            st.metric("Máximo", f"{max_dd:.2f}%")
        
        st.markdown("---")
        
        # Exportación de datos
        export_cols = ['Close', 'Regime_Name', 'State', 'ADX', 'RSI', 'Plus_DI', 'Minus_DI', 
                      'SMA_20', 'SMA_50']
        
        if 'MACD_V' in df.columns:
            export_cols.append('MACD_V')
        if 'MACD_V_Signal' in df.columns:
            export_cols.append('MACD_V_Signal')
        
        available_cols = [col for col in export_cols if col in df.columns]
        
        csv = df[available_cols].to_csv()
        st.download_button(
            label="📥 Descargar Datos Completos (CSV)",
            data=csv,
            file_name=f"market_regime_{st.session_state.last_ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.markdown("""
        <div style='text-align: center; padding: 40px; background: linear-gradient(135deg, #1A1D29 0%, #2D3142 100%); 
                    border-radius: 20px; border: 3px solid #4ECDC4; margin-top: 30px;
                    box-shadow: 0 8px 30px rgba(78, 205, 196, 0.3);'>
            <h2 style='color: #4ECDC4; margin: 0;'>👋 Bienvenido al Market Regime Analyzer Pro</h2>
            <p style='color: #B0B0B0; font-size: 18px; margin: 20px 0;'>
                Configura los parámetros en el panel lateral y presiona 
                <strong style='color: #00D9FF;'>'🚀 ANALIZAR RÉGIMEN'</strong> 
                para comenzar el análisis técnico avanzado.
            </p>
            <p style='color: #8E93A1; font-size: 14px; margin: 10px 0 0 0;'>
                💡 Análisis basado en ADX, RSI, SMAs y MACD-V con timeframe diario
            </p>
        </div>
        """, unsafe_allow_html=True)

# =========================================================================
# PUNTO DE ENTRADA PROTEGIDO
# =========================================================================

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
