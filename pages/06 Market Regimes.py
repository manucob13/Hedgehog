# pages/Market_Regime_ADX.py
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
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

def calculate_macd_v(df, fast_len=12, slow_len=26, signal_len=9, atr_len=26):
    """Calcula MACD-V (MACD normalizado por volatilidad)"""
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
    """Descarga datos SEMANALES y calcula indicadores"""
    try:
        data = yf.download(ticker, start=start_date, interval='1wk', progress=False)
        
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
        
        # Calcular MACD-V
        df['MACD_V'], df['MACD_V_Signal'] = calculate_macd_v(df)
        
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
        
        # RIESGO (Overbought/Oversold)
        if rsi >= 75 and adx > 25 and plus_di > minus_di:
            regime = 'RIESGO'
            state = 'Overbought'
        elif rsi <= 25 and adx > 25 and minus_di > plus_di:
            regime = 'RIESGO'
            state = 'Oversold'
        
        # ALCISTA
        elif (adx > 25 and plus_di > minus_di and 
              price > sma_20 and sma_20 > sma_50):
            regime = 'ALCISTA'
            state = 'Uptrend'
        
        # BAJISTA
        elif (adx > 25 and minus_di > plus_di and 
              price < sma_20 and sma_20 < sma_50):
            regime = 'BAJISTA'
            state = 'Downtrend'
        
        # RANGO
        else:
            regime = 'RANGO'
            state = 'Ranging'
        
        regimes.append(regime)
        states.append(state)
    
    return regimes, states

# =========================================================================
# ANÁLISIS
# =========================================================================

def analyze_regime(ticker, start_date, lookback_weeks):
    """Analiza regímenes"""
    df = prepare_data(ticker, start_date)
    
    if df is None or df.empty:
        return None, None
    
    df['Regime_Name'], df['State'] = classify_regime(df)
    
    # Filtrar últimos meses para visualización
    df_recent = df.tail(lookback_weeks)
    
    return df, df_recent

# =========================================================================
# VISUALIZACIÓN ULTRA MEJORADA CON MACD-V CORREGIDO
# =========================================================================

def plot_regime_dashboard(df_recent, ticker):
    """Dashboard consolidado con MACD-V estilo ThinkOrSwim mejorado"""
    
    # Verificar que las columnas necesarias existen
    required_cols = ['Close', 'Regime_Name', 'SMA_20', 'SMA_50', 'RSI', 'ADX', 
                     'Plus_DI', 'Minus_DI', 'MACD_V_Signal']
    
    missing_cols = [col for col in required_cols if col not in df_recent.columns]
    if missing_cols:
        st.error(f"❌ Columnas faltantes en los datos: {missing_cols}")
        return None
    
    # Paleta de colores moderna con gradientes
    regime_colors = {
        'RIESGO': '#FF6B35',
        'BAJISTA': '#DC143C',
        'RANGO': '#4A5568',
        'ALCISTA': '#10B981'
    }
    
    # Configuración de estilo moderno
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(22, 15), facecolor='#F8F9FA')
    gs = fig.add_gridspec(5, 1, height_ratios=[3.5, 1, 1, 1, 1.2], hspace=0.35)
    
    # =====================================================================
    # GRÁFICO 1: PRECIO CON REGÍMENES
    # =====================================================================
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#FFFFFF')
    
    ax1.fill_between(df_recent.index, df_recent['Close'].min() * 0.98, 
                     df_recent['Close'], color='#E8F4F8', alpha=0.3, zorder=0)
    
    ax1.plot(df_recent.index, df_recent['Close'], 
             color='#A0AEC0', alpha=0.4, linewidth=3, zorder=1)
    ax1.plot(df_recent.index, df_recent['Close'], 
             color='#2D3748', alpha=0.6, linewidth=1.5, zorder=2)
    
    for regime_name, color in regime_colors.items():
        mask = df_recent['Regime_Name'] == regime_name
        if mask.sum() > 0:
            ax1.scatter(df_recent[mask].index, 
                       df_recent[mask]['Close'],
                       c=color, 
                       alpha=0.2, 
                       s=120,
                       edgecolors='none',
                       zorder=4)
            ax1.scatter(df_recent[mask].index, 
                       df_recent[mask]['Close'],
                       c=color, 
                       label=regime_name,
                       alpha=0.9, 
                       s=65,
                       edgecolors='white', 
                       linewidth=1.2,
                       zorder=5)
    
    ax1.plot(df_recent.index, df_recent['SMA_20'], 
             color='#3B82F6', alpha=0.85, linewidth=2.5, linestyle='-', 
             label='SMA(20)', zorder=3)
    ax1.plot(df_recent.index, df_recent['SMA_50'], 
             color='#8B5CF6', alpha=0.85, linewidth=2.5, linestyle='-', 
             label='SMA(50)', zorder=3)
    
    current = df_recent.iloc[-1]
    ax1.scatter(current.name, current['Close'], 
               facecolors='none',
               edgecolors=regime_colors[current['Regime_Name']], 
               s=350,
               linewidth=4,
               alpha=0.4,
               marker='o', 
               zorder=9)
    ax1.scatter(current.name, current['Close'], 
               facecolors='none',
               edgecolors='#1A202C', 
               s=220,
               linewidth=3,
               marker='o', 
               label=f'📍 Actual: {current["Regime_Name"]}', 
               zorder=10)
    
    bbox_color = regime_colors[current['Regime_Name']]
    ax1.annotate(f'{current["Regime_Name"]}\n${current["Close"]:.2f}',
                xy=(current.name, current['Close']),
                xytext=(20, 30), 
                textcoords='offset points',
                fontsize=12,
                fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.8', 
                         facecolor=bbox_color, 
                         alpha=0.95, 
                         edgecolor='white', 
                         linewidth=2.5),
                arrowprops=dict(arrowstyle='->', lw=2.5, 
                               color=bbox_color, 
                               connectionstyle='arc3,rad=0.2'),
                zorder=11)
    
    ax1.text(0.5, 1.08, f'{ticker}', 
            transform=ax1.transAxes,
            fontsize=24, fontweight='bold', 
            ha='center', color='#1A202C')
    ax1.text(0.5, 1.03, 'Market Regime Analysis • Weekly Timeframe', 
            transform=ax1.transAxes,
            fontsize=12, style='italic',
            ha='center', color='#718096')
    
    ax1.set_ylabel('Price ($)', fontsize=14, fontweight='600', color='#2D3748', labelpad=10)
    
    legend = ax1.legend(loc='upper left', fontsize=10.5, framealpha=0.98, 
                       ncol=3, edgecolor='#CBD5E0', fancybox=True,
                       borderpad=1, labelspacing=0.8, columnspacing=1.5)
    legend.get_frame().set_facecolor('#FFFFFF')
    legend.get_frame().set_linewidth(1.5)
    
    ax1.grid(True, alpha=0.15, linestyle='-', linewidth=0.8, color='#CBD5E0')
    ax1.tick_params(labelsize=10.5, colors='#4A5568', width=1.2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#CBD5E0')
    ax1.spines['bottom'].set_color('#CBD5E0')
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['bottom'].set_linewidth(1.5)
    
    # =====================================================================
    # GRÁFICO 2: RSI
    # =====================================================================
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_facecolor('#FFFFFF')
    
    ax2.plot(df_recent.index, df_recent['RSI'], 
             color='#7C3AED', linewidth=3, label='RSI', zorder=3)
    
    ax2.fill_between(df_recent.index, df_recent['RSI'], 50,
                     where=(df_recent['RSI'] >= 50), 
                     color='#10B981', alpha=0.12, zorder=1)
    ax2.fill_between(df_recent.index, df_recent['RSI'], 50,
                     where=(df_recent['RSI'] < 50), 
                     color='#DC143C', alpha=0.12, zorder=1)
    
    ax2.axhline(y=75, color='#DC143C', linestyle='--', linewidth=2, alpha=0.7, zorder=2)
    ax2.axhline(y=70, color='#F59E0B', linestyle=':', linewidth=1.5, alpha=0.6, zorder=2)
    ax2.axhline(y=50, color='#6B7280', linestyle='-', linewidth=1.5, alpha=0.6, zorder=2)
    ax2.axhline(y=30, color='#F59E0B', linestyle=':', linewidth=1.5, alpha=0.6, zorder=2)
    ax2.axhline(y=25, color='#10B981', linestyle='--', linewidth=2, alpha=0.7, zorder=2)
    
    ax2.fill_between(df_recent.index, 75, 100, alpha=0.08, color='#DC143C', zorder=0)
    ax2.fill_between(df_recent.index, 0, 25, alpha=0.08, color='#10B981', zorder=0)
    
    ax2.scatter(current.name, current['RSI'], 
               facecolors='none',
               edgecolors='#1A202C', 
               s=180,
               linewidth=3,
               marker='o',
               zorder=10)
    
    rsi_color = '#DC143C' if current['RSI'] > 70 else '#10B981' if current['RSI'] < 30 else '#6B7280'
    rsi_status = 'Overbought' if current['RSI'] > 70 else 'Oversold' if current['RSI'] < 30 else 'Neutral'
    
    ax2.text(0.02, 0.90, f'RSI: {current["RSI"]:.1f}', 
            transform=ax2.transAxes, 
            fontsize=12, 
            fontweight='bold',
            color='white',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.6', 
                     facecolor=rsi_color, 
                     alpha=0.95, 
                     edgecolor='white', 
                     linewidth=2))
    ax2.text(0.13, 0.90, rsi_status, 
            transform=ax2.transAxes, 
            fontsize=10, 
            style='italic',
            color=rsi_color,
            verticalalignment='top')
    
    ax2.set_ylabel('RSI', fontsize=13, fontweight='600', color='#2D3748', labelpad=10)
    ax2.set_ylim([0, 100])
    ax2.grid(True, alpha=0.15, linestyle='-', linewidth=0.8, color='#CBD5E0')
    ax2.tick_params(labelsize=10, colors='#4A5568', width=1.2)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#CBD5E0')
    ax2.spines['bottom'].set_color('#CBD5E0')
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['bottom'].set_linewidth(1.5)
    
    # =====================================================================
    # GRÁFICO 3: ADX
    # =====================================================================
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.set_facecolor('#FFFFFF')
    
    ax3.plot(df_recent.index, df_recent['ADX'], 
             color='#1A202C', linewidth=3, label='ADX', zorder=3)
    ax3.plot(df_recent.index, df_recent['Plus_DI'], 
             color='#10B981', linewidth=2.5, alpha=0.8, label='+DI', zorder=2)
    ax3.plot(df_recent.index, df_recent['Minus_DI'], 
             color='#DC143C', linewidth=2.5, alpha=0.8, label='-DI', zorder=2)
    
    ax3.axhline(y=25, color='#10B981', linestyle='--', linewidth=2, alpha=0.7)
    ax3.axhline(y=20, color='#F59E0B', linestyle=':', linewidth=1.5, alpha=0.6)
    
    ax3.fill_between(df_recent.index, 0, 20, alpha=0.08, color='#9CA3AF')
    ax3.fill_between(df_recent.index, 25, 100, alpha=0.08, color='#10B981')
    
    ax3.scatter(current.name, current['ADX'], 
               facecolors='none',
               edgecolors='#1A202C', 
               s=180,
               linewidth=3,
               marker='o',
               zorder=10)
    
    adx_color = '#10B981' if current['ADX'] > 25 else '#F59E0B' if current['ADX'] > 20 else '#9CA3AF'
    trend_strength = 'Strong' if current['ADX'] > 25 else 'Moderate' if current['ADX'] > 20 else 'Weak'
    
    ax3.text(0.02, 0.90, f'ADX: {current["ADX"]:.1f}', 
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
    ax3.text(0.13, 0.90, trend_strength, 
            transform=ax3.transAxes, 
            fontsize=10, 
            style='italic',
            color=adx_color,
            verticalalignment='top')
    
    ax3.set_ylabel('ADX / DI', fontsize=13, fontweight='600', color='#2D3748', labelpad=10)
    ax3.set_ylim([0, 70])
    legend = ax3.legend(loc='upper left', fontsize=9.5, framealpha=0.98, 
                       ncol=3, edgecolor='#CBD5E0', fancybox=True)
    legend.get_frame().set_facecolor('#FFFFFF')
    ax3.grid(True, alpha=0.15, linestyle='-', linewidth=0.8, color='#CBD5E0')
    ax3.tick_params(labelsize=10, colors='#4A5568', width=1.2)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_color('#CBD5E0')
    ax3.spines['bottom'].set_color('#CBD5E0')
    ax3.spines['left'].set_linewidth(1.5)
    ax3.spines['bottom'].set_linewidth(1.5)
    
    # =====================================================================
    # GRÁFICO 4: MACD-V MEJORADO (ESTILO THINKORSWIM PROFESIONAL)
    # =====================================================================
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.set_facecolor('#0A0A0A')  # Fondo negro puro
    
    # Verificar que MACD_V_Signal tenga datos válidos
    if 'MACD_V_Signal' in df_recent.columns and not df_recent['MACD_V_Signal'].isna().all():
        macd_signal = df_recent['MACD_V_Signal'].dropna()
        
        if len(macd_signal) > 0:
            # Calcular límites dinámicos
            macd_max = macd_signal.max()
            macd_min = macd_signal.min()
            macd_range = macd_max - macd_min
            
            # Definir límites del gráfico con padding
            y_max = macd_max + (macd_range * 0.2)
            y_min = macd_min - (macd_range * 0.2)
            
            # Líneas de referencia principales (bandas ±50, ±150)
            ax4.axhline(y=150, color='#CC0000', linestyle='--', linewidth=2, alpha=0.9, zorder=2, label='±150')
            ax4.axhline(y=50, color='#FFD700', linestyle='--', linewidth=1.5, alpha=0.7, zorder=2, label='±50')
            ax4.axhline(y=0, color='#FFFFFF', linestyle='-', linewidth=2, alpha=0.5, zorder=2, label='Zero')
            ax4.axhline(y=-50, color='#FFD700', linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)
            ax4.axhline(y=-150, color='#CC0000', linestyle='--', linewidth=2, alpha=0.9, zorder=2)
            
            # Zonas de color (similar a ThinkOrSwim)
            # Zona alcista fuerte (>150)
            ax4.fill_between(df_recent.index, 150, y_max, 
                             color='#00FF00', alpha=0.08, zorder=0)
            
            # Zona alcista moderada (50 a 150)
            ax4.fill_between(df_recent.index, 50, 150, 
                             color='#00CC00', alpha=0.12, zorder=0)
            
            # Zona neutral (-50 a 50)
            ax4.fill_between(df_recent.index, -50, 50, 
                             color='#404040', alpha=0.15, zorder=0)
            
            # Zona bajista moderada (-150 a -50)
            ax4.fill_between(df_recent.index, -150, -50, 
                             color='#CC0000', alpha=0.12, zorder=0)
            
            # Zona bajista fuerte (<-150)
            ax4.fill_between(df_recent.index, y_min, -150, 
                             color='#FF0000', alpha=0.08, zorder=0)
            
            # Línea MACD-V con gradiente de color según dirección
            for i in range(1, len(df_recent)):
                if pd.notna(df_recent['MACD_V_Signal'].iloc[i-1]) and pd.notna(df_recent['MACD_V_Signal'].iloc[i]):
                    x1, x2 = df_recent.index[i-1], df_recent.index[i]
                    y1, y2 = df_recent['MACD_V_Signal'].iloc[i-1], df_recent['MACD_V_Signal'].iloc[i]
                    
                    # Color según dirección Y posición
                    if y2 > y1:  # Subiendo
                        if y2 > 50:
                            color = '#00FF00'  # Verde brillante
                        elif y2 > 0:
                            color = '#7FFF00'  # Verde lima
                        else:
                            color = '#FFFF00'  # Amarillo
                    else:  # Bajando
                        if y2 < -50:
                            color = '#FF0000'  # Rojo brillante
                        elif y2 < 0:
                            color = '#FF6600'  # Naranja
                        else:
                            color = '#FFAA00'  # Amarillo-naranja
                    
                    ax4.plot([x1, x2], [y1, y2], color=color, linewidth=3.5, alpha=0.95, zorder=5)
            
            # Punto actual con doble círculo
            if pd.notna(current['MACD_V_Signal']):
                current_color = '#00FF00' if current['MACD_V_Signal'] > 0 else '#FF0000'
                
                # Círculo exterior (glow)
                ax4.scatter(current.name, current['MACD_V_Signal'], 
                           facecolors=current_color,
                           edgecolors='none', 
                           s=250,
                           alpha=0.3,
                           marker='o',
                           zorder=9)
                
                # Círculo principal
                ax4.scatter(current.name, current['MACD_V_Signal'], 
                           facecolors=current_color,
                           edgecolors='#FFFFFF', 
                           s=150,
                           linewidth=3,
                           marker='o',
                           alpha=0.95,
                           zorder=10)
                
                # Badge informativo mejorado
                if len(df_recent) >= 2 and pd.notna(df_recent['MACD_V_Signal'].iloc[-2]):
                    signal_diff = current['MACD_V_Signal'] - df_recent['MACD_V_Signal'].iloc[-2]
                    signal_direction = '▲' if signal_diff > 0 else '▼'
                    signal_color = '#00FF00' if signal_diff > 0 else '#FF0000'
                    
                    # Badge con valor actual
                    ax4.text(0.02, 0.92, f'MACD-V: {current["MACD_V_Signal"]:.1f}', 
                            transform=ax4.transAxes, 
                            fontsize=13, 
                            fontweight='bold',
                            color='#FFFFFF',
                            verticalalignment='top',
                            bbox=dict(boxstyle='round,pad=0.7', 
                                     facecolor='#1A1A1A', 
                                     alpha=0.95, 
                                     edgecolor=signal_color, 
                                     linewidth=3))
                    
                    # Indicador de dirección
                    ax4.text(0.02, 0.78, f'{signal_direction} {abs(signal_diff):.1f}', 
                            transform=ax4.transAxes, 
                            fontsize=11, 
                            fontweight='bold',
                            color=signal_color,
                            verticalalignment='top')
                    
                    # Estado del momentum
                    if current['MACD_V_Signal'] > 150:
                        momentum_status = 'EXTREMELY BULLISH'
                        status_color = '#00FF00'
                    elif current['MACD_V_Signal'] > 50:
                        momentum_status = 'BULLISH'
                        status_color = '#7FFF00'
                    elif current['MACD_V_Signal'] > -50:
                        momentum_status = 'NEUTRAL'
                        status_color = '#FFFF00'
                    elif current['MACD_V_Signal'] > -150:
                        momentum_status = 'BEARISH'
                        status_color = '#FF6600'
                    else:
                        momentum_status = 'EXTREMELY BEARISH'
                        status_color = '#FF0000'
                    
                    ax4.text(0.98, 0.92, momentum_status, 
                            transform=ax4.transAxes, 
                            fontsize=11, 
                            fontweight='bold',
                            color=status_color,
                            verticalalignment='top',
                            horizontalalignment='right')
            
            # Configurar límites del eje Y
            ax4.set_ylim([y_min, y_max])
            
        else:
            ax4.text(0.5, 0.5, 'Sin datos MACD-V disponibles', 
                    transform=ax4.transAxes, 
                    fontsize=12, 
                    ha='center', 
                    va='center',
                    color='#FFFFFF')
    else:
        ax4.text(0.5, 0.5, 'MACD-V no disponible', 
                transform=ax4.transAxes, 
                fontsize=12, 
                ha='center', 
                va='center',
                color='#FFFFFF')
    
    ax4.set_ylabel('MACD-V', fontsize=13, fontweight='bold', color='#FFFFFF', labelpad=10)
    ax4.grid(True, alpha=0.2, linestyle=':', linewidth=0.5, color='#404040')
    ax4.tick_params(labelsize=10, colors='#CCCCCC', width=1.2)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['left'].set_color('#404040')
    ax4.spines['bottom'].set_color('#404040')
    ax4.spines['left'].set_linewidth(2)
    ax4.spines['bottom'].set_linewidth(2)
    
    # =====================================================================
    # GRÁFICO 5: TIMELINE DE REGÍMENES
    # =====================================================================
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    ax5.set_facecolor('#FFFFFF')
    
    regime_order = ['BAJISTA', 'RANGO', 'ALCISTA', 'RIESGO']
    
    regime_nums = [regime_order.index(r) for r in df_recent['Regime_Name']]
    ax5.plot(df_recent.index, regime_nums, 
            color='#CBD5E0', linewidth=2, alpha=0.5, zorder=1, linestyle='-')
    
    for regime_name in regime_order:
        mask = df_recent['Regime_Name'] == regime_name
        if mask.sum() > 0:
            color = regime_colors[regime_name]
            regime_num = regime_order.index(regime_name)
            
            ax5.scatter(df_recent[mask].index, 
                       [regime_num] * mask.sum(),
                       c=color, 
                       alpha=0.2, 
                       s=160,
                       edgecolors='none',
                       zorder=3)
            ax5.scatter(df_recent[mask].index, 
                       [regime_num] * mask.sum(),
                       c=color, 
                       alpha=0.95, 
                       s=95,
                       edgecolors='white', 
                       linewidth=1.2,
                       zorder=4)
    
    current_regime_num = regime_order.index(current['Regime_Name'])
    ax5.scatter(current.name, current_regime_num, 
               facecolors='none',
               edgecolors='#1A202C', 
               s=250,
               linewidth=3.5,
               marker='o',
               zorder=10)
    
    ax5.set_ylabel('Regime', fontsize=13, fontweight='600', color='#2D3748', labelpad=10)
    ax5.set_yticks(range(4))
    ax5.set_yticklabels(regime_order, fontsize=11, fontweight='600')
    ax5.set_xlabel('Date', fontsize=14, fontweight='600', color='#2D3748', labelpad=10)
    ax5.grid(True, alpha=0.15, linestyle='-', linewidth=0.8, color='#CBD5E0', axis='x')
    ax5.tick_params(labelsize=10, colors='#4A5568', width=1.2)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.spines['left'].set_color('#CBD5E0')
    ax5.spines['bottom'].set_color('#CBD5E0')
    ax5.spines['left'].set_linewidth(1.5)
    ax5.spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    
    return fig

# =========================================================================
# PÁGINA PRINCIPAL
# =========================================================================

def market_regime_page():
    st.title("📊 Market Regime Analyzer (ADX + MACD-V)")
    st.markdown("---")
    st.info("🔍 Análisis de regímenes de mercado usando ADX + RSI + SMAs + MACD-V")
    
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        ticker = st.text_input(
            "🎯 Ticker",
            value="AAPL",
            help="Ingresa el símbolo del ticker (ej: AAPL, MSFT, SPY)"
        ).upper()
        
        st.markdown("---")
        
        lookback_months = st.slider(
            "📅 Meses a mostrar",
            min_value=3,
            max_value=24,
            value=6,
            step=1,
            help="Número de meses de historia a visualizar"
        )
        
        lookback_weeks = int(lookback_months * 4.33)
        
        st.markdown("---")
        
        start_date = st.date_input(
            "📆 Fecha inicio datos",
            value=datetime(2018, 1, 1),
            help="Fecha de inicio para descarga de datos históricos"
        )
        
        st.markdown("---")
        
        analizar_btn = st.button(
            "🚀 Analizar Régimen",
            type="primary",
            use_container_width=True,
            key="sidebar_analyze"
        )
        
        st.markdown("---")
        
        st.markdown("### 📖 Metodología")
        st.markdown("""
        **ADX (Fuerza de Tendencia):**
        - ADX > 25: Tendencia fuerte
        - ADX < 20: Sin tendencia / Rango
        
        **RSI (Overbought/Oversold):**
        - RSI > 75: Overbought (riesgo)
        - RSI < 25: Oversold (riesgo)
        
        **MACD-V (Momentum):**
        - Normalizado por volatilidad (ATR)
        - Verde: momentum alcista
        - Rojo: momentum bajista
        - >150: Extremadamente alcista
        - 50-150: Alcista
        - -50 a 50: Neutral
        - -150 a -50: Bajista
        - <-150: Extremadamente bajista
        
        **Regímenes:**
        - 🟢 **ALCISTA**: ADX>25, +DI>-DI, precio>SMAs
        - 🔴 **BAJISTA**: ADX>25, -DI>+DI, precio<SMAs
        - ⚫ **RANGO**: ADX<20, sin dirección
        - 🟠 **RIESGO**: RSI extremo + ADX alto
        """)
    
    if analizar_btn:
        with st.spinner(f"Descargando y analizando datos para {ticker}..."):
            df, df_recent = analyze_regime(
                ticker,
                start_date.strftime('%Y-%m-%d'),
                lookback_weeks
            )
            
            if df is None or df_recent is None:
                st.error(f"❌ No se pudieron obtener datos para {ticker}")
                st.stop()
            
            st.session_state.last_ticker = ticker
            st.session_state.df = df
            st.session_state.df_recent = df_recent
            st.session_state.lookback_months = lookback_months
    
    if 'df' in st.session_state and 'df_recent' in st.session_state:
        df = st.session_state.df
        df_recent = st.session_state.df_recent
        current = df.iloc[-1]
        
        if 'last_ticker' not in st.session_state or st.session_state.last_ticker != ticker:
            st.warning("⚠️ El ticker ha cambiado. Presiona '🚀 Analizar Régimen' en el sidebar para actualizar.")
        
        st.markdown("---")
        
        st.markdown("### 🎯 Régimen Actual")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            regime_color = {
                'ALCISTA': '🟢',
                'BAJISTA': '🔴',
                'RANGO': '⚫',
                'RIESGO': '🟠'
            }[current['Regime_Name']]
            st.metric(
                "Régimen",
                f"{regime_color} {current['Regime_Name']}"
            )
        
        with col2:
            st.metric("Precio", f"${current['Close']:.2f}")
        
        with col3:
            adx_status = "Fuerte" if current['ADX'] > 25 else "Débil"
            st.metric("ADX", f"{current['ADX']:.1f}", adx_status)
        
        with col4:
            rsi_status = "OB" if current['RSI'] > 70 else "OS" if current['RSI'] < 30 else "Neutral"
            st.metric("RSI", f"{current['RSI']:.1f}", rsi_status)
        
        with col5:
            try:
                if 'MACD_V_Signal' in df.columns and len(df) >= 2:
                    current_macd = current.get('MACD_V_Signal')
                    previous_macd = df['MACD_V_Signal'].iloc[-2]
                    
                    if pd.notna(current_macd) and pd.notna(previous_macd):
                        macd_direction = "↑" if current_macd > previous_macd else "↓"
                        st.metric("MACD-V", f"{current_macd:.1f}", macd_direction)
                    else:
                        st.metric("MACD-V", "N/A", "Sin datos")
                else:
                    st.metric("MACD-V", "N/A", "Insuficientes datos")
            except (KeyError, IndexError, TypeError) as e:
                st.metric("MACD-V", "Error", str(e)[:10])
        
        with col6:
            st.metric("Fecha", current.name.strftime('%Y-%m-%d'))
        
        st.markdown("---")
        st.markdown("### 💡 Recomendación")
        
        if current['Regime_Name'] == 'RIESGO':
            if current['RSI'] > 70:
                st.warning("⚠️ **PRECAUCIÓN**: Zona de sobrecompra - considerar tomar ganancias")
            else:
                st.info("💡 **OPORTUNIDAD**: Zona de sobreventa - posible rebote con confirmación")
        elif current['Regime_Name'] == 'ALCISTA':
            st.success("✅ **MANTENER/COMPRAR**: Tendencia alcista fuerte confirmada")
        elif current['Regime_Name'] == 'BAJISTA':
            st.error("🔴 **EVITAR/VENDER**: Tendencia bajista confirmada - esperar ADX < 20")
        else:
            st.info("⚖️ **ESPERAR**: Sin dirección clara - estrategia de mean reversion")
        
        st.markdown("---")
        lookback_display = st.session_state.get('lookback_months', lookback_months)
        st.markdown(f"### 📈 Análisis Visual ({lookback_display} meses)")
        
        fig = plot_regime_dashboard(df_recent, st.session_state.last_ticker)
        
        if fig is not None:
            st.pyplot(fig)
        else:
            st.error("❌ No se pudo generar el gráfico. Verifica que los datos estén completos.")
        
        st.markdown("---")
        
        # Verificar columnas antes de exportar
        export_cols = ['Close', 'Regime_Name', 'State', 'ADX', 'RSI', 'Plus_DI', 'Minus_DI', 
                      'SMA_20', 'SMA_50']
        
        # Agregar MACD-V solo si existen
        if 'MACD_V' in df.columns:
            export_cols.append('MACD_V')
        if 'MACD_V_Signal' in df.columns:
            export_cols.append('MACD_V_Signal')
        
        # Filtrar solo las columnas que existen
        available_cols = [col for col in export_cols if col in df.columns]
        
        csv = df[available_cols].to_csv()
        st.download_button(
            label="📥 Descargar Datos Completos (CSV)",
            data=csv,
            file_name=f"market_regime_{st.session_state.last_ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.markdown("---")
        st.info("👈 Configura los parámetros en el sidebar y presiona **'🚀 Analizar Régimen'** para comenzar el análisis.")

# =========================================================================
# PUNTO DE ENTRADA PROTEGIDO
# =========================================================================

if __name__ == "__main__":
    if check_password():
        market_regime_page()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Introduce tus credenciales en el menú lateral para acceder.")
