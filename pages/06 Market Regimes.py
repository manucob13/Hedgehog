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
# VISUALIZACIÓN ULTRA MEJORADA
# =========================================================================

def plot_regime_dashboard(df_recent, ticker):
    """Dashboard consolidado con diseño moderno y atractivo"""
    
    # Paleta de colores moderna con gradientes
    regime_colors = {
        'RIESGO': '#FF6B35',    # Naranja coral
        'BAJISTA': '#DC143C',   # Crimson
        'RANGO': '#4A5568',     # Gris pizarra
        'ALCISTA': '#10B981'    # Verde esmeralda
    }
    
    # Configuración de estilo moderno
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(22, 13), facecolor='#F8F9FA')
    gs = fig.add_gridspec(4, 1, height_ratios=[3.5, 1, 1, 1], hspace=0.35)
    
    # =====================================================================
    # GRÁFICO 1: PRECIO CON REGÍMENES - DISEÑO PREMIUM
    # =====================================================================
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#FFFFFF')
    
    # Área de relleno bajo el precio para dar profundidad
    ax1.fill_between(df_recent.index, df_recent['Close'].min() * 0.98, 
                     df_recent['Close'], color='#E8F4F8', alpha=0.3, zorder=0)
    
    # Línea de precio principal con efecto de sombra
    ax1.plot(df_recent.index, df_recent['Close'], 
             color='#A0AEC0', alpha=0.4, linewidth=3, zorder=1)
    ax1.plot(df_recent.index, df_recent['Close'], 
             color='#2D3748', alpha=0.6, linewidth=1.5, zorder=2)
    
    # Puntos de régimen con efecto de halo
    for regime_name, color in regime_colors.items():
        mask = df_recent['Regime_Name'] == regime_name
        if mask.sum() > 0:
            # Halo exterior
            ax1.scatter(df_recent[mask].index, 
                       df_recent[mask]['Close'],
                       c=color, 
                       alpha=0.2, 
                       s=120,
                       edgecolors='none',
                       zorder=4)
            # Punto principal
            ax1.scatter(df_recent[mask].index, 
                       df_recent[mask]['Close'],
                       c=color, 
                       label=regime_name,
                       alpha=0.9, 
                       s=65,
                       edgecolors='white', 
                       linewidth=1.2,
                       zorder=5)
    
    # SMAs con estilo degradado
    ax1.plot(df_recent.index, df_recent['SMA_20'], 
             color='#3B82F6', alpha=0.85, linewidth=2.5, linestyle='-', 
             label='SMA(20)', zorder=3)
    ax1.plot(df_recent.index, df_recent['SMA_50'], 
             color='#8B5CF6', alpha=0.85, linewidth=2.5, linestyle='-', 
             label='SMA(50)', zorder=3)
    
    # Punto actual con círculo doble
    current = df_recent.iloc[-1]
    # Círculo exterior (halo)
    ax1.scatter(current.name, current['Close'], 
               facecolors='none',
               edgecolors=regime_colors[current['Regime_Name']], 
               s=350,
               linewidth=4,
               alpha=0.4,
               marker='o', 
               zorder=9)
    # Círculo interior
    ax1.scatter(current.name, current['Close'], 
               facecolors='none',
               edgecolors='#1A202C', 
               s=220,
               linewidth=3,
               marker='o', 
               label=f'📍 Actual: {current["Regime_Name"]}', 
               zorder=10)
    
    # Anotación con estilo moderno
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
    
    # Título con diseño moderno
    ax1.text(0.5, 1.08, f'{ticker}', 
            transform=ax1.transAxes,
            fontsize=24, fontweight='bold', 
            ha='center', color='#1A202C')
    ax1.text(0.5, 1.03, 'Market Regime Analysis • Weekly Timeframe', 
            transform=ax1.transAxes,
            fontsize=12, style='italic',
            ha='center', color='#718096')
    
    ax1.set_ylabel('Price ($)', fontsize=14, fontweight='600', color='#2D3748', labelpad=10)
    
    # Leyenda moderna
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
    # GRÁFICO 2: RSI CON GRADIENTES
    # =====================================================================
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_facecolor('#FFFFFF')
    
    # Línea RSI con gradiente
    line = ax2.plot(df_recent.index, df_recent['RSI'], 
                    color='#7C3AED', linewidth=3, label='RSI', zorder=3)[0]
    
    # Zonas con gradiente
    ax2.fill_between(df_recent.index, df_recent['RSI'], 50,
                     where=(df_recent['RSI'] >= 50), 
                     color='#10B981', alpha=0.12, zorder=1)
    ax2.fill_between(df_recent.index, df_recent['RSI'], 50,
                     where=(df_recent['RSI'] < 50), 
                     color='#DC143C', alpha=0.12, zorder=1)
    
    # Líneas de referencia con estilo
    ax2.axhline(y=75, color='#DC143C', linestyle='--', linewidth=2, alpha=0.7, zorder=2)
    ax2.axhline(y=70, color='#F59E0B', linestyle=':', linewidth=1.5, alpha=0.6, zorder=2)
    ax2.axhline(y=50, color='#6B7280', linestyle='-', linewidth=1.5, alpha=0.6, zorder=2)
    ax2.axhline(y=30, color='#F59E0B', linestyle=':', linewidth=1.5, alpha=0.6, zorder=2)
    ax2.axhline(y=25, color='#10B981', linestyle='--', linewidth=2, alpha=0.7, zorder=2)
    
    # Zonas extremas
    ax2.fill_between(df_recent.index, 75, 100, alpha=0.08, color='#DC143C', zorder=0)
    ax2.fill_between(df_recent.index, 0, 25, alpha=0.08, color='#10B981', zorder=0)
    
    # Punto actual
    ax2.scatter(current.name, current['RSI'], 
               facecolors='none',
               edgecolors='#1A202C', 
               s=180,
               linewidth=3,
               marker='o',
               zorder=10)
    
    # Badge informativo
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
    # GRÁFICO 3: ADX CON DISEÑO MEJORADO
    # =====================================================================
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.set_facecolor('#FFFFFF')
    
    # Líneas con estilo moderno
    ax3.plot(df_recent.index, df_recent['ADX'], 
             color='#1A202C', linewidth=3, label='ADX', zorder=3)
    ax3.plot(df_recent.index, df_recent['Plus_DI'], 
             color='#10B981', linewidth=2.5, alpha=0.8, label='+DI', zorder=2)
    ax3.plot(df_recent.index, df_recent['Minus_DI'], 
             color='#DC143C', linewidth=2.5, alpha=0.8, label='-DI', zorder=2)
    
    # Líneas de referencia
    ax3.axhline(y=25, color='#10B981', linestyle='--', linewidth=2, alpha=0.7)
    ax3.axhline(y=20, color='#F59E0B', linestyle=':', linewidth=1.5, alpha=0.6)
    
    # Zonas
    ax3.fill_between(df_recent.index, 0, 20, alpha=0.08, color='#9CA3AF')
    ax3.fill_between(df_recent.index, 25, 100, alpha=0.08, color='#10B981')
    
    # Punto actual
    ax3.scatter(current.name, current['ADX'], 
               facecolors='none',
               edgecolors='#1A202C', 
               s=180,
               linewidth=3,
               marker='o',
               zorder=10)
    
    # Badge informativo
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
    # GRÁFICO 4: TIMELINE DE REGÍMENES
    # =====================================================================
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.set_facecolor('#FFFFFF')
    
    regime_order = ['BAJISTA', 'RANGO', 'ALCISTA', 'RIESGO']
    
    # Línea conectora entre regímenes
    regime_nums = [regime_order.index(r) for r in df_recent['Regime_Name']]
    ax4.plot(df_recent.index, regime_nums, 
            color='#CBD5E0', linewidth=2, alpha=0.5, zorder=1, linestyle='-')
    
    # Puntos de régimen con halo
    for regime_name in regime_order:
        mask = df_recent['Regime_Name'] == regime_name
        if mask.sum() > 0:
            color = regime_colors[regime_name]
            regime_num = regime_order.index(regime_name)
            
            # Halo
            ax4.scatter(df_recent[mask].index, 
                       [regime_num] * mask.sum(),
                       c=color, 
                       alpha=0.2, 
                       s=160,
                       edgecolors='none',
                       zorder=3)
            # Punto principal
            ax4.scatter(df_recent[mask].index, 
                       [regime_num] * mask.sum(),
                       c=color, 
                       alpha=0.95, 
                       s=95,
                       edgecolors='white', 
                       linewidth=1.2,
                       zorder=4)
    
    # Punto actual
    current_regime_num = regime_order.index(current['Regime_Name'])
    ax4.scatter(current.name, current_regime_num, 
               facecolors='none',
               edgecolors='#1A202C', 
               s=250,
               linewidth=3.5,
               marker='o',
               zorder=10)
    
    ax4.set_ylabel('Regime', fontsize=13, fontweight='600', color='#2D3748', labelpad=10)
    ax4.set_yticks(range(4))
    ax4.set_yticklabels(regime_order, fontsize=11, fontweight='600')
    ax4.set_xlabel('Date', fontsize=14, fontweight='600', color='#2D3748', labelpad=10)
    ax4.grid(True, alpha=0.15, linestyle='-', linewidth=0.8, color='#CBD5E0', axis='x')
    ax4.tick_params(labelsize=10, colors='#4A5568', width=1.2)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['left'].set_color('#CBD5E0')
    ax4.spines['bottom'].set_color('#CBD5E0')
    ax4.spines['left'].set_linewidth(1.5)
    ax4.spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    
    return fig

# =========================================================================
# PÁGINA PRINCIPAL
# =========================================================================

def market_regime_page():
    st.title("📊 Market Regime Analyzer (ADX Method)")
    st.markdown("---")
    st.info("🔍 Análisis de regímenes de mercado usando ADX + RSI + SMAs (método académico)")
    
    # Sidebar para configuración
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
        
        lookback_weeks = int(lookback_months * 4.33)  # Aproximadamente 4.33 semanas por mes
        
        st.markdown("---")
        
        start_date = st.date_input(
            "📆 Fecha inicio datos",
            value=datetime(2018, 1, 1),
            help="Fecha de inicio para descarga de datos históricos"
        )
        
        st.markdown("---")
        
        # BOTÓN DE ANÁLISIS EN EL SIDEBAR
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
        
        **Regímenes:**
        - 🟢 **ALCISTA**: ADX>25, +DI>-DI, precio>SMAs
        - 🔴 **BAJISTA**: ADX>25, -DI>+DI, precio<SMAs
        - ⚫ **RANGO**: ADX<20, sin dirección
        - 🟠 **RIESGO**: RSI extremo + ADX alto
        """)
    
    # Ejecutar análisis cuando se presiona el botón
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
    
    # Mostrar resultados si existen
    if 'df' in st.session_state and 'df_recent' in st.session_state:
        df = st.session_state.df
        df_recent = st.session_state.df_recent
        current = df.iloc[-1]
        
        # Verificar si el ticker cambió
        if 'last_ticker' not in st.session_state or st.session_state.last_ticker != ticker:
            st.warning("⚠️ El ticker ha cambiado. Presiona '🚀 Analizar Régimen' en el sidebar para actualizar.")
        
        st.markdown("---")
        
        # Métricas del régimen actual
        st.markdown("### 🎯 Régimen Actual")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
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
            st.metric("Fecha", current.name.strftime('%Y-%m-%d'))
        
        # Recomendación
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
            st.info("⚖️ **ESPERAR**: Sin dirección clara - estrategia de mean reversion (win rate: 80-85%)")
        
        # Gráfico
        st.markdown("---")
        lookback_display = st.session_state.get('lookback_months', lookback_months)
        st.markdown(f"### 📈 Análisis Visual ({lookback_display} meses)")
        
        fig = plot_regime_dashboard(df_recent, st.session_state.last_ticker)
        st.pyplot(fig)
        
        # Tabla de datos recientes
        st.markdown("---")
        st.markdown("### 📊 Datos Recientes")
        
        df_table = df[['Close', 'Regime_Name', 'ADX', 'RSI', 'Plus_DI', 'Minus_DI']].tail(10).copy()
        df_table = df_table.round(2)
        df_table.index = df_table.index.strftime('%Y-%m-%d')
        
        st.dataframe(
            df_table,
            use_container_width=True,
            column_config={
                "Close": st.column_config.NumberColumn("💲 Precio", format="%.2f"),
                "Regime_Name": st.column_config.TextColumn("🎯 Régimen"),
                "ADX": st.column_config.NumberColumn("📊 ADX", format="%.1f"),
                "RSI": st.column_config.NumberColumn("📈 RSI", format="%.1f"),
                "Plus_DI": st.column_config.NumberColumn("➕ +DI", format="%.1f"),
                "Minus_DI": st.column_config.NumberColumn("➖ -DI", format="%.1f"),
            }
        )
        
        # Descarga de datos
        st.markdown("---")
        csv = df[['Close', 'Regime_Name', 'State', 'ADX', 'RSI', 'Plus_DI', 'Minus_DI', 'SMA_20', 'SMA_50']].to_csv()
        st.download_button(
            label="📥 Descargar Datos Completos (CSV)",
            data=csv,
            file_name=f"market_regime_{st.session_state.last_ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        # Mensaje inicial cuando no hay datos
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
