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
# VISUALIZACIÓN MEJORADA
# =========================================================================

def plot_regime_dashboard(df_recent, ticker):
    """Dashboard consolidado: Precio + RSI + ADX + Secuencia - VERSIÓN MEJORADA"""
    
    # Colores modernos y más suaves
    regime_colors = {
        'RIESGO': '#FF8C00',    # Naranja más vibrante
        'BAJISTA': '#E74C3C',   # Rojo moderno
        'RANGO': '#34495E',     # Gris azulado
        'ALCISTA': '#27AE60'    # Verde moderno
    }
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)
    
    # =====================================================================
    # GRÁFICO 1: PRECIO CON REGÍMENES
    # =====================================================================
    ax1 = fig.add_subplot(gs[0])
    
    # Línea de precio de fondo
    ax1.plot(df_recent.index, df_recent['Close'], 
             color='#95A5A6', alpha=0.3, linewidth=2, zorder=1)
    
    # Puntos de régimen (tamaño reducido)
    for regime_name, color in regime_colors.items():
        mask = df_recent['Regime_Name'] == regime_name
        if mask.sum() > 0:
            ax1.scatter(df_recent[mask].index, 
                       df_recent[mask]['Close'],
                       c=color, 
                       label=regime_name,
                       alpha=0.85, 
                       s=50,  # Reducido de 80 a 50
                       edgecolors='white', 
                       linewidth=0.8,
                       zorder=5)
    
    # SMAs con estilo más limpio
    ax1.plot(df_recent.index, df_recent['SMA_20'], 
             color='#3498DB', alpha=0.8, linewidth=2, linestyle='--', 
             label='SMA(20)', zorder=2)
    ax1.plot(df_recent.index, df_recent['SMA_50'], 
             color='#9B59B6', alpha=0.8, linewidth=2, linestyle='--', 
             label='SMA(50)', zorder=2)
    
    # Punto actual (estrella más pequeña)
    current = df_recent.iloc[-1]
    ax1.scatter(current.name, current['Close'], 
               color='#FFD700', 
               s=250,  # Reducido de 600 a 250
               marker='*', 
               edgecolors='black', 
               linewidth=2,
               label=f'Actual: {current["Regime_Name"]}', 
               zorder=10)
    
    # Anotación simplificada
    ax1.annotate(f'{current["Regime_Name"]}\n${current["Close"]:.2f}',
                xy=(current.name, current['Close']),
                xytext=(15, 25), 
                textcoords='offset points',
                fontsize=11,  # Reducido de 14 a 11
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.6', 
                         facecolor=regime_colors[current['Regime_Name']], 
                         alpha=0.85, 
                         edgecolor='black', 
                         linewidth=1.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'),
                zorder=11)
    
    # Título limpio
    ax1.set_title(f'{ticker} - Market Regime Analysis (Weekly)', 
                  fontsize=16, fontweight='bold', pad=15, color='#2C3E50')
    ax1.set_ylabel('Price ($)', fontsize=13, fontweight='bold', color='#2C3E50')
    
    # Leyenda mejorada
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.95, 
              ncol=3, edgecolor='#BDC3C7', shadow=False,
              borderpad=0.8, labelspacing=0.5)
    
    ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax1.tick_params(labelsize=10, colors='#34495E')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # =====================================================================
    # GRÁFICO 2: RSI
    # =====================================================================
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    # Línea RSI principal
    ax2.plot(df_recent.index, df_recent['RSI'], 
             color='#8E44AD', linewidth=2.5, label='RSI')
    
    # Zonas de color
    ax2.fill_between(df_recent.index, df_recent['RSI'], 50,
                     where=(df_recent['RSI'] >= 50), 
                     color='#27AE60', alpha=0.15)
    ax2.fill_between(df_recent.index, df_recent['RSI'], 50,
                     where=(df_recent['RSI'] < 50), 
                     color='#E74C3C', alpha=0.15)
    
    # Líneas de referencia
    ax2.axhline(y=75, color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.6)
    ax2.axhline(y=70, color='#E67E22', linestyle=':', linewidth=1, alpha=0.5)
    ax2.axhline(y=50, color='#7F8C8D', linestyle='-', linewidth=1, alpha=0.5)
    ax2.axhline(y=30, color='#E67E22', linestyle=':', linewidth=1, alpha=0.5)
    ax2.axhline(y=25, color='#27AE60', linestyle='--', linewidth=1.5, alpha=0.6)
    
    # Zonas extremas
    ax2.fill_between(df_recent.index, 75, 100, alpha=0.1, color='#E74C3C')
    ax2.fill_between(df_recent.index, 0, 25, alpha=0.1, color='#27AE60')
    
    # Punto actual (estrella pequeña)
    ax2.scatter(current.name, current['RSI'], 
               color='#FFD700', 
               s=150,  # Reducido de 300 a 150
               marker='*', 
               edgecolors='black', 
               linewidth=1.5, 
               zorder=10)
    
    # Texto informativo
    rsi_color = '#E74C3C' if current['RSI'] > 70 else '#27AE60' if current['RSI'] < 30 else '#7F8C8D'
    ax2.text(0.02, 0.92, f'RSI: {current["RSI"]:.1f}', 
            transform=ax2.transAxes, 
            fontsize=11, 
            fontweight='bold',
            color=rsi_color,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', 
                     facecolor='white', 
                     alpha=0.9, 
                     edgecolor=rsi_color, 
                     linewidth=2))
    
    ax2.set_ylabel('RSI', fontsize=12, fontweight='bold', color='#2C3E50')
    ax2.set_ylim([0, 100])
    ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax2.tick_params(labelsize=9, colors='#34495E')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # =====================================================================
    # GRÁFICO 3: ADX
    # =====================================================================
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    
    # Líneas ADX y DI
    ax3.plot(df_recent.index, df_recent['ADX'], 
             color='#2C3E50', linewidth=2.5, label='ADX')
    ax3.plot(df_recent.index, df_recent['Plus_DI'], 
             color='#27AE60', linewidth=2, alpha=0.7, label='+DI')
    ax3.plot(df_recent.index, df_recent['Minus_DI'], 
             color='#E74C3C', linewidth=2, alpha=0.7, label='-DI')
    
    # Líneas de referencia
    ax3.axhline(y=25, color='#27AE60', linestyle='--', linewidth=1.5, alpha=0.6)
    ax3.axhline(y=20, color='#E67E22', linestyle=':', linewidth=1, alpha=0.5)
    
    # Zonas
    ax3.fill_between(df_recent.index, 0, 20, alpha=0.1, color='#95A5A6')
    ax3.fill_between(df_recent.index, 25, 100, alpha=0.1, color='#27AE60')
    
    # Punto actual
    ax3.scatter(current.name, current['ADX'], 
               color='#FFD700', 
               s=150,  # Reducido de 300 a 150
               marker='*', 
               edgecolors='black', 
               linewidth=1.5, 
               zorder=10)
    
    # Texto informativo
    adx_color = '#27AE60' if current['ADX'] > 25 else '#E67E22' if current['ADX'] > 20 else '#95A5A6'
    trend_strength = 'Fuerte' if current['ADX'] > 25 else 'Moderada' if current['ADX'] > 20 else 'Débil'
    
    ax3.text(0.02, 0.92, f'ADX: {current["ADX"]:.1f} ({trend_strength})', 
            transform=ax3.transAxes, 
            fontsize=11, 
            fontweight='bold',
            color=adx_color,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', 
                     facecolor='white', 
                     alpha=0.9, 
                     edgecolor=adx_color, 
                     linewidth=2))
    
    ax3.set_ylabel('ADX / DI', fontsize=12, fontweight='bold', color='#2C3E50')
    ax3.set_ylim([0, 70])
    ax3.legend(loc='upper left', fontsize=9, framealpha=0.9, 
              ncol=3, edgecolor='#BDC3C7')
    ax3.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax3.tick_params(labelsize=9, colors='#34495E')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # =====================================================================
    # GRÁFICO 4: SECUENCIA DE REGÍMENES
    # =====================================================================
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    
    regime_order = ['BAJISTA', 'RANGO', 'ALCISTA', 'RIESGO']
    
    # Puntos de régimen
    for regime_name in regime_order:
        mask = df_recent['Regime_Name'] == regime_name
        if mask.sum() > 0:
            color = regime_colors[regime_name]
            regime_num = regime_order.index(regime_name)
            
            ax4.scatter(df_recent[mask].index, 
                       [regime_num] * mask.sum(),
                       c=color, 
                       alpha=0.85, 
                       s=80,  # Reducido de 100 a 80
                       edgecolors='white', 
                       linewidth=0.8)
    
    # Punto actual
    current_regime_num = regime_order.index(current['Regime_Name'])
    ax4.scatter(current.name, current_regime_num, 
               color='#FFD700', 
               s=250,  # Reducido de 600 a 250
               marker='*', 
               edgecolors='black', 
               linewidth=2, 
               zorder=10)
    
    ax4.set_ylabel('Regime', fontsize=12, fontweight='bold', color='#2C3E50')
    ax4.set_yticks(range(4))
    ax4.set_yticklabels(regime_order, fontsize=10, fontweight='bold')
    ax4.set_xlabel('Date', fontsize=13, fontweight='bold', color='#2C3E50')
    ax4.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='x')
    ax4.tick_params(labelsize=9, colors='#34495E')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # Fondo blanco limpio
    fig.patch.set_facecolor('white')
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor('#FAFAFA')
    
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
    
    # Botón de análisis
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        analizar_btn = st.button(
            "🚀 Analizar Régimen",
            type="primary",
            use_container_width=True
        )
    
    if analizar_btn or 'last_ticker' in st.session_state and st.session_state.last_ticker == ticker:
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
    
    # Mostrar resultados si existen
    if 'df' in st.session_state and 'df_recent' in st.session_state:
        df = st.session_state.df
        df_recent = st.session_state.df_recent
        current = df.iloc[-1]
        
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
        st.markdown(f"### 📈 Análisis Visual ({lookback_months} meses)")
        
        fig = plot_regime_dashboard(df_recent, ticker)
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
            file_name=f"market_regime_{ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# =========================================================================
# PUNTO DE ENTRADA PROTEGIDO
# =========================================================================

if __name__ == "__main__":
    if check_password():
        market_regime_page()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Introduce tus credenciales en el menú lateral para acceder.")
