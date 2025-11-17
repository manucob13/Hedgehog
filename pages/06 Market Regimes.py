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
# VISUALIZACIÓN
# =========================================================================

def plot_regime_dashboard(df_recent, ticker):
    """Dashboard consolidado: Precio + RSI + ADX + Secuencia"""
    
    regime_colors = {
        'RIESGO': 'orange',
        'BAJISTA': 'red',
        'RANGO': 'black',
        'ALCISTA': 'green'
    }
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)
    
    # GRÁFICO 1: PRECIO CON REGÍMENES
    ax1 = fig.add_subplot(gs[0])
    
    for regime_name, color in regime_colors.items():
        mask = df_recent['Regime_Name'] == regime_name
        if mask.sum() > 0:
            ax1.scatter(df_recent[mask].index, 
                       df_recent[mask]['Close'],
                       c=color, 
                       label=regime_name,
                       alpha=0.8, s=80, edgecolors='white', linewidth=1,
                       zorder=5)
    
    ax1.plot(df_recent.index, df_recent['Close'], 
             color='gray', alpha=0.4, linewidth=2, zorder=1)
    
    ax1.plot(df_recent.index, df_recent['SMA_20'], 
             color='blue', alpha=0.7, linewidth=2.5, linestyle='--', 
             label='SMA(20)', zorder=2)
    ax1.plot(df_recent.index, df_recent['SMA_50'], 
             color='purple', alpha=0.7, linewidth=2.5, linestyle='--', 
             label='SMA(50)', zorder=2)
    
    # Marcar punto actual
    current = df_recent.iloc[-1]
    ax1.scatter(current.name, current['Close'], 
               color='gold', s=600, marker='*', 
               edgecolors='black', linewidth=3,
               label=f'HOY: {current["Regime_Name"]}', zorder=10)
    
    ax1.annotate(f'{current["Regime_Name"]}\n${current["Close"]:.2f}',
                xy=(current.name, current['Close']),
                xytext=(10, 30), textcoords='offset points',
                fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.8', 
                         facecolor=regime_colors[current['Regime_Name']], 
                         alpha=0.9, edgecolor='black', linewidth=2),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                zorder=11)
    
    ax1.set_title(f'{ticker} - Régimen de Mercado\nTimeframe: SEMANAL', 
                  fontsize=18, fontweight='bold', pad=20)
    ax1.set_ylabel('Precio ($)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.95, 
              ncol=3, edgecolor='black', shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(labelsize=11)
    
    # GRÁFICO 2: RSI
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    ax2.plot(df_recent.index, df_recent['RSI'], 
             color='purple', linewidth=3, label='RSI')
    ax2.fill_between(df_recent.index, df_recent['RSI'], 50,
                     where=(df_recent['RSI'] >= 50), 
                     color='green', alpha=0.2)
    ax2.fill_between(df_recent.index, df_recent['RSI'], 50,
                     where=(df_recent['RSI'] < 50), 
                     color='red', alpha=0.2)
    
    ax2.axhline(y=75, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.axhline(y=70, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y=50, color='gray', linestyle='-', linewidth=1.5, alpha=0.6)
    ax2.axhline(y=30, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y=25, color='green', linestyle='--', linewidth=2, alpha=0.7)
    
    ax2.fill_between(df_recent.index, 75, 100, alpha=0.15, color='red')
    ax2.fill_between(df_recent.index, 0, 25, alpha=0.15, color='green')
    
    ax2.scatter(current.name, current['RSI'], 
               color='gold', s=300, marker='*', 
               edgecolors='black', linewidth=2, zorder=10)
    
    ax2.text(0.02, 0.95, f'RSI Actual: {current["RSI"]:.1f}', 
            transform=ax2.transAxes, fontsize=12, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                     edgecolor='black', linewidth=1.5))
    
    ax2.set_ylabel('RSI', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, 100])
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(labelsize=10)
    
    # GRÁFICO 3: ADX
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    
    ax3.plot(df_recent.index, df_recent['ADX'], 
             color='black', linewidth=3, label='ADX')
    ax3.plot(df_recent.index, df_recent['Plus_DI'], 
             color='green', linewidth=2, alpha=0.7, label='+DI')
    ax3.plot(df_recent.index, df_recent['Minus_DI'], 
             color='red', linewidth=2, alpha=0.7, label='-DI')
    
    ax3.axhline(y=25, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax3.axhline(y=20, color='orange', linestyle='--', linewidth=1.5, alpha=0.6)
    
    ax3.fill_between(df_recent.index, 0, 20, alpha=0.15, color='gray')
    ax3.fill_between(df_recent.index, 25, 100, alpha=0.15, color='green')
    
    ax3.scatter(current.name, current['ADX'], 
               color='gold', s=300, marker='*', 
               edgecolors='black', linewidth=2, zorder=10)
    
    ax3.text(0.02, 0.95, f'ADX Actual: {current["ADX"]:.1f}', 
            transform=ax3.transAxes, fontsize=12, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                     edgecolor='black', linewidth=1.5))
    
    ax3.set_ylabel('ADX', fontsize=13, fontweight='bold')
    ax3.set_ylim([0, 70])
    ax3.legend(loc='upper left', fontsize=9, framealpha=0.9, ncol=3)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.tick_params(labelsize=10)
    
    # GRÁFICO 4: SECUENCIA DE REGÍMENES
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    
    regime_order = ['BAJISTA', 'RANGO', 'ALCISTA', 'RIESGO']
    
    for regime_name in regime_order:
        mask = df_recent['Regime_Name'] == regime_name
        if mask.sum() > 0:
            color = regime_colors[regime_name]
            regime_num = regime_order.index(regime_name)
            
            ax4.scatter(df_recent[mask].index, 
                       [regime_num] * mask.sum(),
                       c=color, 
                       alpha=0.8, s=100, edgecolors='white', linewidth=1)
    
    current_regime_num = regime_order.index(current['Regime_Name'])
    ax4.scatter(current.name, current_regime_num, 
               color='gold', s=600, marker='*', 
               edgecolors='black', linewidth=3, zorder=10)
    
    ax4.set_ylabel('Régimen', fontsize=13, fontweight='bold')
    ax4.set_yticks(range(4))
    ax4.set_yticklabels(regime_order, fontsize=11, fontweight='bold')
    ax4.set_xlabel('Fecha', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax4.tick_params(labelsize=10)
    
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
