import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Configuración de página
st.set_page_config(
    page_title="ATR Trend Analyzer",
    page_icon="📊",
    layout="wide"
)

# ============= FUNCIONES DE AUTENTICACIÓN =============
def check_password():
    """Verifica credenciales usando secrets de Streamlit"""
    def password_entered():
        if (st.session_state["username"] == st.secrets["credentials"]["username"] and
            st.session_state["password"] == st.secrets["credentials"]["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.markdown("""
        <div style='text-align: center; padding: 40px;'>
            <h1 style='color: #4ECDC4;'>🔐 ATR Trend Analyzer</h1>
            <p style='color: #8E93A1; font-size: 18px;'>Acceso Restringido</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input("Usuario", key="username")
            st.text_input("Contraseña", type="password", key="password")
            st.button("Iniciar Sesión", on_click=password_entered, use_container_width=True)
        return False
    
    elif not st.session_state["password_correct"]:
        st.error("❌ Usuario o contraseña incorrectos")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input("Usuario", key="username")
            st.text_input("Contraseña", type="password", key="password")
            st.button("Iniciar Sesión", on_click=password_entered, use_container_width=True)
        return False
    else:
        return True

# ============= FUNCIONES TÉCNICAS MEJORADAS =============
def nadaraya_watson_kernel(x, y, bandwidth=8):
    """
    Suavizado Nadaraya-Watson con kernel Gaussiano
    Mejora: Validación de inputs y manejo de edge cases
    """
    if len(x) < bandwidth:
        return y.copy()  # Si hay pocos datos, devolver original
    
    est_y = np.zeros_like(y)
    for i, xi in enumerate(x):
        weights = np.exp(-0.5 * ((x - xi) / bandwidth)**2)
        weights_sum = np.sum(weights)
        if weights_sum > 0:
            est_y[i] = np.sum(weights * y) / weights_sum
        else:
            est_y[i] = y[i]
    return est_y

def calculate_atr_improved(data, period=14):
    """
    ATR mejorado usando pandas para mayor eficiencia
    """
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr.values

def calculate_adaptive_bandwidth(volatility, base_bandwidth=8):
    """
    MEJORA: Bandwidth adaptativo según volatilidad reciente
    Alta volatilidad -> Mayor suavizado
    Baja volatilidad -> Menor suavizado (más sensible)
    """
    recent_vol = volatility[-20:].mean() if len(volatility) >= 20 else volatility.mean()
    median_vol = np.median(volatility[volatility > 0])
    
    if median_vol > 0:
        vol_ratio = recent_vol / median_vol
        # Si volatilidad es alta, incrementar bandwidth
        adaptive_bw = base_bandwidth * (0.7 + 0.6 * vol_ratio)
        return np.clip(adaptive_bw, base_bandwidth * 0.5, base_bandwidth * 2)
    return base_bandwidth

def calculate_dynamic_slope_threshold(trend, lookback=20):
    """
    MEJORA: Umbral de pendiente adaptativo
    Basado en el movimiento histórico reciente
    """
    if len(trend) < lookback:
        return trend * 0.002
    
    recent_changes = np.abs(np.diff(trend[-lookback:]))
    threshold = np.percentile(recent_changes, 40)  # 40th percentile
    return np.maximum(threshold, trend * 0.0015)  # Mínimo 0.15%

def classify_trend_state(prices, trend, atr, atr_multiplier=1.5, 
                         use_adaptive_bandwidth=True, 
                         use_dynamic_threshold=True):
    """
    Sistema de clasificación de tendencia de 4 estados MEJORADO
    
    Estados:
    - 2: RIESGO (Naranja) - Precio fuera de zona ATR
    - 1: ALCISTA (Verde) - Tendencia alcista clara
    - -1: BAJISTA (Rojo) - Tendencia bajista clara
    - 0: LATERAL (Gris) - Sin dirección clara
    
    Mejoras implementadas:
    - Bandwidth adaptativo según volatilidad
    - Umbral de pendiente dinámico
    - Confirmación de cambios de estado (reduce whipsaws)
    """
    x = np.arange(len(prices))
    
    # Bandwidth adaptativo si está habilitado
    if use_adaptive_bandwidth:
        bandwidth = calculate_adaptive_bandwidth(atr)
        trend_recalc = nadaraya_watson_kernel(x, prices, bandwidth=bandwidth)
    else:
        trend_recalc = trend
    
    # Zonas de riesgo basadas en ATR
    upper_risk = trend_recalc + (atr * atr_multiplier)
    lower_risk = trend_recalc - (atr * atr_multiplier)
    
    # Cálculo de pendientes
    slopes = np.diff(trend_recalc, prepend=trend_recalc[0])
    
    # Umbral dinámico si está habilitado
    if use_dynamic_threshold:
        slope_thresholds = np.array([calculate_dynamic_slope_threshold(trend_recalc, 20) 
                                     for _ in range(len(trend_recalc))])
    else:
        slope_thresholds = trend_recalc * 0.002
    
    states = np.zeros(len(prices), dtype=int)
    
    for i in range(len(prices)):
        # PRIORIDAD 1: RIESGO - Precio fuera de zona ATR
        if prices[i] > upper_risk[i] or prices[i] < lower_risk[i]:
            # MEJORA: Confirmación adicional con momentum
            if i > 0:
                price_change = (prices[i] - prices[i-1]) / prices[i-1]
                if abs(price_change) > 0.03:  # Cambio > 3% confirma riesgo
                    states[i] = 2
                else:
                    # Si no hay momentum extremo, revisar tendencia
                    if slopes[i] > slope_thresholds[i]:
                        states[i] = 1
                    elif slopes[i] < -slope_thresholds[i]:
                        states[i] = -1
                    else:
                        states[i] = 0
            else:
                states[i] = 2
        
        # PRIORIDAD 2: Dirección de tendencia
        elif slopes[i] > slope_thresholds[i]:
            states[i] = 1  # ALCISTA
        elif slopes[i] < -slope_thresholds[i]:
            states[i] = -1  # BAJISTA
        else:
            states[i] = 0  # LATERAL
    
    # MEJORA: Filtro de confirmación (reduce cambios espurios)
    # Un estado debe mantenerse al menos 2 periodos para ser válido
    filtered_states = states.copy()
    for i in range(1, len(states) - 1):
        if states[i] != states[i-1] and states[i] != states[i+1]:
            filtered_states[i] = states[i-1]  # Mantener estado anterior
    
    return filtered_states, trend_recalc, upper_risk, lower_risk

# ============= DESCARGA Y PROCESAMIENTO DE DATOS =============
@st.cache_data(ttl=3600)
def download_and_process_data(ticker, period="2y", interval="1wk",
                               atr_period=14, atr_multiplier=1.5,
                               use_adaptive=True, use_dynamic=True):
    """
    Descarga datos y calcula todos los indicadores
    """
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if data.empty or len(data) < 30:
            return None, "Datos insuficientes para el análisis"
        
        # Asegurar estructura correcta
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        prices = data['Close'].values
        x = np.arange(len(prices))
        
        # Cálculo de tendencia kernel
        trend = nadaraya_watson_kernel(x, prices, bandwidth=8)
        
        # Cálculo de ATR mejorado
        atr = calculate_atr_improved(data, period=atr_period)
        
        # Clasificación de estados
        states, trend_refined, upper_risk, lower_risk = classify_trend_state(
            prices, trend, atr, 
            atr_multiplier=atr_multiplier,
            use_adaptive_bandwidth=use_adaptive,
            use_dynamic_threshold=use_dynamic
        )
        
        # Crear DataFrame de resultados
        results = pd.DataFrame({
            'Date': data.index,
            'Close': prices,
            'Trend': trend_refined,
            'ATR': atr,
            'Upper_Risk': upper_risk,
            'Lower_Risk': lower_risk,
            'State': states
        })
        
        # Mapeo de estados a nombres
        state_names = {-1: 'BAJISTA', 0: 'LATERAL', 1: 'ALCISTA', 2: 'RIESGO'}
        results['State_Name'] = results['State'].map(state_names)
        
        return results, None
        
    except Exception as e:
        return None, f"Error al descargar datos: {str(e)}"

# ============= VISUALIZACIÓN =============
def plot_atr_analysis(results, ticker):
    """
    Gráfico avanzado del análisis ATR
    """
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 10), facecolor='#0E1117')
    gs = fig.add_gridspec(3, 1, height_ratios=[4, 1, 0.8], hspace=0.3)
    
    # Color mapping
    colors_map = {1: '#4ECDC4', -1: '#FF6B6B', 0: '#95A5A6', 2: '#FFB86C'}
    state_names = {1: 'ALCISTA', -1: 'BAJISTA', 0: 'LATERAL', 2: 'RIESGO'}
    
    # SUBPLOT 1: Precio y Análisis ATR
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#1A1D29')
    
    # Zona de confort ATR
    ax1.fill_between(range(len(results)), results['Lower_Risk'], results['Upper_Risk'],
                     color='#FFB86C', alpha=0.08, label='Zona de Confort (ATR)', zorder=1)
    
    # Líneas de riesgo
    ax1.plot(results['Upper_Risk'], color='#FFB86C', linewidth=1.5, 
             linestyle='--', alpha=0.6, zorder=2)
    ax1.plot(results['Lower_Risk'], color='#FFB86C', linewidth=1.5, 
             linestyle='--', alpha=0.6, zorder=2)
    
    # Tendencia kernel
    ax1.plot(results['Trend'], color='#00D9FF', linewidth=2.5, 
             label='Tendencia Kernel', zorder=3)
    
    # Precio con efectos
    ax1.plot(results['Close'], color='#FFFFFF', alpha=0.2, linewidth=4, zorder=4)
    ax1.plot(results['Close'], color='#E0E0E0', linewidth=1.8, 
             label='Precio', zorder=5)
    
    # Scatter por estado
    for state_val, color in colors_map.items():
        mask = results['State'] == state_val
        if mask.sum() > 0:
            ax1.scatter(results[mask].index, results[mask]['Close'],
                       c=color, s=60, alpha=0.7, edgecolors='white',
                       linewidth=1, label=state_names[state_val], zorder=6)
    
    # Último punto destacado
    last_idx = len(results) - 1
    last_state = results['State'].iloc[-1]
    ax1.scatter(last_idx, results['Close'].iloc[-1],
               facecolors=colors_map[last_state], edgecolors='white',
               s=300, linewidth=3, marker='o', zorder=10)
    
    # Anotación del estado actual
    ax1.annotate(f'{results["State_Name"].iloc[-1]}\n${results["Close"].iloc[-1]:.2f}',
                xy=(last_idx, results['Close'].iloc[-1]),
                xytext=(20, 30), textcoords='offset points',
                fontsize=12, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.8', 
                         facecolor=colors_map[last_state],
                         alpha=0.95, edgecolor='white', linewidth=2),
                arrowprops=dict(arrowstyle='->', lw=2.5,
                               color=colors_map[last_state]),
                zorder=11)
    
    ax1.set_title(f'{ticker} - Análisis de Tendencia con ATR Dinámico',
                 fontsize=18, fontweight='bold', color='#FFFFFF', pad=20)
    ax1.set_ylabel('Precio ($)', fontsize=13, fontweight='bold', color='#FFFFFF')
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.1, linestyle='-', linewidth=0.8)
    ax1.tick_params(labelsize=10, colors='#B0B0B0')
    
    # SUBPLOT 2: ATR Evolution
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_facecolor('#1A1D29')
    
    ax2.plot(results['ATR'], color='#BD93F9', linewidth=2, label='ATR')
    ax2.fill_between(range(len(results)), 0, results['ATR'],
                     color='#BD93F9', alpha=0.2)
    
    ax2.set_ylabel('ATR', fontsize=12, fontweight='bold', color='#FFFFFF')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.1)
    ax2.tick_params(labelsize=9, colors='#B0B0B0')
    
    # SUBPLOT 3: Semáforo de Estados
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.set_facecolor('#1A1D29')
    
    bar_colors = [colors_map[s] for s in results['State']]
    ax3.bar(range(len(results)), 1, color=bar_colors, width=1.0, alpha=0.9)
    
    ax3.set_ylabel('Estado', fontsize=11, fontweight='bold', color='#FFFFFF')
    ax3.set_yticks([])
    ax3.set_xlabel('Periodo', fontsize=12, fontweight='bold', color='#FFFFFF')
    ax3.tick_params(labelsize=9, colors='#B0B0B0')
    
    plt.tight_layout()
    return fig

# ============= INTERFAZ STREAMLIT =============
def main_app():
    """
    Aplicación principal
    """
    # CSS personalizado
    st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stMetric { 
        background: linear-gradient(135deg, #1A1D29 0%, #2D3142 100%);
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #4ECDC4;
        box-shadow: 0 4px 15px rgba(78, 205, 196, 0.2);
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
    h1, h2, h3 { color: #FFFFFF !important; font-weight: 800 !important; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("📊 ATR Trend Analyzer Pro")
    st.markdown("Sistema de Detección de Tendencia con Average True Range Dinámico")
    st.markdown("---")
    
    # Sidebar - Configuración
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 20px; 
                    background: linear-gradient(135deg, #4ECDC4 0%, #00D9FF 100%); 
                    border-radius: 15px; margin-bottom: 20px;'>
            <h2 style='color: white; margin: 0;'>⚙️ Configuración</h2>
        </div>
        """, unsafe_allow_html=True)
        
        ticker = st.text_input("Ticker Symbol", value="AAPL", 
                              help="Símbolo del activo (ej: AAPL, MSFT, ^GSPC)").upper()
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            period = st.selectbox("Periodo", 
                                 ["1y", "2y", "5y", "10y", "max"],
                                 index=1)
        with col2:
            interval = st.selectbox("Intervalo",
                                   ["1d", "1wk", "1mo"],
                                   index=1)
        
        st.markdown("---")
        
        st.subheader("Parámetros ATR")
        atr_period = st.slider("Periodo ATR", 7, 28, 14, 1)
        atr_multiplier = st.slider("Multiplicador ATR", 0.5, 3.0, 1.5, 0.1,
                                   help="Factor para definir zona de confort")
        
        st.markdown("---")
        
        st.subheader("Mejoras Algorítmicas")
        use_adaptive = st.checkbox("Bandwidth Adaptativo", value=True,
                                   help="Ajusta suavizado según volatilidad")
        use_dynamic = st.checkbox("Umbral Dinámico", value=True,
                                 help="Umbral de tendencia adaptativo")
        
        st.markdown("---")
        
        analyze_btn = st.button("🚀 ANALIZAR", type="primary", 
                               use_container_width=True)
        
        st.markdown("---")
        
        # Información metodológica
        with st.expander("ℹ️ Metodología"):
            st.markdown("""
            **Estados del Mercado:**
            - 🟢 **ALCISTA**: Tendencia alcista confirmada
            - 🔴 **BAJISTA**: Tendencia bajista confirmada
            - ⚫ **LATERAL**: Sin dirección clara (rango)
            - 🟠 **RIESGO**: Precio fuera de zona ATR (sobreextendido)
            
            **Indicadores:**
            - **Kernel Nadaraya-Watson**: Suavizado de tendencia
            - **ATR Dinámico**: Volatilidad adaptativa
            - **Bandwidth Adaptativo**: Ajuste según volatilidad reciente
            - **Umbral Dinámico**: Sensibilidad variable según mercado
            """)
    
    # Procesamiento y visualización
    if analyze_btn:
        with st.spinner(f"Analizando {ticker}..."):
            results, error = download_and_process_data(
                ticker, period, interval,
                atr_period, atr_multiplier,
                use_adaptive, use_dynamic
            )
            
            if error:
                st.error(f"❌ {error}")
                return
            
            if results is None:
                st.error("❌ No se pudieron obtener datos")
                return
            
            st.session_state['results'] = results
            st.session_state['ticker'] = ticker
            st.success(f"✅ Datos cargados: {len(results)} periodos")
    
    # Mostrar resultados si existen
    if 'results' in st.session_state:
        results = st.session_state['results']
        ticker = st.session_state['ticker']
        
        # Métricas actuales
        st.markdown("---")
        st.markdown("### 📈 Estado Actual del Mercado")
        
        current = results.iloc[-1]
        prev = results.iloc[-2] if len(results) > 1 else current
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            state_emoji = {'ALCISTA': '🟢', 'BAJISTA': '🔴', 
                          'LATERAL': '⚫', 'RIESGO': '🟠'}
            st.metric("ESTADO", 
                     f"{state_emoji[current['State_Name']]} {current['State_Name']}")
        
        with col2:
            price_change = ((current['Close'] - prev['Close']) / prev['Close'] * 100)
            st.metric("PRECIO", f"${current['Close']:.2f}", 
                     f"{price_change:+.2f}%")
        
        with col3:
            st.metric("ATR", f"${current['ATR']:.2f}",
                     help="Average True Range actual")
        
        with col4:
            trend_pos = ((current['Close'] - current['Lower_Risk']) / 
                        (current['Upper_Risk'] - current['Lower_Risk']) * 100)
            st.metric("Posición ATR", f"{trend_pos:.0f}%",
                     help="Posición dentro de la zona de confort")
        
        with col5:
            distance_trend = ((current['Close'] - current['Trend']) / 
                            current['Trend'] * 100)
            st.metric("Vs Tendencia", f"{distance_trend:+.2f}%")
        
        # Gráfico principal
        st.markdown("---")
        st.markdown("### 📊 Análisis Técnico Completo")
        
        fig = plot_atr_analysis(results, ticker)
        st.pyplot(fig)
        
        # Estadísticas adicionales
        st.markdown("---")
        st.markdown("### 📉 Estadísticas del Periodo")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            state_counts = results['State_Name'].value_counts()
            st.markdown("**Distribución de Estados:**")
            for state, count in state_counts.items():
                pct = (count / len(results) * 100)
                st.write(f"- {state}: {count} ({pct:.1f}%)")
        
        with col2:
            st.markdown("**Métricas de Precio:**")
            st.write(f"- Máximo: ${results['Close'].max():.2f}")
            st.write(f"- Mínimo: ${results['Close'].min():.2f}")
            st.write(f"- Promedio: ${results['Close'].mean():.2f}")
            st.write(f"- Volatilidad: {results['Close'].std():.2f}")
        
        with col3:
            st.markdown("**Métricas ATR:**")
            st.write(f"- ATR Promedio: ${results['ATR'].mean():.2f}")
            st.write(f"- ATR Máximo: ${results['ATR'].max():.2f}")
            st.write(f"- ATR Actual: ${current['ATR']:.2f}")
        
        # Descargar datos
        st.markdown("---")
        csv = results.to_csv(index=False)
        st.download_button(
            label="📥 Descargar Datos (CSV)",
            data=csv,
            file_name=f"atr_analysis_{ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    else:
        # Mensaje de bienvenida
        st.markdown("""
        <div style='text-align: center; padding: 60px 20px;
                    background: linear-gradient(135deg, #1A1D29 0%, #2D3142 100%);
                    border-radius: 20px; border: 3px solid #4ECDC4;
                    box-shadow: 0 8px 30px rgba(78, 205, 196, 0.3);
                    margin-top: 40px;'>
            <h2 style='color: #4ECDC4; margin: 0;'>
                👋 Bienvenido al ATR Trend Analyzer Pro
            </h2>
            <p style='color: #B0B0B0; font-size: 18px; margin: 20px 0;'>
                Sistema avanzado de detección de tendencias con Average True Range dinámico
            </p>
            <p style='color: #8E93A1; font-size: 14px;'>
                Configura los parámetros en el panel lateral y presiona 
                <strong style='color: #00D9FF;'>ANALIZAR</strong> para comenzar
            </p>
        </div>
        """, unsafe_allow_html=True)

# ============= EJECUCIÓN =============
if __name__ == "__main__":
    if check_password():
        main_app()
    else:
        st.stop()
