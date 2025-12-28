import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Configuración de página
st.set_page_config(
    page_title="ATR Trend Analyzer + Projection",
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
            <h1 style='color: #4ECDC4;'>🔐 ATR Trend Analyzer + Projection</h1>
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
        return y.copy()
    
    est_y = np.zeros_like(y, dtype=float)
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
    Bandwidth adaptativo según volatilidad reciente
    """
    recent_vol = volatility[-20:].mean() if len(volatility) >= 20 else volatility.mean()
    median_vol = np.median(volatility[volatility > 0])
    
    if median_vol > 0:
        vol_ratio = recent_vol / median_vol
        adaptive_bw = base_bandwidth * (0.7 + 0.6 * vol_ratio)
        return np.clip(adaptive_bw, base_bandwidth * 0.5, base_bandwidth * 2)
    return base_bandwidth

def calculate_dynamic_slope_threshold(trend, lookback=20):
    """
    Umbral de pendiente adaptativo
    """
    if len(trend) < lookback:
        return np.full_like(trend, np.abs(trend).mean() * 0.002)
    
    recent_changes = np.abs(np.diff(trend[-lookback:]))
    threshold_value = np.percentile(recent_changes, 40)
    
    min_threshold = np.abs(trend) * 0.0015
    threshold_array = np.maximum(threshold_value, min_threshold)
    
    return threshold_array

def project_trend(x, trend, periods_ahead=4, lookback_points=10, poly_degree=2):
    """
    NUEVA FUNCIÓN: Proyección de tendencia usando extrapolación polinomial
    
    Parameters:
    -----------
    x : array - Índices temporales
    trend : array - Valores de tendencia
    periods_ahead : int - Períodos a proyectar hacia adelante
    lookback_points : int - Puntos históricos para calcular inercia
    poly_degree : int - Grado del polinomio (1=lineal, 2=cuadrático)
    
    Returns:
    --------
    x_future : array - Índices futuros
    y_future : array - Valores proyectados
    confidence : float - Confianza de la proyección (0-1)
    """
    # Ajustar lookback si no hay suficientes datos
    lookback_points = min(lookback_points, len(trend))
    
    # Ajustar grado del polinomio según datos disponibles
    poly_degree = min(poly_degree, lookback_points - 1)
    
    # Extraer últimos puntos para proyección
    x_recent = x[-lookback_points:]
    y_recent = trend[-lookback_points:]
    
    # Ajustar polinomio
    poly_coeffs = np.polyfit(x_recent, y_recent, poly_degree)
    polynomial = np.poly1d(poly_coeffs)
    
    # Generar proyección
    x_future = np.arange(len(trend), len(trend) + periods_ahead)
    y_future = polynomial(x_future)
    
    # Calcular confianza basada en R² del ajuste
    y_fit = polynomial(x_recent)
    residuals = y_recent - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_recent - np.mean(y_recent))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    confidence = max(0, min(1, r_squared))
    
    return x_future, y_future, confidence

def project_atr_bands(trend_projection, atr_current, atr_multiplier=1.5):
    """
    Proyecta las bandas ATR hacia el futuro
    Asume ATR constante (conservador)
    """
    upper_projection = trend_projection + (atr_current * atr_multiplier)
    lower_projection = trend_projection - (atr_current * atr_multiplier)
    return upper_projection, lower_projection

def classify_trend_state(prices, trend, atr, atr_multiplier=1.5, 
                         use_adaptive_bandwidth=True, 
                         use_dynamic_threshold=True):
    """
    Sistema de clasificación de tendencia de 4 estados MEJORADO
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
    
    # Umbral dinámico
    if use_dynamic_threshold:
        slope_thresholds = calculate_dynamic_slope_threshold(trend_recalc, 20)
    else:
        slope_thresholds = np.abs(trend_recalc) * 0.002
    
    states = np.zeros(len(prices), dtype=int)
    
    for i in range(len(prices)):
        # PRIORIDAD 1: RIESGO
        if prices[i] > upper_risk[i] or prices[i] < lower_risk[i]:
            if i > 0:
                price_change = (prices[i] - prices[i-1]) / prices[i-1]
                if abs(price_change) > 0.03:
                    states[i] = 2
                else:
                    if slopes[i] > slope_thresholds[i]:
                        states[i] = 1
                    elif slopes[i] < -slope_thresholds[i]:
                        states[i] = -1
                    else:
                        states[i] = 0
            else:
                states[i] = 2
        elif slopes[i] > slope_thresholds[i]:
            states[i] = 1  # ALCISTA
        elif slopes[i] < -slope_thresholds[i]:
            states[i] = -1  # BAJISTA
        else:
            states[i] = 0  # LATERAL
    
    # Filtro de confirmación
    filtered_states = states.copy()
    for i in range(1, len(states) - 1):
        if states[i] != states[i-1] and states[i] != states[i+1]:
            filtered_states[i] = states[i-1]
    
    return filtered_states, trend_recalc, upper_risk, lower_risk

# ============= DESCARGA Y PROCESAMIENTO DE DATOS =============
@st.cache_data(ttl=3600)
def download_and_process_data(ticker, period="2y", interval="1wk",
                               atr_period=14, atr_multiplier=1.5,
                               use_adaptive=True, use_dynamic=True,
                               projection_periods=4, projection_lookback=10,
                               projection_degree=2):
    """
    Descarga datos y calcula todos los indicadores + PROYECCIÓN
    """
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if data.empty or len(data) < 30:
            return None, "Datos insuficientes para el análisis"
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        prices = data['Close'].values
        x = np.arange(len(prices))
        
        # Cálculo de tendencia kernel
        trend = nadaraya_watson_kernel(x, prices, bandwidth=8)
        
        # Cálculo de ATR
        atr = calculate_atr_improved(data, period=atr_period)
        
        # Clasificación de estados
        states, trend_refined, upper_risk, lower_risk = classify_trend_state(
            prices, trend, atr, 
            atr_multiplier=atr_multiplier,
            use_adaptive_bandwidth=use_adaptive,
            use_dynamic_threshold=use_dynamic
        )
        
        # *** NUEVA FUNCIONALIDAD: PROYECCIÓN ***
        x_future, y_future, projection_confidence = project_trend(
            x, trend_refined, 
            periods_ahead=projection_periods,
            lookback_points=projection_lookback,
            poly_degree=projection_degree
        )
        
        # Proyectar bandas ATR
        atr_current = atr[-1] if not np.isnan(atr[-1]) else np.nanmean(atr[-5:])
        upper_future, lower_future = project_atr_bands(
            y_future, atr_current, atr_multiplier
        )
        
        # Crear DataFrame de resultados históricos
        results = pd.DataFrame({
            'Date': data.index,
            'Close': prices,
            'Trend': trend_refined,
            'ATR': atr,
            'Upper_Risk': upper_risk,
            'Lower_Risk': lower_risk,
            'State': states
        })
        
        # DataFrame de proyección
        last_date = data.index[-1]
        if interval == '1wk':
            freq = 'W'
        elif interval == '1d':
            freq = 'D'
        elif interval == '1mo':
            freq = 'M'
        else:
            freq = 'W'
        
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=projection_periods,
            freq=freq
        )
        
        projection_df = pd.DataFrame({
            'Date': future_dates,
            'Trend_Projection': y_future,
            'Upper_Projection': upper_future,
            'Lower_Projection': lower_future
        })
        
        # Mapeo de estados
        state_names = {-1: 'BAJISTA', 0: 'LATERAL', 1: 'ALCISTA', 2: 'RIESGO'}
        results['State_Name'] = results['State'].map(state_names)
        
        # Métricas adicionales
        metrics = {
            'projection_confidence': projection_confidence,
            'projection_change_pct': ((y_future[-1] - prices[-1]) / prices[-1] * 100),
            'projection_target': y_future[-1],
            'atr_current': atr_current
        }
        
        return (results, projection_df, metrics), None
        
    except Exception as e:
        return None, f"Error al descargar datos: {str(e)}"

# ============= VISUALIZACIÓN MEJORADA CON PROYECCIÓN =============
def plot_atr_analysis_with_projection(results, projection_df, metrics, ticker):
    """
    Gráfico avanzado con proyección incluida
    """
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 12), facecolor='#0E1117')
    gs = fig.add_gridspec(3, 1, height_ratios=[5, 1, 0.8], hspace=0.3)
    
    # Color mapping
    colors_map = {1: '#4ECDC4', -1: '#FF6B6B', 0: '#95A5A6', 2: '#FFB86C'}
    state_names = {1: 'ALCISTA', -1: 'BAJISTA', 0: 'LATERAL', 2: 'RIESGO'}
    
    # SUBPLOT 1: Precio, Tendencia y PROYECCIÓN
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#1A1D29')
    
    # Zona histórica de confort ATR
    ax1.fill_between(range(len(results)), results['Lower_Risk'], results['Upper_Risk'],
                     color='#FFB86C', alpha=0.08, label='Zona Histórica (ATR)', zorder=1)
    
    # Zona proyectada de confort ATR
    proj_x_start = len(results) - 1
    proj_x = np.arange(proj_x_start, proj_x_start + len(projection_df) + 1)
    proj_upper = np.concatenate([[results['Upper_Risk'].iloc[-1]], projection_df['Upper_Projection'].values])
    proj_lower = np.concatenate([[results['Lower_Risk'].iloc[-1]], projection_df['Lower_Projection'].values])
    
    ax1.fill_between(proj_x, proj_lower, proj_upper,
                     color='#FFB86C', alpha=0.15, 
                     label=f'Zona Proyectada (Confianza: {metrics["projection_confidence"]*100:.0f}%)', 
                     zorder=1, hatch='//')
    
    # Líneas de riesgo históricas
    ax1.plot(results['Upper_Risk'], color='#FFB86C', linewidth=1.5, 
             linestyle='--', alpha=0.6, zorder=2)
    ax1.plot(results['Lower_Risk'], color='#FFB86C', linewidth=1.5, 
             linestyle='--', alpha=0.6, zorder=2)
    
    # Líneas de riesgo proyectadas
    ax1.plot(proj_x, proj_upper, color='#FF6B6B', linewidth=2, 
             linestyle=':', alpha=0.8, zorder=2)
    ax1.plot(proj_x, proj_lower, color='#FF6B6B', linewidth=2, 
             linestyle=':', alpha=0.8, zorder=2)
    
    # Tendencia histórica
    ax1.plot(results['Trend'], color='#00D9FF', linewidth=2.5, 
             label='Tendencia Kernel', zorder=3)
    
    # *** PROYECCIÓN DE TENDENCIA ***
    proj_trend = np.concatenate([[results['Trend'].iloc[-1]], projection_df['Trend_Projection'].values])
    ax1.plot(proj_x, proj_trend, color='#FF6B6B', linewidth=3, 
             linestyle='--', label=f'Proyección (4 periodos)', zorder=4,
             marker='o', markersize=8, markerfacecolor='#FF6B6B', 
             markeredgecolor='white', markeredgewidth=2)
    
    # Precio histórico
    ax1.plot(results['Close'], color='#FFFFFF', alpha=0.2, linewidth=4, zorder=4)
    ax1.plot(results['Close'], color='#E0E0E0', linewidth=1.8, 
             label='Precio Histórico', zorder=5)
    
    # Scatter por estado
    for state_val, color in colors_map.items():
        mask = results['State'] == state_val
        if mask.sum() > 0:
            ax1.scatter(results[mask].index, results[mask]['Close'],
                       c=color, s=60, alpha=0.7, edgecolors='white',
                       linewidth=1, label=state_names[state_val], zorder=6)
    
    # Último punto histórico destacado
    last_idx = len(results) - 1
    last_state = results['State'].iloc[-1]
    ax1.scatter(last_idx, results['Close'].iloc[-1],
               facecolors=colors_map[last_state], edgecolors='white',
               s=300, linewidth=3, marker='o', zorder=10)
    
    # Punto de proyección final
    proj_last_idx = proj_x[-1]
    ax1.scatter(proj_last_idx, projection_df['Trend_Projection'].iloc[-1],
               facecolors='#FF6B6B', edgecolors='yellow',
               s=400, linewidth=4, marker='*', zorder=11,
               label=f'Target: ${projection_df["Trend_Projection"].iloc[-1]:.2f}')
    
    # Anotación estado actual
    ax1.annotate(f'{results["State_Name"].iloc[-1]}\n${results["Close"].iloc[-1]:.2f}',
                xy=(last_idx, results['Close'].iloc[-1]),
                xytext=(-80, -40), textcoords='offset points',
                fontsize=11, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.8', 
                         facecolor=colors_map[last_state],
                         alpha=0.95, edgecolor='white', linewidth=2),
                arrowprops=dict(arrowstyle='->', lw=2.5,
                               color=colors_map[last_state]),
                zorder=11)
    
    # Anotación proyección
    change_pct = metrics['projection_change_pct']
    arrow_color = '#4ECDC4' if change_pct > 0 else '#FF6B6B'
    ax1.annotate(f'Proyección\n${projection_df["Trend_Projection"].iloc[-1]:.2f}\n({change_pct:+.1f}%)',
                xy=(proj_last_idx, projection_df['Trend_Projection'].iloc[-1]),
                xytext=(30, 30), textcoords='offset points',
                fontsize=12, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.8', 
                         facecolor=arrow_color,
                         alpha=0.95, edgecolor='yellow', linewidth=3),
                arrowprops=dict(arrowstyle='->', lw=3,
                               color='yellow'),
                zorder=12)
    
    # Línea vertical separando histórico de proyección
    ax1.axvline(x=last_idx, color='yellow', linestyle=':', 
                linewidth=2, alpha=0.5, zorder=1)
    ax1.text(last_idx, ax1.get_ylim()[1]*0.98, 'HOY', 
            ha='center', va='top', fontsize=10, fontweight='bold',
            color='yellow', bbox=dict(boxstyle='round,pad=0.4',
                                     facecolor='black', alpha=0.7))
    
    ax1.set_title(f'{ticker} - Análisis ATR + Proyección Inercial',
                 fontsize=20, fontweight='bold', color='#FFFFFF', pad=20)
    ax1.set_ylabel('Precio ($)', fontsize=14, fontweight='bold', color='#FFFFFF')
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9, ncol=2)
    ax1.grid(True, alpha=0.1, linestyle='-', linewidth=0.8)
    ax1.tick_params(labelsize=10, colors='#B0B0B0')
    
    # SUBPLOT 2: ATR Evolution
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_facecolor('#1A1D29')
    
    ax2.plot(results['ATR'], color='#BD93F9', linewidth=2, label='ATR Histórico')
    ax2.fill_between(range(len(results)), 0, results['ATR'],
                     color='#BD93F9', alpha=0.2)
    
    # ATR proyectado (constante)
    ax2.axhline(y=metrics['atr_current'], xmin=last_idx/len(results), 
                color='#BD93F9', linestyle='--', linewidth=2, alpha=0.6,
                label=f'ATR Proyectado: ${metrics["atr_current"]:.2f}')
    
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
    Aplicación principal con PROYECCIÓN
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
    .projection-metric {
        background: linear-gradient(135deg, #FF6B6B 0%, #FFB86C 100%);
        padding: 20px;
        border-radius: 12px;
        border: 3px solid #FFD700;
        box-shadow: 0 4px 20px rgba(255, 215, 0, 0.4);
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
    st.title("📊 ATR Trend Analyzer Pro + Proyección")
    st.markdown("Sistema de Detección de Tendencia con Proyección Inercial")
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
                                   help="Factor para zona de confort")
        
        st.markdown("---")
        
        st.subheader("🔮 Parámetros de Proyección")
        projection_periods = st.slider("Períodos a Proyectar", 1, 12, 4, 1,
                                       help="Cuántos períodos proyectar hacia adelante")
        projection_lookback = st.slider("Puntos de Inercia", 5, 20, 10, 1,
                                        help="Períodos históricos para calcular tendencia")
        projection_degree = st.selectbox("Tipo de Proyección",
                                        options=[1, 2, 3],
                                        format_func=lambda x: {1: "Lineal", 2: "Cuadrática", 3: "Cúbica"}[x],
                                        index=1,
                                        help="Grado del polinomio de proyección")
        
        st.markdown("---")
        
        st.subheader("Mejoras Algorítmicas")
        use_adaptive = st.checkbox("Bandwidth Adaptativo", value=True,
                                   help="Ajusta suavizado según volatilidad")
        use_dynamic = st.checkbox("Umbral Dinámico", value=True,
                                 help="Umbral de tendencia adaptativo")
        
        st.markdown("---")
        
        analyze_btn = st.button("🚀 ANALIZAR + PROYECTAR", type="primary", 
                               use_container_width=True)
        
        st.markdown("---")
        
        # Información metodológica
        with st.expander("ℹ️ Metodología"):
            st.markdown("""
            **Estados del Mercado:**
            - 🟢 **ALCISTA**: Tendencia alcista confirmada
            - 🔴 **BAJISTA**: Tendencia bajista confirmada
            - ⚫ **LATERAL**: Sin dirección clara (rango)
            - 🟠 **RIESGO**: Precio fuera de zona ATR
            
            **Indicadores:**
            - **Kernel NW**: Suavizado de tendencia
            - **ATR**: Volatilidad adaptativa
            - **Proyección**: Extrapolación polinomial
            
            **Proyección:**
            - Usa últimos N puntos para calcular inercia
            - Proyección polinomial (lineal/cuadrática/cúbica)
            - Confianza basada en R² del ajuste
            """)
    
    # Procesamiento y visualización
    if analyze_btn:
        with st.spinner(f"Analizando {ticker} y calculando proyección..."):
            result = download_and_process_data(
                ticker, period, interval,
                atr_period, atr_multiplier,
                use_adaptive, use_dynamic,
                projection_periods, projection_lookback,
                projection_degree
            )
            
            if result[1]:  # Error
                st.error(f"❌ {result[1]}")
                return
            
            results, projection_df, metrics = result[0]
            
            if results is None:
                st.error("❌ No se pudieron obtener datos")
                return
            
            st.session_state['results'] = results
            st.session_state['projection_df'] = projection_df
            st.session_state['metrics'] = metrics
            st.session_state['ticker'] = ticker
            st.success(f"✅ Análisis completado: {len(results)} periodos + {len(projection_df)} proyectados")
    
    # Mostrar resultados si existen
    if 'results' in st.session_state:
        results = st.session_state['results']
        projection_df = st.session_state['projection_df']
        metrics = st.session_state['metrics']
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
            st.metric("ESTADO ACTUAL", 
                     f"{state_emoji[current['State_Name']]} {current['State_Name']}")
        
        with col2:
            price_change = ((current['Close'] - prev['Close']) / prev['Close'] * 100)
            st.metric("PRECIO ACTUAL", f"${current['Close']:.2f}", 
                     f"{price_change:+.2f}%")
        
        with col3:
            st.metric("ATR", f"${current['ATR']:.2f}",
                     help="Average True Range actual")
        
        with col4:
            trend_pos = ((current['Close'] - current['Lower_Risk']) / 
                        (current['Upper_Risk'] - current['Lower_Risk']) * 100)
            st.metric("Posición ATR", f"{trend_pos:.0f}%",
                     help="Posición dentro de zona de confort")
        
        with col5:
            distance_trend = ((current['Close'] - current['Trend']) / 
                            current['Trend'] * 100)
            st.metric("Vs Tendencia", f"{distance_trend:+.2f}%")
        
        # NUEVA SECCIÓN: Métricas de Proyección
        st.markdown("---")
        st.markdown("### 🔮 Proyección a Futuro")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            proj_target = projection_df['Trend_Projection'].iloc[-1]
            st.markdown(f"""
            <div class='projection-metric'>
                <div style='font-size: 14px; color: #FFD700; font-weight: bold;'>
                    TARGET PROYECTADO
                </div>
                <div style='font-size: 28px; color: white; font-weight: bold; margin-top: 10px;'>
                    ${proj_target:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            proj_change = metrics['projection_change_pct']
            color = '#4ECDC4' if proj_change > 0 else '#FF6B6B'
            arrow = '↗' if proj_change > 0 else '↘'
            st.markdown(f"""
            <div class='projection-metric'>
                <div style='font-size: 14px; color: #FFD700; font-weight: bold;'>
                    CAMBIO ESPERADO
                </div>
                <div style='font-size: 28px; color: {color}; font-weight: bold; margin-top: 10px;'>
                    {arrow} {proj_change:+.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            confidence = metrics['projection_confidence']
            conf_color = '#4ECDC4' if confidence > 0.7 else '#FFB86C' if confidence > 0.4 else '#FF6B6B'
            conf_label = 'Alta' if confidence > 0.7 else 'Media' if confidence > 0.4 else 'Baja'
            st.markdown(f"""
            <div class='projection-metric'>
                <div style='font-size: 14px; color: #FFD700; font-weight: bold;'>
                    CONFIANZA (R²)
                </div>
                <div style='font-size: 28px; color: {conf_color}; font-weight: bold; margin-top: 10px;'>
                    {confidence*100:.0f}% ({conf_label})
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            date_target = projection_df['Date'].iloc[-1].strftime('%Y-%m-%d')
            st.markdown(f"""
            <div class='projection-metric'>
                <div style='font-size: 14px; color: #FFD700; font-weight: bold;'>
                    FECHA TARGET
                </div>
                <div style='font-size: 20px; color: white; font-weight: bold; margin-top: 10px;'>
                    {date_target}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Tabla de proyección detallada
        st.markdown("---")
        with st.expander("📅 Ver Proyección Detallada"):
            proj_display = projection_df.copy()
            proj_display['Date'] = proj_display['Date'].dt.strftime('%Y-%m-%d')
            proj_display['Trend_Projection'] = proj_display['Trend_Projection'].round(2)
            proj_display['Upper_Projection'] = proj_display['Upper_Projection'].round(2)
            proj_display['Lower_Projection'] = proj_display['Lower_Projection'].round(2)
            proj_display.columns = ['Fecha', 'Precio Proyectado', 'Banda Superior', 'Banda Inferior']
            st.dataframe(proj_display, use_container_width=True)
        
        # Gráfico principal
        st.markdown("---")
        st.markdown("### 📊 Análisis Técnico + Proyección")
        
        fig = plot_atr_analysis_with_projection(results, projection_df, metrics, ticker)
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
        
        # Interpretación automática
        st.markdown("---")
        st.markdown("### 🤖 Interpretación Automática")
        
        interpretation = f"""
        **Análisis de {ticker}:**
        
        - **Estado Actual**: El mercado se encuentra en estado **{current['State_Name']}**
        - **Precio**: ${current['Close']:.2f} ({distance_trend:+.1f}% respecto a tendencia)
        - **Proyección**: Se espera que el precio alcance **${proj_target:.2f}** ({proj_change:+.1f}%)
        - **Confianza**: La proyección tiene una confianza **{conf_label.lower()}** ({confidence*100:.0f}%)
        - **Zona ATR**: El precio está en el {trend_pos:.0f}% de la zona de confort
        """
        
        if proj_change > 5:
            interpretation += "\n- ⚠️ **Alerta**: Proyección alcista significativa (>5%)"
        elif proj_change < -5:
            interpretation += "\n- ⚠️ **Alerta**: Proyección bajista significativa (<-5%)"
        
        if confidence < 0.5:
            interpretation += "\n- ⚠️ **Precaución**: Baja confianza en la proyección, mercado puede ser volátil"
        
        if current['State_Name'] == 'RIESGO':
            interpretation += "\n- 🚨 **Advertencia**: Precio fuera de zona de confort, posible reversión"
        
        st.info(interpretation)
        
        # Descargar datos
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            csv_hist = results.to_csv(index=False)
            st.download_button(
                label="📥 Descargar Datos Históricos (CSV)",
                data=csv_hist,
                file_name=f"atr_historical_{ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            csv_proj = projection_df.to_csv(index=False)
            st.download_button(
                label="🔮 Descargar Proyección (CSV)",
                data=csv_proj,
                file_name=f"atr_projection_{ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
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
                👋 Bienvenido al ATR Trend Analyzer Pro + Proyección
            </h2>
            <p style='color: #B0B0B0; font-size: 18px; margin: 20px 0;'>
                Sistema avanzado con proyección inercial basada en tendencia kernel
            </p>
            <p style='color: #8E93A1; font-size: 14px;'>
                Configura los parámetros en el panel lateral y presiona 
                <strong style='color: #00D9FF;'>ANALIZAR + PROYECTAR</strong> para comenzar
            </p>
            <div style='margin-top: 30px; padding: 20px; background: rgba(255, 107, 107, 0.1); border-radius: 10px;'>
                <h3 style='color: #FF6B6B; margin: 0 0 10px 0;'>🔮 Nueva Funcionalidad: Proyección</h3>
                <p style='color: #B0B0B0; font-size: 14px;'>
                    Ahora incluye proyección polinomial hacia el futuro con cálculo de confianza
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============= EJECUCIÓN =============
if __name__ == "__main__":
    if check_password():
        main_app()
    else:
        st.stop()
