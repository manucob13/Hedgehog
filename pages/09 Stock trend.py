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
    """Verifica credenciales usando secrets de Streamlit - SIDEBAR"""
    def password_entered():
        if (st.session_state["username"] == st.secrets["credentials"]["username"] and
            st.session_state["password"] == st.secrets["credentials"]["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        with st.sidebar:
            st.markdown("""
            <div style='text-align: center; padding: 20px; 
                        background: linear-gradient(135deg, #FF6B6B 0%, #FFB86C 100%); 
                        border-radius: 15px; margin-bottom: 20px;'>
                <h2 style='color: white; margin: 0;'>🔐 Login</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.text_input("Usuario", key="username")
            st.text_input("Contraseña", type="password", key="password")
            st.button("Iniciar Sesión", on_click=password_entered, use_container_width=True)
        return False
    
    elif not st.session_state["password_correct"]:
        with st.sidebar:
            st.error("❌ Usuario o contraseña incorrectos")
            st.text_input("Usuario", key="username")
            st.text_input("Contraseña", type="password", key="password")
            st.button("Iniciar Sesión", on_click=password_entered, use_container_width=True)
        return False
    else:
        return True

# ============= FUNCIONES TÉCNICAS =============
def nadaraya_watson_kernel(x, y, bandwidth=8):
    """Suavizado Nadaraya-Watson con kernel Gaussiano"""
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
    """ATR mejorado usando pandas"""
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr.values

def calculate_ema(data, period):
    """Calcula EMA"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_macd_v(df, fast_len=12, slow_len=26, signal_len=9, atr_len=26):
    """Calcula MACD-V (MACD normalizado por ATR)"""
    fast_ema = calculate_ema(df['Close'], fast_len)
    slow_ema = calculate_ema(df['Close'], slow_len)
    
    # Calcular ATR para normalización
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=atr_len).mean()
    
    macd = ((fast_ema - slow_ema) / atr) * 100
    signal = calculate_ema(macd, signal_len)
    
    return macd, signal

def calculate_adaptive_bandwidth(volatility, base_bandwidth=8):
    """Bandwidth adaptativo según volatilidad"""
    recent_vol = volatility[-20:].mean() if len(volatility) >= 20 else volatility.mean()
    median_vol = np.median(volatility[volatility > 0])
    
    if median_vol > 0:
        vol_ratio = recent_vol / median_vol
        adaptive_bw = base_bandwidth * (0.7 + 0.6 * vol_ratio)
        return np.clip(adaptive_bw, base_bandwidth * 0.5, base_bandwidth * 2)
    return base_bandwidth

def calculate_dynamic_slope_threshold(trend, lookback=20):
    """Umbral de pendiente adaptativo"""
    if len(trend) < lookback:
        return np.full_like(trend, np.abs(trend).mean() * 0.002)
    
    recent_changes = np.abs(np.diff(trend[-lookback:]))
    threshold_value = np.percentile(recent_changes, 40)
    
    min_threshold = np.abs(trend) * 0.0015
    threshold_array = np.maximum(threshold_value, min_threshold)
    
    return threshold_array

def project_trend(x, trend, periods_ahead=4, lookback_points=10, poly_degree=2):
    """Proyección de tendencia usando extrapolación polinomial"""
    lookback_points = min(lookback_points, len(trend))
    poly_degree = min(poly_degree, lookback_points - 1)
    
    x_recent = x[-lookback_points:]
    y_recent = trend[-lookback_points:]
    
    poly_coeffs = np.polyfit(x_recent, y_recent, poly_degree)
    polynomial = np.poly1d(poly_coeffs)
    
    x_future = np.arange(len(trend), len(trend) + periods_ahead)
    y_future = polynomial(x_future)
    
    y_fit = polynomial(x_recent)
    residuals = y_recent - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_recent - np.mean(y_recent))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    confidence = max(0, min(1, r_squared))
    
    return x_future, y_future, confidence

def project_atr_bands(trend_projection, atr_current, atr_multiplier=1.5):
    """Proyecta las bandas ATR hacia el futuro"""
    upper_projection = trend_projection + (atr_current * atr_multiplier)
    lower_projection = trend_projection - (atr_current * atr_multiplier)
    return upper_projection, lower_projection

def classify_trend_state(prices, trend, atr, atr_multiplier=1.5, 
                         use_adaptive_bandwidth=True, 
                         use_dynamic_threshold=True):
    """Sistema de clasificación de tendencia mejorado para detectar BAJISTA correctamente"""
    x = np.arange(len(prices))
    
    if use_adaptive_bandwidth:
        bandwidth = calculate_adaptive_bandwidth(atr)
        trend_recalc = nadaraya_watson_kernel(x, prices, bandwidth=bandwidth)
    else:
        trend_recalc = trend
    
    upper_risk = trend_recalc + (atr * atr_multiplier)
    lower_risk = trend_recalc - (atr * atr_multiplier)
    
    # Pendiente actual
    slopes = np.diff(trend_recalc, prepend=trend_recalc[0])
    
    if use_dynamic_threshold:
        slope_thresholds = calculate_dynamic_slope_threshold(trend_recalc, 20)
    else:
        slope_thresholds = np.abs(trend_recalc) * 0.002
    
    states = np.zeros(len(prices), dtype=int)
    
    for i in range(len(prices)):
        # 1. Determinamos dirección base por la pendiente de la tendencia
        if slopes[i] > slope_thresholds[i]:
            base_state = 1  # ALCISTA
        elif slopes[i] < -slope_thresholds[i]:
            base_state = -1 # BAJISTA
        else:
            base_state = 0  # LATERAL

        # 2. Refinamos con RIESGO si el precio está muy fuera de las bandas ATR
        # Se prioriza RIESGO si el precio está sobreextendido por ARRIBA (posible burbuja)
        # O si está muy por debajo pero la tendencia aún no ha girado
        if prices[i] > upper_risk[i]:
            states[i] = 2  # RIESGO (Sobrecompra)
        elif prices[i] < lower_risk[i] and base_state != -1:
            states[i] = 2  # RIESGO (Caída brusca sin tendencia confirmada aún)
        else:
            states[i] = base_state
    
    # Filtro de confirmación suave (evita cambios de un solo punto)
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
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if data.empty or len(data) < 30:
            return None, "Datos insuficientes para el análisis"
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        prices = data['Close'].values
        x = np.arange(len(prices))
        
        # Kernel inicial
        trend_init = nadaraya_watson_kernel(x, prices, bandwidth=8)
        atr = calculate_atr_improved(data, period=atr_period)
        
        macd_v, macd_v_signal = calculate_macd_v(data, fast_len=12, slow_len=26, 
                                                  signal_len=9, atr_len=26)
        
        states, trend_refined, upper_risk, lower_risk = classify_trend_state(
            prices, trend_init, atr, 
            atr_multiplier=atr_multiplier,
            use_adaptive_bandwidth=use_adaptive,
            use_dynamic_threshold=use_dynamic
        )
        
        x_future, y_future, projection_confidence = project_trend(
            x, trend_refined, 
            periods_ahead=projection_periods,
            lookback_points=projection_lookback,
            poly_degree=projection_degree
        )
        
        atr_current = atr[-1] if not np.isnan(atr[-1]) else np.nanmean(atr[-5:])
        upper_future, lower_future = project_atr_bands(
            y_future, atr_current, atr_multiplier
        )
        
        results = pd.DataFrame({
            'Date': data.index,
            'Close': prices,
            'Trend': trend_refined,
            'ATR': atr,
            'Upper_Risk': upper_risk,
            'Lower_Risk': lower_risk,
            'State': states,
            'MACD_V': macd_v.values,
            'MACD_V_Signal': macd_v_signal.values
        })
        
        last_date = data.index[-1]
        freq = 'W' if interval == '1wk' else 'D' if interval == '1d' else 'M'
        
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
        
        state_names = {-1: 'BAJISTA', 0: 'LATERAL', 1: 'ALCISTA', 2: 'RIESGO'}
        results['State_Name'] = results['State'].map(state_names)
        
        metrics = {
            'projection_confidence': projection_confidence,
            'projection_change_pct': ((y_future[-1] - prices[-1]) / prices[-1] * 100),
            'projection_target': y_future[-1],
            'atr_current': atr_current
        }
        
        return (results, projection_df, metrics), None
        
    except Exception as e:
        return None, f"Error al descargar datos: {str(e)}"

# ============= VISUALIZACIÓN =============
def plot_atr_analysis_with_projection(results, projection_df, metrics, ticker):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 14), facecolor='#0E1117')
    gs = fig.add_gridspec(4, 1, height_ratios=[4, 1, 1, 0.8], hspace=0.35)
    
    colors_map = {1: '#4ECDC4', -1: '#EE5A6F', 0: '#95A5A6', 2: '#FF6B6B'}
    state_names = {1: 'ALCISTA', -1: 'BAJISTA', 0: 'LATERAL', 2: 'RIESGO'}
    
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#1A1D29')
    
    # Zonas ATR
    ax1.fill_between(range(len(results)), results['Lower_Risk'], results['Upper_Risk'],
                     color='#FFB86C', alpha=0.08, label='Zona ATR Histórica')
    
    # Proyección ATR
    proj_x_start = len(results) - 1
    proj_x = np.arange(proj_x_start, proj_x_start + len(projection_df) + 1)
    proj_upper = np.concatenate([[results['Upper_Risk'].iloc[-1]], projection_df['Upper_Projection'].values])
    proj_lower = np.concatenate([[results['Lower_Risk'].iloc[-1]], projection_df['Lower_Projection'].values])
    
    ax1.fill_between(proj_x, proj_lower, proj_upper,
                     color='#FFB86C', alpha=0.15, hatch='//',
                     label=f'Zona Proyectada (Conf: {metrics["projection_confidence"]*100:.0f}%)')
    
    ax1.plot(results['Trend'], color='#00D9FF', linewidth=2.5, label='Tendencia Kernel')
    
    proj_trend = np.concatenate([[results['Trend'].iloc[-1]], projection_df['Trend_Projection'].values])
    ax1.plot(proj_x, proj_trend, color='#FF6B6B', linewidth=3, linestyle='--', label='Proyección')
    
    ax1.plot(results['Close'], color='#FFFFFF', alpha=0.6, linewidth=1.5, label='Precio')
    
    for state_val, color in colors_map.items():
        mask = results['State'] == state_val
        if mask.sum() > 0:
            ax1.scatter(results[mask].index, results[mask]['Close'],
                       c=color, s=60, alpha=0.8, edgecolors='white', label=state_names[state_val])
    
    # Anotación Precio Actual
    last_idx = len(results) - 1
    last_state = results['State'].iloc[-1]
    ax1.annotate(f'{results["State_Name"].iloc[-1]}\n${results["Close"].iloc[-1]:.2f}',
                xy=(last_idx, results['Close'].iloc[-1]),
                xytext=(-80, -40), textcoords='offset points',
                fontsize=12, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.8', facecolor=colors_map[last_state], alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='white'))

    ax1.set_ylabel('Precio ($)', fontsize=14, color='white')
    ax1.legend(loc='upper left', ncol=2)
    ax1.grid(alpha=0.1)

    # Subplot ATR
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_facecolor('#1A1D29')
    ax2.plot(results['ATR'], color='#BD93F9', linewidth=2)
    ax2.fill_between(range(len(results)), 0, results['ATR'], color='#BD93F9', alpha=0.1)
    ax2.set_ylabel('ATR (Volatilidad)', color='white')

    # Subplot MACD-V
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.set_facecolor('#1A1D29')
    for i in range(1, len(results)):
        if pd.notna(results['MACD_V'].iloc[i]):
            y = results['MACD_V'].iloc[i]
            color = '#4ECDC4' if y > 50 else '#EE5A6F' if y < -50 else '#95A5A6'
            ax3.plot([i-1, i], [results['MACD_V'].iloc[i-1], y], color=color, linewidth=2.5)
    ax3.axhline(0, color='white', alpha=0.3)
    ax3.set_ylabel('MACD-V', color='white')

    # Subplot Semáforo
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.set_facecolor('#1A1D29')
    regime_order = ['BAJISTA', 'LATERAL', 'ALCISTA', 'RIESGO']
    state_to_y = {'BAJISTA': 0, 'LATERAL': 1, 'ALCISTA': 2, 'RIESGO': 3}
    for state_name in regime_order:
        mask = results['State_Name'] == state_name
        if mask.sum() > 0:
            color = colors_map[{-1: -1, 0: 0, 1: 1, 2: 2}[results[mask]['State'].iloc[0]]]
            ax4.scatter(results[mask].index, [state_to_y[state_name]] * mask.sum(), 
                       c=color, s=100, edgecolors='white')
    ax4.set_yticks(range(4))
    ax4.set_yticklabels(regime_order)
    
    plt.tight_layout()
    return fig

# ============= APP PRINCIPAL =============
def main_app():
    st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stMetric { background: #1A1D29; padding: 15px; border-radius: 10px; border: 1px solid #4ECDC4; }
    h1, h2, h3 { color: white !important; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("📊 ATR Trend Analyzer Pro + Proyección")
    
    with st.sidebar:
        st.header("⚙️ Configuración")
        ticker = st.text_input("Ticker Symbol", value="AAPL").upper()
        col1, col2 = st.columns(2)
        period = col1.selectbox("Periodo", ["1y", "2y", "5y", "10y"], index=1)
        interval = col2.selectbox("Intervalo", ["1d", "1wk", "1mo"], index=1)
        
        st.subheader("🔮 Proyección")
        projection_periods = st.slider("Períodos a Proyectar", 1, 12, 4)
        projection_degree = st.selectbox("Modelo", [1, 2, 3], format_func=lambda x: ["Lineal", "Cuadrático", "Cúbico"][x-1], index=1)
        
        atr_multiplier = st.slider("Multiplicador ATR", 0.5, 3.0, 1.5, 0.1)
        analyze_btn = st.button("🚀 ANALIZAR + PROYECTAR", use_container_width=True, type="primary")

    if analyze_btn or 'results' in st.session_state:
        if analyze_btn:
            with st.spinner(f"Procesando {ticker}..."):
                res = download_and_process_data(ticker, period, interval, 
                                               atr_multiplier=atr_multiplier,
                                               projection_periods=projection_periods,
                                               projection_degree=projection_degree)
                if res[1]: 
                    st.error(res[1])
                else:
                    st.session_state['results'], st.session_state['proj'], st.session_state['met'] = res[0]
                    st.session_state['ticker'] = ticker

        if 'results' in st.session_state:
            results = st.session_state['results']
            projection_df = st.session_state['proj']
            metrics = st.session_state['met']
            
            # Dashboard de métricas
            c1, c2, c3, c4 = st.columns(4)
            cur = results.iloc[-1]
            state_colors = {'ALCISTA': '🟢', 'BAJISTA': '🔴', 'LATERAL': '⚪', 'RIESGO': '🟠'}
            c1.metric("ESTADO", f"{state_colors[cur['State_Name']]} {cur['State_Name']}")
            c2.metric("PRECIO", f"${cur['Close']:.2f}")
            c3.metric("TARGET PROYECTADO", f"${metrics['projection_target']:.2f}", f"{metrics['projection_change_pct']:+.1f}%")
            c4.metric("CONFIANZA", f"{metrics['projection_confidence']*100:.0f}%")
            
            st.pyplot(plot_atr_analysis_with_projection(results, projection_df, metrics, st.session_state['ticker']))
            
            with st.expander("📝 Interpretación del Analista"):
                if cur['State_Name'] == 'BAJISTA':
                    st.warning(f"⚠️ {st.session_state['ticker']} está en una fase bajista confirmada por la pendiente de la tendencia.")
                elif cur['State_Name'] == 'ALCISTA':
                    st.success(f"✅ Tendencia alcista sólida. El momentum acompaña el movimiento.")
                elif cur['State_Name'] == 'RIESGO':
                    st.error(f"🚨 Alerta de riesgo: El precio está muy alejado de su tendencia histórica. Posible reversión o corrección.")

if __name__ == "__main__":
    if check_password():
        main_app()
