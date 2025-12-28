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
    return data.ewm(span=period, adjust=False).mean()

def calculate_macd_v(df, fast_len=12, slow_len=26, signal_len=9, atr_len=26):
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

def calculate_adaptive_bandwidth(volatility, base_bandwidth=8):
    recent_vol = volatility[-20:].mean() if len(volatility) >= 20 else volatility.mean()
    median_vol = np.median(volatility[volatility > 0])
    if median_vol > 0:
        vol_ratio = recent_vol / median_vol
        adaptive_bw = base_bandwidth * (0.7 + 0.6 * vol_ratio)
        return np.clip(adaptive_bw, base_bandwidth * 0.5, base_bandwidth * 2)
    return base_bandwidth

def calculate_dynamic_slope_threshold(trend, lookback=20):
    if len(trend) < lookback:
        return np.full_like(trend, np.abs(trend).mean() * 0.002)
    recent_changes = np.abs(np.diff(trend[-lookback:]))
    threshold_value = np.percentile(recent_changes, 40)
    min_threshold = np.abs(trend) * 0.0015
    return np.maximum(threshold_value, min_threshold)

def project_trend(x, trend, periods_ahead=4, lookback_points=10, poly_degree=2):
    """Proyección de tendencia con corrección de GAP (Continuidad)"""
    lookback_points = min(lookback_points, len(trend))
    poly_degree = min(poly_degree, lookback_points - 1)
    
    x_recent = x[-lookback_points:]
    y_recent = trend[-lookback_points:]
    
    # Ajuste del modelo
    poly_coeffs = np.polyfit(x_recent, y_recent, poly_degree)
    polynomial = np.poly1d(poly_coeffs)
    
    # --- CORRECCIÓN DE CONTINUIDAD ---
    # Calculamos cuánto se desvía el modelo del último punto real
    last_real_val = trend[-1]
    last_x_idx = x[-1]
    model_val_at_last_x = polynomial(last_x_idx)
    offset = last_real_val - model_val_at_last_x
    
    # Generamos futuro aplicando el offset para eliminar el gap
    x_future = np.arange(len(trend), len(trend) + periods_ahead)
    y_future = polynomial(x_future) + offset # <--- Aplicamos el desplazamiento
    
    # Cálculo de R2 para confianza
    y_fit = polynomial(x_recent)
    residuals = y_recent - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_recent - np.mean(y_recent))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return x_future, y_future, max(0, min(1, r_squared))

def project_atr_bands(trend_projection, atr_current, atr_multiplier=1.5):
    upper_projection = trend_projection + (atr_current * atr_multiplier)
    lower_projection = trend_projection - (atr_current * atr_multiplier)
    return upper_projection, lower_projection

def classify_trend_state(prices, trend, atr, atr_multiplier=1.5, 
                         use_adaptive_bandwidth=True, 
                         use_dynamic_threshold=True):
    """Sistema de clasificación mejorado con prioridad BAJISTA"""
    x = np.arange(len(prices))
    if use_adaptive_bandwidth:
        bandwidth = calculate_adaptive_bandwidth(atr)
        trend_recalc = nadaraya_watson_kernel(x, prices, bandwidth=bandwidth)
    else:
        trend_recalc = trend
    
    upper_risk = trend_recalc + (atr * atr_multiplier)
    lower_risk = trend_recalc - (atr * atr_multiplier)
    slopes = np.diff(trend_recalc, prepend=trend_recalc[0])
    
    if use_dynamic_threshold:
        slope_thresholds = calculate_dynamic_slope_threshold(trend_recalc, 20)
    else:
        slope_thresholds = np.abs(trend_recalc) * 0.002
    
    states = np.zeros(len(prices), dtype=int)
    for i in range(len(prices)):
        # Lógica de estados con prioridad direccional
        if slopes[i] > slope_thresholds[i]:
            base_state = 1  # ALCISTA
        elif slopes[i] < -slope_thresholds[i]:
            base_state = -1 # BAJISTA
        else:
            base_state = 0  # LATERAL

        if prices[i] > upper_risk[i]:
            states[i] = 2  # RIESGO
        elif prices[i] < lower_risk[i] and base_state != -1:
            states[i] = 2  # RIESGO si no es bajista confirmado
        else:
            states[i] = base_state
    
    filtered_states = states.copy()
    for i in range(1, len(states) - 1):
        if states[i] != states[i-1] and states[i] != states[i+1]:
            filtered_states[i] = states[i-1]
            
    return filtered_states, trend_recalc, upper_risk, lower_risk

# ============= PROCESAMIENTO =============
@st.cache_data(ttl=3600)
def download_and_process_data(ticker, period="2y", interval="1wk",
                               atr_period=14, atr_multiplier=1.5,
                               use_adaptive=True, use_dynamic=True,
                               projection_periods=4, projection_lookback=10,
                               projection_degree=2):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty or len(data) < 30: return None, "Datos insuficientes"
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        
        prices = data['Close'].values
        x = np.arange(len(prices))
        trend_init = nadaraya_watson_kernel(x, prices, bandwidth=8)
        atr = calculate_atr_improved(data, period=atr_period)
        macd_v, macd_v_signal = calculate_macd_v(data)
        
        states, trend_refined, upper_risk, lower_risk = classify_trend_state(
            prices, trend_init, atr, atr_multiplier, use_adaptive, use_dynamic
        )
        
        x_future, y_future, conf = project_trend(
            x, trend_refined, projection_periods, projection_lookback, projection_degree
        )
        
        atr_current = atr[-1] if not np.isnan(atr[-1]) else np.nanmean(atr[-5:])
        u_f, l_f = project_atr_bands(y_future, atr_current, atr_multiplier)
        
        results = pd.DataFrame({
            'Date': data.index, 'Close': prices, 'Trend': trend_refined,
            'ATR': atr, 'Upper_Risk': upper_risk, 'Lower_Risk': lower_risk,
            'State': states, 'MACD_V': macd_v.values, 'MACD_V_Signal': macd_v_signal.values,
            'State_Name': pd.Series(states).map({-1: 'BAJISTA', 0: 'LATERAL', 1: 'ALCISTA', 2: 'RIESGO'}).values
        })
        
        future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=projection_periods, freq='W')
        projection_df = pd.DataFrame({'Date': future_dates, 'Trend_Projection': y_future, 'Upper_Projection': u_f, 'Lower_Projection': l_f})
        
        metrics = {'projection_confidence': conf, 'projection_change_pct': ((y_future[-1] - prices[-1]) / prices[-1] * 100), 'projection_target': y_future[-1]}
        return (results, projection_df, metrics), None
    except Exception as e: return None, str(e)

# ============= VISUALIZACIÓN =============
def plot_atr_analysis_with_projection(results, projection_df, metrics, ticker):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 14), facecolor='#0E1117')
    gs = fig.add_gridspec(4, 1, height_ratios=[4, 1, 1, 0.8], hspace=0.35)
    
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#1A1D29')
    
    # Zonas
    ax1.fill_between(range(len(results)), results['Lower_Risk'], results['Upper_Risk'], color='#FFB86C', alpha=0.08)
    
    proj_x = np.arange(len(results)-1, len(results) + len(projection_df))
    p_u = np.concatenate([[results['Upper_Risk'].iloc[-1]], projection_df['Upper_Projection']])
    p_l = np.concatenate([[results['Lower_Risk'].iloc[-1]], projection_df['Lower_Projection']])
    ax1.fill_between(proj_x, p_l, p_u, color='#FFB86C', alpha=0.15, hatch='//')
    
    ax1.plot(results['Trend'], color='#00D9FF', linewidth=2, label='Tendencia')
    
    # Proyección fluida
    p_t = np.concatenate([[results['Trend'].iloc[-1]], projection_df['Trend_Projection']])
    ax1.plot(proj_x, p_t, color='#FF6B6B', linewidth=3, linestyle='--', label='Proyección (Continuidad OK)')
    
    ax1.plot(results['Close'], color='white', alpha=0.4, linewidth=1)
    
    # Scatter de estados
    colors = {1: '#4ECDC4', -1: '#EE5A6F', 0: '#95A5A6', 2: '#FF6B6B'}
    for s, c in colors.items():
        m = results['State'] == s
        ax1.scatter(results[m].index, results[m]['Close'], c=c, s=50, edgecolors='white')

    ax1.axvline(len(results)-1, color='yellow', linestyle=':', alpha=0.5)
    ax1.legend()
    
    # Subplot MACD-V
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(results['MACD_V'], color='#4ECDC4')
    ax3.axhline(0, color='white', alpha=0.2)
    
    # Subplot Estados (Semáforo corregido)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.set_facecolor('#1A1D29')
    regimes = ['BAJISTA', 'LATERAL', 'ALCISTA', 'RIESGO']
    for i, r in enumerate(regimes):
        m = results['State_Name'] == r
        if m.any(): ax4.scatter(results[m].index, [i]*m.sum(), c=colors[{-1: -1, 0: 0, 1: 1, 2: 2}[results[m]['State'].iloc[0]]], s=80)
    ax4.set_yticks(range(4))
    ax4.set_yticklabels(regimes)
    
    return fig

def main_app():
    st.title("📊 ATR Trend Analyzer Pro")
    with st.sidebar:
        ticker = st.text_input("Ticker", "JEPQ").upper()
        p_deg = st.selectbox("Modelo Proyección", [1, 2, 3], index=1)
        btn = st.button("🚀 ANALIZAR")

    if btn or 'results' in st.session_state:
        if btn:
            res, err = download_and_process_data(ticker, projection_degree=p_deg)
            if err: st.error(err)
            else: st.session_state.update({'results': res[0], 'proj': res[1], 'met': res[2], 't': ticker})

        if 'results' in st.session_state:
            r, p, m, t = st.session_state['results'], st.session_state['proj'], st.session_state['met'], st.session_state['t']
            c1, c2, c3 = st.columns(3)
            c1.metric("ESTADO", r['State_Name'].iloc[-1])
            c2.metric("TARGET", f"${m['projection_target']:.2f}")
            c3.metric("CONFIANZA", f"{m['projection_confidence']*100:.0f}%")
            st.pyplot(plot_atr_analysis_with_projection(r, p, m, t))

if __name__ == "__main__":
    if check_password(): main_app()
