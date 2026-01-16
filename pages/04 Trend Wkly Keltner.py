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
    """Umbral de pendiente adaptativo - MEJORADO"""
    if len(trend) < lookback:
        return np.full_like(trend, np.abs(trend).mean() * 0.001)
    
    recent_changes = np.abs(np.diff(trend[-lookback:]))
    # Usar percentil más bajo para capturar mejor las tendencias bajistas
    threshold_value = np.percentile(recent_changes, 25)  # Antes era 40
    
    min_threshold = np.abs(trend) * 0.0008  # Reducido de 0.0015
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
    """Sistema de clasificación de tendencia de 4 estados - MEJORADO"""
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
        slope_thresholds = np.abs(trend_recalc) * 0.001
    
    states = np.zeros(len(prices), dtype=int)
    
    for i in range(len(prices)):
        price_vs_upper = prices[i] > upper_risk[i]
        price_vs_lower = prices[i] < lower_risk[i]
        
        # RIESGO: precio fuera de bandas ATR
        if price_vs_upper or price_vs_lower:
            # Verificar si hay movimiento extremo
            if i > 0:
                price_change = abs((prices[i] - prices[i-1]) / prices[i-1])
                if price_change > 0.03:  # Movimiento >3%
                    states[i] = 2
                    continue
            
            # Si está fuera de bandas pero la pendiente indica dirección
            if slopes[i] > slope_thresholds[i] * 1.5:  # Alcista fuerte
                states[i] = 1
            elif slopes[i] < -slope_thresholds[i] * 1.5:  # Bajista fuerte
                states[i] = -1
            else:
                states[i] = 2  # Riesgo
        
        # ALCISTA: pendiente positiva significativa
        elif slopes[i] > slope_thresholds[i]:
            states[i] = 1
        
        # BAJISTA: pendiente negativa significativa
        elif slopes[i] < -slope_thresholds[i]:
            states[i] = -1
        
        # LATERAL: sin dirección clara
        else:
            states[i] = 0
    
    # Filtro de confirmación mejorado
    filtered_states = states.copy()
    window = 2
    for i in range(window, len(states) - window):
        # Si el estado es diferente a los vecinos, verificar contexto
        if states[i] != states[i-1] and states[i] != states[i+1]:
            # Contar estados en ventana
            window_states = states[i-window:i+window+1]
            most_common = np.bincount(window_states + 1).argmax() - 1
            filtered_states[i] = most_common
    
    return filtered_states, trend_recalc, upper_risk, lower_risk

# ============= DESCARGA Y PROCESAMIENTO DE DATOS =============
@st.cache_data(ttl=3600)
def download_and_process_data(ticker, period="2y", interval="1wk",
                               atr_period=14, atr_multiplier=1.5,
                               use_adaptive=True, use_dynamic=True,
                               projection_periods=4, projection_lookback=10,
                               projection_degree=2):
    """Descarga datos y calcula todos los indicadores + PROYECCIÓN"""
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if data.empty or len(data) < 30:
            return None, "Datos insuficientes para el análisis"
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        prices = data['Close'].values
        x = np.arange(len(prices))
        
        trend = nadaraya_watson_kernel(x, prices, bandwidth=8)
        atr = calculate_atr_improved(data, period=atr_period)
        
        macd_v, macd_v_signal = calculate_macd_v(data, fast_len=12, slow_len=26, 
                                                  signal_len=9, atr_len=26)
        
        states, trend_refined, upper_risk, lower_risk = classify_trend_state(
            prices, trend, atr, 
            atr_multiplier=atr_multiplier,
            use_adaptive_bandwidth=use_adaptive,
            use_dynamic_threshold=use_dynamic
        )
        
        # PROYECCIÓN
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

# ============= VISUALIZACIÓN CON MACD-V =============
def plot_atr_analysis_with_projection(results, projection_df, metrics, ticker):
    """Gráfico avanzado con proyección + MACD-V - SIN GAP"""
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 14), facecolor='#0E1117')
    gs = fig.add_gridspec(4, 1, height_ratios=[4, 1, 1, 0.8], hspace=0.35)
    
    colors_map = {1: '#4ECDC4', -1: '#EE5A6F', 0: '#95A5A6', 2: '#FF6B6B'}
    state_names = {1: 'ALCISTA', -1: 'BAJISTA', 0: 'LATERAL', 2: 'RIESGO'}
    
    # ============= SUBPLOT 1: Precio y Proyección - SIN GAP =============
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#1A1D29')
    
    # Zona histórica ATR
    ax1.fill_between(range(len(results)), results['Lower_Risk'], results['Upper_Risk'],
                     color='#FFB86C', alpha=0.08, label='Zona Histórica (ATR)', zorder=1)
    
    # Zona proyectada ATR - SIN GAP
    proj_x_start = len(results) - 1
    proj_x = np.arange(proj_x_start, proj_x_start + len(projection_df))
    
    # Conectar suavemente desde el último punto
    proj_upper = np.concatenate([[results['Upper_Risk'].iloc[-1]], 
                                 projection_df['Upper_Projection'].values[:-1]])
    proj_lower = np.concatenate([[results['Lower_Risk'].iloc[-1]], 
                                 projection_df['Lower_Projection'].values[:-1]])
    
    ax1.fill_between(proj_x, proj_lower, proj_upper,
                     color='#FFB86C', alpha=0.15, 
                     label=f'Zona Proyectada (Conf: {metrics["projection_confidence"]*100:.0f}%)', 
                     zorder=1, hatch='//')
    
    # Líneas de riesgo
    ax1.plot(results['Upper_Risk'], color='#FFB86C', linewidth=1.5, 
             linestyle='--', alpha=0.6, zorder=2)
    ax1.plot(results['Lower_Risk'], color='#FFB86C', linewidth=1.5, 
             linestyle='--', alpha=0.6, zorder=2)
    ax1.plot(proj_x, proj_upper, color='#FF6B6B', linewidth=2, 
             linestyle=':', alpha=0.8, zorder=2)
    ax1.plot(proj_x, proj_lower, color='#FF6B6B', linewidth=2, 
             linestyle=':', alpha=0.8, zorder=2)
    
    # Tendencia
    ax1.plot(results['Trend'], color='#00D9FF', linewidth=2.5, 
             label='Tendencia Kernel', zorder=3)
    
    # Proyección - SIN GAP
    proj_trend = np.concatenate([[results['Trend'].iloc[-1]], 
                                 projection_df['Trend_Projection'].values])
    proj_x_trend = np.arange(proj_x_start, proj_x_start + len(projection_df) + 1)
    
    ax1.plot(proj_x_trend, proj_trend, color='#FF6B6B', linewidth=3, 
             linestyle='--', label=f'Proyección ({len(projection_df)} periodos)', zorder=4,
             marker='o', markersize=8, markerfacecolor='#FF6B6B', 
             markeredgecolor='white', markeredgewidth=2)
    
    # Precio
    ax1.plot(results['Close'], color='#FFFFFF', alpha=0.15, linewidth=6, zorder=4)
    ax1.plot(results['Close'], color='#E0E0E0', alpha=0.4, linewidth=3, zorder=5)
    ax1.plot(results['Close'], color='#FFFFFF', linewidth=1.5, 
             label='Precio', zorder=6)
    
    # Scatter por estado
    for state_val, color in colors_map.items():
        mask = results['State'] == state_val
        if mask.sum() > 0:
            ax1.scatter(results[mask].index, results[mask]['Close'],
                       c=color, s=60, alpha=0.7, edgecolors='white',
                       linewidth=1, label=state_names[state_val], zorder=7)
    
    # Último punto
    last_idx = len(results) - 1
    last_state = results['State'].iloc[-1]
    ax1.scatter(last_idx, results['Close'].iloc[-1],
               facecolors=colors_map[last_state], edgecolors='white',
               s=300, linewidth=3, marker='o', zorder=10)
    
    # Punto proyección
    proj_last_idx = proj_x_trend[-1]
    ax1.scatter(proj_last_idx, projection_df['Trend_Projection'].iloc[-1],
               facecolors='#FF6B6B', edgecolors='yellow',
               s=400, linewidth=4, marker='*', zorder=11)
    
    # Anotaciones
    ax1.annotate(f'{results["State_Name"].iloc[-1]}\n${results["Close"].iloc[-1]:.2f}',
                xy=(last_idx, results['Close'].iloc[-1]),
                xytext=(-80, -40), textcoords='offset points',
                fontsize=12, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.8', 
                         facecolor=colors_map[last_state],
                         alpha=0.95, edgecolor='white', linewidth=2),
                arrowprops=dict(arrowstyle='->', lw=2.5,
                               color=colors_map[last_state]),
                zorder=11)
    
    change_pct = metrics['projection_change_pct']
    arrow_color = '#4ECDC4' if change_pct > 0 else '#FF6B6B'
    ax1.annotate(f'Target\n${projection_df["Trend_Projection"].iloc[-1]:.2f}\n({change_pct:+.1f}%)',
                xy=(proj_last_idx, projection_df['Trend_Projection'].iloc[-1]),
                xytext=(30, 30), textcoords='offset points',
                fontsize=12, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.8', 
                         facecolor=arrow_color,
                         alpha=0.95, edgecolor='yellow', linewidth=3),
                arrowprops=dict(arrowstyle='->', lw=3, color='yellow'),
                zorder=12)
    
    # Línea HOY
    ax1.axvline(x=last_idx, color='yellow', linestyle=':', 
                linewidth=2, alpha=0.5, zorder=1)
    ax1.text(last_idx, ax1.get_ylim()[1]*0.98, 'HOY', 
            ha='center', va='top', fontsize=10, fontweight='bold',
            color='yellow', bbox=dict(boxstyle='round,pad=0.4',
                                     facecolor='black', alpha=0.7))
    
    ax1.text(0.5, 1.10, f'{ticker}', transform=ax1.transAxes, 
             fontsize=28, fontweight='bold', ha='center', color='#FFFFFF')
    ax1.text(0.5, 1.05, 'ATR Trend Analysis + Projection (Mejorado)', transform=ax1.transAxes, 
             fontsize=13, style='italic', ha='center', color='#8E93A1')
    
    ax1.set_ylabel('Precio ($)', fontsize=14, fontweight='bold', color='#FFFFFF')
    legend = ax1.legend(loc='upper left', fontsize=9, framealpha=0.95, ncol=2,
                       edgecolor='#00D9FF', fancybox=True)
    legend.get_frame().set_facecolor('#1A1D29')
    legend.get_frame().set_linewidth(2)
    ax1.grid(True, alpha=0.08, linestyle='-', linewidth=0.8, color='#FFFFFF')
    ax1.tick_params(labelsize=10, colors='#B0B0B0')
    for spine in ax1.spines.values():
        spine.set_color('#2D3142')
        spine.set_linewidth(2)
    
    # ============= SUBPLOT 2: ATR =============
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_facecolor('#1A1D29')
    
    ax2.plot(results['ATR'], color='#BD93F9', linewidth=2, label='ATR')
    ax2.fill_between(range(len(results)), 0, results['ATR'],
                     color='#BD93F9', alpha=0.2)
    ax2.axhline(y=metrics['atr_current'], xmin=last_idx/len(results), 
                color='#BD93F9', linestyle='--', linewidth=2, alpha=0.6)
    
    ax2.set_ylabel('ATR', fontsize=12, fontweight='bold', color='#FFFFFF')
    ax2.legend(loc='upper left', fontsize=9, framealpha=0.95)
    ax2.grid(True, alpha=0.08)
    ax2.tick_params(labelsize=9, colors='#B0B0B0')
    for spine in ax2.spines.values():
        spine.set_color('#2D3142')
        spine.set_linewidth(2)
    
    # ============= SUBPLOT 3: MACD-V =============
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.set_facecolor('#1A1D29')
    
    for i in range(1, len(results)):
        if pd.notna(results['MACD_V'].iloc[i-1]) and pd.notna(results['MACD_V'].iloc[i]):
            y2 = results['MACD_V'].iloc[i]
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
            ax3.plot([i-1, i], 
                    [results['MACD_V'].iloc[i-1], y2], 
                    color=color, linewidth=width, alpha=0.95, zorder=5)
    
    if 'MACD_V_Signal' in results.columns:
        ax3.plot(results['MACD_V_Signal'], color='#FFB86C', 
                linewidth=1.5, alpha=0.5, linestyle='--', 
                label='Signal', zorder=3)
    
    ax3.axhline(y=150, color='#FF6B6B', linestyle='--', linewidth=2, alpha=0.8)
    ax3.axhline(y=50, color='#4ECDC4', linestyle=':', linewidth=1.5, alpha=0.7)
    ax3.axhline(y=0, color='#8E93A1', linestyle='-', linewidth=1.5, alpha=0.7)
    ax3.axhline(y=-50, color='#EE5A6F', linestyle=':', linewidth=1.5, alpha=0.7)
    ax3.axhline(y=-150, color='#FF6B6B', linestyle='--', linewidth=2, alpha=0.8)
    
    ax3.fill_between(range(len(results)), 150, 300, alpha=0.12, color='#FF6B6B', zorder=0)
    ax3.fill_between(range(len(results)), -300, -150, alpha=0.12, color='#FF6B6B', zorder=0)
    ax3.fill_between(range(len(results)), -50, 50, alpha=0.08, color='#95A5A6', zorder=0)
    
    current = results.iloc[-1]
    macd_color = '#FF6B6B' if abs(current['MACD_V']) > 150 else '#4ECDC4' if current['MACD_V'] > 50 else '#EE5A6F' if current['MACD_V'] < -50 else '#95A5A6'
    ax3.text(0.02, 0.88, f'MACD-V: {current["MACD_V"]:.1f}', 
            transform=ax3.transAxes, fontsize=13, fontweight='bold', 
            color='white', verticalalignment='top', 
            bbox=dict(boxstyle='round,pad=0.7', facecolor=macd_color, 
                     alpha=0.95, edgecolor='white', linewidth=2.5))
    
    ax3.set_ylabel('MACD-V (raw)', fontsize=12, fontweight='bold', color='#FFFFFF')
    ax3.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax3.grid(True, alpha=0.08, linestyle=':', linewidth=1, color='#FFFFFF')
    ax3.tick_params(labelsize=9, colors='#B0B0B0')
    for spine in ax3.spines.values():
        spine.set_color('#2D3142')
        spine.set_linewidth(2)
    
    # ============= SUBPLOT 4: Semáforo =============
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.set_facecolor('#1A1D29')
    
    regime_order = ['BAJISTA', 'LATERAL', 'ALCISTA', 'RIESGO']
    for regime_name in regime_order:
        state_map = {'BAJISTA': -1, 'LATERAL': 0, 'ALCISTA': 1, 'RIESGO': 2}
        mask = results['State'] == state_map[regime_name]
        if mask.sum() > 0:
            color = colors_map[state_map[regime_name]]
            regime_num = regime_order.index(regime_name)
            ax4.scatter(results[mask].index, [regime_num] * mask.sum(), 
                       c=color, alpha=0.95, s=110, edgecolors='white', 
                       linewidth=1.8, zorder=4)
    
    ax4.set_ylabel('Estado', fontsize=12, fontweight='bold', color='#FFFFFF')
    ax4.set_yticks(range(4))
    ax4.set_yticklabels(regime_order, fontsize=11, fontweight='bold', color='#E0E0E0')
    ax4.set_xlabel('Periodo', fontsize=12, fontweight='bold', color='#FFFFFF')
    ax4.grid(True, alpha=0.08, linestyle='-', linewidth=1, color='#FFFFFF', axis='x')
    ax4.tick_params(labelsize=9, colors='#B0B0B0')
    for spine in ax4.spines.values():
        spine.set_color('#2D3142')
        spine.set_linewidth(2)
    
    plt.tight_layout()
    return fig

# ============= INTERFAZ STREAMLIT =============
def main_app():
    """Aplicación principal"""
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
    .stMetric [data-testid="stMetricDelta"] {
        color: #4ECDC4 !important;
        font-size: 16px !important;
        font-weight: 600 !important;
    }
    .projection-box {
        background: linear-gradient(135deg, #FF6B6B 0%, #FFB86C 100%);
        padding: 20px;
        border-radius: 12px;
        border: 3px solid #FFD700;
        box-shadow: 0 4px 20px rgba(255, 215, 0, 0.4);
        color: white;
        text-align: center;
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
    
    st.title("📊 ATR Trend Analyzer Pro + Proyección (Mejorado)")
    st.markdown("✨ **Mejoras**: Estados BAJISTA visibles + Proyección sin gap")
    st.markdown("---")
    
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 20px; 
                    background: linear-gradient(135deg, #4ECDC4 0%, #00D9FF 100%); 
                    border-radius: 15px; margin-bottom: 20px;'>
            <h2 style='color: white; margin: 0;'>⚙️ Configuración</h2>
        </div>
        """, unsafe_allow_html=True)
        
        ticker = st.text_input("Ticker Symbol", value="AAPL", 
                              help="Símbolo del activo").upper()
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            period = st.selectbox("Periodo", 
                                 ["1y", "2y", "5y", "10y"],
                                 index=1)
        with col2:
            interval = st.selectbox("Intervalo",
                                   ["1d", "1wk", "1mo"],
                                   index=1)
        
        st.markdown("---")
        
        st.subheader("Parámetros ATR")
        atr_period = st.slider("Periodo ATR", 7, 28, 14, 1)
        atr_multiplier = st.slider("Multiplicador ATR", 0.5, 3.0, 1.5, 0.1)
        
        st.markdown("---")
        
        st.subheader("🔮 Proyección")
        projection_periods = st.slider("Períodos a Proyectar", 1, 12, 4, 1)
        projection_lookback = st.slider("Puntos de Inercia", 5, 20, 10, 1)
        projection_degree = st.selectbox("Tipo",
                                        options=[1, 2, 3],
                                        format_func=lambda x: {1: "Lineal", 2: "Cuadrática", 3: "Cúbica"}[x],
                                        index=1)
        
        st.markdown("---")
        
        st.subheader("Algoritmos")
        use_adaptive = st.checkbox("Bandwidth Adaptativo", value=True)
        use_dynamic = st.checkbox("Umbral Dinámico", value=True)
        
        st.markdown("---")
        
        analyze_btn = st.button("🚀 ANALIZAR + PROYECTAR", type="primary", 
                               use_container_width=True)
        
        st.markdown("---")
        
        with st.expander("ℹ️ Metodología"):
            st.markdown("""
            **Estados (MEJORADO):**
            - 🟢 ALCISTA: Pendiente >umbral
            - 🔴 BAJISTA: Pendiente <-umbral
            - ⚫ LATERAL: Sin dirección
            - 🟠 RIESGO: Precio fuera ATR
            
            **Mejoras v2:**
            - Umbral más sensible (25th percentile)
            - Mejor detección de bajistas
            - Proyección sin gap visual
            - Filtrado de estados mejorado
            """)
    
    if analyze_btn:
        with st.spinner(f"Analizando {ticker}..."):
            result = download_and_process_data(
                ticker, period, interval,
                atr_period, atr_multiplier,
                use_adaptive, use_dynamic,
                projection_periods, projection_lookback,
                projection_degree
            )
            
            if result[1]:
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
            st.success(f"✅ {len(results)} periodos + {len(projection_df)} proyectados")
    
    if 'results' in st.session_state:
        results = st.session_state['results']
        projection_df = st.session_state['projection_df']
        metrics = st.session_state['metrics']
        ticker = st.session_state['ticker']
        
        st.markdown("---")
        st.markdown("### 📈 Estado Actual")
        
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
            st.metric("ATR", f"${current['ATR']:.2f}")
        
        with col4:
            trend_pos = ((current['Close'] - current['Lower_Risk']) / 
                        (current['Upper_Risk'] - current['Lower_Risk']) * 100)
            st.metric("Pos. ATR", f"{trend_pos:.0f}%")
        
        with col5:
            distance = ((current['Close'] - current['Trend']) / current['Trend'] * 100)
            st.metric("Vs Tend.", f"{distance:+.2f}%")
        
        st.markdown("---")
        st.markdown("### 🔮 Proyección")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            proj_target = projection_df['Trend_Projection'].iloc[-1]
            st.markdown(f"""
            <div class='projection-box'>
                <div style='font-size: 14px; font-weight: bold; color: #FFD700;'>
                    TARGET
                </div>
                <div style='font-size: 28px; font-weight: bold; margin-top: 10px;'>
                    ${proj_target:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            proj_change = metrics['projection_change_pct']
            color = '#4ECDC4' if proj_change > 0 else '#FF6B6B'
            arrow = '↗' if proj_change > 0 else '↘'
            st.markdown(f"""
            <div class='projection-box'>
                <div style='font-size: 14px; font-weight: bold; color: #FFD700;'>
                    CAMBIO
                </div>
                <div style='font-size: 28px; font-weight: bold; margin-top: 10px; color: {color};'>
                    {arrow} {proj_change:+.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            conf = metrics['projection_confidence']
            conf_color = '#4ECDC4' if conf > 0.7 else '#FFB86C' if conf > 0.4 else '#FF6B6B'
            conf_label = 'Alta' if conf > 0.7 else 'Media' if conf > 0.4 else 'Baja'
            st.markdown(f"""
            <div class='projection-box'>
                <div style='font-size: 14px; font-weight: bold; color: #FFD700;'>
                    CONFIANZA
                </div>
                <div style='font-size: 28px; font-weight: bold; margin-top: 10px; color: {conf_color};'>
                    {conf*100:.0f}%
                </div>
                <div style='font-size: 14px; margin-top: 5px;'>
                    {conf_label}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            date_target = projection_df['Date'].iloc[-1].strftime('%Y-%m-%d')
            st.markdown(f"""
            <div class='projection-box'>
                <div style='font-size: 14px; font-weight: bold; color: #FFD700;'>
                    FECHA
                </div>
                <div style='font-size: 20px; font-weight: bold; margin-top: 10px;'>
                    {date_target}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📊 Análisis Completo")
        
        fig = plot_atr_analysis_with_projection(results, projection_df, metrics, ticker)
        st.pyplot(fig)
        
        st.markdown("---")
        st.markdown("### 📉 Estadísticas")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Distribución Estados:**")
            state_counts = results['State_Name'].value_counts()
            for state in ['ALCISTA', 'BAJISTA', 'LATERAL', 'RIESGO']:
                count = state_counts.get(state, 0)
                pct = (count / len(results) * 100) if len(results) > 0 else 0
                emoji = {'ALCISTA': '🟢', 'BAJISTA': '🔴', 'LATERAL': '⚫', 'RIESGO': '🟠'}
                st.write(f"{emoji[state]} {state}: {count} ({pct:.1f}%)")
        
        with col2:
            st.markdown("**Precio:**")
            st.write(f"- Máximo: ${results['Close'].max():.2f}")
            st.write(f"- Mínimo: ${results['Close'].min():.2f}")
            st.write(f"- Promedio: ${results['Close'].mean():.2f}")
            st.write(f"- Volatilidad: {results['Close'].std():.2f}")
        
        with col3:
            st.markdown("**ATR:**")
            st.write(f"- Promedio: ${results['ATR'].mean():.2f}")
            st.write(f"- Máximo: ${results['ATR'].max():.2f}")
            st.write(f"- Actual: ${current['ATR']:.2f}")
        
        st.markdown("---")
        st.markdown("### 🤖 Interpretación")
        
        interpretation = f"""
        **Análisis de {ticker}:**
        
        - **Estado Actual**: {current['State_Name']}
        - **Precio**: ${current['Close']:.2f} ({distance:+.1f}% vs tendencia)
        - **Proyección**: ${proj_target:.2f} ({proj_change:+.1f}%)
        - **Confianza**: {conf_label} ({conf*100:.0f}%)
        - **MACD-V**: {current['MACD_V']:.1f}
        """
        
        if proj_change > 5:
            interpretation += "\n- ⚠️ Proyección alcista significativa (>5%)"
        elif proj_change < -5:
            interpretation += "\n- ⚠️ Proyección bajista significativa (<-5%)"
        
        if conf < 0.5:
            interpretation += "\n- ⚠️ Baja confianza, mercado volátil"
        
        if current['State_Name'] == 'RIESGO':
            interpretation += "\n- 🚨 Precio sobreextendido, posible reversión"
        
        st.info(interpretation)
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            csv_hist = results.to_csv(index=False)
            st.download_button(
                "📥 Datos Históricos (CSV)",
                data=csv_hist,
                file_name=f"atr_hist_{ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            csv_proj = projection_df.to_csv(index=False)
            st.download_button(
                "🔮 Proyección (CSV)",
                data=csv_proj,
                file_name=f"atr_proj_{ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    else:
        st.markdown("""
        <div style='text-align: center; padding: 60px 20px;
                    background: linear-gradient(135deg, #1A1D29 0%, #2D3142 100%);
                    border-radius: 20px; border: 3px solid #4ECDC4;
                    box-shadow: 0 8px 30px rgba(78, 205, 196, 0.3);
                    margin-top: 40px;'>
            <h2 style='color: #4ECDC4; margin: 0;'>
                👋 ATR Trend Analyzer Pro + Proyección
            </h2>
            <p style='color: #B0B0B0; font-size: 18px; margin: 20px 0;'>
                Versión Mejorada v2.0
            </p>
            <p style='color: #8E93A1; font-size: 14px;'>
                ✨ Estados BAJISTA visibles<br>
                ✨ Proyección sin gap<br>
                ✨ Detección mejorada
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    if check_password():
        main_app()
    else:
        st.stop()
