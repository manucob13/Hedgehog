# pages/ZScore_MACDV_Analyzer.py
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats
import warnings
from utils.utils import check_password

warnings.filterwarnings('ignore')

REGIME_COLORS_ZSCORE = {
    'SOBRECOMPRA': '#FF6B6B',
    'SOBREVENTA': '#9D4EDD',
    'ALCISTA': '#4ECDC4',
    'BAJISTA': '#EE5A6F',
    'RANGO': '#FFD93D'
}

def calculate_atr(df, period=26):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()

def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def calculate_sma(data, period):
    return data.rolling(window=period).mean()

def calculate_donchian(df, period=20):
    upper = df['High'].rolling(window=period).max()
    lower = df['Low'].rolling(window=period).min()
    middle = (upper + lower) / 2
    return upper, middle, lower

def calculate_z_score_macdv(df, fast=12, slow=26, signal=9, z_window=20):
    h_l = df['High'] - df['Low']
    h_pc = np.abs(df['High'] - df['Close'].shift(1))
    l_pc = np.abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    atr = tr.rolling(window=slow).mean()
    
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd_v = ((ema_fast - ema_slow) / atr) * 100
    
    signal_line = macd_v.ewm(span=signal, adjust=False).mean()
    momentum_dist = macd_v - signal_line
    
    m_mean = momentum_dist.rolling(window=z_window).mean()
    m_std = momentum_dist.rolling(window=z_window).std()
    z_score = (momentum_dist - m_mean) / m_std
    
    def rolling_kurtosis(series, window):
        return series.rolling(window).apply(lambda x: stats.kurtosis(x, fisher=True, nan_policy='omit'), raw=False)
    
    kurtosis = rolling_kurtosis(momentum_dist, z_window)
    kurtosis_factor = 1 + (kurtosis / 10).clip(-0.5, 0.5)
    z_score_adjusted = z_score / kurtosis_factor
    
    df['MACD_V'] = macd_v
    df['MACD_V_Signal'] = signal_line
    df['Momentum_Dist'] = momentum_dist
    df['Z_Score'] = z_score
    df['Z_Score_Adjusted'] = z_score_adjusted
    df['Kurtosis'] = kurtosis
    
    return df

def classify_regime_zscore(df):
    regimes = []
    
    for idx, row in df.iterrows():
        z = row['Z_Score_Adjusted']
        price = row['Close']
        sma50 = row['SMA_50']
        
        if pd.isna(z) or pd.isna(price) or pd.isna(sma50):
            regimes.append('RANGO')
            continue
        
        trend_up = price > sma50
        trend_down = price < sma50
        
        if trend_up:
            if z > 2.0:
                regime = 'SOBRECOMPRA'
            elif z > 0.75:
                regime = 'ALCISTA'
            else:
                regime = 'RANGO'
        elif trend_down:
            if z < -2.0:
                regime = 'SOBREVENTA'
            elif z < -0.75:
                regime = 'BAJISTA'
            else:
                regime = 'RANGO'
        else:
            regime = 'RANGO'
        
        regimes.append(regime)
    
    return regimes

@st.cache_data(ttl=timedelta(hours=1))
def download_weekly_data(ticker, start_date=None, years_back=None):
    try:
        if start_date is None and years_back is None:
            years_back = 7
        
        if start_date is None:
            if years_back is None:
                start_date = '1990-01-01'
            else:
                start_date = (datetime.now() - timedelta(days=365*years_back)).strftime('%Y-%m-%d')
        else:
            start_date = start_date if isinstance(start_date, str) else start_date.strftime('%Y-%m-%d')
        
        data = yf.download(ticker, start=start_date, interval='1wk', progress=False, auto_adjust=True, actions=False, timeout=10)
        
        if data is None or data.empty:
            st.error(f"⚠️ No se encontraron datos para '{ticker}'")
            return None
        
        if len(data) < 50:
            st.warning(f"⚠️ Datos insuficientes ({len(data)} semanas). Mínimo: 50")
            return None
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        required_columns = ['Close', 'Open', 'High', 'Low', 'Volume']
        missing_cols = [col for col in required_columns if col not in data.columns]
        
        if missing_cols:
            st.error(f"❌ Faltan columnas: {missing_cols}")
            return None
        
        df = pd.DataFrame({
            'Close': data['Close'].squeeze(),
            'Open': data['Open'].squeeze(),
            'High': data['High'].squeeze(),
            'Low': data['Low'].squeeze(),
            'Volume': data['Volume'].squeeze()
        }, index=data.index)
        
        df = df.dropna(subset=['Close', 'High', 'Low'])
        
        if len(df) < 50:
            st.warning(f"⚠️ Después de limpieza: {len(df)} períodos (mínimo: 50)")
            return None
        
        df['SMA_20'] = calculate_sma(df['Close'], 20)
        df['SMA_50'] = calculate_sma(df['Close'], 50)
        df['Donchian_Upper'], df['Donchian_Middle'], df['Donchian_Lower'] = calculate_donchian(df, period=20)
        df = calculate_z_score_macdv(df, fast=12, slow=26, signal=9, z_window=20)
        df = df.dropna()
        
        if len(df) < 30:
            st.warning(f"⚠️ Después de indicadores: {len(df)} períodos")
            return None
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error descargando '{ticker}': {str(e)}")
        return None

def plot_zscore_dashboard(df_recent, ticker):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(24, 16), facecolor='#0E1117')
    gs = fig.add_gridspec(5, 1, height_ratios=[3.5, 1.2, 1.2, 1, 1], hspace=0.4)
    
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#1A1D29')
    
    ax1.fill_between(df_recent.index, df_recent['Donchian_Upper'], df_recent['Donchian_Lower'], alpha=0.08, color='#00D9FF', zorder=0)
    ax1.plot(df_recent.index, df_recent['Donchian_Upper'], color='#00D9FF', alpha=0.6, linewidth=2, linestyle='--', label='Donchian Canal', zorder=2)
    ax1.plot(df_recent.index, df_recent['Donchian_Middle'], color='#00D9FF', alpha=0.8, linewidth=2.5, linestyle='-', zorder=2)
    ax1.plot(df_recent.index, df_recent['Donchian_Lower'], color='#00D9FF', alpha=0.6, linewidth=2, linestyle='--', zorder=2)
    
    ax1.plot(df_recent.index, df_recent['Close'], color='#FFFFFF', alpha=0.15, linewidth=6, zorder=1)
    ax1.plot(df_recent.index, df_recent['Close'], color='#E0E0E0', alpha=0.4, linewidth=3, zorder=2)
    ax1.plot(df_recent.index, df_recent['Close'], color='#FFFFFF', linewidth=1.5, zorder=3)
    
    for regime_name, color in REGIME_COLORS_ZSCORE.items():
        mask = df_recent['Regime_ZScore'] == regime_name
        if mask.sum() > 0:
            ax1.scatter(df_recent[mask].index, df_recent[mask]['Close'], c=color, alpha=0.15, s=220, edgecolors='none', zorder=4)
            ax1.scatter(df_recent[mask].index, df_recent[mask]['Close'], c=color, label=regime_name.replace('_', ' '), alpha=0.95, s=85, edgecolors='white', linewidth=1.5, zorder=5)
    
    ax1.plot(df_recent.index, df_recent['SMA_20'], color='#50FA7B', alpha=0.9, linewidth=2.5, label='SMA(20)', zorder=3)
    ax1.plot(df_recent.index, df_recent['SMA_50'], color='#BD93F9', alpha=0.9, linewidth=2.5, label='SMA(50)', zorder=3)
    
    current = df_recent.iloc[-1]
    ax1.scatter(current.name, current['Close'], facecolors='none', edgecolors=REGIME_COLORS_ZSCORE[current['Regime_ZScore']], s=450, linewidth=5, alpha=0.4, marker='o', zorder=9)
    ax1.scatter(current.name, current['Close'], facecolors=REGIME_COLORS_ZSCORE[current['Regime_ZScore']], edgecolors='white', s=280, linewidth=4, marker='o', zorder=10)
    
    bbox_color = REGIME_COLORS_ZSCORE[current['Regime_ZScore']]
    regime_text = current['Regime_ZScore'].replace('_', ' ')
    ax1.annotate(f'{regime_text}\n${current["Close"]:.2f}\nZ: {current["Z_Score_Adjusted"]:.2f}', xy=(current.name, current['Close']), xytext=(25, 50), textcoords='offset points', fontsize=13, fontweight='bold', color='white', bbox=dict(boxstyle='round,pad=1', facecolor=bbox_color, alpha=0.95, edgecolor='white', linewidth=3), arrowprops=dict(arrowstyle='->', lw=3, color=bbox_color), zorder=11)
    
    ax1.text(0.5, 1.10, f'{ticker}', transform=ax1.transAxes, fontsize=28, fontweight='bold', ha='center', color='#FFFFFF')
    ax1.text(0.5, 1.05, 'Z-Score MACD-V Market Regime Analysis (20w)', transform=ax1.transAxes, fontsize=13, style='italic', ha='center', color='#8E93A1')
    
    ax1.set_ylabel('Price ($)', fontsize=15, fontweight='700', color='#FFFFFF', labelpad=12)
    legend = ax1.legend(loc='upper left', fontsize=9, framealpha=0.95, ncol=4)
    legend.get_frame().set_facecolor('#1A1D29')
    ax1.grid(True, alpha=0.08, linestyle='-', linewidth=1, color='#FFFFFF')
    ax1.tick_params(labelsize=11, colors='#B0B0B0')
    
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_facecolor('#1A1D29')
    
    for i in range(1, len(df_recent)):
        if pd.notna(df_recent['Z_Score_Adjusted'].iloc[i-1]) and pd.notna(df_recent['Z_Score_Adjusted'].iloc[i]):
            x1, x2 = df_recent.index[i-1], df_recent.index[i]
            y1, y2 = df_recent['Z_Score_Adjusted'].iloc[i-1], df_recent['Z_Score_Adjusted'].iloc[i]
            
            if y2 > 2.0:
                color, width = '#FF6B6B', 3.5
            elif y2 > 0.75:
                color, width = '#4ECDC4', 3
            elif y2 > -0.75:
                color, width = '#FFD93D', 2.5
            elif y2 > -2.0:
                color, width = '#EE5A6F', 3
            else:
                color, width = '#9D4EDD', 3.5
            
            ax2.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=0.95, zorder=5)
    
    ax2.axhline(y=2.0, color='#FF6B6B', linestyle='--', linewidth=2, alpha=0.8, label='Riesgo Alcista > 2.0σ', zorder=2)
    ax2.axhline(y=0.75, color='#4ECDC4', linestyle=':', linewidth=1.5, alpha=0.7, label='Alcista > 0.75σ', zorder=2)
    ax2.axhline(y=0, color='#8E93A1', linestyle='-', linewidth=1.5, alpha=0.7, zorder=2)
    ax2.axhline(y=-0.75, color='#EE5A6F', linestyle=':', linewidth=1.5, alpha=0.7, label='Bajista < -0.75σ', zorder=2)
    ax2.axhline(y=-2.0, color='#9D4EDD', linestyle='--', linewidth=2, alpha=0.8, label='Riesgo Bajista < -2.0σ', zorder=2)
    
    ax2.fill_between(df_recent.index, 2.0, 4, alpha=0.12, color='#FF6B6B', zorder=0)
    ax2.fill_between(df_recent.index, -4, -2.0, alpha=0.12, color='#9D4EDD', zorder=0)
    ax2.fill_between(df_recent.index, -0.75, 0.75, alpha=0.08, color='#FFD93D', zorder=0)
    
    ax2.scatter(current.name, current['Z_Score_Adjusted'], facecolors=REGIME_COLORS_ZSCORE[current['Regime_ZScore']], edgecolors='white', s=220, linewidth=4, marker='o', zorder=10)
    
    ax2.text(0.02, 0.88, f'Z-Score: {current["Z_Score_Adjusted"]:.2f}σ', transform=ax2.transAxes, fontsize=13, fontweight='bold', color='white', verticalalignment='top', bbox=dict(boxstyle='round,pad=0.7', facecolor=REGIME_COLORS_ZSCORE[current['Regime_ZScore']], alpha=0.95, edgecolor='white', linewidth=2.5))
    
    ax2.set_ylabel('Z-Score\n(Adjusted)', fontsize=14, fontweight='700', color='#FFFFFF')
    ax2.set_ylim([-4, 4])
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.95, ncol=2)
    ax2.grid(True, alpha=0.08, linestyle=':', linewidth=1, color='#FFFFFF')
    ax2.tick_params(labelsize=10.5, colors='#B0B0B0')
    
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.set_facecolor('#1A1D29')
    
    ax3.plot(df_recent.index, df_recent['MACD_V'], color='#00D9FF', linewidth=2.5, label='MACD-V', zorder=3)
    ax3.plot(df_recent.index, df_recent['MACD_V_Signal'], color='#FFB86C', linewidth=2, alpha=0.7, linestyle='--', label='Signal', zorder=2)
    ax3.fill_between(df_recent.index, df_recent['MACD_V'], df_recent['MACD_V_Signal'], where=(df_recent['MACD_V'] >= df_recent['MACD_V_Signal']), color='#4ECDC4', alpha=0.2, zorder=1)
    ax3.fill_between(df_recent.index, df_recent['MACD_V'], df_recent['MACD_V_Signal'], where=(df_recent['MACD_V'] < df_recent['MACD_V_Signal']), color='#EE5A6F', alpha=0.2, zorder=1)
    
    ax3.axhline(y=0, color='#8E93A1', linestyle='-', linewidth=1.5, alpha=0.7, zorder=2)
    ax3.scatter(current.name, current['MACD_V'], facecolors='#00D9FF', edgecolors='white', s=220, linewidth=4, marker='o', zorder=10)
    
    ax3.set_ylabel('MACD-V', fontsize=14, fontweight='700', color='#FFFFFF')
    ax3.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax3.grid(True, alpha=0.08, linestyle=':', linewidth=1, color='#FFFFFF')
    ax3.tick_params(labelsize=10.5, colors='#B0B0B0')
    
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.set_facecolor('#1A1D29')
    
    ax4.plot(df_recent.index, df_recent['Kurtosis'], color='#BD93F9', linewidth=2.5, label='Kurtosis (20w)', zorder=3)
    ax4.axhline(y=0, color='#8E93A1', linestyle='-', linewidth=2, alpha=0.8, label='Normal (0)', zorder=2)
    ax4.fill_between(df_recent.index, 0, df_recent['Kurtosis'], where=(df_recent['Kurtosis'] > 0), color='#FF6B6B', alpha=0.2, label='Fat Tails', zorder=1)
    ax4.fill_between(df_recent.index, df_recent['Kurtosis'], 0, where=(df_recent['Kurtosis'] < 0), color='#4ECDC4', alpha=0.2, label='Thin Tails', zorder=1)
    
    ax4.scatter(current.name, current['Kurtosis'], facecolors='#BD93F9', edgecolors='white', s=220, linewidth=4, marker='o', zorder=10)
    
    kurt_status = 'Fat Tails' if current['Kurtosis'] > 0 else 'Thin Tails'
    kurt_color = '#FF6B6B' if current['Kurtosis'] > 0 else '#4ECDC4'
    ax4.text(0.02, 0.88, f'Kurt: {current["Kurtosis"]:.2f} ({kurt_status})', transform=ax4.transAxes, fontsize=12, fontweight='bold', color='white', verticalalignment='top', bbox=dict(boxstyle='round,pad=0.7', facecolor=kurt_color, alpha=0.95, edgecolor='white', linewidth=2.5))
    
    ax4.set_ylabel('Kurtosis', fontsize=14, fontweight='700', color='#FFFFFF')
    ax4.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax4.grid(True, alpha=0.08, linestyle=':', linewidth=1, color='#FFFFFF')
    ax4.tick_params(labelsize=10.5, colors='#B0B0B0')
    
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    ax5.set_facecolor('#1A1D29')
    
    regime_order = ['SOBREVENTA', 'BAJISTA', 'RANGO', 'ALCISTA', 'SOBRECOMPRA']
    
    for regime_name in regime_order:
        mask = df_recent['Regime_ZScore'] == regime_name
        if mask.sum() > 0:
            color = REGIME_COLORS_ZSCORE[regime_name]
            regime_num = regime_order.index(regime_name)
            ax5.scatter(df_recent[mask].index, [regime_num] * mask.sum(), c=color, alpha=0.95, s=110, edgecolors='white', linewidth=1.8, zorder=4)
    
    ax5.set_ylabel('Regime', fontsize=14, fontweight='700', color='#FFFFFF')
    ax5.set_yticks(range(5))
    ax5.set_yticklabels([r.replace('_', '\n') for r in regime_order], fontsize=10, fontweight='700', color='#E0E0E0')
    ax5.set_xlabel('Date', fontsize=15, fontweight='700', color='#FFFFFF')
    ax5.grid(True, alpha=0.08, linestyle='-', linewidth=1, color='#FFFFFF', axis='x')
    ax5.tick_params(labelsize=11, colors='#B0B0B0')
    
    plt.tight_layout()
    return fig

def zscore_analyzer_page():
    st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stMetric { background: linear-gradient(135deg, #1A1D29 0%, #2D3142 100%); padding: 20px; border-radius: 15px; border: 2px solid #00D9FF; box-shadow: 0 4px 15px rgba(0, 217, 255, 0.2); }
    .stMetric label { color: #00D9FF !important; font-weight: 700 !important; font-size: 14px !important; }
    .stMetric [data-testid="stMetricValue"] { color: #FFFFFF !important; font-size: 24px !important; font-weight: 800 !important; }
    .stButton>button { background: linear-gradient(135deg, #4ECDC4 0%, #00D9FF 100%); color: white; font-weight: 700; border: none; padding: 12px 24px; border-radius: 10px; box-shadow: 0 4px 15px rgba(78, 205, 196, 0.4); }
    h1, h2, h3 { color: #FFFFFF !important; font-weight: 800 !important; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("📊 Z-Score MACD-V Analyzer")
    st.markdown("### *Statistical Momentum with Kurtosis Adjustment (20w)*")
    st.markdown("---")
    
    col_cfg1, col_cfg2, col_cfg3, col_cfg4 = st.columns([2, 2, 2, 2])
    
    with col_cfg1:
        ticker = st.text_input("🎯 Ticker Symbol", value="AAPL").upper()
    
    with col_cfg2:
        lookback_months = st.slider("📅 Meses", 1, 24, 6, 1)
        lookback_weeks = int(lookback_months * 4.33)
    
    with col_cfg3:
        years_options = {
            "3 años": 3,
            "5 años": 5,
            "7 años": 7,
            "10 años": 10,
            "Máximo disponible": None
        }
        years_label = st.selectbox("📊 Histórico", list(years_options.keys()), index=2)
        years_back = years_options[years_label]
    
    with col_cfg4:
        use_custom = st.checkbox("📆 Fecha custom", value=False)
        if use_custom:
            start_date = st.date_input("Inicio", value=datetime(2018, 1, 1))
        else:
            start_date = None
    
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        analizar_btn = st.button("🚀 ANALIZAR", type="primary", use_container_width=True)
    
    if analizar_btn:
        with st.spinner(f"⏳ Descargando {ticker}..."):
            if not ticker or len(ticker) > 10:
                st.error("❌ Ticker inválido")
                st.stop()
            
            if use_custom and start_date:
                df_weekly = download_weekly_data(ticker, start_date=start_date)
            else:
                df_weekly = download_weekly_data(ticker, years_back=years_back)
            
            if df_weekly is None:
                st.error(f"❌ Error con {ticker}")
                st.stop()
            
            df_weekly['Regime_ZScore'] = classify_regime_zscore(df_weekly)
            
            st.session_state['ticker'] = ticker
            st.session_state['df_weekly'] = df_weekly
            st.session_state['lookback_months'] = lookback_months
            
            st.success(f"✅ {ticker} cargado")
            
            first_date = df_weekly.index[0].strftime('%Y-%m-%d')
            last_date = df_weekly.index[-1].strftime('%Y-%m-%d')
            total_weeks = len(df_weekly)
            total_years = round(total_weeks / 52, 1)
            
            if total_weeks >= 260:
                quality = "🟢 Excelente"
            elif total_weeks >= 156:
                quality = "🟡 Buena"
            else:
                quality = "🟠 Limitada"
            
            st.info(f"📊 **{total_weeks} semanas** ({total_years} años) | {first_date} → {last_date} | Calidad: {quality}")
    
    if 'df_weekly' in st.session_state:
        df_weekly = st.session_state['df_weekly']
        ticker = st.session_state['ticker']
        
        df_weekly_recent = df_weekly.tail(lookback_weeks)
        current = df_weekly.iloc[-1]
        
        st.markdown("---")
        st.markdown("<div style='text-align:center;'><h2 style='color:#4ECDC4;'>📊 ANÁLISIS Z-SCORE (20w)</h2></div>", unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        regime_emoji = {'ALCISTA': '🟢', 'BAJISTA': '🔴', 'RANGO': '🟡', 'SOBRECOMPRA': '🔴', 'SOBREVENTA': '🟣'}
        
        with col1:
            st.metric("RÉGIMEN", f"{regime_emoji[current['Regime_ZScore']]} {current['Regime_ZScore'].replace('_', ' ')}")
        
        with col2:
            price_chg = ((current['Close'] - df_weekly['Close'].iloc[-5]) / df_weekly['Close'].iloc[-5] * 100)
            st.metric("PRECIO", f"${current['Close']:.2f}", f"{price_chg:+.2f}%")
        
        with col3:
            st.metric("Z-SCORE", f"{current['Z_Score_Adjusted']:.2f}σ")
        
        with col4:
            st.metric("MACD-V", f"{current['MACD_V']:.1f}", "↑" if current['MACD_V'] > 0 else "↓")
        
        with col5:
            st.metric("CURTOSIS", f"{current['Kurtosis']:.2f}", "Fat" if current['Kurtosis'] > 0 else "Thin")
        
        with col6:
            st.metric("FECHA", current.name.strftime('%Y-%m-%d'))
        
        st.markdown("---")
        fig = plot_zscore_dashboard(df_weekly_recent, ticker)
        st.pyplot(fig)
        
        st.markdown("---")
        st.markdown("### 💡 Interpretación")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Estado Actual")
            
            if current['Z_Score_Adjusted'] > 2.0:
                st.markdown("🔴 **SOBRECOMPRA** - Agotamiento alcista, probable corrección")
            elif current['Z_Score_Adjusted'] > 0.75:
                st.markdown("🟢 **ALCISTA** - Momentum saludable")
            elif current['Z_Score_Adjusted'] > -0.75:
                st.markdown("🟡 **RANGO** - Sin tendencia clara")
            elif current['Z_Score_Adjusted'] > -2.0:
                st.markdown("🔴 **BAJISTA** - Presión vendedora")
            else:
                st.markdown("🟣 **SOBREVENTA** - Agotamiento bajista, posible rebote")
        
        with col2:
            st.markdown("#### Metodología")
            st.markdown(f"- **Ventana Z-Score:** 20 semanas (~5 meses)")
            st.markdown(f"- **Datos históricos:** {total_years} años")
            st.markdown("- **Filtro:** SMA50 confirma tendencia")
            st.markdown("- **Ajuste:** Curtosis para fat tails")
            st.markdown("- **Umbrales:** ±0.75σ y ±2.0σ")
    
    else:
        st.info("👆 Introduce un ticker y presiona ANALIZAR")

if __name__ == "__main__":
    if check_password():
        zscore_analyzer_page()
    else:
        st.error("🔒 Acceso Restringido")
