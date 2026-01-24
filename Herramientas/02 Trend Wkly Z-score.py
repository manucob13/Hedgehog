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

# CONFIGURACIÓN DE COLORES
REGIME_COLORS_ZSCORE = {
    'SOBRECOMPRA': '#FF6B6B',
    'SOBREVENTA': '#9D4EDD',
    'ALCISTA': '#4ECDC4',
    'BAJISTA': '#EE5A6F',
    'RANGO': '#FFD93D'
}

# --- FUNCIONES DE CÁLCULO TÉCNICO ---
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
    # Ajuste de sensibilidad de curtosis: reduce el impacto errático
    kurtosis_factor = 1 + (kurtosis / 15).clip(-0.4, 0.4) 
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
        macd_v = row['MACD_V']
        
        if pd.isna(z) or pd.isna(price) or pd.isna(sma50):
            regimes.append('RANGO')
            continue
        
        # LÓGICA MEJORADA: Tendencia (SMA) + Momentum (Z) + Fuerza (MACD-V)
        if price > sma50:
            if z > 2.0:
                regime = 'SOBRECOMPRA'
            elif z > 0.5 or macd_v > 0: # Mantiene ALCISTA si el momentum es positivo aunque el Z baje
                regime = 'ALCISTA'
            else:
                regime = 'RANGO'
        elif price < sma50:
            if z < -2.0:
                regime = 'SOBREVENTA'
            elif z < -0.5 or macd_v < 0:
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
        if start_date is None and years_back is None: years_back = 7
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*years_back)).strftime('%Y-%m-%d')
        else:
            start_date = start_date if isinstance(start_date, str) else start_date.strftime('%Y-%m-%d')
        
        data = yf.download(ticker, start=start_date, interval='1wk', progress=False, auto_adjust=True)
        
        if data is None or data.empty: return None
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        
        df = pd.DataFrame({
            'Close': data['Close'].squeeze(),
            'Open': data['Open'].squeeze(),
            'High': data['High'].squeeze(),
            'Low': data['Low'].squeeze(),
            'Volume': data['Volume'].squeeze()
        }, index=data.index)
        
        df = df.dropna(subset=['Close'])
        df['SMA_20'] = calculate_sma(df['Close'], 20)
        df['SMA_50'] = calculate_sma(df['Close'], 50)
        df['Donchian_Upper'], df['Donchian_Middle'], df['Donchian_Lower'] = calculate_donchian(df, period=20)
        df = calculate_z_score_macdv(df)
        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# --- VISUALIZACIÓN ---
def plot_zscore_dashboard(df_recent, ticker):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(24, 18), facecolor='#0E1117')
    gs = fig.add_gridspec(5, 1, height_ratios=[3.5, 1.2, 1.2, 1, 1], hspace=0.4)
    
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#1A1D29')
    
    # Canales y Precio
    ax1.fill_between(df_recent.index, df_recent['Donchian_Upper'], df_recent['Donchian_Lower'], alpha=0.08, color='#00D9FF')
    ax1.plot(df_recent.index, df_recent['Close'], color='#FFFFFF', alpha=0.4, linewidth=2, zorder=3)
    
    # Scatter de Regímenes
    for regime, color in REGIME_COLORS_ZSCORE.items():
        mask = df_recent['Regime_ZScore'] == regime
        if mask.any():
            ax1.scatter(df_recent[mask].index, df_recent[mask]['Close'], c=color, label=regime, s=100, edgecolors='white', alpha=0.9, zorder=5)
    
    ax1.plot(df_recent.index, df_recent['SMA_50'], color='#BD93F9', linewidth=2, label='SMA(50)', linestyle='--')
    
    # Anotación actual
    current = df_recent.iloc[-1]
    ax1.annotate(f'{current["Regime_ZScore"]}\n${current["Close"]:.2f}', 
                 xy=(current.name, current['Close']), xytext=(20, 30), textcoords='offset points',
                 bbox=dict(boxstyle='round', facecolor=REGIME_COLORS_ZSCORE[current['Regime_ZScore']], alpha=0.8),
                 arrowprops=dict(arrowstyle='->', color='white'))

    ax1.set_title(f"{ticker} - Market Regime Analysis", fontsize=20, pad=20)
    ax1.legend(loc='upper left', ncol=3)
    
    # Subplot Z-Score
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_facecolor('#1A1D29')
    ax2.plot(df_recent.index, df_recent['Z_Score_Adjusted'], color='#00D9FF', linewidth=2)
    ax2.axhline(2, color='#FF6B6B', linestyle='--')
    ax2.axhline(-2, color='#9D4EDD', linestyle='--')
    ax2.axhline(0.75, color='#4ECDC4', linestyle=':')
    ax2.axhline(-0.75, color='#EE5A6F', linestyle=':')
    ax2.set_ylabel('Z-Score')

    # Subplot MACD-V
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.set_facecolor('#1A1D29')
    ax3.plot(df_recent.index, df_recent['MACD_V'], color='#4ECDC4', label='MACD-V')
    ax3.axhline(0, color='white', alpha=0.3)
    ax3.legend()

    # Subplot Kurtosis
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.set_facecolor('#1A1D29')
    ax4.bar(df_recent.index, df_recent['Kurtosis'], color='#BD93F9', alpha=0.5)
    ax4.set_ylabel('Kurtosis')

    # Subplot Lineal de Regímenes
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    ax5.set_facecolor('#1A1D29')
    regime_map = {r: i for i, r in enumerate(REGIME_COLORS_ZSCORE.keys())}
    ax5.scatter(df_recent.index, df_recent['Regime_ZScore'].map(regime_map), c=df_recent['Regime_ZScore'].map(REGIME_COLORS_ZSCORE), s=50)
    ax5.set_yticks(range(len(regime_map)))
    ax5.set_yticklabels(regime_map.keys())

    return fig

# --- INTERFAZ STREAMLIT ---
def zscore_analyzer_page():
    st.title("📊 Z-Score MACD-V Professional Analyzer")
    
    col_cfg1, col_cfg2, col_cfg3 = st.columns([2, 2, 2])
    with col_cfg1: ticker = st.text_input("Ticker", value="SOFI").upper()
    with col_cfg2: lookback_months = st.slider("Meses vista", 1, 24, 12)
    with col_cfg3: years_back = st.selectbox("Histórico", [3, 5, 7, 10], index=2)

    if st.button("🚀 ANALIZAR", use_container_width=True):
        df_weekly = download_weekly_data(ticker, years_back=years_back)
        if df_weekly is not None:
            df_weekly['Regime_ZScore'] = classify_regime_zscore(df_weekly)
            st.session_state['df_z'] = df_weekly
            st.session_state['t_z'] = ticker

    if 'df_z' in st.session_state:
        df = st.session_state['df_z']
        df_recent = df.tail(int(lookback_months * 4.33))
        current = df.iloc[-1]

        # Métricas principales
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RÉGIMEN", current['Regime_ZScore'])
        m2.metric("PRECIO", f"${current['Close']:.2f}")
        m3.metric("Z-SCORE", f"{current['Z_Score_Adjusted']:.2f}σ")
        m4.metric("KURTOSIS", f"{current['Kurtosis']:.2f}")

        st.pyplot(plot_zscore_dashboard(df_recent, st.session_state['t_z']))

        # --- SECCIÓN DE REFERENCIA TÉCNICA (LOGICA DE ESTADOS) ---
        st.markdown("---")
        with st.expander("📘 Diccionario de Lógica y Reglas de Estado (Referencia)"):
            st.markdown("""
            ### 🛠 Metodología de Clasificación
            La clasificación no depende solo del Z-Score, sino de la confluencia de tres capas de datos:
            
            1.  **Capa de Tendencia (Filtro SMA50):** Define el sesgo principal. Si el precio está por encima, el sistema busca estados Alcistas; si está por debajo, Bajistas.
            2.  **Capa de Momentum (Z-Score 20w):** Mide la velocidad del movimiento. Un Z-Score alto indica aceleración.
            3.  **Capa de Confirmación (MACD-V):** Actúa como "memoria de tendencia". Si el Z-Score cae pero el MACD-V sigue siendo positivo, el estado se mantiene **ALCISTA** en lugar de pasar a RANGO.

            | Régimen | Condición SMA | Z-Score (σ) | MACD-V | Acción Típica |
            | :--- | :--- | :--- | :--- | :--- |
            | **🔴 SOBRECOMPRA** | Precio > SMA50 | **> +2.0** | Muy Alto | Tomar beneficios / No comprar |
            | **🟢 ALCISTA** | Precio > SMA50 | **> +0.5** o **MACD-V > 0** | Positivo | Mantener / Comprar retrocesos |
            | **🟡 RANGO** | Neutral | **Entre ±0.5** | Cerca de 0 | Esperar rotura de Donchian |
            | **🔴 BAJISTA** | Precio < SMA50 | **< -0.5** o **MACD-V < 0** | Negativo | Liquidez / Coberturas |
            | **🟣 SOBREVENTA** | Precio < SMA50 | **< -2.0** | Muy Bajo | Vigilancia para rebote técnico |

            **¿Por qué el ajuste de Curtosis?** La Curtosis mide las "Fat Tails" (colas anchas). Si la curtosis es alta, significa que el mercado está lanzando movimientos extremos. El script ajusta el Z-Score dividiéndolo por un factor de curtosis para que no se activen señales de sobrecompra falsas durante "cisnes negros" o rallies institucionales.
            """)

if __name__ == "__main__":
    if check_password():
        zscore_analyzer_page()
