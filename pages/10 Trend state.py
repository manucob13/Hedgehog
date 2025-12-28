# pages/Market_Analysis_Weekly_Pro.py
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
from utils import check_password

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Market Analysis Pro - Weekly", layout="wide")

REGIME_COLORS = {'RIESGO': '#FF6B6B', 'BAJISTA': '#EE5A6F', 'RANGO': '#FFD93D', 'ALCISTA': '#4ECDC4'}
TREND_COLORS = {1: '#4ECDC4', -1: '#EE5A6F', 0: '#95A5A6', 2: '#FF6B6B'}

def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def calculate_atr(df, period=26):
    hl = df['High'] - df['Low']
    hc = np.abs(df['High'] - df['Close'].shift())
    lc = np.abs(df['Low'] - df['Close'].shift())
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(window=period).mean()

def calculate_macd_v(df, fast=12, slow=26, sig=9, atr_len=26):
    fast_ema = calculate_ema(df['Close'], fast)
    slow_ema = calculate_ema(df['Close'], slow)
    atr = calculate_atr(df, atr_len)
    macd = ((fast_ema - slow_ema) / atr) * 100
    return macd, calculate_ema(macd, sig)

def calculate_donchian(df, period=20):
    upper = df['High'].rolling(window=period).max()
    lower = df['Low'].rolling(window=period).min()
    return upper, (upper + lower) / 2, lower

def calculate_choppiness(df, period=14):
    hl = df['High'] - df['Low']
    hc = np.abs(df['High'] - df['Close'].shift())
    lc = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr_sum = tr.rolling(window=period).sum()
    hmax = df['High'].rolling(window=period).max()
    lmin = df['Low'].rolling(window=period).min()
    return 100 * np.log10(atr_sum / (hmax - lmin)) / np.log10(period)

def classify_regime(df):
    regimes = []
    for _, row in df.iterrows():
        mv, ch, pr, dm = row['MACD_V'], row['Choppiness'], row['Close'], row['Donchian_Middle']
        if mv > 150 or mv < -150:
            regimes.append('RIESGO')
        elif ch > 61.8 or (-50 <= mv <= 50):
            regimes.append('RANGO')
        elif 50 < mv <= 150 and ch < 38.2 and pr > dm:
            regimes.append('ALCISTA')
        elif -150 <= mv < -50 and ch < 38.2 and pr < dm:
            regimes.append('BAJISTA')
        elif mv > 50:
            regimes.append('ALCISTA')
        elif mv < -50:
            regimes.append('BAJISTA')
        else:
            regimes.append('RANGO')
    return regimes

def nadaraya_watson(x, y, bw=8):
    est = np.zeros_like(y, dtype=float)
    for i, xi in enumerate(x):
        w = np.exp(-0.5 * ((x - xi) / bw)**2)
        est[i] = np.sum(w * y) / np.sum(w) if np.sum(w) > 0 else y[i]
    return est

def classify_trend(prices, trend, atr, mult=1.5):
    upper, lower = trend + (atr * mult), trend - (atr * mult)
    slopes = np.diff(trend, prepend=trend[0])
    thresh = np.abs(trend) * 0.002
    states = np.where((prices > upper) | (prices < lower), 2,
              np.where(slopes > thresh, 1,
              np.where(slopes < -thresh, -1, 0)))
    return states, trend, upper, lower

def project_trend(x, trend, periods=4, lookback=10, degree=2):
    lb = min(lookback, len(trend))
    deg = min(degree, lb - 1)
    coeffs = np.polyfit(x[-lb:], trend[-lb:], deg)
    poly = np.poly1d(coeffs)
    x_fut = np.arange(len(trend), len(trend) + periods)
    y_fit = poly(x[-lb:])
    ss_res = np.sum((trend[-lb:] - y_fit)**2)
    ss_tot = np.sum((trend[-lb:] - np.mean(trend[-lb:]))**2)
    conf = max(0, min(1, 1 - (ss_res / ss_tot) if ss_tot > 0 else 0))
    return x_fut, poly(x_fut), conf

@st.cache_data(ttl=3600)
def download_data(ticker, start='2018-01-01'):
    try:
        data = yf.download(ticker, start=start, interval='1wk', progress=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        df = pd.DataFrame({c: data[c].squeeze() for c in ['Close', 'Open', 'High', 'Low', 'Volume']}, index=data.index)
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['Donchian_Upper'], df['Donchian_Middle'], df['Donchian_Lower'] = calculate_donchian(df, 20)
        df['Choppiness'] = calculate_choppiness(df, 14)
        df['MACD_V'], df['MACD_V_Signal'] = calculate_macd_v(df)
        df['ATR'] = calculate_atr(df, 14)
        return df.dropna()
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def plot_regime(df, ticker):
    fig, ax = plt.subplots(4, 1, figsize=(24, 14), facecolor='#0E1117', 
                          gridspec_kw={'height_ratios': [3, 1, 1, 1], 'hspace': 0.4})
    for a in ax:
        a.set_facecolor('#1A1D29')
    
    ax[0].plot(df.index, df['Close'], 'w', linewidth=1.5, label='Precio')
    for reg, col in REGIME_COLORS.items():
        mask = df['Regime'] == reg
        if mask.sum() > 0:
            ax[0].scatter(df[mask].index, df[mask]['Close'], c=col, s=80, edgecolors='w', linewidth=1.5, label=reg)
    ax[0].set_title(f'{ticker} - Régimen Semanal', fontsize=20, color='w', pad=20)
    ax[0].legend(loc='upper left', ncol=4)
    ax[0].grid(alpha=0.1)
    
    ax[1].plot(df.index, df['Choppiness'], '#FFD93D', linewidth=2)
    ax[1].axhline(61.8, color='#FFD93D', linestyle='--', alpha=0.8)
    ax[1].axhline(38.2, color='#4ECDC4', linestyle='--', alpha=0.8)
    ax[1].set_ylabel('Choppiness', color='w')
    ax[1].grid(alpha=0.1)
    
    for i in range(1, len(df)):
        y = df['MACD_V'].iloc[i]
        col = '#FF6B6B' if abs(y) > 150 else '#4ECDC4' if y > 50 else '#EE5A6F' if y < -50 else '#95A5A6'
        ax[2].plot([df.index[i-1], df.index[i]], [df['MACD_V'].iloc[i-1], y], color=col, linewidth=2.5)
    ax[2].axhline(0, color='#8E93A1', linestyle='-', alpha=0.7)
    ax[2].set_ylabel('MACD-V', color='w')
    ax[2].grid(alpha=0.1)
    
    for reg in ['BAJISTA', 'RANGO', 'ALCISTA', 'RIESGO']:
        mask = df['Regime'] == reg
        if mask.sum() > 0:
            ax[3].scatter(df[mask].index, [['BAJISTA','RANGO','ALCISTA','RIESGO'].index(reg)]*mask.sum(), 
                         c=REGIME_COLORS[reg], s=100, edgecolors='w', linewidth=1.8)
    ax[3].set_yticks(range(4))
    ax[3].set_yticklabels(['BAJISTA', 'RANGO', 'ALCISTA', 'RIESGO'], color='w')
    ax[3].grid(alpha=0.1, axis='x')
    
    plt.tight_layout()
    return fig

def plot_atr(res, proj, metrics, ticker):
    fig, ax = plt.subplots(4, 1, figsize=(24, 14), facecolor='#0E1117',
                          gridspec_kw={'height_ratios': [3, 1, 1, 0.8], 'hspace': 0.35})
    for a in ax:
        a.set_facecolor('#1A1D29')
    
    ax[0].fill_between(range(len(res)), res['Lower_Risk'], res['Upper_Risk'], color='#FFB86C', alpha=0.08)
    ax[0].plot(res['Trend'], '#00D9FF', linewidth=2.5, label='Tendencia')
    ax[0].plot(res['Close'], 'w', linewidth=1.5, label='Precio')
    
    px_start = len(res) - 1
    px = np.arange(px_start, px_start + len(proj) + 1)
    pt = np.concatenate([[res['Trend'].iloc[-1]], proj['Trend_Projection'].values])
    ax[0].plot(px, pt, '#FF6B6B', linewidth=3, linestyle='--', marker='o', label=f'Proyección ({len(proj)} sem)')
    ax[0].axvline(px_start, color='yellow', linestyle=':', linewidth=2, alpha=0.5)
    ax[0].set_title(f'{ticker} - ATR Trend + Proyección', fontsize=20, color='w', pad=20)
    ax[0].legend(loc='upper left', ncol=2)
    ax[0].grid(alpha=0.1)
    
    ax[1].plot(res['ATR'], '#BD93F9', linewidth=2, label='ATR')
    ax[1].fill_between(range(len(res)), 0, res['ATR'], color='#BD93F9', alpha=0.2)
    ax[1].legend(loc='upper left')
    ax[1].grid(alpha=0.1)
    
    for i in range(1, len(res)):
        y = res['MACD_V'].iloc[i]
        col = '#FF6B6B' if abs(y) > 150 else '#4ECDC4' if y > 50 else '#EE5A6F' if y < -50 else '#95A5A6'
        ax[2].plot([i-1, i], [res['MACD_V'].iloc[i-1], y], color=col, linewidth=2.5)
    ax[2].axhline(0, color='#8E93A1', linestyle='-', alpha=0.7)
    ax[2].grid(alpha=0.1)
    
    for st in [-1, 0, 1, 2]:
        mask = res['State'] == st
        if mask.sum() > 0:
            y_vals = [['BAJISTA','LATERAL','ALCISTA','RIESGO'].index({-1:'BAJISTA',0:'LATERAL',1:'ALCISTA',2:'RIESGO'}[st])]*mask.sum()
            ax[3].scatter(res[mask].index, y_vals, c=TREND_COLORS[st], s=100, edgecolors='w', linewidth=1.8)
    ax[3].set_yticks(range(4))
    ax[3].set_yticklabels(['BAJISTA', 'LATERAL', 'ALCISTA', 'RIESGO'], color='w')
    ax[3].grid(alpha=0.1, axis='x')
    
    plt.tight_layout()
    return fig

def main_app():
    st.markdown("""<style>
    .main{background-color:#0E1117}
    .stMetric{background:linear-gradient(135deg,#1A1D29 0%,#2D3142 100%);padding:20px;border-radius:12px;border:2px solid #4ECDC4}
    .stMetric label{color:#00D9FF!important;font-weight:700!important}
    .stMetric [data-testid="stMetricValue"]{color:#FFF!important;font-size:24px!important;font-weight:800!important}
    .proj-box{background:linear-gradient(135deg,#FF6B6B 0%,#FFB86C 100%);padding:20px;border-radius:12px;border:3px solid #FFD700;color:white;text-align:center}
    h1,h2,h3{color:#FFF!important;font-weight:800!important}
    </style>""", unsafe_allow_html=True)
    
    st.title("📊 Market Analysis Pro - Weekly Edition")
    st.markdown("Análisis Completo Semanal: Régimen + ATR Trend con Proyección")
    
    with st.sidebar:
        st.markdown("<div style='text-align:center;padding:20px;background:linear-gradient(135deg,#4ECDC4 0%,#00D9FF 100%);border-radius:15px;margin-bottom:20px'><h2 style='color:white;margin:0'>⚙️ Config</h2></div>", unsafe_allow_html=True)
        ticker = st.text_input("Ticker", "AAPL").upper()
        lookback = st.slider("Meses", 6, 24, 6)
        start = st.date_input("Inicio", datetime(2018, 1, 1))
        st.markdown("---")
        atr_mult = st.slider("Mult. ATR", 0.5, 3.0, 1.5, 0.1)
        st.markdown("---")
        proj_per = st.slider("Períodos Proy.", 1, 12, 4)
        proj_lb = st.slider("Inercia", 5, 20, 10)
        proj_deg = st.selectbox("Tipo", [1,2,3], index=1, format_func=lambda x:{1:"Lineal",2:"Cuadrática",3:"Cúbica"}[x])
        st.markdown("---")
        btn = st.button("🚀 ANALIZAR", type="primary", use_container_width=True)
    
    if btn:
        with st.spinner(f"Procesando {ticker}..."):
            df = download_data(ticker, start.strftime('%Y-%m-%d'))
            if df is None:
                st.error("❌ Error")
                st.stop()
            
            df['Regime'] = classify_regime(df)
            
            prices = df['Close'].values
            x = np.arange(len(prices))
            trend = nadaraya_watson(x, prices, 8)
            atr = df['ATR'].values
            
            states, trend_ref, upper, lower = classify_trend(prices, trend, atr, atr_mult)
            x_fut, y_fut, conf = project_trend(x, trend_ref, proj_per, proj_lb, proj_deg)
            
            atr_cur = atr[-1] if not np.isnan(atr[-1]) else np.nanmean(atr[-5:])
            
            res = pd.DataFrame({
                'Date': df.index, 'Close': prices, 'Trend': trend_ref, 'ATR': atr,
                'Upper_Risk': upper, 'Lower_Risk': lower, 'State': states,
                'MACD_V': df['MACD_V'].values, 'MACD_V_Signal': df['MACD_V_Signal'].values
            })
            
            proj = pd.DataFrame({
                'Date': pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=proj_per, freq='W'),
                'Trend_Projection': y_fut,
                'Upper_Projection': y_fut + (atr_cur * atr_mult),
                'Lower_Projection': y_fut - (atr_cur * atr_mult)
            })
            
            metrics = {
                'projection_confidence': conf,
                'projection_change_pct': ((y_fut[-1] - prices[-1]) / prices[-1] * 100),
                'projection_target': y_fut[-1],
                'atr_current': atr_cur
            }
            
            st.session_state.update({'ticker':ticker,'df':df,'res':res,'proj':proj,'metrics':metrics,'lb':lookback})
            st.success(f"✅ {len(df)} periodos")
    
    if 'df' in st.session_state:
        df = st.session_state['df']
        res = st.session_state['res']
        proj = st.session_state['proj']
        metrics = st.session_state['metrics']
        ticker = st.session_state['ticker']
        lb = st.session_state['lb']
        
        lbw = int(lb * 4.33)
        df_rec = df.tail(lbw)
        res_rec = res.tail(lbw)
        
        cr = df_rec.iloc[-1]
        ct = res.iloc[-1]
        
        st.markdown("---\n## 📊 PARTE 1: RÉGIMEN SEMANAL\n---")
        
        c1,c2,c3,c4,c5 = st.columns(5)
        emoji = {'ALCISTA':'🟢','BAJISTA':'🔴','RANGO':'🟡','RIESGO':'🟠'}
        c1.metric("RÉGIMEN", f"{emoji[cr['Regime']]} {cr['Regime']}")
        c2.metric("PRECIO", f"${cr['Close']:.2f}")
        chst = "Choppy" if cr['Choppiness']>61.8 else "Trending" if cr['Choppiness']<38.2 else "Neutral"
        c3.metric("CHOP", f"{cr['Choppiness']:.1f}", chst)
        mcst = "Extremo" if abs(cr['MACD_V'])>150 else "Alcista" if cr['MACD_V']>50 else "Bajista" if cr['MACD_V']<-50 else "Neutral"
        c4.metric("MACD-V", f"{cr['MACD_V']:.1f}", mcst)
        dp = ((cr['Close']-cr['Donchian_Lower'])/(cr['Donchian_Upper']-cr['Donchian_Lower'])*100)
        c5.metric("POS DONCH", f"{dp:.0f}%")
        
        st.markdown("---")
        st.pyplot(plot_regime(df_rec, ticker))
        
        st.markdown("---\n## 📈 PARTE 2: ATR TREND + PROYECCIÓN\n---")
        
        stn = {1:'ALCISTA',-1:'BAJISTA',0:'LATERAL',2:'RIESGO'}
        ste = {1:'🟢',-1:'🔴',0:'⚫',2:'🟠'}
        
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("ESTADO", f"{ste[ct['State']]} {stn[ct['State']]}")
        dist = ((ct['Close']-ct['Trend'])/ct['Trend']*100)
        c2.metric("VS TEND", f"{dist:+.2f}%")
        c3.metric("ATR", f"${ct['ATR']:.2f}")
        pos = ((ct['Close']-ct['Lower_Risk'])/(ct['Upper_Risk']-ct['Lower_Risk'])*100)
        c4.metric("POS ATR", f"{pos:.0f}%")
        
        st.markdown("---\n### 🔮 Proyección")
        
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f"<div class='proj-box'><div style='font-size:14px;font-weight:bold;color:#FFD700'>TARGET</div><div style='font-size:28px;font-weight:bold;margin-top:10px'>${metrics['projection_target']:.2f}</div></div>", unsafe_allow_html=True)
        
        chg = metrics['projection_change_pct']
        col = '#4ECDC4' if chg>0 else '#FF6B6B'
        arr = '↗' if chg>0 else '↘'
        c2.markdown(f"<div class='proj-box'><div style='font-size:14px;font-weight:bold;color:#FFD700'>CAMBIO</div><div style='font-size:28px;font-weight:bold;margin-top:10px;color:{col}'>{arr} {chg:+.1f}%</div></div>", unsafe_allow_html=True)
        
        conf = metrics['projection_confidence']
        ccol = '#4ECDC4' if conf>0.7 else '#FFB86C' if conf>0.4 else '#FF6B6B'
        clbl = 'Alta' if conf>0.7 else 'Media' if conf>0.4 else 'Baja'
        c3.markdown(f"<div class='proj-box'><div style='font-size:14px;font-weight:bold;color:#FFD700'>CONFIANZA</div><div style='font-size:28px;font-weight:bold;margin-top:10px;color:{ccol}'>{conf*100:.0f}%</div><div style='font-size:14px;margin-top:5px'>{clbl}</div></div>", unsafe_allow_html=True)
        
        c4.markdown(f"<div class='proj-box'><div style='font-size:14px;font-weight:bold;color:#FFD700'>FECHA</div><div style='font-size:20px;font-weight:bold;margin-top:10px'>{proj['Date'].iloc[-1].strftime('%Y-%m-%d')}</div></div>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.pyplot(plot_atr(res_rec, proj, metrics, ticker))
        
        st.markdown("---\n## 📥 Descargar Datos")
        c1,c2 = st.columns(2)
        c1.download_button("📊 Régimen CSV", df[['Close','Regime','MACD_V','Choppiness']].to_csv(), 
                          f"regime_{ticker}.csv", "text/csv", use_container_width=True)
        c2.download_button("📈 ATR CSV", res[['Close','Trend','State','ATR']].to_csv(), 
                          f"atr_{ticker}.csv", "text/csv", use_container_width=True)
    else:
        st.info("👈 Configura parámetros y presiona ANALIZAR")

if __name__ == "__main__":
    if check_password():
        main_app()
    else:
        st.stop()
