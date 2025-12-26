# pages/Market_Regime_ADX_Weekly.py
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
from utils import check_password

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Market Regime Analyzer", layout="wide")

REGIME_COLORS_DAILY = {
    'RIESGO': '#FF6B6B',
    'BAJISTA': '#EE5A6F',
    'RANGO': '#95A5A6',
    'ALCISTA': '#4ECDC4'
}

REGIME_COLORS_WEEKLY = {
    'RIESGO': '#FF6B6B',
    'DOWNTREND': '#EE5A6F',
    'SIDEWAYS': '#FFD93D',
    'UPTREND': '#4ECDC4'
}

def calculate_adx(df, period=14):
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
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_sma(data, period):
    return data.rolling(window=period).mean()

def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def calculate_donchian(df, period=20):
    upper = df['High'].rolling(window=period).max()
    lower = df['Low'].rolling(window=period).min()
    middle = (upper + lower) / 2
    return upper, middle, lower

def calculate_choppiness(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_sum = true_range.rolling(window=period).sum()
    high_max = df['High'].rolling(window=period).max()
    low_min = df['Low'].rolling(window=period).min()
    chop = 100 * np.log10(atr_sum / (high_max - low_min)) / np.log10(period)
    return chop

def calculate_atr(df, period=26):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def calculate_macd_v(df, fast_len=12, slow_len=26, signal_len=9, atr_len=26):
    fast_ema = calculate_ema(df['Close'], fast_len)
    slow_ema = calculate_ema(df['Close'], slow_len)
    atr = calculate_atr(df, atr_len)
    macd = ((fast_ema - slow_ema) / atr) * 100
    signal = calculate_ema(macd, signal_len)
    return macd, signal

@st.cache_data(ttl=timedelta(hours=1))
def download_daily_data(ticker, start_date='2018-01-01'):
    try:
        data = yf.download(ticker, start=start_date, interval='1d', progress=False)
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
        st.error(f"Error descargando datos diarios para {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=timedelta(hours=1))
def download_weekly_data(ticker, start_date='2018-01-01'):
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
        df['SMA_20'] = calculate_sma(df['Close'], 20)
        df['SMA_50'] = calculate_sma(df['Close'], 50)
        df['Donchian_Upper'], df['Donchian_Middle'], df['Donchian_Lower'] = calculate_donchian(df, period=20)
        df['Choppiness'] = calculate_choppiness(df, period=14)
        df['MACD_V'], df['MACD_V_Signal'] = calculate_macd_v(df, fast_len=12, slow_len=26, signal_len=9, atr_len=26)
        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"Error descargando datos semanales para {ticker}: {str(e)}")
        return None

def classify_regime_daily(df):
    regimes = []
    for idx, row in df.iterrows():
        adx = row['ADX']
        rsi = row['RSI']
        plus_di = row['Plus_DI']
        minus_di = row['Minus_DI']
        price = row['Close']
        sma_20 = row['SMA_20']
        sma_50 = row['SMA_50']
        if rsi >= 75:
            regime = 'RIESGO'
        elif rsi <= 25:
            regime = 'RIESGO'
        elif (adx > 25 and plus_di > minus_di and price > sma_20 and sma_20 > sma_50):
            regime = 'ALCISTA'
        elif (adx > 25 and minus_di > plus_di and price < sma_20 and sma_20 < sma_50):
            regime = 'BAJISTA'
        else:
            regime = 'RANGO'
        regimes.append(regime)
    return regimes

def classify_regime_weekly(df):
    regimes = []
    for idx, row in df.iterrows():
        macd_v = row['MACD_V_Signal']
        chop = row['Choppiness']
        price = row['Close']
        donchian_middle = row['Donchian_Middle']
        if macd_v > 150 or macd_v < -150:
            regime = 'RIESGO'
        elif chop > 61.8:
            regime = 'SIDEWAYS'
        elif chop < 38.2 and price > donchian_middle and macd_v > 0:
            regime = 'UPTREND'
        elif chop < 38.2 and price < donchian_middle and macd_v < 0:
            regime = 'DOWNTREND'
        elif price > donchian_middle and macd_v > 0:
            regime = 'UPTREND'
        elif price < donchian_middle and macd_v < 0:
            regime = 'DOWNTREND'
        else:
            regime = 'SIDEWAYS'
        regimes.append(regime)
    return regimes

def plot_daily_dashboard(df_recent, ticker):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(24, 14), facecolor='#0E1117')
    gs = fig.add_gridspec(4, 1, height_ratios=[3.5, 1, 1, 1], hspace=0.4)
    
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#1A1D29')
    ax1.plot(df_recent.index, df_recent['Close'], color='#FFFFFF', alpha=0.15, linewidth=6, zorder=1)
    ax1.plot(df_recent.index, df_recent['Close'], color='#E0E0E0', alpha=0.4, linewidth=3, zorder=2)
    ax1.plot(df_recent.index, df_recent['Close'], color='#FFFFFF', linewidth=1.5, zorder=3)
    
    for regime_name, color in REGIME_COLORS_DAILY.items():
        mask = df_recent['Regime_Daily'] == regime_name
        if mask.sum() > 0:
            ax1.scatter(df_recent[mask].index, df_recent[mask]['Close'], c=color, alpha=0.15, s=220, edgecolors='none', zorder=4)
            ax1.scatter(df_recent[mask].index, df_recent[mask]['Close'], c=color, label=regime_name, alpha=0.95, s=85, edgecolors='white', linewidth=1.5, zorder=5)
    
    ax1.plot(df_recent.index, df_recent['SMA_20'], color='#00D9FF', alpha=0.9, linewidth=2.5, label='SMA(20)', zorder=3)
    ax1.plot(df_recent.index, df_recent['SMA_50'], color='#BD93F9', alpha=0.9, linewidth=2.5, label='SMA(50)', zorder=3)
    
    current = df_recent.iloc[-1]
    ax1.scatter(current.name, current['Close'], facecolors='none', edgecolors=REGIME_COLORS_DAILY[current['Regime_Daily']], s=450, linewidth=5, alpha=0.4, marker='o', zorder=9)
    ax1.scatter(current.name, current['Close'], facecolors=REGIME_COLORS_DAILY[current['Regime_Daily']], edgecolors='white', s=280, linewidth=4, marker='o', label=f'📍 Actual: {current["Regime_Daily"]}', zorder=10)
    
    bbox_color = REGIME_COLORS_DAILY[current['Regime_Daily']]
    ax1.annotate(f'{current["Regime_Daily"]}\n${current["Close"]:.2f}', xy=(current.name, current['Close']), xytext=(25, 40), textcoords='offset points', fontsize=14, fontweight='bold', color='white', bbox=dict(boxstyle='round,pad=1', facecolor=bbox_color, alpha=0.95, edgecolor='white', linewidth=3), arrowprops=dict(arrowstyle='->', lw=3, color=bbox_color, connectionstyle='arc3,rad=0.3'), zorder=11)
    
    ax1.text(0.5, 1.10, f'{ticker}', transform=ax1.transAxes, fontsize=28, fontweight='bold', ha='center', color='#FFFFFF')
    ax1.text(0.5, 1.05, 'Daily Market Regime Analysis', transform=ax1.transAxes, fontsize=13, style='italic', ha='center', color='#8E93A1')
    ax1.set_ylabel('Price ($)', fontsize=15, fontweight='700', color='#FFFFFF', labelpad=12)
    legend = ax1.legend(loc='upper left', fontsize=11, framealpha=0.95, ncol=3, edgecolor='#00D9FF', fancybox=True, borderpad=1.2, labelspacing=1, columnspacing=2)
    legend.get_frame().set_facecolor('#1A1D29')
    legend.get_frame().set_linewidth(2)
    ax1.grid(True, alpha=0.08, linestyle='-', linewidth=1, color='#FFFFFF')
    ax1.tick_params(labelsize=11, colors='#B0B0B0', width=1.5)
    for spine in ax1.spines.values():
        spine.set_color('#2D3142')
        spine.set_linewidth(2)
    
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_facecolor('#1A1D29')
    ax2.plot(df_recent.index, df_recent['RSI'], color='#FF79C6', linewidth=2.5, label='RSI', zorder=3)
    ax2.fill_between(df_recent.index, df_recent['RSI'], 50, where=(df_recent['RSI'] >= 50), color='#4ECDC4', alpha=0.2, zorder=1)
    ax2.fill_between(df_recent.index, df_recent['RSI'], 50, where=(df_recent['RSI'] < 50), color='#FF6B6B', alpha=0.2, zorder=1)
    ax2.axhline(y=75, color='#FF6B6B', linestyle='--', linewidth=2, alpha=0.8, zorder=2)
    ax2.axhline(y=70, color='#FFB86C', linestyle=':', linewidth=1.5, alpha=0.7, zorder=2)
    ax2.axhline(y=50, color='#8E93A1', linestyle='-', linewidth=1.5, alpha=0.7, zorder=2)
    ax2.axhline(y=30, color='#FFB86C', linestyle=':', linewidth=1.5, alpha=0.7, zorder=2)
    ax2.axhline(y=25, color='#4ECDC4', linestyle='--', linewidth=2, alpha=0.8, zorder=2)
    ax2.fill_between(df_recent.index, 75, 100, alpha=0.12, color='#FF6B6B', zorder=0)
    ax2.fill_between(df_recent.index, 0, 25, alpha=0.12, color='#4ECDC4', zorder=0)
    ax2.scatter(current.name, current['RSI'], facecolors='#FF79C6', edgecolors='white', s=220, linewidth=4, marker='o', zorder=10)
    rsi_color = '#FF6B6B' if current['RSI'] > 70 else '#4ECDC4' if current['RSI'] < 30 else '#8E93A1'
    ax2.text(0.02, 0.88, f'RSI: {current["RSI"]:.1f}', transform=ax2.transAxes, fontsize=13, fontweight='bold', color='white', verticalalignment='top', bbox=dict(boxstyle='round,pad=0.7', facecolor=rsi_color, alpha=0.95, edgecolor='white', linewidth=2.5))
    ax2.set_ylabel('RSI', fontsize=14, fontweight='700', color='#FFFFFF', labelpad=12)
    ax2.set_ylim([0, 100])
    ax2.grid(True, alpha=0.08, linestyle='-', linewidth=1, color='#FFFFFF')
    ax2.tick_params(labelsize=10.5, colors='#B0B0B0', width=1.5)
    for spine in ax2.spines.values():
        spine.set_color('#2D3142')
        spine.set_linewidth(2)
    
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.set_facecolor('#1A1D29')
    ax3.plot(df_recent.index, df_recent['ADX'], color='#F8F8F2', linewidth=2.5, label='ADX', zorder=3)
    ax3.plot(df_recent.index, df_recent['Plus_DI'], color='#4ECDC4', linewidth=2.2, alpha=0.9, label='+DI', zorder=2)
    ax3.plot(df_recent.index, df_recent['Minus_DI'], color='#FF6B6B', linewidth=2.2, alpha=0.9, label='-DI', zorder=2)
    ax3.axhline(y=25, color='#4ECDC4', linestyle='--', linewidth=2, alpha=0.8)
    ax3.axhline(y=20, color='#FFB86C', linestyle=':', linewidth=1.5, alpha=0.7)
    ax3.set_ylabel('ADX / DI', fontsize=14, fontweight='700', color='#FFFFFF', labelpad=12)
    ax3.set_ylim([0, 70])
    legend = ax3.legend(loc='upper left', fontsize=10, framealpha=0.95, ncol=3, edgecolor='#00D9FF', fancybox=True)
    legend.get_frame().set_facecolor('#1A1D29')
    ax3.grid(True, alpha=0.08, linestyle='-', linewidth=1, color='#FFFFFF')
    ax3.tick_params(labelsize=10.5, colors='#B0B0B0', width=1.5)
    for spine in ax3.spines.values():
        spine.set_color('#2D3142')
        spine.set_linewidth(2)
    
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.set_facecolor('#1A1D29')
    regime_order = ['BAJISTA', 'RANGO', 'ALCISTA', 'RIESGO']
    for regime_name in regime_order:
        mask = df_recent['Regime_Daily'] == regime_name
        if mask.sum() > 0:
            color = REGIME_COLORS_DAILY[regime_name]
            regime_num = regime_order.index(regime_name)
            ax4.scatter(df_recent[mask].index, [regime_num] * mask.sum(), c=color, alpha=0.95, s=110, edgecolors='white', linewidth=1.8, zorder=4)
    ax4.set_ylabel('Regime', fontsize=14, fontweight='700', color='#FFFFFF', labelpad=12)
    ax4.set_yticks(range(4))
    ax4.set_yticklabels(regime_order, fontsize=12, fontweight='700', color='#E0E0E0')
    ax4.set_xlabel('Date', fontsize=15, fontweight='700', color='#FFFFFF', labelpad=12)
    ax4.grid(True, alpha=0.08, linestyle='-', linewidth=1, color='#FFFFFF', axis='x')
    ax4.tick_params(labelsize=11, colors='#B0B0B0', width=1.5)
    for spine in ax4.spines.values():
        spine.set_color('#2D3142')
        spine.set_linewidth(2)
    plt.tight_layout()
    return fig

def plot_weekly_dashboard(df_recent, ticker):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(24, 14), facecolor='#0E1117')
    gs = fig.add_gridspec(4, 1, height_ratios=[3.5, 1, 1, 1], hspace=0.4)
    
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#1A1D29')
    ax1.fill_between(df_recent.index, df_recent['Donchian_Upper'], df_recent['Donchian_Lower'], alpha=0.08, color='#00D9FF', zorder=0)
    ax1.plot(df_recent.index, df_recent['Donchian_Upper'], color='#00D9FF', alpha=0.6, linewidth=2, linestyle='--', label='Donchian Upper', zorder=2)
    ax1.plot(df_recent.index, df_recent['Donchian_Middle'], color='#00D9FF', alpha=0.8, linewidth=2.5, linestyle='-', label='Donchian Middle', zorder=2)
    ax1.plot(df_recent.index, df_recent['Donchian_Lower'], color='#00D9FF', alpha=0.6, linewidth=2, linestyle='--', label='Donchian Lower', zorder=2)
    ax1.plot(df_recent.index, df_recent['Close'], color='#FFFFFF', alpha=0.15, linewidth=6, zorder=1)
    ax1.plot(df_recent.index, df_recent['Close'], color='#E0E0E0', alpha=0.4, linewidth=3, zorder=2)
    ax1.plot(df_recent.index, df_recent['Close'], color='#FFFFFF', linewidth=1.5, zorder=3)
    
    for regime_name, color in REGIME_COLORS_WEEKLY.items():
        mask = df_recent['Regime_Weekly'] == regime_name
        if mask.sum() > 0:
            ax1.scatter(df_recent[mask].index, df_recent[mask]['Close'], c=color, alpha=0.15, s=220, edgecolors='none', zorder=4)
            ax1.scatter(df_recent[mask].index, df_recent[mask]['Close'], c=color, label=regime_name, alpha=0.95, s=85, edgecolors='white', linewidth=1.5, zorder=5)
    
    ax1.plot(df_recent.index, df_recent['SMA_20'], color='#50FA7B', alpha=0.9, linewidth=2.5, label='SMA(20) Weekly', zorder=3)
    ax1.plot(df_recent.index, df_recent['SMA_50'], color='#BD93F9', alpha=0.9, linewidth=2.5, label='SMA(50) Weekly', zorder=3)
    
    current = df_recent.iloc[-1]
    ax1.scatter(current.name, current['Close'], facecolors='none', edgecolors=REGIME_COLORS_WEEKLY[current['Regime_Weekly']], s=450, linewidth=5, alpha=0.4, marker='o', zorder=9)
    ax1.scatter(current.name, current['Close'], facecolors=REGIME_COLORS_WEEKLY[current['Regime_Weekly']], edgecolors='white', s=280, linewidth=4, marker='o', label=f'📍 Actual: {current["Regime_Weekly"]}', zorder=10)
    
    bbox_color = REGIME_COLORS_WEEKLY[current['Regime_Weekly']]
    ax1.annotate(f'{current["Regime_Weekly"]}\n${current["Close"]:.2f}', xy=(current.name, current['Close']), xytext=(25, 40), textcoords='offset points', fontsize=14, fontweight='bold', color='white', bbox=dict(boxstyle='round,pad=1', facecolor=bbox_color, alpha=0.95, edgecolor='white', linewidth=3), arrowprops=dict(arrowstyle='->', lw=3, color=bbox_color, connectionstyle='arc3,rad=0.3'), zorder=11)
    
    ax1.text(0.5, 1.10, f'{ticker}', transform=ax1.transAxes, fontsize=28, fontweight='bold', ha='center', color='#FFFFFF')
    ax1.text(0.5, 1.05, 'Weekly Market Regime Analysis', transform=ax1.transAxes, fontsize=13, style='italic', ha='center', color='#8E93A1')
    ax1.set_ylabel('Price ($)', fontsize=15, fontweight='700', color='#FFFFFF', labelpad=12)
    legend = ax1.legend(loc='upper left', fontsize=10, framealpha=0.95, ncol=3, edgecolor='#00D9FF', fancybox=True, borderpad=1.2, labelspacing=1, columnspacing=2)
    legend.get_frame().set_facecolor('#1A1D29')
    legend.get_frame().set_linewidth(2)
    ax1.grid(True, alpha=0.08, linestyle='-', linewidth=1, color='#FFFFFF')
    ax1.tick_params(labelsize=11, colors='#B0B0B0', width=1.5)
    for spine in ax1.spines.values():
        spine.set_color('#2D3142')
        spine.set_linewidth(2)
    
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_facecolor('#1A1D29')
    ax2.plot(df_recent.index, df_recent['Choppiness'], color='#FFD93D', linewidth=2.5, label='Choppiness Index', zorder=3)
    ax2.fill_between(df_recent.index, df_recent['Choppiness'], 50, where=(df_recent['Choppiness'] >= 61.8), color='#FFD93D', alpha=0.2, zorder=1)
    ax2.fill_between(df_recent.index, df_recent['Choppiness'], 50, where=(df_recent['Choppiness'] <= 38.2), color='#4ECDC4', alpha=0.2, zorder=1)
    ax2.axhline(y=61.8, color='#FFD93D', linestyle='--', linewidth=2, alpha=0.8, label='Choppy > 61.8', zorder=2)
    ax2.axhline(y=50, color='#8E93A1', linestyle='-', linewidth=1.5, alpha=0.7, zorder=2)
    ax2.axhline(y=38.2, color='#4ECDC4', linestyle='--', linewidth=2, alpha=0.8, label='Trending < 38.2', zorder=2)
    ax2.fill_between(df_recent.index, 61.8, 100, alpha=0.12, color='#FFD93D', zorder=0)
    ax2.fill_between(df_recent.index, 0, 38.2, alpha=0.12, color='#4ECDC4', zorder=0)
    ax2.scatter(current.name, current['Choppiness'], facecolors='#FFD93D', edgecolors='white', s=220, linewidth=4, marker='o', zorder=10)
    chop_color = '#FFD93D' if current['Choppiness'] > 61.8 else '#4ECDC4' if current['Choppiness'] < 38.2 else '#8E93A1'
    chop_status = 'Choppy' if current['Choppiness'] > 61.8 else 'Trending' if current['Choppiness'] < 38.2 else 'Neutral'
    ax2.text(0.02, 0.88, f'Chop: {current["Choppiness"]:.1f} ({chop_status})', transform=ax2.transAxes, fontsize=13, fontweight='bold', color='white', verticalalignment='top', bbox=dict(boxstyle='round,pad=0.7', facecolor=chop_color, alpha=0.95, edgecolor='white', linewidth=2.5))
    ax2.set_ylabel('Choppiness', fontsize=14, fontweight='700', color='#FFFFFF', labelpad=12)
    ax2.set_ylim([0, 100])
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax2.grid(True, alpha=0.08, linestyle='-', linewidth=1, color='#FFFFFF')
    ax2.tick_params(labelsize=10.5, colors='#B0B0B0', width=1.5)
    for spine in ax2.spines.values():
        spine.set_color('#2D3142')
        spine.set_linewidth(2)
    
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.set_facecolor('#1A1D29')
    if 'MACD_V_Signal' in df_recent.columns and not df_recent['MACD_V_Signal'].isna().all():
        for i in range(1, len(df_recent)):
            if pd.notna(df_recent['MACD_V_Signal'].iloc[i-1]) and pd.notna(df_recent['MACD_V_Signal'].iloc[i]):
                x1, x2 = df_recent.index[i-1], df_recent.index[i]
                y1, y2 = df_recent['MACD_V_Signal'].iloc[i-1], df_recent['MACD_V_Signal'].iloc[i]
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
                ax3.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=0.95, zorder=5)
    ax3.axhline(y=150, color='#FF6B6B', linestyle='--', linewidth=2, alpha=0.8, label='Risk > 150')
    ax3.axhline(y=0, color='#8E93A1', linestyle='-', linewidth=1.5, alpha=0.7)
    ax3.axhline(y=-150, color='#FF6B6B', linestyle='--', linewidth=2, alpha=0.8, label='Risk < -150')
    ax3.fill_between(df_recent.index, 150, 300, alpha=0.12, color='#FF6B6B', zorder=0)
    ax3.fill_between(df_recent.index, -300, -150, alpha=0.12, color='#FF6B6B', zorder=0)
    macd_color = '#FF6B6B' if abs(current['MACD_V_Signal']) > 150 else '#4ECDC4' if current['MACD_V_Signal'] > 0 else '#EE5A6F'
    ax3.text(0.02, 0.88, f'MACD-V: {current["MACD_V_Signal"]:.1f}', transform=ax3.transAxes, fontsize=13, fontweight='bold', color='white', verticalalignment='top', bbox=dict(boxstyle='round,pad=0.7', facecolor=macd_color, alpha=0.95, edgecolor='white', linewidth=2.5))
    ax3.set_ylabel('MACD-V', fontsize=14, fontweight='700', color='#FFFFFF', labelpad=12)
    ax3.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax3.grid(True, alpha=0.08, linestyle=':', linewidth=1, color='#FFFFFF')
    ax3.tick_params(labelsize=10.5, colors='#B0B0B0', width=1.5)
    for spine in ax3.spines.values():
        spine.set_color('#2D3142')
        spine.set_linewidth(2)
    
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.set_facecolor('#1A1D29')
    regime_order = ['DOWNTREND', 'SIDEWAYS', 'UPTREND', 'RIESGO']
    for regime_name in regime_order:
        mask = df_recent['Regime_Weekly'] == regime_name
        if mask.sum() > 0:
            color = REGIME_COLORS_WEEKLY[regime_name]
            regime_num = regime_order.index(regime_name)
            ax4.scatter(df_recent[mask].index, [regime_num] * mask.sum(), c=color, alpha=0.95, s=110, edgecolors='white', linewidth=1.8, zorder=4)
    ax4.set_ylabel('Regime', fontsize=14, fontweight='700', color='#FFFFFF', labelpad=12)
    ax4.set_yticks(range(4))
    ax4.set_yticklabels(regime_order, fontsize=12, fontweight='700', color='#E0E0E0')
    ax4.set_xlabel('Date', fontsize=15, fontweight='700', color='#FFFFFF', labelpad=12)
    ax4.grid(True, alpha=0.08, linestyle='-', linewidth=1, color='#FFFFFF', axis='x')
    ax4.tick_params(labelsize=11, colors='#B0B0B0', width=1.5)
    for spine in ax4.spines.values():
        spine.set_color('#2D3142')
        spine.set_linewidth(2)
    plt.tight_layout()
    return fig

def market_regime_page():
    st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stMetric { background: linear-gradient(135deg, #1A1D29 0%, #2D3142 100%); padding: 20px; border-radius: 15px; border: 2px solid #00D9FF; box-shadow: 0 4px 15px rgba(0, 217, 255, 0.2); }
    .stMetric label { color: #00D9FF !important; font-weight: 700 !important; font-size: 14px !important; }
    .stMetric [data-testid="stMetricValue"] { color: #FFFFFF !important; font-size: 24px !important; font-weight: 800 !important; }
    .stButton>button { background: linear-gradient(135deg, #4ECDC4 0%, #00D9FF 100%); color: white; font-weight: 700; border: none; padding: 12px 24px; border-radius: 10px; box-shadow: 0 4px 15px rgba(78, 205, 196, 0.4); transition: all 0.3s ease; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(78, 205, 196, 0.6); }
    h1, h2, h3 { color: #FFFFFF !important; font-weight: 800 !important; }
    .stAlert { border-radius: 12px; border-left: 5px solid #00D9FF; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("📊 Market Regime Analyzer Pro")
    st.markdown("---")
    
    with st.sidebar:
        st.markdown("""<div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #4ECDC4 0%, #00D9FF 100%); border-radius: 15px; margin-bottom: 20px;'><h2 style='color: white; margin: 0;'>⚙️ Settings</h2></div>""", unsafe_allow_html=True)
        ticker = st.text_input("🎯 Ticker Symbol", value="AAPL", help="Ingresa el símbolo del ticker").upper()
        st.markdown("---")
        lookback_months = st.slider("📅 Meses a Visualizar", min_value=6, max_value=24, value=6, step=1)
        lookback_days = int(lookback_months * 21)
        lookback_weeks = int(lookback_months * 4.33)
        st.markdown("---")
        start_date = st.date_input("📆 Fecha Inicio de Datos", value=datetime(2018, 1, 1))
        st.markdown("---")
        analizar_btn = st.button("🚀 ANALIZAR MERCADO", type="primary", use_container_width=True)
        st.markdown("---")
        st.markdown("""<div style='background: linear-gradient(135deg, #1A1D29 0%, #2D3142 100%); padding: 15px; border-radius: 12px; border: 2px solid #BD93F9;'><h3 style='color: #BD93F9; margin-top: 0;'>📖 Metodología</h3></div>""", unsafe_allow_html=True)
        st.markdown("""
        ### 📊 ANÁLISIS DIARIO
        - **ADX**: Fuerza tendencia (>25)
        - **RSI**: Momentum (75/25 extremos)
        - **SMAs**: 20 y 50 períodos
        - **Regímenes**: ALCISTA, BAJISTA, RANGO, RIESGO
        
        ### 📈 ANÁLISIS SEMANAL
        - **Donchian**: Canal 20 períodos
        - **Choppiness**: Índice choppy (61.8/38.2)
        - **MACD-V**: Normalizado (±150 riesgo)
        - **SMAs**: 20 y 50 períodos semanales
        - **Regímenes**: UPTREND, DOWNTREND, SIDEWAYS, RIESGO
        """)
    
    if analizar_btn:
        with st.spinner(f"🔄 Descargando datos para {ticker}..."):
            df_daily = download_daily_data(ticker, start_date.strftime('%Y-%m-%d'))
            df_weekly = download_weekly_data(ticker, start_date.strftime('%Y-%m-%d'))
            if df_daily is None or df_weekly is None:
                st.error(f"❌ No se pudieron obtener datos para {ticker}")
                st.stop()
            df_daily['Regime_Daily'] = classify_regime_daily(df_daily)
            df_weekly['Regime_Weekly'] = classify_regime_weekly(df_weekly)
            st.session_state['ticker'] = ticker
            st.session_state['df_daily'] = df_daily
            st.session_state['df_weekly'] = df_weekly
            st.session_state['lookback_months'] = lookback_months
            st.success(f"✅ Datos cargados exitosamente para {ticker}")
    
    if 'df_daily' in st.session_state and 'df_weekly' in st.session_state:
        df_daily = st.session_state['df_daily']
        df_weekly = st.session_state['df_weekly']
        ticker = st.session_state['ticker']
        df_daily_recent = df_daily.tail(lookback_days)
        df_weekly_recent = df_weekly.tail(lookback_weeks)
        current_daily = df_daily.iloc[-1]
        current_weekly = df_weekly.iloc[-1]
        
        st.markdown("---")
        st.markdown("""<div style='text-align: center; margin-bottom: 20px;'><h2 style='color: #00D9FF; font-size: 32px; margin: 0;'>📈 ANÁLISIS DIARIO</h2></div>""", unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            regime_emoji = {'ALCISTA': '🟢', 'BAJISTA': '🔴', 'RANGO': '⚫', 'RIESGO': '🟠'}[current_daily['Regime_Daily']]
            st.metric("RÉGIMEN DIARIO", f"{regime_emoji} {current_daily['Regime_Daily']}")
        with col2:
            price_change = ((current_daily['Close'] - df_daily['Close'].iloc[-5]) / df_daily['Close'].iloc[-5] * 100)
            st.metric("PRECIO", f"${current_daily['Close']:.2f}", f"{price_change:+.2f}%")
        with col3:
            adx_status = "🔥 Fuerte" if current_daily['ADX'] > 25 else "⚡ Moderado" if current_daily['ADX'] > 20 else "💤 Débil"
            st.metric("ADX", f"{current_daily['ADX']:.1f}", adx_status)
        with col4:
            rsi_status = "🔴 OB" if current_daily['RSI'] > 70 else "🔵 OS" if current_daily['RSI'] < 30 else "⚪ Neutral"
            st.metric("RSI", f"{current_daily['RSI']:.1f}", rsi_status)
        with col5:
            sma_status = "📈 Alcista" if current_daily['Close'] > current_daily['SMA_20'] > current_daily['SMA_50'] else "📉 Bajista" if current_daily['Close'] < current_daily['SMA_20'] < current_daily['SMA_50'] else "↔️ Mixto"
            st.metric("TENDENCIA SMAs", sma_status)
        with col6:
            st.metric("FECHA", current_daily.name.strftime('%Y-%m-%d'), current_daily.name.strftime('%A')[:3])
        
        st.markdown("---")
        st.markdown(f"""<div style='text-align: center; margin-bottom: 20px;'><h2 style='color: #BD93F9; font-size: 28px;'>📊 Dashboard Técnico Diario</h2><p style='color: #8E93A1; font-size: 16px;'>Últimos {lookback_months} meses • Timeframe: Daily (1d)</p></div>""", unsafe_allow_html=True)
        fig_daily = plot_daily_dashboard(df_daily_recent, ticker)
        st.pyplot(fig_daily)
        
        st.markdown("---")
        st.markdown("---")
        
        st.markdown("""<div style='text-align: center; margin-bottom: 20px;'><h2 style='color: #4ECDC4; font-size: 32px; margin: 0;'>📊 ANÁLISIS SEMANAL</h2></div>""", unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            regime_emoji = {'UPTREND': '🟢', 'DOWNTREND': '🔴', 'SIDEWAYS': '🟡', 'RIESGO': '🟠'}[current_weekly['Regime_Weekly']]
            st.metric("RÉGIMEN SEMANAL", f"{regime_emoji} {current_weekly['Regime_Weekly']}")
        with col2:
            st.metric("PRECIO", f"${current_weekly['Close']:.2f}")
        with col3:
            chop_status = "🟡 Choppy" if current_weekly['Choppiness'] > 61.8 else "🟢 Trending" if current_weekly['Choppiness'] < 38.2 else "⚪ Neutral"
            st.metric("CHOPPINESS", f"{current_weekly['Choppiness']:.1f}", chop_status)
        with col4:
            macd_status = "🟠 Extremo" if abs(current_weekly['MACD_V_Signal']) > 150 else "🟢 Alcista" if current_weekly['MACD_V_Signal'] > 0 else "🔴 Bajista"
            st.metric("MACD-V", f"{current_weekly['MACD_V_Signal']:.1f}", macd_status)
        with col5:
            donch_pos = ((current_weekly['Close'] - current_weekly['Donchian_Lower']) / (current_weekly['Donchian_Upper'] - current_weekly['Donchian_Lower']) * 100)
            st.metric("POSICIÓN DONCHIAN", f"{donch_pos:.0f}%")
        with col6:
            st.metric("FECHA", current_weekly.name.strftime('%Y-%m-%d'), "Semanal")
        
        st.markdown("---")
        st.markdown(f"""<div style='text-align: center; margin-bottom: 20px;'><h2 style='color: #BD93F9; font-size: 28px;'>📊 Dashboard Técnico Semanal</h2><p style='color: #8E93A1; font-size: 16px;'>Últimos {lookback_months} meses • Timeframe: Weekly (1w)</p></div>""", unsafe_allow_html=True)
        fig_weekly = plot_weekly_dashboard(df_weekly_recent, ticker)
        st.pyplot(fig_weekly)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            export_cols_daily = ['Close', 'Regime_Daily', 'ADX', 'RSI', 'Plus_DI', 'Minus_DI', 'SMA_20', 'SMA_50']
            csv_daily = df_daily[export_cols_daily].to_csv()
            st.download_button("📥 Descargar Datos SEMANALES (CSV)", data=csv_weekly, file_name=f"regime_weekly_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", use_container_width=True)
    else:
        st.markdown("""
        <div style='text-align: center; padding: 40px; background: linear-gradient(135deg, #1A1D29 0%, #2D3142 100%); 
                    border-radius: 20px; border: 3px solid #4ECDC4; margin-top: 30px;
                    box-shadow: 0 8px 30px rgba(78, 205, 196, 0.3);'>
            <h2 style='color: #4ECDC4; margin: 0;'>👋 Bienvenido al Market Regime Analyzer Pro</h2>
            <p style='color: #B0B0B0; font-size: 18px; margin: 20px 0;'>
                Configura los parámetros en el panel lateral y presiona 
                <strong style='color: #00D9FF;'>'🚀 ANALIZAR MERCADO'</strong> 
                para comenzar el análisis técnico avanzado.
            </p>
            <p style='color: #8E93A1; font-size: 14px; margin: 10px 0 0 0;'>
                📈 Análisis Diario: ADX + RSI + SMAs | 📊 Análisis Semanal: Donchian + Choppiness + MACD-V
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    if check_password():
        market_regime_page()
    else:
        st.markdown("""
        <div style='text-align: center; padding: 60px 20px;'>
            <h1 style='color: #FF6B6B; font-size: 48px;'>🔒 Acceso Restringido</h1>
            <p style='color: #B0B0B0; font-size: 20px; margin-top: 20px;'>
                Introduce tus credenciales en el menú lateral para acceder al análisis.
            </p>
        </div>
        """, unsafe_allow_html=True) Datos DIARIOS (CSV)", data=csv_daily, file_name=f"regime_daily_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", use_container_width=True)
        with col2:
            export_cols_weekly = ['Close', 'Regime_Weekly', 'Choppiness', 'MACD_V_Signal', 'Donchian_Upper', 'Donchian_Middle', 'Donchian_Lower', 'SMA_20', 'SMA_50']
            csv_weekly = df_weekly[export_cols_weekly].to_csv()
            st.download_button("📥 Descargar
