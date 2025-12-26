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
        if rsi >= 75 or rsi <= 25:
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
        elif chop < 38.2:
            if price > donchian_middle and macd_v > 0:
                regime = 'UPTREND'
            elif price < donchian_middle and macd_v < 0:
                regime = 'DOWNTREND'
            else:
                regime = 'SIDEWAYS'
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
    ax2.axhline(y=61.8, color='#FFD93D', linestyle='--', linewidth=2, alpha=0.8, label='Choppy > 61.8', zorder=2)
    ax2.axhline(y=50, color='#8E93A1', linestyle='-', linewidth=1.5, alpha=0.7, zorder=2)
    ax2.axhline(y=38.2, color='#4ECDC4', linestyle='--', linewidth=2, alpha=0.8, label='Trending < 38.2', zorder=2)
    ax2.scatter(current.name, current['Choppiness'], facecolors='#FFD93D', edgecolors='white', s=220, linewidth=4, marker='o', zorder=10)
    chop_color = '#FFD93D' if current['Choppiness'] > 61.8 else '#4ECDC4' if current['Choppiness'] < 38.2 else '#8E93A1'
    chop_status = 'Choppy' if current['Choppiness'] > 61.8 else 'Trending' if current['Choppiness'] < 38.2 else 'Neutral'
    ax2.text(0.02, 0.88, f'Chop: {current["Choppiness"]:.1f} ({chop_status})', transform=ax2.transAxes, fontsize=13, fontweight='bold', color='white', verticalalignment='top', bbox=dict(boxstyle='round,pad=0.7', facecolor=chop_color, alpha=0.95, edgecolor='white', linewidth=2.5))
    ax2.set_ylabel('Choppiness', fontsize=14, fontweight='700', color='#FFFFFF', labelpad=12)
    ax2.set_ylim([0, 100])
    ax2.grid(True, alpha=0.08, linestyle='-', linewidth=1, color='#FFFFFF')
    ax2.tick_params(labelsize=10.5, colors='#B0B0B0', width=1.5)
    for spine in ax2.spines.values():
        spine.set_color('#2D3142')
        spine.set_linewidth(2)
    
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.set_facecolor('#1A1D29')
    ax3.plot(df_recent.index, df_recent['MACD_V_Signal'], color='#4ECDC4', linewidth=2.5, label='MACD-V', zorder=5)
    ax3.axhline(y=150, color='#FF6B6B', linestyle='--', linewidth=2, alpha=0.8, label='Risk > 150')
    ax3.axhline(y=0, color='#8E93A1', linestyle='-', linewidth=1.5, alpha=0.7)
    ax3.axhline(y=-150, color='#FF6B6B', linestyle='--', linewidth=2, alpha=0.8, label='Risk < -150')
    macd_color = '#FF6B6B' if abs(current['MACD_V_Signal']) > 150 else '#4ECDC4' if current['MACD_V_Signal'] > 0 else '#EE5A6F'
    ax3.text(0.02, 0.88, f'MACD-V: {current["MACD_V_Signal"]:.1f}', transform=ax3.transAxes, fontsize=13, fontweight='bold', color='white', verticalalignment='top', bbox=dict(boxstyle='round,pad=0.7', facecolor=macd_color, alpha=0.95, edgecolor='white', linewidth=2.5))
    ax3.set_ylabel('MACD-V', fontsize=14, fontweight='700', color='#FFFFFF', labelpad=12)
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
    h1, h2, h3 { color: #FFFFFF !important; font-weight: 800 !important; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("📊 Market Regime Analyzer Pro")
    st.markdown("---")
    
    with st.sidebar:
        st.header("⚙️ Settings")
        ticker = st.text_input("🎯 Ticker Symbol", value="AAPL").upper()
        lookback_months = st.slider("📅 Meses a Visualizar", 6, 24, 6)
        start_date = st.date_input("📆 Fecha Inicio de Datos", value=datetime(2018, 1, 1))
        analizar_btn = st.button("🚀 ANALIZAR MERCADO", type="primary", use_container_width=True)
    
    if analizar_btn:
        with st.spinner(f"🔄 Descargando datos para {ticker}..."):
            df_daily = download_daily_data(ticker, start_date.strftime('%Y-%m-%d'))
            df_weekly = download_weekly_data(ticker, start_date.strftime('%Y-%m-%d'))
            if df_daily is not None and df_weekly is not None:
                df_daily['Regime_Daily'] = classify_regime_daily(df_daily)
                df_weekly['Regime_Weekly'] = classify_regime_weekly(df_weekly)
                st.session_state['ticker'] = ticker
                st.session_state['df_daily'] = df_daily
                st.session_state['df_weekly'] = df_weekly
                st.session_state['lookback_months'] = lookback_months
                st.success(f"✅ Datos cargados para {ticker}")
            else:
                st.error("❌ Error al cargar datos")

    if 'df_daily' in st.session_state:
        df_daily = st.session_state['df_daily']
        df_weekly = st.session_state['df_weekly']
        ticker = st.session_state['ticker']
        lookback_months = st.session_state['lookback_months']
        
        # Dashboard Diario
        st.subheader("📈 ANÁLISIS DIARIO")
        current_daily = df_daily.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("RÉGIMEN", current_daily['Regime_Daily'])
        col2.metric("PRECIO", f"${current_daily['Close']:.2f}")
        col3.metric("ADX", f"{current_daily['ADX']:.1f}")
        col4.metric("RSI", f"{current_daily['RSI']:.1f}")
        
        st.pyplot(plot_daily_dashboard(df_daily.tail(lookback_months * 21), ticker))
        
        # Dashboard Semanal
        st.markdown("---")
        st.subheader("📊 ANÁLISIS SEMANAL")
        current_weekly = df_weekly.iloc[-1]
        wcol1, wcol2, wcol3, wcol4 = st.columns(4)
        wcol1.metric("RÉGIMEN", current_weekly['Regime_Weekly'])
        wcol2.metric("CHOPPINESS", f"{current_weekly['Choppiness']:.1f}")
        wcol3.metric("MACD-V", f"{current_weekly['MACD_V_Signal']:.1f}")
        wcol4.metric("FECHA", current_weekly.name.strftime('%Y-%m-%d'))
        
        st.pyplot(plot_weekly_dashboard(df_weekly.tail(lookback_months * 4), ticker))
        
        # Sección de descarga
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            csv_d = df_daily.to_csv().encode('utf-8')
            st.download_button("📥 Descargar Datos Diarios", data=csv_d, file_name=f"{ticker}_daily.csv", mime="text/csv")
        with c2:
            csv_w = df_weekly.to_csv().encode('utf-8')
            st.download_button("📥 Descargar Datos Semanales", data=csv_w, file_name=f"{ticker}_weekly.csv", mime="text/csv")

if __name__ == "__main__":
    if check_password():
        market_regime_page()
    else:
        st.warning("Introduce tus credenciales en el menú lateral.")
