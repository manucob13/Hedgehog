# pages/GEX.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta, datetime, date
from scipy.stats import norm
from utils import check_password
import schwab
from schwab.auth import easy_client
from schwab.client import Client
import json
import os

# =========================================================================
# 0. CONFIGURACIÓN
# =========================================================================

st.set_page_config(page_title="GEX Scanner", layout="wide")

# Cargar variables de Schwab desde secrets
try:
    api_key = st.secrets["schwab"]["api_key"]
    app_secret = st.secrets["schwab"]["app_secret"]
    redirect_uri = st.secrets["schwab"]["redirect_uri"]
except KeyError as e:
    st.error(f"❌ Falta configurar los secrets de Schwab. Clave faltante: {e}")
    st.stop()

token_path = "schwab_token.json"

# =========================================================================
# 1. FUNCIONES AUXILIARES
# =========================================================================

def connect_to_schwab():
    """Conecta con Schwab usando el token existente"""
    if not os.path.exists(token_path):
        st.error("❌ No se encontró 'schwab_token.json'")
        return None
    
    try:
        client = easy_client(
            api_key=api_key,
            app_secret=app_secret,
            callback_url=redirect_uri,
            token_path=token_path
        )
        
        # Verificar token con un símbolo simple
        test_response = client.get_quote("SPY")
        if hasattr(test_response, "status_code") and test_response.status_code != 200:
            raise Exception(f"Respuesta inesperada: {test_response.status_code}")
        
        return client
    except Exception as e:
        st.error(f"❌ Error al conectar con Schwab: {e}")
        return None


def format_ticker_for_schwab(ticker):
    """Formatea el ticker según los requisitos de Schwab"""
    # Para índices, agregar $ si no lo tiene
    indices = ['SPX', 'NDX', 'RUT', 'DJX', 'VIX']
    ticker_upper = ticker.upper()
    
    if ticker_upper in indices and not ticker.startswith('$'):
        return f"${ticker_upper}"
    
    return ticker_upper


def calcGammaEx(S, K, vol, T, r, q, optType, OI):
    """Calcula Gamma Exposure basado en Black-Scholes"""
    if T == 0 or vol == 0:
        return 0
    
    try:
        dp = (np.log(S/K) + (r - q + 0.5*vol**2)*T) / (vol*np.sqrt(T))
        dm = dp - vol*np.sqrt(T)
        
        if optType == 'call':
            gamma = np.exp(-q*T) * norm.pdf(dp) / (S * vol * np.sqrt(T))
        else:
            gamma = K * np.exp(-r*T) * norm.pdf(dm) / (S * S * vol * np.sqrt(T))
        
        return OI * 100 * S * S * 0.01 * gamma
    except:
        return 0


def isThirdFriday(d):
    """Detecta si una fecha es el tercer viernes del mes"""
    return d.weekday() == 4 and 15 <= d.day <= 21


def get_options_data_schwab(client, ticker):
    """
    Obtiene datos de opciones desde Schwab con filtros para reducir el tamaño de la respuesta
    """
    try:
        formatted_ticker = format_ticker_for_schwab(ticker)
        
        # Obtener spot price
        quote_response = client.get_quote(formatted_ticker)
        if quote_response.status_code != 200:
            st.error(f"Error en quote: Status {quote_response.status_code}")
            st.error(f"Respuesta: {quote_response.text}")
            return None, None
        
        quote_data = quote_response.json()
        
        # Manejar diferentes formatos de respuesta
        if formatted_ticker in quote_data:
            spot_price = quote_data[formatted_ticker]['quote']['lastPrice']
        elif ticker in quote_data:
            spot_price = quote_data[ticker]['quote']['lastPrice']
        else:
            st.error(f"No se encontró el ticker en la respuesta: {list(quote_data.keys())}")
            return None, None
        
        # Calcular fechas para filtrar (solo próximos 45 días)
        from_date = date.today()
        to_date = date.today() + timedelta(days=45)
        
        # Obtener cadena de opciones con FILTROS MÍNIMOS
        options_response = client.get_option_chain(
            formatted_ticker,
            contract_type=Client.Options.ContractType.ALL,
            strike_count=40,
            include_underlying_quote=False,
            strategy=Client.Options.Strategy.SINGLE,
            from_date=from_date,
            to_date=to_date
        )
        
        if options_response.status_code != 200:
            st.error(f"Error en options: Status {options_response.status_code}")
            st.error(f"Respuesta: {options_response.text}")
            return None, None
        
        options_data = options_response.json()
        
        return spot_price, options_data
    
    except Exception as e:
        st.error(f"Error obteniendo datos: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None


def process_schwab_options(options_data, spot_price):
    """Procesa los datos de opciones de Schwab en formato DataFrame"""
    
    all_options = []
    
    # Procesar calls
    call_map = options_data.get('callExpDateMap', {})
    for exp_date, strikes in call_map.items():
        exp_date_clean = exp_date.split(':')[0]
        for strike, contracts in strikes.items():
            contract = contracts[0]
            
            call_iv = contract.get('volatility', 0)
            if call_iv > 2:
                call_iv = call_iv / 100
            
            all_options.append({
                'ExpirationDate': exp_date_clean,
                'StrikePrice': float(strike),
                'CallPut': 'C',
                'CallOpenInt': contract.get('openInterest', 0),
                'CallIV': call_iv,
                'CallGamma': contract.get('gamma', 0),
                'CallVol': contract.get('totalVolume', 0),
                'PutOpenInt': 0,
                'PutIV': 0,
                'PutGamma': 0,
                'PutVol': 0
            })
    
    # Procesar puts
    put_map = options_data.get('putExpDateMap', {})
    for exp_date, strikes in put_map.items():
        exp_date_clean = exp_date.split(':')[0]
        for strike, contracts in strikes.items():
            contract = contracts[0]
            
            put_iv = contract.get('volatility', 0)
            if put_iv > 2:
                put_iv = put_iv / 100
            
            found = False
            for opt in all_options:
                if opt['StrikePrice'] == float(strike) and opt['ExpirationDate'] == exp_date_clean:
                    opt['PutOpenInt'] = contract.get('openInterest', 0)
                    opt['PutIV'] = put_iv
                    opt['PutGamma'] = contract.get('gamma', 0)
                    opt['PutVol'] = contract.get('totalVolume', 0)
                    found = True
                    break
            
            if not found:
                all_options.append({
                    'ExpirationDate': exp_date_clean,
                    'StrikePrice': float(strike),
                    'CallPut': 'P',
                    'CallOpenInt': 0,
                    'CallIV': 0,
                    'CallGamma': 0,
                    'CallVol': 0,
                    'PutOpenInt': contract.get('openInterest', 0),
                    'PutIV': put_iv,
                    'PutGamma': contract.get('gamma', 0),
                    'PutVol': contract.get('totalVolume', 0)
                })
    
    df = pd.DataFrame(all_options)
    
    if df.empty:
        return df
    
    df['ExpirationDate'] = pd.to_datetime(df['ExpirationDate'])
    today = date.today()
    df['daysTillExp'] = df['ExpirationDate'].apply(
        lambda x: 1/262 if np.busday_count(today, x.date()) == 0 
        else np.busday_count(today, x.date())/262
    )
    
    # Calcular GEX
    mult = 100 * 100
    df['CallGEX'] = df['CallGamma'] * df['CallOpenInt'] * mult
    df['PutGEX'] = df['PutGamma'] * df['PutOpenInt'] * mult * -1
    df['TotalGamma'] = (df['CallGEX'] + df['PutGEX']) / 10**9
    
    df['call_gex'] = df['CallGamma'] * df['CallOpenInt'] * mult
    df['put_gex'] = df['PutGamma'] * df['PutOpenInt'] * mult
    df['net_gex'] = df['call_gex'] - df['put_gex']
    df['total_oi'] = df['CallOpenInt'] + df['PutOpenInt']
    
    return df


def plot_total_gamma(df, spot_price, ticker, width):
    """Gráfico de Gamma Exposure Total - MEJORADO CON FILTROS"""
    fromStrike = spot_price - width
    toStrike = spot_price + width
    
    # Filtrar solo strikes en el rango relevante
    df_range = df[(df['StrikePrice'] >= fromStrike) & (df['StrikePrice'] <= toStrike)]
    dfAgg = df_range.groupby(['StrikePrice']).sum(numeric_only=True)
    
    # Filtrar strikes con OI mínimo para limpiar el gráfico
    min_oi = dfAgg['total_oi'].quantile(0.10)  # Top 90% de OI
    dfAgg = dfAgg[dfAgg['total_oi'] >= min_oi]
    
    strikes = dfAgg.index.values
    
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Barras más anchas y limpias
    bar_width = (toStrike - fromStrike) / len(strikes) * 0.8
    ax.bar(strikes, dfAgg['TotalGamma'].to_numpy(), width=bar_width, 
           linewidth=0, edgecolor='none', color='steelblue', alpha=0.7,
           label="Gamma Exposure")
    
    ax.set_xlim([fromStrike, toStrike])
    
    chartTitle = f"Total Gamma: ${df['TotalGamma'].sum():.2f} Bn per 1% {ticker} Move"
    ax.set_title(chartTitle, fontweight="bold", fontsize=18, pad=20)
    ax.set_xlabel('Strike', fontweight="bold", fontsize=14)
    ax.set_ylabel('Spot Gamma Exposure ($ billions/1% move)', fontweight="bold", fontsize=14)
    ax.axvline(x=spot_price, color='red', lw=2, label=f"{ticker} Spot: {spot_price:,.0f}", linestyle='--')
    
    # Resaltar strikes clave
    max_gamma_idx = dfAgg['TotalGamma'].idxmax()
    ax.scatter([max_gamma_idx], [dfAgg.loc[max_gamma_idx, 'TotalGamma']], 
               color='green', s=200, zorder=5, marker='*', label=f'Max Gamma: {max_gamma_idx:.0f}')
    
    ax.legend(fontsize=12, loc='upper right')
    ax.tick_params(labelsize=11)
    
    return fig


def plot_open_interest(df, spot_price, ticker, width):
    """Gráfico de Open Interest - MEJORADO"""
    fromStrike = spot_price - width
    toStrike = spot_price + width
    
    # Filtrar rango y agregar
    df_range = df[(df['StrikePrice'] >= fromStrike) & (df['StrikePrice'] <= toStrike)]
    dfAgg = df_range.groupby(['StrikePrice']).sum(numeric_only=True)
    
    # Filtrar OI bajo para limpiar
    min_oi = dfAgg['total_oi'].quantile(0.05)
    dfAgg = dfAgg[dfAgg['total_oi'] >= min_oi]
    
    strikes = dfAgg.index.values
    
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    bar_width = (toStrike - fromStrike) / len(strikes) * 0.8
    
    ax.bar(strikes, dfAgg['CallOpenInt'].to_numpy(), width=bar_width,
           linewidth=0, color='green', alpha=0.6, label="Call OI")
    ax.bar(strikes, -1 * dfAgg['PutOpenInt'].to_numpy(), width=bar_width,
           linewidth=0, color='red', alpha=0.6, label="Put OI")
    
    ax.set_xlim([fromStrike, toStrike])
    ax.axhline(y=0, color='black', linewidth=1, linestyle='-', alpha=0.3)
    
    ax.set_title(f"Open Interest en rango ±{width} del Spot", fontweight="bold", fontsize=18, pad=20)
    ax.set_xlabel('Strike', fontweight="bold", fontsize=14)
    ax.set_ylabel('Open Interest', fontweight="bold", fontsize=14)
    ax.axvline(x=spot_price, color='black', lw=2, linestyle='--', 
               label=f"{ticker} Spot: {spot_price:,.0f}")
    
    # Destacar strikes con mayor OI
    max_call_strike = dfAgg['CallOpenInt'].idxmax()
    max_put_strike = dfAgg['PutOpenInt'].idxmax()
    
    ax.scatter([max_call_strike], [dfAgg.loc[max_call_strike, 'CallOpenInt']], 
               color='darkgreen', s=150, zorder=5, marker='^')
    ax.scatter([max_put_strike], [-dfAgg.loc[max_put_strike, 'PutOpenInt']], 
               color='darkred', s=150, zorder=5, marker='v')
    
    ax.legend(fontsize=12, loc='upper right')
    ax.tick_params(labelsize=11)
    
    return fig


def plot_gex_profile(df, spot_price, ticker, width):
    """Gráfico de Perfil de Gamma Exposure - SIMPLIFICADO"""
    fromStrike = spot_price - width
    toStrike = spot_price + width
    levels = np.linspace(fromStrike, toStrike, 25)  # Menos puntos = más rápido
    
    todayDate = date.today()
    nextExpiry = df['ExpirationDate'].min()
    
    df['IsThirdFriday'] = df['ExpirationDate'].apply(isThirdFriday)
    thirdFridays = df.loc[df['IsThirdFriday'] == True]
    nextMonthlyExp = thirdFridays['ExpirationDate'].min() if not thirdFridays.empty else nextExpiry
    
    totalGamma = []
    totalGammaExNext = []
    
    for level in levels:
        df['callGammaEx'] = df.apply(
            lambda row: calcGammaEx(level, row['StrikePrice'], row['CallIV'],
                                   row['daysTillExp'], 0, 0, "call", row['CallOpenInt']), 
            axis=1
        )
        df['putGammaEx'] = df.apply(
            lambda row: calcGammaEx(level, row['StrikePrice'], row['PutIV'],
                                   row['daysTillExp'], 0, 0, "put", row['PutOpenInt']), 
            axis=1
        )
        
        totalGamma.append(df['callGammaEx'].sum() - df['putGammaEx'].sum())
        
        exNxt = df.loc[df['ExpirationDate'] != nextExpiry]
        totalGammaExNext.append(exNxt['callGammaEx'].sum() - exNxt['putGammaEx'].sum())
    
    totalGamma = np.array(totalGamma) / 10**9
    totalGammaExNext = np.array(totalGammaExNext) / 10**9
    
    # Encontrar punto de flip gamma
    zeroCrossIdx = np.where(np.diff(np.sign(totalGamma)))[0]
    zeroGamma = None
    
    if len(zeroCrossIdx) > 0:
        negGamma = totalGamma[zeroCrossIdx]
        posGamma = totalGamma[zeroCrossIdx+1]
        negStrike = levels[zeroCrossIdx]
        posStrike = levels[zeroCrossIdx+1]
        zeroGamma = posStrike - ((posStrike - negStrike) * posGamma/(posGamma - negGamma))
        zeroGamma = zeroGamma[0]
    
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    ax.plot(levels, totalGamma, linewidth=3, color='steelblue', label="Todas las expiraciones")
    ax.plot(levels, totalGammaExNext, linewidth=2, color='orange', 
            linestyle='--', label="Sin próxima expiración")
    
    chartTitle = f"Perfil Gamma Exposure - {ticker} - {todayDate.strftime('%d %b %Y')}"
    ax.set_title(chartTitle, fontweight="bold", fontsize=18, pad=20)
    ax.set_xlabel('Precio del Índice', fontweight="bold", fontsize=14)
    ax.set_ylabel('Gamma Exposure ($ billions/1% move)', fontweight="bold", fontsize=14)
    
    ax.axvline(x=spot_price, color='red', lw=2, linestyle='--',
               label=f"{ticker} Spot: {spot_price:,.0f}")
    
    if zeroGamma is not None:
        ax.axvline(x=zeroGamma, color='green', lw=2, linestyle=':',
                   label=f"Gamma Flip: {zeroGamma:,.0f}")
    
    ax.axhline(y=0, color='grey', lw=1.5, linestyle='-')
    ax.set_xlim([fromStrike, toStrike])
    
    # Zonas de gamma positivo/negativo
    trans = ax.get_xaxis_transform()
    flip_point = zeroGamma if zeroGamma else fromStrike
    ax.fill_between([fromStrike, flip_point], 0, 1,
                    facecolor='red', alpha=0.1, transform=trans, 
                    label='Gamma Negativo (volátil)')
    ax.fill_between([flip_point, toStrike], 0, 1,
                    facecolor='green', alpha=0.1, transform=trans,
                    label='Gamma Positivo (estable)')
    
    ax.legend(fontsize=11, loc='best')
    ax.tick_params(labelsize=11)
    
    return fig


def plot_gex_by_strike(df_filtered, spot, ticker, width):
    """Gráfico GEX por Strike - LIMPIO"""
    # Filtrar strikes con GEX significativo
    threshold = df_filtered['net_gex'].abs().quantile(0.20)
    df_sig = df_filtered[df_filtered['net_gex'].abs() >= threshold]
    
    pos = df_sig[df_sig['net_gex'] > 0]
    neg = df_sig[df_sig['net_gex'] < 0]
    
    if pos.empty and neg.empty:
        return None
    
    max_gex = df_sig.loc[df_sig['net_gex'].idxmax()]
    min_gex = df_sig.loc[df_sig['net_gex'].idxmin()]
    
    bar_width = width / 15
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    if not pos.empty:
        ax.bar(pos['StrikePrice'], pos['net_gex'], color='limegreen',
               width=bar_width, edgecolor='darkgreen', linewidth=1.5, 
               alpha=0.7, label='GEX Positivo')
    
    if not neg.empty:
        ax.bar(neg['StrikePrice'], neg['net_gex'], color='red',
               width=bar_width, edgecolor='darkred', linewidth=1.5,
               alpha=0.7, label='GEX Negativo')
    
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.axvline(spot, color='blue', linestyle='--', linewidth=2.5, alpha=0.8)
    
    # Etiquetas más claras
    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin
    
    ax.text(spot, ymax - y_range * 0.05, f'Spot: {int(spot)}', 
            ha='center', va='top', fontsize=14, color='blue', fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.5', linewidth=2))
    
    ax.text(max_gex['StrikePrice'], max_gex['net_gex'] + y_range * 0.02,
            f'{int(max_gex["StrikePrice"])}', ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='darkgreen',
            bbox=dict(facecolor='lightgreen', alpha=0.7, boxstyle='round,pad=0.3'))
    
    ax.text(min_gex['StrikePrice'], min_gex['net_gex'] - y_range * 0.02,
            f'{int(min_gex["StrikePrice"])}', ha='center', va='top',
            fontsize=12, fontweight='bold', color='darkred',
            bbox=dict(facecolor='lightcoral', alpha=0.7, boxstyle='round,pad=0.3'))
    
    ax.set_title(f'{ticker} GEX por Strike (±{width} pts del Spot)',
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Strike', fontweight='bold', fontsize=14)
    ax.set_ylabel('Net GEX', fontweight='bold', fontsize=14)
    ax.legend(fontsize=12, loc='upper right')
    ax.tick_params(labelsize=11)
    
    return fig


def plot_gamma_zones(df_filtered, spot, ticker, width):
    """Gráfico de Zonas Gamma - REDISEÑADO COMPLETAMENTE"""
    if df_filtered.empty:
        return None
    
    # Filtrar solo strikes significativos
    threshold = df_filtered['net_gex'].abs().quantile(0.15)
    df_sig = df_filtered[df_filtered['net_gex'].abs() >= threshold]
    
    max_gex = df_sig.loc[df_sig['net_gex'].idxmax()]
    min_gex = df_sig.loc[df_sig['net_gex'].idxmin()]
    
    oi_threshold = df_sig['total_oi'].quantile(0.85)
    high_oi = df_sig[df_sig['total_oi'] >= oi_threshold].sort_values('total_oi', ascending=False).head(5)
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Zona gamma (verde)
    ax.axvspan(min_gex['StrikePrice'], max_gex['StrikePrice'],
               color='lightgreen', alpha=0.25, label='Zona Gamma (consolidación)')
    
    # Líneas de alto OI (naranjas)
    for idx, (_, row) in enumerate(high_oi.iterrows()):
        strike = row['StrikePrice']
        ax.axvline(strike, color='darkorange', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(strike, 0.92 - idx * 0.05, f'{int(strike)} (OI: {int(row["total_oi"]/1000)}k)', 
                rotation=0, ha='left', va='top', fontsize=10,
                transform=ax.get_xaxis_transform(), color='darkorange',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
    
    # Spot price (línea negra gruesa)
    ax.axvline(spot, color='black', linestyle='-', linewidth=3, label=f'Spot: {int(spot)}')
    
    # Strikes clave
    ax.axvline(min_gex['StrikePrice'], color='red', linestyle=':', linewidth=2.5,
               label=f'Mín GEX: {int(min_gex["StrikePrice"])}')
    ax.axvline(max_gex['StrikePrice'], color='green', linestyle=':', linewidth=2.5,
               label=f'Máx GEX: {int(max_gex["StrikePrice"])}')
    
    ax.set_title('Zonas Clave: Gamma y Open Interest', fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Strike', fontsize=16, fontweight='bold')
    ax.set_xlim([spot - width, spot + width])
    ax.set_ylim([0, 1])
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x', linewidth=0.5)
    
    ax.legend(fontsize=13, loc='upper left', framealpha=0.95)
    ax.tick_params(axis='x', labelsize=12, rotation=45)
    
    return fig


# =========================================================================
# 2. INTERFAZ PRINCIPAL
# =========================================================================

def gex_scanner_page():
    st.title("📊 GEX Scanner - Gamma Exposure Analysis")
    st.markdown("---")
    
    # Conectar con Schwab
    if 'schwab_client' not in st.session_state:
        with st.spinner("Conectando con Schwab..."):
            st.session_state.schwab_client = connect_to_schwab()
    
    client = st.session_state.schwab_client
    
    if client is None:
        st.error("❌ No se pudo establecer conexión con Schwab")
        st.stop()
    
    st.success("✅ Conectado con Schwab")
    
    # Parámetros
    col1, col2 = st.columns(2)
    
    with col1:
        ticker = st.text_input("Ticker", value="SPX", help="Ingresa el símbolo del activo (SPX, SPY, QQQ, etc.)")
    
    with col2:
        width = st.number_input("Ancho de strikes (±puntos del spot)", 
                                min_value=10, value=150, step=10,
                                help="Rango de strikes a analizar alrededor del precio spot")
    
    if st.button("🔍 Analizar GEX", type="primary"):
        with st.spinner(f"Obteniendo datos de {ticker}..."):
            spot_price, options_data = get_options_data_schwab(client, ticker)
            
            if spot_price is None or options_data is None:
                st.error(f"No se pudieron obtener datos para {ticker}")
                st.stop()
            
            st.info(f"💰 Precio Spot de {ticker}: **${spot_price:,.2f}**")
            
            # Procesar datos
            with st.spinner("Procesando datos de opciones..."):
                df = process_schwab_options(options_data, spot_price)
            
            if df.empty:
                st.warning("No se encontraron datos de opciones")
                st.stop()
            
            st.success(f"✅ Se procesaron {len(df)} contratos de opciones")
            
            # Filtrar por rango
            lower_bound = spot_price - width
            upper_bound = spot_price + width
            
            df_filtered = df[
                (df['net_gex'].notna()) &
                (df['StrikePrice'].notna()) &
                (df['StrikePrice'] >= lower_bound) &
                (df['StrikePrice'] <= upper_bound)
            ].sort_values(by='StrikePrice').reset_index(drop=True)
            
            # Métricas resumidas ARRIBA
            st.markdown("---")
            st.subheader("📊 Métricas Clave")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                total_gamma = df['TotalGamma'].sum()
                st.metric("Total Gamma", f"${total_gamma:.2f}B")
            
            with col2:
                total_call_oi = df['CallOpenInt'].sum()
                st.metric("Call OI Total", f"{total_call_oi:,.0f}")
            
            with col3:
                total_put_oi = df['PutOpenInt'].sum()
                st.metric("Put OI Total", f"{total_put_oi:,.0f}")
            
            with col4:
                put_call_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0
                st.metric("Put/Call Ratio", f"{put_call_ratio:.2f}")
            
            with col5:
                max_gex_strike = df_filtered.loc[df_filtered['net_gex'].idxmax(), 'StrikePrice']
                st.metric("Strike Max GEX", f"{max_gex_strike:,.0f}")
            
            st.markdown("---")
            
            # Tabs para organizar gráficos
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🎯 Zonas Clave",
                "🔥 GEX por Strike",
                "📈 Open Interest", 
                "📊 Total Gamma",
                "🌊 Perfil GEX"
            ])
            
            with tab1:
                st.subheader("🎯 Zonas de Gamma y Open Interest")
                with st.spinner("Generando análisis de zonas..."):
                    fig5 = plot_gamma_zones(df_filtered, spot_price, ticker, width)
                    if fig5:
                        st.pyplot(fig5)
                        plt.close(fig5)
                        
                        # Recomendaciones
                        st.markdown("---")
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.markdown("### 📋 Interpretación")
                            st.info("""
                            **🟢 Zona Verde (Consolidación)**  
                            El precio tiende a moverse entre los strikes de mín y máx GEX.
                            
                            **🟠 Líneas Naranjas (Alto OI)**  
                            Actúan como imanes o barreras de precio intradía.
                            
                            **⚫ Línea Negra (Spot)**  
                            Precio actual del subyacente.
                            """)
                        
                        with col_b:
                            st.markdown("### 💡 Estrategias Sugeridas")
                            max_gex_val = df_filtered.loc[df_filtered['net_gex'].idxmax(), 'StrikePrice']
                            min_gex_val = df_filtered.loc[df_filtered['net_gex'].idxmin(), 'StrikePrice']
                            
                            if spot_price > (max_gex_val + min_gex_val) / 2:
                                st.success(f"""
                                **📊 Spot por encima del centro de zona GEX**
                                
                                - **Bear Call Spread**: {int(max_gex_val + 20)}/{int(max_gex_val + 40)}
                                - **Bull Put Spread**: {int(spot_price - 30)}/{int(spot_price - 50)}
                                """)
                            else:
                                st.warning(f"""
                                **📉 Spot por debajo del centro de zona GEX**
                                
                                - **Bull Put Spread**: {int(min_gex_val - 20)}/{int(min_gex_val - 40)}
                                - **Bear Call Spread**: {int(spot_price + 30)}/{int(spot_price + 50)}
                                """)
                    else:
                        st.warning("No hay datos suficientes para este gráfico")
            
            with tab2:
                st.subheader("🔥 GEX por Strike")
                with st.spinner("Generando gráfico..."):
                    fig4 = plot_gex_by_strike(df_filtered, spot_price, ticker, width)
                    if fig4:
                        st.pyplot(fig4)
                        plt.close(fig4)
                    else:
                        st.warning("No hay datos suficientes para este gráfico")
            
            with tab3:
                st.subheader("📈 Open Interest")
                with st.spinner("Generando gráfico..."):
                    fig2 = plot_open_interest(df, spot_price, ticker, width)
                    st.pyplot(fig2)
                    plt.close(fig2)
            
            with tab4:
                st.subheader("📊 Total Gamma Exposure")
                with st.spinner("Generando gráfico..."):
                    fig1 = plot_total_gamma(df, spot_price, ticker, width)
                    st.pyplot(fig1)
                    plt.close(fig1)
            
            with tab5:
                st.subheader("🌊 Perfil de Gamma Exposure")
                with st.spinner("Generando gráfico (esto puede tomar un momento)..."):
                    fig3 = plot_gex_profile(df.copy(), spot_price, ticker, width)
                    st.pyplot(fig3)
                    plt.close(fig3)


# =========================================================================
# 3. PUNTO DE ENTRADA
# =========================================================================

if __name__ == "__main__":
    if check_password():
        gex_scanner_page()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Introduce tus credenciales en el menú lateral para acceder.")
