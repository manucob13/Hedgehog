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
from utils import check_password
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
        
        # Verificar token
        test_response = client.get_quote("SPX")
        if hasattr(test_response, "status_code") and test_response.status_code != 200:
            raise Exception(f"Respuesta inesperada: {test_response.status_code}")
        
        return client
    except Exception as e:
        st.error(f"❌ Error al conectar con Schwab: {e}")
        return None

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

@st.cache_data(ttl=300)  # Cache por 5 minutos
def get_options_data_schwab(_client, ticker):
    """Obtiene datos de opciones desde Schwab"""
    try:
        # Obtener spot price
        quote_response = _client.get_quote(ticker)
        if quote_response.status_code != 200:
            return None, None
        
        quote_data = quote_response.json()
        spot_price = quote_data[ticker]['quote']['lastPrice']
        
        # Obtener cadena de opciones
        options_response = _client.get_option_chain(ticker)
        if options_response.status_code != 200:
            return None, None
        
        options_data = options_response.json()
        
        return spot_price, options_data
    except Exception as e:
        st.error(f"Error obteniendo datos: {e}")
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
            all_options.append({
                'ExpirationDate': exp_date_clean,
                'StrikePrice': float(strike),
                'CallPut': 'C',
                'CallOpenInt': contract.get('openInterest', 0),
                'CallIV': contract.get('volatility', 0),
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
            
            # Buscar si ya existe el strike/fecha en all_options
            found = False
            for opt in all_options:
                if opt['StrikePrice'] == float(strike) and opt['ExpirationDate'] == exp_date_clean:
                    opt['PutOpenInt'] = contract.get('openInterest', 0)
                    opt['PutIV'] = contract.get('volatility', 0)
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
                    'PutIV': contract.get('volatility', 0),
                    'PutGamma': contract.get('gamma', 0),
                    'PutVol': contract.get('totalVolume', 0)
                })
    
    df = pd.DataFrame(all_options)
    
    if df.empty:
        return df
    
    # Convertir fecha y calcular días hasta expiración
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
    """Gráfico de Gamma Exposure Total"""
    fromStrike = spot_price * 0.8
    toStrike = spot_price * 1.2
    
    dfAgg = df.groupby(['StrikePrice']).sum(numeric_only=True)
    strikes = dfAgg.index.values
    
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.grid(True, alpha=0.3)
    ax.bar(strikes, dfAgg['TotalGamma'].to_numpy(), width=6, 
           linewidth=0.1, edgecolor='k', label="Gamma Exposure")
    ax.set_xlim([fromStrike, toStrike])
    
    chartTitle = f"Total Gamma: ${df['TotalGamma'].sum():.2f} Bn per 1% {ticker} Move"
    ax.set_title(chartTitle, fontweight="bold", fontsize=20)
    ax.set_xlabel('Strike', fontweight="bold")
    ax.set_ylabel('Spot Gamma Exposure ($ billions/1% move)', fontweight="bold")
    ax.axvline(x=spot_price, color='r', lw=1, label=f"{ticker} Spot: {spot_price:,.0f}")
    ax.legend()
    
    return fig

def plot_open_interest(df, spot_price, ticker, width):
    """Gráfico de Open Interest"""
    fromStrike = spot_price * 0.8
    toStrike = spot_price * 1.2
    
    dfAgg = df.groupby(['StrikePrice']).sum(numeric_only=True)
    strikes = dfAgg.index.values
    
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.grid(True, alpha=0.3)
    ax.bar(strikes, dfAgg['CallOpenInt'].to_numpy(), width=6,
           linewidth=0.1, edgecolor='k', label="Call OI")
    ax.bar(strikes, -1 * dfAgg['PutOpenInt'].to_numpy(), width=6,
           linewidth=0.1, edgecolor='k', label="Put OI")
    ax.set_xlim([fromStrike, toStrike])
    
    ax.set_title(f"Total Open Interest for {ticker}", fontweight="bold", fontsize=20)
    ax.set_xlabel('Strike', fontweight="bold")
    ax.set_ylabel('Open Interest', fontweight="bold")
    ax.axvline(x=spot_price, color='r', lw=1, label=f"{ticker} Spot: {spot_price:,.0f}")
    ax.legend()
    
    return fig

def plot_gex_profile(df, spot_price, ticker, width):
    """Gráfico de Perfil de Gamma Exposure"""
    fromStrike = spot_price - width
    toStrike = spot_price + width
    levels = np.linspace(fromStrike, toStrike, 30)
    
    todayDate = date.today()
    nextExpiry = df['ExpirationDate'].min()
    
    df['IsThirdFriday'] = df['ExpirationDate'].apply(isThirdFriday)
    thirdFridays = df.loc[df['IsThirdFriday'] == True]
    nextMonthlyExp = thirdFridays['ExpirationDate'].min() if not thirdFridays.empty else nextExpiry
    
    totalGamma = []
    totalGammaExNext = []
    totalGammaExFri = []
    
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
        
        exFri = df.loc[df['ExpirationDate'] != nextMonthlyExp]
        totalGammaExFri.append(exFri['callGammaEx'].sum() - exFri['putGammaEx'].sum())
    
    totalGamma = np.array(totalGamma) / 10**9
    totalGammaExNext = np.array(totalGammaExNext) / 10**9
    totalGammaExFri = np.array(totalGammaExFri) / 10**9
    
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
    ax.grid(True, alpha=0.3)
    ax.plot(levels, totalGamma, label="All Expiries")
    ax.plot(levels, totalGammaExNext, label="Ex-Next Expiry")
    ax.plot(levels, totalGammaExFri, label="Ex-Next Monthly Expiry")
    
    chartTitle = f"Gamma Exposure Profile, {ticker}, {todayDate.strftime('%d %b %Y')}"
    ax.set_title(chartTitle, fontweight="bold", fontsize=20)
    ax.set_xlabel('Index Price', fontweight="bold")
    ax.set_ylabel('Gamma Exposure ($ billions/1% move)', fontweight="bold")
    
    ax.axvline(x=spot_price, color='r', lw=1, label=f"{ticker} Spot: {spot_price:,.0f}")
    
    if zeroGamma is not None:
        ax.axvline(x=zeroGamma, color='g', lw=1, label=f"Gamma Flip: {zeroGamma:,.0f}")
    
    ax.axhline(y=0, color='grey', lw=1)
    ax.set_xlim([fromStrike, toStrike])
    
    trans = ax.get_xaxis_transform()
    flip_point = zeroGamma if zeroGamma else fromStrike
    ax.fill_between([fromStrike, flip_point], min(totalGamma), max(totalGamma),
                    facecolor='red', alpha=0.1, transform=trans)
    ax.fill_between([flip_point, toStrike], min(totalGamma), max(totalGamma),
                    facecolor='green', alpha=0.1, transform=trans)
    
    ax.legend()
    return fig

def plot_gex_by_strike(df_filtered, spot, ticker, width):
    """Gráfico GEX por Strike"""
    pos = df_filtered[df_filtered['net_gex'] > 0]
    neg = df_filtered[df_filtered['net_gex'] < 0]
    
    if pos.empty and neg.empty:
        return None
    
    max_gex = df_filtered.loc[df_filtered['net_gex'].idxmax()]
    min_gex = df_filtered.loc[df_filtered['net_gex'].idxmin()]
    
    bar_width = 2
    fig, ax = plt.subplots(figsize=(16, 8))
    
    if not pos.empty:
        ax.bar(pos['StrikePrice'], pos['net_gex'], color='limegreen',
               width=bar_width, edgecolor='black', linewidth=0.8, label='GEX Positivo')
    
    if not neg.empty:
        ax.bar(neg['StrikePrice'], neg['net_gex'], color='red',
               width=bar_width, edgecolor='black', linewidth=0.8, label='GEX Negativo')
    
    ax.axhline(0, color='black', linestyle='--')
    ax.axvline(spot, color='black', linestyle='--', linewidth=1.5)
    
    ymin, ymax = ax.get_ylim()
    y_middle = (ymax + ymin) / 2
    ax.text(spot, y_middle, f'{int(spot)}', ha='center', va='center',
            fontsize=12, color='black', fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
    
    offset = (ymax - ymin) * 0.01
    ax.text(max_gex['StrikePrice'], max_gex['net_gex'] + offset,
            f'{int(max_gex["StrikePrice"])}', ha='right', va='bottom',
            fontsize=10, fontweight='bold', color='green')
    ax.text(min_gex['StrikePrice'], min_gex['net_gex'] - offset,
            f'{int(min_gex["StrikePrice"])}', ha='left', va='top',
            fontsize=10, fontweight='bold', color='darkred')
    
    ax.set_title(f'{ticker} GEX x STK (±{width} pts del Spot {int(spot)})',
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig

def plot_gamma_zones(df_filtered, spot, ticker, width):
    """Gráfico de Zonas Gamma y OI"""
    if df_filtered.empty:
        return None
    
    max_gex = df_filtered.loc[df_filtered['net_gex'].idxmax()]
    min_gex = df_filtered.loc[df_filtered['net_gex'].idxmin()]
    
    oi_threshold = df_filtered['total_oi'].quantile(0.90)
    high_oi = df_filtered[df_filtered['total_oi'] >= oi_threshold]
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    ax.axvspan(min_gex['StrikePrice'], max_gex['StrikePrice'],
               color='lightgreen', alpha=0.3)
    
    for _, row in high_oi.iterrows():
        strike = row['StrikePrice']
        ax.axvline(strike, color='orange', linestyle='--', alpha=0.6)
        ax.text(strike, 0.95, f'{int(strike)}', rotation=90,
                ha='center', va='top', fontsize=8,
                transform=ax.get_xaxis_transform(), color='orange')
    
    ax.axvline(spot, color='black', linestyle='--', linewidth=1.5)
    
    ax.text(min_gex['StrikePrice'], 0.85, f'Mín\n{int(min_gex["StrikePrice"])}',
            rotation=90, ha='center', va='top', fontsize=10,
            transform=ax.get_xaxis_transform(), color='darkred')
    ax.text(max_gex['StrikePrice'], 0.85, f'Máx\n{int(max_gex["StrikePrice"])}',
            rotation=90, ha='center', va='top', fontsize=10,
            transform=ax.get_xaxis_transform(), color='green')
    
    ax.set_title('Zonas clave Gamma y Open Interest', fontsize=16, fontweight='bold')
    ax.set_xlabel('Strike', fontsize=14)
    ax.set_ylabel('Nivel', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    lower_bound = int(min_gex['StrikePrice']) - 200
    upper_bound = int(max_gex['StrikePrice']) + 200
    ax.set_xticks(np.arange(lower_bound, upper_bound + 50, 50))
    ax.tick_params(axis='x', labelrotation=45, labelsize=12)
    
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
        ticker = st.text_input("Ticker", value="SPX", help="Ingresa el símbolo del activo")
    
    with col2:
        width = st.number_input("Ancho de strikes (±puntos del spot)", 
                                min_value=10, value=50, step=10)
    
    if st.button("🔍 Analizar GEX", type="primary"):
        with st.spinner(f"Obteniendo datos de {ticker}..."):
            spot_price, options_data = get_options_data_schwab(client, ticker)
            
            if spot_price is None or options_data is None:
                st.error(f"No se pudieron obtener datos para {ticker}")
                st.stop()
            
            st.info(f"Precio Spot de {ticker}: **${spot_price:,.2f}**")
            
            # Procesar datos
            df = process_schwab_options(options_data, spot_price)
            
            if df.empty:
                st.warning("No se encontraron datos de opciones")
                st.stop()
            
            # Filtrar por rango
            lower_bound = spot_price - width
            upper_bound = spot_price + width
            
            df_filtered = df[
                (df['net_gex'].notna()) &
                (df['StrikePrice'].notna()) &
                (df['StrikePrice'] >= lower_bound) &
                (df['StrikePrice'] <= upper_bound)
            ].sort_values(by='StrikePrice').reset_index(drop=True)
            
            # Tabs para organizar gráficos
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Total Gamma",
                "📈 Open Interest", 
                "🎯 Perfil GEX",
                "🔥 GEX por Strike",
                "🎪 Zonas Gamma"
            ])
            
            with tab1:
                st.subheader("Total Gamma Exposure")
                fig1 = plot_total_gamma(df, spot_price, ticker, width)
                st.pyplot(fig1)
            
            with tab2:
                st.subheader("Open Interest Total")
                fig2 = plot_open_interest(df, spot_price, ticker, width)
                st.pyplot(fig2)
            
            with tab3:
                st.subheader("Perfil de Gamma Exposure")
                fig3 = plot_gex_profile(df, spot_price, ticker, width)
                st.pyplot(fig3)
            
            with tab4:
                st.subheader("GEX por Strike")
                fig4 = plot_gex_by_strike(df_filtered, spot_price, ticker, width)
                if fig4:
                    st.pyplot(fig4)
                else:
                    st.warning("No hay datos suficientes para este gráfico")
            
            with tab5:
                st.subheader("Zonas de Gamma y Open Interest")
                fig5 = plot_gamma_zones(df_filtered, spot_price, ticker, width)
                if fig5:
                    st.pyplot(fig5)
                    
                    # Recomendaciones
                    st.markdown("---")
                    st.markdown("### 📋 Recomendaciones para 0DTE")
                    st.info("""
                    **Interpretación de Zonas:**
                    - 🟢 **Zona Verde**: El precio tiende a consolidarse aquí (entre mín y máx GEX)
                    - 🟠 **Picos OI**: Actúan como imanes de precio o barreras intradía
                    - 📊 **Cerca del Máx GEX**: Posible presión alcista
                    - 📉 **Cerca del Mín GEX**: Posible presión bajista
                    - ⚡ **Fuera de zona GEX**: Mayor probabilidad de movimiento rápido
                    
                    **Estrategias sugeridas:**
                    - **Bull Put Spreads**: Ubicar justo debajo de la zona verde si el spot sube
                    - **Bear Call Spreads**: Ubicar justo arriba de la zona verde si el spot cae
                    - **Iron Condor**: Ideal si el spot está centrado y el mercado tranquilo
                    """)
                else:
                    st.warning("No hay datos suficientes para este gráfico")
            
            # Métricas resumidas
            st.markdown("---")
            st.subheader("📊 Métricas Resumidas")
            
            col1, col2, col3, col4 = st.columns(4)
            
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

# =========================================================================
# 3. PUNTO DE ENTRADA
# =========================================================================

if __name__ == "__main__":
    if check_password():
        gex_scanner_page()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Introduce tus credenciales en el menú lateral para acceder.")
