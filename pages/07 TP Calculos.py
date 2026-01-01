import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import requests
import plotly.graph_objects as go
from utils import check_password

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="TP Cálculos - Expected Move", layout="wide")

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

@st.cache_data(ttl=300)
def get_current_price(ticker):
    """Obtiene el precio actual del ticker desde Yahoo Finance."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        if not data.empty:
            return data['Close'].iloc[-1]
        return None
    except Exception as e:
        st.error(f"Error obteniendo precio actual: {e}")
        return None

@st.cache_data(ttl=300)
def get_option_chain_cboe(ticker):
    """Descarga la cadena de opciones desde CBOE."""
    try:
        url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{ticker}.json"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.cboe.com/",
            "Origin": "https://www.cboe.com"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'data' in data and 'options' in data['data']:
            options_data = data['data']['options']
            df = pd.DataFrame(options_data)
            
            # Procesar los datos de opciones
            df['expiry'] = df['option'].str[-15:-9].apply(
                lambda x: datetime.strptime(x, "%y%m%d").date()
            )
            df['opt_type'] = df['option'].str[-9]
            df['strike'] = df['option'].str[-8:].apply(
                lambda x: float(f"{x[:5]}.{x[5:]}")
            )
            df['Mid_price'] = (df['bid'] + df['ask']) / 2
            
            return df
        return None
    except Exception as e:
        st.error(f"Error descargando opciones de CBOE: {e}")
        return None

def calculate_expected_move(df_options, expiration_date, current_price, std_multiplier=1.0):
    """Calcula el Expected Move basado en el straddle ATM."""
    try:
        df_exp = df_options[df_options['expiry'] == expiration_date].copy()
        
        if df_exp.empty:
            return None, None, "No hay datos para esa fecha de expiración"
        
        # Encontrar el strike ATM
        df_exp['distance'] = abs(df_exp['strike'] - current_price)
        atm_strike = df_exp.loc[df_exp['distance'].idxmin(), 'strike']
        
        # Obtener Call y Put ATM
        call_atm = df_exp[(df_exp['strike'] == atm_strike) & (df_exp['opt_type'] == 'C')]
        put_atm = df_exp[(df_exp['strike'] == atm_strike) & (df_exp['opt_type'] == 'P')]
        
        if call_atm.empty or put_atm.empty:
            return None, None, "No se encontraron opciones ATM"
        
        # Obtener precios
        call_bid = call_atm['bid'].iloc[0]
        call_ask = call_atm['ask'].iloc[0]
        call_mid = (call_bid + call_ask) / 2
        call_last = call_atm['last_trade_price'].iloc[0]
        
        put_bid = put_atm['bid'].iloc[0]
        put_ask = put_atm['ask'].iloc[0]
        put_mid = (put_bid + put_ask) / 2
        put_last = put_atm['last_trade_price'].iloc[0]
        
        # Prioridad: MID > BID > LAST
        if call_bid > 0 and call_ask > 0 and put_bid > 0 and put_ask > 0:
            straddle_price = call_mid + put_mid
            price_type = "Mid Price"
        elif call_bid > 0 and put_bid > 0:
            straddle_price = call_bid + put_bid
            price_type = "Bid Price"
        else:
            straddle_price = call_last + put_last
            price_type = "Last Price"
        
        expected_move = straddle_price * 1.25 * std_multiplier
        
        details = {
            'atm_strike': atm_strike,
            'call_bid': call_bid,
            'call_ask': call_ask,
            'call_mid': call_mid,
            'call_last': call_last,
            'put_bid': put_bid,
            'put_ask': put_ask,
            'put_mid': put_mid,
            'put_last': put_last,
            'straddle_price': straddle_price,
            'price_type': price_type,
            'std_multiplier': std_multiplier
        }
        
        return expected_move, details, None
        
    except Exception as e:
        return None, None, f"Error calculando Expected Move: {e}"

@st.cache_data(ttl=300)
def get_historical_prices(ticker, days=7):
    """Obtiene precios históricos del ticker."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        df = stock.history(start=start_date, end=end_date)
        return df
    except Exception as e:
        st.error(f"Error obteniendo datos históricos: {e}")
        return None

def create_ibkr_basket_csv(ticker, expiration_date, strikes_dict, df_options):
    """
    Crea un CSV para IBKR TWS BasketTrader sin órdenes.
    Solo carga los contratos para que puedas crear órdenes manualmente.
    
    Formato para basket sin órdenes:
    Symbol,SecType,Exchange,Currency,LastTradingDay,Strike,Right,Multiplier
    """
    
    basket_items = []
    
    # Convertir fecha a formato YYYYMMDD
    expiry_str = expiration_date.strftime("%Y%m%d")
    
    # Filtrar opciones para la fecha de expiración seleccionada
    df_exp = df_options[df_options['expiry'] == expiration_date].copy()
    
    # Procesar cada strike
    for strike_label, strike_value in strikes_dict.items():
        
        # Call
        call_data = df_exp[(df_exp['strike'] == strike_value) & (df_exp['opt_type'] == 'C')]
        
        if not call_data.empty:
            basket_items.append({
                'Symbol': ticker,
                'SecType': 'OPT',
                'Exchange': 'SMART',
                'Currency': 'USD',
                'LastTradingDay': expiry_str,
                'Strike': strike_value,
                'Right': 'C',
                'Multiplier': '100',
                'Description': f'{strike_label} Call'
            })
        
        # Put
        put_data = df_exp[(df_exp['strike'] == strike_value) & (df_exp['opt_type'] == 'P')]
        
        if not put_data.empty:
            basket_items.append({
                'Symbol': ticker,
                'SecType': 'OPT',
                'Exchange': 'SMART',
                'Currency': 'USD',
                'LastTradingDay': expiry_str,
                'Strike': strike_value,
                'Right': 'P',
                'Multiplier': '100',
                'Description': f'{strike_label} Put'
            })
    
    df_basket = pd.DataFrame(basket_items)
    return df_basket

# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================

def main_tp_calculos():
    
    # Inicializar session_state
    if 'calculation_done' not in st.session_state:
        st.session_state.calculation_done = False
    if 'df_options' not in st.session_state:
        st.session_state.df_options = None
    if 'final_strikes' not in st.session_state:
        st.session_state.final_strikes = None
    
    st.title("🎯 TP Cálculos - Expected Move Calculator")
    st.markdown("---")
    
    # === PASO 1: INPUT DE PARÁMETROS ===
    st.subheader("📊 1. Configuración Inicial")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ticker = st.text_input("Ticker", value="SPY", help="Símbolo del activo").upper()
    
    with col2:
        std_multiplier = st.number_input(
            "Multiplicador σ",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="1.0 = 1 desviación estándar (~68% probabilidad)"
        )
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        calculate_btn = st.button("🔍 Calcular Expected Move", type="primary", use_container_width=True)
    
    # === PASO 2: CÁLCULO ===
    if calculate_btn:
        with st.spinner("Obteniendo datos..."):
            
            # Obtener precio actual
            current_price = get_current_price(ticker)
            
            if current_price is None:
                st.error("No se pudo obtener el precio actual del ticker")
                return
            
            # Obtener cadena de opciones
            df_options = get_option_chain_cboe(ticker)
            
            if df_options is None or df_options.empty:
                st.error("No se pudieron obtener datos de opciones")
                return
            
            # Guardar en session_state
            st.session_state.current_price = current_price
            st.session_state.df_options = df_options
            st.session_state.ticker = ticker
            st.session_state.std_multiplier = std_multiplier
            
            # Obtener fechas de expiración disponibles
            expiration_dates = sorted(df_options['expiry'].unique())
            st.session_state.expiration_dates = expiration_dates
            
            st.success(f"✓ Precio actual de {ticker}: ${current_price:.2f}")
            st.success(f"✓ Se encontraron {len(expiration_dates)} fechas de expiración")
    
    # === PASO 3: SELECCIÓN DE EXPIRACIÓN Y CÁLCULO DE STRIKES ===
    if 'expiration_dates' in st.session_state:
        
        st.markdown("---")
        st.subheader("📅 2. Seleccionar Fecha de Expiración")
        
        expiration_date = st.selectbox(
            "Fecha de Expiración",
            options=st.session_state.expiration_dates,
            format_func=lambda x: f"{x} ({(x - date.today()).days} días)"
        )
        
        if st.button("📐 Calcular Strikes", type="primary"):
            
            # Calcular expected move
            expected_move, details, error = calculate_expected_move(
                st.session_state.df_options,
                expiration_date,
                st.session_state.current_price,
                st.session_state.std_multiplier
            )
            
            if error:
                st.error(error)
                return
            
            # Calcular strikes
            current_price = st.session_state.current_price
            
            strikes = {
                'Upper Strike': round(current_price + expected_move, 2),
                'ATM Strike': details['atm_strike'],
                'Lower Strike': round(current_price - expected_move, 2)
            }
            
            # Guardar en session_state
            st.session_state.expected_move = expected_move
            st.session_state.details = details
            st.session_state.final_strikes = strikes
            st.session_state.expiration_date = expiration_date
            st.session_state.calculation_done = True
    
    # === PASO 4: MOSTRAR RESULTADOS Y OPCIONES ===
    if st.session_state.calculation_done:
        
        st.markdown("---")
        st.subheader("✅ 3. Resultados Calculados")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Expected Move", f"${st.session_state.expected_move:.2f}")
        
        with col2:
            st.metric("Upper Strike", f"${st.session_state.final_strikes['Upper Strike']:.2f}")
        
        with col3:
            st.metric("Lower Strike", f"${st.session_state.final_strikes['Lower Strike']:.2f}")
        
        # Tabla de strikes
        st.markdown("#### 📋 Strikes Calculados")
        
        strikes_df = pd.DataFrame({
            'Strike Type': list(st.session_state.final_strikes.keys()),
            'Strike Price': list(st.session_state.final_strikes.values())
        })
        
        st.dataframe(strikes_df, use_container_width=True, hide_index=True)
        
        # Mostrar opciones para cada strike
        st.markdown("---")
        st.subheader("📊 4. Datos de Opciones por Strike")
        
        df_exp = st.session_state.df_options[
            st.session_state.df_options['expiry'] == st.session_state.expiration_date
        ].copy()
        
        for strike_label, strike_value in st.session_state.final_strikes.items():
            
            with st.expander(f"🎯 {strike_label}: ${strike_value:.2f}", expanded=True):
                
                col1, col2 = st.columns(2)
                
                # Calls
                with col1:
                    st.markdown("**📈 CALL**")
                    call_data = df_exp[(df_exp['strike'] == strike_value) & (df_exp['opt_type'] == 'C')]
                    
                    if not call_data.empty:
                        call_display = pd.DataFrame({
                            'Bid': [call_data['bid'].iloc[0]],
                            'Ask': [call_data['ask'].iloc[0]],
                            'Mid': [(call_data['bid'].iloc[0] + call_data['ask'].iloc[0]) / 2],
                            'Last': [call_data['last_trade_price'].iloc[0]],
                            'Volume': [call_data['volume'].iloc[0]],
                            'Open Int': [call_data['open_interest'].iloc[0]]
                        })
                        st.dataframe(call_display, use_container_width=True, hide_index=True)
                    else:
                        st.warning("No hay datos de Call disponibles")
                
                # Puts
                with col2:
                    st.markdown("**📉 PUT**")
                    put_data = df_exp[(df_exp['strike'] == strike_value) & (df_exp['opt_type'] == 'P')]
                    
                    if not put_data.empty:
                        put_display = pd.DataFrame({
                            'Bid': [put_data['bid'].iloc[0]],
                            'Ask': [put_data['ask'].iloc[0]],
                            'Mid': [(put_data['bid'].iloc[0] + put_data['ask'].iloc[0]) / 2],
                            'Last': [put_data['last_trade_price'].iloc[0]],
                            'Volume': [put_data['volume'].iloc[0]],
                            'Open Int': [put_data['open_interest'].iloc[0]]
                        })
                        st.dataframe(put_display, use_container_width=True, hide_index=True)
                    else:
                        st.warning("No hay datos de Put disponibles")
        
        # === PASO 5: EXPORTAR CSV PARA IBKR TWS BASKETTRADER ===
        st.markdown("---")
        st.subheader("💾 5. Exportar para IBKR TWS BasketTrader")
        
        st.info("""
        **📝 Exportación para BasketTrader (sin órdenes)**
        - Carga los contratos en TWS BasketTrader
        - NO crea órdenes automáticamente
        - Verás precios Bid/Ask/Mid en tiempo real
        - Tú agregas manualmente: Action (BUY/SELL), Quantity, y LmtPrice
        - Total control sobre cuándo y a qué precio operar
        """)
        
        # Crear el basket CSV
        df_basket = create_ibkr_basket_csv(
            st.session_state.ticker,
            st.session_state.expiration_date,
            st.session_state.final_strikes,
            st.session_state.df_options
        )
        
        # Mostrar preview
        st.markdown("#### 👁️ Vista Previa - Contratos para BasketTrader")
        
        # Mostrar con descripción
        preview_cols = ['Symbol', 'SecType', 'Strike', 'Right', 'LastTradingDay', 'Exchange', 'Description']
        st.dataframe(
            df_basket[preview_cols],
            use_container_width=True,
            hide_index=True,
            height=300
        )
        
        # Mostrar estructura completa en expander
        with st.expander("📋 Ver formato completo del CSV (todas las columnas)"):
            st.dataframe(df_basket, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Botones de descarga
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col2:
            # Preparar CSV para descarga (sin columna Description)
            df_export = df_basket.drop(columns=['Description'])
            csv_data = df_export.to_csv(index=False)
            
            filename = f"{st.session_state.ticker}_{st.session_state.expiration_date}_basket.csv"
            
            st.download_button(
                label="📥 Descargar CSV para BasketTrader",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                type="primary",
                use_container_width=True
            )
        
        # Instrucciones de uso
        st.markdown("---")
        with st.expander("ℹ️ Cómo usar el CSV en IBKR TWS BasketTrader"):
            st.markdown(f"""
            ### 📖 Instrucciones paso a paso:
            
            #### 1️⃣ Abrir BasketTrader en TWS
            
            - Ve a: **Trading Tools → BasketTrader**
            - O usa el atajo: **Ctrl+Shift+B** (Windows) / **Cmd+Shift+B** (Mac)
            
            #### 2️⃣ Importar el archivo CSV
            
            - Click derecho en el área de BasketTrader
            - Selecciona: **Import Basket...**
            - Busca y selecciona el archivo: `{filename}`
            - Los contratos se cargarán en el basket
            
            #### 3️⃣ Verificar los contratos cargados
            
            Los contratos aparecerán con:
            - ✅ Symbol: {st.session_state.ticker}
            - ✅ Strike: Upper, ATM, Lower
            - ✅ Right: C (Call) y P (Put)
            - ✅ Expiry: {st.session_state.expiration_date}
            - ✅ Precios en tiempo real (Bid, Ask, Mid)
            
            #### 4️⃣ Agregar órdenes MANUALMENTE
            
            Para cada contrato que quieras operar:
            
            1. **Click en la columna "Action"**:
               - Selecciona: `BUY` o `SELL`
            
            2. **Click en la columna "Quantity"**:
               - Ingresa el número de contratos (ej: 1, 2, 5, 10)
            
            3. **Click en la columna "Order Type"**:
               - Selecciona: `LMT` (Limit) o `MKT` (Market)
            
            4. **Si elegiste LMT, click en "Lmt Price"**:
               - Ingresa tu precio deseado
               - Tip: Usa el precio **Mid** como referencia
               - Ajusta según quieras: más cerca de Bid (para comprar) o Ask (para vender)
            
            #### 5️⃣ Revisar y transmitir
            
            - Revisa todas las órdenes en el basket
            - Verifica: Action, Quantity, Price
            - Cuando estés listo: **Click en "Transmit All"**
            - O transmite individualmente: Click derecho → Transmit Order
            
            ---
            
            ### 💡 Tips y Recomendaciones:
            
            **Estrategia de precios:**
            - 🟢 **Comprando (BUY)**: Usa precio entre **Mid** y **Bid** para mejor ejecución
            - 🔴 **Vendiendo (SELL)**: Usa precio entre **Mid** y **Ask** para mejor ejecución
            - ⚡ **Ejecución rápida**: Usa precio **Ask** (compra) o **Bid** (venta)
            - 🎯 **Mejor precio**: Usa **LMT** en Mid y espera
            
            **Gestión de órdenes:**
            - Puedes agregar/eliminar contratos del basket antes de transmitir
            - Puedes modificar precios después de transmitir (si no ejecutaron)
            - Usa "Preview Order" para ver el impacto antes de transmitir
            
            **Spreads y estrategias:**
            - Para Iron Condor: SELL Upper Call, SELL Lower Put
            - Para Straddle: BUY ATM Call + BUY ATM Put
            - Para Strangle: BUY Upper Call + BUY Lower Put
            
            ---
            
            ### 📚 Columnas del CSV exportado:
            
            | Columna | Descripción |
            |---------|-------------|
            | Symbol | Ticker del activo ({st.session_state.ticker}) |
            | SecType | OPT (Tipo: Opción) |
            | Exchange | SMART (Enrutamiento inteligente de IBKR) |
            | Currency | USD |
            | LastTradingDay | Fecha de expiración (formato YYYYMMDD) |
            | Strike | Precio de ejercicio de la opción |
            | Right | C (Call) o P (Put) |
            | Multiplier | 100 (tamaño estándar del contrato) |
            
            ---
            
            ### ⚠️ Importante:
            
            - ✅ Este CSV **NO contiene órdenes preconfiguradas**
            - ✅ Tú tienes **control total** sobre Action, Quantity, y Price
            - ✅ Sin riesgo de **ejecuciones accidentales**
            - ✅ Ideal para **analizar el mercado** antes de ejecutar
            - ✅ Precios en **tiempo real** para mejor decisión
            
            ---
            
            ### 🔗 Referencias útiles:
            
            - [IBKR BasketTrader Guide](https://www.interactivebrokers.com/campus/trading-lessons/tws-baskettrader-create-a-basket/)
            - [CSV Import Format](https://www.interactivebrokers.co.uk/en/software/tws/usersguidebook/getstarted/import_tickers_from_a_file.htm)
            - [Options Trading](https://www.interactivebrokers.com/en/trading/orders.php)
            """)

# ==============================================================================
# PUNTO DE ENTRADA
# ==============================================================================
if __name__ == "__main__":
    
    if check_password():
        main_tp_calculos()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales para acceder.")
