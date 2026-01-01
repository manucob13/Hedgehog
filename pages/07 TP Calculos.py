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

@st.cache_data(ttl=300)  # Cache por 5 minutos
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
        response = requests.get(url, timeout=10)
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
        # Filtrar por fecha de expiración
        df_exp = df_options[df_options['expiry'] == expiration_date].copy()
        
        if df_exp.empty:
            return None, None, "No hay datos para esa fecha de expiración"
        
        # Encontrar el strike más cercano al precio actual (ATM)
        df_exp['distance'] = abs(df_exp['strike'] - current_price)
        atm_strike = df_exp.loc[df_exp['distance'].idxmin(), 'strike']
        
        # Obtener el Call y Put ATM
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
        # Usar MID si tanto bid como ask están disponibles
        if call_bid > 0 and call_ask > 0 and put_bid > 0 and put_ask > 0:
            straddle_price = call_mid + put_mid
            price_type = "Mid Price"
        elif call_bid > 0 and put_bid > 0:
            straddle_price = call_bid + put_bid
            price_type = "Bid Price"
        else:
            straddle_price = call_last + put_last
            price_type = "Last Price"
        
        # Expected Move = Straddle Price * 1.25 * std_multiplier
        # 1.25 para 1 desviación estándar completa
        expected_move = straddle_price * 1.25 * std_multiplier
        
        # Crear diccionario con detalles
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

# ==============================================================================
# NUEVA FUNCIÓN: CREAR CSV PARA IBKR BASKETTRADER
# ==============================================================================

def create_ibkr_basket_csv(ticker, expiration_date, strikes_dict, df_options):
    """
    Crea un CSV para IBKR TWS BasketTrader sin órdenes.
    Solo carga los contratos para que puedas crear órdenes manualmente.
    """
    
    basket_items = []
    
    # Convertir fecha a formato YYYYMMDD
    expiry_str = expiration_date.strftime("%Y%m%d")
    
    # Filtrar opciones para la fecha de expiración seleccionada
    df_exp = df_options[df_options['expiry'] == expiration_date].copy()
    
    # Procesar cada strike
    for strike_label, strike_value in strikes_dict.items():
        
        strike_value = float(strike_value)
        
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
# FUNCIÓN PRINCIPAL - TP CÁLCULOS
# ==============================================================================

def main_tp_calculos():
    
    # Inicializar session_state para guardar los resultados
    if 'calculation_done' not in st.session_state:
        st.session_state.calculation_done = False
    if 'current_price' not in st.session_state:
        st.session_state.current_price = None
    if 'expected_move' not in st.session_state:
        st.session_state.expected_move = None
    if 'details' not in st.session_state:
        st.session_state.details = None
    if 'selected_ticker' not in st.session_state:
        st.session_state.selected_ticker = None
    if 'expiration_date' not in st.session_state:
        st.session_state.expiration_date = None
    if 'std_multiplier' not in st.session_state:
        st.session_state.std_multiplier = None
    
    # --- TÍTULO PRINCIPAL ---
    st.markdown("<h1 style='text-align: center; color: #2E86AB;'>🎯 TP Cálculos - Expected Move</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Calcula el movimiento esperado basado en el straddle ATM</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # --- SECCIÓN 1: INPUTS ---
    st.subheader("📊 1. Configuración de Parámetros")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        ticker_input = st.text_input(
            "Ticker del activo",
            value="SPY",
            help="Introduce el símbolo del activo (ej: SPY, QQQ, AAPL)"
        ).upper()
    
    with col2:
        std_input = st.number_input(
            "Multiplicador de Desviación Estándar (σ)",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="1.0 = 68% de probabilidad, 2.0 = 95% de probabilidad"
        )
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        btn_calculate = st.button("🚀 Calcular", type="primary", use_container_width=True)
    
    # --- SECCIÓN 2: CÁLCULO ---
    if btn_calculate:
        with st.spinner("⏳ Obteniendo datos del mercado..."):
            
            # Obtener precio actual
            current_price = get_current_price(ticker_input)
            
            if current_price is None:
                st.error("❌ No se pudo obtener el precio actual del ticker. Verifica que el símbolo sea correcto.")
                st.stop()
            
            # Obtener cadena de opciones
            df_options = get_option_chain_cboe(ticker_input)
            
            if df_options is None or df_options.empty:
                st.error("❌ No se pudieron obtener datos de opciones desde CBOE.")
                st.stop()
            
            # Obtener fechas de expiración disponibles
            available_expiries = sorted(df_options['expiry'].unique())
            
            # Guardar en session_state
            st.session_state.selected_ticker = ticker_input
            st.session_state.current_price = current_price
            st.session_state.std_multiplier = std_input
            st.session_state.df_options = df_options
            st.session_state.available_expiries = available_expiries
            
            st.success(f"✅ Datos obtenidos correctamente para {ticker_input}")
            st.info(f"💲 Precio actual: **${current_price:.2f}** | Expiraciones disponibles: **{len(available_expiries)}**")
    
    # --- SECCIÓN 3: SELECCIÓN DE EXPIRACIÓN ---
    if st.session_state.get('available_expiries') is not None:
        
        st.markdown("---")
        st.subheader("📅 2. Seleccionar Fecha de Expiración")
        
        selected_expiry = st.selectbox(
            "Elige la fecha de expiración",
            options=st.session_state.available_expiries,
            format_func=lambda x: f"{x.strftime('%Y-%m-%d')} ({(x - date.today()).days} días hasta expiración)"
        )
        
        if st.button("📐 Calcular Strikes", type="primary"):
            
            # Calcular Expected Move
            expected_move, details, error = calculate_expected_move(
                st.session_state.df_options,
                selected_expiry,
                st.session_state.current_price,
                st.session_state.std_multiplier
            )
            
            if error:
                st.error(f"❌ {error}")
                st.stop()
            
            # Calcular strikes
            upper_strike = st.session_state.current_price + expected_move
            lower_strike = st.session_state.current_price - expected_move
            atm_strike = details['atm_strike']
            
            # Guardar resultados
            st.session_state.expiration_date = selected_expiry
            st.session_state.expected_move = expected_move
            st.session_state.details = details
            st.session_state.upper_strike = upper_strike
            st.session_state.lower_strike = lower_strike
            st.session_state.atm_strike = atm_strike
            st.session_state.calculation_done = True
    
    # --- SECCIÓN 4: RESULTADOS ---
    if st.session_state.calculation_done:
        
        st.markdown("---")
        st.subheader("✅ 3. Resultados del Expected Move")
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Expected Move",
                value=f"${st.session_state.expected_move:.2f}",
                delta=f"{(st.session_state.expected_move / st.session_state.current_price * 100):.2f}%"
            )
        
        with col2:
            st.metric(
                label="Upper Strike",
                value=f"${st.session_state.upper_strike:.2f}",
                delta=f"+{st.session_state.expected_move:.2f}"
            )
        
        with col3:
            st.metric(
                label="ATM Strike",
                value=f"${st.session_state.atm_strike:.2f}",
                delta="Current"
            )
        
        with col4:
            st.metric(
                label="Lower Strike",
                value=f"${st.session_state.lower_strike:.2f}",
                delta=f"-{st.session_state.expected_move:.2f}"
            )
        
        # Detalles del straddle
        st.markdown("#### 📋 Detalles del Straddle ATM")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 CALL ATM**")
            call_df = pd.DataFrame({
                'Métrica': ['Bid', 'Ask', 'Mid', 'Last'],
                'Valor': [
                    f"${st.session_state.details['call_bid']:.2f}",
                    f"${st.session_state.details['call_ask']:.2f}",
                    f"${st.session_state.details['call_mid']:.2f}",
                    f"${st.session_state.details['call_last']:.2f}"
                ]
            })
            st.dataframe(call_df, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("**📉 PUT ATM**")
            put_df = pd.DataFrame({
                'Métrica': ['Bid', 'Ask', 'Mid', 'Last'],
                'Valor': [
                    f"${st.session_state.details['put_bid']:.2f}",
                    f"${st.session_state.details['put_ask']:.2f}",
                    f"${st.session_state.details['put_mid']:.2f}",
                    f"${st.session_state.details['put_last']:.2f}"
                ]
            })
            st.dataframe(put_df, hide_index=True, use_container_width=True)
        
        st.info(f"💡 **Precio del Straddle utilizado:** ${st.session_state.details['straddle_price']:.2f} ({st.session_state.details['price_type']})")
        
        # --- GRÁFICO DE VISUALIZACIÓN ---
        st.markdown("---")
        st.subheader("📊 4. Visualización del Expected Move")
        
        # Crear gráfico
        fig = go.Figure()
        
        # Línea del precio actual
        fig.add_shape(
            type="line",
            x0=st.session_state.current_price,
            y0=0,
            x1=st.session_state.current_price,
            y1=1,
            line=dict(color="blue", width=3, dash="solid"),
            name="Precio Actual"
        )
        
        # Línea del upper strike
        fig.add_shape(
            type="line",
            x0=st.session_state.upper_strike,
            y0=0,
            x1=st.session_state.upper_strike,
            y1=1,
            line=dict(color="green", width=2, dash="dash"),
            name="Upper Strike"
        )
        
        # Línea del lower strike
        fig.add_shape(
            type="line",
            x0=st.session_state.lower_strike,
            y0=0,
            x1=st.session_state.lower_strike,
            y1=1,
            line=dict(color="red", width=2, dash="dash"),
            name="Lower Strike"
        )
        
        # Zona del expected move
        fig.add_vrect(
            x0=st.session_state.lower_strike,
            x1=st.session_state.upper_strike,
            fillcolor="lightgreen",
            opacity=0.2,
            layer="below",
            line_width=0
        )
        
        # Configuración del gráfico
        fig.update_layout(
            title=f"Expected Move para {st.session_state.selected_ticker}",
            xaxis_title="Precio",
            yaxis_title="",
            height=300,
            showlegend=False,
            yaxis=dict(showticklabels=False),
            xaxis=dict(
                range=[
                    st.session_state.lower_strike - 10,
                    st.session_state.upper_strike + 10
                ]
            )
        )
        
        # Añadir anotaciones
        fig.add_annotation(
            x=st.session_state.current_price,
            y=0.9,
            text=f"Actual: ${st.session_state.current_price:.2f}",
            showarrow=False,
            font=dict(color="blue", size=12)
        )
        
        fig.add_annotation(
            x=st.session_state.upper_strike,
            y=0.7,
            text=f"Upper: ${st.session_state.upper_strike:.2f}",
            showarrow=False,
            font=dict(color="green", size=10)
        )
        
        fig.add_annotation(
            x=st.session_state.lower_strike,
            y=0.7,
            text=f"Lower: ${st.session_state.lower_strike:.2f}",
            showarrow=False,
            font=dict(color="red", size=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # --- TABLA DE STRIKES Y OPCIONES ---
        st.markdown("---")
        st.subheader("🎯 5. Datos de Opciones por Strike")
        
        df_exp = st.session_state.df_options[
            st.session_state.df_options['expiry'] == st.session_state.expiration_date
        ].copy()
        
        strikes_dict = {
            'Upper Strike': st.session_state.upper_strike,
            'ATM Strike': st.session_state.atm_strike,
            'Lower Strike': st.session_state.lower_strike
        }
        
        for strike_label, strike_value in strikes_dict.items():
            
            with st.expander(f"📍 {strike_label}: ${strike_value:.2f}", expanded=True):
                
                col1, col2 = st.columns(2)
                
                # CALLS
                with col1:
                    st.markdown("**📈 CALL**")
                    
                    call_data = df_exp[
                        (df_exp['strike'] == strike_value) & 
                        (df_exp['opt_type'] == 'C')
                    ]
                    
                    if not call_data.empty:
                        call_table = pd.DataFrame({
                            'Métrica': ['Bid', 'Ask', 'Mid', 'Last', 'Volume', 'Open Int'],
                            'Valor': [
                                f"${call_data['bid'].iloc[0]:.2f}",
                                f"${call_data['ask'].iloc[0]:.2f}",
                                f"${call_data['Mid_price'].iloc[0]:.2f}",
                                f"${call_data['last_trade_price'].iloc[0]:.2f}",
                                f"{int(call_data['volume'].iloc[0]):,}",
                                f"{int(call_data['open_interest'].iloc[0]):,}"
                            ]
                        })
                        st.dataframe(call_table, hide_index=True, use_container_width=True)
                    else:
                        st.warning("No hay datos disponibles para este strike")
                
                # PUTS
                with col2:
                    st.markdown("**📉 PUT**")
                    
                    put_data = df_exp[
                        (df_exp['strike'] == strike_value) & 
                        (df_exp['opt_type'] == 'P')
                    ]
                    
                    if not put_data.empty:
                        put_table = pd.DataFrame({
                            'Métrica': ['Bid', 'Ask', 'Mid', 'Last', 'Volume', 'Open Int'],
                            'Valor': [
                                f"${put_data['bid'].iloc[0]:.2f}",
                                f"${put_data['ask'].iloc[0]:.2f}",
                                f"${put_data['Mid_price'].iloc[0]:.2f}",
                                f"${put_data['last_trade_price'].iloc[0]:.2f}",
                                f"{int(put_data['volume'].iloc[0]):,}",
                                f"{int(put_data['open_interest'].iloc[0]):,}"
                            ]
                        })
                        st.dataframe(put_table, hide_index=True, use_container_width=True)
                    else:
                        st.warning("No hay datos disponibles para este strike")
        
        # ==============================================================================
        # NUEVA SECCIÓN: EXPORTAR CSV PARA IBKR BASKETTRADER
        # ==============================================================================
        
        st.markdown("---")
        st.subheader("💾 6. Exportar para IBKR TWS BasketTrader")
        
        st.info("""
        **📝 Exportación para BasketTrader (sin órdenes preconfiguradas)**
        - Carga los contratos en TWS BasketTrader para análisis
        - NO crea órdenes automáticamente
        - Verás precios Bid/Ask/Mid en tiempo real
        - Tú agregas manualmente: Action (BUY/SELL), Quantity, y LmtPrice
        - Control total sobre cuándo y a qué precio operar
        """)
        
        # Preparar diccionario de strikes para la función
        strikes_for_export = {
            'Upper Strike': float(st.session_state.upper_strike),
            'ATM Strike': float(st.session_state.atm_strike),
            'Lower Strike': float(st.session_state.lower_strike)
        }
        
        # Crear el basket CSV
        df_basket = create_ibkr_basket_csv(
            st.session_state.selected_ticker,
            st.session_state.expiration_date,
            strikes_for_export,
            st.session_state.df_options
        )
        
        # Mostrar preview
        st.markdown("#### 👁️ Vista Previa - Contratos para BasketTrader")
        
        preview_cols = ['Symbol', 'SecType', 'Strike', 'Right', 'LastTradingDay', 'Exchange', 'Description']
        st.dataframe(
            df_basket[preview_cols],
            use_container_width=True,
            hide_index=True,
            height=300
        )
        
        with st.expander("📋 Ver formato completo del CSV"):
            st.dataframe(df_basket, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Botón de descarga
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col2:
            df_export = df_basket.drop(columns=['Description'])
            csv_data = df_export.to_csv(index=False)
            
            filename = f"{st.session_state.selected_ticker}_{st.session_state.expiration_date}_basket.csv"
            
            st.download_button(
                label="📥 Descargar CSV",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                type="primary",
                use_container_width=True
            )
        
        # Instrucciones
        with st.expander("ℹ️ Cómo usar el CSV en IBKR TWS BasketTrader"):
            st.markdown(f"""
            ### 📖 Instrucciones:
            
            1. **Abrir BasketTrader**: Trading Tools → BasketTrader
            2. **Importar**: Click derecho → Import Basket → Selecciona `{filename}`
            3. **Verificar contratos**: Verás {st.session_state.selected_ticker} con los 3 strikes y sus Calls/Puts
            4. **Agregar órdenes manualmente**:
               - Action: BUY o SELL
               - Quantity: Número de contratos
               - Order Type: LMT o MKT
               - Lmt Price: Tu precio deseado (usa Mid como referencia)
            5. **Transmitir**: Revisa y transmite cuando estés listo
            
            **💡 Tips:**
            - Comprando: precio entre Mid y Bid
            - Vendiendo: precio entre Mid y Ask
            
            **⚠️ Importante:** Este CSV NO contiene órdenes. Tú las creas manualmente con total control.
            """)

# ==============================================================================
# PUNTO DE ENTRADA PROTEGIDO
# ==============================================================================
if __name__ == "__main__":
    
    if check_password():
        main_tp_calculos()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder a TP Cálculos.")
