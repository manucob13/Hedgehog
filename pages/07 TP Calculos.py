import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import requests
import plotly.graph_objects as go
from utils import check_password
import io

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
        
        # Convertir ticker para Yahoo Finance
        if ticker == 'SPX':
            yf_ticker = '^SPX'
        else:
            yf_ticker = ticker
        
        stock = yf.Ticker(yf_ticker)
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
        
        if ticker == 'SPX':
            yf_ticker = '^SPX'
        else:
            yf_ticker = ticker
        
        stock = yf.Ticker(yf_ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        df = stock.history(start=start_date, end=end_date)
        return df
    except Exception as e:
        st.error(f"Error obteniendo datos históricos: {e}")
        return None

def generate_ibkr_basket_csv(orders_df):
    """Convierte el DataFrame de órdenes a formato CSV de IBKR Basket."""
    output = io.StringIO()
    orders_df.to_csv(output, index=False)
    csv_content = output.getvalue()
    output.close()
    return csv_content

# ==============================================================================
# FUNCIÓN PRINCIPAL - TP CÁLCULOS
# ==============================================================================

def main_tp_calculos():
    
    # Inicializar session_state
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
    st.markdown("<h1><span style='font-size: 1.5em;'>🎯</span> TP Cálculos - Expected Move</h1>", unsafe_allow_html=True)
    st.markdown("""
    Esta herramienta calcula el **Expected Move** (movimiento esperado) de un activo basándose 
    en los precios de las opciones (straddle ATM) para una fecha de expiración específica.
    """)
    st.markdown("---")
    
    # ==============================================================================
    # SECCIÓN 1: CONFIGURACIÓN
    # ==============================================================================
    st.header("1. Configuración")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ticker_options = ['QQQ', 'SPX']
        selected_ticker = st.selectbox(
            "Selecciona el Ticker",
            ticker_options,
            index=0,
            key='ticker_tp'
        )
        st.info(f"📊 Ticker seleccionado: **{selected_ticker}**")
    
    with col2:
        min_date = date.today() + timedelta(days=1)
        max_date = date.today() + timedelta(days=365)
        
        expiration_date = st.date_input(
            "Fecha de Expiración",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            key='expiration_tp'
        )
        st.info(f"📅 Expiración: **{expiration_date.strftime('%Y-%m-%d')}**")
    
    with col3:
        std_options = {
            "1σ (68% probabilidad)": 1.0,
            "1.5σ (87% probabilidad)": 1.5,
            "2σ (95% probabilidad)": 2.0
        }
        
        selected_std_label = st.selectbox(
            "Desviaciones Estándar",
            list(std_options.keys()),
            index=0,
            key='std_selector_tp'
        )
        std_multiplier = std_options[selected_std_label]
        
        st.info(f"📈 Multiplicador: **{std_multiplier}σ**")
    
    st.markdown("---")
    
    # ==============================================================================
    # BOTÓN PARA CALCULAR
    # ==============================================================================
    
    if st.button("🚀 Calcular Expected Move", type="primary", use_container_width=True):
        
        with st.spinner(f"Obteniendo datos de {selected_ticker}..."):
            
            current_price = get_current_price(selected_ticker)
            
            if current_price is None:
                st.error("❌ No se pudo obtener el precio actual del activo.")
                st.stop()
            
            st.success(f"✅ Precio actual de {selected_ticker}: **${current_price:.2f}**")
            
            df_options = get_option_chain_cboe(selected_ticker)
            
            if df_options is None or df_options.empty:
                st.error("❌ No se pudieron obtener los datos de opciones desde CBOE.")
                st.stop()
            
            expected_move, details, error = calculate_expected_move(
                df_options, 
                expiration_date, 
                current_price,
                std_multiplier
            )
            
            if error:
                st.error(f"❌ {error}")
                st.stop()
            
            st.session_state.calculation_done = True
            st.session_state.current_price = current_price
            st.session_state.expected_move = expected_move
            st.session_state.details = details
            st.session_state.selected_ticker = selected_ticker
            st.session_state.expiration_date = expiration_date
            st.session_state.std_multiplier = std_multiplier
            
            st.success(f"✅ Expected Move calculado: **${expected_move:.2f}** ({std_multiplier}σ)")
            st.info(f"💰 Precio del Straddle: **${details['straddle_price']:.2f}** ({details['price_type']})")
    
    # ==============================================================================
    # MOSTRAR RESULTADOS
    # ==============================================================================
    
    if st.session_state.calculation_done:
        
        current_price = st.session_state.current_price
        expected_move = st.session_state.expected_move
        details = st.session_state.details
        selected_ticker = st.session_state.selected_ticker
        expiration_date = st.session_state.expiration_date
        std_multiplier = st.session_state.std_multiplier
        
        st.markdown("---")
        
        # ==============================================================================
        # SECCIÓN 2: RESULTADOS
        # ==============================================================================
        st.header("2. Resultados del Expected Move")
        
        upper_range = current_price + expected_move
        lower_range = current_price - expected_move
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Precio Actual", f"${current_price:.2f}")
        with col2:
            st.metric(f"Expected Move ({std_multiplier}σ)", f"${expected_move:.2f}")
        with col3:
            st.metric("Rango Superior", f"${upper_range:.2f}", f"+{expected_move:.2f}")
        with col4:
            st.metric("Rango Inferior", f"${lower_range:.2f}", f"-{expected_move:.2f}")
        
        st.markdown("### 💰 Detalles del Straddle ATM")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📞 CALL ATM**")
            call_details = {
                'Strike': f"${details['atm_strike']:.2f}",
                'Bid': f"${details['call_bid']:.2f}",
                'Ask': f"${details['call_ask']:.2f}",
                'Mid': f"${details['call_mid']:.2f}",
                'Last': f"${details['call_last']:.2f}"
            }
            df_call = pd.DataFrame(list(call_details.items()), columns=['Métrica', 'Valor'])
            st.dataframe(df_call, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("**📉 PUT ATM**")
            put_details = {
                'Strike': f"${details['atm_strike']:.2f}",
                'Bid': f"${details['put_bid']:.2f}",
                'Ask': f"${details['put_ask']:.2f}",
                'Mid': f"${details['put_mid']:.2f}",
                'Last': f"${details['put_last']:.2f}"
            }
            df_put = pd.DataFrame(list(put_details.items()), columns=['Métrica', 'Valor'])
            st.dataframe(df_put, hide_index=True, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### 📊 Resumen del Expected Move")
        
        days_to_exp = (expiration_date - date.today()).days
        move_pct = (expected_move / current_price) * 100
        
        summary_data = {
            'Métrica': [
                'Precio Actual',
                'Strike ATM',
                f'Straddle Price ({details["price_type"]})',
                f'Expected Move (±) [{std_multiplier}σ]',
                'Rango Superior',
                'Rango Inferior',
                'Movimiento (%)',
                'Días hasta Expiración',
                'Fecha de Expiración'
            ],
            'Valor': [
                f"${current_price:.2f}",
                f"${details['atm_strike']:.2f}",
                f"${details['straddle_price']:.2f}",
                f"${expected_move:.2f}",
                f"${upper_range:.2f}",
                f"${lower_range:.2f}",
                f"±{move_pct:.2f}%",
                f"{days_to_exp} días",
                expiration_date.strftime('%Y-%m-%d')
            ]
        }
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, hide_index=True, use_container_width=True)
        
        st.markdown("---")
        
        # ==============================================================================
        # SECCIÓN 3: GRÁFICO
        # ==============================================================================
        st.header("3. Visualización del Expected Move")
        
        df_hist = get_historical_prices(selected_ticker, days=7)
        
        if df_hist is not None and not df_hist.empty:
            
            fig = go.Figure()
            
            df_hist_plot = df_hist.copy()
            if df_hist_plot.index.tz is not None:
                df_hist_plot.index = df_hist_plot.index.tz_localize(None)
            
            fig.add_trace(go.Candlestick(
                x=df_hist_plot.index,
                open=df_hist_plot['Open'],
                high=df_hist_plot['High'],
                low=df_hist_plot['Low'],
                close=df_hist_plot['Close'],
                name=selected_ticker,
                increasing_line_color='#00B06B',
                decreasing_line_color='#FF4444'
            ))
            
            exp_datetime = datetime.combine(expiration_date, datetime.min.time())
            
            fig.add_hline(
                y=upper_range,
                line_dash="dot",
                line_color="steelblue",
                annotation_text=f"${upper_range:.2f} (+{std_multiplier}σ)",
                annotation_position="right"
            )
            
            fig.add_hline(
                y=lower_range,
                line_dash="dot",
                line_color="steelblue",
                annotation_text=f"${lower_range:.2f} (-{std_multiplier}σ)",
                annotation_position="right"
            )
            
            fig.add_hline(
                y=current_price,
                line_dash="dash",
                line_color="yellow",
                annotation_text=f"Precio Actual: ${current_price:.2f}",
                annotation_position="left"
            )
            
            fig.add_shape(
                type="line",
                x0=exp_datetime,
                x1=exp_datetime,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color="red", width=2, dash="dot")
            )
            
            fig.add_annotation(
                x=exp_datetime,
                y=1.02,
                yref="paper",
                text=f"Expiración: {expiration_date.strftime('%Y-%m-%d')}",
                showarrow=False,
                font=dict(color="red", size=12),
                bgcolor="rgba(0,0,0,0.5)"
            )
            
            end_datetime = exp_datetime + timedelta(days=3)
            
            fig.update_layout(
                title=f"Expected Move - {selected_ticker} ({std_multiplier}σ)",
                xaxis_title="Fecha",
                yaxis_title="Precio",
                template="plotly_dark",
                height=500,
                hovermode='x unified',
                showlegend=True,
                xaxis=dict(range=[df_hist_plot.index[0], end_datetime]),
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("⚠️ No se pudieron obtener datos históricos para el gráfico.")
        
        st.markdown("---")
        
        # ==============================================================================
        # SECCIÓN 4: COMPARACIÓN
        # ==============================================================================
        st.header("4. Comparación de Desviaciones Estándar")
        
        comparison_data = []
        for std_label, std_val in std_options.items():
            em = details['straddle_price'] * 1.25 * std_val
            upper = current_price + em
            lower = current_price - em
            pct = (em / current_price) * 100
            
            comparison_data.append({
                'Desviación Estándar': std_label,
                'Expected Move': f"${em:.2f}",
                'Rango': f"${lower:.2f} - ${upper:.2f}",
                'Movimiento (%)': f"±{pct:.2f}%"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        st.markdown("""
        Esta tabla muestra cómo varían los rangos esperados según diferentes niveles de confianza estadística:
        """)
        
        st.dataframe(df_comparison, hide_index=True, use_container_width=True)
        
        st.markdown("---")
        
        # ==============================================================================
        # SECCIÓN 5: INFORMACIÓN
        # ==============================================================================
        st.header("5. Información Adicional")
        
        if std_multiplier == 1.0:
            prob_text = "68%"
        elif std_multiplier == 1.5:
            prob_text = "87%"
        elif std_multiplier == 2.0:
            prob_text = "95%"
        else:
            prob_text = "N/A"
        
        st.info(f"""
        📌 **Interpretación del Expected Move:**
        
        - El **Expected Move** representa el rango de precio esperado ({std_multiplier} desviación estándar) 
          que el mercado anticipa para la fecha de expiración.
        
        - Este cálculo se basa en el precio del **straddle ATM** (comprar un call y un put 
          al mismo strike más cercano al precio actual).
        
        - Con **{std_multiplier}σ**, aproximadamente el **{prob_text}** de las veces, el precio debería 
          permanecer dentro de este rango.
        
        - **Ticker:** {selected_ticker}
        - **Precio Actual:** ${current_price:.2f}
        - **Straddle Price:** ${details['straddle_price']:.2f}
        - **Expected Move ({std_multiplier}σ):** ±${expected_move:.2f} (±{move_pct:.2f}%)
        - **Rango Esperado:** ${lower_range:.2f} - ${upper_range:.2f}
        - **Días hasta Expiración:** {days_to_exp}
        """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 📚 Fuentes de Datos
        - **Precios de Opciones:** CBOE (Chicago Board Options Exchange)
        - **Precios del Activo:** Yahoo Finance
        
        ### 🧮 Fórmula del Expected Move
        ```
        Expected Move = Straddle Price × 1.25 × σ
        ```
        
        Donde:
        - **Straddle Price** = Call Mid + Put Mid (ATM)
        - **1.25** = Factor de ajuste para 1 desviación estándar completa
        - **σ** = Multiplicador de desviaciones estándar (1.0, 1.5, o 2.0)
        
        ### 💡 Prioridad de Precios
        1. **Mid Price** (Bid + Ask) / 2 - Preferido cuando está disponible
        2. **Bid Price** - Usado si el mid no está disponible
        3. **Last Price** - Usado como último recurso
        
        ### 📊 Niveles de Confianza
        - **1σ** ≈ 68% de probabilidad (rango más conservador)
        - **1.5σ** ≈ 87% de probabilidad (rango intermedio)
        - **2σ** ≈ 95% de probabilidad (rango más amplio)
        """)
        
        st.markdown("---")
        
        # Inicializar variables para punto 7
        if 'strike_atm_p6' not in st.session_state:
            st.session_state.strike_atm_p6 = None
        if 'strike_up_p6' not in st.session_state:
            st.session_state.strike_up_p6 = None
        if 'strike_down_p6' not in st.session_state:
            st.session_state.strike_down_p6 = None
        if 'dte_front_p6' not in st.session_state:
            st.session_state.dte_front_p6 = None
        if 'dte_back_p6' not in st.session_state:
            st.session_state.dte_back_p6 = None
        
        # ==============================================================================
        # SECCIÓN 6: TRIPLE CALENDAR
        # ==============================================================================
        st.header("6. Generador de Estructura - Triple Calendar")
        
        st.markdown("""
        Configura los strikes y fechas de expiración para generar un archivo CSV en formato **IBKR Basket** 
        que podrás descargar y ejecutar manualmente en tu broker.
        """)
        
        expected_move_1std = details['straddle_price'] * 1.25
        upper_range_1std = current_price + expected_move_1std
        lower_range_1std = current_price - expected_move_1std
        
        atm_rounded = round(details['atm_strike'] / 5) * 5
        strike_up_default = round(upper_range_1std / 5) * 5
        strike_down_default = round(lower_range_1std / 5) * 5
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📍 Configuración de Strikes")
            
            st.info(f"""
            💡 **Rangos de Referencia (1σ):**
            - Precio Actual: **${current_price:.2f}**
            - Expected Move: **±${expected_move_1std:.2f}**
            - Rango Superior: **${upper_range_1std:.2f}**
            - Rango Inferior: **${lower_range_1std:.2f}**
            - Strike UP Sugerido (redondeado): **${strike_up_default:.0f}**
            - Strike DOWN Sugerido (redondeado): **${strike_down_default:.0f}**
            """)
            
            st.markdown("---")
            
            st.markdown(f"**Strike ATM** (Precio actual: ${current_price:.0f})")
            strike_atm_input = st.number_input(
                "Strike ATM",
                min_value=0.0,
                value=float(atm_rounded),
                step=5.0,
                key='strike_atm_tc',
                help="Strike al dinero (ATM)",
                label_visibility="collapsed"
            )
            
            option_type_atm = st.selectbox(
                "Tipo de Opción ATM",
                ["PUT", "CALL"],
                index=0,
                key='option_type_atm_tc'
            )
            
            st.markdown("---")
            
            st.markdown(f"**Strike DOWN** (Calculado: ${lower_range_1std:.2f})")
            strike_down_input = st.number_input(
                "Strike DOWN (debajo ATM)",
                min_value=0.0,
                value=float(strike_down_default),
                step=5.0,
                key='strike_down_tc',
                help="Strike por debajo del ATM basado en Expected Move",
                label_visibility="collapsed"
            )
            
            option_type_down = st.selectbox(
                "Tipo de Opción DOWN",
                ["PUT", "CALL"],
                index=0,
                key='option_type_down_tc'
            )
            
            st.markdown("---")
            
            st.markdown(f"**Strike UP** (Calculado: ${upper_range_1std:.2f})")
            strike_up_input = st.number_input(
                "Strike UP (arriba ATM)",
                min_value=0.0,
                value=float(strike_up_default),
                step=5.0,
                key='strike_up_tc',
                help="Strike por arriba del ATM basado en Expected Move",
                label_visibility="collapsed"
            )
            
            option_type_up = st.selectbox(
                "Tipo de Opción UP",
                ["CALL", "PUT"],
                index=0,
                key='option_type_up_tc'
            )
        
        with col2:
            st.markdown("#### 📅 Fechas de Expiración")
            
            dte_front_input = st.date_input(
                "DTE FRONT (Venta)",
                value=expiration_date,
                min_value=date.today() + timedelta(days=1),
                max_value=date.today() + timedelta(days=365),
                key='dte_front_tc',
                help="Fecha de expiración de las opciones vendidas"
            )
            
            default_back = dte_front_input + timedelta(days=7)
            
            dte_back_input = st.date_input(
                "DTE BACK (Compra)",
                value=default_back,
                min_value=dte_front_input + timedelta(days=1),
                max_value=date.today() + timedelta(days=365),
                key='dte_back_tc',
                help="Fecha de expiración de las opciones compradas"
            )
            
            days_diff = (dte_back_input - dte_front_input).days
            st.success(f"📅 Diferencia: **{days_diff} días**")
            
            st.markdown("---")
            
            st.markdown("#### 🏷️ Configuración Adicional")
            
            basket_tag = st.text_input(
                "Basket Tag",
                value="TripleCal1",
                key='basket_tag_tc',
                help="Etiqueta para identificar el grupo de órdenes"
            )
            
            st.markdown("---")
            
            st.markdown("#### 📦 Cantidad de Contratos")
            
            quantity_input = st.number_input(
                "Cantidad por orden",
                min_value=1,
                value=1,
                step=1,
                key='quantity_tc',
                help="Número de contratos por cada orden"
            )
        
        st.markdown("---")
        
        if st.button("🎯 Generar CSV para IBKR", type="primary", use_container_width=True):
            
            st.session_state.strike_atm_p6 = strike_atm_input
            st.session_state.strike_up_p6 = strike_up_input
            st.session_state.strike_down_p6 = strike_down_input
            st.session_state.dte_front_p6 = dte_front_input
            st.session_state.dte_back_p6 = dte_back_input
            
            front_date_str = dte_front_input.strftime("%Y%m%d")
            back_date_str = dte_back_input.strftime("%Y%m%d")
            
            orders = []
            
            strike_configs = [
                {'strike': strike_down_input, 'type': option_type_down, 'label': 'DOWN'},
                {'strike': strike_atm_input, 'type': option_type_atm, 'label': 'ATM'},
                {'strike': strike_up_input, 'type': option_type_up, 'label': 'UP'}
            ]
            
            for config in strike_configs:
                strike_int = int(config['strike'])
                right_letter = "C" if config['type'] == "CALL" else "P"
                
                # SELL - FRONT
                orders.append({
                    'Action': 'SELL',
                    'Quantity': quantity_input,
                    'Symbol': selected_ticker,
                    'SecType': 'OPT',
                    'LastTradingDayOrContractMonth': front_date_str,
                    'Strike': strike_int,
                    'Right': right_letter,
                    'Exchange': 'SMART',
                    'Currency': 'USD',
                    'BasketTag': basket_tag
                })
                
                # BUY - BACK
                orders.append({
                    'Action': 'BUY',
                    'Quantity': quantity_input,
                    'Symbol': selected_ticker,
                    'SecType': 'OPT',
                    'LastTradingDayOrContractMonth': back_date_str,
                    'Strike': strike_int,
                    'Right': right_letter,
                    'Exchange': 'SMART',
                    'Currency': 'USD',
                    'BasketTag': basket_tag
                })
            
            df_orders = pd.DataFrame(orders, columns=[
                'Action', 'Quantity', 'Symbol', 'SecType', 'LastTradingDayOrContractMonth', 
                'Strike', 'Right', 'Exchange', 'Currency', 'BasketTag'
            ])
            
            st.success("✅ Estructura de órdenes generada exitosamente!")
            
            st.markdown("### 📋 Vista Previa - IBKR Basket Orders")
            
            st.dataframe(
                df_orders,
                hide_index=True,
                use_container_width=True,
                column_config={
                    'Action': st.column_config.TextColumn('Action', width="small"),
                    'Quantity': st.column_config.NumberColumn('Quantity', format="%d", width="small"),
                    'Symbol': st.column_config.TextColumn('Symbol', width="small"),
                    'SecType': st.column_config.TextColumn('SecType', width="small"),
                    'LastTradingDayOrContractMonth': st.column_config.TextColumn('Expiry', width="medium"),
                    'Strike': st.column_config.NumberColumn('Strike', format="%d"),
                    'Right': st.column_config.TextColumn('Right', width="small"),
                    'Exchange': st.column_config.TextColumn('Exchange', width="small"),
                    'Currency': st.column_config.TextColumn('Currency', width="small"),
                    'BasketTag': st.column_config.TextColumn('BasketTag', width="medium")
                }
            )
            
            st.markdown("---")
            
            csv_content = generate_ibkr_basket_csv(df_orders)
            filename = f"IBKR_{selected_ticker}_TP_{date.today().strftime('%d_%m_%Y')}.csv"
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    label="📥 Descargar CSV para IBKR",
                    data=csv_content,
                    file_name=filename,
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )
            
            st.markdown("---")
            
            st.markdown("### 📊 Resumen de la Estructura")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📍 Strikes Configurados**")
                st.write(f"- DOWN: ${strike_down_input:.0f} ({option_type_down})")
                st.write(f"- ATM: ${strike_atm_input:.0f} ({option_type_atm})")
                st.write(f"- UP: ${strike_up_input:.0f} ({option_type_up})")
            
            with col2:
                st.markdown("**📊 Total de Órdenes**")
                st.metric("Órdenes Totales", len(df_orders))
                st.write(f"- SELL (Front {dte_front_input.strftime('%Y-%m-%d')}): 3")
                st.write(f"- BUY (Back {dte_back_input.strftime('%Y-%m-%d')}): 3")
            
            with col3:
                st.markdown("**⚙️ Configuración**")
                st.write(f"- Ticker: {selected_ticker}")
                st.write(f"- Cantidad: {quantity_input} contratos")
                st.write(f"- Spread: {days_diff} días")
                st.write(f"- Basket: {basket_tag}")
            
            st.markdown("---")
            
            st.markdown("### 📝 Instrucciones para importar en IBKR")
            
            st.info("""
            **Pasos para importar el archivo CSV en Interactive Brokers:**
            
            1. Descarga el archivo CSV usando el botón de arriba
            2. En TWS (Trader Workstation), ve a **Trading Tools → Basket Trader**
            3. Click en **Import Basket**
            4. Selecciona el archivo CSV descargado
            5. Revisa las órdenes importadas
            6. Ajusta los precios límite según el mercado actual
            7. Transmite las órdenes cuando estés listo
            
            ⚠️ **Importante:** 
            - Este archivo solo contiene la estructura de las órdenes
            - Deberás ingresar los precios manualmente en IBKR
            - Verifica todos los detalles antes de transmitir
            - Asegúrate de tener suficiente margen disponible
            """)
        
        st.markdown("---")
        
        # ==============================================================================
        # SECCIÓN 7: AJUSTES
        # ==============================================================================
        st.header("7. Ajustes - Generador de Calendar Individual")
        
        st.markdown("""
        Esta sección te permite generar un **Calendar Spread individual** basado en el precio actual del mercado 
        y el Expected Move (1σ). Úsalo para ajustar tu posición o abrir nuevas posiciones.
        """)
        
        expected_move_1std_adj = details['straddle_price'] * 1.25
        current_price_adj = get_current_price(selected_ticker)
        
        if current_price_adj is None:
            st.warning("⚠️ No se pudo obtener el precio actual para calcular el ajuste.")
        else:
            st.success(f"✅ Precio actual actualizado: **${current_price_adj:.2f}**")
            
            strike_adj_up_calc = current_price_adj + expected_move_1std_adj
            strike_adj_down_calc = current_price_adj - expected_move_1std_adj
            strike_adj_atm_calc = current_price_adj
            
            strike_adj_up_rounded = round(strike_adj_up_calc / 5) * 5
            strike_adj_down_rounded = round(strike_adj_down_calc / 5) * 5
            strike_adj_atm_rounded = round(strike_adj_atm_calc / 5) * 5
            
            st.info(f"""
            💡 **Cálculos de Ajuste (1σ):**
            - Precio Actual: **${current_price_adj:.2f}**
            - Expected Move (1σ): **±${expected_move_1std_adj:.2f}**
            - Strike UP Calculado: **${strike_adj_up_calc:.2f}** → Redondeado: **${strike_adj_up_rounded:.0f}**
            - Strike ATM Calculado: **${strike_adj_atm_calc:.2f}** → Redondeado: **${strike_adj_atm_rounded:.0f}**
            - Strike DOWN Calculado: **${strike_adj_down_calc:.2f}** → Redondeado: **${strike_adj_down_rounded:.0f}**
            """)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📍 Configuración del Strike de Ajuste")
                
                default_atm = st.session_state.strike_atm_p6 if st.session_state.strike_atm_p6 else strike_adj_atm_rounded
                default_up = st.session_state.strike_up_p6 if st.session_state.strike_up_p6 else strike_adj_up_rounded
                default_down = st.session_state.strike_down_p6 if st.session_state.strike_down_p6 else strike_adj_down_rounded
                
                st.markdown("**Strikes de Referencia (Punto 6)**")
                
                strike_ref_atm = st.number_input(
                    "Strike ATM de referencia",
                    min_value=0.0,
                    value=float(default_atm),
                    step=5.0,
                    key='strike_ref_atm_adj',
                    help="Strike ATM usado en el punto 6 o precio actual"
                )
                
                strike_ref_up = st.number_input(
                    "Strike UP de referencia",
                    min_value=0.0,
                    value=float(default_up),
                    step=5.0,
                    key='strike_ref_up_adj',
                    help="Strike UP usado en el punto 6"
                )
                
                strike_ref_down = st.number_input(
                    "Strike DOWN de referencia",
                    min_value=0.0,
                    value=float(default_down),
                    step=5.0,
                    key='strike_ref_down_adj',
                    help="Strike DOWN usado en el punto 6"
                )
                
                st.markdown("---")
                
                st.markdown("**Selecciona el Strike de Referencia para Comparar**")
                strike_comparison_options = {
                    f"ATM (${strike_ref_atm:.0f})": strike_ref_atm,
                    f"UP (${strike_ref_up:.0f})": strike_ref_up,
                    f"DOWN (${strike_ref_down:.0f})": strike_ref_down
                }
                
                selected_strike_ref_label = st.selectbox(
                    "Strike de referencia",
                    list(strike_comparison_options.keys()),
                    index=0,
                    key='strike_ref_comparison_adj'
                )
                
                strike_ref_selected = strike_comparison_options[selected_strike_ref_label]
                
                if current_price_adj < strike_ref_selected:
                    suggested_option_type = "PUT"
                    suggested_strike_calc = current_price_adj - expected_move_1std_adj
                    direction = "debajo"
                else:
                    suggested_option_type = "CALL"
                    suggested_strike_calc = current_price_adj + expected_move_1std_adj
                    direction = "arriba"
                
                suggested_strike_rounded = round(suggested_strike_calc / 5) * 5
                
                st.info(f"""
                📊 **Análisis de Posición:**
                - Precio Actual: **${current_price_adj:.2f}**
                - Strike Referencia: **${strike_ref_selected:.0f}**
                - Posición: Precio está **{direction}** del strike
                - Tipo Sugerido: **{suggested_option_type}**
                - Strike Sugerido: **${suggested_strike_calc:.2f}** → **${suggested_strike_rounded:.0f}**
                """)
                
                st.markdown("---")
                
                st.markdown("**Strike Final del Ajuste**")
                strike_adjustment = st.number_input(
                    "Strike para el Calendar",
                    min_value=0.0,
                    value=float(suggested_strike_rounded),
                    step=5.0,
                    key='strike_adjustment',
                    help="Strike final que se usará en el Calendar Spread"
                )
                
                option_type_adjustment = st.selectbox(
                    "Tipo de Opción",
                    ["PUT", "CALL"],
                    index=0 if suggested_option_type == "PUT" else 1,
                    key='option_type_adjustment'
                )
            
            with col2:
                st.markdown("#### 📅 Fechas de Expiración")
                
                default_front = st.session_state.dte_front_p6 if st.session_state.dte_front_p6 else expiration_date
                default_back = st.session_state.dte_back_p6 if st.session_state.dte_back_p6 else (expiration_date + timedelta(days=7))
                
                dte_front_adj = st.date_input(
                    "DTE FRONT (Venta) - Mismas del Punto 6",
                    value=default_front,
                    min_value=date.today() + timedelta(days=1),
                    max_value=date.today() + timedelta(days=365),
                    key='dte_front_adj',
                    help="Fecha de expiración de la opción vendida"
                )
                
                dte_back_adj = st.date_input(
                    "DTE BACK (Compra) - Mismas del Punto 6",
                    value=default_back,
                    min_value=dte_front_adj + timedelta(days=1),
                    max_value=date.today() + timedelta(days=365),
                    key='dte_back_adj',
                    help="Fecha de expiración de la opción comprada"
                )
                
                days_diff_adj = (dte_back_adj - dte_front_adj).days
                st.success(f"📅 Diferencia: **{days_diff_adj} días**")
                
                st.markdown("---")
                
                st.markdown("#### 🏷️ Configuración Adicional")
                
                basket_tag_adj = st.text_input(
                    "Basket Tag para Ajuste",
                    value="Adjustment1",
                    key='basket_tag_adj',
                    help="Etiqueta para identificar esta orden de ajuste"
                )
                
                st.markdown("---")
                
                st.markdown("#### 📦 Cantidad de Contratos")
                
                quantity_adj = st.number_input(
                    "Cantidad",
                    min_value=1,
                    value=1,
                    step=1,
                    key='quantity_adj',
                    help="Número de contratos para el ajuste"
                )
            
            st.markdown("---")
            
            if st.button("🎯 Generar CSV de Ajuste", type="primary", use_container_width=True):
                
                front_date_adj_str = dte_front_adj.strftime("%Y%m%d")
                back_date_adj_str = dte_back_adj.strftime("%Y%m%d")
                
                orders_adj = []
                
                strike_int_adj = int(strike_adjustment)
                right_letter_adj = "C" if option_type_adjustment == "CALL" else "P"
                
                # SELL - FRONT
                orders_adj.append({
                    'Action': 'SELL',
                    'Quantity': quantity_adj,
                    'Symbol': selected_ticker,
                    'SecType': 'OPT',
                    'LastTradingDayOrContractMonth': front_date_adj_str,
                    'Strike': strike_int_adj,
                    'Right': right_letter_adj,
                    'Exchange': 'SMART',
                    'Currency': 'USD',
                    'BasketTag': basket_tag_adj
                })
                
                # BUY - BACK
                orders_adj.append({
                    'Action': 'BUY',
                    'Quantity': quantity_adj,
                    'Symbol': selected_ticker,
                    'SecType': 'OPT',
                    'LastTradingDayOrContractMonth': back_date_adj_str,
                    'Strike': strike_int_adj,
                    'Right': right_letter_adj,
                    'Exchange': 'SMART',
                    'Currency': 'USD',
                    'BasketTag': basket_tag_adj
                })
                
                df_orders_adj = pd.DataFrame(orders_adj, columns=[
                    'Action', 'Quantity', 'Symbol', 'SecType', 'LastTradingDayOrContractMonth', 
                    'Strike', 'Right', 'Exchange', 'Currency', 'BasketTag'
                ])
                
                st.success("✅ Orden de ajuste generada exitosamente!")
                
                st.markdown("### 📋 Vista Previa - Ajuste Calendar")
                
                st.dataframe(
                    df_orders_adj,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        'Action': st.column_config.TextColumn('Action', width="small"),
                        'Quantity': st.column_config.NumberColumn('Quantity', format="%d", width="small"),
                        'Symbol': st.column_config.TextColumn('Symbol', width="small"),
                        'SecType': st.column_config.TextColumn('SecType', width="small"),
                        'LastTradingDayOrContractMonth': st.column_config.TextColumn('Expiry', width="medium"),
                        'Strike': st.column_config.NumberColumn('Strike', format="%d"),
                        'Right': st.column_config.TextColumn('Right', width="small"),
                        'Exchange': st.column_config.TextColumn('Exchange', width="small"),
                        'Currency': st.column_config.TextColumn('Currency', width="small"),
                        'BasketTag': st.column_config.TextColumn('BasketTag', width="medium")
                    }
                )
                
                st.markdown("---")
                
                csv_content_adj = generate_ibkr_basket_csv(df_orders_adj)
                filename_adj = f"IBKR_{selected_ticker}_ADJ_{date.today().strftime('%d_%m_%Y')}.csv"
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.download_button(
                        label="📥 Descargar CSV de Ajuste",
                        data=csv_content_adj,
                        file_name=filename_adj,
                        mime="text/csv",
                        type="primary",
                        use_container_width=True
                    )
                
                st.markdown("---")
                
                st.markdown("### 📊 Resumen del Ajuste")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**📍 Configuración**")
                    st.write(f"- Strike: ${strike_adjustment:.0f}")
                    st.write(f"- Tipo: {option_type_adjustment}")
                    st.write(f"- Cantidad: {quantity_adj}")
                
                with col2:
                    st.markdown("**📅 Fechas**")
                    st.write(f"- SELL: {dte_front_adj.strftime('%Y-%m-%d')}")
                    st.write(f"- BUY: {dte_back_adj.strftime('%Y-%m-%d')}")
                    st.write(f"- Spread: {days_diff_adj} días")
                
                with col3:
                    st.markdown("**⚙️ Datos**")
                    st.write(f"- Ticker: {selected_ticker}")
                    st.write(f"- Basket: {basket_tag_adj}")
                    st.write(f"- Órdenes: {len(df_orders_adj)}")

# ==============================================================================
# PUNTO DE ENTRADA PROTEGIDO
# ==============================================================================

if __name__ == "__main__":
    
    if check_password():
        main_tp_calculos()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder a TP Cálculos.")
