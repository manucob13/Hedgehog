import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import plotly.graph_objects as go
from utils.utils import check_password
from utils.utils_schwab import connect_to_schwab, get_current_price_schwab, obtener_datos_opcion, get_atm_strike_schwab
import io

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="TC Cálculos - Expected Move", layout="wide")

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def calculate_expected_move_schwab(client, ticker, expiration_date, current_price, std_multiplier=1.0):
    """Calcula el Expected Move basado en el straddle ATM usando datos de Schwab."""
    try:
        if client is None:
            return None, None, "Cliente de Schwab no disponible"
        
        print(f"\n{'='*60}")
        print(f"CALCULANDO EXPECTED MOVE")
        print(f"Ticker: {ticker}")
        print(f"Fecha expiración: {expiration_date}")
        print(f"Precio actual: {current_price}")
        print(f"Multiplicador: {std_multiplier}σ")
        print(f"{'='*60}\n")
        
        # Obtener el strike ATM real de la cadena de opciones
        atm_strike = get_atm_strike_schwab(client, ticker, current_price, expiration_date)
        
        if atm_strike is None:
            error_msg = f"No se pudo obtener el strike ATM para {ticker} en fecha {expiration_date}"
            print(f"❌ {error_msg}")
            return None, None, error_msg
        
        print(f"\n📌 Strike ATM seleccionado: {atm_strike}")
        print(f"Obteniendo datos del CALL ATM...")
        
        # Obtener datos del CALL ATM
        call_mid, call_delta, call_theta = obtener_datos_opcion(
            client, ticker, atm_strike, 'CALL', expiration_date
        )
        
        if call_mid is None:
            error_msg = f"No se encontraron datos del CALL ATM (strike {atm_strike})"
            print(f"❌ {error_msg}")
            return None, None, error_msg
        
        print(f"\n📌 Obteniendo datos del PUT ATM...")
        
        # Obtener datos del PUT ATM
        put_mid, put_delta, put_theta = obtener_datos_opcion(
            client, ticker, atm_strike, 'PUT', expiration_date
        )
        
        if put_mid is None:
            error_msg = f"No se encontraron datos del PUT ATM (strike {atm_strike})"
            print(f"❌ {error_msg}")
            return None, None, error_msg
        
        # Calcular straddle price
        straddle_price = call_mid + put_mid
        price_type = "Mid Price"
        
        # Expected Move = Straddle Price * 1.25 * std_multiplier
        expected_move = straddle_price * 1.25 * std_multiplier
        
        print(f"\n{'='*60}")
        print(f"RESULTADO EXPECTED MOVE")
        print(f"Straddle Price: ${straddle_price:.2f}")
        print(f"Expected Move ({std_multiplier}σ): ${expected_move:.2f}")
        print(f"{'='*60}\n")
        
        # Crear diccionario con detalles
        details = {
            'atm_strike': atm_strike,
            'call_mid': call_mid,
            'call_delta': call_delta,
            'call_theta': call_theta,
            'put_mid': put_mid,
            'put_delta': put_delta,
            'put_theta': put_theta,
            'straddle_price': straddle_price,
            'price_type': price_type,
            'std_multiplier': std_multiplier
        }
        
        return expected_move, details, None
        
    except Exception as e:
        error_msg = f"Error calculando Expected Move con Schwab: {e}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return None, None, error_msg


@st.cache_data(ttl=600, show_spinner=False)
def get_historical_prices_yf(ticker, days=3):
    """Obtiene precios históricos del ticker usando Yahoo Finance (solo últimos 3 días)."""
    import time
    
    try:
        import yfinance as yf
        
        # Ajustar símbolo para Yahoo Finance
        yf_ticker = '^SPX' if ticker == 'SPX' else ticker
        
        # Solo 1 reintento para no saturar
        max_retries = 2
        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(yf_ticker)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                # Obtener datos históricos con configuración mínima
                df = stock.history(
                    start=start_date, 
                    end=end_date,
                    interval='1d',
                    actions=False
                )
                
                if not df.empty:
                    return df
                
                # Si está vacío, esperar y reintentar solo una vez
                if attempt < max_retries - 1:
                    time.sleep(3)
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(3)
                else:
                    return None
        
        return None
        
    except Exception as e:
        return None


def create_simple_chart_from_current_price(current_price, upper_range, lower_range, 
                                           expiration_date, std_multiplier, selected_ticker):
    """Crea un gráfico simple usando solo el precio actual cuando Yahoo Finance falla."""
    
    fig = go.Figure()
    
    # Crear puntos para simular una línea horizontal del precio actual
    today = datetime.now()
    exp_datetime = datetime.combine(expiration_date, datetime.min.time())
    
    # Generar fechas desde hoy hasta expiración
    num_days = max(3, (expiration_date - date.today()).days + 1)
    dates = [today + timedelta(days=i) for i in range(num_days)]
    prices = [current_price] * num_days
    
    # Línea del precio actual
    fig.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode='lines',
        name='Precio Actual',
        line=dict(color='yellow', width=3, dash='dash')
    ))
    
    # Líneas horizontales para los rangos
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
    
    # Línea vertical - Fecha de Expiración
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
    
    # Configuración del layout
    fig.update_layout(
        title=f"Expected Move - {selected_ticker} ({std_multiplier}σ) - Vista Simplificada",
        xaxis_title="Fecha",
        yaxis_title="Precio",
        template="plotly_dark",
        height=500,
        hovermode='x unified',
        showlegend=True,
        yaxis=dict(range=[lower_range * 0.95, upper_range * 1.05])
    )
    
    return fig


def initialize_session_state():
    """Inicializa las variables de session_state necesarias."""
    state_vars = {
        'calculation_done': False,
        'current_price': None,
        'expected_move': None,
        'details': None,
        'selected_ticker': None,
        'expiration_date': None,
        'std_multiplier': None,
        'strike_atm_p6': None,
        'strike_up_p6': None,
        'strike_down_p6': None,
        'dte_front_p6': None,
        'dte_back_p6': None,
        'df_strategy_tc': None,
        'order_preview_tc': False,
        'df_strategy_adj': None,
        'order_preview_adj': False,
        'schwab_client': None
    }
    
    for var, default_value in state_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value


def create_strategy_orders(strikes_config, front_date, back_date, quantity, ticker):
    """Crea las órdenes para una estrategia de opciones."""
    orders = []
    
    for config in strikes_config:
        strike_int = int(config['strike'])
        right_letter = "C" if config['type'] == "CALL" else "P"
        
        # SELL en DTE FRONT
        orders.append({
            'Action': 'SELL',
            'Quantity': quantity,
            'Symbol': ticker,
            'SecType': 'OPT',
            'Expiry': front_date,
            'Strike': strike_int,
            'Right': right_letter,
            'Exchange': 'SMART',
            'Currency': 'USD',
            'Label': f'{config["label"]} Front'
        })
        
        # BUY en DTE BACK
        orders.append({
            'Action': 'BUY',
            'Quantity': quantity,
            'Symbol': ticker,
            'SecType': 'OPT',
            'Expiry': back_date,
            'Strike': strike_int,
            'Right': right_letter,
            'Exchange': 'SMART',
            'Currency': 'USD',
            'Label': f'{config["label"]} Back'
        })
    
    return pd.DataFrame(orders)


# ==============================================================================
# FUNCIÓN PRINCIPAL - TP CÁLCULOS
# ==============================================================================

def main_tp_calculos():
    
    initialize_session_state()
    
    # --- TÍTULO PRINCIPAL ---
    st.markdown("<h1><span style='font-size: 1.5em;'>🎯</span> TC Cálculos - Expected Move</h1>", unsafe_allow_html=True)
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
        selected_ticker = st.selectbox(
            "Selecciona el Ticker",
            ['QQQ', 'SPX'],
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
        
        with st.spinner(f"Conectando con Schwab y obteniendo datos de {selected_ticker}..."):
            
            # Conectar con Schwab
            schwab_client = connect_to_schwab()
            
            if schwab_client is None:
                st.error("❌ No se pudo conectar con Schwab. Verifica la configuración.")
                st.stop()
            
            st.session_state.schwab_client = schwab_client
            st.success("✅ Conectado con Schwab exitosamente")
            
            # Obtener precio actual desde Schwab
            current_price = get_current_price_schwab(schwab_client, selected_ticker)
            
            if current_price is None:
                st.error(f"❌ No se pudo obtener el precio actual de {selected_ticker} desde Schwab.")
                st.info("💡 Intenta con otro ticker o verifica la conexión con Schwab.")
                st.stop()
            
            st.success(f"✅ Precio actual de {selected_ticker}: **${current_price:.2f}**")
            
            # Calcular Expected Move con Schwab
            expected_move, details, error = calculate_expected_move_schwab(
                schwab_client,
                selected_ticker,
                expiration_date,
                current_price,
                std_multiplier
            )
            
            if error:
                st.error(f"❌ Error: {error}")
                st.stop()
            
            st.success(f"✅ Expected Move calculado: **${expected_move:.2f}**")
            
            # Guardar en session_state
            st.session_state.current_price = current_price
            st.session_state.expected_move = expected_move
            st.session_state.details = details
            st.session_state.selected_ticker = selected_ticker
            st.session_state.expiration_date = expiration_date
            st.session_state.std_multiplier = std_multiplier
            st.session_state.calculation_done = True
    
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
        
        # Detalles del Straddle ATM
        st.markdown("### 💰 Detalles del Straddle ATM")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📞 CALL ATM**")
            call_details = {
                'Strike': f"${details['atm_strike']:.2f}",
                'Mid': f"${details['call_mid']:.2f}"
            }
            if details['call_delta'] is not None:
                call_details['Delta'] = f"{details['call_delta']:.4f}"
            if details['call_theta'] is not None:
                call_details['Theta'] = f"{details['call_theta']:.4f}"
            
            df_call = pd.DataFrame(list(call_details.items()), columns=['Métrica', 'Valor'])
            st.dataframe(df_call, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("**📉 PUT ATM**")
            put_details = {
                'Strike': f"${details['atm_strike']:.2f}",
                'Mid': f"${details['put_mid']:.2f}"
            }
            if details['put_delta'] is not None:
                put_details['Delta'] = f"{details['put_delta']:.4f}"
            if details['put_theta'] is not None:
                put_details['Theta'] = f"{details['put_theta']:.4f}"
            
            df_put = pd.DataFrame(list(put_details.items()), columns=['Métrica', 'Valor'])
            st.dataframe(df_put, hide_index=True, use_container_width=True)
        
        st.markdown("---")
        
        # Tabla resumen
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
        # SECCIÓN 3: GRÁFICO - VELAS JAPONESAS
        # ==============================================================================
        st.header("3. Visualización del Expected Move")
        
        # Intentar obtener datos históricos de Yahoo Finance
        with st.spinner("📊 Obteniendo datos históricos (últimos 3 días)..."):
            df_hist = get_historical_prices_yf(selected_ticker, days=3)
        
        if df_hist is not None and not df_hist.empty:
            
            try:
                fig = go.Figure()
                
                # Convertir el índice a timezone-naive
                df_hist_plot = df_hist.copy()
                if df_hist_plot.index.tz is not None:
                    df_hist_plot.index = df_hist_plot.index.tz_localize(None)
                
                # Gráfico de velas japonesas
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
                
                # Líneas horizontales
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
                
                # Línea vertical - Fecha de Expiración
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
                
                st.success("📊 **Datos históricos de Yahoo Finance (últimos 3 días)**")
                
            except Exception as e:
                st.warning(f"⚠️ Error al crear el gráfico con velas: {e}")
                
                # Fallback a gráfico simple
                fig_simple = create_simple_chart_from_current_price(
                    current_price, upper_range, lower_range, 
                    expiration_date, std_multiplier, selected_ticker
                )
                st.plotly_chart(fig_simple, use_container_width=True)
                st.info("📊 **Mostrando vista simplificada basada en precio actual**")
        
        else:
            # Si Yahoo Finance falla, usar gráfico simple con el precio actual
            st.info("💡 No se pudieron obtener datos históricos. Mostrando vista simplificada.")
            
            fig_simple = create_simple_chart_from_current_price(
                current_price, upper_range, lower_range, 
                expiration_date, std_multiplier, selected_ticker
            )
            
            st.plotly_chart(fig_simple, use_container_width=True)
            
            st.success("📊 **Vista simplificada basada en el precio actual de Schwab**")
        
        st.markdown("---")
        
        # ==============================================================================
        # SECCIÓN 4: COMPARACIÓN DE DESVIACIONES ESTÁNDAR
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
        
        df_comparison_std = pd.DataFrame(comparison_data)
        
        st.markdown("""
        Esta tabla muestra cómo varían los rangos esperados según diferentes niveles de confianza estadística:
        """)
        
        st.dataframe(df_comparison_std, hide_index=True, use_container_width=True)
        
        st.markdown("---")
        
        # ==============================================================================
        # SECCIÓN 5: INFORMACIÓN ADICIONAL
        # ==============================================================================
        st.header("5. Información Adicional")
        
        prob_text = {1.0: "68%", 1.5: "87%", 2.0: "95%"}.get(std_multiplier, "N/A")
        
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
        - **Precios de Opciones:** Schwab API
        - **Precios del Activo:** Schwab API
        - **Datos Históricos:** Yahoo Finance
        
        ### 🧮 Fórmula del Expected Move
        ```
        Expected Move = Straddle Price × 1.25 × σ
        ```
        
        Donde:
        - **Straddle Price** = Call Mid + Put Mid (ATM)
        - **1.25** = Factor de ajuste para 1 desviación estándar completa
        - **σ** = Multiplicador de desviaciones estándar (1.0, 1.5, o 2.0)
        
        ### 📊 Niveles de Confianza
        - **1σ** ≈ 68% de probabilidad (rango más conservador)
        - **1.5σ** ≈ 87% de probabilidad (rango intermedio)
        - **2σ** ≈ 95% de probabilidad (rango más amplio)
        """)
        
        st.markdown("---")
        
        # ==============================================================================
        # SECCIÓN 6: GENERADOR DE ESTRUCTURA TRIPLE CALENDAR - ENVÍO A IBKR
        # ==============================================================================
        st.header("6. Generador de Estructura - Triple Calendar (Envío a IBKR)")
        
        st.markdown("""
        Configura los strikes y fechas de expiración para generar una orden **Triple Calendar** 
        que se enviará directamente a **Interactive Brokers** a través de TWS/Gateway.
        """)
        
        # Calcular Expected Move de 1 desviación estándar
        expected_move_1std = details['straddle_price'] * 1.25
        upper_range_1std = current_price + expected_move_1std
        lower_range_1std = current_price - expected_move_1std
        
        # Redondear strikes para los valores por defecto
        atm_rounded = round(details['atm_strike'] / 5) * 5
        strike_up_default = round(upper_range_1std / 5) * 5
        strike_down_default = round(lower_range_1std / 5) * 5
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Configuración de Strikes")
            
            st.info(f"""
            💡 **Rangos de Referencia (1σ):**
            - Precio Actual: **${current_price:.2f}**
            - Expected Move: **±${expected_move_1std:.2f}**
            - Rango Superior: **${upper_range_1std:.2f}**
            - Rango Inferior: **${lower_range_1std:.2f}**
            - Strike UP Sugerido: **${strike_up_default:.0f}**
            - Strike DOWN Sugerido: **${strike_down_default:.0f}**
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
            
            st.markdown("#### 💰 Configuración de Orden")
            
            quantity_input = st.number_input(
                "Cantidad de Contratos",
                min_value=1,
                value=1,
                step=1,
                key='quantity_tc',
                help="Número de contratos por cada pierna del calendario"
            )
            
            limit_price_input = st.number_input(
                "Precio Límite (Total)",
                min_value=0.0,
                value=1.0,
                step=0.05,
                format="%.2f",
                key='limit_price_tc',
                help="Precio límite total para la estrategia (crédito o débito)"
            )
            
            st.info(f"""
            💡 **Configuración de Precio:**
            - Precio Límite: **${limit_price_input:.2f}**
            - Por contrato: **${limit_price_input / quantity_input if quantity_input > 0 else 0:.2f}**
            """)
            
            st.markdown("---")
            
            st.markdown("#### 📌 Configuración IBKR")
            
            ibkr_host = st.text_input("Host IBKR", value="127.0.0.1", key='ibkr_host_tc')
            ibkr_port = st.number_input("Puerto IBKR", min_value=1, max_value=65535, value=5000, step=1, key='ibkr_port_tc')
            ibkr_client_id = st.number_input("Client ID", min_value=1, value=1, step=1, key='ibkr_client_id_tc')
        
        st.markdown("---")
        
        if st.button("📝 Generar Vista Previa de Orden", type="primary", use_container_width=True):
            
            # Guardar strikes y DTEs en session_state para el punto 7
            st.session_state.strike_atm_p6 = strike_atm_input
            st.session_state.strike_up_p6 = strike_up_input
            st.session_state.strike_down_p6 = strike_down_input
            st.session_state.dte_front_p6 = dte_front_input
            st.session_state.dte_back_p6 = dte_back_input
            
            # Crear la estructura de órdenes
            strikes_config = [
                {'strike': strike_down_input, 'type': option_type_down, 'label': 'DOWN'},
                {'strike': strike_atm_input, 'type': option_type_atm, 'label': 'ATM'},
                {'strike': strike_up_input, 'type': option_type_up, 'label': 'UP'}
            ]
            
            df_strategy = create_strategy_orders(
                strikes_config,
                dte_front_input.strftime("%Y-%m-%d"),
                dte_back_input.strftime("%Y-%m-%d"),
                quantity_input,
                selected_ticker
            )
            
            st.session_state.df_strategy_tc = df_strategy
            st.session_state.order_preview_tc = True
            
            st.success("✅ Vista previa de orden generada exitosamente!")
        
        # Mostrar vista previa si está disponible
        if st.session_state.order_preview_tc and st.session_state.df_strategy_tc is not None:
            
            df_strategy = st.session_state.df_strategy_tc
            
            st.markdown("---")
            st.markdown("### 📋 Vista Previa - Orden Triple Calendar")
            
            st.dataframe(df_strategy, hide_index=True, use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 📊 Resumen de la Orden")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Piernas", len(df_strategy))
            with col2:
                st.metric("Cantidad por Pierna", quantity_input)
            with col3:
                st.metric("Precio Límite", f"${limit_price_input:.2f}")
            with col4:
                st.metric("Spread (días)", days_diff)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🎯 Strikes Configurados**")
                st.write(f"- DOWN: ${strike_down_input:.0f} ({option_type_down})")
                st.write(f"- ATM: ${strike_atm_input:.0f} ({option_type_atm})")
                st.write(f"- UP: ${strike_up_input:.0f} ({option_type_up})")
            
            with col2:
                st.markdown("**📅 Fechas de Expiración**")
                st.write(f"- FRONT (SELL): {dte_front_input.strftime('%Y-%m-%d')}")
                st.write(f"- BACK (BUY): {dte_back_input.strftime('%Y-%m-%d')}")
                st.write(f"- Diferencia: {days_diff} días")
            
            st.markdown("---")
            
            st.markdown("**📌 Configuración de Conexión IBKR**")
            st.write(f"- Host: {ibkr_host} | Puerto: {ibkr_port} | Client ID: {ibkr_client_id}")
            
            st.markdown("---")
            
            st.warning("⚠️ **IMPORTANTE:** Asegúrate de que TWS/Gateway esté ejecutándose y la configuración sea correcta.")
            
            if st.button("🚀 ENVIAR ORDEN A IBKR", type="primary", use_container_width=True):
                try:
                    from utils.utils_ibkr import send_strategy_order_ibkr
                except ImportError:
                    st.error("❌ Error: No se pudo importar send_strategy_order_ibkr")
                    st.stop()
                
                if limit_price_input <= 0:
                    st.error("❌ El precio límite debe ser mayor a 0")
                    st.stop()
                
                with st.spinner("📡 Enviando orden a IBKR..."):
                    result = send_strategy_order_ibkr(
                        df_strategy=df_strategy,
                        limit_price=limit_price_input,
                        host=ibkr_host,
                        port=int(ibkr_port),
                        client_id=int(ibkr_client_id),
                        quantity=quantity_input,
                        tif='DAY',
                        action='BUY',
                        timeout=10
                    )
                
                if result['success']:
                    st.success(f"✅ {result['message']}")
                    if result.get('order_id'):
                        st.info(f"📋 Order ID: {result['order_id']}")
                    if result.get('contracts'):
                        st.markdown("**✅ Contratos Calificados:**")
                        for i, c in enumerate(result['contracts'], 1):
                            st.write(f"{i}. {c.symbol} {c.lastTradeDateOrContractMonth} {c.strike} {c.right}")
                else:
                    st.error(f"❌ {result['message']}")
        
        st.markdown("---")
        
        # ==============================================================================
        # SECCIÓN 7: AJUSTES
        # ==============================================================================
        st.header("7. Ajustes - Generador de Calendar Individual")
        
        st.markdown("Esta sección te permite generar un **Calendar Spread individual** basado en el precio actual del mercado y el Expected Move (1σ).")
        
        # Actualizar precio actual
        schwab_client = st.session_state.schwab_client
        if schwab_client is None:
            st.warning("⚠️ No hay conexión activa con Schwab. Reconecta para actualizar el precio.")
            current_price_adj = current_price
        else:
            current_price_adj = get_current_price_schwab(schwab_client, selected_ticker)
            if current_price_adj is None:
                st.warning("⚠️ No se pudo obtener el precio actual actualizado. Usando precio anterior.")
                current_price_adj = current_price
        
        st.success(f"✅ Precio actual actualizado: **${current_price_adj:.2f}**")
        
        expected_move_1std_adj = details['straddle_price'] * 1.25
        
        strike_adj_up_calc = current_price_adj + expected_move_1std_adj
        strike_adj_down_calc = current_price_adj - expected_move_1std_adj
        strike_adj_atm_calc = current_price_adj
        
        strike_adj_up_rounded = round(strike_adj_up_calc / 5) * 5
        strike_adj_down_rounded = round(strike_adj_down_calc / 5) * 5
        strike_adj_atm_rounded = round(strike_adj_atm_calc / 5) * 5
        
        st.info(f"""
        💡 **Cálculos de Ajuste (1σ):**
        - Precio Actual: **${current_price_adj:.2f}**
        - Expected Move: **±${expected_move_1std_adj:.2f}**
        - Strike UP: **${strike_adj_up_calc:.2f}** → **${strike_adj_up_rounded:.0f}**
        - Strike ATM: **${strike_adj_atm_calc:.2f}** → **${strike_adj_atm_rounded:.0f}**
        - Strike DOWN: **${strike_adj_down_calc:.2f}** → **${strike_adj_down_rounded:.0f}**
        """)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Configuración del Strike de Ajuste")
            
            default_atm = st.session_state.strike_atm_p6 if st.session_state.strike_atm_p6 else strike_adj_atm_rounded
            default_up = st.session_state.strike_up_p6 if st.session_state.strike_up_p6 else strike_adj_up_rounded
            default_down = st.session_state.strike_down_p6 if st.session_state.strike_down_p6 else strike_adj_down_rounded
            
            st.markdown("**Strikes de Referencia (Punto 6)**")
            
            strike_ref_atm = st.number_input("Strike ATM ref", min_value=0.0, value=float(default_atm), step=5.0, key='strike_ref_atm_adj')
            strike_ref_up = st.number_input("Strike UP ref", min_value=0.0, value=float(default_up), step=5.0, key='strike_ref_up_adj')
            strike_ref_down = st.number_input("Strike DOWN ref", min_value=0.0, value=float(default_down), step=5.0, key='strike_ref_down_adj')
            
            st.markdown("---")
            st.markdown("**Selecciona el Strike de Referencia**")
            
            strike_comparison_options = {
                f"ATM (${strike_ref_atm:.0f})": strike_ref_atm,
                f"UP (${strike_ref_up:.0f})": strike_ref_up,
                f"DOWN (${strike_ref_down:.0f})": strike_ref_down
            }
            
            selected_strike_ref_label = st.selectbox("Strike de referencia", list(strike_comparison_options.keys()), key='strike_ref_comparison_adj')
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
            📊 **Análisis:**
            - Precio: **${current_price_adj:.2f}**
            - Strike Ref: **${strike_ref_selected:.0f}**
            - Posición: **{direction}**
            - Tipo Sugerido: **{suggested_option_type}**
            - Strike Sugerido: **${suggested_strike_rounded:.0f}**
            """)
            
            st.markdown("---")
            st.markdown("**Strike Final del Ajuste**")
            strike_adjustment = st.number_input("Strike para el Calendar", min_value=0.0, value=float(suggested_strike_rounded), step=5.0, key='strike_adjustment')
            option_type_adjustment = st.selectbox("Tipo de Opción", ["PUT", "CALL"], index=0 if suggested_option_type == "PUT" else 1, key='option_type_adjustment')
        
        with col2:
            st.markdown("#### 📅 Fechas de Expiración")
            
            # Valores por defecto seguros para las fechas
            if st.session_state.dte_front_p6 and isinstance(st.session_state.dte_front_p6, date):
                default_front = st.session_state.dte_front_p6
            else:
                default_front = expiration_date
            
            if st.session_state.dte_back_p6 and isinstance(st.session_state.dte_back_p6, date):
                default_back = st.session_state.dte_back_p6
            else:
                default_back = default_front + timedelta(days=7)
            
            dte_front_adj = st.date_input(
                "DTE FRONT (Venta)", 
                value=default_front, 
                min_value=date.today() + timedelta(days=1),
                max_value=date.today() + timedelta(days=365),
                key='dte_front_adj'
            )
            
            # Asegurar que default_back sea siempre mayor que dte_front_adj
            min_back_date = dte_front_adj + timedelta(days=1)
            if default_back <= dte_front_adj:
                default_back = min_back_date
            
            dte_back_adj = st.date_input(
                "DTE BACK (Compra)", 
                value=default_back, 
                min_value=min_back_date,
                max_value=date.today() + timedelta(days=365),
                key='dte_back_adj'
            )
            
            days_diff_adj = (dte_back_adj - dte_front_adj).days
            st.success(f"📅 Diferencia: **{days_diff_adj} días**")
            
            st.markdown("---")
            st.markdown("#### 💰 Configuración de Orden")
            
            quantity_adj = st.number_input("Cantidad de Contratos", min_value=1, value=1, step=1, key='quantity_adj')
            limit_price_adj = st.number_input("Precio Límite", min_value=0.0, value=1.0, step=0.05, format="%.2f", key='limit_price_adj')
            
            st.info(f"💡 Por contrato: **${limit_price_adj / quantity_adj if quantity_adj > 0 else 0:.2f}**")
            
            st.markdown("---")
            st.markdown("#### 📌 Configuración IBKR")
            
            ibkr_host_adj = st.text_input("Host", value="127.0.0.1", key='ibkr_host_adj')
            ibkr_port_adj = st.number_input("Puerto", min_value=1, max_value=65535, value=5000, step=1, key='ibkr_port_adj')
            ibkr_client_id_adj = st.number_input("Client ID", min_value=1, value=1, step=1, key='ibkr_client_id_adj')
        
        st.markdown("---")
        
        if st.button("📝 Generar Vista Previa de Ajuste", type="primary", use_container_width=True):
            
            strikes_config_adj = [{'strike': strike_adjustment, 'type': option_type_adjustment, 'label': 'Adjustment'}]
            
            df_strategy_adj = create_strategy_orders(
                strikes_config_adj,
                dte_front_adj.strftime("%Y-%m-%d"),
                dte_back_adj.strftime("%Y-%m-%d"),
                quantity_adj,
                selected_ticker
            )
            
            st.session_state.df_strategy_adj = df_strategy_adj
            st.session_state.order_preview_adj = True
            
            st.success("✅ Vista previa de ajuste generada!")
        
        if st.session_state.order_preview_adj and st.session_state.df_strategy_adj is not None:
            
            df_strategy_adj = st.session_state.df_strategy_adj
            
            st.markdown("---")
            st.markdown("### 📋 Vista Previa - Ajuste Calendar")
            st.dataframe(df_strategy_adj, hide_index=True, use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 📊 Resumen del Ajuste")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**🎯 Configuración**")
                st.write(f"Strike: ${strike_adjustment:.0f}")
                st.write(f"Tipo: {option_type_adjustment}")
                st.write(f"Cantidad: {quantity_adj}")
            
            with col2:
                st.markdown("**📅 Fechas**")
                st.write(f"SELL: {dte_front_adj.strftime('%Y-%m-%d')}")
                st.write(f"BUY: {dte_back_adj.strftime('%Y-%m-%d')}")
                st.write(f"Spread: {days_diff_adj} días")
            
            with col3:
                st.markdown("**⚙️ Datos**")
                st.write(f"Ticker: {selected_ticker}")
                st.write(f"Precio: ${limit_price_adj:.2f}")
                st.write(f"Órdenes: {len(df_strategy_adj)}")
            
            st.markdown("---")
            st.warning("⚠️ Asegúrate de que TWS/Gateway esté ejecutándose")
            
            if st.button("🚀 ENVIAR AJUSTE A IBKR", type="primary", use_container_width=True):
                try:
                    from utils.utils_ibkr import send_strategy_order_ibkr
                except ImportError:
                    st.error("❌ Error: No se pudo importar send_strategy_order_ibkr")
                    st.stop()
                
                if limit_price_adj <= 0:
                    st.error("❌ El precio límite debe ser mayor a 0")
                    st.stop()
                
                with st.spinner("📡 Enviando ajuste..."):
                    result = send_strategy_order_ibkr(
                        df_strategy=df_strategy_adj,
                        limit_price=limit_price_adj,
                        host=ibkr_host_adj,
                        port=int(ibkr_port_adj),
                        client_id=int(ibkr_client_id_adj),
                        quantity=quantity_adj,
                        tif='DAY',
                        action='BUY',
                        timeout=10
                    )
                
                if result['success']:
                    st.success(f"✅ {result['message']}")
                    if result.get('order_id'):
                        st.info(f"📋 Order ID: {result['order_id']}")
                else:
                    st.error(f"❌ {result['message']}")
            
            st.markdown("---")
            st.info("""
            **Sobre el Ajuste:**
            - Propósito: Añadir un calendar spread individual
            - Tipo: LIMIT | Acción: BUY | TIF: DAY
            
            **Cuándo Ajustar:**
            - El precio se ha movido significativamente
            - Necesitas rebalancear tu delta
            - Quieres añadir exposición en un nuevo strike
            """)


# ==============================================================================
# PUNTO DE ENTRADA PROTEGIDO
# ==============================================================================

if __name__ == "__main__":
    if check_password():
        main_tp_calculos()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral para acceder.")
