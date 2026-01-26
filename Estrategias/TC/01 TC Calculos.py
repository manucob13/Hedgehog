import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import plotly.graph_objects as go
from utils.utils import check_password
from utils.utils_schwab import connect_to_schwab, get_current_price_schwab, obtener_datos_opcion, get_atm_strike_schwab
import io

# --- CONFIGURACIÓN DE PÁGINA ---
# st.set_page_config(page_title="🦉 TC Calculos", layout="wide")

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
        'schwab_client': None,
        'calendar_strikes_configured': False,
        'strike_down': None,
        'strike_atm': None,
        'strike_up': None,
        'type_down': 'PUT',
        'type_atm': 'PUT',
        'type_up': 'CALL',
        'send_down': True,
        'send_atm': True,
        'send_up': True,
        'orders_reviewed': False
    }
    
    for var, default_value in state_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value


def create_calendar_pair_order(strike, option_type, front_date, back_date, quantity, ticker):
    """Crea un par de órdenes (SELL front + BUY back) para un calendar spread."""
    right_letter = "C" if option_type == "CALL" else "P"
    
    orders = [
        {
            'Action': 'SELL',
            'Quantity': quantity,
            'Symbol': ticker,
            'SecType': 'OPT',
            'Expiry': front_date,
            'Strike': int(strike),
            'Right': right_letter,
            'Exchange': 'SMART',
            'Currency': 'USD',
            'Label': 'Front'
        },
        {
            'Action': 'BUY',
            'Quantity': quantity,
            'Symbol': ticker,
            'SecType': 'OPT',
            'Expiry': back_date,
            'Strike': int(strike),
            'Right': right_letter,
            'Exchange': 'SMART',
            'Currency': 'USD',
            'Label': 'Back'
        }
    ]
    
    return pd.DataFrame(orders)


# ==============================================================================
# FUNCIÓN PRINCIPAL - TP CÁLCULOS
# ==============================================================================

def main_tp_calculos():
    
    initialize_session_state()
    
    # --- TÍTULO PRINCIPAL ---
    st.markdown("<h1><span style='font-size: 1.5em;'>🦉</span> TC Calculos - Expected Move</h1>", unsafe_allow_html=True)
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
    
    if st.button("🦉 Calcular Expected Move", type="primary", use_container_width=True):
        
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
            
            # Calcular strikes sugeridos
            strike_increment = 1.0 if selected_ticker == 'QQQ' else 5.0
            round_to = 1 if selected_ticker == 'QQQ' else 5
            
            expected_move_1std = details['straddle_price'] * 1.25
            upper_range_1std = current_price + expected_move_1std
            lower_range_1std = current_price - expected_move_1std
            
            atm_rounded = round(details['atm_strike'] / round_to) * round_to
            strike_up_default = round(upper_range_1std / round_to) * round_to
            strike_down_default = round(lower_range_1std / round_to) * round_to
            
            # Guardar en session_state
            st.session_state.current_price = current_price
            st.session_state.expected_move = expected_move
            st.session_state.details = details
            st.session_state.selected_ticker = selected_ticker
            st.session_state.expiration_date = expiration_date
            st.session_state.std_multiplier = std_multiplier
            st.session_state.calculation_done = True
            
            # Inicializar strikes sugeridos
            st.session_state.strike_down = strike_down_default
            st.session_state.strike_atm = atm_rounded
            st.session_state.strike_up = strike_up_default
            st.session_state.calendar_strikes_configured = False
            st.session_state.orders_reviewed = False
    
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
        
        # CALCULAR INCREMENTO DINÁMICO SEGÚN TICKER
        strike_increment = 1.0 if selected_ticker == 'QQQ' else 5.0
        round_to = 1 if selected_ticker == 'QQQ' else 5
        
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
        # SECCIÓN 5: CONFIGURACIÓN DE CALENDAR SPREADS (MEJORADA)
        # ==============================================================================
        st.header("5. Generador de Calendar Spreads")
        
        st.markdown("""
        Configura los strikes para los **Calendar Spreads**. Cada strike se enviará como un **par independiente** (SELL front + BUY back) a IBKR.
        """)
        
        # Calcular Expected Move de 1 desviación estándar para sugerencias
        expected_move_1std = details['straddle_price'] * 1.25
        upper_range_1std = current_price + expected_move_1std
        lower_range_1std = current_price - expected_move_1std
        
        # Configuración global de fechas
        st.markdown("### ⚙️ Configuración de Fechas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dte_front_input = st.date_input(
                "DTE FRONT (Venta)",
                value=expiration_date,
                min_value=date.today() + timedelta(days=1),
                max_value=date.today() + timedelta(days=365),
                key='dte_front_global',
                help="Fecha de expiración de las opciones vendidas"
            )
        
        with col2:
            default_back = dte_front_input + timedelta(days=7)
            
            dte_back_input = st.date_input(
                "DTE BACK (Compra)",
                value=default_back,
                min_value=dte_front_input + timedelta(days=1),
                max_value=date.today() + timedelta(days=365),
                key='dte_back_global',
                help="Fecha de expiración de las opciones compradas"
            )
        
        days_diff = (dte_back_input - dte_front_input).days
        st.success(f"📅 Diferencia entre fechas: **{days_diff} días**")
        
        st.markdown("---")
        
        # Configuración de strikes individuales
        st.markdown("### 🎯 Configuración de Strikes")
        
        st.info(f"""
        💡 **Referencia (Precio Actual: ${current_price:.2f})**
        - Expected Move (1σ): ±${expected_move_1std:.2f}
        - Rango: ${lower_range_1std:.2f} - ${upper_range_1std:.2f}
        - Incremento de Strike: {strike_increment:.0f}
        """)
        
        # Crear 3 columnas para los 3 strikes
        col_down, col_atm, col_up = st.columns(3)
        
        # STRIKE DOWN
        with col_down:
            st.markdown("#### 📉 STRIKE DOWN")
            
            with st.expander("⚙️ Configurar Strike Down", expanded=True):
                st.markdown(f"**Precio Actual:** ${current_price:.2f}")
                st.markdown(f"**Sugerido:** ${st.session_state.strike_down:.0f}")
                
                strike_down = st.number_input(
                    "Strike",
                    min_value=0.0,
                    value=float(st.session_state.strike_down),
                    step=strike_increment,
                    format="%.0f",
                    key='strike_down_input'
                )
                
                type_down = st.selectbox(
                    "Tipo de Opción",
                    ['PUT', 'CALL'],
                    index=0,
                    key='type_down_input'
                )
                
                send_down = st.checkbox(
                    "✅ Enviar este calendar",
                    value=st.session_state.send_down,
                    key='send_down_input'
                )
                
                # Actualizar session state
                st.session_state.strike_down = strike_down
                st.session_state.type_down = type_down
                st.session_state.send_down = send_down
                
                if send_down:
                    st.success(f"✅ ${strike_down:.0f} {type_down}")
                else:
                    st.warning("⏸️ No se enviará")
        
        # STRIKE ATM
        with col_atm:
            st.markdown("#### 🎯 STRIKE ATM")
            
            with st.expander("⚙️ Configurar Strike ATM", expanded=True):
                st.markdown(f"**Precio Actual:** ${current_price:.2f}")
                st.markdown(f"**Sugerido:** ${st.session_state.strike_atm:.0f}")
                
                strike_atm = st.number_input(
                    "Strike",
                    min_value=0.0,
                    value=float(st.session_state.strike_atm),
                    step=strike_increment,
                    format="%.0f",
                    key='strike_atm_input'
                )
                
                type_atm = st.selectbox(
                    "Tipo de Opción",
                    ['PUT', 'CALL'],
                    index=0,
                    key='type_atm_input'
                )
                
                send_atm = st.checkbox(
                    "✅ Enviar este calendar",
                    value=st.session_state.send_atm,
                    key='send_atm_input'
                )
                
                # Actualizar session state
                st.session_state.strike_atm = strike_atm
                st.session_state.type_atm = type_atm
                st.session_state.send_atm = send_atm
                
                if send_atm:
                    st.success(f"✅ ${strike_atm:.0f} {type_atm}")
                else:
                    st.warning("⏸️ No se enviará")
        
        # STRIKE UP
        with col_up:
            st.markdown("#### 📈 STRIKE UP")
            
            with st.expander("⚙️ Configurar Strike UP", expanded=True):
                st.markdown(f"**Precio Actual:** ${current_price:.2f}")
                st.markdown(f"**Sugerido:** ${st.session_state.strike_up:.0f}")
                
                strike_up = st.number_input(
                    "Strike",
                    min_value=0.0,
                    value=float(st.session_state.strike_up),
                    step=strike_increment,
                    format="%.0f",
                    key='strike_up_input'
                )
                
                type_up = st.selectbox(
                    "Tipo de Opción",
                    ['PUT', 'CALL'],
                    index=1,
                    key='type_up_input'
                )
                
                send_up = st.checkbox(
                    "✅ Enviar este calendar",
                    value=st.session_state.send_up,
                    key='send_up_input'
                )
                
                # Actualizar session state
                st.session_state.strike_up = strike_up
                st.session_state.type_up = type_up
                st.session_state.send_up = send_up
                
                if send_up:
                    st.success(f"✅ ${strike_up:.0f} {type_up}")
                else:
                    st.warning("⏸️ No se enviará")
        
        st.markdown("---")
        
        # Parámetros adicionales
        st.markdown("### 💰 Parámetros de Orden")
        
        col1, col2 = st.columns(2)
        
        with col1:
            quantity_global = st.number_input(
                "Cantidad de Contratos por Calendar",
                min_value=1,
                value=1,
                step=1,
                key='quantity_global',
                help="Número de contratos por cada calendar spread"
            )
        
        with col2:
            limit_price_global = st.number_input(
                "Precio Límite por Calendar",
                min_value=0.0,
                value=1.0,
                step=0.05,
                format="%.2f",
                key='limit_price_global',
                help="Precio límite para cada calendar spread"
            )
        
        st.markdown("---")
        
        # Botón para revisar órdenes
        if st.button("📋 REVISAR ÓRDENES", type="primary", use_container_width=True):
            st.session_state.orders_reviewed = True
        
        # ==============================================================================
        # SECCIÓN 6: REVISIÓN DE ÓRDENES
        # ==============================================================================
        if st.session_state.orders_reviewed:
            
            st.markdown("---")
            st.header("6. Revisión de Órdenes")
            
            # Recopilar calendars seleccionados
            calendars_to_send = []
            
            if st.session_state.send_down:
                calendars_to_send.append({
                    'Nombre': 'DOWN',
                    'Strike': st.session_state.strike_down,
                    'Tipo': st.session_state.type_down
                })
            
            if st.session_state.send_atm:
                calendars_to_send.append({
                    'Nombre': 'ATM',
                    'Strike': st.session_state.strike_atm,
                    'Tipo': st.session_state.type_atm
                })
            
            if st.session_state.send_up:
                calendars_to_send.append({
                    'Nombre': 'UP',
                    'Strike': st.session_state.strike_up,
                    'Tipo': st.session_state.type_up
                })
            
            if len(calendars_to_send) == 0:
                st.warning("⚠️ No hay calendars seleccionados para enviar. Marca al menos uno en la configuración.")
            else:
                st.success(f"📋 **Total de Calendars a Enviar: {len(calendars_to_send)}**")
                
                # Mostrar tabla de resumen
                preview_data = []
                for cal in calendars_to_send:
                    preview_data.append({
                        'Nombre': cal['Nombre'],
                        'Strike': f"${cal['Strike']:.0f}",
                        'Tipo': cal['Tipo'],
                        'FRONT (SELL)': dte_front_input.strftime('%Y-%m-%d'),
                        'BACK (BUY)': dte_back_input.strftime('%Y-%m-%d'),
                        'Cantidad': quantity_global,
                        'Precio Límite': f"${limit_price_global:.2f}"
                    })
                
                df_preview = pd.DataFrame(preview_data)
                st.dataframe(df_preview, hide_index=True, use_container_width=True)
                
                # Mostrar detalles expandibles de cada calendar
                st.markdown("### 📄 Detalles de Cada Calendar")
                
                for cal in calendars_to_send:
                    with st.expander(f"🔍 {cal['Nombre']} - ${cal['Strike']:.0f} {cal['Tipo']}"):
                        st.markdown(f"""
                        **Ticker:** {selected_ticker}
                        **Strike:** ${cal['Strike']:.0f}
                        **Tipo de Opción:** {cal['Tipo']}
                        
                        **Pierna 1 - SELL (Front):**
                        - Fecha: {dte_front_input.strftime('%Y-%m-%d')}
                        - Cantidad: {quantity_global}
                        
                        **Pierna 2 - BUY (Back):**
                        - Fecha: {dte_back_input.strftime('%Y-%m-%d')}
                        - Cantidad: {quantity_global}
                        
                        **Precio Límite Total:** ${limit_price_global:.2f}
                        """)
                
                st.markdown("---")
                
                # Configuración IBKR
                st.markdown("### 🔌 Configuración de Conexión IBKR")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    ibkr_host = st.text_input("Host IBKR", value="127.0.0.1", key='ibkr_host_global')
                with col2:
                    ibkr_port = st.number_input("Puerto IBKR", min_value=1, max_value=65535, value=5000, step=1, key='ibkr_port_global')
                with col3:
                    ibkr_client_id = st.number_input("Client ID", min_value=1, value=1, step=1, key='ibkr_client_id_global')
                
                st.markdown("---")
                
                st.warning("""
                ⚠️ **IMPORTANTE - Antes de Enviar:**
                - Verifica que TWS o IB Gateway esté ejecutándose
                - Confirma que la configuración de conexión sea correcta
                - Revisa todos los detalles de las órdenes arriba
                - Una vez enviadas, las órdenes se procesarán en IBKR
                """)
                
                # Botón final para enviar a IBKR
                if st.button("🚀 ENVIAR A IBKR", type="primary", use_container_width=True, key='send_to_ibkr_final'):
                    
                    try:
                        from utils.utils_ibkr import send_strategy_order_ibkr
                    except ImportError:
                        st.error("❌ Error: No se pudo importar send_strategy_order_ibkr")
                        st.stop()
                    
                    if limit_price_global <= 0:
                        st.error("❌ El precio límite debe ser mayor a 0")
                        st.stop()
                    
                    # Enviar cada calendar
                    results = []
                    
                    for cal in calendars_to_send:
                        with st.spinner(f"📡 Enviando calendar {cal['Nombre']} (${cal['Strike']:.0f} {cal['Tipo']})..."):
                            
                            # Crear orden
                            df_order = create_calendar_pair_order(
                                strike=cal['Strike'],
                                option_type=cal['Tipo'],
                                front_date=dte_front_input.strftime("%Y-%m-%d"),
                                back_date=dte_back_input.strftime("%Y-%m-%d"),
                                quantity=quantity_global,
                                ticker=selected_ticker
                            )
                            
                            # Enviar a IBKR
                            result = send_strategy_order_ibkr(
                                df_strategy=df_order,
                                limit_price=limit_price_global,
                                host=ibkr_host,
                                port=int(ibkr_port),
                                client_id=int(ibkr_client_id),
                                quantity=quantity_global,
                                tif='DAY',
                                action='BUY',
                                timeout=10
                            )
                            
                            # Guardar resultado
                            results.append({
                                'nombre': cal['Nombre'],
                                'strike': cal['Strike'],
                                'tipo': cal['Tipo'],
                                'success': result['success'],
                                'message': result['message'],
                                'order_id': result.get('order_id'),
                                'contracts': result.get('contracts')
                            })
                    
                    # Mostrar resultados
                    st.markdown("---")
                    st.markdown("### 📊 Resultados del Envío")
                    
                    success_count = sum(1 for r in results if r['success'])
                    error_count = len(results) - success_count
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Enviados", len(results))
                    with col2:
                        st.metric("Exitosos", success_count)
                    with col3:
                        st.metric("Fallidos", error_count)
                    
                    st.markdown("---")
                    
                    # Detalles de cada envío
                    for r in results:
                        if r['success']:
                            st.success(f"""
                            ✅ **{r['nombre']}** (${r['strike']:.0f} {r['tipo']})
                            - {r['message']}
                            {f"- Order ID: {r['order_id']}" if r['order_id'] else ""}
                            """)
                            
                            if r.get('contracts'):
                                with st.expander(f"Ver contratos - {r['nombre']}"):
                                    for i, c in enumerate(r['contracts'], 1):
                                        st.write(f"{i}. {c.symbol} {c.lastTradeDateOrContractMonth} {c.strike} {c.right}")
                        else:
                            st.error(f"""
                            ❌ **{r['nombre']}** (${r['strike']:.0f} {r['tipo']})
                            - {r['message']}
                            """)
                    
                    st.markdown("---")
                    
                    if success_count == len(results):
                        st.balloons()
                        st.success(f"🎉 ¡Todos los calendars ({len(results)}) se enviaron exitosamente!")
                    elif success_count > 0:
                        st.warning(f"⚠️ Se enviaron {success_count} de {len(results)} calendars.")
                    else:
                        st.error("❌ No se pudo enviar ningún calendar. Verifica la conexión con IBKR.")


# ==============================================================================
# PUNTO DE ENTRADA PROTEGIDO
# ==============================================================================

if __name__ == "__main__":
    if check_password():
        main_tp_calculos()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral para acceder.")
