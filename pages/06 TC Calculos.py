import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import plotly.graph_objects as go
from utils.utils import check_password
from utils.utils_cboe import get_option_chain_cboe
from utils.utils_schwab import connect_to_schwab, get_current_price_schwab, obtener_datos_opcion, get_atm_strike_schwab
import io

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="TC Cálculos - Expected Move", layout="wide")

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def calculate_expected_move_cboe(df_options, expiration_date, current_price, std_multiplier=1.0):
    """Calcula el Expected Move basado en el straddle ATM usando datos de CBOE."""
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


def calculate_expected_move_schwab(client, ticker, expiration_date, current_price, std_multiplier=1.0):
    """Calcula el Expected Move basado en el straddle ATM usando datos de Schwab."""
    try:
        if client is None:
            return None, None, "Cliente de Schwab no disponible"
        
        # Obtener el strike ATM real de la cadena de opciones
        atm_strike = get_atm_strike_schwab(client, ticker, current_price, expiration_date)
        
        if atm_strike is None:
            return None, None, "No se pudo obtener el strike ATM de Schwab"
        
        # Obtener datos del CALL ATM
        call_mid, call_delta, call_theta = obtener_datos_opcion(
            client, ticker, atm_strike, 'CALL', expiration_date
        )
        
        # Obtener datos del PUT ATM
        put_mid, put_delta, put_theta = obtener_datos_opcion(
            client, ticker, atm_strike, 'PUT', expiration_date
        )
        
        if call_mid is None or put_mid is None:
            return None, None, "No se encontraron datos de opciones en Schwab"
        
        # Calcular straddle price
        straddle_price = call_mid + put_mid
        price_type = "Mid Price"
        
        # Expected Move = Straddle Price * 1.25 * std_multiplier
        expected_move = straddle_price * 1.25 * std_multiplier
        
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
        return None, None, f"Error calculando Expected Move con Schwab: {e}"


@st.cache_data(ttl=300)
def get_historical_prices_yf(ticker, days=7):
    """Obtiene precios históricos del ticker usando Yahoo Finance."""
    try:
        import yfinance as yf
        
        # Ajustar símbolo para Yahoo Finance
        yf_ticker = '^SPX' if ticker == 'SPX' else ticker
        
        stock = yf.Ticker(yf_ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Obtener datos históricos
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            return None
        
        return df
        
    except Exception as e:
        st.error(f"Error obteniendo datos históricos de Yahoo Finance: {e}")
        return None


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
        'schwab_client': None,
        'cboe_data': None,
        'schwab_data': None,
        'selected_data_source': 'CBOE'
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
                st.error("❌ No se pudo obtener el precio actual desde Schwab.")
                st.stop()
            
            st.success(f"✅ Precio actual de {selected_ticker} (Schwab): **${current_price:.2f}**")
            
            # Obtener cadena de opciones desde CBOE
            df_options = get_option_chain_cboe(selected_ticker)
            
            if df_options is None or df_options.empty:
                st.error("❌ No se pudieron obtener los datos de opciones desde CBOE.")
                st.stop()
            
            # Calcular Expected Move con CBOE
            expected_move_cboe, details_cboe, error_cboe = calculate_expected_move_cboe(
                df_options, 
                expiration_date, 
                current_price,
                std_multiplier
            )
            
            if error_cboe:
                st.warning(f"⚠️ CBOE: {error_cboe}")
                details_cboe = None
            else:
                st.success(f"✅ Expected Move (CBOE) calculado: **${expected_move_cboe:.2f}**")
            
            # Calcular Expected Move con Schwab
            expected_move_schwab, details_schwab, error_schwab = calculate_expected_move_schwab(
                schwab_client,
                selected_ticker,
                expiration_date,
                current_price,
                std_multiplier
            )
            
            if error_schwab:
                st.warning(f"⚠️ Schwab: {error_schwab}")
                details_schwab = None
            else:
                st.success(f"✅ Expected Move (Schwab) calculado: **${expected_move_schwab:.2f}**")
            
            # Guardar ambos cálculos en session_state
            st.session_state.cboe_data = {
                'expected_move': expected_move_cboe,
                'details': details_cboe,
                'error': error_cboe
            }
            
            st.session_state.schwab_data = {
                'expected_move': expected_move_schwab,
                'details': details_schwab,
                'error': error_schwab
            }
            
            st.session_state.current_price = current_price
            st.session_state.selected_ticker = selected_ticker
            st.session_state.expiration_date = expiration_date
            st.session_state.std_multiplier = std_multiplier
            st.session_state.calculation_done = True
    
    # ==============================================================================
    # MOSTRAR COMPARACIÓN Y SELECTOR
    # ==============================================================================
    
    if st.session_state.calculation_done:
        
        current_price = st.session_state.current_price
        selected_ticker = st.session_state.selected_ticker
        expiration_date = st.session_state.expiration_date
        std_multiplier = st.session_state.std_multiplier
        
        cboe_data = st.session_state.cboe_data
        schwab_data = st.session_state.schwab_data
        
        st.markdown("---")
        st.header("2. Comparación de Fuentes de Datos")
        
        # Crear tabla comparativa
        comparison_rows = []
        
        # Precio Actual
        comparison_rows.append({
            'Métrica': 'Precio Actual',
            'CBOE': f"${current_price:.2f}",
            'Schwab': f"${current_price:.2f}"
        })
        
        # Strike ATM
        if cboe_data['details'] and schwab_data['details']:
            comparison_rows.append({
                'Métrica': 'Strike ATM',
                'CBOE': f"${cboe_data['details']['atm_strike']:.2f}",
                'Schwab': f"${schwab_data['details']['atm_strike']:.2f}"
            })
        
        # Call Mid
        if cboe_data['details'] and schwab_data['details']:
            comparison_rows.append({
                'Métrica': 'Call Mid',
                'CBOE': f"${cboe_data['details']['call_mid']:.2f}",
                'Schwab': f"${schwab_data['details']['call_mid']:.2f}"
            })
        
        # Put Mid
        if cboe_data['details'] and schwab_data['details']:
            comparison_rows.append({
                'Métrica': 'Put Mid',
                'CBOE': f"${cboe_data['details']['put_mid']:.2f}",
                'Schwab': f"${schwab_data['details']['put_mid']:.2f}"
            })
        
        # Straddle Price
        if cboe_data['details'] and schwab_data['details']:
            comparison_rows.append({
                'Métrica': 'Straddle Price',
                'CBOE': f"${cboe_data['details']['straddle_price']:.2f}",
                'Schwab': f"${schwab_data['details']['straddle_price']:.2f}"
            })
        
        # Expected Move
        if cboe_data['expected_move'] and schwab_data['expected_move']:
            comparison_rows.append({
                'Métrica': f'Expected Move ({std_multiplier}σ)',
                'CBOE': f"${cboe_data['expected_move']:.2f}",
                'Schwab': f"${schwab_data['expected_move']:.2f}"
            })
        
        # Tipo de Precio
        if cboe_data['details'] and schwab_data['details']:
            comparison_rows.append({
                'Métrica': 'Tipo de Precio',
                'CBOE': cboe_data['details']['price_type'],
                'Schwab': schwab_data['details']['price_type']
            })
        
        df_comparison = pd.DataFrame(comparison_rows)
        
        st.markdown("### 📊 Tabla Comparativa")
        st.dataframe(df_comparison, hide_index=True, use_container_width=True)
        
        st.markdown("---")
        
        # Selector de fuente de datos
        st.markdown("### 🎛️ Selecciona la Fuente de Datos")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("📊 Usar CBOE", use_container_width=True, type="primary" if st.session_state.selected_data_source == 'CBOE' else "secondary"):
                st.session_state.selected_data_source = 'CBOE'
                st.rerun()
        
        with col2:
            if st.button("💼 Usar Schwab", use_container_width=True, type="primary" if st.session_state.selected_data_source == 'Schwab' else "secondary"):
                st.session_state.selected_data_source = 'Schwab'
                st.rerun()
        
        with col3:
            st.info(f"**Fuente Activa:** {st.session_state.selected_data_source}")
        
        # Seleccionar datos según la fuente elegida
        if st.session_state.selected_data_source == 'CBOE':
            if cboe_data['details'] is None:
                st.error("❌ No hay datos válidos de CBOE disponibles")
                st.stop()
            expected_move = cboe_data['expected_move']
            details = cboe_data['details']
        else:
            if schwab_data['details'] is None:
                st.error("❌ No hay datos válidos de Schwab disponibles")
                st.stop()
            expected_move = schwab_data['expected_move']
            details = schwab_data['details']
        
        # Guardar en session_state
        st.session_state.expected_move = expected_move
        st.session_state.details = details
        
        st.markdown("---")
        
        # ==============================================================================
        # SECCIÓN 3: RESULTADOS
        # ==============================================================================
        st.header("3. Resultados del Expected Move")
        
        st.info(f"📌 **Mostrando resultados de:** {st.session_state.selected_data_source}")
        
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
            # Agregar datos adicionales si están disponibles (CBOE)
            if 'call_bid' in details:
                call_details['Bid'] = f"${details['call_bid']:.2f}"
                call_details['Ask'] = f"${details['call_ask']:.2f}"
                call_details['Last'] = f"${details['call_last']:.2f}"
            # Agregar greeks si están disponibles (Schwab)
            if 'call_delta' in details and details['call_delta'] is not None:
                call_details['Delta'] = f"{details['call_delta']:.4f}"
            if 'call_theta' in details and details['call_theta'] is not None:
                call_details['Theta'] = f"{details['call_theta']:.4f}"
            
            df_call = pd.DataFrame(list(call_details.items()), columns=['Métrica', 'Valor'])
            st.dataframe(df_call, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("**📉 PUT ATM**")
            put_details = {
                'Strike': f"${details['atm_strike']:.2f}",
                'Mid': f"${details['put_mid']:.2f}"
            }
            # Agregar datos adicionales si están disponibles (CBOE)
            if 'put_bid' in details:
                put_details['Bid'] = f"${details['put_bid']:.2f}"
                put_details['Ask'] = f"${details['put_ask']:.2f}"
                put_details['Last'] = f"${details['put_last']:.2f}"
            # Agregar greeks si están disponibles (Schwab)
            if 'put_delta' in details and details['put_delta'] is not None:
                put_details['Delta'] = f"{details['put_delta']:.4f}"
            if 'put_theta' in details and details['put_theta'] is not None:
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
                'Fuente de Datos',
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
                st.session_state.selected_data_source,
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
        # SECCIÓN 4: GRÁFICO - VELAS JAPONESAS (YAHOO FINANCE)
        # ==============================================================================
        st.header("4. Visualización del Expected Move")
        
        # Usar Yahoo Finance directamente
        df_hist = get_historical_prices_yf(selected_ticker, days=7)
        
        if df_hist is not None and not df_hist.empty:
            
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
                title=f"Expected Move - {selected_ticker} ({std_multiplier}σ) - Fuente: {st.session_state.selected_data_source}",
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
            
            st.info("📊 **Datos históricos obtenidos de Yahoo Finance**")
            
        else:
            st.warning("⚠️ No se pudieron obtener datos históricos de Yahoo Finance para el gráfico.")
        
        st.markdown("---")
        
        # ==============================================================================
        # SECCIÓN 5: COMPARACIÓN DE DESVIACIONES ESTÁNDAR
        # ==============================================================================
        st.header("5. Comparación de Desviaciones Estándar")
        
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
        # SECCIÓN 6: INFORMACIÓN ADICIONAL
        # ==============================================================================
        st.header("6. Información Adicional")
        
        prob_text = {1.0: "68%", 1.5: "87%", 2.0: "95%"}.get(std_multiplier, "N/A")
        
        st.info(f"""
        📌 **Interpretación del Expected Move:**
        
        - El **Expected Move** representa el rango de precio esperado ({std_multiplier} desviación estándar) 
          que el mercado anticipa para la fecha de expiración.
        
        - Este cálculo se basa en el precio del **straddle ATM** (comprar un call y un put 
          al mismo strike más cercano al precio actual).
        
        - Con **{std_multiplier}σ**, aproximadamente el **{prob_text}** de las veces, el precio debería 
          permanecer dentro de este rango.
        
        - **Fuente de Datos:** {st.session_state.selected_data_source}
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
        - **Precios de Opciones:** CBOE (Chicago Board Options Exchange) y Schwab
        - **Precios del Activo:** Schwab
        - **Datos Históricos:** Yahoo Finance
        
        ### 🧮 Fórmula del Expected Move
        ```
        Expected Move = Straddle Price × 1.25 × σ
        ```
        
        Donde:
        - **Straddle Price** = Call Mid + Put Mid (ATM)
        - **1.25** = Factor de ajuste para 1 desviación estándar completa
        - **σ** = Multiplicador de desviaciones estándar (1.0, 1.5, o 2.0)
        
        ### 💡 Diferencias entre Fuentes
        - **CBOE:** Proporciona Bid, Ask, Mid y Last Price con mayor granularidad
        - **Schwab:** Proporciona Mid Price y Greeks (Delta, Theta) en tiempo real, strike ATM basado en cadena real
        
        ### 📊 Niveles de Confianza
        - **1σ** ≈ 68% de probabilidad (rango más conservador)
        - **1.5σ** ≈ 87% de probabilidad (rango intermedio)
        - **2σ** ≈ 95% de probabilidad (rango más amplio)
        """)
        
        st.markdown("---")
        
        # ==============================================================================
        # SECCIÓN 7: GENERADOR DE ESTRUCTURA TRIPLE CALENDAR - ENVÍO A IBKR
        # ==============================================================================
        st.header("7. Generador de Estructura - Triple Calendar (Envío a IBKR)")
        
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
            💡 **Rangos de Referencia (1σ) - Fuente: {st.session_state.selected_data_source}:**
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
            
            # Guardar strikes y DTEs en session_state para el punto 8
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
        # SECCIÓN 8: AJUSTES
        # ==============================================================================
        st.header("8. Ajustes - Generador de Calendar Individual")
        
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
        💡 **Cálculos de Ajuste (1σ) - Fuente: {st.session_state.selected_data_source}:**
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
            
            st.markdown("**Strikes de Referencia (Punto 7)**")
            
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
            
            default_front = st.session_state.dte_front_p6 if st.session_state.dte_front_p6 else expiration_date
            default_back = st.session_state.dte_back_p6 if st.session_state.dte_back_p6 else (expiration_date + timedelta(days=7))
            
            dte_front_adj = st.date_input("DTE FRONT (Venta)", value=default_front, min_value=date.today() + timedelta(days=1), key='dte_front_adj')
            dte_back_adj = st.date_input("DTE BACK (Compra)", value=default_back, min_value=dte_front_adj + timedelta(days=1), key='dte_back_adj')
            
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
