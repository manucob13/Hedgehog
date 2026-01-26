import streamlit as st
import pandas as pd
from datetime import date, timedelta
from utils.utils import check_password
from utils.utils_schwab import connect_to_schwab, get_current_price_schwab, get_atm_strike_schwab

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="🔧 Ajustes Calendar", layout="wide")

def create_calendar_order(strike, option_type, front_date, back_date, quantity, ticker):
    """Crea las órdenes para un calendar spread individual."""
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
            'Label': 'Adjustment Front'
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
            'Label': 'Adjustment Back'
        }
    ]
    
    return pd.DataFrame(orders)

def initialize_session_state():
    """Inicializa las variables de session_state."""
    if 'df_strategy_adj' not in st.session_state:
        st.session_state.df_strategy_adj = None
    if 'order_preview_adj' not in st.session_state:
        st.session_state.order_preview_adj = False
    if 'schwab_client' not in st.session_state:
        st.session_state.schwab_client = None

def main_adjustments():
    initialize_session_state()
    
    # --- TÍTULO ---
    st.markdown("<h1><span style='font-size: 1.5em;'>🔧</span> Ajustes - Calendar Individual</h1>", unsafe_allow_html=True)
    st.markdown("""
    Esta herramienta te permite generar un **Calendar Spread individual** basado en el precio actual del mercado.
    Solo necesitas calcular el strike ATM y enviar la orden directamente a IBKR.
    """)
    st.markdown("---")
    
    # ==============================================================================
    # SECCIÓN 1: CONFIGURACIÓN BÁSICA
    # ==============================================================================
    st.header("1. Configuración Básica")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_ticker = st.selectbox(
            "Selecciona el Ticker",
            ['QQQ', 'SPX'],
            index=0,
            key='ticker_adj'
        )
        st.info(f"📊 Ticker: **{selected_ticker}**")
    
    with col2:
        # Calcular incremento según ticker
        strike_increment = 1.0 if selected_ticker == 'QQQ' else 5.0
        round_to = 1 if selected_ticker == 'QQQ' else 5
        
        st.info(f"📏 Incremento de Strike: **{strike_increment:.0f}**")
    
    st.markdown("---")
    
    # ==============================================================================
    # BOTÓN PARA OBTENER PRECIO ACTUAL
    # ==============================================================================
    
    if st.button("📡 Obtener Precio Actual", type="primary", use_container_width=True):
        with st.spinner(f"Conectando con Schwab y obteniendo precio de {selected_ticker}..."):
            
            # Conectar con Schwab
            schwab_client = connect_to_schwab()
            
            if schwab_client is None:
                st.error("❌ No se pudo conectar con Schwab. Verifica la configuración.")
                st.stop()
            
            st.session_state.schwab_client = schwab_client
            st.success("✅ Conectado con Schwab exitosamente")
            
            # Obtener precio actual
            current_price = get_current_price_schwab(schwab_client, selected_ticker)
            
            if current_price is None:
                st.error(f"❌ No se pudo obtener el precio actual de {selected_ticker}")
                st.stop()
            
            st.session_state.current_price = current_price
            st.success(f"✅ Precio actual de {selected_ticker}: **${current_price:.2f}**")
    
    # ==============================================================================
    # SECCIÓN 2: CÁLCULO DE STRIKE ATM
    # ==============================================================================
    
    if 'current_price' in st.session_state and st.session_state.current_price:
        
        current_price = st.session_state.current_price
        schwab_client = st.session_state.schwab_client
        
        st.markdown("---")
        st.header("2. Cálculo de Strike ATM")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Precio Actual", f"${current_price:.2f}")
        
        with col2:
            # Calcular strike ATM redondeado
            strike_atm_calc = round(current_price / round_to) * round_to
            st.metric("Strike ATM Calculado", f"${strike_atm_calc:.0f}")
        
        st.markdown("---")
        
        # ==============================================================================
        # SECCIÓN 3: CONFIGURACIÓN DEL CALENDAR
        # ==============================================================================
        
        st.header("3. Configuración del Calendar Spread")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Strike y Tipo")
            
            strike_adjustment = st.number_input(
                "Strike para el Calendar",
                min_value=0.0,
                value=float(strike_atm_calc),
                step=strike_increment,
                key='strike_adjustment'
            )
            
            option_type_adjustment = st.selectbox(
                "Tipo de Opción",
                ["CALL", "PUT"],
                index=0,
                key='option_type_adjustment'
            )
            
            st.info(f"""
            📊 **Configuración del Strike:**
            - Strike seleccionado: **${strike_adjustment:.0f}**
            - Tipo: **{option_type_adjustment}**
            - Incremento: **{strike_increment:.0f}**
            """)
        
        with col2:
            st.markdown("#### 📅 Fechas de Expiración")
            
            min_date = date.today() + timedelta(days=1)
            
            dte_front_adj = st.date_input(
                "DTE FRONT (Venta)",
                value=min_date,
                min_value=min_date,
                max_value=date.today() + timedelta(days=365),
                key='dte_front_adj'
            )
            
            min_back_date = dte_front_adj + timedelta(days=1)
            default_back = dte_front_adj + timedelta(days=7)
            
            dte_back_adj = st.date_input(
                "DTE BACK (Compra)",
                value=default_back,
                min_value=min_back_date,
                max_value=date.today() + timedelta(days=365),
                key='dte_back_adj'
            )
            
            days_diff = (dte_back_adj - dte_front_adj).days
            
            st.success(f"📅 Diferencia: **{days_diff} días**")
            
            st.info(f"""
            📆 **Fechas configuradas:**
            - FRONT: **{dte_front_adj.strftime('%Y-%m-%d')}**
            - BACK: **{dte_back_adj.strftime('%Y-%m-%d')}**
            """)
        
        st.markdown("---")
        
        # ==============================================================================
        # SECCIÓN 4: CONFIGURACIÓN DE ORDEN E IBKR
        # ==============================================================================
        
        st.header("4. Configuración de Orden e IBKR")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💰 Parámetros de Orden")
            
            quantity_adj = st.number_input(
                "Cantidad de Contratos",
                min_value=1,
                value=1,
                step=1,
                key='quantity_adj'
            )
            
            limit_price_adj = st.number_input(
                "Precio Límite (Total)",
                min_value=0.0,
                value=1.0,
                step=0.05,
                format="%.2f",
                key='limit_price_adj'
            )
            
            st.info(f"""
            💡 **Configuración de Precio:**
            - Precio Límite Total: **${limit_price_adj:.2f}**
            - Por contrato: **${limit_price_adj / quantity_adj if quantity_adj > 0 else 0:.2f}**
            """)
        
        with col2:
            st.markdown("#### 📌 Configuración IBKR")
            
            ibkr_host = st.text_input(
                "Host IBKR",
                value="127.0.0.1",
                key='ibkr_host_adj'
            )
            
            ibkr_port = st.number_input(
                "Puerto IBKR",
                min_value=1,
                max_value=65535,
                value=5000,
                step=1,
                key='ibkr_port_adj'
            )
            
            ibkr_client_id = st.number_input(
                "Client ID",
                min_value=1,
                value=1,
                step=1,
                key='ibkr_client_id_adj'
            )
            
            st.info(f"""
            🔌 **Conexión IBKR:**
            - Host: **{ibkr_host}**
            - Puerto: **{ibkr_port}**
            - Client ID: **{ibkr_client_id}**
            """)
        
        st.markdown("---")
        
        # ==============================================================================
        # GENERAR VISTA PREVIA
        # ==============================================================================
        
        if st.button("📝 Generar Vista Previa", type="primary", use_container_width=True):
            
            df_strategy_adj = create_calendar_order(
                strike_adjustment,
                option_type_adjustment,
                dte_front_adj.strftime("%Y-%m-%d"),
                dte_back_adj.strftime("%Y-%m-%d"),
                quantity_adj,
                selected_ticker
            )
            
            st.session_state.df_strategy_adj = df_strategy_adj
            st.session_state.order_preview_adj = True
            
            st.success("✅ Vista previa generada exitosamente!")
        
        # ==============================================================================
        # MOSTRAR VISTA PREVIA Y ENVIAR
        # ==============================================================================
        
        if st.session_state.order_preview_adj and st.session_state.df_strategy_adj is not None:
            
            df_strategy_adj = st.session_state.df_strategy_adj
            
            st.markdown("---")
            st.header("5. Vista Previa y Envío")
            
            st.markdown("### 📋 Vista Previa de la Orden")
            st.dataframe(df_strategy_adj, hide_index=True, use_container_width=True)
            
            st.markdown("---")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Ticker", selected_ticker)
            with col2:
                st.metric("Strike", f"${strike_adjustment:.0f}")
            with col3:
                st.metric("Tipo", option_type_adjustment)
            with col4:
                st.metric("Cantidad", quantity_adj)
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📅 Fechas**")
                st.write(f"FRONT: {dte_front_adj.strftime('%Y-%m-%d')}")
                st.write(f"BACK: {dte_back_adj.strftime('%Y-%m-%d')}")
                st.write(f"Spread: {days_diff} días")
            
            with col2:
                st.markdown("**💰 Precio**")
                st.write(f"Límite Total: ${limit_price_adj:.2f}")
                st.write(f"Por contrato: ${limit_price_adj / quantity_adj:.2f}")
            
            with col3:
                st.markdown("**🔌 IBKR**")
                st.write(f"Host: {ibkr_host}")
                st.write(f"Puerto: {ibkr_port}")
                st.write(f"Client ID: {ibkr_client_id}")
            
            st.markdown("---")
            
            st.warning("⚠️ **IMPORTANTE:** Asegúrate de que TWS/Gateway esté ejecutándose y la configuración sea correcta.")
            
            if st.button("🚀 ENVIAR ORDEN A IBKR", type="primary", use_container_width=True):
                
                try:
                    from utils.utils_ibkr import send_strategy_order_ibkr
                except ImportError:
                    st.error("❌ Error: No se pudo importar send_strategy_order_ibkr")
                    st.stop()
                
                if limit_price_adj <= 0:
                    st.error("❌ El precio límite debe ser mayor a 0")
                    st.stop()
                
                with st.spinner("📡 Enviando orden a IBKR..."):
                    result = send_strategy_order_ibkr(
                        df_strategy=df_strategy_adj,
                        limit_price=limit_price_adj,
                        host=ibkr_host,
                        port=int(ibkr_port),
                        client_id=int(ibkr_client_id),
                        quantity=quantity_adj,
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
            
            st.info("""
            **💡 Sobre el Calendar Spread de Ajuste:**
            
            - **Propósito**: Añadir un calendar spread individual para ajustar tu posición
            - **Tipo de Orden**: LIMIT
            - **Acción**: BUY (compras el back, vendes el front)
            - **TIF**: DAY
            
            **Cuándo usar ajustes:**
            - El precio del subyacente se ha movido significativamente
            - Necesitas rebalancear tu delta
            - Quieres añadir exposición en un nuevo strike
            - Deseas ajustar la estructura de tu posición
            """)

# ==============================================================================
# PUNTO DE ENTRADA
# ==============================================================================

if __name__ == "__main__":
    if check_password():
        main_adjustments()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral para acceder.")
