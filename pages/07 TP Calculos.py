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
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'data' in data and 'options' in data['data']:
            options_data = data['data']['options']
            df = pd.DataFrame(options_data)

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

        df_exp['distance'] = abs(df_exp['strike'] - current_price)
        atm_strike = df_exp.loc[df_exp['distance'].idxmin(), 'strike']

        call_atm = df_exp[(df_exp['strike'] == atm_strike) & (df_exp['opt_type'] == 'C')]
        put_atm = df_exp[(df_exp['strike'] == atm_strike) & (df_exp['opt_type'] == 'P')]

        if call_atm.empty or put_atm.empty:
            return None, None, "No se encontraron opciones ATM"

        call_bid = call_atm['bid'].iloc[0]
        call_ask = call_atm['ask'].iloc[0]
        call_mid = (call_bid + call_ask) / 2
        call_last = call_atm['last_trade_price'].iloc[0]

        put_bid = put_atm['bid'].iloc[0]
        put_ask = put_atm['ask'].iloc[0]
        put_mid = (put_bid + put_ask) / 2
        put_last = put_atm['last_trade_price'].iloc[0]

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

def create_ibkr_basket_csv(ticker, strikes_data, expiration_date, action="BUY"):
    """
    Crea un DataFrame con el formato CSV requerido por IBKR BasketTrader.

    Formato IBKR:
    Symbol, SecType, Expiry, Strike, Right, Exchange, Currency, Quantity, Side, LimitPrice, OrderType

    Args:
        ticker: Símbolo del subyacente
        strikes_data: Lista de diccionarios con información de strikes
        expiration_date: Fecha de expiración (formato YYYYMMDD)
        action: "BUY" o "SELL"
    """
    basket_orders = []

    for strike_info in strikes_data:
        strike = strike_info['strike']
        opt_type = strike_info['type']  # 'C' o 'P'
        quantity = strike_info.get('quantity', 1)
        limit_price = strike_info.get('limit_price', '')

        # Convertir fecha al formato YYYYMMDD si viene en otro formato
        if isinstance(expiration_date, date):
            expiry_str = expiration_date.strftime("%Y%m%d")
        else:
            expiry_str = str(expiration_date)

        order = {
            'Symbol': ticker,
            'SecType': 'OPT',
            'Expiry': expiry_str,
            'Strike': strike,
            'Right': 'C' if opt_type == 'C' else 'P',
            'Exchange': 'SMART',
            'Currency': 'USD',
            'Quantity': quantity,
            'Side': action,
            'LimitPrice': limit_price if limit_price else '',
            'OrderType': 'LMT' if limit_price else 'MKT'
        }

        basket_orders.append(order)

    df_basket = pd.DataFrame(basket_orders)
    return df_basket

# ==============================================================================
# FUNCIÓN PRINCIPAL - TP CÁLCULOS
# ==============================================================================

def main_tp_calculos():
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
    if 'df_options' not in st.session_state:
        st.session_state.df_options = None

    st.markdown("<h1 style='text-align: center;'>📊 TP Cálculos - Expected Move</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # SIDEBAR - Parámetros de entrada
    with st.sidebar:
        st.header("⚙️ Parámetros de Entrada")

        ticker = st.text_input("Ticker del Subyacente", value="SPY", help="Ejemplo: SPY, QQQ, AAPL").upper()

        std_multiplier = st.number_input(
            "Multiplicador de Desviación Estándar",
            min_value=0.1,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="1.0 = 1 desviación estándar (~68%), 2.0 = 2 desviaciones (~95%)"
        )

        min_dte = st.number_input("DTE Mínimo", min_value=0, max_value=365, value=0)
        max_dte = st.number_input("DTE Máximo", min_value=0, max_value=365, value=60)

        if st.button("🔄 Calcular Expected Move", use_container_width=True):
            with st.spinner("Descargando datos..."):
                current_price = get_current_price(ticker)

                if current_price is None:
                    st.error("❌ No se pudo obtener el precio actual del ticker")
                    st.stop()

                st.session_state.current_price = current_price
                st.success(f"✅ Precio actual de {ticker}: ${current_price:.2f}")

                df_options = get_option_chain_cboe(ticker)

                if df_options is None or df_options.empty:
                    st.error("❌ No se pudo descargar la cadena de opciones")
                    st.stop()

                st.session_state.df_options = df_options
                df_options['dte'] = (df_options['expiry'] - date.today()).dt.days

                available_expirations = df_options[
                    (df_options['dte'] >= min_dte) & (df_options['dte'] <= max_dte)
                ]['expiry'].unique()

                if len(available_expirations) == 0:
                    st.error("❌ No hay expiraciones disponibles en el rango de DTE seleccionado")
                    st.stop()

                st.session_state.available_expirations = sorted(available_expirations)
                st.session_state.calculation_done = True
                st.session_state.selected_ticker = ticker
                st.session_state.std_multiplier = std_multiplier

    # AREA PRINCIPAL
    if st.session_state.calculation_done:
        st.success(f"✅ Datos cargados correctamente para {st.session_state.selected_ticker}")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Precio Actual", f"${st.session_state.current_price:.2f}")
        with col2:
            st.metric("Multiplicador σ", f"{st.session_state.std_multiplier}x")

        st.markdown("---")
        st.subheader("📅 Seleccionar Fecha de Expiración")

        expiration_options = [
            f"{exp} (DTE: {(exp - date.today()).days})"
            for exp in st.session_state.available_expirations
        ]

        selected_exp_str = st.selectbox(
            "Fecha de Expiración",
            options=expiration_options,
            help="Selecciona la fecha de expiración para calcular el Expected Move"
        )

        selected_index = expiration_options.index(selected_exp_str)
        expiration_date = st.session_state.available_expirations[selected_index]

        if st.button("📊 Calcular para esta Expiración"):
            expected_move, details, error = calculate_expected_move(
                st.session_state.df_options,
                expiration_date,
                st.session_state.current_price,
                st.session_state.std_multiplier
            )

            if error:
                st.error(f"❌ {error}")
            else:
                st.session_state.expected_move = expected_move
                st.session_state.details = details
                st.session_state.expiration_date = expiration_date

                st.success("✅ Expected Move calculado exitosamente")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Strike ATM", f"${details['atm_strike']:.2f}")
                with col2:
                    st.metric("Precio Straddle", f"${details['straddle_price']:.2f}", 
                             help=f"Usando {details['price_type']}")
                with col3:
                    st.metric("Expected Move", f"${expected_move:.2f}",
                             help=f"{st.session_state.std_multiplier}σ")

                upper_range = st.session_state.current_price + expected_move
                lower_range = st.session_state.current_price - expected_move

                st.markdown("### 📈 Rango de Movimiento Esperado")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Límite Inferior", f"${lower_range:.2f}")
                with col2:
                    st.metric("Precio Actual", f"${st.session_state.current_price:.2f}")
                with col3:
                    st.metric("Límite Superior", f"${upper_range:.2f}")

                # SECCIÓN NUEVA: Exportar a IBKR CSV
                st.markdown("---")
                st.markdown("### 📤 Exportar Basket para IBKR TWS")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Configuración de Strikes")

                    # Permitir al usuario seleccionar strikes
                    df_exp = st.session_state.df_options[
                        st.session_state.df_options['expiry'] == expiration_date
                    ].copy()

                    # Sugerir strikes basados en el expected move
                    suggested_call_strike = round((st.session_state.current_price + expected_move) / 5) * 5
                    suggested_put_strike = round((st.session_state.current_price - expected_move) / 5) * 5

                    st.info(f"💡 Strikes sugeridos basados en Expected Move:\n" +
                           f"- Call: ${suggested_call_strike:.2f}\n" +
                           f"- Put: ${suggested_put_strike:.2f}")

                    # Input para strikes personalizados
                    call_strike = st.number_input("Strike CALL", value=float(suggested_call_strike), step=5.0)
                    put_strike = st.number_input("Strike PUT", value=float(suggested_put_strike), step=5.0)

                    quantity = st.number_input("Cantidad (contratos)", min_value=1, value=1, step=1)

                    action = st.selectbox("Acción", ["BUY", "SELL"])

                    # Precios límite opcionales
                    use_limit = st.checkbox("Usar precios límite", value=False)

                    call_limit_price = ""
                    put_limit_price = ""

                    if use_limit:
                        call_limit_price = st.number_input("Precio límite CALL", min_value=0.0, value=0.0, step=0.05)
                        put_limit_price = st.number_input("Precio límite PUT", min_value=0.0, value=0.0, step=0.05)

                with col2:
                    st.markdown("#### Preview del Basket")

                    strikes_data = [
                        {
                            'strike': call_strike,
                            'type': 'C',
                            'quantity': quantity,
                            'limit_price': call_limit_price if use_limit else ''
                        },
                        {
                            'strike': put_strike,
                            'type': 'P',
                            'quantity': quantity,
                            'limit_price': put_limit_price if use_limit else ''
                        }
                    ]

                    df_basket = create_ibkr_basket_csv(
                        st.session_state.selected_ticker,
                        strikes_data,
                        expiration_date,
                        action
                    )

                    st.dataframe(df_basket, use_container_width=True)

                # Botón de descarga
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])

                with col2:
                    # Convertir a CSV
                    csv_buffer = io.StringIO()
                    df_basket.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()

                    filename = f"{st.session_state.selected_ticker}_basket_{expiration_date.strftime('%Y%m%d')}.csv"

                    st.download_button(
                        label="⬇️ Descargar CSV para IBKR",
                        data=csv_data,
                        file_name=filename,
                        mime="text/csv",
                        use_container_width=True
                    )

                    st.caption("💡 Este archivo puede ser importado en IBKR TWS usando BasketTrader")

# ==============================================================================
# EJECUCIÓN
# ==============================================================================

if __name__ == "__main__":
    if check_password():
        main_tp_calculos()
