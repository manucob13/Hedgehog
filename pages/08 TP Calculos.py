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

def calculate_expected_move(df_options, expiration_date, current_price):
    """Calcula el Expected Move basado en el straddle ATM."""
    try:
        # Filtrar por fecha de expiración
        df_exp = df_options[df_options['expiry'] == expiration_date].copy()
        
        if df_exp.empty:
            return None, "No hay datos para esa fecha de expiración"
        
        # Encontrar el strike más cercano al precio actual (ATM)
        df_exp['distance'] = abs(df_exp['strike'] - current_price)
        atm_strike = df_exp.loc[df_exp['distance'].idxmin(), 'strike']
        
        # Obtener el Call y Put ATM
        call_atm = df_exp[(df_exp['strike'] == atm_strike) & (df_exp['opt_type'] == 'C')]
        put_atm = df_exp[(df_exp['strike'] == atm_strike) & (df_exp['opt_type'] == 'P')]
        
        if call_atm.empty or put_atm.empty:
            return None, "No se encontraron opciones ATM"
        
        # Calcular el precio del straddle
        call_bid = call_atm['bid'].iloc[0]
        call_ask = call_atm['ask'].iloc[0]
        put_bid = put_atm['bid'].iloc[0]
        put_ask = put_atm['ask'].iloc[0]
        
        # Usar BID si está disponible, sino ASK, sino LAST
        if call_bid > 0 and put_bid > 0:
            straddle_price = call_bid + put_bid
        elif call_ask > 0 and put_ask > 0:
            straddle_price = call_ask + put_ask
        else:
            call_last = call_atm['last_trade_price'].iloc[0]
            put_last = put_atm['last_trade_price'].iloc[0]
            straddle_price = call_last + put_last
        
        # Expected Move = Straddle Price * 0.85 (aproximadamente 1 desviación estándar)
        expected_move = straddle_price * 0.85
        
        return expected_move, None
        
    except Exception as e:
        return None, f"Error calculando Expected Move: {e}"

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
# FUNCIÓN PRINCIPAL - TP CÁLCULOS
# ==============================================================================

def main_tp_calculos():
    
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Selector de Ticker
        ticker_options = ['SPX', 'SPY', 'QQQ', 'XSP']
        selected_ticker = st.selectbox(
            "Selecciona el Ticker",
            ticker_options,
            index=1,  # SPY por defecto
            key='ticker_tp'
        )
        st.info(f"📊 Ticker seleccionado: **{selected_ticker}**")
    
    with col2:
        # Selector de Fecha de Expiración
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
    
    st.markdown("---")
    
    # ==============================================================================
    # BOTÓN PARA CALCULAR
    # ==============================================================================
    
    if st.button("🚀 Calcular Expected Move", type="primary", use_container_width=True):
        
        with st.spinner(f"Obteniendo datos de {selected_ticker}..."):
            
            # Obtener precio actual
            current_price = get_current_price(selected_ticker)
            
            if current_price is None:
                st.error("❌ No se pudo obtener el precio actual del activo.")
                st.stop()
            
            st.success(f"✅ Precio actual de {selected_ticker}: **${current_price:.2f}**")
            
            # Obtener cadena de opciones
            df_options = get_option_chain_cboe(selected_ticker)
            
            if df_options is None or df_options.empty:
                st.error("❌ No se pudieron obtener los datos de opciones desde CBOE.")
                st.stop()
            
            # Calcular Expected Move
            expected_move, error = calculate_expected_move(
                df_options, 
                expiration_date, 
                current_price
            )
            
            if error:
                st.error(f"❌ {error}")
                st.stop()
            
            st.success(f"✅ Expected Move calculado: **${expected_move:.2f}**")
        
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
            st.metric("Expected Move", f"${expected_move:.2f}")
        with col3:
            st.metric("Rango Superior", f"${upper_range:.2f}", f"+{expected_move:.2f}")
        with col4:
            st.metric("Rango Inferior", f"${lower_range:.2f}", f"-{expected_move:.2f}")
        
        # Tabla resumen
        st.markdown("### 📊 Resumen del Expected Move")
        
        days_to_exp = (expiration_date - date.today()).days
        move_pct = (expected_move / current_price) * 100
        
        summary_data = {
            'Métrica': [
                'Precio Actual',
                'Expected Move (±)',
                'Rango Superior',
                'Rango Inferior',
                'Movimiento (%)',
                'Días hasta Expiración',
                'Fecha de Expiración'
            ],
            'Valor': [
                f"${current_price:.2f}",
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
        
        # Obtener datos históricos
        df_hist = get_historical_prices(selected_ticker, days=7)
        
        if df_hist is not None and not df_hist.empty:
            
            # Crear el gráfico
            fig = go.Figure()
            
            # Convertir el índice a timezone-naive
            df_hist_plot = df_hist.copy()
            if df_hist_plot.index.tz is not None:
                df_hist_plot.index = df_hist_plot.index.tz_localize(None)
            
            # Línea del precio histórico
            fig.add_trace(go.Scatter(
                x=df_hist_plot.index,
                y=df_hist_plot['Close'],
                mode='lines',
                name=selected_ticker,
                line=dict(color='#00B06B', width=2)
            ))
            
            # Crear fechas extendidas sin timezone
            last_date = df_hist_plot.index[-1]
            exp_datetime = pd.Timestamp(expiration_date)
            end_datetime = exp_datetime + timedelta(days=3)
            
            # Línea horizontal - Rango Superior
            fig.add_hline(
                y=upper_range,
                line_dash="dot",
                line_color="steelblue",
                annotation_text=f"${upper_range:.2f}",
                annotation_position="right"
            )
            
            # Línea horizontal - Rango Inferior
            fig.add_hline(
                y=lower_range,
                line_dash="dot",
                line_color="steelblue",
                annotation_text=f"${lower_range:.2f}",
                annotation_position="right"
            )
            
            # Línea vertical - Fecha de Expiración
            fig.add_vline(
                x=exp_datetime,
                line_dash="dot",
                line_color="red",
                annotation_text=expiration_date.strftime('%Y-%m-%d'),
                annotation_position="top"
            )
            
            # Configuración del layout
            fig.update_layout(
                title=f"Expected Move - {selected_ticker}",
                xaxis_title="Fecha",
                yaxis_title="Precio",
                template="plotly_dark",
                height=500,
                hovermode='x unified',
                showlegend=True,
                xaxis=dict(
                    range=[df_hist_plot.index[0], end_datetime]
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("⚠️ No se pudieron obtener datos históricos para el gráfico.")
        
        st.markdown("---")
        
        # ==============================================================================
        # SECCIÓN 4: INFORMACIÓN ADICIONAL
        # ==============================================================================
        st.header("4. Información Adicional")
        
        st.info(f"""
        📌 **Interpretación del Expected Move:**
        
        - El **Expected Move** representa el rango de precio esperado (±1 desviación estándar) 
          que el mercado anticipa para la fecha de expiración.
        
        - Este cálculo se basa en el precio del **straddle ATM** (comprar un call y un put 
          al mismo strike más cercano al precio actual).
        
        - Aproximadamente el **68%** de las veces, el precio debería permanecer dentro de este rango.
        
        - **Ticker:** {selected_ticker}
        - **Precio Actual:** ${current_price:.2f}
        - **Expected Move:** ±${expected_move:.2f} (±{move_pct:.2f}%)
        - **Rango Esperado:** ${lower_range:.2f} - ${upper_range:.2f}
        - **Días hasta Expiración:** {days_to_exp}
        """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 📚 Fuentes de Datos
        - **Precios de Opciones:** CBOE (Chicago Board Options Exchange)
        - **Precios del Activo:** Yahoo Finance
        - **Cálculo:** Basado en el método del straddle ATM × 0.85
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
