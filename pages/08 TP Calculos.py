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
    
    col1, col2, col3 = st.columns(3)
    
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
    
    with col3:
        # Selector de Desviaciones Estándar
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
            expected_move, details, error = calculate_expected_move(
                df_options, 
                expiration_date, 
                current_price,
                std_multiplier
            )
            
            if error:
                st.error(f"❌ {error}")
                st.stop()
            
            st.success(f"✅ Expected Move calculado: **${expected_move:.2f}** ({std_multiplier}σ)")
            st.info(f"💰 Precio del Straddle: **${details['straddle_price']:.2f}** ({details['price_type']})")
        
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
            
            # Convertir expiration_date a datetime para el gráfico
            exp_datetime = datetime.combine(expiration_date, datetime.min.time())
            
            # Línea horizontal - Rango Superior
            fig.add_hline(
                y=upper_range,
                line_dash="dot",
                line_color="steelblue",
                annotation_text=f"${upper_range:.2f} (+{std_multiplier}σ)",
                annotation_position="right"
            )
            
            # Línea horizontal - Rango Inferior
            fig.add_hline(
                y=lower_range,
                line_dash="dot",
                line_color="steelblue",
                annotation_text=f"${lower_range:.2f} (-{std_multiplier}σ)",
                annotation_position="right"
            )
            
            # Línea horizontal - Precio Actual
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
            
            # Añadir anotación para la fecha de expiración
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
            end_datetime = exp_datetime + timedelta(days=3)
            
            fig.update_layout(
                title=f"Expected Move - {selected_ticker} ({std_multiplier}σ)",
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
        # SECCIÓN 4: COMPARACIÓN DE DESVIACIONES ESTÁNDAR
        # ==============================================================================
        st.header("4. Comparación de Desviaciones Estándar")
        
        # Calcular para todas las desviaciones estándar
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
        # SECCIÓN 5: INFORMACIÓN ADICIONAL
        # ==============================================================================
        st.header("5. Información Adicional")
        
        # Determinar el nivel de probabilidad según el std_multiplier
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

# ==============================================================================
# PUNTO DE ENTRADA PROTEGIDO
# ==============================================================================

if __name__ == "__main__":
    
    if check_password():
        main_tp_calculos()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder a TP Cálculos.")
