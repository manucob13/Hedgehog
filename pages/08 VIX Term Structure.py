import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils import check_password

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="VIX Term Structure - Yahoo Finance", layout="wide")

# ==============================================================================
# OBTENCIÓN DE DATOS VIX DESDE YAHOO FINANCE
# ==============================================================================

@st.cache_data(ttl=300)
def get_vix_spot():
    """Obtiene el precio actual del índice VIX (Spot) desde Yahoo Finance."""
    try:
        import yfinance as yf
        vix = yf.Ticker("^VIX")
        price = vix.history(period="1d")['Close'].iloc[-1]
        return price
    except Exception as e:
        st.error(f"Error obteniendo VIX Spot: {e}")
        return None

@st.cache_data(ttl=300)
def get_vix_futures_data():
    """Obtiene datos de futuros VIX desde Yahoo Finance usando índices mensuales."""
    try:
        import yfinance as yf
        
        # Yahoo Finance tiene símbolos específicos para cada mes de futuros VIX
        # Formato: ^VIX + código mes (JAN, FEB, MAR, APR, MAY, JUN, JUL, AUG, SEP, OCT, NOV, DEC)
        
        # Determinar los próximos 8 meses de contratos
        month_codes = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                       'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        
        current_date = datetime.now()
        futures_data = []
        
        # Intentar obtener los próximos 12 meses (para asegurar 8 válidos)
        for i in range(12):
            future_month = current_date + timedelta(days=30*i)
            month_name = month_codes[future_month.month - 1]
            
            # Yahoo Finance usa formato: ^VIX + MES (ej: ^VIXJAN, ^VIXFEB)
            symbol = f"^VIX{month_name}"
            
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                
                if not hist.empty:
                    last_price = hist['Close'].iloc[-1]
                    
                    # Estimar fecha de expiración (tercer miércoles del mes)
                    # VIX expira el miércoles anterior al tercer viernes del mes
                    year = future_month.year
                    month = future_month.month
                    
                    # Encontrar el tercer viernes
                    first_day = datetime(year, month, 1)
                    first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
                    third_friday = first_friday + timedelta(weeks=2)
                    expiration = third_friday - timedelta(days=2)  # Miércoles anterior
                    
                    futures_data.append({
                        'symbol': symbol,
                        'month': month_name,
                        'close': last_price,
                        'expiration_date': expiration.strftime('%Y-%m-%d'),
                        'expiry': expiration,
                        'volume': hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0
                    })
                    
            except Exception as e:
                continue
        
        if not futures_data:
            return None
        
        # Crear DataFrame y ordenar por fecha de expiración
        df = pd.DataFrame(futures_data)
        df = df.sort_values('expiry').reset_index(drop=True)
        
        # Seleccionar los primeros 8 contratos
        df = df.head(8)
        df['label'] = [f"F{i+1}" for i in range(len(df))]
        
        return df
        
    except Exception as e:
        st.error(f"Error obteniendo futuros VIX: {e}")
        return None

@st.cache_data(ttl=300)
def get_vix_futures_generic():
    """Método alternativo: usar contratos genéricos de Yahoo Finance."""
    try:
        import yfinance as yf
        
        # Yahoo Finance también tiene índices agregados de futuros VIX
        vix_indices = {
            '^SPVIX1M': 'VIX 1-Month',
            '^SPVIX3M': 'VIX 3-Month', 
            '^SPVIX6M': 'VIX 6-Month',
            '^SPVIX9M': 'VIX 9-Month'
        }
        
        futures_data = []
        
        for symbol, label in vix_indices.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                
                if not hist.empty:
                    futures_data.append({
                        'symbol': symbol,
                        'label': label,
                        'close': hist['Close'].iloc[-1]
                    })
            except:
                continue
        
        if futures_data:
            return pd.DataFrame(futures_data)
        
        return None
        
    except Exception as e:
        return None

# ==============================================================================
# FUNCIÓN PRINCIPAL (CONTENIDO DE LA APP)
# ==============================================================================

def main_vix_structure():
    st.markdown("<h1 style='text-align: center;'>📊 VIX Term Structure - Yahoo Finance</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Obtener datos
    with st.spinner("Obteniendo datos desde Yahoo Finance..."):
        vix_spot = get_vix_spot()
        df_futures = get_vix_futures_data()
        
        # Si el método principal falla, intentar método genérico
        if df_futures is None or df_futures.empty:
            st.info("Intentando método alternativo con índices agregados...")
            df_futures_alt = get_vix_futures_generic()
    
    # Mostrar métricas principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if vix_spot:
            st.metric("VIX Spot", f"{vix_spot:.2f}")
        else:
            st.metric("VIX Spot", "N/A")
    
    with col2:
        if df_futures is not None and not df_futures.empty:
            f1_price = df_futures.iloc[0]['close']
            st.metric("VIX F1 (Front Month)", f"{f1_price:.2f}")
        else:
            st.metric("VIX F1 (Front Month)", "N/A")
    
    with col3:
        if df_futures is not None and not df_futures.empty and vix_spot:
            f1_price = df_futures.iloc[0]['close']
            contango = f1_price - vix_spot
            st.metric("Contango/Backwardation", f"{contango:.2f}", 
                     delta="Contango" if contango > 0 else "Backwardation")
        else:
            st.metric("Contango/Backwardation", "N/A")
    
    st.markdown("---")
    
    # Mostrar datos y gráfico principal
    if df_futures is not None and not df_futures.empty:
        
        # Tabla de datos
        st.subheader("📋 Futuros VIX - Estructura de Términos")
        
        # Preparar tabla para mostrar
        display_cols = ['label', 'month', 'expiration_date', 'close']
        if 'volume' in df_futures.columns:
            display_cols.append('volume')
        
        df_display = df_futures[display_cols].copy()
        df_display.columns = ['Contrato', 'Mes', 'Expiración', 'Precio', 'Volumen'][:len(display_cols)]
        
        st.dataframe(
            df_display.style.format({
                'Precio': '{:.2f}',
                'Volumen': '{:,.0f}' if 'Volumen' in df_display.columns else None
            }),
            use_container_width=True, 
            hide_index=True
        )
        
        # Gráfico de la curva de futuros
        st.subheader("📈 Curva de Futuros VIX")
        
        fig = go.Figure()
        
        # Línea de futuros
        fig.add_trace(go.Scatter(
            x=df_futures['label'],
            y=df_futures['close'],
            mode='lines+markers',
            name='VIX Futures',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=10),
            hovertemplate='<b>%{x}</b><br>Precio: %{y:.2f}<extra></extra>'
        ))
        
        # Añadir línea horizontal de spot si está disponible
        if vix_spot:
            fig.add_hline(
                y=vix_spot, 
                line_dash="dash", 
                line_color="#4ECDC4",
                annotation_text=f"VIX Spot: {vix_spot:.2f}",
                annotation_position="right"
            )
        
        fig.update_layout(
            title="VIX Term Structure",
            xaxis_title="Contrato",
            yaxis_title="Precio",
            hovermode='x unified',
            template='plotly_white',
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Análisis de contango/backwardation
        st.subheader("📊 Análisis de Estructura")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Calcular pendiente promedio
            if len(df_futures) >= 2:
                slope = (df_futures.iloc[-1]['close'] - df_futures.iloc[0]['close']) / len(df_futures)
                
                if slope > 0.5:
                    status = "🔴 Contango Fuerte"
                    color = "red"
                elif slope > 0:
                    status = "🟡 Contango Moderado"
                    color = "orange"
                elif slope > -0.5:
                    status = "🟢 Backwardation Moderado"
                    color = "green"
                else:
                    status = "🟢 Backwardation Fuerte"
                    color = "green"
                
                st.markdown(f"### {status}")
                st.metric("Pendiente promedio", f"{slope:.3f} por contrato")
        
        with col2:
            # Spread F1-F8
            if len(df_futures) >= 8:
                spread = df_futures.iloc[7]['close'] - df_futures.iloc[0]['close']
                st.metric(
                    "Spread F1-F8", 
                    f"{spread:.2f}",
                    delta=f"{(spread/df_futures.iloc[0]['close']*100):.1f}%"
                )
            
            # Premium sobre spot
            if vix_spot:
                premium = df_futures.iloc[0]['close'] - vix_spot
                st.metric(
                    "Premium F1 sobre Spot",
                    f"{premium:.2f}",
                    delta=f"{(premium/vix_spot*100):.1f}%"
                )
        
    else:
        st.warning("⚠️ No se pudieron obtener datos de futuros mensuales")
        
        # Mostrar datos alternativos si están disponibles
        if 'df_futures_alt' in locals() and df_futures_alt is not None:
            st.subheader("📊 Índices de Futuros VIX Agregados")
            st.info("Mostrando índices agregados en lugar de contratos individuales")
            
            # Tabla alternativa
            df_alt_display = df_futures_alt.copy()
            st.dataframe(
                df_alt_display.style.format({'close': '{:.2f}'}),
                use_container_width=True,
                hide_index=True
            )
            
            # Gráfico alternativo
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=df_futures_alt['label'],
                y=df_futures_alt['close'],
                marker_color='#FF6B6B',
                text=df_futures_alt['close'].round(2),
                textposition='outside'
            ))
            
            if vix_spot:
                fig.add_hline(
                    y=vix_spot,
                    line_dash="dash",
                    line_color="#4ECDC4",
                    annotation_text=f"VIX Spot: {vix_spot:.2f}",
                    annotation_position="right"
                )
            
            fig.update_layout(
                title="Índices de Futuros VIX Agregados",
                xaxis_title="Índice",
                yaxis_title="Valor",
                template='plotly_white',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("❌ No se pudieron obtener datos de Yahoo Finance")
            st.info("💡 **Nota**: Yahoo Finance puede tener disponibilidad limitada para futuros VIX.\n\n"
                    "**Alternativas recomendadas:**\n"
                    "- Usar broker con acceso a datos (Interactive Brokers, TD Ameritrade)\n"
                    "- APIs comerciales (Bloomberg, Quandl, IVolatility)\n"
                    "- Datos históricos de CBOE: https://www.cboe.com/us/futures/market_statistics/historical_data/")
    
    # Información adicional
    st.markdown("---")
    with st.expander("ℹ️ Información sobre Futuros VIX"):
        st.markdown("""
        ### Sobre los Futuros VIX
        
        - **Símbolo**: VX en CBOE
        - **Tamaño del contrato**: $1,000 × nivel del índice VIX
        - **Expiración**: Miércoles, 30 días antes del tercer viernes del mes calendario
        - **Horario de trading**: Domingo - Viernes, 5:00 PM - 8:15 AM CT (siguiente día)
        
        ### Estructura de Términos
        
        - **Contango**: Cuando los futuros tienen precio superior al spot (curva ascendente)
        - **Backwardation**: Cuando los futuros tienen precio inferior al spot (curva descendente)
        
        ### Fuente de Datos
        
        Datos obtenidos de Yahoo Finance mediante la librería yfinance.
        Los símbolos utilizados son: ^VIXJAN, ^VIXFEB, ^VIXMAR, etc.
        """)

# ==============================================================================
# PUNTO DE ENTRADA PROTEGIDO
# ==============================================================================
if __name__ == "__main__":
    
    if check_password():
        main_vix_structure()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder a VIX Term Structure.")
