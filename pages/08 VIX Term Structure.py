import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
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
    """Obtiene estructura de términos usando ETFs y proxies de futuros VIX disponibles en Yahoo."""
    try:
        import yfinance as yf
        
        # Yahoo Finance tiene ETFs y proxies que rastrean futuros VIX
        # Estos son los instrumentos más confiables disponibles
        vix_instruments = {
            '^VIX': {'label': 'Spot', 'days': 0, 'type': 'spot'},
            '^VIX1D': {'label': '1-Day', 'days': 1, 'type': 'future'},
            '^VIX3M': {'label': '3-Month', 'days': 90, 'type': 'future'},
            '^VIX6M': {'label': '6-Month', 'days': 180, 'type': 'future'},
            '^VXAZN': {'label': 'VXAZN', 'days': 30, 'type': 'future'},
            '^VXAPL': {'label': 'VXAPL', 'days': 60, 'type': 'future'},
        }
        
        futures_data = []
        
        # Primero intentar con los índices disponibles
        for symbol, info in vix_instruments.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                
                if not hist.empty and len(hist) > 0:
                    last_price = hist['Close'].iloc[-1]
                    
                    futures_data.append({
                        'symbol': symbol,
                        'label': info['label'],
                        'close': last_price,
                        'days': info['days'],
                        'type': info['type']
                    })
            except:
                continue
        
        # Si no hay suficientes datos, usar proxies de ETFs de futuros VIX
        if len(futures_data) < 3:
            etf_proxies = {
                'VXX': {'label': 'VXX (1-2M)', 'days': 45},
                'VIXY': {'label': 'VIXY (1M)', 'days': 30},
                'VIXM': {'label': 'VIXM (5M)', 'days': 150},
            }
            
            for symbol, info in etf_proxies.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="5d")
                    
                    if not hist.empty:
                        last_price = hist['Close'].iloc[-1]
                        
                        futures_data.append({
                            'symbol': symbol,
                            'label': info['label'],
                            'close': last_price,
                            'days': info['days'],
                            'type': 'etf'
                        })
                except:
                    continue
        
        if not futures_data:
            return None
        
        # Crear DataFrame y ordenar por días
        df = pd.DataFrame(futures_data)
        df = df.sort_values('days').reset_index(drop=True)
        
        return df
        
    except Exception as e:
        st.error(f"Error obteniendo datos VIX: {e}")
        return None

@st.cache_data(ttl=300)
def get_vix_term_structure_from_csv():
    """Intenta obtener datos directos de CBOE usando archivos CSV públicos."""
    try:
        import yfinance as yf
        
        # Usar el índice de term structure de S&P que Yahoo sí tiene
        term_structure_symbols = [
            '^SPVIX1M',  # 1 mes
            '^SPVIX3M',  # 3 meses  
            '^SPVIX6M',  # 6 meses
            '^SPVIX9M',  # 9 meses
        ]
        
        futures_data = []
        days_map = [30, 90, 180, 270]
        
        for i, symbol in enumerate(term_structure_symbols):
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                
                if not hist.empty:
                    last_price = hist['Close'].iloc[-1]
                    
                    futures_data.append({
                        'symbol': symbol,
                        'label': f'{days_map[i]//30}M',
                        'close': last_price,
                        'days': days_map[i],
                        'month': f'{days_map[i]//30} Month'
                    })
            except:
                continue
        
        if futures_data:
            df = pd.DataFrame(futures_data)
            df = df.sort_values('days').reset_index(drop=True)
            df['label'] = [f"F{i+1}" for i in range(len(df))]
            return df
        
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
        
        # Intentar primero con índices de term structure
        df_futures = get_vix_term_structure_from_csv()
        
        # Si falla, usar método de ETFs/proxies
        if df_futures is None or df_futures.empty:
            st.info("Usando proxies de ETFs para aproximar la estructura...")
            df_futures = get_vix_futures_data()
    
    # Mostrar métricas principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if vix_spot:
            st.metric("VIX Spot", f"{vix_spot:.2f}")
        else:
            st.metric("VIX Spot", "N/A")
    
    with col2:
        if df_futures is not None and not df_futures.empty:
            # Buscar el contrato más cercano (excluyendo spot si está)
            front_month = df_futures[df_futures['days'] > 0].iloc[0] if len(df_futures[df_futures['days'] > 0]) > 0 else df_futures.iloc[0]
            st.metric("Front Month", f"{front_month['close']:.2f}", delta=front_month['label'])
        else:
            st.metric("Front Month", "N/A")
    
    with col3:
        if df_futures is not None and not df_futures.empty and vix_spot:
            front_month = df_futures[df_futures['days'] > 0].iloc[0] if len(df_futures[df_futures['days'] > 0]) > 0 else df_futures.iloc[0]
            contango = front_month['close'] - vix_spot
            st.metric("Contango/Backwardation", f"{contango:.2f}", 
                     delta="Contango" if contango > 0 else "Backwardation")
        else:
            st.metric("Contango/Backwardation", "N/A")
    
    st.markdown("---")
    
    # Mostrar datos y gráfico
    if df_futures is not None and not df_futures.empty:
        
        # Tabla de datos
        st.subheader("📋 VIX Term Structure")
        
        # Preparar tabla
        display_cols = ['label', 'symbol', 'close', 'days']
        if 'month' in df_futures.columns:
            display_cols.insert(1, 'month')
        
        df_display = df_futures[display_cols].copy()
        
        # Renombrar columnas
        col_names = ['Contrato', 'Símbolo', 'Precio', 'Días']
        if 'month' in display_cols:
            col_names.insert(1, 'Período')
        
        df_display.columns = col_names
        
        st.dataframe(
            df_display.style.format({
                'Precio': '{:.2f}',
                'Días': '{:.0f}'
            }),
            use_container_width=True, 
            hide_index=True
        )
        
        # Gráfico de la curva
        st.subheader("📈 Curva de Term Structure")
        
        fig = go.Figure()
        
        # Línea de estructura
        fig.add_trace(go.Scatter(
            x=df_futures['label'],
            y=df_futures['close'],
            mode='lines+markers',
            name='VIX Term Structure',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=10),
            hovertemplate='<b>%{x}</b><br>Precio: %{y:.2f}<extra></extra>'
        ))
        
        # Línea de referencia de spot
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
            xaxis_title="Contrato / Período",
            yaxis_title="Valor",
            hovermode='x unified',
            template='plotly_white',
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Análisis
        st.subheader("📊 Análisis de Estructura")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Calcular pendiente
            if len(df_futures) >= 2:
                # Usar contratos que no sean spot
                non_spot = df_futures[df_futures['days'] > 0]
                if len(non_spot) >= 2:
                    slope = (non_spot.iloc[-1]['close'] - non_spot.iloc[0]['close']) / len(non_spot)
                    
                    if slope > 0.5:
                        status = "🔴 Contango Fuerte"
                    elif slope > 0:
                        status = "🟡 Contango Moderado"
                    elif slope > -0.5:
                        status = "🟢 Backwardation Moderado"
                    else:
                        status = "🟢 Backwardation Fuerte"
                    
                    st.markdown(f"### {status}")
                    st.metric("Pendiente", f"{slope:.3f}")
        
        with col2:
            # Spread entre primer y último
            if len(df_futures) >= 2:
                spread = df_futures.iloc[-1]['close'] - df_futures.iloc[0]['close']
                st.metric(
                    "Spread Total", 
                    f"{spread:.2f}",
                    delta=f"{(spread/df_futures.iloc[0]['close']*100):.1f}%"
                )
        
        # Nota sobre los datos
        st.info("📝 **Nota**: Debido a limitaciones de Yahoo Finance, estos datos representan una aproximación "
                "de la estructura de términos de futuros VIX usando índices agregados disponibles públicamente.")
        
    else:
        st.error("❌ No se pudieron obtener datos de Yahoo Finance")
        
        st.markdown("### 💡 Soluciones Alternativas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Opción 1: Datos Históricos CBOE**
            
            Descarga archivos CSV directamente:
            """)
            st.link_button("📊 CBOE Historical Data", 
                          "https://www.cboe.com/us/futures/market_statistics/historical_data/")
        
        with col2:
            st.markdown("""
            **Opción 2: Ver en Web**
            
            Consulta la estructura actual en tiempo real:
            """)
            st.link_button("🌐 CBOE VIX Term Structure", 
                          "https://www.cboe.com/tradable-products/vix/term-structure/")
    
    # Información adicional
    st.markdown("---")
    with st.expander("ℹ️ Información sobre los Datos"):
        st.markdown("""
        ### Fuentes de Datos Utilizadas
        
        **Símbolos de Yahoo Finance:**
        - `^VIX` - Índice VIX Spot
        - `^SPVIX1M` - S&P 500 VIX 1-Month Futures Index
        - `^SPVIX3M` - S&P 500 VIX 3-Month Futures Index
        - `^SPVIX6M` - S&P 500 VIX 6-Month Futures Index
        - `^SPVIX9M` - S&P 500 VIX 9-Month Futures Index
        
        **ETFs Alternativos (si es necesario):**
        - `VXX` - iPath Series B S&P 500 VIX Short-Term Futures ETN
        - `VIXY` - ProShares VIX Short-Term Futures ETF
        - `VIXM` - ProShares VIX Mid-Term Futures ETF
        
        ### Limitaciones
        
        Yahoo Finance no proporciona acceso directo a contratos individuales de futuros VIX (VXF26, VXG26, etc.).
        Los datos mostrados son índices agregados que rastrean cestas de futuros VIX.
        
        Para datos de contratos individuales, se requiere acceso a:
        - CBOE DataShop (comercial)
        - Brokers con acceso a datos (Interactive Brokers, TD Ameritrade)
        - APIs comerciales (Bloomberg, Quandl, IVolatility)
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
