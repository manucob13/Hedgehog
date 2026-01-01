import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime
from utils import check_password

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="VIX Term Structure - CBOE", layout="wide")

# ==============================================================================
# OBTENCIÓN DE DATOS CBOE Y SPOT
# ==============================================================================

@st.cache_data(ttl=300)
def get_vix_spot():
    """Obtiene el precio actual del índice VIX (Spot) desde Yahoo Finance."""
    try:
        import yfinance as yf
        vix = yf.Ticker("^VIX")
        price = vix.history(period="1d")['Close'].iloc[-1]
        return price
    except:
        return None

@st.cache_data(ttl=300)
def get_vix_futures_data():
    """Descarga datos de futuros VIX desde CBOE con headers correctos."""
    try:
        # Probar múltiples endpoints posibles
        urls = [
            "https://cdn.cboe.com/api/global/delayed_quotes/futures/_VX.json",
            "https://cdn.cboe.com/api/global/delayed_quotes/futures/VX.json",
            "https://cdn.cboe.com/api/global/delayed_quotes/futures/VIX.json"
        ]
        
        # Headers simplificados basados en el código que funciona
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.cboe.com/",
            "Origin": "https://www.cboe.com"
        }
        
        data = None
        for url in urls:
            try:
                response = requests.get(url, headers=headers, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    break
            except:
                continue
        
        if not data:
            st.error("No se pudo conectar con ningún endpoint de CBOE")
            return None
        
        # Procesar la respuesta según la estructura
        futures_list = []
        
        if 'data' in data:
            # Opción 1: estructura con 'futures' key
            if 'futures' in data['data']:
                futures_list = data['data']['futures']
            # Opción 2: estructura con keys de contratos directos
            else:
                for key, value in data['data'].items():
                    if isinstance(value, dict) and 'expiration_date' in value:
                        futures_list.append(value)
        
        if futures_list:
            df = pd.DataFrame(futures_list)
            df['expiry'] = pd.to_datetime(df['expiration_date'])
            df = df.sort_values('expiry').reset_index(drop=True)
            
            # Seleccionamos los primeros 8 meses (F1 a F8)
            df = df.head(8)
            df['label'] = [f"F{i+1}" for i in range(len(df))]
            
            return df
        
        return None
        
    except Exception as e:
        st.error(f"Error al conectar con CBOE: {e}")
        return None

# ==============================================================================
# FUNCIÓN PRINCIPAL (CONTENIDO DE LA APP)
# ==============================================================================

def main_vix_structure():
    st.markdown("<h1 style='text-align: center;'>📊 VIX Term Structure - CBOE</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Obtener datos
    with st.spinner("Obteniendo datos de CBOE..."):
        vix_spot = get_vix_spot()
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
            f1_price = df_futures.iloc[0].get('close', df_futures.iloc[0].get('last', None))
            if f1_price:
                st.metric("VIX F1 (Front Month)", f"{f1_price:.2f}")
            else:
                st.metric("VIX F1 (Front Month)", "N/A")
        else:
            st.metric("VIX F1 (Front Month)", "N/A")
    
    with col3:
        if df_futures is not None and not df_futures.empty and vix_spot:
            f1_price = df_futures.iloc[0].get('close', df_futures.iloc[0].get('last', None))
            if f1_price:
                contango = f1_price - vix_spot
                st.metric("Contango/Backwardation", f"{contango:.2f}", 
                         delta="Contango" if contango > 0 else "Backwardation")
            else:
                st.metric("Contango/Backwardation", "N/A")
        else:
            st.metric("Contango/Backwardation", "N/A")
    
    st.markdown("---")
    
    # Mostrar datos y gráfico
    if df_futures is not None and not df_futures.empty:
        
        # Tabla de datos
        st.subheader("📋 Datos de Futuros VIX")
        
        # Preparar columnas para mostrar
        display_cols = ['label', 'expiration_date']
        
        # Detectar qué columna de precio existe
        if 'close' in df_futures.columns:
            display_cols.append('close')
            price_col = 'close'
        elif 'last' in df_futures.columns:
            display_cols.append('last')
            price_col = 'last'
        else:
            price_col = None
        
        if 'volume' in df_futures.columns:
            display_cols.append('volume')
        if 'open_interest' in df_futures.columns:
            display_cols.append('open_interest')
        
        df_display = df_futures[display_cols].copy()
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        # Gráfico de la curva de futuros
        st.subheader("📈 Curva de Futuros VIX")
        
        if price_col:
            fig = go.Figure()
            
            # Añadir línea de futuros
            fig.add_trace(go.Scatter(
                x=df_futures['label'],
                y=df_futures[price_col],
                mode='lines+markers',
                name='VIX Futures',
                line=dict(color='#FF6B6B', width=3),
                marker=dict(size=10)
            ))
            
            # Añadir línea de spot si está disponible
            if vix_spot:
                fig.add_trace(go.Scatter(
                    x=['Spot'],
                    y=[vix_spot],
                    mode='markers',
                    name='VIX Spot',
                    marker=dict(size=15, color='#4ECDC4', symbol='star')
                ))
            
            fig.update_layout(
                title="VIX Term Structure",
                xaxis_title="Contrato",
                yaxis_title="Precio",
                hovermode='x unified',
                template='plotly_white',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No se encontró columna de precios en los datos")
    else:
        st.error("❌ No se pudieron obtener los datos de futuros VIX desde CBOE")
        st.info("💡 Posibles causas:\n"
                "- El endpoint de CBOE ha cambiado\n"
                "- Restricciones de acceso temporal\n"
                "- Problemas de conectividad")

# ==============================================================================
# PUNTO DE ENTRADA PROTEGIDO
# ==============================================================================
if __name__ == "__main__":
    
    if check_password():
        main_vix_structure()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder a VIX Term Structure.")
