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
    """Descarga datos de futuros con Headers avanzados para evitar el Error 403."""
    try:
        url = "https://cdn.cboe.com/api/global/delayed_quotes/futures/VIX.json"
        
        # Cabeceras completas para simular un navegador real al 100%
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
            "Referer": "https://www.cboe.com/indices/vix/vix_futures/",
            "Origin": "https://www.cboe.com"
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        # Esto lanzará una excepción si hay un error 403 o 404
        response.raise_for_status()
        
        data = response.json()
        
        if 'data' in data and 'futures' in data['data']:
            df = pd.DataFrame(data['data']['futures'])
            df['expiry'] = pd.to_datetime(df['expiration_date'])
            df = df.sort_values('expiry').reset_index(drop=True)
            
            # Seleccionamos los primeros 8 meses (F1 a F8)
            df = df.head(8)
            df['label'] = [f"F{i+1}" for i in range(len(df))]
            return df
        return None
    except Exception as e:
        # Mostramos el error específico para depuración
        st.error(f"Error al conectar con CBOE: {e}")
        return None

# ==============================================================================
# FUNCIÓN PRINCIPAL (CONTENIDO DE LA APP)
# ==============================================================================
def main_vix_structure():
    st.markdown("<h1><span style='font-size: 1.5em;'>📈</span> VIX Term Structure</h1>", unsafe_allow_html=True)
    st.markdown("Datos oficiales de futuros provenientes de CBOE (Estilo VIXCentral).")
    st.divider()

    with st.spinner('Obteniendo datos de mercado...'):
        df = get_vix_futures_data()
        spot_price = get_vix_spot()

    if df is not None:
        # --- TABLA DE DIFERENCIAS (ESTILO VIX CENTRAL) ---
        st.subheader("Tablas de Diferencia (Contango / Backwardation)")
        
        diff_data = []
        
        # 1. Diferencia VIX Spot vs F1
        if spot_price:
            diff_s = df.iloc[0]['price'] - spot_price
            pct_s = (diff_s / spot_price) * 100
            diff_data.append({
                "Meses": "Spot - F1",
                "Diferencia ($)": round(diff_s, 3),
                "Contango %": f"{round(pct_s, 2)}%"
            })

        # 2. Diferencias entre meses consecutivos
        for i in range(len(df) - 1):
            m1 = df.iloc[i]
            m2 = df.iloc[i+1]
            diff = m2['price'] - m1['price']
            pct = (diff / m1['price']) * 100
            
            diff_data.append({
                "Meses": f"{m1['label']} - {m2['label']}",
                "Diferencia ($)": round(diff, 3),
                "Contango %": f"{round(pct, 2)}%"
            })
        
        # Métrica Contango F1-F2 destacada
        idx_f1f2 = 1 if spot_price else 0
        f1_f2_val = float(diff_data[idx_f1f2]['Contango %'].replace('%',''))
        
        st.metric("Contango F1-F2", f"{f1_f2_val}%", delta=f"{f1_f2_val}%")
        
        # Mostrar tabla estática
        st.table(pd.DataFrame(diff_data))
        st.divider()

        # --- GRÁFICO DE ESTRUCTURA DE PLAZOS ---
        st.subheader("VIX Futures Term Structure")

        

        fig = go.Figure()

        # Línea de Futuros (F1-F8)
        fig.add_trace(go.Scatter(
            x=df['label'], y=df['price'],
            mode='lines+markers',
            name='VIX Futures',
            line=dict(color='#1f77b4', width=4),
            marker=dict(size=12, color='#1f77b4', line=dict(width=2, color='white')),
            hovertemplate="<b>%{x}</b><br>Precio: %{y}<extra></extra>"
        ))

        # Punto Spot (Índice VIX)
        if spot_price:
            fig.add_trace(go.Scatter(
                x=["Spot"], y=[spot_price],
                mode='markers',
                name='VIX Spot',
                marker=dict(color='yellow', size=14, symbol='diamond'),
                hovertemplate="<b>VIX Index</b><br>Precio: %{y:.2f}<extra></extra>"
            ))

        fig.update_layout(
            template="plotly_dark", 
            height=500,
            xaxis=dict(title="Mes del Futuro", gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(title="Precio ($)", gridcolor='rgba(255,255,255,0.1)'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PUNTO DE ENTRADA PROTEGIDO
# ==============================================================================
if __name__ == "__main__":
    
    if check_password():
        main_vix_structure()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder a VIX Term Structure.")
