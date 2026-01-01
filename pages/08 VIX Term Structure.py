import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime
from utils import check_password

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="VIX Term Structure - CBOE", layout="wide")

# ==============================================================================
# OBTENCIÓN DE DATOS CBOE
# ==============================================================================
@st.cache_data(ttl=300)
def get_vix_futures_data():
    try:
        url = "https://cdn.cboe.com/api/global/delayed_quotes/futures/VIX.json"
        
        # Añadimos cabeceras para evitar el error de bloqueo (JSON Decode Error)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        # Verificamos si la respuesta es correcta antes de intentar leer el JSON
        if response.status_code != 200:
            st.error(f"Error de servidor CBOE: Código {response.status_code}")
            return None
            
        data = response.json()
        
        if 'data' in data and 'futures' in data['data']:
            df = pd.DataFrame(data['data']['futures'])
            df['expiry'] = pd.to_datetime(df['expiration_date'])
            df = df.sort_values('expiry').reset_index(drop=True)
            
            # Filtramos los primeros 7-8 meses como en VIXCentral
            df = df.head(8)
            df['label'] = [f"F{i+1}" for i in range(len(df))]
            return df
        return None
    except Exception as e:
        st.error(f"Error en la conexión o formato de datos: {e}")
        return None

# ==============================================================================
# FUNCIÓN PRINCIPAL (CONTENIDO DE LA APP)
# ==============================================================================
def main_vix_structure():
    st.markdown("<h1><span style='font-size: 1.5em;'>📈</span> VIX Term Structure</h1>", unsafe_allow_html=True)
    st.markdown("Datos oficiales de futuros provenientes de CBOE.")
    st.divider()

    # Spinner para indicar que está cargando
    with st.spinner('Obteniendo datos de CBOE...'):
        df = get_vix_futures_data()

    if df is not None:
        # --- TABLA DE DIFERENCIAS (ESTILO VIX CENTRAL) ---
        st.subheader("Tablas de Diferencia (Contango / Backwardation)")
        
        diff_data = []
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
        
        # Métrica F1-F2 destacada
        if len(diff_data) > 0:
            f1_f2_pct_str = diff_data[0]['Contango %'].replace('%','')
            f1_f2_pct = float(f1_f2_pct_str)
            
            color_delta = "normal" if f1_f2_pct > 0 else "inverse"
            st.metric("Contango F1-F2", f"{f1_f2_pct}%", delta=f"{f1_f2_pct}%", delta_color=color_delta)
        
        st.table(pd.DataFrame(diff_data))
        st.divider()

        # --- GRÁFICO DE 8 MESES ---
        st.subheader("VIX Futures Curve")
        
        # Preparar datos para el gráfico
        fig = go.Figure()
        
        # Añadir la curva de futuros
        fig.add_trace(go.Scatter(
            x=df['label'], 
            y=df['price'],
            mode='lines+markers',
            name='VIX Futures',
            line=dict(color='#1f77b4', width=4),
            marker=dict(size=12, color='#1f77b4', line=dict(width=2, color='white')),
            hovertemplate="<b>%{x}</b><br>Precio: %{y}<extra></extra>"
        ))

        fig.update_layout(
            template="plotly_dark", 
            height=550,
            xaxis=dict(
                title="Future Month", 
                gridcolor='rgba(255,255,255,0.1)',
                showline=True,
                linewidth=1,
                linecolor='white'
            ),
            yaxis=dict(
                title="Price", 
                gridcolor='rgba(255,255,255,0.1)',
                showline=True,
                linewidth=1,
                linecolor='white'
            ),
            margin=dict(l=40, r=40, t=20, b=40),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PUNTO DE ENTRADA PROTEGIDO (SEGÚN TU BASE)
# ==============================================================================
if __name__ == "__main__":
    
    if check_password():
        main_vix_structure()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder a VIX Term Structure.")
