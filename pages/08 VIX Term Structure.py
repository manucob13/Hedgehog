import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="VIX Term Structure - CBOE", layout="wide")

# ==============================================================================
# SISTEMA DE AUTENTICACIÓN (EN EL SIDEBAR)
# ==============================================================================
def check_password():
    """Maneja la autenticación desde el sidebar."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    # Estructura de login en el lateral
    with st.sidebar:
        st.title("🔒 Acceso")
        pwd = st.text_input("Introduce la contraseña", type="password")
        if st.button("Ingresar"):
            if pwd == st.secrets["password"]:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("⚠️ Contraseña incorrecta")
    
    # Si no está autenticado, muestra mensaje en el cuerpo principal
    st.title("🔒 Acceso Restringido")
    st.info("Por favor, introduce la contraseña en el menú lateral para acceder.")
    return False

# ==============================================================================
# OBTENCIÓN DE DATOS CBOE
# ==============================================================================
@st.cache_data(ttl=300)
def get_vix_futures_data():
    """Descarga datos de futuros directamente de CBOE."""
    try:
        url = "https://cdn.cboe.com/api/global/delayed_quotes/futures/VIX.json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'data' in data and 'futures' in data['data']:
            df = pd.DataFrame(data['data']['futures'])
            # Convertir fecha y ordenar
            df['expiry'] = pd.to_datetime(df['expiration_date'])
            df = df.sort_values('expiry').reset_index(drop=True)
            # Tomamos los primeros 8 meses (F1 a F8)
            df = df.head(8)
            df['label'] = [f"F{i+1}" for i in range(len(df))]
            return df
        return None
    except Exception as e:
        st.error(f"Error descargando datos de CBOE: {e}")
        return None

# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================
def main_vix_central():
    st.markdown("<h1><span style='font-size: 1.5em;'>📈</span> VIX Term Structure</h1>", unsafe_allow_html=True)
    st.markdown("Réplica de **VIXCentral.com** utilizando datos oficiales de CBOE.")
    st.divider()

    df = get_vix_futures_data()

    if df is not None:
        # --- SECCIÓN 1: TABLA DE DIFERENCIAS (SPREADS) ---
        st.subheader("Tablas de Diferencia (Contango / Backwardation)")
        
        diff_list = []
        for i in range(len(df) - 1):
            f_current = df.iloc[i]
            f_next = df.iloc[i+1]
            diff_val = f_next['price'] - f_current['price']
            contango_pct = (diff_val / f_current['price']) * 100
            
            diff_list.append({
                "Meses": f"{f_current['label']} - {f_next['label']}",
                "Diferencia ($)": round(diff_val, 3),
                "Contango %": f"{round(contango_pct, 2)}%"
            })
        
        df_diff = pd.DataFrame(diff_list)
        
        # Métrica destacada F1-F2
        f1_f2_val = float(df_diff.iloc[0]['Contango %'].replace('%',''))
        st.metric("Contango F1-F2", f"{f1_f2_val}%", delta=f"{f1_f2_val}%")
        
        st.table(df_diff)
        st.divider()

        # --- SECCIÓN 2: GRÁFICO ---
        st.subheader("VIX Futures Curve")
        
        fig = go.Figure()
        
        # Añadir curva de futuros
        fig.add_trace(go.Scatter(
            x=df['label'],
            y=df['price'],
            mode='lines+markers',
            name='Curva Actual',
            line=dict(color='#1f77b4', width=4),
            marker=dict(size=12, color='#1f77b4', line=dict(width=2, color='white')),
            hovertemplate="<b>%{x}</b><br>Precio: %{y}<extra></extra>"
        ))

        fig.update_layout(
            template="plotly_dark",
            height=600,
            xaxis=dict(title="Mes del Futuro", gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(title="Precio", gridcolor='rgba(255,255,255,0.1)'),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # --- SECCIÓN 3: TABLA DE DATOS CRUDOS ---
        with st.expander("Ver detalle de contratos"):
            st.dataframe(df[['label', 'symbol', 'expiration_date', 'price', 'volume']], 
                         hide_index=True, use_container_width=True)

# --- EJECUCIÓN ---
if __name__ == "__main__":
    # Si el password es correcto, ejecuta la app
    if check_password():
        main_vix_central()
