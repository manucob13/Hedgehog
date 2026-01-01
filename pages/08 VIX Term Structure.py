import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="VIX Term Structure - CBOE", layout="wide")

# ==============================================================================
# FUNCIÓN DE AUTENTICACIÓN (SIGUIENDO TU EJEMPLO)
# ==============================================================================
def check_password():
    """Retorna True si el usuario ingresó la contraseña correcta."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    # Login en el sidebar como en tu ejemplo
    with st.sidebar:
        st.title("🔒 Acceso")
        password_input = st.text_input("Introduce la contraseña", type="password")
        
        if st.button("Ingresar"):
            # Aquí busca en st.secrets["password"]
            if password_input == st.secrets["password"]:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("⚠️ Contraseña incorrecta")
    return False

# ==============================================================================
# OBTENCIÓN DE DATOS CBOE
# ==============================================================================
@st.cache_data(ttl=300)
def get_vix_futures_data():
    try:
        url = "https://cdn.cboe.com/api/global/delayed_quotes/futures/VIX.json"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if 'data' in data and 'futures' in data['data']:
            df = pd.DataFrame(data['data']['futures'])
            df['expiry'] = pd.to_datetime(df['expiration_date'])
            df = df.sort_values('expiry').reset_index(drop=True)
            # Filtramos los primeros 8 meses como en VIXCentral
            df = df.head(8)
            df['label'] = [f"F{i+1}" for i in range(len(df))]
            return df
        return None
    except Exception as e:
        st.error(f"Error CBOE: {e}")
        return None

# ==============================================================================
# FUNCIÓN PRINCIPAL (CONTENIDO DE LA APP)
# ==============================================================================
def main_vix_structure():
    st.markdown("<h1><span style='font-size: 1.5em;'>📈</span> VIX Term Structure</h1>", unsafe_allow_html=True)
    st.markdown("Datos oficiales de futuros provenientes de CBOE.")
    st.divider()

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
        f1_f2_pct = float(diff_data[0]['Contango %'].replace('%',''))
        st.metric("Contango F1-F2", f"{f1_f2_pct}%", delta=f"{f1_f2_pct}%")
        
        st.table(pd.DataFrame(diff_data))
        st.divider()

        # --- GRÁFICO DE 8 MESES ---
        st.subheader("VIX Futures Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['label'], y=df['price'],
            mode='lines+markers',
            line=dict(color='#1f77b4', width=4),
            marker=dict(size=12, color='#1f77b4', line=dict(width=2, color='white')),
            hovertemplate="<b>%{x}</b><br>Precio: %{y}<extra></extra>"
        ))

        fig.update_layout(
            template="plotly_dark", height=500,
            xaxis=dict(title="Mes del Futuro", gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(title="Precio", gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PUNTO DE ENTRADA PROTEGIDO (IDÉNTICO A TU EJEMPLO)
# ==============================================================================
if __name__ == "__main__":
    # Si el password es correcto (usando la lógica del sidebar)
    if check_password():
        main_vix_structure()
    else:
        # Mensaje de bloqueo igual que en 07 TP Calculos.py
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder.")
