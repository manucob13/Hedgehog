import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="VIX Central Clone - CBOE Data", layout="wide")

# ==============================================================================
# AUTENTICACIÓN
# ==============================================================================
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated:
        return True

    st.title("🔒 Acceso VIX Central")
    pwd = st.text_input("Contraseña", type="password")
    if st.button("Entrar"):
        if pwd == st.secrets["password"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta")
    return False

# ==============================================================================
# OBTENCIÓN Y PROCESAMIENTO DE DATOS
# ==============================================================================
@st.cache_data(ttl=300)
def get_vix_data():
    try:
        # Datos de Futuros desde CBOE
        url = "https://cdn.cboe.com/api/global/delayed_quotes/futures/VIX.json"
        res = requests.get(url, timeout=10)
        data = res.json()
        
        df = pd.DataFrame(data['data']['futures'])
        df['expiry'] = pd.to_datetime(df['expiration_date'])
        df = df.sort_values('expiry').reset_index(drop=True)
        
        # VIX Central suele mostrar los meses principales (F1, F2, etc.)
        # Filtramos para quedarnos con los primeros 8 contratos
        df = df.head(8)
        df['month_label'] = [f"F{i+1}" for i in range(len(df))]
        
        return df
    except Exception as e:
        st.error(f"Error en CBOE API: {e}")
        return None

# ==============================================================================
# LÓGICA PRINCIPAL
# ==============================================================================
def main():
    st.title("📊 VIX Term Structure (VIX Central Style)")
    st.markdown("---")

    df = get_vix_data()
    
    if df is not None:
        # --- SECCIÓN 1: TABLA DE DIFERENCIAS (SPREADS) ---
        st.subheader("Tablas de Diferencia (Contango / Backwardation)")
        
        diff_data = []
        for i in range(len(df) - 1):
            m1 = df.iloc[i]
            m2 = df.iloc[i+1]
            spread = m2['price'] - m1['price']
            pct = (spread / m1['price']) * 100
            
            diff_data.append({
                "Meses": f"{m1['month_label']} - {m2['month_label']}",
                "Diferencia ($)": round(spread, 3),
                "Contango %": f"{round(pct, 2)}%"
            })
        
        df_diff = pd.DataFrame(diff_data)
        
        # Mostrar métricas clave arriba de la tabla
        f1_f2_pct = float(df_diff.iloc[0]['Contango %'].replace('%', ''))
        col1, col2 = st.columns(2)
        col1.metric("F1 vs F2 (Contango)", f"{f1_f2_pct}%", 
                   delta_color="normal" if f1_f2_pct > 0 else "inverse")
        
        st.table(df_diff) # Usamos st.table para que se vea estático como en la web
        
        st.markdown("---")

        # --- SECCIÓN 2: GRÁFICO DE 8 MESES ---
        st.subheader("VIX Futures Term Structure")

        fig = go.Figure()

        # Línea de la estructura de plazos
        fig.add_trace(go.Scatter(
            x=df['month_label'],
            y=df['price'],
            mode='lines+markers',
            name='Current',
            line=dict(color='#1f77b4', width=4),
            marker=dict(size=12, color='#1f77b4', line=dict(width=2, color='white')),
            text=[f"Exp: {d.strftime('%Y-%m-%d')}" for d in df['expiry']],
            hovertemplate="<b>%{x}</b><br>Precio: %{y}<br>%{text}<extra></extra>"
        ))

        # Estilo visual VIX Central
        fig.update_layout(
            template="plotly_dark",
            height=600,
            margin=dict(l=50, r=50, t=50, b=50),
            xaxis=dict(
                title="Future Month",
                gridcolor='rgba(255,255,255,0.1)',
                zeroline=False
            ),
            yaxis=dict(
                title="Price",
                gridcolor='rgba(255,255,255,0.1)',
                zeroline=False,
                range=[df['price'].min() - 2, df['price'].max() + 2]
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- SECCIÓN 3: DETALLE DE CONTRATOS ---
        with st.expander("Ver detalles de los contratos (Símbolos CBOE)"):
            st.dataframe(df[['month_label', 'symbol', 'expiration_date', 'price', 'volume']], 
                         hide_index=True, use_container_width=True)

# ==============================================================================
# EJECUCIÓN
# ==============================================================================
if __name__ == "__main__":
    if check_password():
        main()
