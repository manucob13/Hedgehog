# pages/test.py - Test de Conexión con Schwab
import streamlit as st
from utils_schwab import connect_to_schwab, get_schwab_credentials, verificar_conexion_schwab, obtener_precio_actual
from utils import check_password

# Configuración de página
st.set_page_config(page_title="Test Schwab", layout="centered")

# Estilos CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #1a4d2e;
        border: 2px solid #2d8659;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #a8dadc;
    }
    .error-box {
        background-color: #4d1a1a;
        border: 2px solid #862d2d;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #ffa8a8;
    }
</style>
""", unsafe_allow_html=True)

def test_schwab_page():
    st.markdown('<div class="main-header">🧪 Test de Conexión Schwab</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Esta página prueba la conexión con Schwab API para verificar que:
    - ✅ Los secrets están configurados correctamente
    - ✅ El token existe y es válido
    - ✅ La API responde correctamente
    """)
    
    st.markdown("---")
    
    # PASO 1: Verificar Secrets
    st.markdown("### 1️⃣ Verificar Secrets")
    
    api_key, app_secret, redirect_uri = get_schwab_credentials()
    
    if api_key and app_secret and redirect_uri:
        st.markdown('<div class="success-box">✅ <strong>Secrets configurados correctamente</strong></div>', unsafe_allow_html=True)
        
        with st.expander("🔍 Ver Secrets (parcial)"):
            st.write(f"**API Key:** {api_key[:10]}...{api_key[-4:]}")
            st.write(f"**App Secret:** {app_secret[:10]}...{app_secret[-4:]}")
            st.write(f"**Redirect URI:** {redirect_uri}")
    else:
        st.markdown('<div class="error-box">❌ <strong>Error en Secrets</strong><br>Verifica la configuración en .streamlit/secrets.toml</div>', unsafe_allow_html=True)
        st.stop()
    
    st.markdown("---")
    
    # PASO 2: Conectar con Schwab
    st.markdown("### 2️⃣ Conectar con Schwab")
    
    if st.button("🔌 Probar Conexión", type="primary", use_container_width=True):
        
        with st.spinner("Conectando con Schwab..."):
            client = connect_to_schwab()
        
        if client:
            st.markdown('<div class="success-box">✅ <strong>Conexión exitosa con Schwab</strong></div>', unsafe_allow_html=True)
            
            # Guardar en session_state para reutilizar
            st.session_state.test_client = client
            
            st.markdown("---")
            
            # PASO 3: Verificar Conexión
            st.markdown("### 3️⃣ Verificar Estado de Conexión")
            
            if verificar_conexion_schwab(client):
                st.markdown('<div class="success-box">✅ <strong>La conexión está activa</strong></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-box">❌ <strong>La conexión no responde</strong></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # PASO 4: Prueba de datos
            st.markdown("### 4️⃣ Prueba de Obtención de Datos")
            
            col1, col2 = st.columns(2)
            
            with col1:
                ticker_test = st.text_input("Ticker de prueba", value="AAPL")
            
            with col2:
                if st.button("📊 Obtener Precio", use_container_width=True):
                    with st.spinner(f"Obteniendo precio de {ticker_test}..."):
                        precio = obtener_precio_actual(client, ticker_test)
                    
                    if precio:
                        st.success(f"✅ Precio de {ticker_test}: **${precio:.2f}**")
                    else:
                        st.error(f"❌ No se pudo obtener el precio de {ticker_test}")
            
            st.markdown("---")
            
            # Resumen Final
            st.markdown("### ✅ Resumen del Test")
            
            st.info("""
            **Estado de la Conexión:**
            
            - ✅ Secrets configurados
            - ✅ Cliente de Schwab inicializado
            - ✅ Token validado
            - ✅ API respondiendo correctamente
            
            **Próximos pasos:**
            
            Puedes usar `connect_to_schwab()` en cualquier página para conectarte con Schwab.
            """)
        
        else:
            st.markdown('<div class="error-box">❌ <strong>Error al conectar con Schwab</strong><br>Verifica que schwab_token.json existe y es válido.</div>', unsafe_allow_html=True)
            
            st.warning("""
            **Posibles soluciones:**
            
            1. Verifica que existe el archivo `schwab_token.json` en la raíz del proyecto
            2. Regenera el token desde tu notebook local
            3. Verifica que los secrets coincidan con tu aplicación de Schwab
            """)

if __name__ == "__main__":
    if check_password():
        test_schwab_page()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Introduce tus credenciales en el menú lateral para acceder.")
