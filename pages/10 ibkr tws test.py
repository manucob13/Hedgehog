# test_streamlit_ibkr.py
import streamlit as st
from utils_ibkr import test_ibkr_connection, send_strategy_order_ibkr
import pandas as pd

# Configuración de la página
st.set_page_config(
    page_title="Test IBKR Connection",
    page_icon="🔌",
    layout="wide"
)

# Título
st.title("🔌 Test de Conexión IBKR")
st.markdown("---")

# ==============================================================================
# SECCIÓN 1: CONFIGURACIÓN
# ==============================================================================
st.header("1. Configuración de Conexión")

col1, col2, col3 = st.columns(3)

with col1:
    host = st.text_input(
        "Host",
        value="127.0.0.1",
        help="Dirección IP de TWS/Gateway"
    )

with col2:
    port = st.number_input(
        "Puerto",
        min_value=1,
        max_value=65535,
        value=5000,
        step=1,
        help="Puerto de TWS/Gateway"
    )

with col3:
    client_id = st.number_input(
        "Client ID",
        min_value=1,
        max_value=9999,
        value=999,
        step=1,
        help="ID único del cliente"
    )

st.info(f"📝 Configuración actual: **{host}:{port}** | Client ID: **{client_id}**")

st.markdown("---")

# ==============================================================================
# SECCIÓN 2: TEST DE CONEXIÓN
# ==============================================================================
st.header("2. Prueba de Conexión")

col1, col2 = st.columns([1, 2])

with col1:
    if st.button("🧪 Probar Conexión", type="primary", use_container_width=True):
        with st.spinner("Conectando a IBKR..."):
            result = test_ibkr_connection(
                host=host,
                port=port,
                client_id=client_id
            )
            
            # Guardar resultado en session_state
            st.session_state.connection_result = result

with col2:
    st.markdown("""
    **Instrucciones:**
    1. Asegúrate de que TWS/Gateway esté ejecutándose
    2. Verifica que la API esté habilitada
    3. Haz clic en "Probar Conexión"
    """)

# Mostrar resultado de la conexión
if 'connection_result' in st.session_state:
    result = st.session_state.connection_result
    
    st.markdown("### 📊 Resultado de la Prueba")
    
    if result['success']:
        st.success(f"✅ {result['message']}")
        
        st.markdown("**📋 Información de la Cuenta:**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Estado", "CONECTADO ✅")
        with col2:
            st.metric("Cuentas", len(result['accounts']))
        
        if result['accounts']:
            st.write("**Cuentas disponibles:**")
            for acc in result['accounts']:
                st.write(f"- {acc}")
    else:
        st.error(f"❌ {result['message']}")
        
        st.markdown("### 🔍 Posibles Soluciones")
        
        with st.expander("📖 Ver soluciones comunes"):
            st.markdown("""
            **Si ves el error: `[Errno 111] Connect call failed`**
            
            1. **Verifica que TWS esté ejecutándose:**
               - Abre TWS
               - Verifica que esté conectado (no solo abierto)
            
            2. **Habilita la API en TWS:**
               - Ve a: File → Global Configuration → API → Settings
               - Marca: ✅ Enable ActiveX and Socket Clients
               - Verifica el Socket Port (debe ser el mismo que configuraste arriba)
               - Aplica los cambios
            
            3. **Verifica el puerto:**
               - Prueba con diferentes puertos:
                 - 5000 (tu puerto personalizado)
                 - 7497 (TWS Paper Trading)
                 - 7496 (TWS Live Trading)
                 - 4002 (Gateway Paper)
                 - 4001 (Gateway Live)
            
            4. **Verifica el Client ID:**
               - Si tienes otras conexiones activas, usa un Client ID diferente
               - Prueba con: 1, 2, 100, 999, etc.
            
            5. **Reinicia TWS:**
               - Cierra TWS completamente
               - Vuelve a abrirlo
               - Intenta conectar nuevamente
            """)

st.markdown("---")

# ==============================================================================
# SECCIÓN 3: TEST DE PUERTOS (OPCIONAL)
# ==============================================================================
st.header("3. Escáner de Puertos (Opcional)")

st.markdown("""
Si no estás seguro de qué puerto usar, esta herramienta probará los puertos más comunes.
""")

if st.button("🔍 Escanear Puertos Comunes", use_container_width=True):
    
    puertos_comunes = [
        (5000, "Puerto Personalizado"),
        (7497, "TWS Paper Trading"),
        (7496, "TWS Live Trading"),
        (4002, "Gateway Paper Trading"),
        (4001, "Gateway Live Trading"),
    ]
    
    st.markdown("### Resultados del Escaneo")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    found = False
    
    for i, (puerto, descripcion) in enumerate(puertos_comunes):
        status_text.text(f"Probando {descripcion} (puerto {puerto})...")
        
        result = test_ibkr_connection(
            host=host,
            port=puerto,
            client_id=client_id
        )
        
        if result['success']:
            st.success(f"✅ **{descripcion}** - Puerto {puerto} funciona!")
            st.write(f"Cuentas: {result['accounts']}")
            found = True
            break
        else:
            st.warning(f"❌ **{descripcion}** - Puerto {puerto} no responde")
        
        progress_bar.progress((i + 1) / len(puertos_comunes))
    
    status_text.empty()
    progress_bar.empty()
    
    if not found:
        st.error("❌ No se encontró ningún puerto funcionando. Verifica que TWS/Gateway esté ejecutándose.")

st.markdown("---")

# ==============================================================================
# SECCIÓN 4: TEST DE ORDEN (SOLO SI CONEXIÓN EXITOSA)
# ==============================================================================
if 'connection_result' in st.session_state and st.session_state.connection_result['success']:
    
    st.header("4. Prueba de Orden (Opcional)")
    
    st.warning("⚠️ Esta sección enviará una orden REAL a IBKR. Úsala con precaución.")
    
    with st.expander("🧪 Configurar Orden de Prueba"):
        
        st.markdown("### Configuración de Orden Simple")
        
        col1, col2 = st.columns(2)
        
        with col1:
            symbol_test = st.text_input("Símbolo", value="QQQ")
            expiry_test = st.date_input("Fecha de Expiración")
            strike_test = st.number_input("Strike", min_value=0.0, value=500.0, step=5.0)
        
        with col2:
            right_test = st.selectbox("Tipo", ["P", "C"])
            quantity_test = st.number_input("Cantidad", min_value=1, value=1)
            limit_price_test = st.number_input("Precio Límite", min_value=0.0, value=0.01, step=0.01)
        
        st.markdown("---")
        
        # Crear DataFrame de prueba
        if st.button("📋 Generar Vista Previa", use_container_width=True):
            
            expiry_str = expiry_test.strftime("%Y-%m-%d")
            
            df_test = pd.DataFrame([
                {
                    'Action': 'BUY',
                    'Quantity': quantity_test,
                    'Symbol': symbol_test,
                    'SecType': 'OPT',
                    'Expiry': expiry_str,
                    'Strike': int(strike_test),
                    'Right': right_test,
                    'Exchange': 'SMART',
                    'Currency': 'USD'
                }
            ])
            
            st.session_state.df_test_order = df_test
            
            st.success("✅ Vista previa generada")
        
        if 'df_test_order' in st.session_state:
            st.markdown("### 📋 Vista Previa de la Orden")
            st.dataframe(st.session_state.df_test_order, use_container_width=True)
            
            st.markdown("---")
            
            st.error("🚨 **ADVERTENCIA:** Esta acción enviará una orden REAL a IBKR")
            
            confirm = st.checkbox("Entiendo que esto enviará una orden real")
            
            if confirm:
                if st.button("🚀 ENVIAR ORDEN DE PRUEBA", type="primary", use_container_width=True):
                    
                    with st.spinner("Enviando orden a IBKR..."):
                        
                        result = send_strategy_order_ibkr(
                            df_strategy=st.session_state.df_test_order,
                            limit_price=limit_price_test,
                            host=host,
                            port=port,
                            client_id=client_id,
                            quantity=quantity_test,
                            tif='DAY',
                            action='BUY',
                            timeout=10
                        )
                    
                    if result['success']:
                        st.success(f"✅ {result['message']}")
                        if result['order_id']:
                            st.info(f"📋 Order ID: {result['order_id']}")
                    else:
                        st.error(f"❌ {result['message']}")

else:
    st.info("💡 Primero conecta exitosamente en la Sección 2 para habilitar las pruebas de órdenes")

st.markdown("---")

# ==============================================================================
# SECCIÓN 5: INFORMACIÓN
# ==============================================================================
st.header("5. Información y Ayuda")

with st.expander("📚 Configuración de TWS"):
    st.markdown("""
    ### Cómo habilitar la API en TWS
    
    1. Abre **TWS** (Trader Workstation)
    
    2. Ve al menú: **File → Global Configuration**
    
    3. En el panel izquierdo, selecciona: **API → Settings**
    
    4. Configura lo siguiente:
       - ✅ **Enable ActiveX and Socket Clients**
       - ✅ **Allow connections from localhost only** (para seguridad)
       - ❌ **Read-Only API** (desmarca si quieres enviar órdenes)
       - **Socket Port:** Verifica que sea el número correcto (ej: 5000, 7497, etc.)
       - **Trusted IPs:** Asegúrate que `127.0.0.1` esté en la lista
    
    5. Haz clic en **OK** o **Apply**
    
    6. Reinicia TWS si es necesario
    """)

with st.expander("🔧 Puertos Comunes"):
    st.markdown("""
    | Puerto | Descripción |
    |--------|-------------|
    | 5000   | Tu puerto personalizado |
    | 7497   | TWS Paper Trading (Simulado) |
    | 7496   | TWS Live Trading (Real) |
    | 4002   | Gateway Paper Trading |
    | 4001   | Gateway Live Trading |
    """)

with st.expander("❓ Preguntas Frecuentes"):
    st.markdown("""
    **P: ¿Por qué no se conecta?**
    
    R: Las causas más comunes son:
    - TWS no está ejecutándose
    - API no está habilitada
    - Puerto incorrecto
    - Client ID en uso por otra conexión
    
    **P: ¿Qué Client ID debo usar?**
    
    R: Cualquier número entre 1 y 9999 que no esté en uso. Si tienes dudas, usa 999.
    
    **P: ¿Puedo tener múltiples conexiones?**
    
    R: Sí, pero cada una debe usar un Client ID diferente.
    
    **P: ¿Qué es Paper Trading vs Live?**
    
    R: Paper Trading es simulado (sin dinero real). Live es trading real.
    """)

st.markdown("---")
st.caption("🔌 Test de Conexión IBKR v1.0 | Desarrollado para pruebas")
