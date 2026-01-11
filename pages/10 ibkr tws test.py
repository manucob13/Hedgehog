# test_ibkr.py
import streamlit as st
import asyncio
import sys

st.set_page_config(page_title="Test IBKR", page_icon="🔌")

st.title("🔌 Test Conexión IBKR")

# Configuración
col1, col2 = st.columns(2)
with col1:
    port = st.number_input("Puerto", value=5000, step=1)
with col2:
    client_id = st.number_input("Client ID", value=999, step=1)

# Botón de prueba
if st.button("🧪 Probar Conexión", type="primary"):
    
    try:
        # Configurar event loop ANTES de importar ib_insync
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
        except RuntimeError:
            if sys.platform == 'win32':
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        from ib_insync import IB
        
        with st.spinner("Conectando..."):
            ib = IB()
            ib.connect('127.0.0.1', port, clientId=client_id, timeout=5)
            
            if ib.isConnected():
                cuentas = ib.managedAccounts()
                st.success(f"✅ CONECTADO!")
                st.write(f"**Puerto:** {port}")
                st.write(f"**Cuentas:** {cuentas}")
                ib.disconnect()
            else:
                st.error("❌ No se pudo conectar")
                
    except ImportError:
        st.error("❌ ib_insync no está instalado. Ejecuta: pip install ib-insync")
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.info("""
        **Verifica:**
        - TWS está ejecutándose
        - API habilitada (File → Global Config → API → Settings)
        - Puerto correcto en TWS
        """)
