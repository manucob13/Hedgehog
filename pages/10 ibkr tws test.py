# test_ibkr.py
import streamlit as st

st.set_page_config(page_title="Test IBKR", page_icon="🔌")

st.title("🔌 Test Conexión IBKR")

# Configuración
col1, col2 = st.columns(2)
with col1:
    port = st.number_input("Puerto", value=5000, step=1)
with col2:
    client_id = st.number_input("Client ID", value=1, step=1)

# Botón de prueba
if st.button("🧪 Probar Conexión", type="primary"):
    
    try:
        # IMPORTANTE: nest_asyncio permite event loops anidados
        import nest_asyncio
        nest_asyncio.apply()
        
        from ib_insync import IB, util
        
        st.info(f"Conectando a 127.0.0.1:{port} con Client ID {client_id}...")
        
        with st.spinner("Conectando..."):
            # IGUAL QUE EN TU JUPYTER NOTEBOOK
            util.startLoop()
            ib = IB()
            ib.connect('127.0.0.1', port, clientId=client_id, timeout=10)
            
            if ib.isConnected():
                cuentas = ib.managedAccounts()
                st.success(f"✅ ¡CONECTADO!")
                st.write(f"**Puerto:** {port}")
                st.write(f"**Client ID:** {client_id}")
                st.write(f"**Cuentas:** {cuentas}")
                ib.disconnect()
                st.success("Conexión cerrada")
            else:
                st.error("❌ No se pudo conectar")
                
    except ImportError as e:
        if 'nest_asyncio' in str(e):
            st.error("❌ Ejecuta: pip install nest-asyncio")
            st.code("pip install nest-asyncio")
        else:
            st.error(f"❌ Error: {e}")
    except Exception as e:
        st.error(f"❌ Error: {e}")
        with st.expander("Ver detalles"):
            import traceback
            st.code(traceback.format_exc())

st.markdown("---")

st.info("""
**Requisitos:**
1. Instala: `pip install nest-asyncio`
2. TWS debe estar ejecutándose
3. API habilitada en TWS
""")
