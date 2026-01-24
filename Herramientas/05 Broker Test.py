# Mostrar resumen
            secrets_ok = all([
                results['has_api_key'],
                results['has_app_secret'],
                results['has_redirect_uri'],
                results['redirect_uri_valid'],
                results['has_token']
            ])
            
            if secrets_ok:
                st.success("✅ Todos los secrets están configurados correctamente")
            else:
                st.error("❌ Faltan algunos secrets críticos o tienen formato incorrecto")
            
            # Detalles
            col_# pages/schwab_debug.py
import streamlit as st
import os
import json
from datetime import datetime
from utils.utils import check_password
from utils.utils_schwab import (
    connect_to_schwab,
    get_schwab_credentials,
    setup_token_from_secrets,
    get_current_price_schwab,
    normalize_ticker
)

st.set_page_config(
    page_title="🔌 Broker Debug",
    page_icon="🔌",
    layout="wide"
)

def initialize_session_state():
    """Inicializa variables de session_state para debug."""
    if 'schwab_debug_client' not in st.session_state:
        st.session_state.schwab_debug_client = None
    if 'connection_log' not in st.session_state:
        st.session_state.connection_log = []
    if 'last_test_time' not in st.session_state:
        st.session_state.last_test_time = None


def log_message(message, level="INFO"):
    """
    Añade un mensaje al log con timestamp.
    
    Args:
        message (str): Mensaje a registrar
        level (str): Nivel del log (INFO, SUCCESS, WARNING, ERROR)
    """
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log_entry = {
        'timestamp': timestamp,
        'level': level,
        'message': message
    }
    st.session_state.connection_log.append(log_entry)


def display_log():
    """Muestra el log de conexión con colores según el nivel."""
    if st.session_state.connection_log:
        st.markdown("### 📋 Log de Conexión")
        
        log_container = st.container()
        with log_container:
            for entry in reversed(st.session_state.connection_log[-50:]):  # Últimos 50 mensajes
                timestamp = entry['timestamp']
                level = entry['level']
                message = entry['message']
                
                if level == "ERROR":
                    st.markdown(f"🔴 `{timestamp}` **[ERROR]** {message}")
                elif level == "WARNING":
                    st.markdown(f"🟡 `{timestamp}` **[WARNING]** {message}")
                elif level == "SUCCESS":
                    st.markdown(f"🟢 `{timestamp}` **[SUCCESS]** {message}")
                else:  # INFO
                    st.markdown(f"ℹ️ `{timestamp}` **[INFO]** {message}")


def check_secrets_configuration():
    """
    Verifica la configuración de secrets de Schwab.
    
    Returns:
        dict: Diccionario con resultados de la verificación
    """
    results = {
        'has_api_key': False,
        'has_app_secret': False,
        'has_redirect_uri': False,
        'redirect_uri_valid': False,
        'redirect_uri_value': None,
        'has_token': False,
        'token_fields': []
    }
    
    try:
        # Verificar credenciales principales
        if 'schwab' in st.secrets:
            log_message("✅ Sección 'schwab' encontrada en secrets", "SUCCESS")
            
            if 'api_key' in st.secrets['schwab']:
                results['has_api_key'] = True
                api_key_preview = st.secrets['schwab']['api_key'][:8] + "..." if len(st.secrets['schwab']['api_key']) > 8 else "***"
                log_message(f"✅ API Key encontrada: {api_key_preview}", "SUCCESS")
            else:
                log_message("❌ API Key NO encontrada en secrets", "ERROR")
            
            if 'app_secret' in st.secrets['schwab']:
                results['has_app_secret'] = True
                log_message("✅ App Secret encontrada", "SUCCESS")
            else:
                log_message("❌ App Secret NO encontrada en secrets", "ERROR")
            
            if 'redirect_uri' in st.secrets['schwab']:
                results['has_redirect_uri'] = True
                redirect_uri = st.secrets['schwab']['redirect_uri']
                results['redirect_uri_value'] = redirect_uri
                log_message(f"✅ Redirect URI encontrada: {redirect_uri}", "SUCCESS")
                
                # Validar formato del redirect_uri
                if ':' in redirect_uri and redirect_uri.count(':') >= 2:
                    # Tiene el formato correcto con puerto (ej: https://127.0.0.1:8182)
                    results['redirect_uri_valid'] = True
                    log_message("✅ Redirect URI tiene formato válido (incluye puerto)", "SUCCESS")
                else:
                    results['redirect_uri_valid'] = False
                    log_message("❌ Redirect URI SIN PUERTO - debe incluir puerto (ej: https://127.0.0.1:8182)", "ERROR")
                    log_message(f"  ℹ️ Formato actual: {redirect_uri}", "WARNING")
                    log_message("  ℹ️ Formato correcto: https://127.0.0.1:8182", "INFO")
            else:
                log_message("❌ Redirect URI NO encontrada en secrets", "ERROR")
            
            # Verificar token
            if 'token' in st.secrets['schwab']:
                results['has_token'] = True
                log_message("✅ Sección 'token' encontrada", "SUCCESS")
                
                token_fields = ['creation_timestamp', 'expires_in', 'token_type', 'scope', 
                               'refresh_token', 'access_token', 'id_token', 'expires_at']
                
                for field in token_fields:
                    if field in st.secrets['schwab']['token']:
                        results['token_fields'].append(field)
                        if field == 'access_token':
                            token_preview = st.secrets['schwab']['token'][field][:10] + "..." if len(st.secrets['schwab']['token'][field]) > 10 else "***"
                            log_message(f"  ✅ {field}: {token_preview}", "INFO")
                        elif field == 'expires_at':
                            expires_at = st.secrets['schwab']['token'][field]
                            log_message(f"  ✅ {field}: {expires_at}", "INFO")
                        else:
                            log_message(f"  ✅ {field}: configurado", "INFO")
                    else:
                        log_message(f"  ❌ {field}: FALTANTE", "ERROR")
            else:
                log_message("❌ Sección 'token' NO encontrada en secrets", "ERROR")
        else:
            log_message("❌ Sección 'schwab' NO encontrada en secrets", "ERROR")
    
    except Exception as e:
        log_message(f"❌ Error verificando secrets: {str(e)}", "ERROR")
    
    return results


def check_token_file():
    """
    Verifica la existencia y contenido del archivo schwab_token.json.
    
    Returns:
        dict: Información sobre el archivo token
    """
    token_path = "schwab_token.json"
    results = {
        'exists': False,
        'readable': False,
        'valid_json': False,
        'has_token_field': False,
        'token_data': None
    }
    
    try:
        if os.path.exists(token_path):
            results['exists'] = True
            log_message(f"✅ Archivo {token_path} existe", "SUCCESS")
            
            try:
                with open(token_path, 'r') as f:
                    token_data = json.load(f)
                    results['readable'] = True
                    results['valid_json'] = True
                    results['token_data'] = token_data
                    log_message("✅ Archivo token es JSON válido", "SUCCESS")
                    
                    if 'token' in token_data:
                        results['has_token_field'] = True
                        log_message("✅ Campo 'token' encontrado en archivo", "SUCCESS")
                        
                        if 'expires_at' in token_data['token']:
                            expires_at = token_data['token']['expires_at']
                            log_message(f"  📅 Token expira en: {expires_at}", "INFO")
                        
                        if 'creation_timestamp' in token_data:
                            creation = token_data['creation_timestamp']
                            log_message(f"  📅 Token creado en: {creation}", "INFO")
                    else:
                        log_message("❌ Campo 'token' NO encontrado en archivo", "ERROR")
                        
            except json.JSONDecodeError as e:
                log_message(f"❌ Error leyendo JSON: {str(e)}", "ERROR")
            except Exception as e:
                log_message(f"❌ Error leyendo archivo: {str(e)}", "ERROR")
        else:
            log_message(f"⚠️ Archivo {token_path} NO existe", "WARNING")
            log_message("  ℹ️ Se creará automáticamente desde secrets al conectar", "INFO")
    
    except Exception as e:
        log_message(f"❌ Error verificando archivo token: {str(e)}", "ERROR")
    
    return results


def test_schwab_connection():
    """
    Prueba la conexión completa con Schwab y reporta detalles.
    
    Returns:
        bool: True si la conexión fue exitosa
    """
    import threading
    import time
    
    log_message("🚀 Iniciando prueba de conexión con Schwab...", "INFO")
    
    try:
        # 1. Verificar credenciales
        log_message("1️⃣ Obteniendo credenciales...", "INFO")
        api_key, app_secret, redirect_uri = get_schwab_credentials()
        
        if not all([api_key, app_secret, redirect_uri]):
            log_message("❌ Credenciales incompletas", "ERROR")
            return False
        
        log_message(f"✅ Credenciales obtenidas: API Key: {api_key[:8]}...", "SUCCESS")
        log_message(f"✅ Redirect URI: {redirect_uri}", "SUCCESS")
        
        # 2. Verificar/crear archivo token
        log_message("2️⃣ Verificando archivo token...", "INFO")
        token_setup = setup_token_from_secrets()
        
        if not token_setup:
            log_message("❌ No se pudo configurar el archivo token", "ERROR")
            return False
        
        log_message("✅ Archivo token configurado", "SUCCESS")
        
        # 3. Intentar conexión con timeout
        log_message("3️⃣ Conectando con Schwab API...", "INFO")
        log_message("  ⏳ Esperando respuesta del servidor (timeout: 30s)...", "INFO")
        
        # Variable para almacenar resultado
        result_container = {'client': None, 'error': None, 'completed': False}
        
        def connect_with_timeout():
            try:
                log_message("  🔄 Llamando a connect_to_schwab()...", "INFO")
                client = connect_to_schwab()
                result_container['client'] = client
                result_container['completed'] = True
                log_message("  ✅ connect_to_schwab() completado", "SUCCESS")
            except Exception as e:
                result_container['error'] = e
                result_container['completed'] = True
                log_message(f"  ❌ Error en connect_to_schwab(): {str(e)}", "ERROR")
        
        # Ejecutar conexión en thread separado
        thread = threading.Thread(target=connect_with_timeout)
        thread.daemon = True
        thread.start()
        
        # Esperar con timeout
        timeout = 30
        start_time = time.time()
        while time.time() - start_time < timeout:
            if result_container['completed']:
                break
            time.sleep(0.5)
        
        if not result_container['completed']:
            log_message("❌ TIMEOUT: La conexión tardó más de 30 segundos", "ERROR")
            log_message("  ℹ️ Esto puede indicar que el redirect_uri está mal configurado", "WARNING")
            log_message("  ℹ️ O que el token ha expirado y necesita renovación manual", "WARNING")
            return False
        
        if result_container['error']:
            log_message(f"❌ Error durante la conexión: {str(result_container['error'])}", "ERROR")
            return False
        
        client = result_container['client']
        
        if client is None:
            log_message("❌ Fallo al crear cliente de Schwab (retornó None)", "ERROR")
            return False
        
        log_message("✅ Cliente de Schwab creado exitosamente", "SUCCESS")
        st.session_state.schwab_debug_client = client
        
        # 4. Prueba básica - obtener precio de SPX
        log_message("4️⃣ Probando API con ticker SPX...", "INFO")
        test_ticker = "SPX"
        
        try:
            price = get_current_price_schwab(client, test_ticker)
            
            if price is None:
                log_message(f"⚠️ No se pudo obtener precio de {test_ticker}", "WARNING")
                log_message("  ℹ️ La conexión funciona pero puede haber problemas con permisos de mercado", "INFO")
                return True  # Conexión OK aunque no se pudo obtener precio
            
            log_message(f"✅ Precio de {test_ticker}: ${price:.2f}", "SUCCESS")
            log_message("🎉 CONEXIÓN EXITOSA - Todos los tests pasaron", "SUCCESS")
            
            return True
        except Exception as e:
            log_message(f"❌ Error obteniendo precio: {str(e)}", "ERROR")
            log_message("  ℹ️ La conexión puede estar OK, pero hay un problema con la API", "WARNING")
            return True  # Cliente creado OK
        
    except Exception as e:
        log_message(f"❌ Error durante la conexión: {str(e)}", "ERROR")
        log_message(f"  📝 Tipo de error: {type(e).__name__}", "ERROR")
        import traceback
        log_message(f"  📋 Traceback: {traceback.format_exc()}", "ERROR")
        return False


# ==============================================================================
# PUNTO DE ENTRADA PROTEGIDO
# ==============================================================================
if check_password():
    
    initialize_session_state()
    
    st.markdown(
        "<h1><span style='font-size: 1.5em;'>🔌</span> Broker Connection Debug</h1>", 
        unsafe_allow_html=True
    )
    
    st.markdown("""
    Herramienta de diagnóstico para verificar la conexión con diferentes brokers.
    Verifica configuración de secrets, archivos token y realiza pruebas de conexión completas.
    """)
    
    st.markdown("---")
    
    # ==============================================================================
    # SECCIÓN 1: BROKER SCHWAB
    # ==============================================================================
    
    st.header("1. 🔌 Broker Schwab")
    
    st.markdown("""
    Diagnóstico completo de conexión con Schwab API.
    """)
    
    # Verificación de Configuración
    st.subheader("⚙️ Verificación de Configuración")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Verificar Secrets", use_container_width=True, key="verify_secrets_schwab"):
            st.session_state.connection_log = []  # Limpiar log
            log_message("=" * 60, "INFO")
            log_message("VERIFICACIÓN DE SECRETS DE SCHWAB", "INFO")
            log_message("=" * 60, "INFO")
            
            results = check_secrets_configuration()
            
            st.markdown("### 📋 Resultados de Secrets")
            
            # Mostrar resumen
            secrets_ok = all([
                results['has_api_key'],
                results['has_app_secret'],
                results['has_redirect_uri'],
                results['has_token']
            ])
            
            if secrets_ok:
                st.success("✅ Todos los secrets están configurados correctamente")
            else:
                st.error("❌ Faltan algunos secrets críticos")
            
            # Detalles
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("**Credenciales:**")
                st.write(f"{'✅' if results['has_api_key'] else '❌'} API Key")
                st.write(f"{'✅' if results['has_app_secret'] else '❌'} App Secret")
                st.write(f"{'✅' if results['has_redirect_uri'] else '❌'} Redirect URI")
            
            with col_b:
                st.markdown("**Token:**")
                st.write(f"{'✅' if results['has_token'] else '❌'} Sección Token")
                st.write(f"Campos: {len(results['token_fields'])}/8")
    
    with col2:
        if st.button("📄 Verificar Archivo Token", use_container_width=True, key="verify_token_schwab"):
            st.session_state.connection_log = []  # Limpiar log
            log_message("=" * 60, "INFO")
            log_message("VERIFICACIÓN DE ARCHIVO TOKEN", "INFO")
            log_message("=" * 60, "INFO")
            
            results = check_token_file()
            
            st.markdown("### 📋 Resultados del Archivo")
            
            if results['exists']:
                if results['has_token_field']:
                    st.success("✅ Archivo token correcto y completo")
                    
                    if results['token_data']:
                        with st.expander("📊 Ver Detalles del Token"):
                            # Mostrar solo información no sensible
                            safe_data = {
                                'creation_timestamp': results['token_data'].get('creation_timestamp', 'N/A'),
                                'expires_in': results['token_data']['token'].get('expires_in', 'N/A'),
                                'token_type': results['token_data']['token'].get('token_type', 'N/A'),
                                'expires_at': results['token_data']['token'].get('expires_at', 'N/A')
                            }
                            st.json(safe_data)
                else:
                    st.warning("⚠️ Archivo existe pero estructura incorrecta")
            else:
                st.info("ℹ️ Archivo no existe - se creará al conectar")
    
    st.markdown("---")
    
    # Test de Conexión
    st.subheader("🚀 Test de Conexión")
    
    st.markdown("""
    Ejecuta un test completo de conexión que incluye:
    1. Verificación de credenciales
    2. Configuración de archivo token
    3. Creación del cliente de Schwab
    4. Prueba de API con ticker SPX
    """)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("🔌 CONECTAR A SCHWAB", type="primary", use_container_width=True, key="connect_schwab"):
            st.session_state.connection_log = []  # Limpiar log anterior
            st.session_state.last_test_time = datetime.now()
            
            with st.spinner("Probando conexión..."):
                success = test_schwab_connection()
            
            if success:
                st.success("✅ Conexión exitosa con Schwab API")
            else:
                st.error("❌ Fallo en la conexión - revisa el log abajo")
    
    with col2:
        if st.button("🧹 Limpiar Log", use_container_width=True, key="clear_log_schwab"):
            st.session_state.connection_log = []
            st.rerun()
    
    with col3:
        if st.button("🔄 Refrescar", use_container_width=True, key="refresh_schwab"):
            st.rerun()
    
    # Mostrar última vez que se ejecutó el test
    if st.session_state.last_test_time:
        st.caption(f"Último test: {st.session_state.last_test_time.strftime('%H:%M:%S')}")
    
    st.markdown("---")
    
    # Log de Conexión
    if st.session_state.connection_log:
        display_log()
        
        # Botón para descargar log
        log_text = "\n".join([
            f"[{entry['timestamp']}] [{entry['level']}] {entry['message']}"
            for entry in st.session_state.connection_log
        ])
        
        st.download_button(
            label="📥 Descargar Log Completo",
            data=log_text,
            file_name=f"schwab_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            key="download_log_schwab"
        )
    else:
        st.info("ℹ️ No hay logs disponibles. Ejecuta una verificación para ver resultados.")
    
    st.markdown("---")
    
    # Información de Ayuda
    with st.expander("❓ Ayuda - Problemas Comunes"):
        st.markdown("""
        ### Problemas Comunes y Soluciones
        
        #### ❌ "Credenciales incompletas"
        - Verifica que todos los secrets estén configurados en `.streamlit/secrets.toml`
        - Campos requeridos: `api_key`, `app_secret`, `redirect_uri`
        
        #### ❌ "No se pudo configurar el archivo token"
        - Verifica que la sección `[schwab.token]` exista en secrets
        - Todos los campos del token deben estar presentes
        
        #### ❌ "Fallo al crear cliente de Schwab"
        - El token puede haber expirado - genera uno nuevo
        - Verifica que `redirect_uri` coincida con la configuración de Schwab
        - Asegúrate de que la app esté activa en Schwab Developer Portal
        
        #### ⚠️ "No se pudo obtener precio"
        - Verifica que el ticker exista
        - Algunos tickers requieren permisos especiales (ej: índices)
        - El mercado puede estar cerrado
        
        #### 📚 Estructura de secrets.toml
        ```toml
        [schwab]
        api_key = "tu_api_key"
        app_secret = "tu_app_secret"
        redirect_uri = "https://127.0.0.1"
        
        [schwab.token]
        creation_timestamp = 1234567890
        expires_in = 1800
        token_type = "Bearer"
        scope = "api"
        refresh_token = "tu_refresh_token"
        access_token = "tu_access_token"
        id_token = "tu_id_token"
        expires_at = 1234569690
        ```
        
        ### 🔗 Links Útiles
        - [Schwab Developer Portal](https://developer.schwab.com/)
        - [Documentación schwab-py](https://schwab-py.readthedocs.io/)
        """)

else:
    st.title("🔒 Acceso Restringido")
    st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder.")
    st.markdown("""
    ### 🔌 Broker Connection Debug Tool
    
    Esta herramienta te permite:
    - ✅ Verificar la configuración de secrets
    - ✅ Diagnosticar problemas de conexión
    - ✅ Probar la API de diferentes brokers
    - ✅ Ver logs detallados de debugging
    
    **Inicia sesión para comenzar.**
    """)
