"""
Utilidades para interactuar con la API de Schwab
Este módulo contiene funciones para conectar con Schwab y obtener datos de opciones.
"""

import streamlit as st
import os
import json
from schwab.auth import easy_client


def get_schwab_credentials():
    """
    Obtiene las credenciales de Schwab desde st.secrets.
    
    Returns:
        tuple: (api_key, app_secret, redirect_uri) o (None, None, None) si hay error
    """
    try:
        api_key = st.secrets["schwab"]["api_key"]
        app_secret = st.secrets["schwab"]["app_secret"]
        redirect_uri = st.secrets["schwab"]["redirect_uri"]
        return api_key, app_secret, redirect_uri
    except KeyError as e:
        st.error(f"❌ Falta configurar los secrets de Schwab. Clave faltante: {e}")
        return None, None, None


def setup_token_from_secrets(token_path="schwab_token.json"):
    """
    Crea schwab_token.json desde st.secrets si no existe localmente.
    Esto permite trabajar en Streamlit Cloud sin subir el archivo a GitHub.
    
    Args:
        token_path (str): Ruta donde crear el archivo token
    
    Returns:
        bool: True si el token se creó/existe correctamente, False en caso contrario
    """
    # Si el archivo ya existe, no hacer nada
    if os.path.exists(token_path):
        return True
    
    try:
        # Obtener datos del token desde secrets
        token_data = {
            "creation_timestamp": st.secrets["schwab"]["token"]["creation_timestamp"],
            "token": {
                "expires_in": st.secrets["schwab"]["token"]["expires_in"],
                "token_type": st.secrets["schwab"]["token"]["token_type"],
                "scope": st.secrets["schwab"]["token"]["scope"],
                "refresh_token": st.secrets["schwab"]["token"]["refresh_token"],
                "access_token": st.secrets["schwab"]["token"]["access_token"],
                "id_token": st.secrets["schwab"]["token"]["id_token"],
                "expires_at": st.secrets["schwab"]["token"]["expires_at"]
            }
        }
        
        # Crear el archivo JSON
        with open(token_path, "w") as f:
            json.dump(token_data, f, indent=2)
        
        st.success(f"✅ Token recreado desde secrets en: {token_path}")
        return True
        
    except KeyError as e:
        st.error(f"❌ Falta configurar el token en secrets: {e}")
        st.info("""
        **Añade esto a tus Secrets en Streamlit Cloud:**
        
        ```toml
        [schwab.token]
        creation_timestamp = ...
        expires_in = ...
        token_type = "Bearer"
        scope = "api"
        refresh_token = "..."
        access_token = "..."
        id_token = "..."
        expires_at = ...
        ```
        """)
        return False
    except Exception as e:
        st.error(f"❌ Error al crear token desde secrets: {e}")
        return False


def connect_to_schwab(api_key=None, app_secret=None, redirect_uri=None, token_path="schwab_token.json"):
def connect_to_schwab(api_key=None, app_secret=None, redirect_uri=None, token_path="schwab_token.json"):
    """
    Conecta con Schwab usando el token existente.
    Si no existe el token, permite subirlo mediante file_uploader.
    Si no se proporcionan credenciales, las obtiene automáticamente de st.secrets.
    
    Args:
        api_key (str, optional): API Key de Schwab. Si es None, se obtiene de secrets.
        app_secret (str, optional): App Secret de Schwab. Si es None, se obtiene de secrets.
        redirect_uri (str, optional): Redirect URI configurado en Schwab. Si es None, se obtiene de secrets.
        token_path (str): Ruta al archivo token.json (default: "schwab_token.json")
    
    Returns:
        schwab.client.Client: Cliente de Schwab si la conexión es exitosa, None en caso contrario
    """
    # Si no se proporcionan credenciales, obtenerlas de secrets
    if api_key is None or app_secret is None or redirect_uri is None:
        api_key, app_secret, redirect_uri = get_schwab_credentials()
        
        # Si aún son None, no se pudieron obtener
        if api_key is None or app_secret is None or redirect_uri is None:
            return None
    
    # Si no existe el token, permitir subirlo
    if not os.path.exists(token_path):
        st.warning("⚠️ No se encontró schwab_token.json")
        st.info("👇 **Sube el archivo de token que generaste localmente**")
        
        with st.expander("ℹ️ ¿Cómo generar el token?", expanded=False):
            st.markdown("""
            **Ejecuta este script en tu PC local:**
            
            ```python
            from schwab import auth
            
            client = auth.client_from_manual_flow(
                api_key="TU_API_KEY",
                app_secret="TU_SECRET",
                callback_url="https://127.0.0.1",


def obtener_datos_opcion(client, ticker, strike, tipo, fecha_salida):
    """
    Obtiene precio (mid), delta y theta de una opción desde Schwab.
    
    Args:
        client (schwab.client.Client): Cliente de Schwab autenticado
        ticker (str): Símbolo del ticker (ej: 'AAPL', 'SPY')
        strike (float): Precio de ejercicio (strike)
        tipo (str): Tipo de opción ('CALL' o 'PUT')
        fecha_salida (date): Fecha de expiración de la opción
    
    Returns:
        tuple: (mid_price, delta, theta)
            - mid_price (float): Precio medio (bid + ask) / 2, o None si no está disponible
            - delta (float): Delta de la opción, o None si no está disponible
            - theta (float): Theta de la opción, o None si no está disponible
    """
    try:
        if client is None:
            return None, None, None
        
        # Obtener cadena de opciones
        response = client.get_option_chain(ticker)
        if response.status_code != 200:
            return None, None, None
        
        opciones = response.json()
        
        # Seleccionar el mapa correcto según el tipo
        if tipo == 'CALL':
            option_map = opciones.get('callExpDateMap', {})
        else:
            option_map = opciones.get('putExpDateMap', {})
        
        # Formatear fecha para búsqueda
        fecha_str = fecha_salida.strftime('%Y-%m-%d')
        
        # Buscar la clave de fecha que coincida
        fecha_key_match = None
        for key in option_map.keys():
            if key.startswith(fecha_str):
                fecha_key_match = key
                break
        
        if fecha_key_match:
            strikes = option_map[fecha_key_match]
            strike_str = str(float(strike))
            
            if strike_str in strikes:
                # Obtener el primer contrato (usualmente hay uno por strike)
                contrato = strikes[strike_str][0]
                
                # Extraer datos
                bid = contrato.get('bid', 0)
                ask = contrato.get('ask', 0)
                delta = contrato.get('delta', None)
                theta = contrato.get('theta', None)
                
                # Calcular precio medio
                if bid > 0 and ask > 0:
                    mid_price = (bid + ask) / 2
                else:
                    mid_price = None
                
                return mid_price, delta, theta
        
        return None, None, None
        
    except Exception:
        return None, None, None


def obtener_precio_actual(client, ticker):
    """
    Obtiene el precio actual (último precio) de un ticker desde Schwab.
    
    Args:
        client (schwab.client.Client): Cliente de Schwab autenticado
        ticker (str): Símbolo del ticker (ej: 'AAPL', 'SPY')
    
    Returns:
        float: Precio actual del ticker, o None si hay error
    """
    try:
        if client is None:
            return None
        
        response = client.get_quote(ticker)
        if response.status_code != 200:
            return None
        
        data = response.json()
        
        # La estructura puede variar, intentar acceder al precio
        if ticker in data:
            quote_data = data[ticker]
            # Intentar obtener el último precio
            precio = quote_data.get('quote', {}).get('lastPrice')
            if precio is None:
                precio = quote_data.get('lastPrice')
            return precio
        
        return None
        
    except Exception:
        return None


def verificar_conexion_schwab(client):
    """
    Verifica que la conexión con Schwab esté activa.
    
    Args:
        client (schwab.client.Client): Cliente de Schwab a verificar
    
    Returns:
        bool: True si la conexión es válida, False en caso contrario
    """
    try:
        if client is None:
            return False
        
        # Hacer una llamada simple de prueba
        response = client.get_quote("AAPL")
        return hasattr(response, "status_code") and response.status_code == 200
        
    except Exception:
        return False
