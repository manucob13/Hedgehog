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
        
        return True
        
    except KeyError as e:
        st.error(f"❌ Falta configurar el token en secrets: {e}")
        return False
    except Exception as e:
        st.error(f"❌ Error al crear token desde secrets: {e}")
        return False


def connect_to_schwab(token_path="schwab_token.json"):
    """
    Conecta con Schwab usando credenciales y token desde st.secrets.
    
    Args:
        token_path (str): Ruta al archivo token.json (default: "schwab_token.json")
    
    Returns:
        schwab.client.Client: Cliente de Schwab si la conexión es exitosa, None en caso contrario
    """
    # Obtener credenciales de secrets
    api_key, app_secret, redirect_uri = get_schwab_credentials()
    
    if api_key is None or app_secret is None or redirect_uri is None:
        return None
    
    # Crear token desde secrets si no existe
    if not setup_token_from_secrets(token_path):
        st.error("❌ No se pudo configurar el token de Schwab")
        return None

    try:
        client = easy_client(
            api_key=api_key,
            app_secret=app_secret,
            callback_url=redirect_uri,
            token_path=token_path
        )

        # Verificar token con una llamada de prueba
        test_response = client.get_quote("AAPL")
        if hasattr(test_response, "status_code") and test_response.status_code != 200:
            raise Exception(f"Respuesta inesperada: {test_response.status_code}")

        return client

    except Exception as e:
        st.error(f"❌ Error al inicializar Schwab Client: {e}")
        return None


def obtener_datos_opcion(client, ticker, strike, tipo, fecha_salida):
    """
    Obtiene precio (mid), delta y theta de una opción desde Schwab.
    
    Args:
        client: Cliente de Schwab autenticado
        ticker (str): Símbolo del ticker (ej: 'AAPL', 'SPY')
        strike (float): Precio de ejercicio
        tipo (str): 'CALL' o 'PUT'
        fecha_salida (date): Fecha de expiración
    
    Returns:
        tuple: (mid_price, delta, theta)
    """
    try:
        if client is None:
            return None, None, None
        
        response = client.get_option_chain(ticker)
        if response.status_code != 200:
            return None, None, None
        
        opciones = response.json()
        
        # Seleccionar el mapa correcto según el tipo
        if tipo == 'CALL':
            option_map = opciones.get('callExpDateMap', {})
        else:
            option_map = opciones.get('putExpDateMap', {})
        
        # Buscar fecha
        fecha_str = fecha_salida.strftime('%Y-%m-%d')
        fecha_key_match = None
        for key in option_map.keys():
            if key.startswith(fecha_str):
                fecha_key_match = key
                break
        
        if fecha_key_match:
            strikes = option_map[fecha_key_match]
            strike_str = str(float(strike))
            
            if strike_str in strikes:
                contrato = strikes[strike_str][0]
                
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



def get_current_price_schwab(client, ticker):
    """
    Obtiene el precio actual del ticker desde Schwab.
    
    Args:
        client: Cliente de Schwab autenticado
        ticker (str): Símbolo del ticker (ej: 'AAPL', 'SPY', 'QQQ')
    
    Returns:
        float: Precio actual del ticker, None si hay error
    """
    try:
        if client is None:
            return None
        
        # Obtener quote desde Schwab
        response = client.get_quote(ticker)
        
        if response.status_code != 200:
            return None
        
        quote_data = response.json()
        
        # La estructura de respuesta de Schwab es un diccionario con el ticker como clave
        # Ejemplo: {'AAPL': {'quote': {...}}}
        if ticker in quote_data:
            ticker_data = quote_data[ticker]
            
            # Intentar obtener diferentes campos de precio en orden de prioridad
            # 1. lastPrice - precio de la última transacción
            # 2. mark - precio mark (mid entre bid/ask)
            # 3. closePrice - precio de cierre
            
            if 'quote' in ticker_data:
                quote = ticker_data['quote']
                
                # Prioridad 1: Last Price
                if 'lastPrice' in quote and quote['lastPrice'] is not None:
                    return float(quote['lastPrice'])
                
                # Prioridad 2: Mark Price
                if 'mark' in quote and quote['mark'] is not None:
                    return float(quote['mark'])
                
                # Prioridad 3: Close Price
                if 'closePrice' in quote and quote['closePrice'] is not None:
                    return float(quote['closePrice'])
                
                # Prioridad 4: Calcular desde bid/ask
                if 'bidPrice' in quote and 'askPrice' in quote:
                    bid = quote.get('bidPrice')
                    ask = quote.get('askPrice')
                    if bid is not None and ask is not None and bid > 0 and ask > 0:
                        return float((bid + ask) / 2)
        
        return None
        
    except Exception as e:
        # En producción podrías usar logging en lugar de print
        print(f"Error obteniendo precio actual de Schwab: {e}")
        return None


def get_atm_strike_schwab(client, ticker, current_price, expiration_date):
    """
    Obtiene el strike ATM (at-the-money) más cercano al precio actual.
    """
    try:
        from datetime import datetime, date

        # 1. Ajustar el símbolo para SPX
        symbol = '$SPX' if ticker == 'SPX' else ticker
        
        # 2. CORRECCIÓN: Asegurar que expiration_date sea un objeto date
        # Si es un string, lo convertimos. Si ya es date/datetime, nos aseguramos que sea date.
        if isinstance(expiration_date, str):
            # Intentar parsear el string (asumiendo formato YYYY-MM-DD)
            target_date = datetime.strptime(expiration_date, "%Y-%m-%d").date()
        elif isinstance(expiration_date, datetime):
            target_date = expiration_date.date()
        else:
            target_date = expiration_date # Ya es un objeto date

        # 3. Llamada a la API usando el objeto date (target_date)
        response = client.get_option_chain(
            symbol,
            from_date=target_date,  # Ahora es tipo datetime.date
            to_date=target_date     # Ahora es tipo datetime.date
        )
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        available_strikes = set()
        
        # 4. Extraer strikes (usamos el string para comparar en el diccionario)
        exp_date_str = target_date.strftime("%Y-%m-%d")
        
        for map_type in ['callExpDateMap', 'putExpDateMap']:
            exp_map = data.get(map_type, {})
            for date_key, strikes_dict in exp_map.items():
                if date_key.startswith(exp_date_str):
                    for strike_key in strikes_dict.keys():
                        available_strikes.add(float(strike_key))
        
        if not available_strikes:
            return None
        
        # 5. Encontrar el strike más cercano
        atm_strike = min(available_strikes, key=lambda x: abs(x - current_price))
        return atm_strike
        
    except Exception as e:
        st.error(f"Error detallado en Schwab ATM: {e}")
        return None
