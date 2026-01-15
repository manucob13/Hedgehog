"""
Utilidades para interactuar con la API de Schwab
Este módulo contiene funciones para conectar con Schwab y obtener datos de opciones.
"""

import streamlit as st
import os
import json
from datetime import datetime, date
from schwab.auth import easy_client


def normalize_ticker(ticker):
    """
    Convierte tickers especiales al formato de Schwab.
    
    Args:
        ticker (str): Símbolo del ticker
    
    Returns:
        str: Ticker normalizado para Schwab API
    """
    ticker_upper = ticker.upper().strip()
    return '$SPX' if ticker_upper == 'SPX' else ticker_upper


def normalize_date(date_input):
    """
    Convierte diferentes formatos de fecha a datetime.date.
    
    Args:
        date_input: Puede ser str, datetime, o date
    
    Returns:
        date: Objeto datetime.date
    """
    if isinstance(date_input, str):
        return datetime.strptime(date_input, "%Y-%m-%d").date()
    elif isinstance(date_input, datetime):
        return date_input.date()
    else:
        return date_input


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
    if os.path.exists(token_path):
        return True
    
    try:
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
    api_key, app_secret, redirect_uri = get_schwab_credentials()
    
    if api_key is None or app_secret is None or redirect_uri is None:
        return None
    
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
        return client

    except Exception as e:
        st.error(f"❌ Error al inicializar Schwab Client: {e}")
        return None


def obtener_datos_opcion(client, ticker, strike, tipo, fecha_salida):
    """
    Obtiene precio (mid), delta y theta de una opción desde Schwab.
    Versión mejorada con manejo robusto de strikes y fechas.
    
    Args:
        client: Cliente autenticado de Schwab
        ticker (str): Símbolo del ticker
        strike (float): Precio strike deseado
        tipo (str): 'CALL' o 'PUT'
        fecha_salida: Fecha de expiración (str, datetime, o date)
    
    Returns:
        tuple: (mid_price, delta, theta) o (None, None, None) si hay error
    """
    try:
        if client is None:
            print("❌ Cliente es None")
            return None, None, None
        
        symbol = normalize_ticker(ticker)
        fecha_normalizada = normalize_date(fecha_salida)
        
        print(f"📞 Consultando opciones para {symbol} ({tipo}) - Fecha: {fecha_normalizada}")
        
        response = client.get_option_chain(symbol)
        if response.status_code != 200:
            print(f"❌ Status code para {symbol}: {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return None, None, None
        
        opciones = response.json()
        option_map = opciones.get('callExpDateMap' if tipo == 'CALL' else 'putExpDateMap', {})
        
        if not option_map:
            print(f"❌ No hay {tipo} en la respuesta")
            print(f"Claves disponibles: {list(opciones.keys())}")
            return None, None, None
        
        fecha_str = fecha_normalizada.strftime('%Y-%m-%d')
        fecha_key_match = None
        
        print(f"🔍 Buscando fecha {fecha_str}")
        print(f"Fechas disponibles: {list(option_map.keys())[:5]}...")
        
        for key in option_map.keys():
            if key.startswith(fecha_str):
                fecha_key_match = key
                break
        
        if not fecha_key_match:
            print(f"❌ No hay fecha que coincida con {fecha_str}")
            print(f"Todas las fechas disponibles: {list(option_map.keys())[:10]}")
            return None, None, None
        
        print(f"✅ Fecha encontrada: {fecha_key_match}")
        
        strikes_dict = option_map[fecha_key_match]
        
        # Obtener todos los strikes disponibles como floats
        available_strikes = []
        strike_key_map = {}  # Mapeo de strike float a su key original
        
        for strike_key in strikes_dict.keys():
            try:
                strike_float = float(strike_key)
                available_strikes.append(strike_float)
                strike_key_map[strike_float] = strike_key
            except ValueError:
                print(f"⚠️ Strike no numérico ignorado: {strike_key}")
                continue
        
        if not available_strikes:
            print(f"❌ Sin strikes disponibles para {tipo}")
            return None, None, None
        
        print(f"Strikes disponibles: {sorted(available_strikes)[:10]}... (total: {len(available_strikes)})")
        
        # Encontrar el strike más cercano
        closest_strike_float = min(available_strikes, key=lambda x: abs(x - float(strike)))
        
        # Obtener la key original del strike más cercano
        strike_key_to_use = strike_key_map[closest_strike_float]
        
        print(f"🔍 {tipo}: Buscando {strike} → Encontrado {closest_strike_float} (key: '{strike_key_to_use}')")
        
        if strike_key_to_use not in strikes_dict:
            print(f"❌ Strike key '{strike_key_to_use}' no encontrado en diccionario")
            print(f"Keys disponibles: {list(strikes_dict.keys())[:10]}")
            return None, None, None
        
        contratos_list = strikes_dict[strike_key_to_use]
        
        if not contratos_list or len(contratos_list) == 0:
            print(f"❌ Lista de contratos vacía para strike {strike_key_to_use}")
            return None, None, None
        
        contrato = contratos_list[0]
        
        bid = contrato.get('bid', 0)
        ask = contrato.get('ask', 0)
        mark = contrato.get('mark', 0)
        
        print(f"   Bid: {bid}, Ask: {ask}, Mark: {mark}")
        
        # Para índices como SPX, priorizar mark si bid/ask no están disponibles
        if bid > 0 and ask > 0:
            mid_price = (bid + ask) / 2
            print(f"   ✅ Usando Mid (Bid+Ask)/2: {mid_price}")
        elif mark > 0:
            mid_price = mark
            print(f"   ⚠️ Bid/Ask inválidos, usando Mark: {mid_price}")
        else:
            print(f"❌ Bid/Ask y Mark inválidos o cero")
            print(f"Contrato completo: {json.dumps(contrato, indent=2)[:500]}")
            return None, None, None
        
        delta = contrato.get('delta', None)
        theta = contrato.get('theta', None)
        
        print(f"   ✅ Precio final: {mid_price}, Delta: {delta}, Theta: {theta}")
        
        return mid_price, delta, theta
        
    except Exception as e:
        print(f"❌ Error en obtener_datos_opcion ({tipo}): {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def get_current_price_schwab(client, ticker):
    """
    Obtiene el precio actual del ticker desde Schwab.
    
    Args:
        client: Cliente de Schwab autenticado
        ticker (str): Símbolo del ticker (ej: 'AAPL', 'SPY', 'QQQ', 'SPX')
    
    Returns:
        float: Precio actual del ticker, None si hay error
    """
    try:
        if client is None:
            print("❌ Cliente es None en get_current_price_schwab")
            return None
        
        # Normalizar el ticker (SPX -> $SPX)
        symbol = normalize_ticker(ticker)
        
        print(f"🔍 Obteniendo precio actual de: {ticker} → {symbol}")
        
        response = client.get_quote(symbol)
        
        if response.status_code != 200:
            print(f"❌ Error en get_quote: Status code {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return None
        
        quote_data = response.json()
        
        # Buscar con el símbolo normalizado
        if symbol in quote_data:
            ticker_data = quote_data[symbol]
            
            if 'quote' in ticker_data:
                quote = ticker_data['quote']
                
                if 'lastPrice' in quote and quote['lastPrice'] is not None:
                    price = float(quote['lastPrice'])
                    print(f"✅ Precio obtenido (lastPrice): {price}")
                    return price
                
                if 'mark' in quote and quote['mark'] is not None:
                    price = float(quote['mark'])
                    print(f"✅ Precio obtenido (mark): {price}")
                    return price
                
                if 'closePrice' in quote and quote['closePrice'] is not None:
                    price = float(quote['closePrice'])
                    print(f"✅ Precio obtenido (closePrice): {price}")
                    return price
                
                if 'bidPrice' in quote and 'askPrice' in quote:
                    bid = quote.get('bidPrice')
                    ask = quote.get('askPrice')
                    if bid is not None and ask is not None and bid > 0 and ask > 0:
                        price = float((bid + ask) / 2)
                        print(f"✅ Precio obtenido (mid bid/ask): {price}")
                        return price
        
        print(f"❌ No se encontró precio para {symbol} en la respuesta")
        return None
        
    except Exception as e:
        print(f"❌ Error obteniendo precio actual de Schwab: {e}")
        import traceback
        traceback.print_exc()
        return None


def diagnose_option_chain(client, ticker):
    """
    Función de diagnóstico para ver qué fechas están disponibles.
    """
    try:
        symbol = normalize_ticker(ticker)
        print(f"\n{'🔬'*30}")
        print(f"DIAGNÓSTICO DE CADENA DE OPCIONES")
        print(f"Ticker: {ticker} → {symbol}")
        print(f"{'🔬'*30}\n")
        
        response = client.get_option_chain(symbol)
        
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}")
            return
        
        data = response.json()
        
        print(f"📊 Estructura de respuesta:")
        print(f"   Keys: {list(data.keys())}\n")
        
        for map_type in ['callExpDateMap', 'putExpDateMap']:
            exp_map = data.get(map_type, {})
            if exp_map:
                dates = sorted(list(exp_map.keys()))
                print(f"\n{map_type}:")
                print(f"   Total fechas: {len(dates)}")
                print(f"   Primera fecha: {dates[0] if dates else 'N/A'}")
                print(f"   Última fecha: {dates[-1] if dates else 'N/A'}")
                print(f"\n   Primeras 20 fechas:")
                for i, date_key in enumerate(dates[:20], 1):
                    print(f"      {i:2d}. {date_key}")
        
        print(f"\n{'🔬'*30}\n")
        
    except Exception as e:
        print(f"❌ Error en diagnóstico: {e}")
        import traceback
        traceback.print_exc()

def get_atm_strike_schwab(client, ticker, current_price, expiration_date):
    try:
        symbol = normalize_ticker(ticker)
        target_date = normalize_date(expiration_date)
        exp_date_str = target_date.strftime("%Y-%m-%d")
        
        # OPTIMIZACIÓN: Añadimos parámetros para que SPX no sature la conexión
        # 'strike_count' limita los resultados cerca del precio actual
        response = client.get_option_chain(
            symbol, 
            strike_count=20, 
            from_date=target_date, 
            to_date=target_date
        )
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        available_strikes = set()
        
        # El SPX a veces devuelve los mapas vacíos si la fecha no es EXACTA
        # Schwab usa el formato "YYYY-MM-DD:Días"
        for map_type in ['callExpDateMap', 'putExpDateMap']:
            exp_map = data.get(map_type, {})
            for date_key, strikes_dict in exp_map.items():
                if date_key.startswith(exp_date_str):
                    for strike_key in strikes_dict.keys():
                        available_strikes.add(float(strike_key))
        
        if not available_strikes:
            # Si no encuentra nada, imprimimos qué fechas SÍ hay para debug
            print(f"Fechas disponibles en API: {list(data.get('callExpDateMap', {}).keys())}")
            return None
            
        return min(available_strikes, key=lambda x: abs(x - current_price))

    except Exception as e:
        print(f"Error: {e}")
        return None

