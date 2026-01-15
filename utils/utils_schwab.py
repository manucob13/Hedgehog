"""
Utilidades para interactuar con la API de Schwab
Este módulo contiene funciones para conectar con Schwab y obtener datos de opciones.
"""

import streamlit as st
import os
import json
from datetime import datetime, date, timedelta
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


def get_date_range_for_ticker(ticker, target_date):
    """
    Determina el rango de fechas óptimo según el ticker.
    
    Args:
        ticker (str): Símbolo del ticker
        target_date (date): Fecha objetivo
    
    Returns:
        tuple: (from_date, to_date)
    """
    if ticker in ['$SPX', 'SPX']:
        # SPX tiene muchas expiraciones, usar rango amplio
        from_date = target_date - timedelta(days=5)
        to_date = target_date + timedelta(days=35)
    else:
        # QQQ y otros tickers usan rango más corto
        from_date = target_date - timedelta(days=2)
        to_date = target_date + timedelta(days=2)
    
    return from_date, to_date


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
            return None, None, None
        
        symbol = normalize_ticker(ticker)
        fecha_normalizada = normalize_date(fecha_salida)
        
        # Usar el mismo sistema de rango de fechas
        from_date, to_date = get_date_range_for_ticker(symbol, fecha_normalizada)
        
        response = client.get_option_chain(
            symbol,
            from_date=from_date,
            to_date=to_date
        )
        
        if response.status_code != 200:
            return None, None, None
        
        opciones = response.json()
        option_map = opciones.get('callExpDateMap' if tipo == 'CALL' else 'putExpDateMap', {})
        
        if not option_map:
            return None, None, None
        
        # Buscar la fecha en el formato de Schwab (YYYY-MM-DD:XX)
        fecha_str = fecha_normalizada.strftime('%Y-%m-%d')
        fecha_key_match = None
        
        for key in option_map.keys():
            if key.startswith(fecha_str):
                fecha_key_match = key
                break
        
        if not fecha_key_match:
            return None, None, None
        
        strikes_dict = option_map[fecha_key_match]
        
        # Obtener todos los strikes disponibles
        available_strikes = []
        strike_key_map = {}
        
        for strike_key in strikes_dict.keys():
            try:
                strike_float = float(strike_key)
                available_strikes.append(strike_float)
                strike_key_map[strike_float] = strike_key
            except ValueError:
                continue
        
        if not available_strikes:
            return None, None, None
        
        # Encontrar el strike más cercano
        closest_strike_float = min(available_strikes, key=lambda x: abs(x - float(strike)))
        strike_key_to_use = strike_key_map[closest_strike_float]
        
        if strike_key_to_use not in strikes_dict:
            return None, None, None
        
        contratos_list = strikes_dict[strike_key_to_use]
        
        if not contratos_list or len(contratos_list) == 0:
            return None, None, None
        
        contrato = contratos_list[0]
        
        bid = contrato.get('bid', 0)
        ask = contrato.get('ask', 0)
        mark = contrato.get('mark', 0)
        
        # Para índices como SPX, priorizar mark si bid/ask no están disponibles
        if bid > 0 and ask > 0:
            mid_price = (bid + ask) / 2
        elif mark > 0:
            mid_price = mark
        else:
            return None, None, None
        
        delta = contrato.get('delta', None)
        theta = contrato.get('theta', None)
        
        return mid_price, delta, theta
        
    except Exception as e:
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
            return None
        
        symbol = normalize_ticker(ticker)
        response = client.get_quote(symbol)
        
        if response.status_code != 200:
            return None
        
        quote_data = response.json()
        
        if symbol in quote_data:
            ticker_data = quote_data[symbol]
            
            if 'quote' in ticker_data:
                quote = ticker_data['quote']
                
                # Prioridad: lastPrice -> mark -> closePrice -> mid(bid,ask)
                if 'lastPrice' in quote and quote['lastPrice'] is not None:
                    return float(quote['lastPrice'])
                
                if 'mark' in quote and quote['mark'] is not None:
                    return float(quote['mark'])
                
                if 'closePrice' in quote and quote['closePrice'] is not None:
                    return float(quote['closePrice'])
                
                if 'bidPrice' in quote and 'askPrice' in quote:
                    bid = quote.get('bidPrice')
                    ask = quote.get('askPrice')
                    if bid is not None and ask is not None and bid > 0 and ask > 0:
                        return float((bid + ask) / 2)
        
        return None
        
    except Exception as e:
        return None


def get_atm_strike_schwab(client, ticker, current_price, expiration_date):
    """
    Obtiene el strike ATM (at-the-money) más cercano al precio actual.
    
    Args:
        client: Cliente autenticado de Schwab
        ticker (str): Símbolo del ticker
        current_price (float): Precio actual del subyacente
        expiration_date: Fecha de expiración (str, datetime, o date)
    
    Returns:
        float: Strike ATM más cercano, None si hay error
    """
    try:
        symbol = normalize_ticker(ticker)
        target_date = normalize_date(expiration_date)
        
        # Determinar rango de fechas según el ticker
        from_date, to_date = get_date_range_for_ticker(symbol, target_date)
        
        print(f"\n{'='*60}")
        print(f"GET ATM STRIKE - {symbol}")
        print(f"Precio actual: {current_price}")
        print(f"Fecha objetivo: {target_date}")
        print(f"Rango búsqueda: {from_date} a {to_date}")
        print(f"{'='*60}\n")
        
        response = client.get_option_chain(
            symbol,
            from_date=from_date,
            to_date=to_date
        )
        
        if response.status_code != 200:
            print(f"❌ Error en respuesta: status_code={response.status_code}")
            return None
        
        data = response.json()
        available_strikes = set()
        exp_date_str = target_date.strftime("%Y-%m-%d")
        
        print(f"Buscando fecha: {exp_date_str}")
        
        # Buscar en ambos mapas (calls y puts)
        for map_type in ['callExpDateMap', 'putExpDateMap']:
            exp_map = data.get(map_type, {})
            
            if not exp_map:
                print(f"⚠️ {map_type} está vacío")
                continue
            
            print(f"\n📋 Fechas disponibles en {map_type}:")
            for date_key in exp_map.keys():
                print(f"  - {date_key}")
            
            # Buscar la fecha exacta
            for date_key, strikes_dict in exp_map.items():
                if date_key.startswith(exp_date_str):
                    print(f"\n✅ Encontrada fecha coincidente: {date_key}")
                    for strike_key in strikes_dict.keys():
                        try:
                            strike_float = float(strike_key)
                            available_strikes.add(strike_float)
                        except ValueError:
                            print(f"⚠️ Strike inválido ignorado: {strike_key}")
                            continue
        
        if not available_strikes:
            print(f"\n❌ No se encontraron strikes para {exp_date_str}")
            print("Fechas disponibles en la respuesta:")
            
            # Debug: mostrar todas las fechas disponibles
            all_dates = set()
            for map_type in ['callExpDateMap', 'putExpDateMap']:
                exp_map = data.get(map_type, {})
                all_dates.update(exp_map.keys())
            
            for d in sorted(all_dates):
                print(f"  - {d}")
            
            return None
        
        # Encontrar el strike más cercano al precio actual
        available_strikes_list = sorted(available_strikes)
        print(f"\n📊 Strikes disponibles ({len(available_strikes_list)}):")
        print(f"  Rango: ${min(available_strikes_list):.2f} - ${max(available_strikes_list):.2f}")
        
        atm_strike = min(available_strikes, key=lambda x: abs(x - current_price))
        
        print(f"\n✅ Strike ATM seleccionado: ${atm_strike:.2f}")
        print(f"   (más cercano a precio actual: ${current_price:.2f})")
        print(f"{'='*60}\n")
        
        return atm_strike
        
    except Exception as e:
        print(f"\n❌ Error en get_atm_strike_schwab: {e}")
        import traceback
        traceback.print_exc()
        return None
