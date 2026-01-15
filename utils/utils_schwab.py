"""
Utilidades para interactuar con la API de Schwab
Este módulo contiene funciones para conectar con Schwab y obtener datos de opciones.
"""

import streamlit as st
import os
import json
from datetime import datetime, date, timedelta
from schwab.auth import easy_client


def log_info(message):
    """Muestra mensaje INFO en pantalla."""
    st.write(f"ℹ️ {message}")
    print(f"INFO: {message}")


def log_success(message):
    """Muestra mensaje SUCCESS en pantalla."""
    st.success(f"✅ {message}")
    print(f"SUCCESS: {message}")


def log_warning(message):
    """Muestra mensaje WARNING en pantalla."""
    st.warning(f"⚠️ {message}")
    print(f"WARNING: {message}")


def log_error(message):
    """Muestra mensaje ERROR en pantalla."""
    st.error(f"❌ {message}")
    print(f"ERROR: {message}")


def log_debug(message):
    """Muestra mensaje DEBUG en pantalla."""
    st.text(f"🔍 {message}")
    print(f"DEBUG: {message}")


def normalize_ticker(ticker):
    """
    Convierte tickers especiales al formato de Schwab.
    
    Args:
        ticker (str): Símbolo del ticker
    
    Returns:
        str: Ticker normalizado para Schwab API
    """
    ticker_upper = ticker.upper().strip()
    normalized = '$SPX' if ticker_upper == 'SPX' else ticker_upper
    log_debug(f"normalize_ticker: '{ticker}' -> '{normalized}'")
    return normalized


def normalize_date(date_input):
    """
    Convierte diferentes formatos de fecha a datetime.date.
    
    Args:
        date_input: Puede ser str, datetime, o date
    
    Returns:
        date: Objeto datetime.date
    """
    if isinstance(date_input, str):
        result = datetime.strptime(date_input, "%Y-%m-%d").date()
    elif isinstance(date_input, datetime):
        result = date_input.date()
    else:
        result = date_input
    
    log_debug(f"normalize_date: {date_input} ({type(date_input).__name__}) -> {result}")
    return result


def get_date_range_for_ticker(ticker, target_date):
    """
    Determina el rango de fechas óptimo según el ticker.
    
    Args:
        ticker (str): Símbolo del ticker (puede ser normalizado o no)
        target_date (date): Fecha objetivo
    
    Returns:
        tuple: (from_date, to_date)
    """
    normalized = normalize_ticker(ticker)
    
    if normalized == '$SPX':
        # SPX: usar rango MUY corto para evitar error 502 "Body buffer overflow"
        # SPX tiene muchos strikes (cada 5 puntos) y muchas expiraciones
        from_date = target_date - timedelta(days=1)
        to_date = target_date + timedelta(days=1)
    else:
        # QQQ y otros tickers usan rango corto
        from_date = target_date - timedelta(days=2)
        to_date = target_date + timedelta(days=2)
    
    log_info(f"📅 Rango de fechas para {ticker} ({normalized}): {from_date} a {to_date} ({(to_date - from_date).days} días)")
    
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
        log_success("Credenciales de Schwab obtenidas")
        return api_key, app_secret, redirect_uri
    except KeyError as e:
        log_error(f"Falta configurar los secrets de Schwab. Clave faltante: {e}")
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
        log_info(f"Token file ya existe: {token_path}")
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
        
        log_success(f"Token file creado: {token_path}")
        return True
        
    except KeyError as e:
        log_error(f"Falta configurar el token en secrets: {e}")
        return False
    except Exception as e:
        log_error(f"Error al crear token desde secrets: {e}")
        return False


def connect_to_schwab(token_path="schwab_token.json"):
    """
    Conecta con Schwab usando credenciales y token desde st.secrets.
    
    Args:
        token_path (str): Ruta al archivo token.json (default: "schwab_token.json")
    
    Returns:
        schwab.client.Client: Cliente de Schwab si la conexión es exitosa, None en caso contrario
    """
    log_info("🔌 Iniciando conexión con Schwab...")
    
    api_key, app_secret, redirect_uri = get_schwab_credentials()
    
    if api_key is None or app_secret is None or redirect_uri is None:
        log_error("No se pudieron obtener las credenciales")
        return None
    
    if not setup_token_from_secrets(token_path):
        log_error("No se pudo configurar el token de Schwab")
        return None

    try:
        client = easy_client(
            api_key=api_key,
            app_secret=app_secret,
            callback_url=redirect_uri,
            token_path=token_path
        )
        log_success("Cliente de Schwab inicializado exitosamente")
        return client

    except Exception as e:
        log_error(f"Error al inicializar Schwab Client: {e}")
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
    st.markdown("---")
    log_info(f"📊 **OBTENER_DATOS_OPCION** - Ticker: {ticker}, Strike: {strike}, Tipo: {tipo}, Fecha: {fecha_salida}")
    
    try:
        if client is None:
            log_error("Cliente es None")
            return None, None, None
        
        symbol = normalize_ticker(ticker)
        fecha_normalizada = normalize_date(fecha_salida)
        
        # Usar el sistema de rango de fechas dinámico
        from_date, to_date = get_date_range_for_ticker(symbol, fecha_normalizada)
        
        log_info(f"📡 Llamando API get_option_chain para {symbol}")
        
        response = client.get_option_chain(
            symbol,
            from_date=from_date,
            to_date=to_date
        )
        
        log_info(f"📥 Respuesta recibida - Status Code: {response.status_code}")
        
        if response.status_code != 200:
            log_error(f"Status code no exitoso: {response.status_code}")
            st.code(response.text[:500], language="text")
            return None, None, None
        
        opciones = response.json()
        
        log_debug(f"Keys principales en respuesta: {list(opciones.keys())}")
        
        map_key = 'callExpDateMap' if tipo == 'CALL' else 'putExpDateMap'
        option_map = opciones.get(map_key, {})
        
        log_info(f"📋 {map_key}: {len(option_map)} fechas disponibles")
        
        if not option_map:
            log_error(f"{map_key} está vacío - NO HAY DATOS")
            return None, None, None
        
        # Log de fechas disponibles
        st.write(f"📅 **Fechas disponibles en {map_key}:**")
        fechas_list = sorted(option_map.keys())
        for idx, date_key in enumerate(fechas_list[:15], 1):
            st.text(f"  {idx}. {date_key}")
        if len(fechas_list) > 15:
            st.text(f"  ... y {len(fechas_list) - 15} fechas más")
        
        # Buscar la fecha en el formato de Schwab (YYYY-MM-DD:XX)
        fecha_str = fecha_normalizada.strftime('%Y-%m-%d')
        fecha_key_match = None
        
        log_info(f"🔍 Buscando fecha objetivo: **{fecha_str}**")
        
        for key in option_map.keys():
            if key.startswith(fecha_str):
                fecha_key_match = key
                log_success(f"Fecha encontrada: {fecha_key_match}")
                break
        
        if not fecha_key_match:
            log_error(f"NO se encontró la fecha {fecha_str}")
            st.write("**❌ Fechas disponibles completas:**")
            for key in sorted(option_map.keys()):
                st.text(f"  - {key}")
            return None, None, None
        
        strikes_dict = option_map[fecha_key_match]
        log_info(f"📊 Strikes disponibles para {fecha_key_match}: {len(strikes_dict)}")
        
        # Obtener todos los strikes disponibles
        available_strikes = []
        strike_key_map = {}
        
        for strike_key in strikes_dict.keys():
            try:
                strike_float = float(strike_key)
                available_strikes.append(strike_float)
                strike_key_map[strike_float] = strike_key
            except ValueError:
                log_warning(f"Strike inválido ignorado: {strike_key}")
                continue
        
        if not available_strikes:
            log_error("No hay strikes válidos")
            return None, None, None
        
        log_success(f"Total strikes válidos: {len(available_strikes)}")
        log_info(f"Rango de strikes: ${min(available_strikes):.2f} - ${max(available_strikes):.2f}")
        
        # Encontrar el strike más cercano
        closest_strike_float = min(available_strikes, key=lambda x: abs(x - float(strike)))
        strike_key_to_use = strike_key_map[closest_strike_float]
        
        log_info(f"🎯 Strike seleccionado: ${closest_strike_float:.2f} (solicitado: ${strike:.2f}, diferencia: ${abs(closest_strike_float - float(strike)):.2f})")
        
        if strike_key_to_use not in strikes_dict:
            log_error("Strike key no existe en strikes_dict")
            return None, None, None
        
        contratos_list = strikes_dict[strike_key_to_use]
        
        log_info(f"📄 Contratos encontrados: {len(contratos_list)}")
        
        if not contratos_list or len(contratos_list) == 0:
            log_error("Lista de contratos vacía")
            return None, None, None
        
        contrato = contratos_list[0]
        
        bid = contrato.get('bid', 0)
        ask = contrato.get('ask', 0)
        mark = contrato.get('mark', 0)
        
        log_info(f"💰 Precios - Bid: {bid}, Ask: {ask}, Mark: {mark}")
        
        # Para índices como SPX, priorizar mark si bid/ask no están disponibles
        if bid > 0 and ask > 0:
            mid_price = (bid + ask) / 2
            log_success(f"Mid calculado de bid/ask: ${mid_price:.2f}")
        elif mark > 0:
            mid_price = mark
            log_success(f"Usando mark como mid: ${mid_price:.2f}")
        else:
            log_error("No hay precios válidos disponibles")
            return None, None, None
        
        delta = contrato.get('delta', None)
        theta = contrato.get('theta', None)
        
        log_info(f"📊 Greeks - Delta: {delta}, Theta: {theta}")
        log_success(f"**RESULTADO:** Mid: ${mid_price:.2f}, Delta: {delta}, Theta: {theta}")
        
        return mid_price, delta, theta
        
    except Exception as e:
        log_error(f"EXCEPCIÓN en obtener_datos_opcion: {str(e)}")
        import traceback
        st.code(traceback.format_exc(), language="python")
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
    st.markdown("---")
    log_info(f"💵 **GET_CURRENT_PRICE** - Ticker: {ticker}")
    
    try:
        if client is None:
            log_error("Cliente es None")
            return None
        
        symbol = normalize_ticker(ticker)
        
        log_info(f"📡 Llamando API get_quote para: {symbol}")
        response = client.get_quote(symbol)
        
        log_info(f"📥 Respuesta - Status Code: {response.status_code}")
        
        if response.status_code != 200:
            log_error(f"Status code: {response.status_code}")
            st.code(response.text[:500], language="text")
            return None
        
        quote_data = response.json()
        
        log_debug(f"Keys en respuesta: {list(quote_data.keys())}")
        
        if symbol in quote_data:
            ticker_data = quote_data[symbol]
            log_debug(f"Keys en ticker_data: {list(ticker_data.keys())}")
            
            if 'quote' in ticker_data:
                quote = ticker_data['quote']
                
                # Prioridad: lastPrice -> mark -> closePrice -> mid(bid,ask)
                if 'lastPrice' in quote and quote['lastPrice'] is not None:
                    price = float(quote['lastPrice'])
                    log_success(f"Usando lastPrice: ${price:.2f}")
                    return price
                
                if 'mark' in quote and quote['mark'] is not None:
                    price = float(quote['mark'])
                    log_success(f"Usando mark: ${price:.2f}")
                    return price
                
                if 'closePrice' in quote and quote['closePrice'] is not None:
                    price = float(quote['closePrice'])
                    log_success(f"Usando closePrice: ${price:.2f}")
                    return price
                
                if 'bidPrice' in quote and 'askPrice' in quote:
                    bid = quote.get('bidPrice')
                    ask = quote.get('askPrice')
                    if bid is not None and ask is not None and bid > 0 and ask > 0:
                        price = float((bid + ask) / 2)
                        log_success(f"Usando mid(bid/ask): ${price:.2f}")
                        return price
                
                log_error("No se encontró ningún precio válido en quote")
                st.write(f"Contenido de quote: {list(quote.keys())}")
            else:
                log_error("No existe 'quote' en ticker_data")
        else:
            log_error(f"Symbol {symbol} no existe en quote_data")
        
        return None
        
    except Exception as e:
        log_error(f"EXCEPCIÓN en get_current_price_schwab: {str(e)}")
        import traceback
        st.code(traceback.format_exc(), language="python")
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
    st.markdown("---")
    log_info(f"🎯 **GET_ATM_STRIKE** - Ticker: {ticker}, Precio: ${current_price:.2f}, Fecha: {expiration_date}")
    
    try:
        symbol = normalize_ticker(ticker)
        target_date = normalize_date(expiration_date)
        
        # Usar el sistema de rango de fechas dinámico
        from_date, to_date = get_date_range_for_ticker(symbol, target_date)
        
        log_info(f"📡 Llamando API get_option_chain para {symbol}")
        
        response = client.get_option_chain(
            symbol,
            from_date=from_date,
            to_date=to_date
        )
        
        log_info(f"📥 Respuesta - Status Code: {response.status_code}")
        
        if response.status_code != 200:
            log_error(f"Status code: {response.status_code}")
            st.code(response.text[:500], language="text")
            return None
        
        data = response.json()
        available_strikes = set()
        exp_date_str = target_date.strftime("%Y-%m-%d")
        
        log_info(f"🔍 Buscando fecha: **{exp_date_str}**")
        
        # Buscar en ambos mapas (calls y puts)
        for map_type in ['callExpDateMap', 'putExpDateMap']:
            exp_map = data.get(map_type, {})
            
            if not exp_map:
                log_warning(f"{map_type} está vacío")
                continue
            
            st.write(f"📋 **Fechas en {map_type}** ({len(exp_map)} fechas):")
            fechas_list = sorted(exp_map.keys())
            for idx, date_key in enumerate(fechas_list[:15], 1):
                st.text(f"  {idx}. {date_key}")
            if len(fechas_list) > 15:
                st.text(f"  ... y {len(fechas_list) - 15} fechas más")
            
            # Buscar la fecha exacta
            for date_key, strikes_dict in exp_map.items():
                if date_key.startswith(exp_date_str):
                    log_success(f"Fecha coincidente: {date_key} ({len(strikes_dict)} strikes)")
                    
                    for strike_key in strikes_dict.keys():
                        try:
                            strike_float = float(strike_key)
                            available_strikes.add(strike_float)
                        except ValueError:
                            log_warning(f"Strike inválido ignorado: {strike_key}")
                            continue
        
        if not available_strikes:
            log_error(f"NO se encontraron strikes para {exp_date_str}")
            
            st.write("**❌ Todas las fechas disponibles:**")
            all_dates = set()
            for map_type in ['callExpDateMap', 'putExpDateMap']:
                exp_map = data.get(map_type, {})
                all_dates.update(exp_map.keys())
            
            for idx, d in enumerate(sorted(all_dates)[:30], 1):
                st.text(f"  {idx}. {d}")
            if len(all_dates) > 30:
                st.text(f"  ... y {len(all_dates) - 30} fechas más")
            
            return None
        
        # Encontrar el strike más cercano al precio actual
        available_strikes_list = sorted(available_strikes)
        
        log_success(f"Total strikes: {len(available_strikes_list)}")
        log_info(f"Rango: ${min(available_strikes_list):.2f} - ${max(available_strikes_list):.2f}")
        
        atm_strike = min(available_strikes, key=lambda x: abs(x - current_price))
        
        log_success(f"**Strike ATM:** ${atm_strike:.2f} (diferencia: ${abs(atm_strike - current_price):.2f})")
        
        return atm_strike
        
    except Exception as e:
        log_error(f"EXCEPCIÓN en get_atm_strike_schwab: {str(e)}")
        import traceback
        st.code(traceback.format_exc(), language="python")
        return None
