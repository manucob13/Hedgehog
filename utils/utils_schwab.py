"""
Utilidades para interactuar con la API de Schwab
Este módulo contiene funciones para conectar con Schwab y obtener datos de opciones.
"""

import streamlit as st
import os
import json
from datetime import datetime, date, timedelta
from schwab.auth import easy_client


class StreamlitLogger:
    """Logger que muestra mensajes en Streamlit en tiempo real."""
    
    def __init__(self):
        if 'debug_logs' not in st.session_state:
            st.session_state.debug_logs = []
    
    def log(self, message, level="INFO"):
        """Agrega un mensaje al log."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] {level}: {message}"
        st.session_state.debug_logs.append(log_entry)
        
        # Mostrar en consola también
        print(log_entry)
    
    def debug(self, message):
        self.log(message, "DEBUG")
    
    def info(self, message):
        self.log(message, "INFO")
    
    def warning(self, message):
        self.log(message, "WARNING")
    
    def error(self, message):
        self.log(message, "ERROR")
    
    def clear(self):
        """Limpia todos los logs."""
        st.session_state.debug_logs = []
    
    def display(self):
        """Muestra todos los logs en un expander de Streamlit."""
        if st.session_state.debug_logs:
            with st.expander(f"🔍 Debug Logs ({len(st.session_state.debug_logs)} entradas)", expanded=False):
                log_text = "\n".join(st.session_state.debug_logs)
                st.text_area("Logs", log_text, height=400, key="debug_log_display")
                
                if st.button("🗑️ Limpiar Logs"):
                    self.clear()
                    st.rerun()

# Crear instancia global del logger
logger = StreamlitLogger()


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
    logger.debug(f"normalize_ticker: '{ticker}' -> '{normalized}'")
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
    
    logger.debug(f"normalize_date: {date_input} ({type(date_input).__name__}) -> {result}")
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
        # SPX tiene muchas expiraciones, usar rango amplio
        from_date = target_date - timedelta(days=5)
        to_date = target_date + timedelta(days=35)
    else:
        # QQQ y otros tickers usan rango más corto
        from_date = target_date - timedelta(days=2)
        to_date = target_date + timedelta(days=2)
    
    logger.info(f"📅 get_date_range_for_ticker: {ticker} ({normalized})")
    logger.info(f"   Target date: {target_date}")
    logger.info(f"   Range: {from_date} to {to_date} ({(to_date - from_date).days} days)")
    
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
        logger.info("✅ Credenciales de Schwab obtenidas exitosamente")
        return api_key, app_secret, redirect_uri
    except KeyError as e:
        logger.error(f"❌ Falta configurar los secrets de Schwab. Clave faltante: {e}")
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
        logger.info(f"✅ Token file ya existe: {token_path}")
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
        
        logger.info(f"✅ Token file creado: {token_path}")
        return True
        
    except KeyError as e:
        logger.error(f"❌ Falta configurar el token en secrets: {e}")
        st.error(f"❌ Falta configurar el token en secrets: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error al crear token desde secrets: {e}")
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
    logger.info("🔌 Iniciando conexión con Schwab...")
    
    api_key, app_secret, redirect_uri = get_schwab_credentials()
    
    if api_key is None or app_secret is None or redirect_uri is None:
        logger.error("❌ No se pudieron obtener las credenciales")
        return None
    
    if not setup_token_from_secrets(token_path):
        logger.error("❌ No se pudo configurar el token de Schwab")
        st.error("❌ No se pudo configurar el token de Schwab")
        return None

    try:
        client = easy_client(
            api_key=api_key,
            app_secret=app_secret,
            callback_url=redirect_uri,
            token_path=token_path
        )
        logger.info("✅ Cliente de Schwab inicializado exitosamente")
        return client

    except Exception as e:
        logger.error(f"❌ Error al inicializar Schwab Client: {e}")
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
    logger.info("="*80)
    logger.info(f"📊 OBTENER_DATOS_OPCION - Inicio")
    logger.info(f"   Ticker: {ticker}")
    logger.info(f"   Strike: {strike}")
    logger.info(f"   Tipo: {tipo}")
    logger.info(f"   Fecha salida: {fecha_salida}")
    
    try:
        if client is None:
            logger.error("❌ Cliente es None")
            return None, None, None
        
        symbol = normalize_ticker(ticker)
        fecha_normalizada = normalize_date(fecha_salida)
        
        # Usar el sistema de rango de fechas dinámico
        from_date, to_date = get_date_range_for_ticker(symbol, fecha_normalizada)
        
        logger.info(f"📡 Llamando API get_option_chain:")
        logger.info(f"   Symbol: {symbol}")
        logger.info(f"   From: {from_date}")
        logger.info(f"   To: {to_date}")
        
        response = client.get_option_chain(
            symbol,
            from_date=from_date,
            to_date=to_date
        )
        
        logger.info(f"📥 Respuesta recibida - Status: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"❌ Status code no exitoso: {response.status_code}")
            logger.error(f"   Response text (primeros 500 chars): {response.text[:500]}")
            return None, None, None
        
        opciones = response.json()
        
        # Log de estructura de respuesta
        logger.info(f"📊 Estructura de respuesta:")
        logger.info(f"   Keys principales: {list(opciones.keys())}")
        
        map_key = 'callExpDateMap' if tipo == 'CALL' else 'putExpDateMap'
        option_map = opciones.get(map_key, {})
        
        logger.info(f"📋 Mapa de opciones ({map_key}):")
        logger.info(f"   ¿Vacío?: {len(option_map) == 0}")
        logger.info(f"   Cantidad de fechas: {len(option_map)}")
        
        if not option_map:
            logger.error(f"❌ {map_key} está vacío")
            return None, None, None
        
        # Log de fechas disponibles
        logger.info(f"📅 Fechas disponibles en {map_key}:")
        for idx, date_key in enumerate(sorted(option_map.keys())[:10], 1):  # Primeras 10
            logger.info(f"   {idx}. {date_key}")
        if len(option_map) > 10:
            logger.info(f"   ... y {len(option_map) - 10} fechas más")
        
        # Buscar la fecha en el formato de Schwab (YYYY-MM-DD:XX)
        fecha_str = fecha_normalizada.strftime('%Y-%m-%d')
        fecha_key_match = None
        
        logger.info(f"🔍 Buscando fecha objetivo: {fecha_str}")
        
        for key in option_map.keys():
            if key.startswith(fecha_str):
                fecha_key_match = key
                logger.info(f"✅ Fecha encontrada: {fecha_key_match}")
                break
        
        if not fecha_key_match:
            logger.error(f"❌ No se encontró la fecha {fecha_str}")
            logger.error(f"   Fechas disponibles completas:")
            for key in sorted(option_map.keys()):
                logger.error(f"     - {key}")
            return None, None, None
        
        strikes_dict = option_map[fecha_key_match]
        logger.info(f"📊 Strikes disponibles para {fecha_key_match}: {len(strikes_dict)}")
        
        # Obtener todos los strikes disponibles
        available_strikes = []
        strike_key_map = {}
        
        logger.info(f"🔢 Procesando strikes...")
        for strike_key in strikes_dict.keys():
            try:
                strike_float = float(strike_key)
                available_strikes.append(strike_float)
                strike_key_map[strike_float] = strike_key
            except ValueError:
                logger.warning(f"   ⚠️ Strike inválido ignorado: {strike_key}")
                continue
        
        if not available_strikes:
            logger.error(f"❌ No hay strikes válidos")
            return None, None, None
        
        logger.info(f"✅ Total strikes válidos: {len(available_strikes)}")
        logger.info(f"   Rango: {min(available_strikes):.2f} - {max(available_strikes):.2f}")
        
        # Encontrar el strike más cercano
        closest_strike_float = min(available_strikes, key=lambda x: abs(x - float(strike)))
        strike_key_to_use = strike_key_map[closest_strike_float]
        
        logger.info(f"🎯 Strike más cercano:")
        logger.info(f"   Solicitado: {strike}")
        logger.info(f"   Seleccionado: {closest_strike_float} (key: {strike_key_to_use})")
        logger.info(f"   Diferencia: {abs(closest_strike_float - float(strike)):.2f}")
        
        if strike_key_to_use not in strikes_dict:
            logger.error(f"❌ Strike key no existe en strikes_dict")
            return None, None, None
        
        contratos_list = strikes_dict[strike_key_to_use]
        
        logger.info(f"📄 Contratos para strike {strike_key_to_use}:")
        logger.info(f"   Cantidad: {len(contratos_list)}")
        
        if not contratos_list or len(contratos_list) == 0:
            logger.error(f"❌ Lista de contratos vacía")
            return None, None, None
        
        contrato = contratos_list[0]
        
        logger.info(f"📋 Datos del contrato (keys): {list(contrato.keys())[:10]}")
        
        bid = contrato.get('bid', 0)
        ask = contrato.get('ask', 0)
        mark = contrato.get('mark', 0)
        
        logger.info(f"💰 Precios:")
        logger.info(f"   Bid: {bid}")
        logger.info(f"   Ask: {ask}")
        logger.info(f"   Mark: {mark}")
        
        # Para índices como SPX, priorizar mark si bid/ask no están disponibles
        if bid > 0 and ask > 0:
            mid_price = (bid + ask) / 2
            logger.info(f"✅ Mid calculado de bid/ask: {mid_price}")
        elif mark > 0:
            mid_price = mark
            logger.info(f"✅ Usando mark como mid: {mid_price}")
        else:
            logger.error(f"❌ No hay precios válidos disponibles")
            return None, None, None
        
        delta = contrato.get('delta', None)
        theta = contrato.get('theta', None)
        
        logger.info(f"📊 Greeks:")
        logger.info(f"   Delta: {delta}")
        logger.info(f"   Theta: {theta}")
        
        logger.info(f"✅ RESULTADO FINAL:")
        logger.info(f"   Mid Price: {mid_price}")
        logger.info(f"   Delta: {delta}")
        logger.info(f"   Theta: {theta}")
        logger.info("="*80)
        
        return mid_price, delta, theta
        
    except Exception as e:
        logger.error(f"❌ EXCEPCIÓN en obtener_datos_opcion: {str(e)}")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")
        logger.info("="*80)
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
    logger.info("="*80)
    logger.info(f"💵 GET_CURRENT_PRICE - Inicio")
    logger.info(f"   Ticker: {ticker}")
    
    try:
        if client is None:
            logger.error("❌ Cliente es None")
            return None
        
        symbol = normalize_ticker(ticker)
        
        logger.info(f"📡 Llamando API get_quote para: {symbol}")
        response = client.get_quote(symbol)
        
        logger.info(f"📥 Respuesta - Status: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"❌ Status code: {response.status_code}")
            logger.error(f"   Response text (primeros 500 chars): {response.text[:500]}")
            return None
        
        quote_data = response.json()
        
        logger.info(f"📊 Keys en respuesta: {list(quote_data.keys())}")
        
        if symbol in quote_data:
            ticker_data = quote_data[symbol]
            logger.info(f"📋 Keys en ticker_data: {list(ticker_data.keys())}")
            
            if 'quote' in ticker_data:
                quote = ticker_data['quote']
                logger.info(f"💰 Keys en quote (primeras 10): {list(quote.keys())[:10]}")
                
                # Prioridad: lastPrice -> mark -> closePrice -> mid(bid,ask)
                if 'lastPrice' in quote and quote['lastPrice'] is not None:
                    price = float(quote['lastPrice'])
                    logger.info(f"✅ Usando lastPrice: {price}")
                    logger.info("="*80)
                    return price
                
                if 'mark' in quote and quote['mark'] is not None:
                    price = float(quote['mark'])
                    logger.info(f"✅ Usando mark: {price}")
                    logger.info("="*80)
                    return price
                
                if 'closePrice' in quote and quote['closePrice'] is not None:
                    price = float(quote['closePrice'])
                    logger.info(f"✅ Usando closePrice: {price}")
                    logger.info("="*80)
                    return price
                
                if 'bidPrice' in quote and 'askPrice' in quote:
                    bid = quote.get('bidPrice')
                    ask = quote.get('askPrice')
                    if bid is not None and ask is not None and bid > 0 and ask > 0:
                        price = float((bid + ask) / 2)
                        logger.info(f"✅ Usando mid(bid/ask): {price}")
                        logger.info("="*80)
                        return price
                
                logger.error("❌ No se encontró ningún precio válido")
            else:
                logger.error("❌ No existe 'quote' en ticker_data")
        else:
            logger.error(f"❌ Symbol {symbol} no existe en quote_data")
        
        logger.info("="*80)
        return None
        
    except Exception as e:
        logger.error(f"❌ EXCEPCIÓN en get_current_price_schwab: {str(e)}")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")
        logger.info("="*80)
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
    logger.info("="*80)
    logger.info(f"🎯 GET_ATM_STRIKE - Inicio")
    logger.info(f"   Ticker: {ticker}")
    logger.info(f"   Precio actual: {current_price}")
    logger.info(f"   Fecha expiración: {expiration_date}")
    
    try:
        symbol = normalize_ticker(ticker)
        target_date = normalize_date(expiration_date)
        
        # Usar el sistema de rango de fechas dinámico
        from_date, to_date = get_date_range_for_ticker(symbol, target_date)
        
        logger.info(f"📡 Llamando API get_option_chain:")
        logger.info(f"   Symbol: {symbol}")
        logger.info(f"   From: {from_date}")
        logger.info(f"   To: {to_date}")
        
        response = client.get_option_chain(
            symbol,
            from_date=from_date,
            to_date=to_date
        )
        
        logger.info(f"📥 Respuesta - Status: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"❌ Status code: {response.status_code}")
            logger.error(f"   Response text (primeros 500 chars): {response.text[:500]}")
            return None
        
        data = response.json()
        available_strikes = set()
        exp_date_str = target_date.strftime("%Y-%m-%d")
        
        logger.info(f"🔍 Buscando fecha: {exp_date_str}")
        
        # Buscar en ambos mapas (calls y puts)
        for map_type in ['callExpDateMap', 'putExpDateMap']:
            exp_map = data.get(map_type, {})
            
            if not exp_map:
                logger.warning(f"⚠️ {map_type} está vacío")
                continue
            
            logger.info(f"📋 Fechas disponibles en {map_type} ({len(exp_map)} fechas):")
            for idx, date_key in enumerate(sorted(exp_map.keys())[:10], 1):
                logger.info(f"   {idx}. {date_key}")
            if len(exp_map) > 10:
                logger.info(f"   ... y {len(exp_map) - 10} fechas más")
            
            # Buscar la fecha exacta
            for date_key, strikes_dict in exp_map.items():
                if date_key.startswith(exp_date_str):
                    logger.info(f"✅ Fecha coincidente encontrada: {date_key}")
                    logger.info(f"   Strikes en esta fecha: {len(strikes_dict)}")
                    
                    for strike_key in strikes_dict.keys():
                        try:
                            strike_float = float(strike_key)
                            available_strikes.add(strike_float)
                        except ValueError:
                            logger.warning(f"   ⚠️ Strike inválido ignorado: {strike_key}")
                            continue
        
        if not available_strikes:
            logger.error(f"❌ No se encontraron strikes para {exp_date_str}")
            logger.error("   Resumen de fechas disponibles:")
            
            all_dates = set()
            for map_type in ['callExpDateMap', 'putExpDateMap']:
                exp_map = data.get(map_type, {})
                all_dates.update(exp_map.keys())
            
            for idx, d in enumerate(sorted(all_dates)[:20], 1):
                logger.error(f"     {idx}. {d}")
            if len(all_dates) > 20:
                logger.error(f"     ... y {len(all_dates) - 20} fechas más")
            
            logger.info("="*80)
            return None
        
        # Encontrar el strike más cercano al precio actual
        available_strikes_list = sorted(available_strikes)
        
        logger.info(f"📊 Strikes disponibles totales: {len(available_strikes_list)}")
        logger.info(f"   Rango: ${min(available_strikes_list):.2f} - ${max(available_strikes_list):.2f}")
        
        atm_strike = min(available_strikes, key=lambda x: abs(x - current_price))
        
        logger.info(f"✅ Strike ATM seleccionado: ${atm_strike:.2f}")
        logger.info(f"   (más cercano a precio actual: ${current_price:.2f})")
        logger.info(f"   Diferencia: ${abs(atm_strike - current_price):.2f}")
        logger.info("="*80)
        
        return atm_strike
        
    except Exception as e:
        logger.error(f"❌ EXCEPCIÓN en get_atm_strike_schwab: {str(e)}")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")
        logger.info("="*80)
        return None
