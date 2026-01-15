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
    """
    Obtiene el strike ATM (at-the-money) más cercano al precio actual.
    VERSIÓN MEJORADA con mejor debugging y manejo de fechas.
    
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
        
        print(f"\n{'='*60}")
        print(f"GET ATM STRIKE - INICIO")
        print(f"Ticker original: {ticker}")
        print(f"Symbol normalizado: {symbol}")
        print(f"Fecha objetivo: {target_date}")
        print(f"Precio actual: {current_price}")
        print(f"{'='*60}\n")
        
        # NUEVO: Ejecutar diagnóstico primero
        print("⚙️ Ejecutando diagnóstico de cadena de opciones...")
        diagnose_option_chain(client, ticker)
        
        # Llamar con el símbolo normalizado
        print(f"📡 Llamando a get_option_chain({symbol})...")
        response = client.get_option_chain(symbol)
        
        if response.status_code != 200:
            print(f"❌ Error en get_option_chain: Status code {response.status_code}")
            print(f"Response text: {response.text[:1000]}")
            return None
        
        print(f"✅ Respuesta recibida de Schwab API")
        
        data = response.json()
        
        # Verificar qué keys están disponibles en la respuesta
        print(f"\n📋 Keys principales en respuesta: {list(data.keys())}")
        
        available_strikes = set()
        exp_date_str = target_date.strftime("%Y-%m-%d")
        
        print(f"\n🔍 Buscando strikes para fecha: {exp_date_str}")
        
        fechas_encontradas = []
        
        for map_type in ['callExpDateMap', 'putExpDateMap']:
            exp_map = data.get(map_type, {})
            
            if not exp_map:
                print(f"⚠️ {map_type} está vacío o no existe")
                continue
            
            print(f"\n📊 Procesando {map_type}")
            print(f"   Total fechas disponibles: {len(exp_map)}")
            
            # Mostrar las primeras 10 fechas para debugging
            all_dates = list(exp_map.keys())
            print(f"   Primeras 10 fechas: {all_dates[:10]}")
            
            fecha_match_found = False
            
            for date_key, strikes_dict in exp_map.items():
                # Buscar coincidencia exacta o parcial
                if date_key.startswith(exp_date_str):
                    print(f"   ✅ MATCH encontrado: {date_key}")
                    fechas_encontradas.append(date_key)
                    fecha_match_found = True
                    
                    # Extraer strikes
                    for strike_key in strikes_dict.keys():
                        try:
                            strike_float = float(strike_key)
                            available_strikes.add(strike_float)
                        except ValueError:
                            print(f"      ⚠️ Strike no numérico: {strike_key}")
                            continue
            
            if not fecha_match_found:
                print(f"   ❌ No se encontró match para {exp_date_str} en {map_type}")
        
        if not available_strikes:
            print(f"\n❌ ERROR: No se encontraron strikes para la fecha {exp_date_str}")
            print(f"\nFechas que sí están disponibles (primeras 20):")
            
            all_available_dates = set()
            for map_type in ['callExpDateMap', 'putExpDateMap']:
                exp_map = data.get(map_type, {})
                all_available_dates.update(list(exp_map.keys())[:20])
            
            for fecha in sorted(list(all_available_dates))[:20]:
                print(f"   - {fecha}")
            
            print(f"\n💡 Sugerencia: Usa una de estas fechas disponibles")
            return None
        
        available_strikes_list = sorted(list(available_strikes))
        
        print(f"\n✅ Strikes encontrados: {len(available_strikes_list)}")
        print(f"   Rango: {min(available_strikes_list):.2f} - {max(available_strikes_list):.2f}")
        print(f"   Primeros 10: {available_strikes_list[:10]}")
        print(f"   Últimos 10: {available_strikes_list[-10:]}")
        
        # Encontrar el strike más cercano al precio actual
        atm_strike = min(available_strikes, key=lambda x: abs(x - current_price))
        
        diferencia = abs(atm_strike - current_price)
        diferencia_pct = (diferencia / current_price) * 100
        
        print(f"\n{'='*60}")
        print(f"RESULTADO ATM STRIKE")
        print(f"Strike ATM seleccionado: {atm_strike}")
        print(f"Precio actual: {current_price}")
        print(f"Diferencia: ${diferencia:.2f} ({diferencia_pct:.3f}%)")
        print(f"Fechas procesadas: {fechas_encontradas}")
        print(f"{'='*60}\n")
        
        return atm_strike
        
    except Exception as e:
        print(f"\n❌ EXCEPCIÓN en get_atm_strike_schwab: {e}")
        import traceback
        traceback.print_exc()
        return None
