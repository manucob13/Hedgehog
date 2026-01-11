"""
Utilidades para interactuar con la API de CBOE (Chicago Board Options Exchange)
Este módulo contiene la función para obtener datos de opciones desde CBOE.
"""

import streamlit as st
import pandas as pd
import requests
from datetime import datetime


@st.cache_data(ttl=300)
def get_option_chain_cboe(ticker):
    """
    Descarga la cadena de opciones desde CBOE.
    
    Args:
        ticker (str): Símbolo del ticker (ej: 'SPX', 'QQQ')
    
    Returns:
        pd.DataFrame: DataFrame con los datos de opciones procesados, o None si hay error
        
    Columns del DataFrame retornado:
        - option: Código de la opción
        - expiry: Fecha de expiración (date object)
        - opt_type: Tipo de opción ('C' para Call, 'P' para Put)
        - strike: Precio de ejercicio (strike)
        - Mid_price: Precio medio (bid + ask) / 2
        - bid: Precio bid
        - ask: Precio ask
        - last_trade_price: Último precio de operación
        - volume: Volumen
        - open_interest: Interés abierto
        - Y otros campos disponibles en la API de CBOE
    """
    try:
        url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{ticker}.json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'data' in data and 'options' in data['data']:
            options_data = data['data']['options']
            df = pd.DataFrame(options_data)
            
            # Procesar los datos de opciones
            # El formato del código de opción es: TICKER + YYMMDD + C/P + STRIKE
            # Ejemplo: SPX240315C05000000 -> SPX, 24/03/15, Call, 5000.00
            
            # Extraer fecha de expiración (posiciones -15 a -9)
            df['expiry'] = df['option'].str[-15:-9].apply(
                lambda x: datetime.strptime(x, "%y%m%d").date()
            )
            
            # Extraer tipo de opción (posición -9: 'C' o 'P')
            df['opt_type'] = df['option'].str[-9]
            
            # Extraer strike (últimos 8 caracteres)
            # Formato: XXXXX.XXX (5 enteros + 3 decimales)
            df['strike'] = df['option'].str[-8:].apply(
                lambda x: float(f"{x[:5]}.{x[5:]}")
            )
            
            # Calcular precio medio
            df['Mid_price'] = (df['bid'] + df['ask']) / 2
            
            return df
        
        return None
        
    except requests.exceptions.Timeout:
        st.error(f"⏱️ Timeout al conectar con CBOE para {ticker}. Intenta nuevamente.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ Error HTTP al descargar opciones de CBOE: {e}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error de conexión con CBOE: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Error procesando opciones de CBOE: {e}")
        return None
