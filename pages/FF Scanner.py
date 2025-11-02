# pages/FF Scanner.py - VERSIÓN CON AUTENTICACIÓN Y CONEXIÓN SCHWAB

import streamlit as st
import pandas as pd
import requests
import yfinance as yf
from datetime import timedelta, date, datetime 
from io import StringIO
from concurrent.futures import ThreadPoolExecutor
import numpy as np 
import os 
import time 
# Importaciones que SÍ funcionan:
import schwab
from schwab.auth import easy_client
from schwab.client import Client
from utils import check_password
from urllib.parse import urlparse, parse_qs # Necesario para procesar la URL de callback

# =========================================================================
# 0. CONFIGURACIÓN Y VARIABLES
# =========================================================================

st.set_page_config(page_title="FF Scanner", layout="wide")

# **IMPORTANTE:** Cargar variables de Schwab desde st.secrets
try:
    api_key = st.secrets["schwab"]["api_key"]
    app_secret = st.secrets["schwab"]["app_secret"]
    redirect_uri = st.secrets["schwab"]["redirect_uri"] 
except KeyError as e:
    st.error(f"❌ Error: Falta configurar los secrets de Schwab. Clave faltante: {e}. Asegúrate de que tienes [schwab] en secrets.toml")
    st.stop()

# Ruta local del token
token_path = "schwab_token.json" 

# =========================================================================
# 1. FASE DE PREPARACIÓN (Validación de Tickers)
# =========================================================================

def is_valid_ticker(ticker):
    """Verifica si un ticker es válido usando yfinance."""
    try:
        t = yf.Ticker(ticker)
        fi = getattr(t, "fast_info", None)
        if fi and isinstance(fi, dict) and fi.get('last_price') is not None:
            return ticker
        info = t.info 
        if isinstance(info, dict) and (info.get('regularMarketPrice') is not None or info.get('previousClose') is not None):
            return ticker
    except Exception:
        return None
    return None

# Usamos st.cache_resource con clear_on_click para permitir la repetición
@st.cache_resource(ttl=timedelta(hours=24), show_spinner=False)
def perform_initial_preparation():
    """Realiza la lectura, descarga y validación en PARALELO de tickers."""
    st.subheader("1. Preparación y Validación de Tickers")
    
    # Placeholder para mensajes de estado
    status_text = st.empty()
    
    # 1.1 Leer Tickers.csv existentes
    status_text.text("1. Leyendo tickers existentes (Tickers.csv)...")
    existing_tickers = set()
    try:
        if os.path.exists('Tickers.csv'):
            df_existing = pd.read_csv('Tickers.csv')
            existing_tickers = set(df_existing.iloc[:, 0].astype(str).str.upper().str.strip())
            st.info(f"✅ Se encontró 'Tickers.csv'. Leídos **{len(existing_tickers)}** tickers existentes.")
        else:
            # Ahora que el usuario movió el archivo, este mensaje solo aparecerá si falta
            st.warning("⚠️ Archivo 'Tickers.csv' NO ENCONTRADO en el repositorio. Iniciando con 0 tickers existentes.")
    except Exception as e:
        st.error(f"❌ Error crítico al leer 'Tickers.csv'. Error: {e}")
        
    # 1.2 Descargar tickers del S&P 500
    status_text.text("2. Descargando lista de tickers del S&P 500 de Wikipedia...")
    sp500_tickers = set()
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        sp500_df = pd.read_html(StringIO(response.text))[0] 
        sp500_tickers = set(sp500_df['Symbol'].astype(str).str.upper().str.strip())
        st.success(f"✅ Obtenidos {len(sp500_tickers)} tickers del S&P 500.")
    except Exception as e:
        st.error(f"❌ Error al descargar el S&P 500. Usando solo tickers existentes. Error: {e}")

    # 1.3 Combinar
    all_tickers = sp500_tickers.union(existing_tickers)
    st.info(f"Total de tickers combinados a validar: **{len(all_tickers)}**")
    
    # 1.4 Validar en PARALELO
    status_text.text(f"3. Validando {len(all_tickers)} tickers con yfinance (esto será rápido)...")
    progress_bar = st.progress(0)
    
    valid_tickers = []
    sorted_tickers = sorted(all_tickers)
    
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(is_valid_ticker, ticker): ticker for ticker in sorted_tickers}
        
        for i, future in enumerate(futures):
            result = future.result()
            if result:
                valid_tickers.append(result)
            progress_bar.progress((i + 1) / len(sorted_tickers))
            status_text.text(f"3. Validando tickers: {i + 1}/{len(sorted_tickers)} procesados. Válidos encontrados: {len(valid_tickers)}")

    progress_bar.empty()
    status_text.empty()
    
    # 1.5 Guardar y Resumir
    valid_tickers = sorted(set(valid_tickers))
    invalid_tickers = sorted(set(all_tickers) - set(valid_tickers))

    try:
        pd.DataFrame({'Ticker': valid_tickers}).to_csv('Tickers.csv', index=False)
        pd.DataFrame({'Ticker': invalid_tickers}).to_csv('Tickers_invalidos.csv', index=False)
    except Exception as e:
        st.warning(f"⚠️ No se pudieron guardar Tickers.csv/Tickers_invalidos.csv. Error: {e}")

    valid_count = len(valid_tickers)
    invalid_count = len(invalid_tickers)

    st.success(f"✅ Validación de preparación finalizada.")
    st.markdown(f"**✅ {valid_count} tickers válidos guardados en 'Tickers.csv'**")
    st.markdown(f"**🗑️ Eliminados: {invalid_count} inválidos**")
    st.divider() 
    
    return valid_tickers

# =========================================================================
# 2. CONEXIÓN CON BROKER SCHWAB (LÓGICA CORREGIDA con easy_client)
# =========================================================================

def connect_to_schwab():
    """
    Intenta conectar con Schwab API usando el método easy_client
    que maneja el flujo de token y la ruta del archivo.
    """
    st.subheader("2. Conexión con Broker Schwab")
    
    # 1. Inicializar el cliente usando easy_client
    try:
        # easy_client intentará cargar el token existente.
        client = easy_client(
            token_path=token_path,
            api_key=api_key,
            app_secret=app_secret,
            callback_url=redirect_uri # Se requiere 'callback_url' para esta versión de schwab-py.
        )
    except Exception as e:
        # En caso de un error de inicialización, probablemente por credenciales incorrectas
        st.error(f"❌ Error crítico al inicializar el cliente Schwab: {e}")
        st.info("Revisa que `api_key` y `app_secret` sean correctos en tus secretos de Streamlit.")
        return None

    # 2. Verificar la existencia del token y la validez de la conexión
    if os.path.exists(token_path):
        st.info(f"📄 Archivo de token encontrado: `{token_path}`. Verificando conexión...")
        try:
            # Prueba de conexión simple
            test_response = client.get_quote("AAPL")
            if test_response.status_code == 200:
                st.success("✅ Conexión a Schwab API verificada: Token ACTIVO.")
                return client
            else:
                # Si el estado no es 200, el token puede ser inválido o haber expirado
                raise Exception(f"Respuesta API inesperada: {test_response.status_code}")
                
        except Exception as e:
            st.error(f"❌ Error al usar el token existente: {e}")
            st.warning("⚠️ El token puede haber expirado. Por favor, regenera el token.")
            if st.button("🗑️ Eliminar token y regenerar", key="delete_token"):
                os.remove(token_path)
                st.rerun()
            return None

    # 3. Si NO existe el token, mostrar el proceso de autenticación manual
    st.warning(f"⚠️ No se encontró el archivo de token: `{token_path}`. Inicia la autenticación.")
    
    # Obtener la URL de autorización
    auth_url = client.oauth.get_oauth_url(redirect_uri=redirect_uri) 
    
    st.markdown("---")
    st.markdown("### 🔧 Generación de Token - Proceso Manual")
    
    st.markdown("#### Paso 1: Autorización")
    st.markdown(f"Haz clic en este enlace para autorizar la aplicación:")
    st.markdown(f"[🔗 Autorizar con Schwab]({auth_url})")
    
    st.info("""
    - Serás redirigido a una URL que **NO carga** (es normal).
    - **ATENCIÓN:** Copia **TODA la URL** de la barra de direcciones que comienza con `https://127.0.0.1/?code=...` **inmediatamente**. El código expira muy rápido.
    """)
    
    st.markdown("#### Paso 2: Copiar URL de Callback y Generar Token")
    callback_url = st.text_input(
        "Pega aquí la URL completa de callback:",
        placeholder="https://127.0.0.1/?code=C0.b2F1dGgyLm...",
        key="callback_url_input"
    )
    
    if st.button("🔐 Generar Token y Conectar", type="primary", key="generate_token"):
        if not callback_url or not callback_url.startswith("https://127.0.0.1"):
            st.error("❌ Por favor, pega la URL de callback completa y correcta.")
        else:
            try:
                with st.spinner("Generando y guardando token..."):
                    # Esto intercambia el código por tokens y los guarda en 'schwab_token.json'
                    client.oauth.from_callback_url(callback_url)
                    
                    # Verificación explícita (MEJORA):
                    if os.path.exists(token_path):
                        st.success("✅ Token generado y guardado exitosamente!")
                        st.info("🔄 Recarga la página para verificar la conexión y continuar.")
                        time.sleep(1) # Pequeña pausa para que el mensaje se vea
                        st.rerun() 
                    else:
                        st.error(f"❌ Error de guardado: La API respondió, pero el archivo '{token_path}' no se creó.")
                        
            except Exception as e:
                # Este error es el más probable si el código ha expirado o hubo un error de API (401 Unauthorized, etc.)
                st.error(f"❌ Error al intentar generar el token (Paso 2). Esto puede ser porque el código de autorización ha expirado o las credenciales (API Key/Secret) son incorrectas. Error detallado: {e}")
    
    return None

# =========================================================================
# 3. FUNCIÓN PRINCIPAL DE LA PÁGINA (FF Scanner)
# =========================================================================

def ff_scanner_page():
    st.title("🛡️ FF Scanner - Módulos de Preparación y Conexión")
    st.markdown("---")
    
    # Contenedor para el botón de validación de tickers
    col1, col2 = st.columns([1, 4])

    with col1:
        st.button("🔄 Actualizar/Validar Tickers", 
                  type="primary",
                  help="Borra la caché y fuerza la re-lectura de Tickers.csv y la re-descarga del S&P 500.",
                  on_click=perform_initial_preparation.clear)

    with col2:
        st.markdown("_(La validación se ejecuta automáticamente cada 24h, o al hacer clic en el botón)_")

    st.divider()

    # FASE 1: Preparación
    valid_tickers = perform_initial_preparation()
    
    st.divider()
    
    # FASE 2: Conexión con Schwab
    schwab_client = connect_to_schwab()
    
    # Mensaje final
    if schwab_client:
        st.success(f"🎯 Sistema listo con {len(valid_tickers)} tickers válidos y conexión Schwab activa.")
    else:
        st.info("⏳ Completa la conexión con Schwab para activar las funciones de trading.")


# =========================================================================
# 4. PUNTO DE ENTRADA PROTEGIDO (CON AUTENTICACIÓN)
# =========================================================================

if __name__ == "__main__":
    # LLAMADA AL LOGIN (Muestra el formulario si es necesario)
    if check_password():
        # SI EL LOGIN ES EXITOSO, EJECUTA LA APP PRINCIPAL
        ff_scanner_page()
    else:
        # Mensaje cuando no se ha autenticado
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder a la aplicación.")
