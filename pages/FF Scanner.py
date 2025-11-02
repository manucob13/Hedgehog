# pages/FF Scanner.py - VERSIÓN CORREGIDA: PREPARACIÓN Y CONEXIÓN

import streamlit as st
import pandas as pd
import requests
import yfinance as yf
from datetime import timedelta, date, datetime 
from io import StringIO
from concurrent.futures import ThreadPoolExecutor
import os 
from utils import check_password

# =========================================================================
# 0. CONFIGURACIÓN Y VARIABLES
# =========================================================================

st.set_page_config(page_title="FF Scanner", layout="wide")

# Variables de Schwab (cargadas desde secrets)
try:
    api_key = st.secrets["schwab"]["api_key"]
    app_secret = st.secrets["schwab"]["app_secret"]
    redirect_uri = st.secrets["schwab"]["redirect_uri"]
except KeyError as e:
    st.error(f"❌ Error: Falta configurar los secrets de Schwab. Clave faltante: {e}")
    st.stop()

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

@st.cache_resource(ttl=timedelta(hours=24), show_spinner=False)
def perform_initial_preparation():
    """Realiza la lectura, descarga y validación en PARALELO de tickers."""
    st.subheader("1. Preparación y Validación de Tickers")
    
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
            st.warning("⚠️ Archivo 'Tickers.csv' NO ENCONTRADO. Iniciando con 0 tickers existentes.")
    except Exception as e:
        st.error(f"❌ Error al leer 'Tickers.csv'. Error: {e}")
        
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
    status_text.text(f"3. Validando {len(all_tickers)} tickers con yfinance...")
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
            status_text.text(f"3. Validando tickers: {i + 1}/{len(sorted_tickers)} procesados. Válidos: {len(valid_tickers)}")

    progress_bar.empty()
    status_text.empty()
    
    # 1.5 Guardar y Resumir
    valid_tickers = sorted(set(valid_tickers))
    invalid_tickers = sorted(set(all_tickers) - set(valid_tickers))

    try:
        pd.DataFrame({'Ticker': valid_tickers}).to_csv('Tickers.csv', index=False)
        pd.DataFrame({'Ticker': invalid_tickers}).to_csv('Tickers_invalidos.csv', index=False)
    except Exception as e:
        st.warning(f"⚠️ No se pudieron guardar los archivos. Error: {e}")

    valid_count = len(valid_tickers)
    invalid_count = len(invalid_tickers)

    st.success(f"✅ Validación completada.")
    st.markdown(f"**✅ {valid_count} tickers válidos | 🗑️ {invalid_count} inválidos**")
    st.divider() 
    
    return valid_tickers

# =========================================================================
# 2. CONEXIÓN CON BROKER SCHWAB
# =========================================================================

def connect_to_schwab():
    """
    Intenta conectar con Schwab API.
    - Si existe el token, lo usa
    - Si no existe, muestra proceso de generación manual
    """
    st.subheader("2. Conexión con Broker Schwab")
    
    # Verificar si schwab-py está instalado y obtener las funciones CORRECTAS
    try:
        # Modificado: Ahora importamos client_from_token_file y SchwabOauth
        from schwab.auth import client_from_token_file, SchwabOauth
        from urllib.parse import urlparse, parse_qs # Necesaria para extraer el código
    except ImportError:
        st.error("❌ La librería 'schwab-py' no está instalada.")
        st.code("pip install schwab-py", language="bash")
        st.stop()
    
    # Verificar si existe el archivo de token
    if os.path.exists(token_path):
        st.info(f"📄 Archivo de token encontrado: `{token_path}`")
        
        try:
            with st.spinner("🔐 Conectando con Schwab API..."):
                client = client_from_token_file(
                    token_path=token_path,
                    api_key=api_key,
                    app_secret=app_secret
                )
            
            st.success("✅ Conexión a Schwab API establecida correctamente.")
            
            # Verificar con petición de prueba
            try:
                test_response = client.get_quote("AAPL")
                if test_response.status_code == 200:
                    st.success("✅ Token válido - Conexión verificada.")
                else:
                    st.warning(f"⚠️ Respuesta inesperada: {test_response.status_code}")
            except Exception as e:
                st.warning(f"⚠️ Error al verificar la conexión: {e}")
            
            return client
            
        except Exception as e:
            st.error(f"❌ Error al conectar: {e}")
            st.warning("⚠️ El token puede haber expirado. Necesitas regenerarlo.")
            # Borrar el token inválido
            if st.button("🗑️ Eliminar token inválido y regenerar"):
                os.remove(token_path)
                st.rerun()
            return None
    
    else:
        # El token no existe - proceso de generación manual
        st.warning(f"⚠️ No se encontró el archivo de token: `{token_path}`")
        
        st.markdown("""
        ### 🔧 Generación de Token - Proceso Manual
        
        Como estás en Streamlit Cloud, vamos a generar el token manualmente siguiendo estos pasos:
        """)
        
        # Paso 1: Mostrar la URL de autorización
        auth_url = f"https://api.schwabapi.com/v1/oauth/authorize?response_type=code&client_id={api_key}&redirect_uri={redirect_uri}"
        
        st.markdown("#### Paso 1: Autorización")
        st.markdown(f"Haz clic en este enlace para autorizar la aplicación:")
        st.markdown(f"[🔗 Autorizar con Schwab]({auth_url})")
        
        st.info("""
        - Se abrirá la página de Schwab
        - Inicia sesión con tus credenciales
        - Autoriza la aplicación
        - Serás redirigido a una página que NO carga (es normal)
        """)
        
        # Paso 2: Capturar la URL de callback
        st.markdown("#### Paso 2: Copiar URL de Callback")
        st.markdown("""
        Después de autorizar, tu navegador intentará ir a `https://127.0.0.1/?code=...`
        
        La página NO cargará, pero la URL es lo importante. Copia **TODA la URL** de la barra de direcciones.
        """)
        
        callback_url = st.text_input(
            "Pega aquí la URL completa de callback:",
            placeholder="https://127.0.0.1/?code=C0.b2F1dGgyLm...",
            key="callback_url_input"
        )
        
        # Paso 3: Generar el token (Lógica CORREGIDA)
        if st.button("🔐 Generar Token", type="primary"):
            if not callback_url or not callback_url.startswith("https://127.0.0.1"):
                st.error("❌ Por favor, pega la URL de callback completa.")
            else:
                try:
                    with st.spinner("Generando token..."):
                        # Extraer el código de la URL
                        parsed_url = urlparse(callback_url)
                        code = parse_qs(parsed_url.query).get('code', [None])[0]
                        
                        if not code:
                            st.error("❌ No se pudo extraer el código de autorización de la URL.")
                            st.stop()
                        
                        # **CÓDIGO CORREGIDO:** Usamos SchwabOauth para el intercambio
                        oauth = SchwabOauth(
                            client_id=api_key, 
                            client_secret=app_secret, 
                            redirect_uri=redirect_uri,
                            token_path=token_path
                        )
                        
                        # Esto intercambia el código por tokens y los guarda en 'schwab_token.json'
                        oauth.generate_tokens_from_code(code)
                        # **FIN DEL CÓDIGO CORREGIDO**

                    st.success("✅ Token generado y guardado exitosamente!")
                    st.balloons()
                    
                    # El token ya está guardado. Recargamos para que el código principal lo use
                    st.info("🔄 Recarga la página para verificar la conexión y continuar.")
                    st.rerun() # Esto hará que la siguiente ejecución use client_from_token_file

                except Exception as e:
                    st.error(f"❌ Error al generar el token: {e}")
                    st.markdown("""
                    **Posibles causas:**
                    - La URL de callback no es correcta
                    - El código de autorización ya fue usado (genera uno nuevo)
                    - Las credenciales API son incorrectas
                    """)
        
        return None

# =========================================================================
# 3. FUNCIÓN PRINCIPAL
# =========================================================================

def ff_scanner_page():
    st.title("🛡️ FF Scanner - Preparación y Conexión")
    st.markdown("---")
    
    # Botón de actualización
    col1, col2 = st.columns([1, 4])
    with col1:
        st.button("🔄 Actualizar Tickers", 
                  type="primary",
                  on_click=perform_initial_preparation.clear)
    with col2:
        st.markdown("_(La validación se ejecuta cada 24h o al hacer clic)_")

    st.divider()

    # FASE 1: Preparación
    valid_tickers = perform_initial_preparation()
    
    st.divider()
    
    # FASE 2: Conexión con Schwab
    client = connect_to_schwab()
    
    # Mensaje final
    if client:
        st.success(f"🎯 Sistema listo con {len(valid_tickers)} tickers válidos y conexión Schwab activa.")
    else:
        st.info("⏳ Completa la conexión con Schwab para continuar.")

# =========================================================================
# 4. PUNTO DE ENTRADA
# =========================================================================

if __name__ == "__main__":
    if check_password():
        ff_scanner_page()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar).")
