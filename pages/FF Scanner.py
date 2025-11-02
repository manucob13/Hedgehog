# pages/FF Scanner.py - VERSIÓN SIMPLIFICADA: PREPARACIÓN Y CONEXIÓN

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
    - Si no existe, muestra instrucciones para generarlo
    """
    st.subheader("2. Conexión con Broker Schwab")
    
    # Verificar si schwab-py está instalado
    try:
        from schwab.auth import client_from_token_file
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
            return None
    
    else:
        # El token no existe - mostrar instrucciones
        st.warning(f"⚠️ No se encontró el archivo de token: `{token_path}`")
        
        st.markdown("""
        ### 📋 Instrucciones para generar el token
        
        **Ejecuta este código en tu computadora local** (no en Streamlit Cloud):
        
        ```python
        from schwab.auth import easy_client
        
        api_key = "n9ydCRbM3Gv5bBAGA1ZvVl6GAqo5IG9So6pMwjO9slvJXEa6"
        app_secret = "DAFletN79meCi4yBYGzlDvlrNcJiISH0HuMuThydxYANTWghMxXxXbrpQOVjsdsx"
        redirect_uri = "https://127.0.0.1"
        
        client = easy_client(
            api_key=api_key,
            app_secret=app_secret,
            callback_url=redirect_uri,
            token_path="schwab_token.json"
        )
        
        print("✅ Token generado en schwab_token.json")
        ```
        
        ### Pasos:
        1. Ejecuta el código anterior localmente
        2. Autentícate en el navegador con Schwab
        3. Copia la URL completa después de autenticarte
        4. Pégala cuando te lo pida
        5. Sube `schwab_token.json` a tu repositorio
        6. Recarga esta página
        
        ⚠️ **Nota:** Si tu repositorio es público, añade `schwab_token.json` al `.gitignore`
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
