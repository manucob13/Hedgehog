# pages/FF Scanner.py - VERSIÓN FINAL CORREGIDA (usa token existente, sin puerto)
import streamlit as st
import pandas as pd
import requests
import yfinance as yf
from datetime import timedelta
from io import StringIO
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
import time
import schwab
from schwab.auth import easy_client
from schwab.client import Client
from utils import check_password
from urllib.parse import urlparse, parse_qs

# =========================================================================
# 0. CONFIGURACIÓN Y VARIABLES
# =========================================================================

st.set_page_config(page_title="FF Scanner", layout="wide")

# Cargar variables de Schwab desde secrets
try:
    api_key = st.secrets["schwab"]["api_key"]
    app_secret = st.secrets["schwab"]["app_secret"]
    redirect_uri = st.secrets["schwab"]["redirect_uri"]
except KeyError as e:
    st.error(f"❌ Falta configurar los secrets de Schwab. Clave faltante: {e}. Asegúrate de que tienes [schwab] en secrets.toml")
    st.stop()

# Ruta local del token
token_path = "schwab_token.json"

# =========================================================================
# 1. PREPARACIÓN DE TICKERS
# =========================================================================

def is_valid_ticker(ticker):
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
    st.subheader("1. Preparación y Validación de Tickers")

    status_text = st.empty()

    # 1.1 Leer tickers existentes
    existing_tickers = set()
    if os.path.exists('Tickers.csv'):
        df_existing = pd.read_csv('Tickers.csv')
        existing_tickers = set(df_existing.iloc[:, 0].astype(str).str.upper().str.strip())
        st.info(f"✅ 'Tickers.csv' encontrado con {len(existing_tickers)} tickers.")
    else:
        st.warning("⚠️ 'Tickers.csv' no encontrado. Iniciando desde cero.")

    # 1.2 Descargar tickers del S&P 500
    try:
        status_text.text("Descargando lista del S&P 500...")
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        sp500_df = pd.read_html(StringIO(response.text))[0]
        sp500_tickers = set(sp500_df['Symbol'].astype(str).str.upper().str.strip())
        st.success(f"✅ Obtenidos {len(sp500_tickers)} tickers del S&P 500.")
    except Exception as e:
        st.error(f"❌ Error al descargar el S&P 500: {e}")
        sp500_tickers = set()

    all_tickers = sp500_tickers.union(existing_tickers)
    st.info(f"Validando {len(all_tickers)} tickers con yfinance...")

    progress_bar = st.progress(0)
    valid_tickers = []
    sorted_tickers = sorted(all_tickers)

    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(is_valid_ticker, t): t for t in sorted_tickers}
        for i, future in enumerate(futures):
            result = future.result()
            if result:
                valid_tickers.append(result)
            progress_bar.progress((i + 1) / len(sorted_tickers))

    progress_bar.empty()

    valid_tickers = sorted(set(valid_tickers))
    invalid_tickers = sorted(set(all_tickers) - set(valid_tickers))

    try:
        pd.DataFrame({'Ticker': valid_tickers}).to_csv('Tickers.csv', index=False)
        pd.DataFrame({'Ticker': invalid_tickers}).to_csv('Tickers_invalidos.csv', index=False)
    except Exception as e:
        st.warning(f"⚠️ No se pudieron guardar los CSV: {e}")

    st.success(f"✅ Validación finalizada con {len(valid_tickers)} tickers válidos.")
    st.divider()

    return valid_tickers

# =========================================================================
# 2. CONEXIÓN CON BROKER SCHWAB
# =========================================================================

def connect_to_schwab():
    """
    Usa el token existente si está disponible.
    Si no existe, ofrece un flujo manual de autenticación.
    """
    st.subheader("2. Conexión con Broker Schwab")

    try:
        client = easy_client(
            api_key=api_key,
            app_secret=app_secret,
            callback_url=redirect_uri,
            token_path=token_path
        )
    except Exception as e:
        st.error(f"❌ Error al inicializar Schwab Client: {e}")
        return None

    # Si ya existe token, probarlo
    if os.path.exists(token_path):
        try:
            test_response = client.get_quote("AAPL")
            if test_response.status_code == 200:
                st.success("✅ Conexión a Schwab verificada (token activo).")
                return client
            else:
                raise Exception(f"Respuesta inesperada: {test_response.status_code}")
        except Exception as e:
            st.warning(f"⚠️ El token puede haber expirado: {e}")
            if st.button("🗑️ Eliminar token y regenerar", key="regen"):
                os.remove(token_path)
                st.rerun()
            return None

    # Si no hay token, autenticación manual
    st.warning("⚠️ No se encontró un token. Genera uno nuevo desde la URL de autorización.")

    try:
        auth_url = client.oauth.get_oauth_url(redirect_uri=redirect_uri)
        st.markdown(f"[🔗 Autorizar con Schwab]({auth_url})")
    except Exception as e:
        st.error(f"❌ Error al generar URL de autorización: {e}")
        return None

    callback_url = st.text_input("Pega aquí la URL completa del callback:", key="cb_url")

    if st.button("🔐 Generar Token y Conectar", type="primary"):
        if not callback_url.startswith("https://127.0.0.1"):
            st.error("❌ Pega la URL completa que inicia con https://127.0.0.1/?code=")
        else:
            try:
                with st.spinner("Generando token..."):
                    client.oauth.from_callback_url(callback_url)
                    if os.path.exists(token_path):
                        st.success("✅ Token guardado exitosamente. Recarga para continuar.")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ La API respondió, pero el archivo de token no se creó.")
            except Exception as e:
                st.error(f"❌ Error al generar token: {e}")
    return None

# =========================================================================
# 3. FUNCIÓN PRINCIPAL
# =========================================================================

def ff_scanner_page():
    st.title("🛡️ FF Scanner - Preparación y Conexión")
    st.markdown("---")

    col1, col2 = st.columns([1, 4])
    with col1:
        st.button("🔄 Actualizar/Validar Tickers", type="primary",
                  help="Borra la caché y fuerza la re-lectura de Tickers.csv",
                  on_click=perform_initial_preparation.clear)
    with col2:
        st.markdown("_(Se valida automáticamente cada 24h o al pulsar el botón.)_")

    st.divider()

    valid_tickers = perform_initial_preparation()
    st.divider()
    schwab_client = connect_to_schwab()

    if schwab_client:
        st.success(f"🎯 Sistema listo con {len(valid_tickers)} tickers válidos y conexión Schwab activa.")
    else:
        st.info("⏳ Completa la conexión con Schwab para activar funciones de trading.")

# =========================================================================
# 4. PUNTO DE ENTRADA PROTEGIDO
# =========================================================================

if __name__ == "__main__":
    if check_password():
        ff_scanner_page()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Introduce tus credenciales en el menú lateral para acceder.")
