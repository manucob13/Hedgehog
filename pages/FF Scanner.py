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
import datetime
import schwab
from schwab.auth import easy_client
from schwab.client import Client
from utils import check_password

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
# 3. FECHAS DE ENTRADA Y DTE
# =========================================================================

def get_next_thursday(today=None):
    """Devuelve el jueves de esta semana o el próximo si ya pasó."""
    if today is None:
        today = datetime.date.today()
    thursday = today + datetime.timedelta((3 - today.weekday()) % 7)
    # Si ya pasó jueves, sumamos 7 días
    if thursday <= today:
        thursday += datetime.timedelta(days=7)
    return thursday

def fechas_section():
    st.subheader("3. Fechas de Entrada y DTE")

    # Valores por defecto
    default_fecha = get_next_thursday()
    default_dte_front = 15
    default_dte_back = 22

    # Inputs
    fecha_entrada = st.date_input("Fecha de Entrada", value=default_fecha)
    dte_front = st.number_input("DTE Front", min_value=1, value=default_dte_front)
    dte_back = st.number_input("DTE Back", min_value=1, value=default_dte_back)

    # Cálculo fechas adicionales
    fecha_dte_front = fecha_entrada + datetime.timedelta(days=int(dte_front))
    fecha_dte_back = fecha_entrada + datetime.timedelta(days=int(dte_back))

    # Mostrar resultados
    st.markdown(f"- **DTE Front Fecha:** {fecha_dte_front}")
    st.markdown(f"- **DTE Back Fecha:** {fecha_dte_back}")

    return fecha_entrada, dte_front, dte_back, fecha_dte_front, fecha_dte_back

# =========================================================================
# 2. CONEXIÓN CON BROKER SCHWAB (solo usa token existente)
# =========================================================================

def connect_to_schwab():
    """
    Usa el token existente si está disponible.
    No abre flujo OAuth ni usa puerto; solo valida token.json.
    """
    st.subheader("2. Conexión con Broker Schwab")

    if not os.path.exists(token_path):
        st.error("❌ No se encontró 'schwab_token.json'. Genera el token desde tu notebook local antes de usar esta página.")
        return None

    try:
        client = easy_client(
            api_key=api_key,
            app_secret=app_secret,
            callback_url=redirect_uri,
            token_path=token_path
        )

        # Verificar token
        test_response = client.get_quote("AAPL")
        if hasattr(test_response, "status_code") and test_response.status_code != 200:
            raise Exception(f"Respuesta inesperada: {test_response.status_code}")

        st.success("✅ Conexión con Schwab verificada (token activo).")
        return client

    except Exception as e:
        st.error(f"❌ Error al inicializar Schwab Client: {e}")
        st.warning("⚠️ Si el error persiste, elimina el archivo 'schwab_token.json' y vuelve a generarlo desde tu entorno local.")
        return None

# =========================================================================
# 4. FUNCIÓN PRINCIPAL
# =========================================================================

def ff_scanner_page():
    st.title("🛡️ FF Scanner - Preparación y Conexión")
    st.markdown("---")

    # --- Punto 1: Preparación de Tickers ---
    col1, col2 = st.columns([1, 4])
    with col1:
        st.button("🔄 Actualizar/Validar Tickers", type="primary",
                  help="Borra la caché y fuerza la re-lectura de Tickers.csv",
                  on_click=perform_initial_preparation.clear)
    with col2:
        st.markdown("_(Se valida automáticamente cada 24h o al pulsar el botón.)_")

    st.divider()
    valid_tickers = perform_initial_preparation()

    # --- Punto 3: Fechas ---
    st.divider()
    fecha_entrada, dte_front, dte_back, fecha_dte_front, fecha_dte_back = fechas_section()

    # --- Punto 2: Conexión Schwab ---
    st.divider()
    schwab_client = connect_to_schwab()

    if schwab_client:
        st.success(f"🎯 Sistema listo con {len(valid_tickers)} tickers válidos y conexión Schwab activa.")
    else:
        st.info("⏳ Conecta tu token Schwab para activar funciones de trading.")

# =========================================================================
# 5. PUNTO DE ENTRADA PROTEGIDO
# =========================================================================

if __name__ == "__main__":
    if check_password():
        ff_scanner_page()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Introduce tus credenciales en el menú lateral para acceder.")

