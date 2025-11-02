# pages/FF Scanner.py - VERSIÓN CON AUTENTICACIÓN Y BOTÓN DE ACTUALIZACIÓN

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
import schwab
from schwab.auth import easy_client
from schwab.client import Client
from utils import check_password

# =========================================================================
# 0. CONFIGURACIÓN Y VARIABLES
# =========================================================================

st.set_page_config(page_title="FF Scanner", layout="wide")

# Variables de Schwab (se mantienen, aunque no se usen aún)
api_key = "n9ydCRbM3Gv5bBAGA1ZvVl6GAqo5IG9So6pMwjO9slvJXEa6"
app_secret = "DAFletN79meCi4yBYGzlDvlrNcJiISH0HuMuThydxYANTWghMxXxXbrpQOVjsdsx"
redirect_uri = "https://127.0.0.1" 
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
# 2. FUNCIÓN PRINCIPAL DE LA PÁGINA (FF Scanner)
# =========================================================================

def ff_scanner_page():
    st.title("🛡️ FF Scanner (Preparación y Validación de Tickers)")
    st.markdown("---")
    
    # Contenedor para el botón
    col1, col2 = st.columns([1, 4])

    with col1:
        # st.cache_resource se borrará al hacer clic en este botón, forzando la re-ejecución
        st.button("🔄 Actualizar/Validar Tickers", 
                  type="primary",
                  help="Borra la caché y fuerza la re-lectura de Tickers.csv y la re-descarga del S&P 500.",
                  on_click=perform_initial_preparation.clear)

    with col2:
        st.markdown("_(La validación se ejecuta automáticamente cada 24h, o al hacer clic en el botón)_")

    st.divider()

    # FASE 1: Preparación (Se llama aquí, pero se ejecuta desde la caché, a menos que se borre)
    valid_tickers = perform_initial_preparation()

# =========================================================================
# 2. PUNTO DE ENTRADA PROTEGIDO (CON AUTENTICACIÓN)
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
