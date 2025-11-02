# pages/FF Scanner.py - VERSIÓN SIMPLIFICADA Y CORREGIDA

import streamlit as st
import pandas as pd
import requests
import yfinance as yf
from datetime import timedelta
from io import StringIO
from concurrent.futures import ThreadPoolExecutor
import numpy as np 
import os # Necesario para la manipulación de archivos

# =========================================================================
# 0. CONFIGURACIÓN DE LA PÁGINA
# =========================================================================

st.set_page_config(page_title="FF Scanner", layout="wide")

# =========================================================================
# 1. FUNCIONES AUXILIARES (Validación)
# =========================================================================

def is_valid_ticker(ticker):
    """Verifica si un ticker es válido usando yfinance."""
    try:
        t = yf.Ticker(ticker)
        # Intentamos obtener la información de forma eficiente
        fi = getattr(t, "fast_info", None)
        if fi and isinstance(fi, dict) and fi.get('last_price') is not None:
            return ticker
        # Caída de respaldo si fast_info falla
        info = t.info 
        if isinstance(info, dict) and (info.get('regularMarketPrice') is not None or info.get('previousClose') is not None):
            return ticker
    except Exception:
        return None
    return None

# Usamos st.cache_resource para asegurar que esta tarea costosa se ejecuta 
# a lo sumo una vez al día (TTL de 24 horas).
@st.cache_resource(ttl=timedelta(hours=24), show_spinner=False)
def perform_initial_preparation():
    """
    Realiza la lectura, descarga y validación en paralelo de tickers.
    """
    st.subheader("1. Preparación y Validación de Tickers")
    
    # Placeholder para mensajes de estado
    status_text = st.empty()
    
    # 1.1 Leer Tickers.csv existentes
    status_text.text("1. Leyendo tickers existentes (Tickers.csv)...")
    try:
        if os.path.exists('Tickers.csv'):
            df_existing = pd.read_csv('Tickers.csv')
            # Limpieza y upper case
            existing_tickers = set(df_existing.iloc[:, 0].astype(str).str.upper().str.strip())
        else:
            existing_tickers = set()
    except Exception:
        existing_tickers = set()
        
    st.success(f"✅ Leídos {len(existing_tickers)} tickers existentes.")


    # 1.2 Descargar tickers del S&P 500
    # NOTA: Esto ahora funciona gracias a que agregamos 'lxml' a requirements.txt
    status_text.text("2. Descargando lista de tickers del S&P 500 de Wikipedia...")
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        sp500_df = pd.read_html(StringIO(response.text))[0]
        sp500_tickers = set(sp500_df['Symbol'].astype(str).str.upper().str.strip())
        st.success(f"✅ Obtenidos {len(sp500_tickers)} tickers del S&P 500.")
    except Exception as e:
        st.error(f"❌ Error al descargar el S&P 500. Usando solo tickers existentes. Error: {e}")
        sp500_tickers = set()

    # 1.3 Combinar
    # La unión asegura que tus tickers existentes se mantengan y se añadan los nuevos.
    all_tickers = sp500_tickers.union(existing_tickers)
    st.info(f"Total de tickers combinados a validar: **{len(all_tickers)}**")
    
    # 1.4 Validar en paralelo
    status_text.text(f"3. Validando {len(all_tickers)} tickers con yfinance (esto puede tardar varios minutos)...")
    progress_bar = st.progress(0)
    
    valid_tickers = []
    sorted_tickers = sorted(all_tickers)
    
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(is_valid_ticker, ticker): ticker for ticker in sorted_tickers}
        
        for i, future in enumerate(futures):
            # Obtener resultado
            result = future.result()
            if result:
                valid_tickers.append(result)
                
            # Actualizar progreso y estado
            progress_bar.progress((i + 1) / len(sorted_tickers))
            status_text.text(f"3. Validando tickers: {i + 1}/{len(sorted_tickers)} procesados. Válidos encontrados: {len(valid_tickers)}")

    progress_bar.empty()
    status_text.empty()
    
    # --- 1.5 Guardar el CSV actualizado (Lógica de MANTENIMIENTO y LIMPIEZA) ---
    
    # Conjunto de todos los tickers que pasaron la validación
    valid_tickers = sorted(set(valid_tickers))
    
    # Los tickers inválidos son los que estaban en el conjunto TOTAL, pero NO en el conjunto FINAL válido.
    invalid_tickers = sorted(set(all_tickers) - set(valid_tickers))

    try:
        # Se guarda la lista limpia y ampliada. (Mantiene tus 901 si son válidos).
        pd.DataFrame({'Ticker': valid_tickers}).to_csv('Tickers.csv', index=False)
        pd.DataFrame({'Ticker': invalid_tickers}).to_csv('Tickers_invalidos.csv', index=False)
    except Exception as e:
        st.warning(f"⚠️ No se pudieron guardar Tickers.csv/Tickers_invalidos.csv en el servidor. (Error: {e})")

    st.success("✅ Validación de preparación finalizada.")
    st.markdown(f"""
        <p style='font-style: italic;'>
        **Resumen final:** <br>
        ✔️ Tickers válidos guardados: **{len(valid_tickers)}** <br>
        ❌ Tickers inválidos eliminados: **{len(invalid_tickers)}**
        </p>
    """, unsafe_allow_html=True)
    
    return valid_tickers

# =========================================================================
# 2. FUNCIÓN PRINCIPAL DE LA PÁGINA (FF Scanner)
# =========================================================================

def ff_scanner_page():
    st.title("🛡️ FF Scanner (Preparación de Datos)")
    st.markdown("---")
    
    # Ejecutar la fase de preparación
    valid_tickers = perform_initial_preparation()
    
    # --- Estructura para la fase 2. ESCANER (Pendiente) ---
    st.divider()
    st.subheader("2. Escaneo de Cadenas de Opciones (Siguiente Fase)")
    if valid_tickers:
        st.info(f"El siguiente paso usará los **{len(valid_tickers)}** tickers validados. Aquí agregaremos la lógica de Schwab.")
    else:
        st.error("No hay tickers válidos para continuar el escaneo. Revisa 'Tickers.csv'.")

# =========================================================================
# 3. EJECUCIÓN DEL SCRIPT
# =========================================================================

ff_scanner_page()
