# pages/FF Scanner.py - VERSIÓN COMPLETA CON AUTENTICACIÓN Y ESCANEO

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
from math import sqrt

# === IMPORTAR SCHWAB ===
import schwab
from schwab.auth import easy_client
from schwab.client import Client

# === IMPORTAR LA FUNCIÓN DE AUTENTICACIÓN ===
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
            st.warning("⚠️ Archivo 'Tickers.csv' NO ENCONTRADO. Iniciando con 0 tickers existentes.")
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
# 2. FASE DE ESCANEO (Cálculo de fechas y búsqueda de opciones)
# =========================================================================

def calculate_ff_dates():
    """Calcula las fechas de vencimiento objetivo basadas en DTE."""
    
    # Parámetros
    DTE_FRONT_DAYS = 15
    DTE_BACK_DAYS = 22
    THURSDAY = 3  # Jueves (Lunes=0)
    
    # Fecha de hoy
    today = datetime.now().date()
    
    # Calcular el siguiente Jueves (DTE_ENTRY)
    current_day_of_week = today.weekday()
    days_until_next_thursday = THURSDAY - current_day_of_week
    if days_until_next_thursday < 0:
        days_until_next_thursday += 7
    
    dte_entry = today + timedelta(days=days_until_next_thursday)
    
    # Calcular DTE_FRONT y DTE_BACK
    dte_front = dte_entry + timedelta(days=DTE_FRONT_DAYS)
    dte_back = dte_entry + timedelta(days=DTE_BACK_DAYS)
    
    return today, dte_entry, dte_front, dte_back, DTE_FRONT_DAYS, DTE_BACK_DAYS

def obtener_strike_valido(client, ticker, fecha_front, fecha_back):
    """
    Función para obtener el strike ATM con IVs válidos en ambas fechas.
    Basada en el notebook original.
    """
    try:
        response = client.get_option_chain(ticker)
        if response.status_code != 200:
            return None, None, None, None
        
        opciones = response.json()
        precio_actual = opciones.get('underlyingPrice')
        call_map = opciones.get('callExpDateMap', {})
        
        # Buscar ambas fechas
        fecha_front_str = fecha_front.strftime('%Y-%m-%d')
        fecha_back_str = fecha_back.strftime('%Y-%m-%d')
        
        strikes_front = None
        strikes_back = None
        
        for fecha, strikes in call_map.items():
            if fecha_front_str in fecha:
                strikes_front = strikes
            if fecha_back_str in fecha:
                strikes_back = strikes
        
        if not strikes_front or not strikes_back:
            return None, None, None, None
        
        # Obtener strikes disponibles en ambas fechas
        strikes_comunes = set(strikes_front.keys()) & set(strikes_back.keys())
        
        if not strikes_comunes:
            return None, None, None, None
        
        # Ordenar por cercanía al precio actual
        strikes_ordenados = sorted(strikes_comunes, key=lambda x: abs(float(x) - precio_actual))
        
        # Buscar el primer strike con IVs válidos en ambas fechas
        for strike_str in strikes_ordenados:
            iv_front = strikes_front[strike_str][0].get('volatility')
            iv_back = strikes_back[strike_str][0].get('volatility')
            
            # Verificar que ambos IVs sean válidos (no None, no -999, positivos)
            if (iv_front and iv_back and 
                iv_front > 0 and iv_back > 0 and 
                iv_front < 200 and iv_back < 200):  # Filtrar valores absurdos
                
                return precio_actual, float(strike_str), iv_front, iv_back
        
        return None, None, None, None
    
    except Exception as e:
        return None, None, None, None

def scan_options_ff(valid_tickers, client):
    """
    Escanea opciones para todos los tickers válidos.
    Basado en la sección 2 del notebook.
    """
    
    st.subheader("2. Escaneo de Opciones (Factor Forward)")
    
    if not valid_tickers:
        st.error("No hay tickers válidos para iniciar el escaneo.")
        return None
    
    # Calcular fechas
    today, dte_entry, dte_front, dte_back, DTE_FRONT_DAYS, DTE_BACK_DAYS = calculate_ff_dates()
    
    # Mostrar fechas calculadas
    st.markdown(f"""
    **📅 Fechas Calculadas:**
    - **Hoy:** `{today.strftime('%Y-%m-%d')}`
    - **Jueves de Entrada (DTE_ENTRY):** `{dte_entry.strftime('%Y-%m-%d')}`
    - **DTE_FRONT ({DTE_FRONT_DAYS} días):** `{dte_front.strftime('%Y-%m-%d')}`
    - **DTE_BACK ({DTE_BACK_DAYS} días):** `{dte_back.strftime('%Y-%m-%d')}`
    """)
    
    st.divider()
    
    # Procesar tickers con barra de progreso
    st.info(f"🔍 Escaneando {len(valid_tickers)} tickers...")
    
    progress_bar = st.progress(0)
    progress_text = st.empty()
    resultados = []
    
    for idx, ticker in enumerate(valid_tickers):
        # Buscar el strike más cercano con IVs válidos en ambas fechas
        precio_atm, strike_atm, iv_front, iv_back = obtener_strike_valido(
            client, ticker, dte_front, dte_back
        )
        
        if precio_atm and strike_atm:
            resultados.append({
                'Ticker': ticker,
                'DTE_Pair': f"{DTE_FRONT_DAYS}-{DTE_BACK_DAYS}",
                'DTE_Front': dte_front,
                'DTE_Back': dte_back,
                'Precio': f"{precio_atm:.2f}",
                'Strike': f"{strike_atm:.2f}",
                'IV_F (%)': f"{iv_front:.2f}",
                'IV_B (%)': f"{iv_back:.2f}",
                'FF (%)': '',
                'Market': '',
                'Banda_FF': '',
                'Operar': ''
            })
        
        # Actualizar progreso
        progress_bar.progress((idx + 1) / len(valid_tickers))
        progress_text.text(f"Procesando: {idx + 1}/{len(valid_tickers)} - Encontrados: {len(resultados)}")
        
        # Pequeña pausa para evitar rate limiting (opcional)
        time.sleep(0.1)
    
    progress_bar.empty()
    progress_text.empty()
    
    # Crear DataFrame
    if resultados:
        df = pd.DataFrame(resultados)
        st.success(f"✅ Escaneo completado: {len(resultados)} tickers con datos válidos.")
        
        # Aplicar cálculos de FF, Market, Banda_FF y Operar
        df = calculate_ff_metrics(df, DTE_FRONT_DAYS, DTE_BACK_DAYS)
        
        st.dataframe(df, use_container_width=True)
        return df
    else:
        st.warning("⚠️ No se encontraron tickers con datos de opciones válidos.")
        return None

def calculate_ff_metrics(df, dte_front_days, dte_back_days):
    """
    Calcula FF, Market, Banda_FF y Operar para cada fila del DataFrame.
    Basado en el notebook original.
    """
    
    def calc_row(row):
        try:
            iv_front = float(row['IV_F (%)']) / 100
            iv_back = float(row['IV_B (%)']) / 100
        except (ValueError, TypeError):
            return np.nan, 'ERROR', 'ERROR', False
        
        dte_f = dte_front_days
        dte_b = dte_back_days
        diff_dte = dte_b - dte_f
        
        if diff_dte <= 0:
            return np.nan, 'ERROR', 'ERROR', False
        
        # Cálculo del Factor Forward (FF) al cuadrado
        ff_squared = (iv_back**2 * dte_b - iv_front**2 * dte_f) / diff_dte
        
        # Validar y calcular la raíz (FF)
        if ff_squared < 0:
            ff = 0.0
        else:
            ff = sqrt(ff_squared)
        
        # Asignar Market
        market = 'CONTANGO' if iv_back > iv_front else 'BACKWARDATION'
        
        # Asignar Banda (FF en decimal)
        if ff < 0.25:
            banda = '<25%'
        elif ff <= 0.35:
            banda = '25-35%'
        else:
            banda = '>35%'
        
        # Determinar si operar
        operar = True if market == 'CONTANGO' and banda == '25-35%' else False
        
        return ff * 100, market, banda, operar
    
    # Aplicar cálculo a cada fila
    df[['FF_calc', 'Market_calc', 'Banda_FF_calc', 'Operar_calc']] = df.apply(
        calc_row, axis=1, result_type='expand'
    )
    
    # Actualizar columnas originales
    df['FF (%)'] = df['FF_calc'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else '')
    df['Market'] = df['Market_calc']
    df['Banda_FF'] = df['Banda_FF_calc']
    df['Operar'] = df['Operar_calc']
    
    # Eliminar columnas temporales
    df = df.drop(columns=['FF_calc', 'Market_calc', 'Banda_FF_calc', 'Operar_calc'])
    
    return df

# =========================================================================
# 3. FUNCIÓN PRINCIPAL DE LA PÁGINA (FF Scanner)
# =========================================================================

def ff_scanner_page():
    st.title("🛡️ FF Scanner (Preparación y Escaneo)")
    st.markdown("---")
    
    # Contenedor para el botón
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
    
    # FASE 2: Escaneo (solo si hay tickers válidos)
    if valid_tickers:
        try:
            # Verificar si existe el token
            if not os.path.exists(token_path):
                st.error(f"❌ No se encontró el archivo de token: `{token_path}`")
                st.info("""
                **Para generar el token:**
                
                1. Ejecuta este código localmente (en tu computadora):
                ```python
                from schwab.auth import easy_client
                
                client = easy_client(
                    api_key="tu_api_key",
                    app_secret="tu_app_secret",
                    callback_url="https://127.0.0.1",
                    token_path="schwab_token.json"
                )
                ```
                2. Completa la autenticación en el navegador
                3. Sube el archivo `schwab_token.json` generado a tu repositorio
                """)
                st.stop()
            
            # Importar schwab
            from schwab.auth import client_from_token_file
            
            with st.spinner("🔐 Conectando con Schwab API usando token existente..."):
                # Usar el token existente en lugar de easy_client
                client = client_from_token_file(
                    token_path=token_path,
                    api_key=api_key,
                    app_secret=app_secret
                )
            
            st.success("✅ Conexión a Schwab API establecida.")
            
            # Ejecutar el escaneo
            df_resultados = scan_options_ff(valid_tickers, client)
            
        except ImportError:
            st.error("❌ La librería 'schwab-py' no está instalada. Instálala con: `pip install schwab-py`")
        except FileNotFoundError:
            st.error(f"❌ No se encontró el archivo de token: `{token_path}`")
            st.info("Asegúrate de haber generado el token y subirlo al repositorio.")
        except Exception as e:
            st.error(f"❌ Error al conectar con Schwab API: {e}")
            st.info("💡 Verifica que el token sea válido y no haya expirado.")
    else:
        st.error("❌ No hay tickers válidos para iniciar el escaneo.")

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
