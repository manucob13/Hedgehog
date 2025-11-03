# pages/FF Scanner.py - VERSIÓN CON LÓGICA EXACTA DE JUPYTER
import streamlit as st
import pandas as pd
import requests
import yfinance as yf
from datetime import timedelta, datetime
from io import StringIO
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
import time
from math import sqrt
from tqdm import tqdm
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
    st.error(f"❌ Falta configurar los secrets de Schwab. Clave faltante: {e}")
    st.stop()

# Rutas de archivos
token_path = "schwab_token.json"
TICKERS_SOURCE_FILE = 'Tickers.csv'

# =========================================================================
# 1. PREPARACIÓN DE TICKERS - LÓGICA EXACTA DE JUPYTER
# =========================================================================

def is_valid_ticker(ticker):
    """
    LÓGICA EXACTA DE TU JUPYTER QUE FUNCIONABA.
    Verifica si un ticker es válido usando yfinance.
    """
    try:
        t = yf.Ticker(ticker)
        fi = getattr(t, "fast_info", None)
        # fast_info es más rápido y suficiente
        if fi and isinstance(fi, dict) and fi.get('last_price') is not None:
            return ticker
        info = t.info
        if isinstance(info, dict) and (info.get('regularMarketPrice') is not None or info.get('previousClose') is not None):
            return ticker
    except Exception:
        return None
    return None


def perform_initial_preparation():
    """
    Preparación y validación de tickers - LÓGICA EXACTA DE JUPYTER
    """
    st.subheader("1. Preparación y Validación de Tickers")
    status_text = st.empty()

    # ==========================================
    # 1.1 LEER TICKERS EXISTENTES
    # ==========================================
    existing_tickers = set()
    if os.path.exists(TICKERS_SOURCE_FILE):
        try:
            df_existing = pd.read_csv(TICKERS_SOURCE_FILE)
            existing_tickers = set(df_existing.iloc[:, 0].astype(str).str.upper().str.strip())
            existing_tickers = {t for t in existing_tickers if t and t != 'NAN' and len(t) > 0}
            st.info(f"✅ '{TICKERS_SOURCE_FILE}' encontrado con {len(existing_tickers)} tickers.")
        except Exception as e:
            st.error(f"❌ Error al leer {TICKERS_SOURCE_FILE}: {e}")
            existing_tickers = set()
    else:
        st.warning(f"⚠️ '{TICKERS_SOURCE_FILE}' no encontrado. Iniciando solo con S&P 500.")

    # ==========================================
    # 1.2 DESCARGAR S&P 500
    # ==========================================
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

    # ==========================================
    # 1.3 COMBINAR TODAS LAS FUENTES
    # ==========================================
    all_tickers = sp500_tickers.union(existing_tickers)
    sorted_tickers = sorted(all_tickers)
    
    st.info(f"📊 **{len(all_tickers)} tickers** para validar (S&P 500 + Tickers.csv)")
    st.warning("⚠️ **Validación robusta activada**: Prioridad en precisión, tomará varios minutos.")
    
    # ==========================================
    # 1.4 VALIDACIÓN - LÓGICA EXACTA DE JUPYTER
    # ==========================================
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    valid_tickers = []
    
    # Usar la misma configuración que en Jupyter
    start_time = time.time()
    
    # IMPORTANTE: Ejecutar sin chunks, como en Jupyter
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = []
        for ticker in sorted_tickers:
            futures.append(executor.submit(is_valid_ticker, ticker))
        
        completed = 0
        for future in futures:
            result = future.result()
            if result:
                valid_tickers.append(result)
            
            completed += 1
            progress = completed / len(sorted_tickers)
            progress_bar.progress(progress)
            
            elapsed = time.time() - start_time
            estimated_total = elapsed / progress if progress > 0 else 0
            remaining = estimated_total - elapsed
            
            progress_text.text(
                f"Procesados: {completed}/{len(sorted_tickers)} | "
                f"Válidos: {len(valid_tickers)} | "
                f"Tiempo restante: ~{remaining/60:.1f} min"
            )
    
    progress_bar.empty()
    progress_text.empty()
    
    # ==========================================
    # 1.5 GUARDAR RESULTADOS
    # ==========================================
    valid_tickers = sorted(set(valid_tickers))
    invalid_tickers = sorted(set(all_tickers) - set(valid_tickers))
    
    # Guardar cache de validación (NO sobrescribir Tickers.csv original)
    try:
        pd.DataFrame({'Ticker': valid_tickers}).to_csv('Tickers_VAL_CACHE.csv', index=False)
        if invalid_tickers:
            pd.DataFrame({'Ticker': invalid_tickers}).to_csv('Tickers_invalidos.csv', index=False)
    except Exception as e:
        st.warning(f"⚠️ No se pudieron guardar los archivos de caché: {e}")
    
    # ==========================================
    # 1.6 MOSTRAR RESULTADOS
    # ==========================================
    total_time = time.time() - start_time
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("✅ Válidos", len(valid_tickers))
    with col2:
        st.metric("❌ Inválidos", len(invalid_tickers))
    with col3:
        st.metric("📊 Total", len(all_tickers))
    with col4:
        st.metric("⏱️ Tiempo", f"{total_time/60:.1f} min")
    
    # Mostrar sample de inválidos
    if invalid_tickers:
        with st.expander(f"🔍 Ver {len(invalid_tickers)} tickers inválidos"):
            st.write(", ".join(invalid_tickers[:50]))
            if len(invalid_tickers) > 50:
                st.caption(f"... y {len(invalid_tickers) - 50} más")
    
    st.success(f"✅ Validación completada: **{len(valid_tickers)} tickers válidos** listos para escaneo")
    
    # Mostrar comparación con Jupyter
    expected_valid = 915  # Lo que obtuviste en Jupyter
    diff = len(valid_tickers) - expected_valid
    if abs(diff) > 50:
        st.warning(f"⚠️ Diferencia con Jupyter: {diff:+d} tickers (esperado: ~{expected_valid})")
    
    st.divider()
    
    return valid_tickers

# =========================================================================
# 2. CONEXIÓN CON BROKER SCHWAB
# =========================================================================

def connect_to_schwab():
    """Conecta usando el token existente."""
    st.subheader("2. Conexión con Broker Schwab")

    if not os.path.exists(token_path):
        st.error("❌ No se encontró 'schwab_token.json'. Genera el token desde tu notebook local.")
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
        return None

# =========================================================================
# 3. FECHAS DE ENTRADA Y DTE
# =========================================================================

def get_next_thursday(today=None):
    """Devuelve el jueves de esta semana o el próximo si ya pasó."""
    if today is None:
        today = datetime.now().date()
    thursday = today + timedelta((3 - today.weekday()) % 7)
    if thursday <= today:
        thursday += timedelta(days=7)
    return thursday

def fechas_section():
    st.subheader("3. Fechas de Entrada y DTE")
    
    with st.container():
        default_fecha = get_next_thursday()
        default_dte_front = 15
        default_dte_back = 22

        col1, col2, col3 = st.columns(3)
        
        with col1:
            fecha_entrada = st.date_input(
                "📅 Fecha de Entrada", 
                value=default_fecha,
                format="DD/MM/YYYY"
            )
        
        with col2:
            dte_front = st.number_input("⏱️ DTE Front", min_value=1, value=default_dte_front)
        
        with col3:
            dte_back = st.number_input("⏱️ DTE Back", min_value=1, value=default_dte_back)

        st.markdown("---")

        fecha_dte_front = fecha_entrada + timedelta(days=int(dte_front))
        fecha_dte_back = fecha_entrada + timedelta(days=int(dte_back))

        st.markdown("#### 📊 Resumen de Fechas Calculadas")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Fecha de Entrada", fecha_entrada.strftime("%d/%m/%Y"))
            st.caption(f"📆 {fecha_entrada.strftime('%A, %d de %B de %Y')}")
        
        with col2:
            dias_hasta_front = (fecha_dte_front - fecha_entrada).days
            st.metric("Expiración Front", fecha_dte_front.strftime("%d/%m/%Y"), delta=f"{dias_hasta_front} días")
            st.caption(f"📆 {fecha_dte_front.strftime('%A, %d de %B de %Y')}")
        
        with col3:
            dias_hasta_back = (fecha_dte_back - fecha_entrada).days
            st.metric("Expiración Back", fecha_dte_back.strftime("%d/%m/%Y"), delta=f"{dias_hasta_back} días")
            st.caption(f"📆 {fecha_dte_back.strftime('%A, %d de %B de %Y')}")

        st.markdown("---")
        st.markdown("#### 📋 Tabla Resumen")
        
        df_fechas = pd.DataFrame({
            "Concepto": ["Entrada", "Front", "Back"],
            "Fecha": [
                fecha_entrada.strftime("%d/%m/%Y"),
                fecha_dte_front.strftime("%d/%m/%Y"),
                fecha_dte_back.strftime("%d/%m/%Y")
            ],
            "Días desde Entrada": [0, dias_hasta_front, dias_hasta_back],
            "Día de la Semana": [
                fecha_entrada.strftime("%A"),
                fecha_dte_front.strftime("%A"),
                fecha_dte_back.strftime("%A")
            ]
        })
        
        st.dataframe(df_fechas, hide_index=True, use_container_width=True)

    return fecha_entrada, dte_front, dte_back, fecha_dte_front, fecha_dte_back

# =========================================================================
# 4. CÁLCULOS Y ESCANEO
# =========================================================================

def obtener_strike_valido(client, ticker, fecha_front, fecha_back):
    """Obtiene el strike ATM con IVs válidos en ambas fechas"""
    try:
        response = client.get_option_chain(ticker)
        if response.status_code != 200:
            return None, None, None, None
        
        opciones = response.json()
        precio_actual = opciones.get('underlyingPrice')
        call_map = opciones.get('callExpDateMap', {})
        
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
        
        strikes_comunes = set(strikes_front.keys()) & set(strikes_back.keys())
        
        if not strikes_comunes:
            return None, None, None, None
        
        strikes_ordenados = sorted(strikes_comunes, key=lambda x: abs(float(x) - precio_actual))
        
        for strike_str in strikes_ordenados:
            iv_front = strikes_front[strike_str][0].get('volatility')
            iv_back = strikes_back[strike_str][0].get('volatility')
            
            if (iv_front and iv_back and 
                iv_front > 0 and iv_back > 0 and 
                iv_front < 200 and iv_back < 200):
                
                return precio_actual, float(strike_str), iv_front, iv_back
        
        return None, None, None, None
    except Exception:
        return None, None, None, None

def procesar_ticker_ivs(args):
    """Obtiene IVs y Strike ATM."""
    client, ticker, fecha_front, fecha_back, dte_front_days, dte_back_days = args
    
    try:
        precio_atm, strike_atm, iv_front, iv_back = obtener_strike_valido(
            client, ticker, fecha_front, fecha_back
        )
        
        if precio_atm and strike_atm:
            return {
                'Ticker': ticker,
                'DTE_Pair': f"{dte_front_days}-{dte_back_days}",
                'DTE_Front': fecha_front,
                'DTE_Back': fecha_back,
                'Precio': f"{precio_atm:.2f}",
                'Strike': f"{strike_atm:.2f}",
                'IV_F (%)': f"{iv_front:.2f}",
                'IV_B (%)': f"{iv_back:.2f}",
            }
    except Exception:
        pass
    
    return None

def calculate_ff_metrics(row, dte_front_days, dte_back_days):
    """Calcula FF, Market, Banda y Operar"""
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

    ff_squared = (iv_back**2 * dte_b - iv_front**2 * dte_f) / diff_dte
    
    if ff_squared < 0:
        ff = 0.0
    else:
        ff = sqrt(ff_squared)

    market = 'CONTANGO' if iv_back > iv_front else 'BACKWARDATION'
            
    if ff < 0.25:
        banda = '<25%'
    elif ff <= 0.35:
        banda = '25-35%'
    else:
        banda = '>35%'
            
    operar = True if market == 'CONTANGO' and banda == '25-35%' else False
    
    return ff * 100, market, banda, operar

def obtener_mid_price(client, ticker, fecha, strike):
    """Obtiene mid price para un strike específico"""
    try:
        response = client.get_option_chain(ticker)
        if response.status_code != 200:
            return None
        
        opciones = response.json()
        call_map = opciones.get('callExpDateMap', {})
        
        fecha_str = fecha.strftime('%Y-%m-%d')
        
        for fecha_key, strikes in call_map.items():
            if fecha_str in fecha_key:
                strike_str = str(float(strike))
                if strike_str in strikes:
                    contrato = strikes[strike_str][0]
                    bid = contrato.get('bid', 0)
                    ask = contrato.get('ask', 0)
                    
                    if bid and ask and bid > 0 and ask > 0:
                        mid_price = (bid + ask) / 2
                        return mid_price
        
        return None
    except Exception:
        return None

def procesar_ticker_precios(args):
    """Obtiene Mid Price."""
    client, ticker, fecha_front, strike = args
    try:
        mid_price = obtener_mid_price(client, ticker, fecha_front, float(strike))
        return {'ticker': ticker, 'mid_price': mid_price if mid_price else None}
    except Exception:
        return {'ticker': ticker, 'mid_price': None}

def check_earnings(ticker_symbol, fecha_inicio, fecha_fin):
    """Verifica si hay earnings entre fecha_inicio y fecha_fin"""
    try:
        ticker = yf.Ticker(ticker_symbol)
        earnings_df = ticker.earnings_dates

        if earnings_df is not None and not earnings_df.empty:
            earnings_dates = pd.to_datetime(earnings_df.index).date
            has_earnings = any((d >= fecha_inicio) and (d <= fecha_fin) for d in earnings_dates)
            return has_earnings
        return False
    except Exception:
        return False

def procesar_ticker_earnings(args):
    """Verifica Earnings."""
    ticker_symbol, fecha_inicio, fecha_fin = args
    return check_earnings(ticker_symbol, fecha_inicio, fecha_fin)

def obtener_volumen_opciones_ultimo_dia(ticker_symbol):
    """Obtiene el volumen total de opciones del último día"""
    try:
        ticker = yf.Ticker(ticker_symbol)
        expiration_dates = ticker.options
        
        if not expiration_dates:
            return 0
        
        total_volume = 0
        
        for exp_date_str in expiration_dates:
            try:
                chain = ticker.option_chain(exp_date_str)
                vol_calls = chain.calls['volume'].fillna(0).sum()
                vol_puts = chain.puts['volume'].fillna(0).sum()
                total_volume += vol_calls + vol_puts
            except Exception:
                continue
        
        return int(total_volume)
        
    except Exception:
        return 0

def procesar_ticker_volumen(args):
    """Obtiene Volumen del Último Día."""
    ticker_symbol = args
    return obtener_volumen_opciones_ultimo_dia(ticker_symbol)

def ejecutar_escaneo(client, tickers, fecha_entrada, dte_front_days, dte_back_days, fecha_dte_front, fecha_dte_back):
    """Ejecuta el escaneo completo"""
    
    status_container = st.empty()
    progress_bar = st.progress(0)
    
    # PASO 1: IVs y strikes
    status_container.info("📊 Paso 1/5: Obteniendo IVs y strikes ATM...")
    
    args_list = [(client, ticker, fecha_dte_front, fecha_dte_back, dte_front_days, dte_back_days) 
                 for ticker in tickers]
    
    resultados = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = [executor.submit(procesar_ticker_ivs, args) for args in args_list]
        for i, future in enumerate(futures):
            try:
                result = future.result(timeout=15)
                if result:
                    resultados.append(result)
            except Exception:
                pass
            progress_bar.progress((i + 1) / len(futures))
    
    if not resultados:
        status_container.error("❌ No se encontraron tickers con datos válidos")
        progress_bar.empty()
        return None
    
    df = pd.DataFrame(resultados)
    status_container.success(f"✅ Paso 1: {len(df)} tickers con datos")
    
    # PASO 2: Calcular FF
    status_container.info("🧮 Paso 2/5: Calculando FF...")
    df[['FF_calc', 'Market_calc', 'Banda_FF_calc', 'Operar_calc']] = df.apply(
        lambda row: pd.Series(calculate_ff_metrics(row, dte_front_days, dte_back_days)),
        axis=1
    )
    
    df['FF (%)'] = df['FF_calc'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else '')
    df['Market'] = df['Market_calc']
    df['Banda_FF'] = df['Banda_FF_calc']
    df['Operar'] = df['Operar_calc']
    df = df.drop(columns=['FF_calc', 'Market_calc', 'Banda_FF_calc', 'Operar_calc'])
    
    df_operar = df[df['Operar'] == True].copy()
    status_container.success(f"✅ Paso 2: {len(df_operar)} cumplen condiciones")
    
    if df_operar.empty:
        status_container.warning("⚠️ No hay tickers que cumplan condiciones")
        progress_bar.empty()
        return None
    
    # PASO 3: Precios
    status_container.info("💰 Paso 3/5: Obteniendo precios...")
    
    args_list = [(client, row['Ticker'], fecha_dte_front, row['Strike']) 
                 for _, row in df_operar.iterrows()]
    
    precios_results = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = [executor.submit(procesar_ticker_precios, args) for args in args_list]
        for i, future in enumerate(futures):
            try:
                precios_results.append(future.result(timeout=10))
            except Exception:
                precios_results.append({'ticker': '', 'mid_price': None})
            progress_bar.progress((i + 1) / len(futures))
    
    df_operar.insert(5, 'MID_Price', [f"{r['mid_price']:.2f}" if r['mid_price'] else "N/A" for r in precios_results])
    status_container.success("✅ Paso 3: Precios obtenidos")
    
    # PASO 4: Earnings
    status_container.info("📅 Paso 4/5: Verificando earnings...")
    
    args_list = [(row['Ticker'], fecha_entrada, fecha_dte_back) for _, row in df_operar.iterrows()]
    
    earnings_flags = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = [executor.submit(procesar_ticker_earnings, args) for args in args_list]
        for i, future in enumerate(futures):
            try:
                earnings_flags.append(future.result(timeout=10))
            except Exception:
                earnings_flags.append(False)
            progress_bar.progress((i + 1) / len(futures))
    
    df_operar['Earnings_temp'] = earnings_flags
    df_operar = df_operar[df_operar['Earnings_temp'] == False].copy()
    df_operar = df_operar.drop(columns=['Earnings_temp'])
    
    status_container.success(f"✅ Paso 4: {len(df_operar)} sin earnings")
    
    if df_operar.empty:
        status_container.warning("⚠️ No hay tickers sin earnings")
        progress_bar.empty()
        return None
    
    # PASO 5: Volúmenes
    status_container.info("📊 Paso 5/5: Obteniendo volúmenes...")
    
    args_list = [row['Ticker'] for _, row in df_operar.iterrows()]
    
    volumenes = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = [executor.submit(procesar_ticker_volumen, ticker) for ticker in args_list]
        for i, future in enumerate(futures):
            try:
                volumenes.append(future.result(timeout=15))
            except Exception:
                volumenes.append(0)
            progress_bar.progress((i + 1) / len(futures))
    
    df_operar['Vol_Ult_Dia'] = volumenes
    status_container.success("✅ Paso 5: Volúmenes obtenidos")
    
    # Top 5 por volumen
    df_final = df_operar.sort_values('Vol_Ult_Dia', ascending=False).head(5)
    
    columnas_finales = ['Ticker', 'DTE_Pair', 'DTE_Front', 'DTE_Back', 'Precio', 'MID_Price', 
                         'Strike', 'IV_F (%)', 'IV_B (%)', 'FF (%)', 'Market', 'Banda_FF', 'Vol_Ult_Dia']
    df_final = df_final[columnas_finales].reset_index(drop=True)
    
    progress_bar.empty()
    status_container.success(f"🎉 Escaneo completado: Top {len(df_final)} por volumen")
    
    return df_final

# =========================================================================
# 5. PRESENTACIÓN DE RESULTADOS
# =========================================================================

def mostrar_resultados(df_resultados):
    """Muestra resultados"""
    st.subheader("5. Resultados del Escaneo")
    
    if df_resultados is None or df_resultados.empty:
        st.warning("⚠️ No hay resultados")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Operaciones", len(df_resultados))
    with col2:
        avg_ff = df_resultados['FF (%)'].apply(lambda x: float(x) if x else 0).mean()
        st.metric("📊 FF Promedio", f"{avg_ff:.2f}%")
    with col3:
        contango_count = (df_resultados['Market'] == 'CONTANGO').sum()
        st.metric("📈 Contango", contango_count)
    with col4:
        total_vol = df_resultados['Vol_Ult_Dia'].sum()
        st.metric("📊 Vol. Total", f"{total_vol:,}")
    
    st.markdown("---")
    
    df_display = df_resultados.copy()
    df_display['DTE_Front'] = pd.to_datetime(df_display['DTE_Front']).dt.strftime('%d/%m/%Y')
    df_display['DTE_Back'] = pd.to_datetime(df_display['DTE_Back']).dt.strftime('%d/%m/%Y')
    
    st.markdown("#### 📋 Top 5 por Volumen")
    st.dataframe(df_display, hide_index=True, use_container_width=True)
    
    csv = df_resultados.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar CSV",
        data=csv,
        file_name=f"ff_scanner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# =========================================================================
# 6. FUNCIÓN PRINCIPAL
# =========================================================================

def ff_scanner_page():
    st.title("🛡️ FF Scanner - Preparación y Conexión")
    st.markdown("---")

    # Punto 1: Validación
    col1, col2 = st.columns([1, 4])
    with col1:
        update_btn = st.button("🔄 Actualizar/Validar Tickers", type="primary", key="update_btn")
    with col2:
        st.markdown("_(Lógica exacta de Jupyter)_")

    st.divider()
    
    if update_btn or 'valid_tickers' not in st.session_state:
        valid_tickers = perform_initial_preparation()
        st.session_state.valid_tickers = valid_tickers
    else:
        valid_tickers = st.session_state.valid_tickers
        st.info(f"✅ Usando {len(valid_tickers)} tickers validados")

    # Punto 2: Schwab
    st.divider()
    
    if 'schwab_client' not in st.session_state:
        st.session_state.schwab_client = connect_to_schwab()
    else:
        st.subheader("2. Conexión con Broker Schwab")
        st.success("✅ Schwab conectado")
    
    schwab_client = st.session_state.schwab_client

    # Punto 3: Fechas
    st.divider()
    fecha_entrada, dte_front, dte_back, fecha_dte_front, fecha_dte_back = fechas_section()

    # Punto 4: Escaneo
    st.divider()
    st.subheader("4. Escaneo de Mercado")
    
    if schwab_client is None:
        st.error("❌ Conecta Schwab primero")
    else:
        st.info(f"📊 {len(valid_tickers)} tickers listos")
        st.warning("⚠️ Escaneo: 3-5 minutos")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            scan_btn = st.button("🚀 Ejecutar Escaneo", type="primary", key="scan_btn")
        with col2:
            if 'df_resultados' in st.session_state and st.session_state.df_resultados is not None:
                st.success(f"✅ {len(st.session_state.df_resultados)} resultados")
        with col3:
            if st.button("🗑️ Limpiar", key="clear_btn"):
                if 'df_resultados' in st.session_state:
                    del st.session_state.df_resultados
                st.rerun()
        
        if scan_btn:
            start = time.time()
            df_resultados = ejecutar_escaneo(
                schwab_client, valid_tickers, fecha_entrada,
                int(dte_front), int(dte_back), fecha_dte_front, fecha_dte_back
            )
            st.session_state.df_resultados = df_resultados
            elapsed = time.time() - start
            
            if df_resultados is not None:
                st.balloons()
                st.success(f"🎉 Completado en {elapsed:.1f}s")

    # Punto 5: Resultados
    st.divider()
    if 'df_resultados' in st.session_state:
        mostrar_resultados(st.session_state.df_resultados)
    else:
        st.subheader("5. Resultados")
        st.info("👆 Ejecuta escaneo primero")

    st.divider()
    if schwab_client:
        st.success(f"🎯 {len(valid_tickers)} tickers válidos")

# =========================================================================
# 7. PUNTO DE ENTRADA
# =========================================================================

if __name__ == "__main__":
    if check_password():
        ff_scanner_page()
    else:
        st.title("🔒 Acceso Restringido")
