# pages/FF Scanner.py - VERSIÓN COMPLETA CON ESCANEO PARALELO
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
# 1. PREPARACIÓN DE TICKERS (SIMPLIFICADA - SOLO LECTURA)
# =========================================================================

@st.cache_resource(ttl=timedelta(hours=24), show_spinner=False)
def perform_initial_preparation():
    st.subheader("1. Preparación de Tickers")

    status_text = st.empty()

    # Leer tickers del archivo CSV
    if os.path.exists('Tickers.csv'):
        try:
            df_tickers = pd.read_csv('Tickers.csv')
            tickers = df_tickers.iloc[:, 0].astype(str).str.upper().str.strip().tolist()
            tickers = sorted(set(tickers))  # Eliminar duplicados y ordenar
            
            st.success(f"✅ 'Tickers.csv' encontrado con {len(tickers)} tickers.")
            st.info("ℹ️ Los tickers se usan directamente sin validación adicional.")
            
            status_text.empty()
            st.divider()
            
            return tickers
            
        except Exception as e:
            st.error(f"❌ Error al leer 'Tickers.csv': {e}")
            st.stop()
    else:
        st.error("❌ 'Tickers.csv' no encontrado en el directorio raíz.")
        st.info("📝 Crea un archivo 'Tickers.csv' con una columna de tickers (uno por línea)")
        st.stop()

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
# 3. FECHAS DE ENTRADA Y DTE (después de conectar al broker)
# =========================================================================

def get_next_thursday(today=None):
    """Devuelve el jueves de esta semana o el próximo si ya pasó."""
    if today is None:
        today = datetime.now().date()
    thursday = today + timedelta((3 - today.weekday()) % 7)
    # Si ya pasó jueves, sumamos 7 días
    if thursday <= today:
        thursday += timedelta(days=7)
    return thursday

def fechas_section():
    st.subheader("3. Fechas de Entrada y DTE")
    
    # Crear un contenedor con estilo
    with st.container():
        # Valores por defecto
        default_fecha = get_next_thursday()
        default_dte_front = 15
        default_dte_back = 22

        # Layout en 3 columnas para los inputs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fecha_entrada = st.date_input(
                "📅 Fecha de Entrada", 
                value=default_fecha,
                help="Fecha en la que se realizará la entrada al trade",
                format="DD/MM/YYYY"
            )
        
        with col2:
            dte_front = st.number_input(
                "⏱️ DTE Front", 
                min_value=1, 
                value=default_dte_front,
                help="Días hasta expiración de la opción front"
            )
        
        with col3:
            dte_back = st.number_input(
                "⏱️ DTE Back", 
                min_value=1, 
                value=default_dte_back,
                help="Días hasta expiración de la opción back"
            )

        st.markdown("---")

        # Cálculo fechas adicionales
        fecha_dte_front = fecha_entrada + timedelta(days=int(dte_front))
        fecha_dte_back = fecha_entrada + timedelta(days=int(dte_back))

        # Mostrar resultados en cards usando métricas
        st.markdown("#### 📊 Resumen de Fechas Calculadas")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Fecha de Entrada",
                value=fecha_entrada.strftime("%d/%m/%Y"),
                help="Fecha seleccionada para realizar la entrada"
            )
            st.caption(f"📆 {fecha_entrada.strftime('%A, %d de %B de %Y')}")
        
        with col2:
            dias_hasta_front = (fecha_dte_front - fecha_entrada).days
            st.metric(
                label="Expiración Front",
                value=fecha_dte_front.strftime("%d/%m/%Y"),
                delta=f"{dias_hasta_front} días",
                help="Fecha de expiración de la opción front"
            )
            st.caption(f"📆 {fecha_dte_front.strftime('%A, %d de %B de %Y')}")
        
        with col3:
            dias_hasta_back = (fecha_dte_back - fecha_entrada).days
            st.metric(
                label="Expiración Back",
                value=fecha_dte_back.strftime("%d/%m/%Y"),
                delta=f"{dias_hasta_back} días",
                help="Fecha de expiración de la opción back"
            )
            st.caption(f"📆 {fecha_dte_back.strftime('%A, %d de %B de %Y')}")

        # Tabla adicional con información compacta
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
        
        # Mostrar tabla sin índice usando HTML personalizado
        st.dataframe(
            df_fechas,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Concepto": st.column_config.TextColumn("📌 Concepto", width="medium"),
                "Fecha": st.column_config.TextColumn("📅 Fecha", width="medium"),
                "Días desde Entrada": st.column_config.NumberColumn("⏰ Días", width="small"),
                "Día de la Semana": st.column_config.TextColumn("🗓️ Día", width="medium")
            }
        )

    return fecha_entrada, dte_front, dte_back, fecha_dte_front, fecha_dte_back

# =========================================================================
# 4. CÁLCULOS Y ESCANEO (PARALELO)
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
            
            # Verificar que ambos IVs sean válidos
            if (iv_front and iv_back and 
                iv_front > 0 and iv_back > 0 and 
                iv_front < 200 and iv_back < 200):
                
                return precio_actual, float(strike_str), iv_front, iv_back
        
        return None, None, None, None
    except Exception as e:
        return None, None, None, None

def procesar_ticker_ivs(args):
    """Función helper para paralelizar paso 1"""
    client, ticker, fecha_front, fecha_back, dte_front_days, dte_back_days = args
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

def obtener_mid_price(client, ticker, fecha, strike):
    """Obtiene bid, ask y calcula mid price para un strike específico"""
    try:
        response = client.get_option_chain(ticker)
        if response.status_code != 200:
            return None
        
        opciones = response.json()
        call_map = opciones.get('callExpDateMap', {})
        
        # Buscar la fecha
        fecha_str = fecha.strftime('%Y-%m-%d')
        
        for fecha_key, strikes in call_map.items():
            if fecha_str in fecha_key:
                strike_str = str(float(strike))
                if strike_str in strikes:
                    contrato = strikes[strike_str][0]
                    bid = contrato.get('bid', 0)
                    ask = contrato.get('ask', 0)
                    
                    # Calcular mid price
                    if bid and ask and bid > 0 and ask > 0:
                        mid_price = (bid + ask) / 2
                        return mid_price
        
        return None
    except Exception as e:
        return None

def procesar_ticker_precios(args):
    """Función helper para paralelizar paso 3"""
    client, ticker, fecha_front, strike = args
    mid_price = obtener_mid_price(client, ticker, fecha_front, float(strike))
    return {
        'ticker': ticker,
        'mid_price': mid_price if mid_price else None
    }

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
    """Función helper para paralelizar paso 4"""
    ticker_symbol, fecha_inicio, fecha_fin = args
    return check_earnings(ticker_symbol, fecha_inicio, fecha_fin)

def obtener_volumen_opciones_30dias(ticker_symbol):
    """
    Obtiene volumen promedio de opciones (calls + puts) de los últimos 30 días.
    Usa todas las fechas de expiración disponibles.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # Obtener todas las fechas de expiración disponibles
        expiration_dates = ticker.options
        
        if not expiration_dates:
            return 0
        
        total_volume = 0
        fecha_actual = datetime.now().date()
        
        # Iterar sobre todas las fechas de expiración
        for exp_date_str in expiration_dates:
            try:
                # Convertir fecha de expiración a date object
                exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d').date()
                
                # Solo considerar opciones que expiran en los próximos 30 días o menos
                dias_hasta_expiracion = (exp_date - fecha_actual).days
                
                if 0 <= dias_hasta_expiracion <= 30:
                    # Obtener chain de opciones para esta fecha
                    chain = ticker.option_chain(exp_date_str)
                    
                    # Sumar volumen de calls y puts
                    vol_calls = chain.calls['volume'].fillna(0).sum()
                    vol_puts = chain.puts['volume'].fillna(0).sum()
                    
                    total_volume += vol_calls + vol_puts
                    
            except Exception:
                # Si falla una fecha específica, continuar con la siguiente
                continue
        
        return int(total_volume)
        
    except Exception:
        return 0

def procesar_ticker_volumen(args):
    """Función helper para paralelizar paso 5"""
    ticker_symbol = args
    return obtener_volumen_opciones_30dias(ticker_symbol)

def ejecutar_escaneo(client, tickers, fecha_entrada, dte_front_days, dte_back_days, fecha_dte_front, fecha_dte_back):
    """Ejecuta el escaneo completo de todos los tickers EN PARALELO"""
    
    # Contenedores para mostrar progreso
    status_container = st.empty()
    progress_bar = st.progress(0)
    
    # PASO 1: Obtener IVs y strikes (PARALELO)
    status_container.info("📊 Paso 1/5: Obteniendo IVs y strikes ATM (paralelo)...")
    
    args_list = [(client, ticker, fecha_dte_front, fecha_dte_back, dte_front_days, dte_back_days) 
                 for ticker in tickers]
    
    resultados = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = [executor.submit(procesar_ticker_ivs, args) for args in args_list]
        for i, future in enumerate(futures):
            result = future.result()
            if result:
                resultados.append(result)
            progress_bar.progress((i + 1) / len(futures))
    
    if not resultados:
        status_container.error("❌ No se encontraron tickers con datos válidos")
        progress_bar.empty()
        return None
    
    df = pd.DataFrame(resultados)
    status_container.success(f"✅ Paso 1 completado: {len(df)} tickers con datos válidos")
    
    # PASO 2: Calcular FF y métricas (LOCAL, rápido)
    status_container.info("🧮 Paso 2/5: Calculando Factor Forward y métricas...")
    df[['FF_calc', 'Market_calc', 'Banda_FF_calc', 'Operar_calc']] = df.apply(
        lambda row: pd.Series(calculate_ff_metrics(row, dte_front_days, dte_back_days)),
        axis=1
    )
    
    df['FF (%)'] = df['FF_calc'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else '')
    df['Market'] = df['Market_calc']
    df['Banda_FF'] = df['Banda_FF_calc']
    df['Operar'] = df['Operar_calc']
    df = df.drop(columns=['FF_calc', 'Market_calc', 'Banda_FF_calc', 'Operar_calc'])
    
    # Filtrar solo los que cumplen condiciones
    df_operar = df[df['Operar'] == True].copy()
    status_container.success(f"✅ Paso 2 completado: {len(df_operar)} tickers cumplen condiciones")
    
    if df_operar.empty:
        status_container.warning("⚠️ No hay tickers que cumplan las condiciones de trading")
        progress_bar.empty()
        return None
    
    # PASO 3: Obtener precios (Mid Price solo) (PARALELO)
    status_container.info("💰 Paso 3/5: Obteniendo precios Mid (paralelo)...")
    
    args_list = [(client, row['Ticker'], fecha_dte_front, row['Strike']) 
                 for _, row in df_operar.iterrows()]
    
    precios_results = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = [executor.submit(procesar_ticker_precios, args) for args in args_list]
        for i, future in enumerate(futures):
            precios_results.append(future.result())
            progress_bar.progress((i + 1) / len(futures))
    
    df_operar.insert(5, 'MID_Price', [f"{r['mid_price']:.2f}" if r['mid_price'] else "N/A" for r in precios_results])
    status_container.success("✅ Paso 3 completado: Precios obtenidos")
    
    # PASO 4: Verificar earnings (PARALELO) - desde fecha_entrada hasta fecha_dte_back
    status_container.info("📅 Paso 4/5: Verificando earnings (paralelo)...")
    
    args_list = [(row['Ticker'], fecha_entrada, fecha_dte_back) for _, row in df_operar.iterrows()]
    
    earnings_flags = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = [executor.submit(procesar_ticker_earnings, args) for args in args_list]
        for i, future in enumerate(futures):
            has_earnings = future.result()
            earnings_flags.append(has_earnings)
            progress_bar.progress((i + 1) / len(futures))
    
    # Filtrar directamente los que tienen earnings (no los guardamos)
    df_operar['Earnings_temp'] = earnings_flags
    df_operar = df_operar[df_operar['Earnings_temp'] == False].copy()
    df_operar = df_operar.drop(columns=['Earnings_temp'])
    
    status_container.success(f"✅ Paso 4 completado: {len(df_operar)} tickers sin earnings")
    
    if df_operar.empty:
        status_container.warning("⚠️ No hay tickers sin earnings en el período")
        progress_bar.empty()
        return None
    
    # PASO 5: Obtener volúmenes de últimos 30 días (PARALELO)
    status_container.info("📊 Paso 5/5: Obteniendo volúmenes de opciones (últimos 30 días, paralelo)...")
    
    args_list = [row['Ticker'] for _, row in df_operar.iterrows()]
    
    volumenes = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = [executor.submit(procesar_ticker_volumen, ticker) for ticker in args_list]
        for i, future in enumerate(futures):
            volumenes.append(future.result())
            progress_bar.progress((i + 1) / len(futures))
    
    df_operar['Vol_30d'] = volumenes
    status_container.success("✅ Paso 5 completado: Volúmenes de 30 días obtenidos")
    
    # Filtrar por volumen >= 1000
    df_final = df_operar[df_operar['Vol_30d'] >= 1000].copy()
    
    if df_final.empty:
        status_container.warning("⚠️ No hay tickers con volumen >= 1000 en últimos 30 días")
        progress_bar.empty()
        return None
    
    # Ordenar por volumen descendente
    df_final = df_final.sort_values('Vol_30d', ascending=False)
    
    # Reorganizar columnas - quitar Operar
    columnas_finales = ['Ticker', 'DTE_Pair', 'DTE_Front', 'DTE_Back', 'Precio', 'MID_Price', 
                        'Strike', 'IV_F (%)', 'IV_B (%)', 'FF (%)', 'Market', 'Banda_FF', 'Vol_30d']
    df_final = df_final[columnas_finales]
    df_final = df_final.reset_index(drop=True)
    
    progress_bar.empty()
    status_container.success(f"🎉 Escaneo completado: {len(df_final)} operaciones válidas encontradas")
    
    return df_final

# =========================================================================
# 5. PRESENTACIÓN DE RESULTADOS
# =========================================================================

def mostrar_resultados(df_resultados):
    """Muestra los resultados del escaneo en formato tabla interactiva"""
    st.subheader("5. Resultados del Escaneo")
    
    if df_resultados is None or df_resultados.empty:
        st.warning("⚠️ No hay resultados para mostrar. Ejecuta el escaneo primero.")
        return
    
    # Compatibilidad con resultados antiguos: renombrar 'Vol' a 'Vol_30d' si existe
    if 'Vol' in df_resultados.columns and 'Vol_30d' not in df_resultados.columns:
        df_resultados = df_resultados.rename(columns={'Vol': 'Vol_30d'})
    
    # Métricas resumen
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Operaciones Válidas", len(df_resultados))
    with col2:
        avg_ff = df_resultados['FF (%)'].apply(lambda x: float(x) if x else 0).mean()
        st.metric("📊 FF Promedio", f"{avg_ff:.2f}%")
    with col3:
        contango_count = (df_resultados['Market'] == 'CONTANGO').sum()
        st.metric("📈 Contango", contango_count)
    with col4:
        total_vol = df_resultados['Vol_30d'].sum()
        st.metric("📊 Vol. Total (30d)", f"{total_vol:,}")
    
    st.markdown("---")
    
    # Convertir fechas a formato DD/MM/YYYY para display
    df_display = df_resultados.copy()
    df_display['DTE_Front'] = pd.to_datetime(df_display['DTE_Front']).dt.strftime('%d/%m/%Y')
    df_display['DTE_Back'] = pd.to_datetime(df_display['DTE_Back']).dt.strftime('%d/%m/%Y')
    
    # Tabla de resultados
    st.markdown("#### 📋 Tabla de Operaciones Recomendadas")
    st.dataframe(
        df_display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Ticker": st.column_config.TextColumn("🎯 Ticker", width="small"),
            "DTE_Pair": st.column_config.TextColumn("📅 DTE", width="small"),
            "DTE_Front": st.column_config.TextColumn("📅 Front", width="medium"),
            "DTE_Back": st.column_config.TextColumn("📅 Back", width="medium"),
            "Precio": st.column_config.TextColumn("💵 Precio", width="small"),
            "MID_Price": st.column_config.TextColumn("💰 Mid", width="small"),
            "Strike": st.column_config.TextColumn("🎯 Strike", width="small"),
            "IV_F (%)": st.column_config.TextColumn("📊 IV Front", width="small"),
            "IV_B (%)": st.column_config.TextColumn("📊 IV Back", width="small"),
            "FF (%)": st.column_config.TextColumn("🔥 FF", width="small"),
            "Market": st.column_config.TextColumn("📈 Market", width="medium"),
            "Banda_FF": st.column_config.TextColumn("🎯 Banda", width="small"),
            "Vol_30d": st.column_config.NumberColumn("📊 Vol 30d", width="small", format="%d")
        }
    )
    
    # Botón de descarga
    csv = df_resultados.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar Resultados (CSV)",
        data=csv,
        file_name=f"ff_scanner_resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# =========================================================================
# 6. FUNCIÓN PRINCIPAL
# =========================================================================

def ff_scanner_page():
    st.title("🛡️ FF Scanner - Preparación y Conexión")
    st.markdown("---")

    # --- Punto 1: Preparación de Tickers ---
    col1, col2 = st.columns([1, 4])
    with col1:
        st.button("🔄 Recargar Tickers", type="primary",
                  help="Borra la caché y recarga Tickers.csv",
                  on_click=perform_initial_preparation.clear)
    with col2:
        st.markdown("_(Los tickers se cargan desde Tickers.csv sin validación.)_")

    st.divider()
    valid_tickers = perform_initial_preparation()

    # --- Punto 2: Conexión Schwab ---
    st.divider()
    
    # Guardar cliente en session_state si no existe
    if 'schwab_client' not in st.session_state:
        st.session_state.schwab_client = connect_to_schwab()
    else:
        st.subheader("2. Conexión con Broker Schwab")
        st.success("✅ Conexión con Schwab verificada (ya conectado en esta sesión).")
    
    schwab_client = st.session_state.schwab_client

    # --- Punto 3: Fechas (después de conectar al broker) ---
    st.divider()
    fecha_entrada, dte_front, dte_back, fecha_dte_front, fecha_dte_back = fechas_section()

    # Guardar fechas en session_state para uso posterior
    st.session_state.fecha_entrada = fecha_entrada
    st.session_state.dte_front = dte_front
    st.session_state.dte_back = dte_back
    st.session_state.fecha_dte_front = fecha_dte_front
    st.session_state.fecha_dte_back = fecha_dte_back

    # --- Punto 4: Cálculos y Escaneo ---
    st.divider()
    st.subheader("4. Cálculos y Escaneo de Mercado")
    
    if schwab_client is None:
        st.error("❌ Necesitas conectar con Schwab antes de ejecutar el escaneo")
    else:
        st.info(f"📊 Tickers listos para escanear: **{len(valid_tickers)}** | 🚀 Modo: **Paralelo (15 hilos)**")
        st.warning("⚠️ El escaneo tardará 2-4 minutos. **No cambies de página durante el proceso.**")
        st.info("📊 **Nuevo**: El volumen se calcula sobre las opciones de los últimos 30 días")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            ejecutar_btn = st.button("🚀 Ejecutar Escaneo Completo", type="primary", use_container_width=True)
        with col2:
            if 'df_resultados' in st.session_state and st.session_state.df_resultados is not None:
                st.success(f"✅ Último escaneo: {len(st.session_state.df_resultados)} resultados")
        with col3:
            if st.button("🗑️ Limpiar Resultados", use_container_width=True):
                if 'df_resultados' in st.session_state:
                    del st.session_state.df_resultados
                st.rerun()
        
        if ejecutar_btn:
            start_time = time.time()
            with st.spinner("Ejecutando escaneo paralelo..."):
                df_resultados = ejecutar_escaneo(
                    schwab_client,
                    valid_tickers,
                    fecha_entrada,
                    int(dte_front),
                    int(dte_back),
                    fecha_dte_front,
                    fecha_dte_back
                )
                st.session_state.df_resultados = df_resultados
                
            elapsed_time = time.time() - start_time
            
            if df_resultados is not None and not df_resultados.empty:
                st.balloons()
                st.success(f"🎉 Escaneo completado en {elapsed_time:.1f} segundos con {len(df_resultados)} operaciones válidas")
            else:
                st.warning("⚠️ No se encontraron operaciones que cumplan los criterios")

    # --- Punto 5: Resultados ---
    st.divider()
    if 'df_resultados' in st.session_state:
        mostrar_resultados(st.session_state.df_resultados)
    else:
        st.subheader("5. Resultados del Escaneo")
        st.info("👆 Ejecuta el escaneo primero para ver los resultados aquí")

    # --- Estado final ---
    st.divider()
    if schwab_client:
        st.success(f"🎯 Sistema listo con {len(valid_tickers)} tickers válidos y conexión Schwab activa.")
    else:
        st.info("⏳ Conecta tu token Schwab para activar funciones de trading.")

# =========================================================================
# 7. PUNTO DE ENTRADA PROTEGIDO
# =========================================================================

if __name__ == "__main__":
    if check_password():
        ff_scanner_page()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Introduce tus credenciales en el menú lateral para acceder.")
