# pages/Option Tracker.py - CON PERSISTENCIA EN GITHUB
import streamlit as st
import pandas as pd
from datetime import timedelta, datetime
import os
import schwab
from schwab.auth import easy_client
from schwab.client import Client
from utils import check_password
import base64
import requests
import json
from io import StringIO

# =========================================================================
# 0. CONFIGURACIÓN
# =========================================================================

st.set_page_config(page_title="Option Tracker", layout="wide", initial_sidebar_state="expanded")

# Cargar variables de Schwab desde secrets
try:
    api_key = st.secrets["schwab"]["api_key"]
    app_secret = st.secrets["schwab"]["app_secret"]
    redirect_uri = st.secrets["schwab"]["redirect_uri"]
except KeyError as e:
    st.error(f"❌ Falta configurar los secrets de Schwab. Clave faltante: {e}")
    api_key, app_secret, redirect_uri = None, None, None

# Cargar variables de GitHub desde secrets
try:
    GITHUB_TOKEN = st.secrets["github"]["token"]
    GITHUB_REPO_OWNER = st.secrets["github"]["repo_owner"]
    GITHUB_REPO_NAME = st.secrets["github"]["repo_name"]
    GITHUB_BRANCH = st.secrets["github"]["branch"]
    GITHUB_FILE_PATH = "data/option_tracker.csv"
except KeyError as e:
    st.error(f"❌ Falta configurar los secrets de GitHub. Clave faltante: {e}")
    GITHUB_TOKEN = None

token_path = "schwab_token.json"

# Estilos CSS personalizados para modo oscuro
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: bold;
    }
    .success-box {
        background-color: #1a4d2e;
        border: 2px solid #2d8659;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #a8dadc;
    }
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1.5rem;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================================
# 1. FUNCIONES DE GITHUB API
# =========================================================================

def get_github_file(file_path):
    """Descarga un archivo desde GitHub"""
    if not GITHUB_TOKEN:
        return None, None
    
    url = f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/contents/{file_path}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    try:
        response = requests.get(url, headers=headers, params={"ref": GITHUB_BRANCH})
        if response.status_code == 200:
            content = response.json()
            file_content = base64.b64decode(content['content']).decode('utf-8')
            return file_content, content['sha']
        elif response.status_code == 404:
            return None, None
        else:
            st.error(f"Error al descargar desde GitHub: {response.status_code}")
            return None, None
    except Exception as e:
        st.error(f"Error al conectar con GitHub: {e}")
        return None, None

def update_github_file(file_path, content, sha=None):
    """Actualiza o crea un archivo en GitHub"""
    if not GITHUB_TOKEN:
        st.error("❌ No hay token de GitHub configurado")
        return False
    
    url = f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/contents/{file_path}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Convertir contenido a base64
    content_bytes = content.encode('utf-8')
    content_base64 = base64.b64encode(content_bytes).decode('utf-8')
    
    data = {
        "message": f"Update {file_path} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "content": content_base64,
        "branch": GITHUB_BRANCH
    }
    
    if sha:
        data["sha"] = sha
    
    try:
        response = requests.put(url, headers=headers, json=data)
        if response.status_code in [200, 201]:
            return True
        else:
            st.error(f"Error al guardar en GitHub: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        st.error(f"Error al conectar con GitHub: {e}")
        return False

# =========================================================================
# 2. CONEXIÓN SCHWAB
# =========================================================================

def connect_to_schwab():
    """Usa el token existente si está disponible"""
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

        test_response = client.get_quote("AAPL")
        if hasattr(test_response, "status_code") and test_response.status_code != 200:
            raise Exception(f"Respuesta inesperada: {test_response.status_code}")

        return client

    except Exception as e:
        st.error(f"❌ Error al inicializar Schwab Client: {e}")
        st.warning("⚠️ Si el error persiste, elimina el archivo 'schwab_token.json' y vuelve a generarlo desde tu entorno local.")
        return None

# =========================================================================
# 3. FUNCIONES DE DATOS CON GITHUB
# =========================================================================

def cargar_operaciones():
    """Carga operaciones desde GitHub"""
    file_content, sha = get_github_file(GITHUB_FILE_PATH)
    
    if file_content:
        try:
            df = pd.read_csv(StringIO(file_content))
            
            # Convertir fechas
            if 'Fecha_Entrada' in df.columns:
                df['Fecha_Entrada'] = pd.to_datetime(df['Fecha_Entrada']).dt.date
            
            if 'Fecha_Salida' not in df.columns and 'DTE' in df.columns:
                fechas_salida = []
                for _, row in df.iterrows():
                    fecha_salida = row['Fecha_Entrada'] + timedelta(days=int(row['DTE']))
                    fechas_salida.append(fecha_salida)
                df['Fecha_Salida'] = fechas_salida
            elif 'Fecha_Salida' in df.columns:
                df['Fecha_Salida'] = pd.to_datetime(df['Fecha_Salida']).dt.date
            
            if 'Prima_Entrada' in df.columns:
                if 'Es_Credito' not in df.columns:
                    df['Es_Credito'] = df['Prima_Entrada'].apply(lambda x: x < 0 if pd.notna(x) else True)
                    df['Prima_Entrada'] = df['Prima_Entrada'].abs()
            else:
                df['Prima_Entrada'] = 0.0
                df['Es_Credito'] = True
            
            if 'Comision_Leg1' not in df.columns:
                df['Comision_Leg1'] = 0.65
            if 'Comision_Leg2' not in df.columns:
                df['Comision_Leg2'] = 0.65
            if 'Comision' not in df.columns:
                df['Comision'] = df['Comision_Leg1'].fillna(0) + df['Comision_Leg2'].fillna(0)
            
            # Guardar el SHA para futuras actualizaciones
            st.session_state['csv_sha'] = sha
            
            return df
        except Exception as e:
            st.error(f"Error al procesar CSV desde GitHub: {e}")
            return crear_dataframe_vacio()
    else:
        # Crear DataFrame vacío si no existe el archivo
        st.session_state['csv_sha'] = None
        return crear_dataframe_vacio()

def crear_dataframe_vacio():
    """Crea un DataFrame vacío con las columnas necesarias"""
    return pd.DataFrame(columns=[
        'ID', 'Ticker', 'Estrategia',
        'Strike_1', 'Tipo_1', 'Posicion_1',
        'Strike_2', 'Tipo_2', 'Posicion_2',
        'Fecha_Entrada', 'Fecha_Salida', 'DTE',
        'Prima_Entrada', 'Es_Credito', 'Comision_Leg1', 'Comision_Leg2', 'Comision',
        'Precio_Actual_1', 'Delta_1', 'Theta_1',
        'Precio_Actual_2', 'Delta_2', 'Theta_2',
        'PnL_Bruto', 'PnL_Neto', 'PnL_Porcentaje'
    ])

def guardar_operaciones(df):
    """Guarda operaciones en GitHub"""
    csv_content = df.to_csv(index=False)
    sha = st.session_state.get('csv_sha', None)
    
    success = update_github_file(GITHUB_FILE_PATH, csv_content, sha)
    
    if success:
        # Actualizar el SHA después de guardar
        file_content, new_sha = get_github_file(GITHUB_FILE_PATH)
        if new_sha:
            st.session_state['csv_sha'] = new_sha
    
    return success

def agregar_operacion(ticker, estrategia, strike_1, tipo_1, posicion_1,
                      strike_2, tipo_2, posicion_2,
                      fecha_entrada, fecha_salida, prima_entrada, es_credito, 
                      comision_leg1, comision_leg2):
    """Agrega una nueva operación"""
    df = cargar_operaciones()
    
    dte = (fecha_salida - fecha_entrada).days
    nuevo_id = df['ID'].max() + 1 if len(df) > 0 else 1
    
    if estrategia == "Single Leg":
        strike_2 = None
        tipo_2 = None
        posicion_2 = None
        comision_leg2 = 0
    
    comision_total = comision_leg1 + comision_leg2
    prima_normalizada = abs(prima_entrada)
    
    nueva_operacion = pd.DataFrame([{
        'ID': nuevo_id,
        'Ticker': ticker.upper(),
        'Estrategia': estrategia,
        'Strike_1': strike_1,
        'Tipo_1': tipo_1,
        'Posicion_1': posicion_1,
        'Strike_2': strike_2,
        'Tipo_2': tipo_2,
        'Posicion_2': posicion_2,
        'Fecha_Entrada': fecha_entrada,
        'Fecha_Salida': fecha_salida,
        'DTE': dte,
        'Prima_Entrada': prima_normalizada,
        'Es_Credito': es_credito,
        'Comision_Leg1': comision_leg1,
        'Comision_Leg2': comision_leg2,
        'Comision': comision_total,
        'Precio_Actual_1': None,
        'Delta_1': None,
        'Theta_1': None,
        'Precio_Actual_2': None,
        'Delta_2': None,
        'Theta_2': None,
        'PnL_Bruto': None,
        'PnL_Neto': None,
        'PnL_Porcentaje': None
    }])
    
    df = pd.concat([df, nueva_operacion], ignore_index=True)
    return guardar_operaciones(df)

def eliminar_operacion(id_operacion):
    """Elimina una operación por ID"""
    df = cargar_operaciones()
    df = df[df['ID'] != id_operacion]
    return guardar_operaciones(df)

# =========================================================================
# 4. OBTENER DATOS DE SCHWAB
# =========================================================================

def obtener_datos_opcion(client, ticker, strike, tipo, fecha_salida):
    """Obtiene precio, delta y theta desde Schwab"""
    try:
        if client is None:
            return None, None, None
        
        response = client.get_option_chain(ticker)
        if response.status_code != 200:
            return None, None, None
        
        opciones = response.json()
        
        if tipo == 'CALL':
            option_map = opciones.get('callExpDateMap', {})
        else:
            option_map = opciones.get('putExpDateMap', {})
        
        fecha_str = fecha_salida.strftime('%Y-%m-%d')
        
        fecha_key_match = None
        for key in option_map.keys():
            if key.startswith(fecha_str):
                fecha_key_match = key
                break
        
        if fecha_key_match:
            strikes = option_map[fecha_key_match]
            strike_str = str(float(strike))
            
            if strike_str in strikes:
                contrato = strikes[strike_str][0] 
                
                bid = contrato.get('bid', 0)
                ask = contrato.get('ask', 0)
                delta = contrato.get('delta', None)
                theta = contrato.get('theta', None)
                
                if bid > 0 and ask > 0:
                    mid_price = (bid + ask) / 2
                else:
                    mid_price = None 
                
                return mid_price, delta, theta
        
        return None, None, None
    except Exception:
        return None, None, None

# =========================================================================
# 5. CÁLCULO DE P&L CORREGIDO
# =========================================================================

def calcular_pnl_correcto(prima_entrada, precio_cierre_actual, es_credito, comision):
    """Calcula P&L correctamente según el tipo de operación"""
    if es_credito:
        pnl_bruto = (prima_entrada - precio_cierre_actual) * 100
    else:
        pnl_bruto = (precio_cierre_actual - prima_entrada) * 100
    
    pnl_neto = pnl_bruto - comision
    
    return pnl_bruto, pnl_neto

def refrescar_todas_operaciones(client):
    """Refresca datos de todas las operaciones desde Schwab y calcula P&L"""
    df = cargar_operaciones()
    
    if df.empty or client is None:
        return df
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, row in df.iterrows():
        status_text.text(f"🔄 Actualizando {row['Ticker']} ({idx+1}/{len(df)})...")
        
        # Obtener datos del Leg 1
        precio_1, delta_1, theta_1 = obtener_datos_opcion(
            client, row['Ticker'], row['Strike_1'], row['Tipo_1'], row['Fecha_Salida']
        )
        
        df.at[idx, 'Precio_Actual_1'] = precio_1
        df.at[idx, 'Delta_1'] = delta_1
        df.at[idx, 'Theta_1'] = theta_1
        
        # Obtener datos del Leg 2 si es Spread
        precio_2 = None
        if row['Estrategia'] == "Spread" and pd.notna(row['Strike_2']):
            precio_2, delta_2, theta_2 = obtener_datos_opcion(
                client, row['Ticker'], row['Strike_2'], row['Tipo_2'], row['Fecha_Salida']
            )
            df.at[idx, 'Precio_Actual_2'] = precio_2
            df.at[idx, 'Delta_2'] = delta_2
            df.at[idx, 'Theta_2'] = theta_2
        else:
            df.at[idx, 'Precio_Actual_2'] = None 
            df.at[idx, 'Delta_2'] = None
            df.at[idx, 'Theta_2'] = None
        
        # Cálculo de P&L
        pnl_bruto = None
        pnl_neto = None
        pnl_porcentaje = None
        
        if float(row['Prima_Entrada']) > 0:
            prima_entrada = float(row['Prima_Entrada'])
            comision = float(row['Comision'])
            es_credito = bool(row['Es_Credito'])
            
            if row['Estrategia'] == "Single Leg" and precio_1 is not None:
                pnl_bruto, pnl_neto = calcular_pnl_correcto(
                    prima_entrada, precio_1, es_credito, comision
                )
            
            elif row['Estrategia'] == "Spread" and precio_1 is not None and precio_2 is not None:
                valor_actual_spread = abs(precio_1 - precio_2)
                pnl_bruto, pnl_neto = calcular_pnl_correcto(
                    prima_entrada, valor_actual_spread, es_credito, comision
                )
            
            if pnl_neto is not None:
                prima_total_invertida = prima_entrada * 100
                if prima_total_invertida != 0:
                    pnl_porcentaje = (pnl_neto / prima_total_invertida) * 100
        
        df.at[idx, 'PnL_Bruto'] = pnl_bruto
        df.at[idx, 'PnL_Neto'] = pnl_neto
        df.at[idx, 'PnL_Porcentaje'] = pnl_porcentaje
        
        progress_bar.progress((idx + 1) / len(df))
    
    progress_bar.empty()
    status_text.empty()
    
    guardar_operaciones(df)
    return df

# =========================================================================
# 6. EXPANDIR SPREADS EN LEGS
# =========================================================================

def expandir_operaciones_en_legs(df):
    """Expande cada spread en múltiples filas (una por leg)"""
    if df.empty:
        return df
    
    filas_expandidas = []
    
    for idx, row in df.iterrows():
        if row['Estrategia'] == "Single Leg":
            row_copy = row.copy()
            row_copy['ID_Original'] = row['ID']
            row_copy['Leg_Num'] = 1
            
            strike = f"{row['Strike_1']:.2f}"
            tipo = row['Tipo_1'][0]
            posicion = row['Posicion_1']
            row_copy['Leg_Descripcion'] = f"{posicion} {strike}{tipo}"
            
            filas_expandidas.append(row_copy)
        
        else:  # Spread
            pnl_neto_total = float(row['PnL_Neto']) if pd.notna(row['PnL_Neto']) else 0
            prima_total = float(row['Prima_Entrada'])
            comision_total = float(row['Comision'])
            delta_total = float(row['Delta_1'] or 0) + float(row['Delta_2'] or 0)
            theta_total = float(row['Theta_1'] or 0) + float(row['Theta_2'] or 0)
            
            pnl_por_leg = pnl_neto_total / 2
            prima_por_leg = prima_total / 2
            comision_por_leg = comision_total / 2
            delta_por_leg = delta_total / 2
            theta_por_leg = theta_total / 2
            
            pnl_porcentaje_leg = None
            if prima_por_leg * 100 != 0:
                pnl_porcentaje_leg = (pnl_por_leg / (prima_por_leg * 100)) * 100
            
            # LEG 1
            row_leg1 = row.copy()
            row_leg1['ID_Original'] = row['ID']
            row_leg1['Leg_Num'] = 1
            strike_1 = f"{row['Strike_1']:.2f}"
            tipo_1 = row['Tipo_1'][0]
            posicion_1 = row['Posicion_1']
            row_leg1['Leg_Descripcion'] = f"{posicion_1} {strike_1}{tipo_1}"
            row_leg1['Prima_Entrada'] = prima_por_leg
            row_leg1['Comision'] = comision_por_leg
            row_leg1['PnL_Neto'] = pnl_por_leg
            row_leg1['PnL_Porcentaje'] = pnl_porcentaje_leg
            row_leg1['Delta_Total_Display'] = delta_por_leg
            row_leg1['Theta_Total_Display'] = theta_por_leg
            filas_expandidas.append(row_leg1)
            
            # LEG 2
            row_leg2 = row.copy()
            row_leg2['ID_Original'] = row['ID']
            row_leg2['Leg_Num'] = 2
            strike_2 = f"{row['Strike_2']:.2f}"
            tipo_2 = row['Tipo_2'][0]
            posicion_2 = row['Posicion_2']
            row_leg2['Leg_Descripcion'] = f"{posicion_2} {strike_2}{tipo_2}"
            row_leg2['Prima_Entrada'] = prima_por_leg
            row_leg2['Comision'] = comision_por_leg
            row_leg2['PnL_Neto'] = pnl_por_leg
            row_leg2['PnL_Porcentaje'] = pnl_porcentaje_leg
            row_leg2['Delta_Total_Display'] = delta_por_leg
            row_leg2['Theta_Total_Display'] = theta_por_leg
            filas_expandidas.append(row_leg2)
    
    df_expandido = pd.DataFrame(filas_expandidas)
    return df_expandido

# =========================================================================
# 7. INTERFAZ PRINCIPAL
# =========================================================================

def option_tracker_page():
    st.markdown('<div class="main-header">📊 Option Tracker Pro</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Monitor profesional de opciones y spreads</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Conexión Schwab
    if 'schwab_client_tracker' not in st.session_state:
        with st.spinner("🔌 Conectando con Schwab..."):
            st.session_state.schwab_client_tracker = connect_to_schwab()
    
    client = st.session_state.schwab_client_tracker
    
    if client:
        st.markdown('<div class="success-box">✅ <strong>Conexión activa</strong> con Schwab API</div>', unsafe_allow_html=True)
    else:
        st.error("❌ No hay conexión con Schwab. Verifica que 'schwab_token.json' existe y es válido.")
    
    # Verificar conexión GitHub
    if GITHUB_TOKEN:
        st.markdown('<div class="success-box">✅ <strong>Persistencia activada</strong> en GitHub</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ No hay conexión con GitHub. Los datos no se guardarán permanentemente.")
    
    st.markdown("---")
    
    # SECCIÓN 1: AGREGAR NUEVA OPERACIÓN
    st.markdown("### ➕ Nueva Operación")
    
    with st.expander("📝 Formulario de entrada", expanded=False):
        # Selectores fuera del formulario
        st.markdown("#### 📋 Información Básica")
        col1, col2 = st.columns(2)
        
        with col1:
            estrategia = st.selectbox(
                "📊 Estrategia", 
                ["Single Leg", "Spread"], 
                key="estrategia_selector"
            )
        
        with col2:
            tipo_comision = st.selectbox(
                "💳 Tipo Comisión", 
                ["Por Leg", "Total"],
                key="tipo_comision_selector"
            )
        
        st.markdown("---")
        
        # Formulario
        with st.form("form_nueva_operacion", clear_on_submit=True):
            col1, col2 = st.columns(2)
            
            with col1:
                ticker = st.text_input("🎯 Ticker", placeholder="AAPL", help="Símbolo del activo")
            
            with col2:
                es_credito = st.checkbox("💰 Es Crédito", value=True, help="Marca si recibiste crédito (vendiste). Desmarca si pagaste débito (compraste)")
            
            st.markdown("---")
            
            # Comisiones
            if tipo_comision == "Total":
                comision_total_input = st.number_input(
                    "💵 Comisión Total ($)", 
                    min_value=0.0, 
                    value=1.30 if estrategia == "Spread" else 0.65, 
                    step=0.01, 
                    format="%.2f",
                    key="comision_total_input",
                    help="Comisión total de toda la operación (apertura + cierre)"
                )
                if estrategia == "Spread":
                    comision_leg1 = comision_total_input / 2
                    comision_leg2 = comision_total_input / 2
                else:
                    comision_leg1 = comision_total_input
                    comision_leg2 = 0
            else:  # Por Leg
                if estrategia == "Spread":
                    col1, col2 = st.columns(2)
                    with col1:
                        comision_leg1 = st.number_input(
                            "💵 Comisión Leg 1 ($)", 
                            min_value=0.0, 
                            value=0.65, 
                            step=0.01, 
                            format="%.2f",
                            key="comision_leg1_input"
                        )
                    with col2:
                        comision_leg2 = st.number_input(
                            "💵 Comisión Leg 2 ($)", 
                            min_value=0.0, 
                            value=0.65, 
                            step=0.01, 
                            format="%.2f",
                            key="comision_leg2_input"
                        )
                else:  # Single Leg
                    comision_leg1 = st.number_input(
                        "💵 Comisión ($)", 
                        min_value=0.0, 
                        value=0.65, 
                        step=0.01, 
                        format="%.2f",
                        key="comision_single_input"
                    )
                    comision_leg2 = 0
            
            st.markdown("---")
            
            st.markdown("#### 🎯 Leg 1")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                strike_1 = st.number_input("Strike 1", min_value=0.0, step=0.5, format="%.2f")
            
            with col2:
                tipo_1 = st.selectbox("Tipo 1", ["CALL", "PUT"])
            
            with col3:
                posicion_1 = st.selectbox("Posición 1", ["LONG", "SHORT"])
            
            strike_2 = None
            tipo_2 = None
            posicion_2 = None
            
            if estrategia == "Spread":
                st.markdown("#### 🎯 Leg 2")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    strike_2 = st.number_input("Strike 2", min_value=0.0, step=0.5, format="%.2f")
                
                with col2:
                    tipo_2 = st.selectbox("Tipo 2", ["CALL", "PUT"], index=0)
                
                with col3:
                    posicion_2_default = "LONG" if posicion_1 == "SHORT" else "SHORT"
                    posicion_2 = st.selectbox("Posición 2", ["LONG", "SHORT"], 
                                              index=0 if posicion_2_default == "LONG" else 1,
                                              help="En un Spread, la Posición 2 debe ser opuesta a la Posición 1.")
            
            st.markdown("---")
            
            st.markdown("#### 📅 Fechas y Prima")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fecha_entrada = st.date_input(
                    "Fecha Entrada",
                    value=datetime.now().date(),
                    format="DD/MM/YYYY"
                )
            
            with col2:
                fecha_salida = st.date_input(
                    "Fecha Salida (Expiración)",
                    value=datetime.now().date() + timedelta(days=15),
                    format="DD/MM/YYYY"
                )
            
            with col3:
                prima_entrada = st.number_input(
                    "Prima ($)", 
                    min_value=0.0,
                    max_value=1000.0,
                    value=0.0,
                    step=0.01, 
                    format="%.2f",
                    help="Monto de la prima (siempre positivo). Usa el checkbox 'Es Crédito' para indicar si es crédito o débito"
                )
            
            # Info boxes
            if fecha_salida and fecha_entrada:
                dte_calculado = (fecha_salida - fecha_entrada).days
                
                if tipo_comision == "Total":
                    comision_total_display = comision_total_input
                else:
                    comision_total_display = comision_leg1 + comision_leg2
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"📊 DTE Calculado: **{dte_calculado} días**")
                with col2:
                    st.info(f"💵 Comisión Total: **${comision_total_display:.2f}**")
            
            st.markdown("---")
            
            submit_button = st.form_submit_button("✅ Agregar Operación", use_container_width=True, type="primary")
            
            if submit_button:
                if ticker and strike_1 > 0:
                    if estrategia == "Spread" and (not strike_2 or strike_2 <= 0):
                        st.error("❌ Para Spreads debes completar ambos legs")
                    else:
                        with st.spinner("💾 Guardando en GitHub..."):
                            if agregar_operacion(ticker, estrategia, strike_1, tipo_1, posicion_1,
                                                 strike_2, tipo_2, posicion_2,
                                                 fecha_entrada, fecha_salida, prima_entrada, es_credito,
                                                 comision_leg1, comision_leg2):
                                st.success(f"✅ Operación agregada y guardada: {ticker} {estrategia} ({'CRÉDITO' if es_credito else 'DÉBITO'})")
                                st.rerun()
                            else:
                                st.error("❌ Error al guardar en GitHub")
                else:
                    st.error("❌ Completa todos los campos obligatorios")
    
    st.markdown("---")
    
    # SECCIÓN 2: OPERACIONES ACTIVAS
    st.markdown("### 📊 Portfolio Activo")
    
    df = cargar_operaciones()
    
    if client and not df.empty and df['Precio_Actual_1'].isna().all():
        with st.spinner("Cargando datos iniciales desde Schwab..."):
            df = refrescar_todas_operaciones(client)
    
    if df.empty:
        st.markdown('<div class="info-box">📝 No hay operaciones activas. ¡Agrega tu primera operación arriba!</div>', unsafe_allow_html=True)
    else:
        col1, col2, col3 = st.columns([2, 2, 3])
        
        with col1:
            if st.button("🔄 Refrescar Datos", type="primary", disabled=(client is None), use_container_width=True):
                if client:
                    with st.spinner("Actualizando desde Schwab..."):
                        df = refrescar_todas_operaciones(client)
                    st.success("✅ Datos actualizados y guardados en GitHub")
                    st.rerun()
        
        with col2:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Descargar CSV (Backup)",
                data=csv,
                file_name=f"option_tracker_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            if not df.empty and df['PnL_Neto'].notna().any():
                st.info(f"📅 Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        
        st.markdown("---")
        
        # RESUMEN DEL PORTFOLIO
        st.markdown("#### 💼 Resumen del Portfolio")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        pnl_neto_total = df['PnL_Neto'].apply(lambda x: float(x) if pd.notna(x) else 0).sum()
        delta_color = "normal" if pnl_neto_total >= 0 else "inverse"
        
        with col1:
            st.metric("📊 Operaciones", len(df))
        
        with col2:
            st.metric("💰 P&L Neto Total", f"${pnl_neto_total:.2f}", 
                      delta=f"{pnl_neto_total:.2f}", delta_color=delta_color)
        
        with col3:
            delta_agregado = df['Delta_1'].fillna(0).sum() + df['Delta_2'].fillna(0).sum()
            st.metric("Δ Portfolio", f"{delta_agregado:.2f}")

        with col4:
            theta_agregado = df['Theta_1'].fillna(0).sum() + df['Theta_2'].fillna(0).sum()
            st.metric("Θ Portfolio", f"{theta_agregado:.2f}")

        with col5:
            prima_total_invertida = df['Prima_Entrada'].apply(lambda x: float(x) if pd.notna(x) else 0).sum() * 100
            if prima_total_invertida != 0:
                pnl_neto_portfolio = df['PnL_Neto'].apply(lambda x: float(x) if pd.notna(x) else 0).sum()
                pnl_porcentaje_portfolio = (pnl_neto_portfolio / prima_total_invertida) * 100
                st.metric("📈 P&L % Portfolio", f"{pnl_porcentaje_portfolio:.1f}%")
            else:
                st.metric("📈 P&L % Portfolio", "0.0%")
        
        st.markdown("---")
        
        # DETALLE DE OPERACIONES
        st.markdown("### 📋 Detalle de Operaciones (DataFrame)")
        
        df_expandido = expandir_operaciones_en_legs(df)
        df_display = df_expandido.copy()

        df_display['P&L Neto'] = df_display['PnL_Neto'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
        df_display['P&L %'] = df_display['PnL_Porcentaje'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
        
        def format_delta(row):
            if pd.notna(row.get('Delta_Total_Display')):
                return f"{row['Delta_Total_Display']:.2f}"
            elif pd.notna(row['Delta_1']) and pd.notna(row['Delta_2']):
                total = row['Delta_1'] + row['Delta_2']
                return f"{total:.2f}"
            elif pd.notna(row['Delta_1']):
                return f"{row['Delta_1']:.2f}"
            else:
                return "N/A"
        
        def format_theta(row):
            if pd.notna(row.get('Theta_Total_Display')):
                return f"{row['Theta_Total_Display']:.2f}"
            elif pd.notna(row['Theta_1']) and pd.notna(row['Theta_2']):
                total = row['Theta_1'] + row['Theta_2']
                return f"{total:.2f}"
            elif pd.notna(row['Theta_1']):
                return f"{row['Theta_1']:.2f}"
            else:
                return "N/A"
        
        df_display['Δ Total'] = df_display.apply(format_delta, axis=1)
        df_display['Θ Total'] = df_display.apply(format_theta, axis=1)
        
        df_display['Prima'] = df_display['Prima_Entrada'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "$0.00")
        df_display['Comis.'] = df_display['Comision'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "$0.00")
        
        df_display['Fch. Entrada'] = df_display['Fecha_Entrada'].apply(lambda x: x.strftime('%d/%m/%Y') if pd.notna(x) else '')
        df_display['Fch. Salida'] = df_display['Fecha_Salida'].apply(lambda x: x.strftime('%d/%m/%Y') if pd.notna(x) else '')

        columnas_finales = [
            'ID_Original',
            'Leg_Num',
            'Ticker',
            'Leg_Descripcion',
            'Fch. Entrada',
            'Fch. Salida',
            'P&L Neto',
            'P&L %',
            'Δ Total',
            'Θ Total',
            'Prima',
            'Comis.'
        ]
        
        df_final = df_display[columnas_finales].rename(columns={
            'ID_Original': 'ID',
            'Leg_Num': 'Leg',
            'Leg_Descripcion': 'Estrategia Detalle'
        })

        st.dataframe(
            df_final,
            hide_index=True,
            use_container_width=True,
            column_config={
                "ID": st.column_config.NumberColumn("ID", width="small"),
                "Leg": st.column_config.NumberColumn("Leg", width="small"),
                "Ticker": st.column_config.TextColumn("🎯 Ticker", width="small"),
                "Estrategia Detalle": st.column_config.TextColumn("📊 Estrategia", width="medium"),
                "Fch. Entrada": st.column_config.TextColumn("📅 Entrada", width="small"),
                "Fch. Salida": st.column_config.TextColumn("📅 Salida", width="small"),
                "P&L Neto": st.column_config.TextColumn("💵 P&L", width="small"),
                "P&L %": st.column_config.TextColumn("📈 P&L %", width="small"),
                "Δ Total": st.column_config.TextColumn("📊 Delta", width="small"),
                "Θ Total": st.column_config.TextColumn("⏱️ Theta", width="small"),
                "Prima": st.column_config.TextColumn("💰 Prima", width="small"),
                "Comis.": st.column_config.TextColumn("💳 Comis", width="small")
            }
        )
        
        st.markdown("---")
        
        # Información sobre cálculos
        with st.expander("ℹ️ Información sobre Cálculos de P&L"):
            st.markdown("""
            ### 📊 Cálculo de P&L Corregido
            
            **Para CRÉDITOS (recibiste prima al abrir):**
            - ✅ P&L = (Prima Recibida - Precio Cierre Actual) × 100 - Comisión
            - Si el precio de cierre es **menor** que la prima recibida → **Ganancia** 💰
            - Ejemplo: Prima $2.10, Precio Actual $0.06 → P&L = ($2.10 - $0.06) × 100 - $0.68 = **+$203.32**
            
            **Para DÉBITOS (pagaste prima al abrir):**
            - ✅ P&L = (Precio Cierre Actual - Prima Pagada) × 100 - Comisión
            - Si el precio de cierre es **mayor** que la prima pagada → **Ganancia** 💰
            - Ejemplo: Prima $1.50, Precio Actual $1.80 → P&L = ($1.80 - $1.50) × 100 - $0.68 = **+$29.32**
            
            **Para SPREADS:**
            - Se calcula el valor actual del spread como la diferencia entre ambos legs
            - Se aplica la misma lógica de crédito/débito sobre el valor del spread
            - En la tabla expandida, cada leg muestra el 50% del P&L total
            
            ### 📋 Visualización de Legs
            
            - **Single Leg**: Aparece en 1 fila con toda la información
            - **Spread**: Aparece en 2 filas, una por cada leg
              - Cada fila muestra strike, tipo (C/P) y posición (LONG/SHORT)
              - Las métricas (P&L, Delta, Theta, Prima, Comisión) se distribuyen 50/50
              - El resumen del portfolio agrupa correctamente por ID para no duplicar
            
            ### 💾 Persistencia de Datos
            
            - **Todos los datos se guardan automáticamente en GitHub**
            - Puedes cerrar y volver a abrir la app sin perder información
            - El archivo se actualiza en `data/option_tracker.csv` en tu repositorio
            - Cada cambio crea un commit automático con timestamp
            """)
        
        st.markdown("---")
        
        # Eliminar operación
        with st.expander("🗑️ Eliminar Operación"):
            col1, col2 = st.columns([1, 2])
            
            max_id = int(df['ID'].max()) if not df.empty else 1
            
            with col1:
                id_eliminar = st.number_input(
                    "ID a eliminar",
                    min_value=1,
                    max_value=max_id,
                    step=1
                )
            
            with col2:
                if st.button("🗑️ Confirmar Eliminación", type="secondary", use_container_width=True):
                    if id_eliminar in df['ID'].values:
                        operacion = df[df['ID'] == id_eliminar].iloc[0]
                        with st.spinner("🗑️ Eliminando de GitHub..."):
                            if eliminar_operacion(id_eliminar):
                                st.success(f"✅ Eliminada y guardado en GitHub: {operacion['Ticker']} - {operacion['Estrategia']}")
                                st.rerun()
                            else:
                                st.error("❌ Error al guardar cambios en GitHub")
                    else:
                        st.error(f"❌ No existe ID {id_eliminar}")

# =========================================================================
# 8. PUNTO DE ENTRADA
# =========================================================================

if __name__ == "__main__":
    if check_password():
        option_tracker_page()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Introduce tus credenciales en el menú lateral para acceder.")
