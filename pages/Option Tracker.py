# pages/Option Tracker.py - MONITOREO DE OPCIONES CON TARJETAS SIMPLIFICADAS (CORRECCIÓN FINAL BORDER-LEFT)
import streamlit as st
import pandas as pd
from datetime import timedelta, datetime, date
import os

# Dependencias de Schwab (puede que tu entorno use otro nombre/cliente)
try:
    import schwab
    from schwab.auth import easy_client
    from schwab.client import Client
except Exception:
    # Si no existe la librería en entorno de desarrollo, no romperá al editar/ejecutar offline.
    easy_client = None

# Asume que check_password se encuentra en un archivo utils.py
try:
    from utils import check_password 
except ImportError:
    # Definición dummy si no existe utils.py (solo para que el código corra)
    def check_password():
        return True 

# =========================================================================
# 0. CONFIGURACIÓN
# =========================================================================

st.set_page_config(page_title="Option Tracker", layout="wide", initial_sidebar_state="expanded")

# Cargar variables de Schwab desde secrets
try:
    api_key = st.secrets["schwab"]["api_key"]
    app_secret = st.secrets["schwab"]["app_secret"]
    redirect_uri = st.secrets["schwab"]["redirect_uri"]
except Exception as e:
    api_key = app_secret = redirect_uri = None

token_path = "schwab_token.json"
TRACKER_CSV = "option_tracker.csv"

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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
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
    .op-card {
        background-color: #2e303c;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .op-header {
        display: flex; 
        justify-content: space-between; 
        align-items: center; 
        margin-bottom: 10px;
        border-bottom: 1px dashed #444;
        padding-bottom: 5px;
    }
    .op-data {
        display: flex; 
        justify-content: space-between; 
        align-items: top; 
        padding-top: 10px;
    }
    .op-item {
        flex: 1;
        text-align: center;
        min-width: 80px;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================================
# 1. CONEXIÓN SCHWAB
# =========================================================================

def connect_to_schwab():
    """Conecta con Schwab usando token existente (si la librería está disponible)."""
    if easy_client is None:
        return None

    if not os.path.exists(token_path):
        st.error("❌ No se encontró 'schwab_token.json'. Genera el token primero.")
        return None

    try:
        client = easy_client(
            api_key=api_key,
            app_secret=app_secret,
            callback_url=redirect_uri,
            token_path=token_path
        )
        
        # Prueba de conexión (si la API disponible)
        try:
            test_response = client.get_quote("AAPL")
            if hasattr(test_response, "status_code") and test_response.status_code != 200:
                raise Exception(f"Respuesta inesperada: {test_response.status_code}")
        except Exception:
            # No fallar aquí si la prueba no es concluyente; devolvemos client y permitimos que funciones manejen errores.
            pass
        
        return client
    except Exception as e:
        st.error(f"❌ Error al conectar con Schwab: {e}")
        return None

# =========================================================================
# 2. FUNCIONES DE DATOS
# =========================================================================

def cargar_operaciones():
    """Carga operaciones desde CSV con manejo robusto de fechas y columnas faltantes."""
    if os.path.exists(TRACKER_CSV):
        df = pd.read_csv(TRACKER_CSV)

        # Asegurar columnas mínimas
        expected_cols = [
            'ID', 'Ticker', 'Estrategia',
            'Strike_1', 'Tipo_1', 'Posicion_1',
            'Strike_2', 'Tipo_2', 'Posicion_2',
            'Fecha_Entrada', 'Fecha_Salida', 'DTE',
            'Prima_Entrada', 'Es_Credito', 'Comision_Leg1', 'Comision_Leg2', 'Comision',
            'Precio_Actual_1', 'Delta_1', 'Theta_1',
            'Precio_Actual_2', 'Delta_2', 'Theta_2',
            'PnL_Bruto', 'PnL_Neto', 'PnL_Porcentaje'
        ]
        for c in expected_cols:
            if c not in df.columns:
                df[c] = pd.NA

        # Convertir fechas de forma segura (pueden venir vacías)
        df['Fecha_Entrada'] = pd.to_datetime(df['Fecha_Entrada'], errors='coerce')
        df['Fecha_Salida'] = pd.to_datetime(df['Fecha_Salida'], errors='coerce')

        # Convertir a objetos date o None
        df['Fecha_Entrada'] = df['Fecha_Entrada'].apply(lambda x: x.date() if pd.notna(x) else None)
        df['Fecha_Salida'] = df['Fecha_Salida'].apply(lambda x: x.date() if pd.notna(x) else None)

        # Compatibilidad con CSV antiguo que no tiene Fecha_Salida pero tiene DTE
        if df['Fecha_Salida'].isna().all() and 'DTE' in df.columns:
            fechas_salida = []
            for _, row in df.iterrows():
                fecha_ent = row.get('Fecha_Entrada', None)
                try:
                    dte_val = int(row['DTE']) if pd.notna(row['DTE']) else None
                except Exception:
                    dte_val = None

                if isinstance(fecha_ent, date) and dte_val is not None:
                    fecha_salida = fecha_ent + timedelta(days=dte_val)
                else:
                    fecha_salida = None
                fechas_salida.append(fecha_salida)
            df['Fecha_Salida'] = fechas_salida

        # Migrar Prima_Entrada si existe con valores negativos (para compatibilidad)
        if 'Prima_Entrada' in df.columns:
            # Convertir a float seguro
            df['Prima_Entrada'] = pd.to_numeric(df['Prima_Entrada'], errors='coerce').fillna(0.0)
            if 'Es_Credito' not in df.columns:
                # Si Prima_Entrada es negativa originalmente (ya normalizada arriba), asumimos crédito.
                # Nota: si el CSV ya convirtió a abs, no hay forma de recuperar signo; esto es heurístico.
                df['Es_Credito'] = df['Prima_Entrada'].apply(lambda x: True if x >= 0 else True)
        else:
            df['Prima_Entrada'] = 0.0
            df['Es_Credito'] = True

        # Añadir columnas de comisión si no existen
        if 'Comision_Leg1' not in df.columns:
            df['Comision_Leg1'] = 0.65
        if 'Comision_Leg2' not in df.columns:
            df['Comision_Leg2'] = 0.65
        if 'Comision' not in df.columns:
            # Recalcular Comisión total para evitar errores de NaN
            df['Comision'] = df['Comision_Leg1'].fillna(0).astype(float) + df['Comision_Leg2'].fillna(0).astype(float)

        # Asegurar tipos numéricos para griegas / precios
        for col in ['Precio_Actual_1','Delta_1','Theta_1','Precio_Actual_2','Delta_2','Theta_2',
                    'PnL_Bruto','PnL_Neto','PnL_Porcentaje','DTE','Strike_1','Strike_2']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Asegurar ID como entero cuando sea posible
        if 'ID' in df.columns:
            try:
                df['ID'] = pd.to_numeric(df['ID'], errors='coerce')
            except Exception:
                pass

        return df
    else:
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
    """Guarda operaciones en CSV; convierte fechas a ISO para evitar problemas."""
    df_to_save = df.copy()

    # Convertir fechas a ISO strings para persistencia
    for col in ['Fecha_Entrada', 'Fecha_Salida']:
        if col in df_to_save.columns:
            df_to_save[col] = df_to_save[col].apply(lambda x: x.isoformat() if isinstance(x, date) else (x if pd.isna(x) else str(x)))

    df_to_save.to_csv(TRACKER_CSV, index=False)

def agregar_operacion(ticker, estrategia, strike_1, tipo_1, posicion_1,
                      strike_2, tipo_2, posicion_2,
                      fecha_entrada, fecha_salida, prima_entrada, es_credito, 
                      comision_leg1, comision_leg2):
    """Agrega una nueva operación"""
    df = cargar_operaciones()
    
    # Calcular DTE
    dte = None
    try:
        if isinstance(fecha_salida, date) and isinstance(fecha_entrada, date):
            dte = (fecha_salida - fecha_entrada).days
    except Exception:
        dte = None
    
    # Generar ID único robusto
    if 'ID' in df.columns and df['ID'].notna().any():
        try:
            nuevo_id = int(df['ID'].max()) + 1
        except Exception:
            nuevo_id = 1
    else:
        nuevo_id = 1

    # Para single leg, los campos del leg 2 son None y la comisión es solo la del leg 1
    if estrategia == "Single Leg":
        strike_2 = None
        tipo_2 = None
        posicion_2 = None
        comision_leg2 = 0.0
    
    # Calcular comisión total
    try:
        comision_total = float(comision_leg1 or 0.0) + float(comision_leg2 or 0.0)
    except Exception:
        comision_total = 0.0
    
    # Normalizar prima: guardarla como positiva, el signo se maneja con 'Es_Credito'
    prima_normalizada = abs(float(prima_entrada or 0.0))
    
    nueva_operacion = pd.DataFrame([{
        'ID': nuevo_id,
        'Ticker': (ticker or "").upper(),
        'Estrategia': estrategia,
        'Strike_1': float(strike_1) if strike_1 is not None else pd.NA,
        'Tipo_1': tipo_1,
        'Posicion_1': posicion_1,
        'Strike_2': float(strike_2) if strike_2 is not None else pd.NA,
        'Tipo_2': tipo_2,
        'Posicion_2': posicion_2,
        'Fecha_Entrada': fecha_entrada,
        'Fecha_Salida': fecha_salida,
        'DTE': dte,
        'Prima_Entrada': prima_normalizada,
        'Es_Credito': bool(es_credito),
        'Comision_Leg1': float(comision_leg1 or 0.0),
        'Comision_Leg2': float(comision_leg2 or 0.0),
        'Comision': comision_total,
        'Precio_Actual_1': pd.NA,
        'Delta_1': pd.NA,
        'Theta_1': pd.NA,
        'Precio_Actual_2': pd.NA,
        'Delta_2': pd.NA,
        'Theta_2': pd.NA,
        'PnL_Bruto': pd.NA,
        'PnL_Neto': pd.NA,
        'PnL_Porcentaje': pd.NA
    }])
    
    df = pd.concat([df, nueva_operacion], ignore_index=True, sort=False)
    guardar_operaciones(df)
    return True

def eliminar_operacion(id_operacion):
    """Elimina una operación por ID"""
    df = cargar_operaciones()
    df = df[df['ID'] != id_operacion]
    guardar_operaciones(df)
    return True

# =========================================================================
# 3. OBTENER DATOS DE SCHWAB
# =========================================================================

def obtener_datos_opcion(client, ticker, strike, tipo, fecha_salida):
    """Obtiene precio, delta y theta desde Schwab. Devuelve (mid_price, delta, theta) o (None, None, None)."""
    try:
        if client is None:
            return None, None, None

        response = client.get_option_chain(ticker)
        # Si la librería devuelve algo distinto, validar
        if hasattr(response, "status_code") and response.status_code != 200:
            return None, None, None

        opciones = response.json() if hasattr(response, "json") else response

        if tipo == 'CALL':
            option_map = opciones.get('callExpDateMap', {}) if isinstance(opciones, dict) else {}
        else:
            option_map = opciones.get('putExpDateMap', {}) if isinstance(opciones, dict) else {}

        if not option_map:
            return None, None, None

        fecha_str = fecha_salida.strftime('%Y-%m-%d') if isinstance(fecha_salida, date) else None

        # Las claves tienen formato 'YYYY-MM-DD:DTE'
        fecha_key_match = None
        if fecha_str:
            for key in option_map.keys():
                if key.startswith(fecha_str):
                    fecha_key_match = key
                    break

        if fecha_key_match:
            strikes = option_map.get(fecha_key_match, {})
            # En algunos formatos el strike viene como string convertible a float
            strike_str = str(float(strike)) if strike is not None else None

            if strike_str in strikes:
                contrato = strikes[strike_str][0] 
                bid = contrato.get('bid', 0) or 0
                ask = contrato.get('ask', 0) or 0
                delta = contrato.get('delta', None)
                theta = contrato.get('theta', None)

                if bid > 0 and ask > 0:
                    mid_price = (bid + ask) / 2
                else:
                    # Si no hay bid/ask válidos, intentar con lastPrice o mark
                    mid_price = contrato.get('mark') or contrato.get('last') or None

                return mid_price, delta, theta

        return None, None, None
    except Exception:
        return None, None, None

def refrescar_todas_operaciones(client):
    """Refresca datos de todas las operaciones desde Schwab y calcula P&L."""
    df = cargar_operaciones()
    
    if df.empty:
        return df
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(df)
    for idx, row in df.iterrows():
        status_text.text(f"🔄 Actualizando {row.get('Ticker','')} ({idx+1}/{total})...")
        
        # 1. Obtener datos Leg 1
        precio_1, delta_1, theta_1 = obtener_datos_opcion(
            client, row.get('Ticker'), row.get('Strike_1'), row.get('Tipo_1'), row.get('Fecha_Salida')
        )
        
        df.at[idx, 'Precio_Actual_1'] = precio_1
        df.at[idx, 'Delta_1'] = delta_1
        df.at[idx, 'Theta_1'] = theta_1
        
        # 2. Obtener datos Leg 2 (si existe)
        precio_2 = None
        delta_2 = None
        theta_2 = None
        if str(row.get('Estrategia')).lower() == "spread" and pd.notna(row.get('Strike_2')):
            precio_2, delta_2, theta_2 = obtener_datos_opcion(
                client, row.get('Ticker'), row.get('Strike_2'), row.get('Tipo_2'), row.get('Fecha_Salida')
            )
            df.at[idx, 'Precio_Actual_2'] = precio_2
            df.at[idx, 'Delta_2'] = delta_2
            df.at[idx, 'Theta_2'] = theta_2
        else:
            # Aseguramos None para Single Leg
            df.at[idx, 'Precio_Actual_2'] = pd.NA 
            df.at[idx, 'Delta_2'] = pd.NA
            df.at[idx, 'Theta_2'] = pd.NA
        
        # 3. Inicializar P&L
        pnl_bruto = None
        pnl_neto = None
        pnl_porcentaje = None
        
        # Solo calcular si tenemos la Prima de Entrada y algún precio de mercado (o si spread tiene ambos legs)
        try:
            prima_entrada = float(row.get('Prima_Entrada') or 0.0)
        except Exception:
            prima_entrada = 0.0
        try:
            comision = float(row.get('Comision') or 0.0)
        except Exception:
            comision = 0.0
        es_credito = bool(row.get('Es_Credito')) if row.get('Es_Credito') is not None else True
        
        # --- LÓGICA DE CÁLCULO CORREGIDA PARA CRÉDITO Y DÉBITO ---
        if prima_entrada > 0:
            # Cálculo para Single Leg
            if str(row.get('Estrategia')).lower() == "single leg" and precio_1 is not None:
                precio_cierre_actual = precio_1
                if es_credito:
                    pnl_bruto = (prima_entrada - precio_cierre_actual) * 100
                else:
                    pnl_bruto = (precio_cierre_actual - prima_entrada) * 100

            # Cálculo para Spread
            elif str(row.get('Estrategia')).lower() == "spread" and precio_1 is not None and precio_2 is not None:
                valor_actual_spread = abs(float(precio_1) - float(precio_2))
                if es_credito:
                    pnl_bruto = (prima_entrada - valor_actual_spread) * 100
                else:
                    pnl_bruto = (valor_actual_spread - prima_entrada) * 100

            # 4. Cálculo Neto y Porcentaje
            if pnl_bruto is not None:
                pnl_neto = pnl_bruto - comision
                try:
                    pnl_porcentaje = (pnl_neto / (prima_entrada * 100)) * 100
                except Exception:
                    pnl_porcentaje = None
        
        # 5. Guardar resultados
        df.at[idx, 'PnL_Bruto'] = pnl_bruto
        df.at[idx, 'PnL_Neto'] = pnl_neto
        df.at[idx, 'PnL_Porcentaje'] = pnl_porcentaje
        
        progress_bar.progress((idx + 1) / total)
    
    progress_bar.empty()
    status_text.empty()
    
    guardar_operaciones(df)
    return df

# =========================================================================
# 4. FUNCIONES DE VISUALIZACIÓN
# =========================================================================

def safe_strftime(d):
    """Formatea date -> dd/mm/YYYY o devuelve 'N/A' si es None/invalid."""
    if isinstance(d, date):
        return d.strftime('%d/%m/%Y')
    return "N/A"

def safe_number_text(x, fmt="${:.2f}", na_text="N/A"):
    try:
        if pd.isna(x) or x is None:
            return na_text
        val = float(x)
        return fmt.format(val)
    except Exception:
        return na_text

def display_operation_card_v2(row):
    """Muestra una operación activa con PnL, Delta y Theta en formato de tarjeta."""
    
    ticker = row.get('Ticker', 'N/A')
    estrategia = row.get('Estrategia', 'N/A')
    fecha_entrada = safe_strftime(row.get('Fecha_Entrada'))
    fecha_salida = safe_strftime(row.get('Fecha_Salida'))
    
    pnl_neto = row.get('PnL_Neto')
    delta_1 = row.get('Delta_1')
    delta_2 = row.get('Delta_2')
    theta_1 = row.get('Theta_1')
    theta_2 = row.get('Theta_2')
    
    # PnL
    if pnl_neto is None or pd.isna(pnl_neto):
        pnl_color = "#f0ad4e" # Naranja para No Data
        pnl_text = "N/A"
    else:
        try:
            pnl_val = float(pnl_neto)
            if pnl_val >= 0:
                pnl_color = "#5cb85c"
            else:
                pnl_color = "#d9534f"
            pnl_text = f"${pnl_val:.2f}"
        except Exception:
            pnl_color = "#f0ad4e"
            pnl_text = "N/A"

    # Delta Total (sumar solo valores válidos)
    delta_total = 0.0
    delta_present = False
    for v in (delta_1, delta_2):
        try:
            if pd.notna(v):
                delta_total += float(v)
                delta_present = True
        except Exception:
            continue
    delta_text = f"{delta_total:.2f}" if delta_present else "N/A"

    # Theta Total
    theta_total = 0.0
    theta_present = False
    for v in (theta_1, theta_2):
        try:
            if pd.notna(v):
                theta_total += float(v)
                theta_present = True
        except Exception:
            continue
    theta_text = f"{theta_total:.2f}" if theta_present else "N/A"
    
    # Descripción breve de la estrategia y strikes (con seguridad)
    strike_1_txt = "N/A"
    try:
        s1 = row.get('Strike_1')
        if pd.notna(s1):
            strike_1_txt = f"{float(s1):.2f}"
    except Exception:
        pass
    tipo_1 = row.get('Tipo_1', '')
    tipo_1_letter = tipo_1[0] if tipo_1 else ''

    brief_description = f"{strike_1_txt}{tipo_1_letter}"
    if str(estrategia).lower() == "spread":
        strike_2_txt = ""
        tipo_2_letter = ""
        try:
            s2 = row.get('Strike_2')
            if pd.notna(s2):
                strike_2_txt = f"{float(s2):.2f}"
        except Exception:
            pass
        tipo_2 = row.get('Tipo_2', '')
        tipo_2_letter = tipo_2[0] if tipo_2 else ''
        if strike_2_txt:
            brief_description = f"{strike_1_txt}{tipo_1_letter}/{strike_2_txt}{tipo_2_letter}"

    # Estilo del borde izquierdo para todos menos el primero (fácil de ajustar si quieres)
    border_style = "border-left: 1px dotted #444;" 
    
    st.markdown(f"""
    <div class="op-card">
        <div class="op-header">
            <h4 style="color: white; margin: 0; font-size: 1.5rem;">
                {int(row.get('ID')) if pd.notna(row.get('ID')) else ''} - <strong>{ticker}</strong> ({estrategia} - {brief_description})
            </h4>
            <span style="font-size: 0.9rem; color: #ccc;">DTE: <strong>{int(row.get('DTE')) if pd.notna(row.get('DTE')) else 'N/A'}</strong></span>
        </div>
        
        <div class="op-data">
            <div class="op-item"> 
                <span style="color: #aaa; font-size: 0.8rem;">Fecha Entrada</span><br>
                <strong style="color: white; font-size: 1rem;">{fecha_entrada}</strong>
            </div>
            <div class="op-item" style="{border_style}"> 
                <span style="color: #aaa; font-size: 0.8rem;">Fecha Salida</span><br>
                <strong style="color: white; font-size: 1rem;">{fecha_salida}</strong>
            </div>
            <div class="op-item" style="{border_style}"> 
                <span style="color: #aaa; font-size: 0.8rem;">💰 P&L Neto</span><br>
                <strong style="color: {pnl_color}; font-size: 1.2rem;">{pnl_text}</strong>
            </div>
            <div class="op-item" style="{border_style}"> 
                <span style="color: #aaa; font-size: 0.8rem;">Δ Total</span><br>
                <strong style="color: white; font-size: 1.2rem;">{delta_text}</strong>
            </div>
            <div class="op-item" style="{border_style}"> 
                <span style="color: #aaa; font-size: 0.8rem;">Θ Total</span><br>
                <strong style="color: white; font-size: 1.2rem;">{theta_text}</strong>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================================
# 5. INTERFAZ PRINCIPAL
# =========================================================================

def option_tracker_page():
    # Header con estilo
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
        if api_key is None or app_secret is None or redirect_uri is None:
            st.error("❌ Falta configurar los secrets de Schwab. Funciones limitadas.")
        else:
            st.error("❌ No hay conexión con Schwab. Funciones limitadas.")
    
    st.markdown("---")
    
    # SECCIÓN 1: AGREGAR NUEVA OPERACIÓN
    st.markdown("### ➕ Nueva Operación")
    
    with st.expander("📝 Formulario de entrada", expanded=False):
        # Primero seleccionar estrategia para condicionar el resto
        estrategia = st.selectbox("📊 Estrategia", ["Single Leg", "Spread"], key="estrategia_select")
        
        with st.form("form_nueva_operacion", clear_on_submit=True):
            # Información básica
            st.markdown("#### 📋 Información Básica")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                ticker = st.text_input("🎯 Ticker", placeholder="AAPL", help="Símbolo del activo")
            
            with col2:
                es_credito = st.checkbox("💰 Es Crédito", value=True, help="Marca si recibiste crédito (vendiste). Desmarca si pagaste débito (compraste)")
            
            with col3:
                tipo_comision = st.selectbox("💳 Tipo Comisión", ["Por Leg", "Total"])
            
            st.markdown("---")
            
            # Comisiones según tipo
            if tipo_comision == "Total":
                comision_total_input = st.number_input("💵 Comisión Total ($)", min_value=0.0, value=1.30, step=0.01, format="%.2f")
                if estrategia == "Spread":
                    comision_leg1 = comision_total_input / 2
                    comision_leg2 = comision_total_input / 2
                else:
                    comision_leg1 = comision_total_input
                    comision_leg2 = 0.0
            else:
                col1, col2 = st.columns(2)
                with col1:
                    comision_leg1 = st.number_input("💵 Comisión Leg 1 ($)", min_value=0.0, value=0.65, step=0.01, format="%.2f")
                with col2:
                    if estrategia == "Spread":
                        comision_leg2 = st.number_input("💵 Comisión Leg 2 ($)", min_value=0.0, value=0.65, step=0.01, format="%.2f")
                    else:
                        comision_leg2 = 0.0
            
            st.markdown("---")
            
            # LEG 1
            st.markdown("#### 🎯 Leg 1")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                strike_1 = st.number_input("Strike 1", min_value=0.0, step=0.5, format="%.2f")
            
            with col2:
                tipo_1 = st.selectbox("Tipo 1", ["CALL", "PUT"])
            
            with col3:
                posicion_1 = st.selectbox("Posición 1", ["LONG", "SHORT"])
            
            # LEG 2 (solo si es Spread)
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
            
            # Fechas y Prima
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
                    value=(datetime.now().date() + timedelta(days=15)),
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
            
            # Calcular DTE automáticamente
            if fecha_salida and fecha_entrada:
                dte_calculado = (fecha_salida - fecha_entrada).days
                comision_total_display = float(comision_leg1 or 0.0) + float(comision_leg2 or 0.0)
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"📊 DTE Calculado: **{dte_calculado} días**")
                with col2:
                    st.info(f"💵 Comisión Total: **${comision_total_display:.2f}**")
            
            st.markdown("---")
            
            submit_button = st.form_submit_button("✅ Agregar Operación", use_container_width=True, type="primary")
            
            if submit_button:
                if ticker and strike_1 > 0:
                    if estrategia == "Spread" and (strike_2 is None or strike_2 <= 0):
                        st.error("❌ Para Spreads debes completar ambos legs")
                    else:
                        if agregar_operacion(ticker, estrategia, strike_1, tipo_1, posicion_1,
                                             strike_2, tipo_2, posicion_2,
                                             fecha_entrada, fecha_salida, prima_entrada, es_credito,
                                             comision_leg1, comision_leg2):
                            st.success(f"✅ Operación agregada: {ticker} {estrategia} ({'CRÉDITO' if es_credito else 'DÉBITO'})")
                            st.rerun()
                else:
                    st.error("❌ Completa todos los campos obligatorios")
    
    st.markdown("---")
    
    # SECCIÓN 2: OPERACIONES ACTIVAS
    st.markdown("### 📊 Portfolio Activo")
    
    df = cargar_operaciones()
    
    # Lógica para refrescar automáticamente al cargar, si es la primera vez y hay conexión
    if client and not df.empty and df['Precio_Actual_1'].isna().all():
        with st.spinner("Cargando datos iniciales desde Schwab..."):
            df = refrescar_todas_operaciones(client)
    
    if df.empty:
        st.markdown('<div class="info-box">📝 No hay operaciones activas. ¡Agrega tu primera operación arriba!</div>', unsafe_allow_html=True)
    else:
        # Botones de acción
        col1, col2, col3 = st.columns([2, 2, 3])
        
        with col1:
            if st.button("🔄 Refrescar Datos", type="primary", disabled=(client is None), use_container_width=True):
                if client:
                    with st.spinner("Actualizando desde Schwab..."):
                        df = refrescar_todas_operaciones(client)
                    st.success("✅ Datos actualizados")
                    st.rerun()
        
        with col2:
            # Botón de Descargar CSV
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
        
        # Métricas del portfolio
        st.markdown("#### 💼 Resumen del Portfolio")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Cálculo del P&L Neto Total
        pnl_neto_total = df['PnL_Neto'].apply(lambda x: float(x) if pd.notna(x) else 0).sum()
        
        # Para el color de la métrica:
        delta_color = "normal" if pnl_neto_total >= 0 else "inverse"
        
        with col1:
            st.metric("📊 Operaciones", len(df))
        
        with col2:
            st.metric("💰 P&L Neto Total", f"${pnl_neto_total:.2f}", 
                      delta=f"{pnl_neto_total:.2f}", delta_color=delta_color)
        
        with col3:
            # Delta Agregado
            delta_agregado = df['Delta_1'].fillna(0).sum() + df['Delta_2'].fillna(0).sum()
            st.metric("Δ Portfolio", f"{delta_agregado:.2f}")

        with col4:
            # Theta Agregado
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
        
        # =========================================================================
        # SECCIÓN DE TARJETAS DE OPERACIONES (El formato solicitado)
        # =========================================================================
        
        st.markdown("### 📋 Detalle de Operaciones")
        
        # Iterar sobre cada operación y mostrarla como una Tarjeta (Card)
        for _, row in df.iterrows():
            display_operation_card_v2(row)
        
        st.markdown("---")
        
        # Eliminar operación
        with st.expander("🗑️ Eliminar Operación"):
            col1, col2 = st.columns([1, 2])
            
            # Ajustar max_value para evitar errores si no hay filas
            max_id = int(df['ID'].max()) if not df.empty and pd.notna(df['ID'].max()) else 1
            
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
                        if eliminar_operacion(id_eliminar):
                            st.success(f"✅ Eliminada: {operacion['Ticker']} - {operacion['Estrategia']}")
                            st.rerun()
                    else:
                        st.error(f"❌ No existe ID {id_eliminar}")

# =========================================================================
# 6. PUNTO DE ENTRADA
# =========================================================================

if __name__ == "__main__":
    if check_password():
        option_tracker_page()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Introduce tus credenciales en el menú lateral para acceder.")
