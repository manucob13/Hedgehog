# pages/Option Tracker.py - MONITOREO DE OPCIONES CON SPREADS (FORMATO UNIFICADO)
import streamlit as st
import pandas as pd
from datetime import timedelta, datetime
import os
import schwab
from schwab.auth import easy_client
from schwab.client import Client
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
except KeyError as e:
    st.error(f"❌ Falta configurar los secrets de Schwab. Clave faltante: {e}")
    st.stop()

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
</style>
""", unsafe_allow_html=True)

# =========================================================================
# 1. CONEXIÓN SCHWAB
# =========================================================================

def connect_to_schwab():
    """Conecta con Schwab usando token existente"""
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
        
        # Prueba de conexión
        test_response = client.get_quote("AAPL")
        if hasattr(test_response, "status_code") and test_response.status_code != 200:
            raise Exception(f"Respuesta inesperada: {test_response.status_code}")
        
        return client
    except Exception as e:
        st.error(f"❌ Error al conectar con Schwab: {e}")
        return None

# =========================================================================
# 2. FUNCIONES DE DATOS
# =========================================================================

def cargar_operaciones():
    """Carga operaciones desde CSV"""
    if os.path.exists(TRACKER_CSV):
        df = pd.read_csv(TRACKER_CSV)
        
        # Convertir fechas
        if 'Fecha_Entrada' in df.columns:
            df['Fecha_Entrada'] = pd.to_datetime(df['Fecha_Entrada']).dt.date
        
        # Compatibilidad con CSV antiguo que no tiene Fecha_Salida
        if 'Fecha_Salida' not in df.columns and 'DTE' in df.columns:
            fechas_salida = []
            for _, row in df.iterrows():
                fecha_salida = row['Fecha_Entrada'] + timedelta(days=int(row['DTE']))
                fechas_salida.append(fecha_salida)
            df['Fecha_Salida'] = fechas_salida
        elif 'Fecha_Salida' in df.columns:
            df['Fecha_Salida'] = pd.to_datetime(df['Fecha_Salida']).dt.date
        
        # Migrar Prima_Entrada si existe con valores negativos (para compatibilidad)
        if 'Prima_Entrada' in df.columns:
            if 'Es_Credito' not in df.columns:
                # Si Prima_Entrada es negativa, se asume Crédito.
                df['Es_Credito'] = df['Prima_Entrada'].apply(lambda x: x < 0 if pd.notna(x) else True)
                # Normalizar Prima_Entrada a valores absolutos
                df['Prima_Entrada'] = df['Prima_Entrada'].abs()
        else:
            # Si no existe Prima_Entrada, crear columnas vacías
            df['Prima_Entrada'] = 0.0
            df['Es_Credito'] = True
        
        # Añadir columnas de comisión si no existen
        if 'Comision_Leg1' not in df.columns:
            df['Comision_Leg1'] = 0.65
        if 'Comision_Leg2' not in df.columns:
            df['Comision_Leg2'] = 0.65
        if 'Comision' not in df.columns:
            # Recalcular Comisión total para evitar errores de NaN
            df['Comision'] = df['Comision_Leg1'].fillna(0) + df['Comision_Leg2'].fillna(0)
        
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
    """Guarda operaciones en CSV"""
    df.to_csv(TRACKER_CSV, index=False)

def agregar_operacion(ticker, estrategia, strike_1, tipo_1, posicion_1,
                      strike_2, tipo_2, posicion_2,
                      fecha_entrada, fecha_salida, prima_entrada, es_credito, 
                      comision_leg1, comision_leg2):
    """Agrega una nueva operación"""
    df = cargar_operaciones()
    
    # Calcular DTE
    dte = (fecha_salida - fecha_entrada).days
    
    # Generar ID único
    nuevo_id = df['ID'].max() + 1 if len(df) > 0 else 1
    
    # Para single leg, los campos del leg 2 son None y la comisión es solo la del leg 1
    if estrategia == "Single Leg":
        strike_2 = None
        tipo_2 = None
        posicion_2 = None
        comision_leg2 = 0
    
    # Calcular comisión total
    comision_total = comision_leg1 + comision_leg2
    
    # Normalizar prima: guardarla como positiva, el signo se maneja con 'Es_Credito'
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
    """Obtiene precio, delta y theta desde Schwab"""
    try:
        response = client.get_option_chain(ticker)
        if response.status_code != 200:
            return None, None, None
        
        opciones = response.json()
        
        if tipo == 'CALL':
            option_map = opciones.get('callExpDateMap', {})
        else:
            option_map = opciones.get('putExpDateMap', {})
        
        fecha_str = fecha_salida.strftime('%Y-%m-%d')
        
        # Las claves tienen formato 'YYYY-MM-DD:DTE'
        fecha_key_match = None
        for key in option_map.keys():
            if key.startswith(fecha_str):
                fecha_key_match = key
                break
        
        if fecha_key_match:
            strikes = option_map[fecha_key_match]
            strike_str = str(float(strike))
            
            if strike_str in strikes:
                # Tomamos el primer contrato (no deberia haber mas de uno para el mismo strike/fecha/tipo)
                contrato = strikes[strike_str][0] 
                
                bid = contrato.get('bid', 0)
                ask = contrato.get('ask', 0)
                delta = contrato.get('delta', None)
                theta = contrato.get('theta', None)
                
                # Usamos el precio medio (mid price) para el valor actual del contrato
                if bid > 0 and ask > 0:
                    mid_price = (bid + ask) / 2
                else:
                    mid_price = None # No se pudo obtener un precio de mercado valido
                
                return mid_price, delta, theta
        
        return None, None, None
    except Exception:
        return None, None, None

def refrescar_todas_operaciones(client):
    """Refresca datos de todas las operaciones desde Schwab y calcula P&L"""
    df = cargar_operaciones()
    
    if df.empty:
        return df
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, row in df.iterrows():
        status_text.text(f"🔄 Actualizando {row['Ticker']} ({idx+1}/{len(df)})...")
        
        # 1. Obtener datos Leg 1
        precio_1, delta_1, theta_1 = obtener_datos_opcion(
            client, row['Ticker'], row['Strike_1'], row['Tipo_1'], row['Fecha_Salida']
        )
        
        df.at[idx, 'Precio_Actual_1'] = precio_1
        df.at[idx, 'Delta_1'] = delta_1
        df.at[idx, 'Theta_1'] = theta_1
        
        # 2. Obtener datos Leg 2 (si existe)
        precio_2 = None
        if row['Estrategia'] == "Spread" and pd.notna(row['Strike_2']):
            precio_2, delta_2, theta_2 = obtener_datos_opcion(
                client, row['Ticker'], row['Strike_2'], row['Tipo_2'], row['Fecha_Salida']
            )
            df.at[idx, 'Precio_Actual_2'] = precio_2
            df.at[idx, 'Delta_2'] = delta_2
            df.at[idx, 'Theta_2'] = theta_2
        else:
            # Aseguramos None para Single Leg
            df.at[idx, 'Precio_Actual_2'] = None 
            df.at[idx, 'Delta_2'] = None
            df.at[idx, 'Theta_2'] = None
        
        # 3. Inicializar P&L
        pnl_bruto = None
        pnl_neto = None
        pnl_porcentaje = None
        
        # Solo calcular si tenemos la Prima de Entrada y algún precio de mercado
        if float(row['Prima_Entrada']) > 0:
            prima_entrada = float(row['Prima_Entrada'])
            comision = float(row['Comision'])
            es_credito = bool(row['Es_Credito'])
            
            # --- LÓGICA DE CÁLCULO CORREGIDA PARA CRÉDITO Y DÉBITO ---
            
            # Cálculo para Single Leg
            if row['Estrategia'] == "Single Leg" and precio_1 is not None:
                precio_cierre_actual = precio_1 # Valor actual del contrato
                
                if es_credito:
                    # ✅ CRÉDITO: PnL = (Prima Recibida - Costo de Cierre) * 100
                    pnl_bruto = (prima_entrada - precio_cierre_actual) * 100
                else:
                    # 🔵 DÉBITO: PnL = (Valor Actual - Prima Pagada) * 100
                    pnl_bruto = (precio_cierre_actual - prima_entrada) * 100
            
            # Cálculo para Spread
            elif row['Estrategia'] == "Spread" and precio_1 is not None and precio_2 is not None:
                # El valor actual del Spread es la diferencia absoluta de los precios de los legs
                valor_actual_spread = abs(precio_1 - precio_2)
                
                if es_credito:
                    # ✅ CRÉDITO: PnL = (Prima Recibida - Costo de Cierre del Spread) * 100
                    pnl_bruto = (prima_entrada - valor_actual_spread) * 100
                else:
                    # 🔵 DÉBITO: PnL = (Valor Actual del Spread - Prima Pagada) * 100
                    pnl_bruto = (valor_actual_spread - prima_entrada) * 100
            
            # 4. Cálculo Neto y Porcentaje
            if pnl_bruto is not None:
                pnl_neto = pnl_bruto - comision
                # La base para el porcentaje siempre es la prima pagada/recibida (en dólares, por contrato)
                pnl_porcentaje = (pnl_neto / (prima_entrada * 100)) * 100
            
        # 5. Guardar resultados
        df.at[idx, 'PnL_Bruto'] = pnl_bruto
        df.at[idx, 'PnL_Neto'] = pnl_neto
        df.at[idx, 'PnL_Porcentaje'] = pnl_porcentaje
        
        progress_bar.progress((idx + 1) / len(df))
    
    progress_bar.empty()
    status_text.empty()
    
    guardar_operaciones(df)
    return df

# =========================================================================
# 4. INTERFAZ PRINCIPAL
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
        st.error("❌ No hay conexión con Schwab. Funciones limitadas.")
    
    st.markdown("---")
    
    # SECCIÓN 1: AGREGAR NUEVA OPERACIÓN
    st.markdown("### ➕ Nueva Operación")
    
    with st.expander("📝 Formulario de entrada", expanded=True):
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
                    comision_leg2 = 0
            else:
                col1, col2 = st.columns(2)
                with col1:
                    comision_leg1 = st.number_input("💵 Comisión Leg 1 ($)", min_value=0.0, value=0.65, step=0.01, format="%.2f")
                with col2:
                    if estrategia == "Spread":
                        comision_leg2 = st.number_input("💵 Comisión Leg 2 ($)", min_value=0.0, value=0.65, step=0.01, format="%.2f")
                    else:
                        comision_leg2 = 0
            
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
                    # En un spread, la segunda posición DEBE ser opuesta a la primera.
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
            
            # Calcular DTE automáticamente
            if fecha_salida and fecha_entrada:
                dte_calculado = (fecha_salida - fecha_entrada).days
                comision_total_display = comision_leg1 + (comision_leg2 if estrategia == "Spread" else 0)
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
                label="💾 Descargar CSV",
                data=csv,
                file_name=f"option_tracker_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            if not df.empty and df['PnL_Neto'].notna().any():
                st.info(f"📅 Actualizado: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        
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
            spreads = len(df[df['Estrategia'] == 'Spread'])
            st.metric("📐 Spreads", spreads)
        
        with col4:
            singles = len(df[df['Estrategia'] == 'Single Leg'])
            st.metric("🎯 Single Legs", singles)
        
        with col5:
            prima_total_invertida = df['Prima_Entrada'].apply(lambda x: float(x) if pd.notna(x) else 0).sum() * 100
            if prima_total_invertida != 0:
                pnl_neto_portfolio = df['PnL_Neto'].apply(lambda x: float(x) if pd.notna(x) else 0).sum()
                pnl_porcentaje_portfolio = (pnl_neto_portfolio / prima_total_invertida) * 100
                st.metric("📈 P&L % Portfolio", f"{pnl_porcentaje_portfolio:.1f}%")
            else:
                st.metric("📈 P&L % Portfolio", "0.0%")
        
        st.markdown("---")
        
        # Tabla de operaciones
        st.markdown("#### 📋 Detalle de Operaciones")
        
        df_display = df.copy()
        df_display['Fecha_Entrada'] = pd.to_datetime(df_display['Fecha_Entrada']).dt.strftime('%d/%m/%Y')
        df_display['Fecha_Salida'] = pd.to_datetime(df_display['Fecha_Salida']).dt.strftime('%d/%m/%Y')
        
        # =================================================================
        # UNIFICACIÓN DE FORMATO
        # =================================================================
        # Rellenar con guiones para operaciones Single Leg en Leg 2
        cols_to_fill = ['Strike_2', 'Tipo_2', 'Posicion_2']
        
        for col in cols_to_fill:
            df_display[col] = df_display.apply(
                lambda row: '-' if row['Estrategia'] == 'Single Leg' else (f"{row[col]:.2f}" if pd.notna(row[col]) and isinstance(row[col], (int, float)) else str(row[col])),
                axis=1
            )
        
        # Formatear el resto de columnas numéricas
        numeric_cols_for_format = ['Strike_1', 'Prima_Entrada', 'Comision_Leg1', 'Comision_Leg2', 'Comision',
                                   'Precio_Actual_1', 'Delta_1', 'Theta_1',
                                   'Precio_Actual_2', 'Delta_2', 'Theta_2',
                                   'PnL_Bruto', 'PnL_Neto', 'PnL_Porcentaje']
        
        for col in numeric_cols_for_format:
            # Solo aplicar si la columna no fue ya formateada con el condicional de Spread
            if col not in cols_to_fill: 
                df_display[col] = df_display[col].apply(
                    lambda x: f"{x:.2f}" if pd.notna(x) and isinstance(x, (int, float)) else "-"
                )

        
        st.dataframe(
            df_display,
            hide_index=True,
            use_container_width=True,
            column_config={
                # =========================================================
                # COLUMNAS CLAVE: Visibles para todas las operaciones
                # =========================================================
                "ID": st.column_config.NumberColumn("ID", width="small"),
                "Ticker": st.column_config.TextColumn("🎯 Ticker", width="small"),
                "Estrategia": st.column_config.TextColumn("📊 Estrategia", width="medium"),
                "DTE": st.column_config.NumberColumn("DTE", width="small"),
                "Prima_Entrada": st.column_config.TextColumn("💵 Prima", width="small"),
                "Comision": st.column_config.TextColumn("💳 ComTotal", width="small"),
                "PnL_Neto": st.column_config.TextColumn("💰 P&L Neto", width="small"),
                "PnL_Porcentaje": st.column_config.TextColumn("📊 P&L %", width="small"),

                # =========================================================
                # DETALLE UNIFICADO (Leg 1 y Leg 2)
                # =========================================================
                "Strike_1": st.column_config.TextColumn("Strike 1", width="small"),
                "Tipo_1": st.column_config.TextColumn("Tipo 1", width="small"),
                "Posicion_1": st.column_config.TextColumn("Pos 1", width="small"),
                
                "Strike_2": st.column_config.TextColumn("Strike 2", width="small"), # Será '-' en Single Leg
                "Tipo_2": st.column_config.TextColumn("Tipo 2", width="small"),     # Será '-' en Single Leg
                "Posicion_2": st.column_config.TextColumn("Pos 2", width="small"),  # Será '-' en Single Leg

                # Precio actual (P1)
                "Precio_Actual_1": st.column_config.TextColumn("💰 P1", width="small"),
                "Delta_1": st.column_config.TextColumn("Δ1", width="small"),
                "Fecha_Salida": st.column_config.TextColumn("📅 Expiración", width="medium"),
                
                # =========================================================
                # COLUMNAS OCULTAS PARA SIMPLIFICAR EL DETALLE
                # =========================================================
                "Fecha_Entrada": "hidden",
                "Es_Credito": "hidden",
                "Comision_Leg1": "hidden",
                "Comision_Leg2": "hidden",
                "PnL_Bruto": "hidden",
                "Theta_1": "hidden",
                "Precio_Actual_2": "hidden",
                "Delta_2": "hidden",
                "Theta_2": "hidden"
            }
        )
        
        st.markdown("---")
        
        # Eliminar operación
        with st.expander("🗑️ Eliminar Operación"):
            col1, col2 = st.columns([1, 2])
            
            # Ajustar max_value para evitar errores si no hay filas
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
                        if eliminar_operacion(id_eliminar):
                            st.success(f"✅ Eliminada: {operacion['Ticker']} - {operacion['Estrategia']}")
                            st.rerun()
                    else:
                        st.error(f"❌ No existe ID {id_eliminar}")

# =========================================================================
# 5. PUNTO DE ENTRADA
# =========================================================================

if __name__ == "__main__":
    if check_password():
        option_tracker_page()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Introduce tus credenciales en el menú lateral para acceder.")
