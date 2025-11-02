# pages/Option Tracker.py - MONITOREO DE OPCIONES CON SPREADS
import streamlit as st
import pandas as pd
from datetime import timedelta, datetime
import os
import schwab
from schwab.auth import easy_client
from schwab.client import Client
from utils import check_password

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
        df['Fecha_Entrada'] = pd.to_datetime(df['Fecha_Entrada']).dt.date
        
        # Compatibilidad con CSV antiguo que no tiene Fecha_Salida
        if 'Fecha_Salida' not in df.columns:
            # Migrar desde DTE a Fecha_Salida
            fechas_salida = []
            for _, row in df.iterrows():
                fecha_salida = row['Fecha_Entrada'] + timedelta(days=int(row['DTE']))
                fechas_salida.append(fecha_salida)
            df['Fecha_Salida'] = fechas_salida
        else:
            df['Fecha_Salida'] = pd.to_datetime(df['Fecha_Salida']).dt.date
        
        # Añadir columnas de comisión si no existen
        if 'Comision_Leg1' not in df.columns:
            df['Comision_Leg1'] = 0.65
        if 'Comision_Leg2' not in df.columns:
            df['Comision_Leg2'] = 0.65
        if 'Comision' not in df.columns:
            df['Comision'] = df['Comision_Leg1'] + df['Comision_Leg2']
        
        return df
    else:
        return pd.DataFrame(columns=[
            'ID', 'Ticker', 'Estrategia',
            'Strike_1', 'Tipo_1', 'Posicion_1',
            'Strike_2', 'Tipo_2', 'Posicion_2',
            'Fecha_Entrada', 'Fecha_Salida', 'DTE',
            'Prima_Entrada', 'Comision_Leg1', 'Comision_Leg2', 'Comision',
            'Precio_Actual_1', 'Delta_1', 'Theta_1',
            'Precio_Actual_2', 'Delta_2', 'Theta_2',
            'PnL_Bruto', 'PnL_Neto', 'PnL_Porcentaje'
        ])

def guardar_operaciones(df):
    """Guarda operaciones en CSV"""
    df.to_csv(TRACKER_CSV, index=False)

def agregar_operacion(ticker, estrategia, strike_1, tipo_1, posicion_1,
                      strike_2, tipo_2, posicion_2,
                      fecha_entrada, fecha_salida, prima_entrada, comision_leg1, comision_leg2):
    """Agrega una nueva operación"""
    df = cargar_operaciones()
    
    # Calcular DTE
    dte = (fecha_salida - fecha_entrada).days
    
    # Generar ID único
    nuevo_id = df['ID'].max() + 1 if len(df) > 0 else 1
    
    # Calcular comisión total
    comision_total = comision_leg1 + comision_leg2
    
    # Para single leg, los campos del leg 2 son None
    if estrategia == "Single Leg":
        strike_2 = None
        tipo_2 = None
        posicion_2 = None
        comision_leg2 = 0
        comision_total = comision_leg1
    
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
        'Prima_Entrada': prima_entrada,
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
        
        for fecha_key, strikes in option_map.items():
            if fecha_str in fecha_key:
                strike_str = str(float(strike))
                if strike_str in strikes:
                    contrato = strikes[strike_str][0]
                    
                    bid = contrato.get('bid', 0)
                    ask = contrato.get('ask', 0)
                    delta = contrato.get('delta', None)
                    theta = contrato.get('theta', None)
                    
                    if bid and ask and bid > 0 and ask > 0:
                        mid_price = (bid + ask) / 2
                    else:
                        mid_price = None
                    
                    return mid_price, delta, theta
        
        return None, None, None
    except Exception:
        return None, None, None

def refrescar_todas_operaciones(client):
    """Refresca datos de todas las operaciones desde Schwab"""
    df = cargar_operaciones()
    
    if df.empty:
        return df
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, row in df.iterrows():
        status_text.text(f"🔄 Actualizando {row['Ticker']} ({idx+1}/{len(df)})...")
        
        # Obtener datos Leg 1
        precio_1, delta_1, theta_1 = obtener_datos_opcion(
            client, row['Ticker'], row['Strike_1'], row['Tipo_1'], row['Fecha_Salida']
        )
        
        df.at[idx, 'Precio_Actual_1'] = precio_1
        df.at[idx, 'Delta_1'] = delta_1
        df.at[idx, 'Theta_1'] = theta_1
        
        # Obtener datos Leg 2 (si existe)
        if row['Estrategia'] == "Spread" and pd.notna(row['Strike_2']):
            precio_2, delta_2, theta_2 = obtener_datos_opcion(
                client, row['Ticker'], row['Strike_2'], row['Tipo_2'], row['Fecha_Salida']
            )
            df.at[idx, 'Precio_Actual_2'] = precio_2
            df.at[idx, 'Delta_2'] = delta_2
            df.at[idx, 'Theta_2'] = theta_2
        
        # Calcular P&L
        if precio_1 is not None:
            prima_entrada = float(row['Prima_Entrada'])
            comision = float(row['Comision'])
            
            if row['Estrategia'] == "Single Leg":
                # Single leg
                if row['Posicion_1'] == 'SHORT':
                    # Crédito: prima_entrada negativa, ganas cuando baja
                    pnl_bruto = (abs(prima_entrada) - precio_1) * 100
                else:  # LONG
                    # Débito: prima_entrada positiva, ganas cuando sube
                    pnl_bruto = (precio_1 - abs(prima_entrada)) * 100
            else:
                # Spread
                if precio_1 and row['Precio_Actual_2']:
                    precio_2 = float(row['Precio_Actual_2'])
                    
                    # Calcular valor actual del spread
                    if row['Posicion_1'] == 'SHORT':
                        valor_actual_spread = precio_1 - precio_2
                    else:
                        valor_actual_spread = precio_2 - precio_1
                    
                    # P&L: prima entrada (con signo) - valor actual
                    pnl_bruto = (abs(prima_entrada) - valor_actual_spread) * 100
                else:
                    pnl_bruto = None
            
            if pnl_bruto is not None:
                pnl_neto = pnl_bruto - comision
                pnl_porcentaje = (pnl_neto / (abs(prima_entrada) * 100)) * 100 if prima_entrada != 0 else 0
                
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
        with st.form("form_nueva_operacion", clear_on_submit=True):
            # Información básica
            st.markdown("#### 📋 Información Básica")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                ticker = st.text_input("🎯 Ticker", placeholder="AAPL", help="Símbolo del activo")
            
            with col2:
                estrategia = st.selectbox("📊 Estrategia", ["Single Leg", "Spread"])
            
            with col3:
                comision_leg1 = st.number_input("💵 Comisión Leg 1 ($)", min_value=0.0, value=0.65, step=0.01, format="%.2f")
            
            st.markdown("---")
            
            # LEG 1
            st.markdown("#### 🎯 Leg 1")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                strike_1 = st.number_input("Strike 1", min_value=0.0, step=0.5, format="%.2f", key="strike_1")
            
            with col2:
                tipo_1 = st.selectbox("Tipo 1", ["CALL", "PUT"], key="tipo_1")
            
            with col3:
                posicion_1 = st.selectbox("Posición 1", ["LONG", "SHORT"], key="pos_1")
            
            # LEG 2 (solo si es Spread)
            strike_2 = None
            tipo_2 = None
            posicion_2 = None
            comision_leg2 = 0
            
            if estrategia == "Spread":
                st.markdown("#### 🎯 Leg 2")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    strike_2 = st.number_input("Strike 2", min_value=0.0, step=0.5, format="%.2f", key="strike_2")
                
                with col2:
                    tipo_2 = st.selectbox("Tipo 2", ["CALL", "PUT"], key="tipo_2")
                
                with col3:
                    posicion_2 = st.selectbox("Posición 2", ["LONG", "SHORT"], key="pos_2")
                
                with col4:
                    comision_leg2 = st.number_input("💵 Comisión Leg 2 ($)", min_value=0.0, value=0.65, step=0.01, format="%.2f", key="com_leg2")
            
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
                    "Prima Entrada ($)", 
                    min_value=-1000.0,
                    max_value=1000.0,
                    value=0.0,
                    step=0.01, 
                    format="%.2f",
                    help="Negativo = Crédito recibido, Positivo = Débito pagado"
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
                                           fecha_entrada, fecha_salida, prima_entrada, comision_leg1, comision_leg2):
                            st.success(f"✅ Operación agregada exitosamente: {ticker} {estrategia}")
                            st.rerun()
                else:
                    st.error("❌ Completa todos los campos obligatorios")
    
    st.markdown("---")
    
    # SECCIÓN 2: OPERACIONES ACTIVAS
    st.markdown("### 📊 Portfolio Activo")
    
    df = cargar_operaciones()
    
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
            if st.button("💾 Descargar CSV", use_container_width=True):
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Descargar",
                    data=csv,
                    file_name=f"option_tracker_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            if not df.empty and df['Precio_Actual_1'].notna().any():
                st.info(f"📅 Actualizado: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        
        st.markdown("---")
        
        # Métricas del portfolio
        st.markdown("#### 💼 Resumen del Portfolio")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📊 Operaciones", len(df))
        
        with col2:
            pnl_neto_total = df['PnL_Neto'].apply(lambda x: float(x) if pd.notna(x) else 0).sum()
            delta_color = "normal" if pnl_neto_total >= 0 else "inverse"
            st.metric("💰 P&L Neto", f"${pnl_neto_total:.2f}", delta=f"${pnl_neto_total:.2f}", delta_color=delta_color)
        
        with col3:
            spreads = len(df[df['Estrategia'] == 'Spread'])
            st.metric("📐 Spreads", spreads)
        
        with col4:
            singles = len(df[df['Estrategia'] == 'Single Leg'])
            st.metric("🎯 Single Legs", singles)
        
        with col5:
            if pnl_neto_total != 0:
                avg_pnl_pct = df['PnL_Porcentaje'].apply(lambda x: float(x) if pd.notna(x) else 0).mean()
                st.metric("📈 Avg P&L %", f"{avg_pnl_pct:.1f}%")
            else:
                st.metric("📈 Avg P&L %", "0.0%")
        
        st.markdown("---")
        
        # Tabla de operaciones
        st.markdown("#### 📋 Detalle de Operaciones")
        
        df_display = df.copy()
        df_display['Fecha_Entrada'] = pd.to_datetime(df_display['Fecha_Entrada']).dt.strftime('%d/%m/%Y')
        df_display['Fecha_Salida'] = pd.to_datetime(df_display['Fecha_Salida']).dt.strftime('%d/%m/%Y')
        
        # Formatear columnas numéricas
        numeric_cols = ['Strike_1', 'Strike_2', 'Prima_Entrada', 'Comision_Leg1', 'Comision_Leg2', 'Comision',
                       'Precio_Actual_1', 'Delta_1', 'Theta_1',
                       'Precio_Actual_2', 'Delta_2', 'Theta_2',
                       'PnL_Bruto', 'PnL_Neto', 'PnL_Porcentaje']
        
        for col in numeric_cols:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(
                    lambda x: f"{x:.2f}" if pd.notna(x) else "-"
                )
        
        st.dataframe(
            df_display,
            hide_index=True,
            use_container_width=True,
            column_config={
                "ID": st.column_config.NumberColumn("ID", width="small"),
                "Ticker": st.column_config.TextColumn("🎯 Ticker", width="small"),
                "Estrategia": st.column_config.TextColumn("📊 Estrategia", width="medium"),
                "Strike_1": st.column_config.TextColumn("Strike 1", width="small"),
                "Tipo_1": st.column_config.TextColumn("Tipo 1", width="small"),
                "Posicion_1": st.column_config.TextColumn("Pos 1", width="small"),
                "Strike_2": st.column_config.TextColumn("Strike 2", width="small"),
                "Tipo_2": st.column_config.TextColumn("Tipo 2", width="small"),
                "Posicion_2": st.column_config.TextColumn("Pos 2", width="small"),
                "Fecha_Entrada": st.column_config.TextColumn("📅 Entrada", width="medium"),
                "Fecha_Salida": st.column_config.TextColumn("📅 Salida", width="medium"),
                "DTE": st.column_config.NumberColumn("DTE", width="small"),
                "Prima_Entrada": st.column_config.TextColumn("💵 Prima", width="small"),
                "Comision_Leg1": st.column_config.TextColumn("💳 Com1", width="small"),
                "Comision_Leg2": st.column_config.TextColumn("💳 Com2", width="small"),
                "Comision": st.column_config.TextColumn("💳 ComTotal", width="small"),
                "Precio_Actual_1": st.column_config.TextColumn("💰 P1", width="small"),
                "Delta_1": st.column_config.TextColumn("Δ1", width="small"),
                "Theta_1": st.column_config.TextColumn("Θ1", width="small"),
                "Precio_Actual_2": st.column_config.TextColumn("💰 P2", width="small"),
                "Delta_2": st.column_config.TextColumn("Δ2", width="small"),
                "Theta_2": st.column_config.TextColumn("Θ2", width="small"),
                "PnL_Bruto": st.column_config.TextColumn("💵 P&L Bruto", width="small"),
                "PnL_Neto": st.column_config.TextColumn("💰 P&L Neto", width="small"),
                "PnL_Porcentaje": st.column_config.TextColumn("📊 P&L %", width="small")
            }
        )
        
        st.markdown("---")
        
        # Eliminar operación
        with st.expander("🗑️ Eliminar Operación"):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                id_eliminar = st.number_input(
                    "ID a eliminar",
                    min_value=1,
                    max_value=int(df['ID'].max()) if not df.empty else 1,
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
