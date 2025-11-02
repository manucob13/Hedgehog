# pages/Option Tracker.py - MONITOREO DE OPCIONES
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

st.set_page_config(page_title="Option Tracker", layout="wide")

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
        
        # Verificar token
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
        df['Fecha_Entrada'] = pd.to_datetime(df['Fecha_Entrada']).dt.date
        df['Fecha_Expiracion'] = pd.to_datetime(df['Fecha_Expiracion']).dt.date
        return df
    else:
        # Crear DataFrame vacío con estructura
        return pd.DataFrame(columns=[
            'ID', 'Ticker', 'Strike', 'Tipo', 'Posicion', 
            'Fecha_Entrada', 'DTE', 'Fecha_Expiracion', 'Precio_Entrada',
            'Precio_Actual', 'Delta', 'Theta', 'PnL_Dolares', 'PnL_Porcentaje'
        ])

def guardar_operaciones(df):
    """Guarda operaciones en CSV"""
    df.to_csv(TRACKER_CSV, index=False)

def agregar_operacion(ticker, strike, tipo, posicion, fecha_entrada, dte, precio_entrada):
    """Agrega una nueva operación"""
    df = cargar_operaciones()
    
    # Calcular fecha de expiración
    fecha_expiracion = fecha_entrada + timedelta(days=int(dte))
    
    # Generar ID único
    if len(df) > 0:
        nuevo_id = df['ID'].max() + 1
    else:
        nuevo_id = 1
    
    # Crear nueva fila
    nueva_operacion = pd.DataFrame([{
        'ID': nuevo_id,
        'Ticker': ticker.upper(),
        'Strike': strike,
        'Tipo': tipo,
        'Posicion': posicion,
        'Fecha_Entrada': fecha_entrada,
        'DTE': dte,
        'Fecha_Expiracion': fecha_expiracion,
        'Precio_Entrada': precio_entrada,
        'Precio_Actual': None,
        'Delta': None,
        'Theta': None,
        'PnL_Dolares': None,
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

def obtener_datos_opcion(client, ticker, strike, tipo, fecha_expiracion):
    """Obtiene precio, delta y theta desde Schwab"""
    try:
        response = client.get_option_chain(ticker)
        if response.status_code != 200:
            return None, None, None
        
        opciones = response.json()
        
        # Seleccionar map según tipo
        if tipo == 'CALL':
            option_map = opciones.get('callExpDateMap', {})
        else:
            option_map = opciones.get('putExpDateMap', {})
        
        fecha_str = fecha_expiracion.strftime('%Y-%m-%d')
        
        # Buscar la fecha
        for fecha_key, strikes in option_map.items():
            if fecha_str in fecha_key:
                strike_str = str(float(strike))
                if strike_str in strikes:
                    contrato = strikes[strike_str][0]
                    
                    bid = contrato.get('bid', 0)
                    ask = contrato.get('ask', 0)
                    delta = contrato.get('delta', None)
                    theta = contrato.get('theta', None)
                    
                    # Calcular mid price
                    if bid and ask and bid > 0 and ask > 0:
                        mid_price = (bid + ask) / 2
                    else:
                        mid_price = None
                    
                    return mid_price, delta, theta
        
        return None, None, None
    except Exception as e:
        return None, None, None

def refrescar_todas_operaciones(client):
    """Refresca datos de todas las operaciones desde Schwab"""
    df = cargar_operaciones()
    
    if df.empty:
        return df
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, row in df.iterrows():
        status_text.text(f"Actualizando {row['Ticker']} ({idx+1}/{len(df)})...")
        
        precio_actual, delta, theta = obtener_datos_opcion(
            client,
            row['Ticker'],
            row['Strike'],
            row['Tipo'],
            row['Fecha_Expiracion']
        )
        
        # Actualizar datos
        df.at[idx, 'Precio_Actual'] = precio_actual
        df.at[idx, 'Delta'] = delta
        df.at[idx, 'Theta'] = theta
        
        # Calcular P&L si tenemos precio actual
        if precio_actual and row['Precio_Entrada']:
            precio_entrada = float(row['Precio_Entrada'])
            
            # Lógica P&L según posición
            if row['Posicion'] == 'SHORT':
                # SHORT: ganas cuando baja el precio (vendes caro, compras barato)
                pnl_dolares = (precio_entrada - precio_actual) * 100
                pnl_porcentaje = ((precio_entrada - precio_actual) / precio_entrada) * 100
            else:  # LONG
                # LONG: ganas cuando sube el precio (compras barato, vendes caro)
                pnl_dolares = (precio_actual - precio_entrada) * 100
                pnl_porcentaje = ((precio_actual - precio_entrada) / precio_entrada) * 100
            
            df.at[idx, 'PnL_Dolares'] = pnl_dolares
            df.at[idx, 'PnL_Porcentaje'] = pnl_porcentaje
        
        progress_bar.progress((idx + 1) / len(df))
    
    progress_bar.empty()
    status_text.empty()
    
    # Guardar datos actualizados
    guardar_operaciones(df)
    return df

# =========================================================================
# 4. INTERFAZ PRINCIPAL
# =========================================================================

def option_tracker_page():
    st.title("📊 Option Tracker - Monitor de Operaciones")
    st.markdown("---")
    
    # Conexión Schwab
    if 'schwab_client_tracker' not in st.session_state:
        with st.spinner("Conectando con Schwab..."):
            st.session_state.schwab_client_tracker = connect_to_schwab()
    
    client = st.session_state.schwab_client_tracker
    
    if client:
        st.success("✅ Conectado con Schwab")
    else:
        st.error("❌ No hay conexión con Schwab. Algunas funciones no estarán disponibles.")
    
    st.markdown("---")
    
    # SECCIÓN 1: AGREGAR NUEVA OPERACIÓN
    st.subheader("➕ Agregar Nueva Operación")
    
    with st.form("form_nueva_operacion", clear_on_submit=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ticker = st.text_input("Ticker", placeholder="AAPL", help="Símbolo del activo")
        
        with col2:
            strike = st.number_input("Strike", min_value=0.0, step=0.5, format="%.2f")
        
        with col3:
            tipo = st.selectbox("Tipo", ["CALL", "PUT"])
        
        with col4:
            posicion = st.selectbox("Posición", ["LONG", "SHORT"])
        
        col5, col6, col7 = st.columns(3)
        
        with col5:
            fecha_entrada = st.date_input(
                "Fecha de Entrada",
                value=datetime.now().date(),
                format="DD/MM/YYYY"
            )
        
        with col6:
            dte = st.number_input("DTE", min_value=1, value=15, help="Días hasta expiración")
        
        with col7:
            precio_entrada = st.number_input(
                "Precio Entrada ($)", 
                min_value=0.0, 
                step=0.01, 
                format="%.2f",
                help="Precio al que entraste (crédito recibido o débito pagado)"
            )
        
        submit_button = st.form_submit_button("✅ Agregar Operación", use_container_width=True)
        
        if submit_button:
            if ticker and strike > 0 and precio_entrada > 0:
                if agregar_operacion(ticker, strike, tipo, posicion, fecha_entrada, dte, precio_entrada):
                    st.success(f"✅ Operación agregada: {posicion} {ticker} ${strike} {tipo}")
                    st.rerun()
            else:
                st.error("❌ Por favor completa todos los campos")
    
    st.markdown("---")
    
    # SECCIÓN 2: OPERACIONES ACTUALES
    st.subheader("📋 Operaciones Activas")
    
    df = cargar_operaciones()
    
    if df.empty:
        st.info("👆 No hay operaciones. Agrega tu primera operación arriba.")
    else:
        # Botón refrescar
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🔄 Refrescar Datos desde Schwab", type="primary", disabled=(client is None)):
                if client:
                    with st.spinner("Actualizando datos..."):
                        df = refrescar_todas_operaciones(client)
                    st.success("✅ Datos actualizados correctamente")
                    st.rerun()
        
        with col2:
            if not df.empty and df['Precio_Actual'].notna().any():
                ultima_actualizacion = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                st.info(f"📅 Última actualización: {ultima_actualizacion}")
        
        st.markdown("---")
        
        # Preparar DataFrame para mostrar
        df_display = df.copy()
        df_display['Fecha_Entrada'] = pd.to_datetime(df_display['Fecha_Entrada']).dt.strftime('%d/%m/%Y')
        df_display['Fecha_Expiracion'] = pd.to_datetime(df_display['Fecha_Expiracion']).dt.strftime('%d/%m/%Y')
        
        # Formatear columnas numéricas
        for col in ['Precio_Entrada', 'Precio_Actual', 'Delta', 'Theta', 'PnL_Dolares', 'PnL_Porcentaje']:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(
                    lambda x: f"{x:.2f}" if pd.notna(x) else "-"
                )
        
        # Mostrar tabla
        st.dataframe(
            df_display,
            hide_index=True,
            use_container_width=True,
            column_config={
                "ID": st.column_config.NumberColumn("ID", width="small"),
                "Ticker": st.column_config.TextColumn("🎯 Ticker", width="small"),
                "Strike": st.column_config.NumberColumn("Strike", width="small", format="%.2f"),
                "Tipo": st.column_config.TextColumn("Tipo", width="small"),
                "Posicion": st.column_config.TextColumn("Pos", width="small"),
                "Fecha_Entrada": st.column_config.TextColumn("📅 Entrada", width="medium"),
                "DTE": st.column_config.NumberColumn("DTE", width="small"),
                "Fecha_Expiracion": st.column_config.TextColumn("📅 Exp", width="medium"),
                "Precio_Entrada": st.column_config.TextColumn("💵 P.Entrada", width="small"),
                "Precio_Actual": st.column_config.TextColumn("💵 P.Actual", width="small"),
                "Delta": st.column_config.TextColumn("Δ Delta", width="small"),
                "Theta": st.column_config.TextColumn("Θ Theta", width="small"),
                "PnL_Dolares": st.column_config.TextColumn("💰 P&L $", width="small"),
                "PnL_Porcentaje": st.column_config.TextColumn("📊 P&L %", width="small")
            }
        )
        
        st.markdown("---")
        
        # SECCIÓN 3: ELIMINAR OPERACIONES
        st.subheader("🗑️ Eliminar Operación")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            id_eliminar = st.number_input(
                "ID a eliminar",
                min_value=1,
                max_value=int(df['ID'].max()) if not df.empty else 1,
                step=1
            )
        
        with col2:
            if st.button("🗑️ Eliminar Operación", type="secondary"):
                if id_eliminar in df['ID'].values:
                    operacion = df[df['ID'] == id_eliminar].iloc[0]
                    if eliminar_operacion(id_eliminar):
                        st.success(f"✅ Eliminada: {operacion['Posicion']} {operacion['Ticker']} ${operacion['Strike']} {operacion['Tipo']}")
                        st.rerun()
                else:
                    st.error(f"❌ No existe operación con ID {id_eliminar}")
        
        # Métricas resumen
        st.markdown("---")
        st.subheader("📊 Resumen del Portfolio")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Operaciones", len(df))
        
        with col2:
            pnl_total = df['PnL_Dolares'].apply(lambda x: float(x) if pd.notna(x) and x != "-" else 0).sum()
            st.metric("P&L Total ($)", f"${pnl_total:.2f}")
        
        with col3:
            calls = len(df[df['Tipo'] == 'CALL'])
            st.metric("Calls", calls)
        
        with col4:
            puts = len(df[df['Tipo'] == 'PUT'])
            st.metric("Puts", puts)

# =========================================================================
# 5. PUNTO DE ENTRADA
# =========================================================================

if __name__ == "__main__":
    if check_password():
        option_tracker_page()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Introduce tus credenciales en el menú lateral para acceder.")
