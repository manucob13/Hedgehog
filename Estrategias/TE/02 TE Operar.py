# 02 TE Operar.py
import streamlit as st
import pandas as pd
from datetime import date, timedelta
from utils.utils import check_password
from utils.utils_schwab import (
    connect_to_schwab,
    get_current_price_schwab,
    normalize_ticker
)

# --- CONFIGURACIÓN DE PÁGINA ---
# st.set_page_config(page_title="📊 TE Operar", layout="wide")

# ==============================================================================
# HELPERS
# ==============================================================================

def inicializar_session_state():
    """Inicializa las variables de session_state necesarias para esta página."""
    if 'te_operar_client' not in st.session_state:
        st.session_state['te_operar_client'] = None
    if 'te_operar_precio_spx' not in st.session_state:
        st.session_state['te_operar_precio_spx'] = None
    if 'te_operar_chain_raw' not in st.session_state:
        st.session_state['te_operar_chain_raw'] = None
    if 'te_operar_expiraciones' not in st.session_state:
        st.session_state['te_operar_expiraciones'] = []
    if 'te_operar_df_calls' not in st.session_state:
        st.session_state['te_operar_df_calls'] = None
    if 'te_operar_df_puts' not in st.session_state:
        st.session_state['te_operar_df_puts'] = None


def conectar_y_cargar():
    """
    Conecta a Schwab, obtiene precio SPX y descarga la cadena de opciones.
    Guarda todo en session_state.
    
    Returns:
        bool: True si todo fue exitoso, False en caso contrario
    """
    # 1. Conexión
    with st.spinner("🔌 Conectando con Schwab..."):
        client = connect_to_schwab()

    if client is None:
        st.error("❌ No se pudo conectar con Schwab. Verifica los secrets y el token.")
        return False

    st.session_state['te_operar_client'] = client

    # 2. Precio actual del SPX
    with st.spinner("📡 Obteniendo precio actual del SPX..."):
        precio = get_current_price_schwab(client, "SPX")

    if precio is None:
        st.error("❌ No se pudo obtener el precio del SPX. El mercado puede estar cerrado.")
        return False

    st.session_state['te_operar_precio_spx'] = precio

    # 3. Cadena de opciones — rango de 45 días para cubrir DTE típicos de Time Edge
    with st.spinner("📥 Descargando cadena de opciones del SPX (0–45 DTE)..."):
        hoy = date.today()
        desde = hoy
        hasta = hoy + timedelta(days=45)

        try:
            response = client.get_option_chain(
                "$SPX",
                from_date=desde,
                to_date=hasta
            )

            if response.status_code != 200:
                st.error(f"❌ Error al obtener la cadena de opciones: HTTP {response.status_code}")
                return False

            chain_data = response.json()
            st.session_state['te_operar_chain_raw'] = chain_data

        except Exception as e:
            st.error(f"❌ Excepción al descargar la cadena: {e}")
            return False

    # 4. Extraer fechas de expiración disponibles
    expiraciones = set()
    for map_type in ['callExpDateMap', 'putExpDateMap']:
        for date_key in chain_data.get(map_type, {}).keys():
            fecha_str = date_key.split(":")[0]  # formato "YYYY-MM-DD:DTE"
            expiraciones.add(fecha_str)

    expiraciones_sorted = sorted(list(expiraciones))
    st.session_state['te_operar_expiraciones'] = expiraciones_sorted

    return True


def parsear_cadena_para_fecha(chain_data, fecha_str, precio_spx):
    """
    Extrae calls y puts de la cadena para una fecha de expiración específica.
    
    Args:
        chain_data (dict): Respuesta cruda de get_option_chain
        fecha_str (str): Fecha en formato 'YYYY-MM-DD'
        precio_spx (float): Precio actual del SPX para calcular moneyness
    
    Returns:
        tuple: (df_calls, df_puts) — DataFrames con los strikes y greeks
    """
    def extraer_strikes(exp_map, fecha_str):
        rows = []
        for date_key, strikes_dict in exp_map.items():
            if not date_key.startswith(fecha_str):
                continue
            dte = int(date_key.split(":")[1]) if ":" in date_key else 0
            for strike_key, contratos in strikes_dict.items():
                if not contratos:
                    continue
                c = contratos[0]
                bid = c.get('bid', 0) or 0
                ask = c.get('ask', 0) or 0
                mark = c.get('mark', 0) or 0
                mid = (bid + ask) / 2 if bid > 0 and ask > 0 else mark
                rows.append({
                    'Strike': float(strike_key),
                    'DTE': dte,
                    'Bid': round(bid, 2),
                    'Ask': round(ask, 2),
                    'Mid': round(mid, 2),
                    'Mark': round(mark, 2),
                    'Delta': round(c.get('delta') or 0, 4),
                    'Gamma': round(c.get('gamma') or 0, 4),
                    'Theta': round(c.get('theta') or 0, 4),
                    'Vega': round(c.get('vega') or 0, 4),
                    'IV': round((c.get('volatility') or 0) / 100, 4),
                    'OI': c.get('openInterest') or 0,
                    'Volumen': c.get('totalVolume') or 0,
                })
        return rows

    calls_raw = extraer_strikes(chain_data.get('callExpDateMap', {}), fecha_str)
    puts_raw  = extraer_strikes(chain_data.get('putExpDateMap', {}), fecha_str)

    df_calls = pd.DataFrame(calls_raw).sort_values('Strike').reset_index(drop=True) if calls_raw else pd.DataFrame()
    df_puts  = pd.DataFrame(puts_raw).sort_values('Strike').reset_index(drop=True)  if puts_raw  else pd.DataFrame()

    return df_calls, df_puts


def highlight_atm(df, precio_spx, col='Strike'):
    """
    Aplica resaltado al strike más cercano al precio actual (ATM).
    """
    if df.empty or col not in df.columns:
        return df.style

    atm_idx = (df[col] - precio_spx).abs().idxmin()

    def resaltar(row):
        if row.name == atm_idx:
            return ['background-color: #1a3a5c; color: white; font-weight: bold'] * len(row)
        return [''] * len(row)

    return df.style.apply(resaltar, axis=1)


# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================

def main():

    st.markdown(
        "<h1><span style='font-size: 1.5em;'>📊</span> TE Operar — Cadena de Opciones SPX</h1>",
        unsafe_allow_html=True
    )
    st.markdown("""
    Conexión en tiempo real con Schwab para consultar la cadena de opciones del **SPX**.
    Seleccioná la fecha de expiración para ver strikes, precios y greeks.
    """)
    st.markdown("---")

    inicializar_session_state()

    # ==========================================================================
    # SECCIÓN 2.1 — CONEXIÓN Y CARGA DE DATOS
    # ==========================================================================
    st.header("2.1 Conexión y Carga de Datos")

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("🔌 Conectar y Cargar Cadena SPX", type="primary", use_container_width=True):
            # Limpiar datos anteriores
            for key in ['te_operar_client', 'te_operar_precio_spx', 'te_operar_chain_raw',
                        'te_operar_expiraciones', 'te_operar_df_calls', 'te_operar_df_puts']:
                st.session_state[key] = None if 'df' not in key else None
            st.session_state['te_operar_expiraciones'] = []

            exito = conectar_y_cargar()

            if exito:
                n_exp = len(st.session_state['te_operar_expiraciones'])
                st.success(f"✅ Conexión exitosa — {n_exp} fechas de expiración disponibles (0–45 DTE)")

    with col2:
        if st.button("🧹 Limpiar", use_container_width=True):
            for key in ['te_operar_client', 'te_operar_precio_spx', 'te_operar_chain_raw',
                        'te_operar_expiraciones', 'te_operar_df_calls', 'te_operar_df_puts']:
                st.session_state[key] = None
            st.session_state['te_operar_expiraciones'] = []
            st.rerun()

    # Mostrar precio actual si está disponible
    precio_spx = st.session_state.get('te_operar_precio_spx')
    if precio_spx:
        st.markdown(
            f"<div style='font-size: 1.1em; padding: 8px 14px; background-color: #1a3a5c; "
            f"color: white; border-radius: 5px; display: inline-block;'>"
            f"📈 <strong>SPX:</strong> {precio_spx:,.2f}</div>",
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("---")

    # ==========================================================================
    # SECCIÓN 2.2 — SELECCIÓN DE EXPIRACIÓN Y TABLA DE STRIKES
    # ==========================================================================
    st.header("2.2 Cadena de Opciones por Expiración")

    expiraciones = st.session_state.get('te_operar_expiraciones', [])
    chain_data   = st.session_state.get('te_operar_chain_raw')

    if not expiraciones or chain_data is None:
        st.info("ℹ️ Cargá la cadena primero usando el botón de arriba.")
        return

    # Calcular DTE para cada expiración y mostrarlo en el selector
    hoy = date.today()
    opciones_selector = []
    for f in expiraciones:
        try:
            dt = date.fromisoformat(f)
            dte = (dt - hoy).days
            opciones_selector.append(f"{f}  ({dte} DTE)")
        except Exception:
            opciones_selector.append(f)

    seleccion = st.selectbox(
        "📅 Seleccioná la fecha de expiración:",
        options=opciones_selector,
        index=0,
        key="te_operar_selector_fecha"
    )

    # Extraer solo la fecha del label
    fecha_seleccionada = seleccion.split(" ")[0]

    # Parsear la cadena para esa fecha
    df_calls, df_puts = parsear_cadena_para_fecha(chain_data, fecha_seleccionada, precio_spx)

    st.session_state['te_operar_df_calls'] = df_calls
    st.session_state['te_operar_df_puts']  = df_puts

    if df_calls.empty and df_puts.empty:
        st.warning(f"⚠️ No se encontraron contratos para la fecha {fecha_seleccionada}.")
        return

    st.markdown(f"**Mostrando strikes para:** `{fecha_seleccionada}` — SPX @ {precio_spx:,.2f}")
    st.markdown("El strike resaltado en azul es el **ATM** más cercano al precio actual.")

    # Tabs para Calls y Puts
    tab_calls, tab_puts = st.tabs(["📈 Calls", "📉 Puts"])

    columnas_mostrar = ['Strike', 'DTE', 'Bid', 'Ask', 'Mid', 'Delta', 'Gamma', 'Theta', 'Vega', 'IV', 'OI', 'Volumen']

    with tab_calls:
        if df_calls.empty:
            st.info("No hay calls disponibles para esta fecha.")
        else:
            st.dataframe(
                highlight_atm(df_calls[columnas_mostrar], precio_spx),
                hide_index=True,
                use_container_width=True
            )
            st.caption(f"Total strikes disponibles (Calls): {len(df_calls)}")

    with tab_puts:
        if df_puts.empty:
            st.info("No hay puts disponibles para esta fecha.")
        else:
            st.dataframe(
                highlight_atm(df_puts[columnas_mostrar], precio_spx),
                hide_index=True,
                use_container_width=True
            )
            st.caption(f"Total strikes disponibles (Puts): {len(df_puts)}")

    st.markdown("---")


# ==============================================================================
# PUNTO DE ENTRADA PROTEGIDO
# ==============================================================================

if __name__ == "__main__":

    if check_password():
        main()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder a la aplicación.")
