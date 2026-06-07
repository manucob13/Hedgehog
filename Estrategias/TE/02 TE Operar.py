# 02 TE Operar.py
import streamlit as st
import pandas as pd
from datetime import date, timedelta
from utils.utils import check_password
from utils.utils_schwab import (
    connect_to_schwab,
    get_current_price_schwab,
)

# --- CONFIGURACIÓN DE PÁGINA ---
# st.set_page_config(page_title="📊 TE Operar", layout="wide")

# ==============================================================================
# HELPERS
# ==============================================================================

def inicializar_session_state():
    """Inicializa las variables de session_state necesarias para esta página."""
    defaults = {
        'te_operar_client':       None,
        'te_operar_precio_spx':   None,
        'te_operar_expiraciones': [],   # lista de str "YYYY-MM-DD"
        'te_operar_fecha_cargada': None, # última fecha cuya cadena se bajó
        'te_operar_df_calls':     None,
        'te_operar_df_puts':      None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def limpiar_todo():
    """Resetea todos los datos de esta página."""
    keys = ['te_operar_client', 'te_operar_precio_spx', 'te_operar_expiraciones',
            'te_operar_fecha_cargada', 'te_operar_df_calls', 'te_operar_df_puts']
    for k in keys:
        st.session_state[k] = [] if k == 'te_operar_expiraciones' else None


# ------------------------------------------------------------------------------
# PASO 1 — Conectar y obtener lista de expiraciones (llamada liviana: ±1 día)
# ------------------------------------------------------------------------------

def conectar_y_obtener_expiraciones():
    """
    Conecta a Schwab, obtiene el precio del SPX y descarga UNA sola fecha
    (hoy ±1 día) para extraer el listado de expiraciones disponibles sin
    provocar el 502 'Body buffer overflow' del SPX.

    La lista completa de expiraciones se arma haciendo llamadas de 1 día
    avanzando de a semana hasta cubrir 60 DTE.

    Returns:
        bool: True si todo fue exitoso
    """
    # 1. Conexión
    with st.spinner("🔌 Conectando con Schwab..."):
        client = connect_to_schwab()
    if client is None:
        st.error("❌ No se pudo conectar con Schwab. Verificá los secrets y el token.")
        return False
    st.session_state['te_operar_client'] = client

    # 2. Precio SPX
    with st.spinner("📡 Obteniendo precio actual del SPX..."):
        precio = get_current_price_schwab(client, "SPX")
    if precio is None:
        st.error("❌ No se pudo obtener el precio del SPX.")
        return False
    st.session_state['te_operar_precio_spx'] = precio

    # 3. Escanear expiraciones de a 1 día cada vez (evita 502)
    #    Barremos hoy + cada lunes/miércoles/viernes hasta 60 DTE
    with st.spinner("📅 Escaneando fechas de expiración disponibles (0–60 DTE)..."):
        hoy = date.today()
        expiraciones_encontradas = set()

        # Probamos cada día hábil del calendario hasta 60 días adelante
        # Para el SPX el rango ±1 día funciona sin overflow
        dias_a_probar = list(range(0, 61))  # 0 a 60 DTE

        for delta in dias_a_probar:
            target = hoy + timedelta(days=delta)
            # Saltear fines de semana (el SPX no vence sábado/domingo normalmente)
            if target.weekday() >= 5:
                continue
            try:
                resp = client.get_option_chain(
                    "$SPX",
                    from_date=target - timedelta(days=1),
                    to_date=target + timedelta(days=1)
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for map_type in ['callExpDateMap', 'putExpDateMap']:
                        for date_key in data.get(map_type, {}).keys():
                            fecha_str = date_key.split(":")[0]
                            expiraciones_encontradas.add(fecha_str)
            except Exception:
                continue  # Si una fecha falla, seguimos con la siguiente

    expiraciones_sorted = sorted(list(expiraciones_encontradas))
    st.session_state['te_operar_expiraciones'] = expiraciones_sorted
    return True


# ------------------------------------------------------------------------------
# PASO 2 — Bajar la cadena completa SOLO para la fecha seleccionada (±1 día)
# ------------------------------------------------------------------------------

def cargar_cadena_para_fecha(client, fecha_str):
    """
    Descarga la cadena de opciones del SPX solo para una fecha específica
    usando el rango ±1 día que ya funciona en utils_schwab.

    Args:
        client: Cliente Schwab autenticado
        fecha_str (str): Fecha en formato 'YYYY-MM-DD'

    Returns:
        dict | None: Respuesta JSON de la cadena, None si hay error
    """
    try:
        target = date.fromisoformat(fecha_str)
        from_date = target - timedelta(days=1)
        to_date   = target + timedelta(days=1)

        resp = client.get_option_chain(
            "$SPX",
            from_date=from_date,
            to_date=to_date
        )

        if resp.status_code != 200:
            st.error(f"❌ Error HTTP {resp.status_code} al bajar cadena para {fecha_str}")
            return None

        return resp.json()

    except Exception as e:
        st.error(f"❌ Excepción al descargar cadena: {e}")
        return None


# ------------------------------------------------------------------------------
# Parser y helpers de visualización
# ------------------------------------------------------------------------------

def parsear_cadena_para_fecha(chain_data, fecha_str):
    """
    Extrae calls y puts de la cadena para una fecha específica.

    Returns:
        tuple: (df_calls, df_puts)
    """
    def extraer_strikes(exp_map):
        rows = []
        for date_key, strikes_dict in exp_map.items():
            if not date_key.startswith(fecha_str):
                continue
            dte = int(date_key.split(":")[1]) if ":" in date_key else 0
            for strike_key, contratos in strikes_dict.items():
                if not contratos:
                    continue
                c = contratos[0]
                bid  = c.get('bid', 0)  or 0
                ask  = c.get('ask', 0)  or 0
                mark = c.get('mark', 0) or 0
                mid  = (bid + ask) / 2 if bid > 0 and ask > 0 else mark
                rows.append({
                    'Strike':  float(strike_key),
                    'DTE':     dte,
                    'Bid':     round(bid,  2),
                    'Ask':     round(ask,  2),
                    'Mid':     round(mid,  2),
                    'Mark':    round(mark, 2),
                    'Delta':   round(c.get('delta')      or 0, 4),
                    'Gamma':   round(c.get('gamma')      or 0, 4),
                    'Theta':   round(c.get('theta')      or 0, 4),
                    'Vega':    round(c.get('vega')       or 0, 4),
                    'IV':      round((c.get('volatility') or 0) / 100, 4),
                    'OI':      c.get('openInterest')  or 0,
                    'Volumen': c.get('totalVolume')   or 0,
                })
        return rows

    calls_raw = extraer_strikes(chain_data.get('callExpDateMap', {}))
    puts_raw  = extraer_strikes(chain_data.get('putExpDateMap',  {}))

    df_calls = pd.DataFrame(calls_raw).sort_values('Strike').reset_index(drop=True) if calls_raw else pd.DataFrame()
    df_puts  = pd.DataFrame(puts_raw ).sort_values('Strike').reset_index(drop=True) if puts_raw  else pd.DataFrame()

    return df_calls, df_puts


def highlight_atm(df, precio_spx, col='Strike'):
    """Resalta el strike ATM más cercano al precio actual."""
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
    La carga se hace en **dos pasos** para evitar el límite de buffer de la API.
    """)
    st.markdown("---")

    inicializar_session_state()

    # ==========================================================================
    # SECCIÓN 2.1 — CONEXIÓN Y LISTA DE EXPIRACIONES
    # ==========================================================================
    st.header("2.1 Conexión y Fechas Disponibles")

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("🔌 Conectar y Escanear Expiraciones", type="primary", use_container_width=True):
            limpiar_todo()
            exito = conectar_y_obtener_expiraciones()
            if exito:
                n = len(st.session_state['te_operar_expiraciones'])
                st.success(f"✅ Conectado — {n} fechas de expiración encontradas (0–60 DTE)")

    with col2:
        if st.button("🧹 Limpiar", use_container_width=True):
            limpiar_todo()
            st.rerun()

    # Precio SPX
    precio_spx = st.session_state.get('te_operar_precio_spx')
    if precio_spx:
        st.markdown(
            f"<div style='font-size: 1.1em; padding: 8px 14px; background-color: #1a3a5c; "
            f"color: white; border-radius: 5px; display: inline-block; margin-top: 8px;'>"
            f"📈 <strong>SPX:</strong> {precio_spx:,.2f}</div>",
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("---")

    # ==========================================================================
    # SECCIÓN 2.2 — SELECCIÓN DE FECHA Y CARGA DE CADENA
    # ==========================================================================
    st.header("2.2 Cadena de Opciones por Expiración")

    expiraciones = st.session_state.get('te_operar_expiraciones', [])
    client       = st.session_state.get('te_operar_client')

    if not expiraciones or client is None:
        st.info("ℹ️ Primero conectate y escaneá las expiraciones (botón de arriba).")
        return

    # Selector con DTE calculado
    hoy = date.today()
    opciones_selector = []
    for f in expiraciones:
        try:
            dte = (date.fromisoformat(f) - hoy).days
            opciones_selector.append(f"{f}  ({dte} DTE)")
        except Exception:
            opciones_selector.append(f)

    col_sel, col_btn = st.columns([3, 1])

    with col_sel:
        seleccion = st.selectbox(
            "📅 Seleccioná la fecha de expiración:",
            options=opciones_selector,
            index=0,
            key="te_operar_selector_fecha"
        )

    fecha_seleccionada = seleccion.split(" ")[0]

    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)  # alinear verticalmente
        cargar_btn = st.button("📥 Cargar Cadena", type="primary", use_container_width=True)

    if cargar_btn:
        with st.spinner(f"📥 Bajando cadena para {fecha_seleccionada}..."):
            chain_data = cargar_cadena_para_fecha(client, fecha_seleccionada)

        if chain_data:
            df_calls, df_puts = parsear_cadena_para_fecha(chain_data, fecha_seleccionada)
            st.session_state['te_operar_df_calls']      = df_calls
            st.session_state['te_operar_df_puts']       = df_puts
            st.session_state['te_operar_fecha_cargada'] = fecha_seleccionada

            n_calls = len(df_calls)
            n_puts  = len(df_puts)
            st.success(f"✅ Cadena cargada — {n_calls} calls | {n_puts} puts")

    # Mostrar tabla si hay datos cargados
    df_calls       = st.session_state.get('te_operar_df_calls')
    df_puts        = st.session_state.get('te_operar_df_puts')
    fecha_cargada  = st.session_state.get('te_operar_fecha_cargada')

    if df_calls is None and df_puts is None:
        st.info("ℹ️ Seleccioná una fecha y presioná **Cargar Cadena**.")
        return

    if (df_calls is not None and df_calls.empty) and (df_puts is not None and df_puts.empty):
        st.warning(f"⚠️ No se encontraron contratos para {fecha_cargada}.")
        return

    st.markdown(f"**Cadena activa:** `{fecha_cargada}` — SPX @ **{precio_spx:,.2f}**")
    st.markdown("El strike resaltado en azul es el **ATM** más cercano al precio actual.")

    columnas = ['Strike', 'DTE', 'Bid', 'Ask', 'Mid', 'Delta', 'Gamma', 'Theta', 'Vega', 'IV', 'OI', 'Volumen']

    tab_calls, tab_puts = st.tabs(["📈 Calls", "📉 Puts"])

    with tab_calls:
        if df_calls is None or df_calls.empty:
            st.info("No hay calls disponibles para esta fecha.")
        else:
            cols_disponibles = [c for c in columnas if c in df_calls.columns]
            st.dataframe(
                highlight_atm(df_calls[cols_disponibles], precio_spx),
                hide_index=True,
                use_container_width=True
            )
            st.caption(f"Total strikes (Calls): {len(df_calls)}")

    with tab_puts:
        if df_puts is None or df_puts.empty:
            st.info("No hay puts disponibles para esta fecha.")
        else:
            cols_disponibles = [c for c in columnas if c in df_puts.columns]
            st.dataframe(
                highlight_atm(df_puts[cols_disponibles], precio_spx),
                hide_index=True,
                use_container_width=True
            )
            st.caption(f"Total strikes (Puts): {len(df_puts)}")

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
