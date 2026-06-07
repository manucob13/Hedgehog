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
    defaults = {
        'te_operar_client':        None,
        'te_operar_precio_spx':    None,
        'te_operar_expiraciones':  [],
        'te_operar_fecha_cargada': None,
        'te_operar_df_puts':       None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def limpiar_todo():
    keys = ['te_operar_client', 'te_operar_precio_spx', 'te_operar_expiraciones',
            'te_operar_fecha_cargada', 'te_operar_df_puts']
    for k in keys:
        st.session_state[k] = [] if k == 'te_operar_expiraciones' else None


# ------------------------------------------------------------------------------
# PASO 1 — Conectar y escanear solo VIERNES con DTE 60–120
# ------------------------------------------------------------------------------

def conectar_y_obtener_expiraciones():
    """
    Conecta a Schwab, obtiene precio SPX y escanea únicamente los VIERNES
    dentro del rango 60–120 DTE para evitar el 502 de la API.
    """
    with st.spinner("🔌 Conectando con Schwab..."):
        client = connect_to_schwab()
    if client is None:
        st.error("❌ No se pudo conectar con Schwab. Verificá los secrets y el token.")
        return False
    st.session_state['te_operar_client'] = client

    with st.spinner("📡 Obteniendo precio actual del SPX..."):
        precio = get_current_price_schwab(client, "SPX")
    if precio is None:
        st.error("❌ No se pudo obtener el precio del SPX.")
        return False
    st.session_state['te_operar_precio_spx'] = precio

    with st.spinner("📅 Escaneando todas las expiraciones entre 60 y 120 DTE..."):
        hoy = date.today()
        expiraciones_encontradas = set()

        for delta in range(60, 121):
            target = hoy + timedelta(days=delta)
            # Saltear fines de semana
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
                    for map_type in ["callExpDateMap", "putExpDateMap"]:
                        for date_key in data.get(map_type, {}).keys():
                            fecha_str = date_key.split(":")[0]
                            try:
                                dte_real = (date.fromisoformat(fecha_str) - hoy).days
                                if 60 <= dte_real <= 120:
                                    expiraciones_encontradas.add(fecha_str)
                            except Exception:
                                pass
            except Exception:
                continue

    expiraciones_sorted = sorted(list(expiraciones_encontradas))
    st.session_state['te_operar_expiraciones'] = expiraciones_sorted
    return True


# ------------------------------------------------------------------------------
# PASO 2 — Bajar cadena solo para la fecha seleccionada (±1 día)
# ------------------------------------------------------------------------------

def cargar_cadena_para_fecha(client, fecha_str):
    try:
        target    = date.fromisoformat(fecha_str)
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
# Parser — solo PUTS, ±5 strikes del ATM
# ------------------------------------------------------------------------------

def parsear_puts_atm(chain_data, fecha_str, precio_spx, n_strikes=5):
    """
    Extrae puts de la cadena para una fecha, limitado a ±5 strikes del ATM.

    Args:
        chain_data (dict): Respuesta cruda de get_option_chain
        fecha_str  (str):  Fecha en formato 'YYYY-MM-DD'
        precio_spx (float): Precio actual del SPX
        n_strikes  (int):  Número de strikes hacia arriba y hacia abajo del ATM

    Returns:
        pd.DataFrame: Puts filtrados y ordenados por strike
    """
    rows = []
    put_map = chain_data.get('putExpDateMap', {})

    for date_key, strikes_dict in put_map.items():
        if not date_key.startswith(fecha_str):
            continue
        dte = int(date_key.split(":")[1]) if ":" in date_key else 0
        for strike_key, contratos in strikes_dict.items():
            if not contratos:
                continue
            c    = contratos[0]
            bid  = c.get('bid',  0) or 0
            ask  = c.get('ask',  0) or 0
            mark = c.get('mark', 0) or 0
            mid  = (bid + ask) / 2 if bid > 0 and ask > 0 else mark
            rows.append({
                'Strike':  float(strike_key),
                'DTE':     dte,
                'Bid':     round(bid,  2),
                'Ask':     round(ask,  2),
                'Mid':     round(mid,  2),
                'Mark':    round(mark, 2),
                'Delta':   round(c.get('delta')       or 0, 4),
                'Gamma':   round(c.get('gamma')       or 0, 4),
                'Theta':   round(c.get('theta')       or 0, 4),
                'Vega':    round(c.get('vega')        or 0, 4),
                'IV':      round((c.get('volatility') or 0) / 100, 4),
                'OI':      c.get('openInterest')  or 0,
                'Volumen': c.get('totalVolume')   or 0,
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values('Strike').reset_index(drop=True)

    # Encontrar índice del strike ATM más cercano
    atm_idx = (df['Strike'] - precio_spx).abs().idxmin()

    # Filtrar ±n_strikes alrededor del ATM
    idx_min = max(0, atm_idx - n_strikes)
    idx_max = min(len(df) - 1, atm_idx + n_strikes)
    df_filtrado = df.iloc[idx_min:idx_max + 1].reset_index(drop=True)

    return df_filtrado


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
    Cadena de **puts** del SPX en tiempo real via Schwab.
    Expiraciones: **solo viernes**, entre **60 y 120 DTE**.
    Strikes: **±5 alrededor del ATM**.
    """)
    st.markdown("---")

    inicializar_session_state()

    # ==========================================================================
    # 2.1 — CONEXIÓN Y ESCANEO DE EXPIRACIONES
    # ==========================================================================
    st.header("2.1 Conexión y Fechas Disponibles")

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("🔌 Conectar y Escanear Expiraciones", type="primary", use_container_width=True):
            limpiar_todo()
            exito = conectar_y_obtener_expiraciones()
            if exito:
                n = len(st.session_state['te_operar_expiraciones'])
                st.success(f"✅ Conectado — {n} expiraciones encontradas entre 60 y 120 DTE")

    with col2:
        if st.button("🧹 Limpiar", use_container_width=True):
            limpiar_todo()
            st.rerun()

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
    # 2.2 — SELECCIÓN DE FECHA Y CARGA DE CADENA
    # ==========================================================================
    st.header("2.2 Puts SPX — ±5 Strikes del ATM")

    expiraciones = st.session_state.get('te_operar_expiraciones', [])
    client       = st.session_state.get('te_operar_client')

    if not expiraciones or client is None:
        st.info("ℹ️ Primero conectate y escaneá las expiraciones.")
        return

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
        st.markdown("<br>", unsafe_allow_html=True)
        cargar_btn = st.button("📥 Cargar Puts", type="primary", use_container_width=True)

    if cargar_btn:
        with st.spinner(f"📥 Bajando puts para {fecha_seleccionada}..."):
            chain_data = cargar_cadena_para_fecha(client, fecha_seleccionada)

        if chain_data:
            df_puts = parsear_puts_atm(chain_data, fecha_seleccionada, precio_spx, n_strikes=5)
            st.session_state['te_operar_df_puts']       = df_puts
            st.session_state['te_operar_fecha_cargada'] = fecha_seleccionada

            if df_puts.empty:
                st.warning("⚠️ No se encontraron puts para esta fecha.")
            else:
                st.success(f"✅ {len(df_puts)} puts cargados (±5 strikes del ATM)")

    # Mostrar tabla
    df_puts       = st.session_state.get('te_operar_df_puts')
    fecha_cargada = st.session_state.get('te_operar_fecha_cargada')

    if df_puts is None:
        st.info("ℹ️ Seleccioná una fecha y presioná **Cargar Puts**.")
        return

    if df_puts.empty:
        st.warning(f"⚠️ Sin datos para {fecha_cargada}.")
        return

    st.markdown(f"**Cadena activa:** `{fecha_cargada}` — SPX @ **{precio_spx:,.2f}**")
    st.caption("Strike en azul = ATM más cercano al precio actual.")

    columnas = ['Strike', 'DTE', 'Bid', 'Ask', 'Mid', 'Delta', 'Gamma', 'Theta', 'Vega', 'IV', 'OI', 'Volumen']
    cols_ok   = [c for c in columnas if c in df_puts.columns]

    st.dataframe(
        highlight_atm(df_puts[cols_ok], precio_spx),
        hide_index=True,
        use_container_width=True
    )
    st.caption(f"Mostrando {len(df_puts)} strikes (5 por encima y 5 por debajo del ATM)")

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
