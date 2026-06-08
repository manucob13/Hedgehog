# 02 TE Operar.py
import streamlit as st
import pandas as pd
from datetime import date, timedelta, datetime
from utils.utils import check_password
from utils.utils_schwab import (
    connect_to_schwab,
    get_current_price_schwab,
)

# --- CONFIGURACIÓN DE PÁGINA ---
# st.set_page_config(page_title="📊 TE Operar", layout="wide")

# ==============================================================================
# HELPERS — SESSION STATE
# ==============================================================================

def inicializar_session_state():
    defaults = {
        'te_operar_client':          None,
        'te_operar_precio_spx':      None,
        'te_operar_expiraciones':    [],
        'te_short_fecha_cargada':    None,
        'te_short_df_puts':          None,
        'te_long_fecha_cargada':     None,
        'te_long_df_puts':           None,
        'te_order_preview':          False,
        'te_order_df':               None,
        'te_calendar_registros':     [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def limpiar_todo():
    keys = ['te_operar_client', 'te_operar_precio_spx', 'te_operar_expiraciones',
            'te_short_fecha_cargada', 'te_short_df_puts',
            'te_long_fecha_cargada',  'te_long_df_puts',
            'te_order_preview',       'te_order_df']
    for k in keys:
        if k == 'te_operar_expiraciones':
            st.session_state[k] = []
        elif k == 'te_order_preview':
            st.session_state[k] = False
        else:
            st.session_state[k] = None


# ==============================================================================
# PASO 1 — Conectar y escanear expiraciones 60–120 DTE
# ==============================================================================

def conectar_y_obtener_expiraciones():
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

    with st.spinner("📅 Escaneando expiraciones entre 60 y 120 DTE..."):
        hoy = date.today()
        expiraciones_encontradas = set()
        for delta in range(60, 121):
            target = hoy + timedelta(days=delta)
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

    st.session_state['te_operar_expiraciones'] = sorted(list(expiraciones_encontradas))
    return True


# ==============================================================================
# PASO 2 — Bajar cadena para una fecha (±1 día)
# ==============================================================================

def cargar_cadena_para_fecha(client, fecha_str):
    try:
        target = date.fromisoformat(fecha_str)
        resp = client.get_option_chain(
            "$SPX",
            from_date=target - timedelta(days=1),
            to_date=target + timedelta(days=1)
        )
        if resp.status_code != 200:
            st.error(f"❌ Error HTTP {resp.status_code} al bajar cadena para {fecha_str}")
            return None
        return resp.json()
    except Exception as e:
        st.error(f"❌ Excepción al descargar cadena: {e}")
        return None


# ==============================================================================
# PARSER — Puts y Calls, ATM ±1 strike (3 strikes total)
# ==============================================================================

def limpiar_greek(valor):
    """Convierte -999 (sin datos) a None."""
    if valor is None:
        return None
    try:
        v = float(valor)
        return None if v <= -999 or v >= 999 else round(v, 4)
    except Exception:
        return None


def parsear_cadena_atm(chain_data, fecha_str, precio_spx, n_strikes=1, strike_atm_fijo=None):
    """
    Extrae puts Y calls para una fecha, limitado a ATM ±n_strikes.
    strike_atm_fijo: si se pasa, se usa ese strike como centro en lugar de
                     calcular el más cercano (garantiza consistencia entre patas).
    Retorna (df_puts, df_calls).
    """
    def extraer(exp_map):
        rows = []
        for date_key, strikes_dict in exp_map.items():
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
                    'Strike': float(strike_key),
                    'DTE':    dte,
                    'Bid':    round(bid,  2),
                    'Ask':    round(ask,  2),
                    'Mid':    round(mid,  2),
                    'Delta':  limpiar_greek(c.get('delta')),
                    'Theta':  limpiar_greek(c.get('theta')),
                    'Vega':   limpiar_greek(c.get('vega')),
                    'IV':     round((c.get('volatility') or 0) / 100, 4),
                })
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).sort_values('Strike').reset_index(drop=True)

        # Centro ATM: usar strike fijo si se provee, sino el más cercano al precio
        centro = strike_atm_fijo if strike_atm_fijo is not None else precio_spx
        distancias = (df['Strike'] - centro).abs()
        atm_idx = int(distancias.idxmin())
        idx_min = max(0, atm_idx - n_strikes)
        idx_max = min(len(df) - 1, atm_idx + n_strikes)
        return df.iloc[idx_min:idx_max + 1].reset_index(drop=True)

    df_puts  = extraer(chain_data.get('putExpDateMap',  {}))
    df_calls = extraer(chain_data.get('callExpDateMap', {}))
    return df_puts, df_calls


def highlight_atm(df, precio_spx):
    if df.empty or 'Strike' not in df.columns:
        return df.style
    atm_idx = (df['Strike'] - precio_spx).abs().idxmin()
    def resaltar(row):
        if row.name == atm_idx:
            return ['background-color: #1a3a5c; color: white; font-weight: bold'] * len(row)
        return [''] * len(row)
    return df.style.apply(resaltar, axis=1)


def mostrar_tabla_cadena(df_puts, df_calls, precio_spx, label):
    """Muestra puts y calls en tabs para una pata."""
    if (df_puts is None or df_puts.empty) and (df_calls is None or df_calls.empty):
        return
    tab_p, tab_c = st.tabs(["📉 Puts", "📈 Calls"])
    cols = ['Strike', 'DTE', 'Bid', 'Ask', 'Mid', 'Delta', 'Theta', 'Vega', 'IV']
    with tab_p:
        if df_puts is not None and not df_puts.empty:
            cols_ok = [c for c in cols if c in df_puts.columns]
            st.dataframe(highlight_atm(df_puts[cols_ok], precio_spx),
                         hide_index=True, use_container_width=True)
        else:
            st.info("Sin puts disponibles.")
    with tab_c:
        if df_calls is not None and not df_calls.empty:
            cols_ok = [c for c in cols if c in df_calls.columns]
            st.dataframe(highlight_atm(df_calls[cols_ok], precio_spx),
                         hide_index=True, use_container_width=True)
        else:
            st.info("Sin calls disponibles.")


def bloque_cadena(pata, key_fecha, key_df_puts, key_df_calls,
                  expiraciones, client, precio_spx, color_header,
                  dte_referencia=None, strike_atm_fijo=None):
    """
    Renderiza selector + botón + tabla (puts y calls) para una pata.
    dte_referencia : DTE del SHORT → el LONG muestra solo el incremento.
    strike_atm_fijo: strike centro fijo → garantiza mismos strikes en ambas patas.
    Retorna (df_puts, df_calls, fecha_cargada).
    """
    hoy = date.today()

    def label(f):
        try:
            dte = (date.fromisoformat(f) - hoy).days
            if dte_referencia is not None:
                diff = dte - dte_referencia
                return f"{f}  (+{diff} DTE)"
            return f"{f}  ({dte} DTE)"
        except Exception:
            return f

    opciones = [label(f) for f in expiraciones]

    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        seleccion = st.selectbox(
            f"📅 Fecha ({pata}):",
            options=opciones,
            index=0,
            key=f"te_sel_fecha_{pata.lower()}"
        )
    fecha_sel = seleccion.split(" ")[0]

    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(f"📥 Cargar {pata}", type="primary",
                     use_container_width=True, key=f"btn_cargar_{pata.lower()}"):
            with st.spinner(f"Bajando cadena {pata} para {fecha_sel}..."):
                chain = cargar_cadena_para_fecha(client, fecha_sel)
            if chain:
                df_p, df_c = parsear_cadena_atm(
                    chain, fecha_sel, precio_spx,
                    n_strikes=1,
                    strike_atm_fijo=strike_atm_fijo
                )
                st.session_state[key_df_puts]  = df_p
                st.session_state[key_df_calls] = df_c
                st.session_state[key_fecha]    = fecha_sel
                if (df_p is None or df_p.empty) and (df_c is None or df_c.empty):
                    st.warning(f"⚠️ Sin datos para {fecha_sel}.")

    df_p   = st.session_state.get(key_df_puts)
    df_c   = st.session_state.get(key_df_calls)
    f_carg = st.session_state.get(key_fecha)

    if f_carg and ((df_p is not None and not df_p.empty) or
                   (df_c is not None and not df_c.empty)):
        st.markdown(
            f"<span style='background:{color_header}; color:white; padding:3px 10px; "
            f"border-radius:4px; font-size:0.85em;'>Cadena {pata}: {f_carg}</span>",
            unsafe_allow_html=True
        )
        mostrar_tabla_cadena(df_p, df_c, precio_spx, pata)

    return df_p, df_c, f_carg


# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================

def main():

    st.markdown(
        "<h1><span style='font-size: 1.5em;'>📊</span> TE Operar — Calendar Put Spread SPX</h1>",
        unsafe_allow_html=True
    )
    st.markdown("Cadena SPX en tiempo real · 60–120 DTE · ATM ±1 strike")
    st.markdown("---")

    inicializar_session_state()

    # =========================================================================
    # 2.1 — CONEXIÓN
    # =========================================================================
    st.header("2.1 Conexión y Fechas Disponibles")

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("🔌 Conectar y Escanear Expiraciones", type="primary", use_container_width=True):
            limpiar_todo()
            if conectar_y_obtener_expiraciones():
                n = len(st.session_state['te_operar_expiraciones'])
                st.success(f"✅ Conectado — {n} expiraciones encontradas")
    with col2:
        if st.button("🧹 Limpiar Todo", use_container_width=True):
            limpiar_todo()
            st.rerun()

    precio_spx   = st.session_state.get('te_operar_precio_spx')
    expiraciones = st.session_state.get('te_operar_expiraciones', [])
    client       = st.session_state.get('te_operar_client')

    if precio_spx:
        st.markdown(
            f"<div style='font-size:1.1em; padding:8px 14px; background:#1a3a5c; "
            f"color:white; border-radius:5px; display:inline-block; margin-top:8px;'>"
            f"📈 <strong>SPX:</strong> {precio_spx:,.2f}</div>",
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("---")

    if not expiraciones or client is None:
        st.info("ℹ️ Primero conectate y escaneá las expiraciones.")
        return

    # =========================================================================
    # 2.2 — CADENAS: SHORT (near) y LONG (far)
    # =========================================================================
    st.header("2.2 Cadenas — Short y Long")
    st.caption("Cada pata muestra Puts y Calls · ATM ±1 strike · Strike azul = ATM")

    col_s, col_l = st.columns(2)

    # ATM = múltiplo de 5 más cercano al precio actual del SPX
    # Ej: SPX=7383 → 7385, SPX=7377 → 7375
    strike_atm_global = round(precio_spx / 5) * 5

    with col_s:
        st.markdown("### 📤 Short")
        df_short_puts, df_short_calls, fecha_short = bloque_cadena(
            pata           = 'SHORT',
            key_fecha      = 'te_short_fecha_cargada',
            key_df_puts    = 'te_short_df_puts',
            key_df_calls   = 'te_short_df_calls',
            expiraciones   = expiraciones,
            client         = client,
            precio_spx     = precio_spx,
            color_header   = '#8B0000',
            dte_referencia = None,
            strike_atm_fijo= strike_atm_global
        )

    # DTE del SHORT seleccionado (desde el selector en tiempo real)
    hoy = date.today()
    dte_short_ref = None
    fecha_short_sel = st.session_state.get('te_sel_fecha_short', '')
    if fecha_short_sel:
        f_str = fecha_short_sel.split(" ")[0]
        try:
            dte_short_ref = (date.fromisoformat(f_str) - hoy).days
        except Exception:
            pass
    if dte_short_ref is None and fecha_short:
        try:
            dte_short_ref = (date.fromisoformat(fecha_short) - hoy).days
        except Exception:
            pass

    with col_l:
        st.markdown("### 📥 Long")
        df_long_puts, df_long_calls, fecha_long = bloque_cadena(
            pata           = 'LONG',
            key_fecha      = 'te_long_fecha_cargada',
            key_df_puts    = 'te_long_df_puts',
            key_df_calls   = 'te_long_df_calls',
            expiraciones   = expiraciones,
            client         = client,
            precio_spx     = precio_spx,
            color_header   = '#1a5c1a',
            dte_referencia = dte_short_ref,
            strike_atm_fijo= strike_atm_global   # mismo ATM para ambas patas
        )

    st.markdown("---")

    # =========================================================================
    # 2.3 — CONFIGURAR ORDEN
    # =========================================================================
    st.header("2.3 Configurar Orden")

    cadena_short_ok = (df_short_puts is not None and not df_short_puts.empty) or \
                      (df_short_calls is not None and not df_short_calls.empty)
    cadena_long_ok  = (df_long_puts  is not None and not df_long_puts.empty)  or \
                      (df_long_calls is not None and not df_long_calls.empty)

    if not cadena_short_ok:
        st.info("ℹ️ Cargá la cadena **Short** arriba para configurar la orden.")
        return
    if not cadena_long_ok:
        st.info("ℹ️ Cargá la cadena **Long** arriba para configurar la orden.")
        return

    # Strikes disponibles por pata — mínimo ATM-50 para SHORT, ATM-30 para LONG
    strikes_short_raw = sorted(set(
        (df_short_puts['Strike'].tolist()  if df_short_puts  is not None and not df_short_puts.empty  else []) +
        (df_short_calls['Strike'].tolist() if df_short_calls is not None and not df_short_calls.empty else [])
    ))
    strikes_long_raw = sorted(set(
        (df_long_puts['Strike'].tolist()   if df_long_puts   is not None and not df_long_puts.empty   else []) +
        (df_long_calls['Strike'].tolist()  if df_long_calls  is not None and not df_long_calls.empty  else [])
    ))
    atm_aprox = min(strikes_short_raw, key=lambda x: abs(x - precio_spx)) if strikes_short_raw else precio_spx

    strikes_short_all = sorted([s for s in strikes_short_raw if s >= atm_aprox - 50])
    strikes_long_all  = sorted([s for s in strikes_long_raw  if s >= atm_aprox - 30])

    col_front, col_back = st.columns(2)

    # ------------------------------------------------------------------
    # PATA SHORT
    # ------------------------------------------------------------------
    with col_front:
        st.markdown(f"### 📤 Short — `{fecha_short}`")
        st.markdown("---")

        tipo_short = st.selectbox(
            "Tipo de Opción SHORT",
            ["PUT", "CALL"],
            index=0,
            key='orden_tipo_short'
        )
        accion_short = st.selectbox(
            "Acción SHORT",
            ["SELL", "BUY"],
            index=0,
            key='orden_accion_short'
        )
        strike_short_label = st.selectbox(
            "Strike SHORT",
            options=[f"{s:,.0f}" for s in strikes_short_all],
            key='orden_strike_short'
        )
        strike_short_val = float(strike_short_label.replace(",", ""))

        # Buscar mid de referencia en la cadena correspondiente
        df_ref_short = df_short_puts if tipo_short == "PUT" else df_short_calls
        mid_ref_short = None
        if df_ref_short is not None and not df_ref_short.empty:
            fila = df_ref_short[df_ref_short['Strike'] == strike_short_val]
            if not fila.empty:
                mid_ref_short = fila.iloc[0]['Mid']
                bid_s = fila.iloc[0]['Bid']
                ask_s = fila.iloc[0]['Ask']
                st.caption(f"Bid: {bid_s}  |  Ask: {ask_s}  |  Mid referencia: **{mid_ref_short:.2f}**")

        mid_short_input = st.number_input(
            "Mid Price SHORT (editable)",
            min_value=0.0,
            value=float(mid_ref_short) if mid_ref_short else 0.0,
            step=0.05,
            format="%.2f",
            key='orden_mid_short',
            help="Podés editar el mid price manualmente"
        )

        # Análisis ITM/ATM/OTM
        diff_s = strike_short_val - precio_spx
        pct_s  = (diff_s / precio_spx) * 100
        if tipo_short == "PUT":
            estado_s = "ITM 🔴" if diff_s > 0 else ("ATM 🟡" if abs(diff_s) <= 10 else "OTM 🟢")
        else:
            estado_s = "ITM 🔴" if diff_s < 0 else ("ATM 🟡" if abs(diff_s) <= 10 else "OTM 🟢")
        st.markdown(f"**{diff_s:+.0f} pts** ({pct_s:+.2f}%) · {estado_s}")

    # ------------------------------------------------------------------
    # PATA LONG
    # ------------------------------------------------------------------
    with col_back:
        st.markdown(f"### 📥 Long — `{fecha_long}`")
        st.markdown("---")

        tipo_long = st.selectbox(
            "Tipo de Opción LONG",
            ["PUT", "CALL"],
            index=0,
            key='orden_tipo_long'
        )
        accion_long = st.selectbox(
            "Acción LONG",
            ["BUY", "SELL"],
            index=0,
            key='orden_accion_long'
        )
        strike_long_label = st.selectbox(
            "Strike LONG",
            options=[f"{s:,.0f}" for s in strikes_long_all],
            key='orden_strike_long'
        )
        strike_long_val = float(strike_long_label.replace(",", ""))

        df_ref_long = df_long_puts if tipo_long == "PUT" else df_long_calls
        mid_ref_long = None
        if df_ref_long is not None and not df_ref_long.empty:
            fila = df_ref_long[df_ref_long['Strike'] == strike_long_val]
            if not fila.empty:
                mid_ref_long = fila.iloc[0]['Mid']
                bid_l = fila.iloc[0]['Bid']
                ask_l = fila.iloc[0]['Ask']
                st.caption(f"Bid: {bid_l}  |  Ask: {ask_l}  |  Mid referencia: **{mid_ref_long:.2f}**")

        mid_long_input = st.number_input(
            "Mid Price LONG (editable)",
            min_value=0.0,
            value=float(mid_ref_long) if mid_ref_long else 0.0,
            step=0.05,
            format="%.2f",
            key='orden_mid_long',
            help="Podés editar el mid price manualmente"
        )

        diff_l = strike_long_val - precio_spx
        pct_l  = (diff_l / precio_spx) * 100
        if tipo_long == "PUT":
            estado_l = "ITM 🔴" if diff_l > 0 else ("ATM 🟡" if abs(diff_l) <= 10 else "OTM 🟢")
        else:
            estado_l = "ITM 🔴" if diff_l < 0 else ("ATM 🟡" if abs(diff_l) <= 10 else "OTM 🟢")
        st.markdown(f"**{diff_l:+.0f} pts** ({pct_l:+.2f}%) · {estado_l}")

    # ------------------------------------------------------------------
    # CANTIDAD Y DÉBITO
    # ------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    col_qty, col_debito = st.columns([1, 2])

    with col_qty:
        cantidad = st.number_input(
            "Cantidad de contratos",
            min_value=1, value=1, step=1,
            key='orden_cantidad'
        )

    with col_debito:
        debito = round(mid_long_input - mid_short_input, 2)
        color_debito = "#ffb74d" if debito > 0 else "#ef5350"
        signo = "DÉBITO" if debito > 0 else "CRÉDITO"
        st.markdown(
            f"<div style='padding:12px; background:#1a1a2e; border-radius:6px; margin-top:8px;'>"
            f"<span style='font-size:0.9em; color:#aaa;'>Resultado neto ({signo})</span><br>"
            f"<span style='font-size:1.5em; font-weight:bold; color:{color_debito};'>"
            f"${abs(debito):.2f} / contrato &nbsp;·&nbsp; ${abs(debito)*100*cantidad:.2f} total</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # BOTÓN VISTA PREVIA
    # ------------------------------------------------------------------
    if st.button("📝 Generar Vista Previa de Orden", type="primary", use_container_width=True):
        orders = [
            {
                'Pata':       'SHORT',
                'Acción':     accion_short,
                'Tipo':       tipo_short,
                'Strike':     int(strike_short_val),
                'Expiración': fecha_short,
                'Mid Price':  mid_short_input,
                'Contratos':  cantidad,
            },
            {
                'Pata':       'LONG',
                'Acción':     accion_long,
                'Tipo':       tipo_long,
                'Strike':     int(strike_long_val),
                'Expiración': fecha_long,
                'Mid Price':  mid_long_input,
                'Contratos':  cantidad,
            },
        ]
        st.session_state['te_order_df']      = pd.DataFrame(orders)
        st.session_state['te_order_preview'] = True

    # ------------------------------------------------------------------
    # VISTA PREVIA
    # ------------------------------------------------------------------
    if st.session_state.get('te_order_preview') and st.session_state.get('te_order_df') is not None:
        df_order = st.session_state['te_order_df']

        st.markdown("---")
        st.markdown("### 📋 Vista Previa de Orden")
        st.dataframe(df_order, hide_index=True, use_container_width=True)

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            st.metric("SPX al momento", f"{precio_spx:,.2f}")
        with col_r2:
            st.metric("Strike", f"{int(strike_short_val):,}")
        with col_r3:
            st.metric(signo, f"${abs(debito):.2f} / cto")
        with col_r4:
            st.metric("Total", f"${abs(debito)*100*cantidad:.2f}")

        st.markdown("<br>", unsafe_allow_html=True)

        col_reg, col_cancel = st.columns([2, 1])
        with col_reg:
            if st.button("✅ Registrar este Calendar", type="primary", use_container_width=True):
                registro = {
                    'Timestamp':      datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'SPX al abrir':   f"{precio_spx:,.2f}",
                    'Strike':         int(strike_short_val),
                    'Short Tipo':     tipo_short,
                    'Short Acción':   accion_short,
                    'Short Fecha':    fecha_short,
                    'Short Mid':      mid_short_input,
                    'Long Tipo':      tipo_long,
                    'Long Acción':    accion_long,
                    'Long Fecha':     fecha_long,
                    'Long Mid':       mid_long_input,
                    f'{signo}':       abs(debito),
                    'Total ($)':      round(abs(debito) * 100 * cantidad, 2),
                    'Contratos':      cantidad,
                }
                st.session_state['te_calendar_registros'].append(registro)
                st.session_state['te_order_preview'] = False
                st.success("✅ Calendar registrado.")
                st.rerun()

        with col_cancel:
            if st.button("✖ Cancelar", use_container_width=True):
                st.session_state['te_order_preview'] = False
                st.rerun()

    st.markdown("---")

    # =========================================================================
    # 2.4 — REGISTRO DE CALENDARS
    # =========================================================================
    st.header("2.4 Registro de Calendars")

    registros = st.session_state.get('te_calendar_registros', [])

    if not registros:
        st.info("ℹ️ Todavía no registraste ningún calendar.")
    else:
        df_reg = pd.DataFrame(registros)

        def highlight_last(row):
            if row.name == len(df_reg) - 1:
                return ['background-color: #1a3a5c; color: white'] * len(row)
            return [''] * len(row)

        st.dataframe(
            df_reg.style.apply(highlight_last, axis=1),
            hide_index=True,
            use_container_width=True
        )
        st.caption(f"Total registros: {len(registros)}")

        col_dl, col_clear = st.columns([2, 1])
        with col_dl:
            csv = df_reg.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Exportar CSV",
                data=csv,
                file_name=f"calendars_TE_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        with col_clear:
            if st.button("🗑️ Limpiar Registro", use_container_width=True):
                st.session_state['te_calendar_registros'] = []
                st.rerun()

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
