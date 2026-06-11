import streamlit as st
import pandas as pd
from datetime import date, timedelta, datetime
from utils.utils import check_password
from utils.utils_schwab import (
    connect_to_schwab,
    get_current_price_schwab,
)
from utils.risk_profile import seccion_risk_profile
from utils.github_storage import cargar_calendars_csv, guardar_calendars_csv

# ==============================================================================
# CONFIGURACIÓN POR SUBYACENTE
# ==============================================================================

SUBYACENTES = {
    "SPX": {
        "symbol":         "SPX",          # para get_current_price_schwab
        "chain_symbol":   "$SPX",         # para get_option_chain
        "display":        "SPX (S&P 500 Index)",
        "atm_multiplo":   5,              # ATM se redondea al múltiplo más cercano
        "dte_min":        60,
        "dte_max":        120,
        "csv_filename":   "calendars_TE.csv",   # nombre del archivo en GitHub
        "color_header":   "#1a3a5c",
    },
    "SPY": {
        "symbol":         "SPY",
        "chain_symbol":   "SPY",
        "display":        "SPY (S&P 500 ETF)",
        "atm_multiplo":   1,              # SPY cotiza ~$550, múltiplo de $1
        "dte_min":        30,
        "dte_max":        90,
        "csv_filename":   "calendars_TE_SPY.csv",
        "color_header":   "#1a4a2a",
    },
}

def get_cfg() -> dict:
    """Devuelve la config del subyacente actualmente seleccionado."""
    return SUBYACENTES[st.session_state.get("te_subyacente", "SPX")]


# ==============================================================================
# HELPERS — SESSION STATE
# ==============================================================================

def inicializar_session_state():
    defaults = {
        "te_subyacente":             "SPX",
        "te_operar_client":          None,
        "te_operar_precio":          None,
        "te_operar_expiraciones":    [],
        "te_short_fecha_cargada":    None,
        "te_short_df_puts":          None,
        "te_short_df_calls":         None,
        "te_long_fecha_cargada":     None,
        "te_long_df_puts":           None,
        "te_long_df_calls":          None,
        "te_order_preview":          False,
        "te_order_df":               None,
        "te_calendar_registros":     [],
        "te_rp_mids_refrescados":    {},
        "te_csv_cargado":            False,
        "te_modo":                   None,   # 'trading' | 'visualizar'
        "te_viz_client":             None,
        "te_viz_precio":             None,
        # Para detectar cambio de subyacente y limpiar estado
        "_te_ultimo_subyacente":     None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def limpiar_todo():
    keys = [
        "te_operar_client", "te_operar_precio", "te_operar_expiraciones",
        "te_short_fecha_cargada", "te_short_df_puts", "te_short_df_calls",
        "te_long_fecha_cargada",  "te_long_df_puts",  "te_long_df_calls",
        "te_order_preview", "te_order_df",
        "te_viz_client", "te_viz_precio",
        "te_rp_mids_refrescados",
    ]
    for k in keys:
        if k == "te_operar_expiraciones":
            st.session_state[k] = []
        elif k == "te_order_preview":
            st.session_state[k] = False
        elif k == "te_rp_mids_refrescados":
            st.session_state[k] = {}
        else:
            st.session_state[k] = None


def limpiar_si_cambio_subyacente():
    """Limpia cadenas y conexión si el usuario cambió de subyacente."""
    actual   = st.session_state.get("te_subyacente", "SPX")
    anterior = st.session_state.get("_te_ultimo_subyacente")
    if anterior is not None and anterior != actual:
        limpiar_todo()
        st.session_state["te_csv_cargado"] = False
        st.session_state["te_calendar_registros"] = []
    st.session_state["_te_ultimo_subyacente"] = actual


# ==============================================================================
# PASO 1 — Conectar y escanear expiraciones según DTE del subyacente
# ==============================================================================

def conectar_y_obtener_expiraciones():
    cfg = get_cfg()
    with st.spinner("🔌 Conectando con Schwab..."):
        client = connect_to_schwab()
    if client is None:
        st.error("❌ No se pudo conectar con Schwab. Verificá los secrets y el token.")
        return False
    st.session_state["te_operar_client"] = client

    with st.spinner(f"📡 Obteniendo precio actual de {cfg['symbol']}..."):
        precio = get_current_price_schwab(client, cfg["symbol"])
    if precio is None:
        st.error(f"❌ No se pudo obtener el precio de {cfg['symbol']}.")
        return False
    st.session_state["te_operar_precio"] = precio

    dte_min = cfg["dte_min"]
    dte_max = cfg["dte_max"]
    with st.spinner(f"📅 Escaneando expiraciones entre {dte_min} y {dte_max} DTE..."):
        hoy = date.today()
        expiraciones_encontradas = set()
        for delta in range(dte_min, dte_max + 1):
            target = hoy + timedelta(days=delta)
            if target.weekday() >= 5:
                continue
            try:
                resp = client.get_option_chain(
                    cfg["chain_symbol"],
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
                                if dte_min <= dte_real <= dte_max:
                                    expiraciones_encontradas.add(fecha_str)
                            except Exception:
                                pass
            except Exception:
                continue

    st.session_state["te_operar_expiraciones"] = sorted(list(expiraciones_encontradas))
    return True


def conectar_solo_precio():
    """Conexión mínima para modo visualización: solo cliente + precio."""
    cfg = get_cfg()
    with st.spinner("🔌 Conectando con Schwab..."):
        client = connect_to_schwab()
    if client is None:
        st.error("❌ No se pudo conectar con Schwab.")
        return False
    st.session_state["te_viz_client"] = client

    with st.spinner(f"📡 Obteniendo precio de {cfg['symbol']}..."):
        precio = get_current_price_schwab(client, cfg["symbol"])
    if precio is None:
        st.error(f"❌ No se pudo obtener el precio de {cfg['symbol']}.")
        return False
    st.session_state["te_viz_precio"] = precio
    return True


# ==============================================================================
# PASO 2 — Bajar cadena para una fecha (±1 día)
# ==============================================================================

def cargar_cadena_para_fecha(client, fecha_str):
    cfg = get_cfg()
    try:
        target = date.fromisoformat(fecha_str)
        resp = client.get_option_chain(
            cfg["chain_symbol"],
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
# PARSER — Puts y Calls, ATM ±1 strike
# ==============================================================================

def limpiar_greek(valor):
    if valor is None:
        return None
    try:
        v = float(valor)
        return None if v <= -999 or v >= 999 else round(v, 4)
    except Exception:
        return None


def parsear_cadena_atm(chain_data, fecha_str, precio, n_strikes=1, strike_atm_fijo=None):
    cfg = get_cfg()
    multiplo = cfg["atm_multiplo"]

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
                bid  = c.get("bid",  0) or 0
                ask  = c.get("ask",  0) or 0
                mark = c.get("mark", 0) or 0
                mid  = (bid + ask) / 2 if bid > 0 and ask > 0 else mark
                rows.append({
                    "Strike": float(strike_key),
                    "DTE":    dte,
                    "Bid":    round(bid,  2),
                    "Ask":    round(ask,  2),
                    "Mid":    round(mid,  2),
                    "Delta":  limpiar_greek(c.get("delta")),
                    "Theta":  limpiar_greek(c.get("theta")),
                    "Vega":   limpiar_greek(c.get("vega")),
                    "IV":     limpiar_greek(c.get("volatility")),
                })
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).sort_values("Strike").reset_index(drop=True)
        centro = strike_atm_fijo if strike_atm_fijo is not None else round(precio / multiplo) * multiplo
        distancias = (df["Strike"] - centro).abs()
        atm_idx = int(distancias.idxmin())
        idx_min = max(0, atm_idx - n_strikes)
        idx_max = min(len(df) - 1, atm_idx + n_strikes)
        return df.iloc[idx_min:idx_max + 1].reset_index(drop=True)

    df_puts  = extraer(chain_data.get("putExpDateMap",  {}))
    df_calls = extraer(chain_data.get("callExpDateMap", {}))
    return df_puts, df_calls


def highlight_atm(df, precio):
    if df.empty or "Strike" not in df.columns:
        return df.style
    atm_idx = (df["Strike"] - precio).abs().idxmin()
    def resaltar(row):
        if row.name == atm_idx:
            return ["background-color: #1a3a5c; color: white; font-weight: bold"] * len(row)
        return [""] * len(row)
    return df.style.apply(resaltar, axis=1)


def mostrar_tabla_cadena(df_puts, df_calls, precio, label):
    if (df_puts is None or df_puts.empty) and (df_calls is None or df_calls.empty):
        return
    tab_p, tab_c = st.tabs(["📉 Puts", "📈 Calls"])
    cols = ["Strike", "DTE", "Bid", "Ask", "Mid", "Delta", "Theta", "Vega", "IV"]
    with tab_p:
        if df_puts is not None and not df_puts.empty:
            cols_ok = [c for c in cols if c in df_puts.columns]
            st.dataframe(highlight_atm(df_puts[cols_ok], precio),
                         hide_index=True, use_container_width=True)
        else:
            st.info("Sin puts disponibles.")
    with tab_c:
        if df_calls is not None and not df_calls.empty:
            cols_ok = [c for c in cols if c in df_calls.columns]
            st.dataframe(highlight_atm(df_calls[cols_ok], precio),
                         hide_index=True, use_container_width=True)
        else:
            st.info("Sin calls disponibles.")


def bloque_cadena(pata, key_fecha, key_df_puts, key_df_calls,
                  expiraciones, client, precio, color_header,
                  dte_referencia=None, strike_atm_fijo=None):
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
                    chain, fecha_sel, precio,
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

    if f_carg and f_carg != fecha_sel:
        st.session_state[key_df_puts]  = None
        st.session_state[key_df_calls] = None
        st.session_state[key_fecha]    = None
        df_p, df_c, f_carg = None, None, None

    if f_carg and ((df_p is not None and not df_p.empty) or
                   (df_c is not None and not df_c.empty)):
        st.markdown(
            f"<span style='background:{color_header}; color:white; padding:3px 10px; "
            f"border-radius:4px; font-size:0.85em;'>Cadena {pata}: {f_carg}</span>",
            unsafe_allow_html=True
        )
        mostrar_tabla_cadena(df_p, df_c, precio, pata)

    return df_p, df_c, f_carg


# ==============================================================================
# Refresca mids de un registro consultando Schwab
# ==============================================================================

def refrescar_mids_desde_schwab(client, registro, precio_live):
    try:
        K           = float(registro.get("Strike", 0))
        short_fecha = registro.get("Short Fecha", "")
        long_fecha  = registro.get("Long Fecha", "")
        option_type = str(registro.get("Short Tipo", "PUT")).upper()
    except Exception as e:
        st.error(f"❌ Error leyendo registro: {e}")
        return None

    resultado = {}

    with st.spinner(f"📡 Refrescando SHORT {short_fecha} · Strike {K:,.0f}..."):
        chain_short = cargar_cadena_para_fecha(client, short_fecha)
    if chain_short is None:
        return None

    df_s_puts, df_s_calls = parsear_cadena_atm(
        chain_short, short_fecha, precio_live, n_strikes=3, strike_atm_fijo=K,
    )
    df_ref_s = df_s_puts if option_type == "PUT" else df_s_calls
    mid_s = None
    if df_ref_s is not None and not df_ref_s.empty:
        fila = df_ref_s[df_ref_s["Strike"] == K]
        if not fila.empty:
            mid_s = round(float(fila.iloc[0]["Mid"]), 2)
            resultado["bid_short"] = float(fila.iloc[0]["Bid"])
            resultado["ask_short"] = float(fila.iloc[0]["Ask"])
    if mid_s is None:
        st.warning(f"⚠️ No se encontró el strike {K:,.0f} en cadena SHORT para {short_fecha}.")
        return None
    resultado["mid_short"] = mid_s

    with st.spinner(f"📡 Refrescando LONG {long_fecha} · Strike {K:,.0f}..."):
        chain_long = cargar_cadena_para_fecha(client, long_fecha)
    if chain_long is None:
        return None

    df_l_puts, df_l_calls = parsear_cadena_atm(
        chain_long, long_fecha, precio_live, n_strikes=3, strike_atm_fijo=K,
    )
    df_ref_l = df_l_puts if option_type == "PUT" else df_l_calls
    mid_l = None
    if df_ref_l is not None and not df_ref_l.empty:
        fila = df_ref_l[df_ref_l["Strike"] == K]
        if not fila.empty:
            mid_l = round(float(fila.iloc[0]["Mid"]), 2)
            resultado["bid_long"] = float(fila.iloc[0]["Bid"])
            resultado["ask_long"] = float(fila.iloc[0]["Ask"])
    if mid_l is None:
        st.warning(f"⚠️ No se encontró el strike {K:,.0f} en cadena LONG para {long_fecha}.")
        return None

    resultado["mid_long"]          = mid_l
    resultado["spx_al_refrescar"]  = precio_live
    resultado["timestamp"]         = datetime.now().strftime("%H:%M:%S")
    return resultado


# ==============================================================================
# BLOQUE 2.4 + 2.5 — compartido entre modo trading y modo visualización
# ==============================================================================

def bloque_registro_y_risk_profile(precio, client, registros):
    cfg    = get_cfg()
    symbol = cfg["symbol"]

    # =========================================================================
    # 2.4 — REGISTRO DE CALENDARS
    # =========================================================================
    st.header("2.4 Registro de Calendars")

    col_gh1, col_gh2 = st.columns([1, 3])
    with col_gh1:
        if st.button("📂 Recargar desde GitHub", use_container_width=True, key="btn_recargar_gh"):
            with st.spinner("📂 Cargando desde GitHub..."):
                registros_gh = cargar_calendars_csv(cfg["csv_filename"])
            if registros_gh:
                st.session_state["te_calendar_registros"] = registros_gh
                st.success(f"✅ {len(registros_gh)} posiciones cargadas desde GitHub.")
                st.rerun()
            else:
                st.info("ℹ️ No hay posiciones guardadas en GitHub todavía.")
    with col_gh2:
        st.caption(f"Los registros se guardan en `utils/data/{cfg['csv_filename']}` del repo.")

    if not registros:
        st.info("ℹ️ Todavía no registraste ningún calendar.")
    else:
        df_reg = pd.DataFrame(registros)

        def highlight_last(row):
            if row.name == len(df_reg) - 1:
                return ["background-color: #1a3a5c; color: white"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df_reg.style.apply(highlight_last, axis=1),
            hide_index=True,
            use_container_width=True
        )
        st.caption(f"Total registros: {len(registros)}")

        col_dl, col_clear = st.columns([2, 1])
        with col_dl:
            csv = df_reg.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Exportar CSV",
                data=csv,
                file_name=f"calendars_TE_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        with col_clear:
            if st.button("🗑️ Limpiar Registro", use_container_width=True):
                st.session_state["te_calendar_registros"] = []
                st.session_state["te_csv_cargado"] = False
                with st.spinner("🗑️ Limpiando en GitHub..."):
                    guardar_calendars_csv([], cfg["csv_filename"])
                st.rerun()

    st.markdown("---")

    # =========================================================================
    # 2.5 — RISK PROFILE
    # =========================================================================
    st.header("2.5 Risk Profile — Calendar Spread")

    if not registros:
        st.info("ℹ️ Registrá un calendar en el punto 2.4 para ver el Risk Profile.")
        return

    # Selector de registro
    if len(registros) == 1:
        idx_reg = 0
    else:
        opciones_reg = [
            f"#{i+1} · Strike {r.get('Strike','')} · "
            f"{r.get('Short Fecha','')} / {r.get('Long Fecha','')} · "
            f"{r.get('Timestamp','')}"
            for i, r in enumerate(registros)
        ]
        sel = st.selectbox("Seleccioná el calendar:", options=opciones_reg, key="rp_selector_25")
        idx_reg = opciones_reg.index(sel)

    registro_sel = registros[idx_reg]

    # ------------------------------------------------------------------
    # Bloque de conexión Schwab
    # ------------------------------------------------------------------
    client_activo = client or st.session_state.get("te_viz_client")
    precio_activo = precio or st.session_state.get("te_viz_precio")

    if client_activo is None:
        st.markdown(
            "<div style='background:#1a1a0d; border:1px solid #ffc107; border-radius:6px; "
            "padding:10px 16px; margin-bottom:12px;'>"
            f"<span style='color:#ffc107; font-weight:bold;'>⚡ Conectá a Schwab para actualizar el gráfico con precios reales</span><br>"
            "<span style='color:#aaa; font-size:0.85em;'>Sin conexión el Risk Profile usa los mids guardados al momento del registro.</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        col_conn, _ = st.columns([1, 3])
        with col_conn:
            if st.button("🔌 Conectar a Schwab", type="primary",
                         use_container_width=True, key="btn_viz_conectar"):
                if conectar_solo_precio():
                    client_activo = st.session_state["te_viz_client"]
                    precio_activo = st.session_state["te_viz_precio"]
                    st.success(f"✅ Conectado · {symbol}: {precio_activo:,.2f}")
                    st.rerun()
    else:
        col_spx_info, col_spx_btn = st.columns([3, 1])
        with col_spx_info:
            if precio_activo:
                st.markdown(
                    f"<div style='background:#0d1a0d; border:1px solid #1a5c1a; border-radius:5px; "
                    f"padding:6px 14px; display:inline-block; font-size:0.9em;'>"
                    f"📈 <b style='color:#ffc107;'>{symbol}: {precio_activo:,.2f}</b>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        with col_spx_btn:
            if st.button("🔄 Actualizar precio", use_container_width=True, key="btn_actualizar_precio"):
                with st.spinner(f"📡 Actualizando precio {symbol}..."):
                    precio_nuevo = get_current_price_schwab(client_activo, symbol)
                if precio_nuevo:
                    if st.session_state.get("te_modo") == "trading":
                        st.session_state["te_operar_precio"] = precio_nuevo
                    else:
                        st.session_state["te_viz_precio"] = precio_nuevo
                    precio_activo = precio_nuevo
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Botón Refrescar Mids
    # ------------------------------------------------------------------
    col_btn_r, col_info_r = st.columns([1, 3])

    with col_btn_r:
        btn_disabled = client_activo is None
        if st.button(
            "🔄 Refrescar Mids desde Schwab",
            type="primary",
            use_container_width=True,
            key="btn_refrescar_mids",
            disabled=btn_disabled,
            help="Conectate a Schwab primero para refrescar los mids" if btn_disabled else None,
        ):
            with st.spinner(f"📡 Actualizando precio {symbol}..."):
                precio_fresco = get_current_price_schwab(client_activo, symbol)
            if precio_fresco:
                if st.session_state.get("te_modo") == "trading":
                    st.session_state["te_operar_precio"] = precio_fresco
                else:
                    st.session_state["te_viz_precio"] = precio_fresco
                precio_activo = precio_fresco

            mids = refrescar_mids_desde_schwab(client_activo, registro_sel, precio_activo)
            if mids:
                st.session_state["te_rp_mids_refrescados"][idx_reg] = mids
                st.success(
                    f"✅ Refrescado a las {mids['timestamp']} · "
                    f"{symbol}: {precio_activo:,.2f} · "
                    f"Mid Short: {mids['mid_short']} · "
                    f"Mid Long: {mids['mid_long']}"
                )

    mids_frescos = st.session_state["te_rp_mids_refrescados"].get(idx_reg)

    with col_info_r:
        if mids_frescos:
            ts_r  = mids_frescos.get("timestamp", "")
            spx_r = mids_frescos.get("spx_al_refrescar", precio_activo)
            ms    = mids_frescos.get("mid_short", "-")
            ml    = mids_frescos.get("mid_long",  "-")
            bs    = mids_frescos.get("bid_short", "-")
            as_   = mids_frescos.get("ask_short", "-")
            bl    = mids_frescos.get("bid_long",  "-")
            al    = mids_frescos.get("ask_long",  "-")
            st.markdown(
                f"<div style='background:#0d1a0d; border:1px solid #1a5c1a; border-radius:6px; "
                f"padding:8px 14px; font-size:0.85em; color:#aaa;'>"
                f"⏱ Último refresco: <b style='color:white'>{ts_r}</b> · "
                f"{symbol}: <b style='color:#ffc107'>{spx_r:,.2f}</b><br>"
                f"SHORT — Bid: {bs} | Ask: {as_} | "
                f"<b style='color:#ef9a9a'>Mid: {ms}</b> &nbsp;·&nbsp; "
                f"LONG — Bid: {bl} | Ask: {al} | "
                f"<b style='color:#a5d6a7'>Mid: {ml}</b>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            if client_activo:
                st.caption("Sin refresco aún — presioná 'Refrescar Mids' para actualizar con precios reales.")
            else:
                st.caption("Sin refresco aún — el Risk Profile usa los mids del momento de registro.")

    st.markdown("<br>", unsafe_allow_html=True)

    seccion_risk_profile(
        precio_spx_live  = precio_activo,
        mids_refrescados = mids_frescos,
        idx_registro     = idx_reg,
    )


# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================

def main():

    inicializar_session_state()

    # =========================================================================
    # SELECTOR DE SUBYACENTE — al tope de todo
    # =========================================================================
    col_titulo, col_sub = st.columns([3, 1])
    with col_titulo:
        st.markdown(
            "<h1><span style='font-size: 1.5em;'>📊</span> TE Operar — Calendar Put Spread</h1>",
            unsafe_allow_html=True
        )
    with col_sub:
        subyacente_sel = st.selectbox(
            "🎯 Subyacente",
            options=list(SUBYACENTES.keys()),
            index=list(SUBYACENTES.keys()).index(st.session_state.get("te_subyacente", "SPX")),
            key="te_subyacente_selector",
            help="Cambiá el subyacente — limpia cadenas y registros de sesión",
        )
        if subyacente_sel != st.session_state.get("te_subyacente"):
            st.session_state["te_subyacente"] = subyacente_sel

    # Limpiar estado si cambió el subyacente
    limpiar_si_cambio_subyacente()

    cfg    = get_cfg()
    symbol = cfg["symbol"]

    st.markdown(
        f"<span style='color:#aaa; font-size:0.9em;'>"
        f"{cfg['display']} · Cadena en tiempo real · "
        f"{cfg['dte_min']}–{cfg['dte_max']} DTE · ATM ±1 strike"
        f"</span>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # ── Carga automática del CSV de GitHub al arrancar (solo una vez por subyacente) ──
    if not st.session_state.get("te_csv_cargado"):
        with st.spinner(f"📂 Cargando posiciones {symbol} desde GitHub..."):
            registros_gh = cargar_calendars_csv(cfg["csv_filename"])
        if registros_gh:
            st.session_state["te_calendar_registros"] = registros_gh
            st.session_state["te_csv_cargado"] = True
            st.toast(f"✅ {len(registros_gh)} posiciones {symbol} cargadas desde GitHub", icon="📂")
        else:
            st.session_state["te_csv_cargado"] = True

    registros = st.session_state.get("te_calendar_registros", [])

    # =========================================================================
    # SELECTOR DE MODO
    # =========================================================================
    modo_opciones = {
        "📊 Solo visualizar posiciones": "visualizar",
        "⚡ Modo trading (conectar a Schwab y operar)": "trading",
    }
    modo_actual = st.session_state.get("te_modo")
    idx_default = 0 if modo_actual != "trading" else 1

    modo_label = st.radio(
        "¿Qué querés hacer?",
        options=list(modo_opciones.keys()),
        index=idx_default,
        horizontal=True,
        key="te_modo_radio",
    )
    modo = modo_opciones[modo_label]
    st.session_state["te_modo"] = modo

    st.markdown("---")

    # =========================================================================
    # MODO VISUALIZACIÓN
    # =========================================================================
    if modo == "visualizar":
        if registros:
            n = len(registros)
            st.success(f"📂 {n} posición{'es' if n > 1 else ''} {symbol} cargada{'s' if n > 1 else ''} desde GitHub.")
        else:
            st.info("ℹ️ No hay posiciones guardadas. Podés recargar desde GitHub en la sección 2.4.")

        bloque_registro_y_risk_profile(
            precio    = st.session_state.get("te_viz_precio"),
            client    = st.session_state.get("te_viz_client"),
            registros = registros,
        )
        return

    # =========================================================================
    # MODO TRADING
    # =========================================================================

    # 2.1 — CONEXIÓN
    st.header("2.1 Conexión y Fechas Disponibles")

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("🔌 Conectar y Escanear Expiraciones", type="primary", use_container_width=True):
            limpiar_todo()
            if conectar_y_obtener_expiraciones():
                n = len(st.session_state["te_operar_expiraciones"])
                st.success(f"✅ Conectado — {n} expiraciones encontradas")
    with col2:
        if st.button("🧹 Limpiar Todo", use_container_width=True):
            limpiar_todo()
            st.rerun()

    precio       = st.session_state.get("te_operar_precio")
    expiraciones = st.session_state.get("te_operar_expiraciones", [])
    client       = st.session_state.get("te_operar_client")

    if precio:
        multiplo         = cfg["atm_multiplo"]
        strike_atm_display = round(precio / multiplo) * multiplo
        st.markdown(
            f"<div style='font-size:1.1em; padding:8px 14px; background:#1a3a5c; "
            f"color:white; border-radius:5px; display:inline-block; margin-top:8px;'>"
            f"📈 <strong>{symbol}:</strong> {precio:,.2f} &nbsp;·&nbsp; "
            f"🎯 <strong>ATM:</strong> {strike_atm_display:,.0f}</div>",
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("---")

    if not expiraciones or client is None:
        st.info("ℹ️ Primero conectate y escaneá las expiraciones.")
        bloque_registro_y_risk_profile(precio=precio, client=client, registros=registros)
        return

    # 2.2 — CADENAS
    st.header("2.2 Cadenas — Short y Long")
    st.caption(f"Cada pata muestra Puts y Calls · ATM ±1 strike · Strike azul = ATM · {symbol}")

    col_s, col_l = st.columns(2)
    multiplo         = cfg["atm_multiplo"]
    strike_atm_global = round(precio / multiplo) * multiplo

    with col_s:
        st.markdown("### 📤 Short")
        df_short_puts, df_short_calls, fecha_short = bloque_cadena(
            pata            = "SHORT",
            key_fecha       = "te_short_fecha_cargada",
            key_df_puts     = "te_short_df_puts",
            key_df_calls    = "te_short_df_calls",
            expiraciones    = expiraciones,
            client          = client,
            precio          = precio,
            color_header    = "#8B0000",
            dte_referencia  = None,
            strike_atm_fijo = strike_atm_global
        )

    hoy = date.today()
    dte_short_ref   = None
    fecha_short_sel = st.session_state.get("te_sel_fecha_short", "")
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
            pata            = "LONG",
            key_fecha       = "te_long_fecha_cargada",
            key_df_puts     = "te_long_df_puts",
            key_df_calls    = "te_long_df_calls",
            expiraciones    = expiraciones,
            client          = client,
            precio          = precio,
            color_header    = "#1a5c1a",
            dte_referencia  = dte_short_ref,
            strike_atm_fijo = strike_atm_global
        )

    st.markdown("---")

    # 2.3 — CONFIGURAR ORDEN
    st.header("2.3 Configurar Orden")

    cadena_short_ok = (df_short_puts  is not None and not df_short_puts.empty)  or \
                      (df_short_calls is not None and not df_short_calls.empty)
    cadena_long_ok  = (df_long_puts   is not None and not df_long_puts.empty)   or \
                      (df_long_calls  is not None and not df_long_calls.empty)

    if not cadena_short_ok:
        st.info("ℹ️ Cargá la cadena **Short** arriba para configurar la orden.")
        bloque_registro_y_risk_profile(precio, client, registros)
        return
    if not cadena_long_ok:
        st.info("ℹ️ Cargá la cadena **Long** arriba para configurar la orden.")
        bloque_registro_y_risk_profile(precio, client, registros)
        return

    strikes_short_raw = sorted(set(
        (df_short_puts["Strike"].tolist()  if df_short_puts  is not None and not df_short_puts.empty  else []) +
        (df_short_calls["Strike"].tolist() if df_short_calls is not None and not df_short_calls.empty else [])
    ))
    strikes_long_raw = sorted(set(
        (df_long_puts["Strike"].tolist()   if df_long_puts   is not None and not df_long_puts.empty   else []) +
        (df_long_calls["Strike"].tolist()  if df_long_calls  is not None and not df_long_calls.empty  else [])
    ))
    atm_aprox = min(strikes_short_raw, key=lambda x: abs(x - precio)) if strikes_short_raw else precio

    # SPX: filtro looser (±50 pts); SPY: ±10 pts
    margen_short = 50 if multiplo >= 5 else 10
    margen_long  = 30 if multiplo >= 5 else 10
    strikes_short_all = sorted([s for s in strikes_short_raw if s >= atm_aprox - margen_short])
    strikes_long_all  = sorted([s for s in strikes_long_raw  if s >= atm_aprox - margen_long])

    col_front, col_back = st.columns(2)

    def idx_atm(strikes_list, atm):
        if not strikes_list:
            return 0
        return min(range(len(strikes_list)), key=lambda i: abs(strikes_list[i] - atm))

    atm_idx_short = idx_atm(strikes_short_all, strike_atm_global)
    atm_idx_long  = idx_atm(strikes_long_all,  strike_atm_global)

    with col_front:
        st.markdown(f"### 📤 Short — `{fecha_short}`")
        st.markdown("---")
        tipo_short   = st.selectbox("Tipo de Opción SHORT", ["PUT", "CALL"], index=0, key="orden_tipo_short")
        accion_short = st.selectbox("Acción SHORT", ["SELL", "BUY"], index=0, key="orden_accion_short")
        strike_short_label = st.selectbox(
            "Strike SHORT",
            options=[f"{s:,.0f}" for s in strikes_short_all],
            index=atm_idx_short,
            key="orden_strike_short"
        )
        strike_short_val = float(strike_short_label.replace(",", ""))

        df_ref_short  = df_short_puts if tipo_short == "PUT" else df_short_calls
        mid_ref_short = 0.0
        if df_ref_short is not None and not df_ref_short.empty:
            fila = df_ref_short[df_ref_short["Strike"] == strike_short_val]
            if not fila.empty:
                mid_ref_short = float(fila.iloc[0]["Mid"]) if fila.iloc[0]["Mid"] is not None else 0.0
                bid_s = fila.iloc[0]["Bid"]
                ask_s = fila.iloc[0]["Ask"]
                st.caption(f"Bid: {bid_s}  |  Ask: {ask_s}  |  Mid referencia: **{mid_ref_short:.2f}**")

        clave_mid_s = f"mid_short_{strike_short_val}_{tipo_short}"
        if st.session_state.get("_last_mid_short_key") != clave_mid_s:
            st.session_state["orden_mid_short"] = mid_ref_short
            st.session_state["_last_mid_short_key"] = clave_mid_s

        mid_short_input = st.number_input(
            "Mid Price SHORT", min_value=0.0, step=0.05, format="%.2f", key="orden_mid_short"
        )

        diff_s = strike_short_val - precio
        pct_s  = (diff_s / precio) * 100
        if tipo_short == "PUT":
            estado_s = "ITM 🔴" if diff_s > 0 else ("ATM 🟡" if abs(diff_s) <= 10 else "OTM 🟢")
        else:
            estado_s = "ITM 🔴" if diff_s < 0 else ("ATM 🟡" if abs(diff_s) <= 10 else "OTM 🟢")
        st.markdown(f"**{diff_s:+.0f} pts** ({pct_s:+.2f}%) · {estado_s}")

    with col_back:
        st.markdown(f"### 📥 Long — `{fecha_long}`")
        st.markdown("---")
        tipo_long   = st.selectbox("Tipo de Opción LONG", ["PUT", "CALL"], index=0, key="orden_tipo_long")
        accion_long = st.selectbox("Acción LONG", ["BUY", "SELL"], index=0, key="orden_accion_long")
        strike_long_label = st.selectbox(
            "Strike LONG",
            options=[f"{s:,.0f}" for s in strikes_long_all],
            index=atm_idx_long,
            key="orden_strike_long"
        )
        strike_long_val = float(strike_long_label.replace(",", ""))

        df_ref_long  = df_long_puts if tipo_long == "PUT" else df_long_calls
        mid_ref_long = 0.0
        if df_ref_long is not None and not df_ref_long.empty:
            fila = df_ref_long[df_ref_long["Strike"] == strike_long_val]
            if not fila.empty:
                mid_ref_long = float(fila.iloc[0]["Mid"]) if fila.iloc[0]["Mid"] is not None else 0.0
                bid_l = fila.iloc[0]["Bid"]
                ask_l = fila.iloc[0]["Ask"]
                st.caption(f"Bid: {bid_l}  |  Ask: {ask_l}  |  Mid referencia: **{mid_ref_long:.2f}**")

        mid_long_input = st.number_input(
            "Mid Price LONG (editable)",
            min_value=0.0,
            value=float(mid_ref_long) if mid_ref_long else 0.0,
            step=0.05, format="%.2f",
            key="orden_mid_long",
            help="Podés editar el mid price manualmente"
        )

        diff_l = strike_long_val - precio
        pct_l  = (diff_l / precio) * 100
        if tipo_long == "PUT":
            estado_l = "ITM 🔴" if diff_l > 0 else ("ATM 🟡" if abs(diff_l) <= 10 else "OTM 🟢")
        else:
            estado_l = "ITM 🔴" if diff_l < 0 else ("ATM 🟡" if abs(diff_l) <= 10 else "OTM 🟢")
        st.markdown(f"**{diff_l:+.0f} pts** ({pct_l:+.2f}%) · {estado_l}")

    st.markdown("<br>", unsafe_allow_html=True)
    col_qty, col_debito = st.columns([1, 2])

    with col_qty:
        cantidad = st.number_input(
            "Cantidad de contratos", min_value=1, value=1, step=1, key="orden_cantidad"
        )

    # ------------------------------------------------------------------
    # Resultado neto teórico (a partir de los mids de referencia)
    # ------------------------------------------------------------------
    debito_teorico = round(mid_long_input - mid_short_input, 2)
    multiplicador  = 100  # SPX y SPY usan x100

    # ------------------------------------------------------------------
    # Ajuste manual del resultado neto (según precio real del broker)
    # Si cambia el contexto de la orden (strikes/fechas/tipos/cantidad),
    # reseteamos el ajuste manual para no arrastrar valores viejos.
    # ------------------------------------------------------------------
    clave_ajuste = (
        f"{strike_short_val}_{tipo_short}_{fecha_short}_"
        f"{strike_long_val}_{tipo_long}_{fecha_long}_{cantidad}"
    )
    if st.session_state.get("_last_ajuste_key") != clave_ajuste:
        st.session_state["orden_resultado_neto_real"] = abs(debito_teorico)
        st.session_state["_last_ajuste_key"] = clave_ajuste

    with col_debito:
        signo_teorico = "DÉBITO" if debito_teorico > 0 else "CRÉDITO"
        color_teorico = "#ffb74d" if debito_teorico > 0 else "#ef5350"
        st.markdown(
            f"<div style='padding:12px; background:#1a1a2e; border-radius:6px; margin-top:8px;'>"
            f"<span style='font-size:0.9em; color:#aaa;'>Resultado neto teórico ({signo_teorico}, según mids)</span><br>"
            f"<span style='font-size:1.5em; font-weight:bold; color:{color_teorico};'>"
            f"${abs(debito_teorico):.2f} / contrato &nbsp;·&nbsp; ${abs(debito_teorico)*multiplicador*cantidad:.2f} total</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Resultado neto REAL (editable) — ajusta los mids equitativamente
    # ------------------------------------------------------------------
    st.markdown("##### 🎯 Resultado neto real (según ejecución en el broker)")
    st.caption(
        "Si el fill real difiere del teórico (por desplazamiento de mercado), "
        "ingresá aquí el débito/crédito real por contrato. El sistema repartirá "
        "la diferencia equitativamente entre el Mid SHORT y el Mid LONG, "
        "manteniendo el signo original de la operación (débito o crédito), "
        "para que el registro quede coherente."
    )

    col_signo_real, col_valor_real, col_resumen_real = st.columns([1, 1, 2])

    # El signo de la operación lo determina el resultado teórico (Calendar
    # típico = débito). Si el teórico fuera 0 (caso borde), default a débito.
    signo_operacion = "DÉBITO" if debito_teorico >= 0 else "CRÉDITO"
    factor_signo    = 1 if signo_operacion == "DÉBITO" else -1

    with col_signo_real:
        st.markdown(
            f"<div style='padding:8px 12px; background:#222; border-radius:6px; "
            f"text-align:center; margin-top:6px;'>"
            f"<span style='font-size:0.85em; color:#aaa;'>Signo</span><br>"
            f"<span style='font-size:1.1em; font-weight:bold;'>{signo_operacion}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_valor_real:
        resultado_neto_real = st.number_input(
            f"{signo_operacion} real / contrato",
            min_value=0.0,
            step=0.05,
            format="%.2f",
            key="orden_resultado_neto_real",
            help="Valor absoluto del débito o crédito real por contrato, según el broker."
        )

    # Diferencia con signo (teórico vs real), y reparto equitativo
    debito_real_signed = factor_signo * resultado_neto_real
    diferencia         = round(debito_real_signed - debito_teorico, 2)
    ajuste_por_pata    = round(diferencia / 2, 2)

    # debito = mid_long - mid_short  =>  para mover el débito en +diferencia,
    # repartimos: mid_long sube la mitad, mid_short baja la mitad
    mid_long_ajustado  = round(mid_long_input  + ajuste_por_pata, 2)
    mid_short_ajustado = round(mid_short_input - ajuste_por_pata, 2)

    # Evitar mids negativos: si el ajuste lleva alguno por debajo de 0,
    # recortamos a 0 y dejamos el resto en la otra pata (mejor esfuerzo).
    aviso_clip = False
    if mid_long_ajustado < 0:
        falta = -mid_long_ajustado
        mid_long_ajustado = 0.0
        mid_short_ajustado = round(mid_short_ajustado - falta, 2)
        aviso_clip = True
    if mid_short_ajustado < 0:
        falta = -mid_short_ajustado
        mid_short_ajustado = 0.0
        mid_long_ajustado = round(mid_long_ajustado + falta, 2)
        aviso_clip = True

    debito_final = round(mid_long_ajustado - mid_short_ajustado, 2)

    with col_resumen_real:
        color_final = "#ffb74d" if debito_final > 0 else "#ef5350"
        signo_final = "DÉBITO" if debito_final > 0 else "CRÉDITO"
        st.markdown(
            f"<div style='padding:8px 12px; background:#1a1a2e; border-radius:6px; margin-top:6px;'>"
            f"<span style='font-size:0.85em; color:#aaa;'>Ajuste por pata: "
            f"<b>{ajuste_por_pata:+.2f}</b> &nbsp;|&nbsp; "
            f"Mid SHORT ajustado: <b style='color:#ef9a9a'>{mid_short_ajustado:.2f}</b> "
            f"&nbsp;|&nbsp; Mid LONG ajustado: <b style='color:#a5d6a7'>{mid_long_ajustado:.2f}</b></span><br>"
            f"<span style='font-size:1.2em; font-weight:bold; color:{color_final};'>"
            f"Resultado final ({signo_final}): ${abs(debito_final):.2f} / cto &nbsp;·&nbsp; "
            f"${abs(debito_final)*multiplicador*cantidad:.2f} total</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    if aviso_clip:
        st.warning(
            "⚠️ El ajuste hizo que uno de los mids llegara a 0; se recortó y la "
            "diferencia restante se aplicó a la otra pata. Revisá los valores."
        )

    # Estos son los valores que se usarán de acá en adelante (vista previa y registro)
    mid_short_final = mid_short_ajustado
    mid_long_final  = mid_long_ajustado
    debito          = debito_final
    color_debito    = color_final
    signo           = signo_final

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("📝 Generar Vista Previa de Orden", type="primary", use_container_width=True):
        orders = [
            {
                "Pata": "SHORT", "Acción": accion_short, "Tipo": tipo_short,
                "Strike": int(strike_short_val), "Expiración": fecha_short,
                "Mid Price": mid_short_final, "Contratos": cantidad,
            },
            {
                "Pata": "LONG", "Acción": accion_long, "Tipo": tipo_long,
                "Strike": int(strike_long_val), "Expiración": fecha_long,
                "Mid Price": mid_long_final, "Contratos": cantidad,
            },
        ]
        st.session_state["te_order_df"]      = pd.DataFrame(orders)
        st.session_state["te_order_preview"] = True

    if st.session_state.get("te_order_preview") and st.session_state.get("te_order_df") is not None:
        df_order = st.session_state["te_order_df"]

        st.markdown("---")
        st.markdown("### 📋 Vista Previa de Orden")
        st.dataframe(df_order, hide_index=True, use_container_width=True)

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            st.metric(f"{symbol} al momento", f"{precio:,.2f}")
        with col_r2:
            st.metric("Strike", f"{int(strike_short_val):,}")
        with col_r3:
            st.metric(signo, f"${abs(debito):.2f} / cto")
        with col_r4:
            st.metric("Total", f"${abs(debito)*multiplicador*cantidad:.2f}")

        st.markdown("<br>", unsafe_allow_html=True)

        col_reg, col_cancel = st.columns([2, 1])
        with col_reg:
            if st.button("✅ Registrar este Calendar", type="primary", use_container_width=True):
                registro = {
                    "Timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "Subyacente":   symbol,
                    f"{symbol} al abrir": f"{precio:,.2f}",
                    "Strike":       int(strike_short_val),
                    "Short Tipo":   tipo_short,
                    "Short Acción": accion_short,
                    "Short Fecha":  fecha_short,
                    "Short Mid":    mid_short_final,
                    "Long Tipo":    tipo_long,
                    "Long Acción":  accion_long,
                    "Long Fecha":   fecha_long,
                    "Long Mid":     mid_long_final,
                    f"{signo}":     abs(debito),
                    "Total ($)":    round(abs(debito) * multiplicador * cantidad, 2),
                    "Contratos":    cantidad,
                }
                st.session_state["te_calendar_registros"].append(registro)
                st.session_state["te_order_preview"] = False
                with st.spinner("💾 Guardando en GitHub..."):
                    ok = guardar_calendars_csv(
                        st.session_state["te_calendar_registros"],
                        cfg["csv_filename"]
                    )
                if ok:
                    st.success(f"✅ Calendar {symbol} registrado y guardado en GitHub.")
                else:
                    st.warning("✅ Calendar registrado en sesión pero no se pudo guardar en GitHub.")
                st.rerun()

        with col_cancel:
            if st.button("✖ Cancelar", use_container_width=True):
                st.session_state["te_order_preview"] = False
                st.rerun()

    st.markdown("---")

    bloque_registro_y_risk_profile(precio, client, registros)


# ==============================================================================
# PUNTO DE ENTRADA PROTEGIDO
# ==============================================================================

if __name__ == "__main__":
    if check_password():
        main()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder a la aplicación.")
