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
        # Pata SHORT (near)
        'te_short_fecha_cargada':    None,
        'te_short_df_puts':          None,
        # Pata LONG (far)
        'te_long_fecha_cargada':     None,
        'te_long_df_puts':           None,
        # Registro de calendars
        'te_calendar_registros':     [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def limpiar_todo():
    keys = ['te_operar_client', 'te_operar_precio_spx', 'te_operar_expiraciones',
            'te_short_fecha_cargada', 'te_short_df_puts',
            'te_long_fecha_cargada',  'te_long_df_puts']
    for k in keys:
        st.session_state[k] = [] if k == 'te_operar_expiraciones' else None


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
# PARSER — Puts ±5 strikes del ATM
# ==============================================================================

def parsear_puts_atm(chain_data, fecha_str, precio_spx, n_strikes=5):
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
    atm_idx = (df['Strike'] - precio_spx).abs().idxmin()
    idx_min = max(0, atm_idx - n_strikes)
    idx_max = min(len(df) - 1, atm_idx + n_strikes)
    return df.iloc[idx_min:idx_max + 1].reset_index(drop=True)


def highlight_atm(df, precio_spx, col='Strike'):
    if df.empty or col not in df.columns:
        return df.style
    atm_idx = (df[col] - precio_spx).abs().idxmin()
    def resaltar(row):
        if row.name == atm_idx:
            return ['background-color: #1a3a5c; color: white; font-weight: bold'] * len(row)
        return [''] * len(row)
    return df.style.apply(resaltar, axis=1)


def bloque_cadena(pata, key_fecha, key_df, expiraciones, client, precio_spx, color_header):
    """
    Renderiza el bloque completo de selección + carga + tabla para una pata.

    Args:
        pata        (str): 'SHORT' o 'LONG'
        key_fecha   (str): clave en session_state para la fecha cargada
        key_df      (str): clave en session_state para el DataFrame de puts
        expiraciones(list): lista de fechas disponibles
        client      : cliente Schwab
        precio_spx  (float): precio actual SPX
        color_header(str): color del badge de header
    """
    hoy = date.today()

    def label(f):
        try:
            return f"{f}  ({(date.fromisoformat(f) - hoy).days} DTE)"
        except Exception:
            return f

    opciones = [label(f) for f in expiraciones]

    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        seleccion = st.selectbox(
            f"📅 Fecha expiración ({pata}):",
            options=opciones,
            index=0,
            key=f"te_sel_fecha_{pata.lower()}"
        )
    fecha_sel = seleccion.split(" ")[0]

    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(f"📥 Cargar {pata}", type="primary", use_container_width=True, key=f"btn_cargar_{pata.lower()}"):
            with st.spinner(f"Bajando puts {pata} para {fecha_sel}..."):
                chain = cargar_cadena_para_fecha(client, fecha_sel)
            if chain:
                df = parsear_puts_atm(chain, fecha_sel, precio_spx)
                st.session_state[key_df]    = df
                st.session_state[key_fecha] = fecha_sel
                if df.empty:
                    st.warning(f"⚠️ Sin puts para {fecha_sel}.")
                else:
                    st.success(f"✅ {len(df)} puts {pata} cargados")

    df      = st.session_state.get(key_df)
    f_carg  = st.session_state.get(key_fecha)

    if df is not None and not df.empty:
        st.markdown(
            f"<span style='background:{color_header}; color:white; padding:3px 10px; "
            f"border-radius:4px; font-size:0.85em;'>Cadena {pata}: {f_carg}</span>",
            unsafe_allow_html=True
        )
        columnas = ['Strike', 'DTE', 'Bid', 'Ask', 'Mid', 'Delta', 'Gamma', 'Theta', 'Vega', 'IV']
        cols_ok  = [c for c in columnas if c in df.columns]
        st.dataframe(
            highlight_atm(df[cols_ok], precio_spx),
            hide_index=True,
            use_container_width=True
        )

    return df, f_carg


# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================

def main():

    st.markdown(
        "<h1><span style='font-size: 1.5em;'>📊</span> TE Operar — Cadena de Opciones SPX</h1>",
        unsafe_allow_html=True
    )
    st.markdown("Cadena de **puts** del SPX · 60–120 DTE · ±5 strikes del ATM")
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
    # 2.2 — CADENAS: SHORT (near) y LONG (far) en dos columnas
    # =========================================================================
    st.header("2.2 Cadenas de Puts — Short y Long")
    st.markdown("Cargá cada pata de forma independiente. El **Short** es la expiración más corta (near), el **Long** la más larga (far).")

    col_s, col_l = st.columns(2)

    with col_s:
        st.markdown("### 📤 Short Put *(near — vendés)*")
        df_short, fecha_short = bloque_cadena(
            pata         = 'SHORT',
            key_fecha    = 'te_short_fecha_cargada',
            key_df       = 'te_short_df_puts',
            expiraciones = expiraciones,
            client       = client,
            precio_spx   = precio_spx,
            color_header = '#8B0000'
        )

    with col_l:
        st.markdown("### 📥 Long Put *(far — comprás)*")
        df_long, fecha_long = bloque_cadena(
            pata         = 'LONG',
            key_fecha    = 'te_long_fecha_cargada',
            key_df       = 'te_long_df_puts',
            expiraciones = expiraciones,
            client       = client,
            precio_spx   = precio_spx,
            color_header = '#1a5c1a'
        )

    st.markdown("---")

    # =========================================================================
    # 2.3 — ARMAR CALENDAR PUT SPREAD
    # =========================================================================
    st.header("2.3 Armar Calendar Put Spread")

    if df_short is None or df_short.empty:
        st.info("ℹ️ Cargá la pata **Short** arriba para armar el calendar.")
        return
    if df_long is None or df_long.empty:
        st.info("ℹ️ Cargá la pata **Long** arriba para armar el calendar.")
        return

    # Strikes comunes a ambas cadenas para el selectbox
    strikes_short = sorted(df_short['Strike'].unique().tolist())
    strikes_long  = sorted(df_long['Strike'].unique().tolist())
    strikes_comunes = sorted(set(strikes_short) & set(strikes_long))

    # Si no hay strikes comunes, permitir selección independiente
    usar_comunes = len(strikes_comunes) > 0

    col_short, col_long = st.columns(2)

    with col_short:
        st.markdown("#### 📤 Short Put (vendés)")
        opciones_s = [f"{s:,.0f}" for s in (strikes_comunes if usar_comunes else strikes_short)]
        short_label = st.selectbox("Strike Short:", options=opciones_s, key="cal_short_strike")
        short_strike = float(short_label.replace(",", ""))
        fila_short = df_short[df_short['Strike'] == short_strike]
        short_row  = fila_short.iloc[0] if not fila_short.empty else None
        if short_row is not None:
            st.metric("Mid Short", f"${short_row['Mid']:.2f}")
            st.caption(f"Bid: {short_row['Bid']}  |  Ask: {short_row['Ask']}  |  DTE: {short_row['DTE']}  |  Δ: {short_row['Delta']}")

    with col_long:
        st.markdown("#### 📥 Long Put (comprás)")
        opciones_l = [f"{s:,.0f}" for s in (strikes_comunes if usar_comunes else strikes_long)]
        long_label = st.selectbox("Strike Long:", options=opciones_l, key="cal_long_strike")
        long_strike = float(long_label.replace(",", ""))
        fila_long = df_long[df_long['Strike'] == long_strike]
        long_row  = fila_long.iloc[0] if not fila_long.empty else None
        if long_row is not None:
            st.metric("Mid Long", f"${long_row['Mid']:.2f}")
            st.caption(f"Bid: {long_row['Bid']}  |  Ask: {long_row['Ask']}  |  DTE: {long_row['DTE']}  |  Δ: {long_row['Delta']}")

    st.markdown("<br>", unsafe_allow_html=True)

    if short_row is not None and long_row is not None:
        mid_short = short_row['Mid']
        mid_long  = long_row['Mid']
        dte_short = short_row['DTE']
        dte_long  = long_row['DTE']
        debito    = round(mid_long - mid_short, 2)

        # Validaciones
        if dte_long <= dte_short:
            st.warning(f"⚠️ El Long ({dte_long} DTE) debe tener **más DTE** que el Short ({dte_short} DTE).")
        elif debito <= 0:
            st.warning(f"⚠️ Débito calculado: ${debito:.2f} — el Long debería ser más caro que el Short.")
        else:
            st.markdown(
                f"<div style='background:#0e2a0e; border:1px solid #2e7d32; border-radius:8px; padding:16px;'>"
                f"<h4 style='color:#81c784; margin:0 0 12px 0;'>📋 Calendar Put Spread — Resumen</h4>"
                f"<table style='width:100%; color:white; font-size:0.95em; border-collapse:collapse;'>"
                f"<tr><td style='padding:4px 8px;'>Strike</td><td style='padding:4px 8px;'><strong>{short_strike:,.0f}</strong></td></tr>"
                f"<tr><td style='padding:4px 8px;'>📤 Short Put</td><td style='padding:4px 8px;'>{fecha_short} · {dte_short} DTE · Mid <strong>${mid_short:.2f}</strong></td></tr>"
                f"<tr><td style='padding:4px 8px;'>📥 Long Put</td><td style='padding:4px 8px;'>{fecha_long} · {dte_long} DTE · Mid <strong>${mid_long:.2f}</strong></td></tr>"
                f"<tr><td style='padding:10px 8px 4px; font-size:1.1em;'>💰 Débito Neto</td>"
                f"<td style='padding:10px 8px 4px; font-size:1.1em; color:#ffb74d;'>"
                f"<strong>${debito:.2f} / contrato &nbsp;·&nbsp; ${debito*100:.2f} / lote</strong></td></tr>"
                f"</table></div>",
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("✅ Registrar este Calendar", type="primary"):
                registro = {
                    'Timestamp':       datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'SPX al abrir':    f"{precio_spx:,.2f}",
                    'Strike':          int(short_strike),
                    'Short Fecha':     fecha_short,
                    'Short DTE':       dte_short,
                    'Short Mid':       mid_short,
                    'Long Fecha':      fecha_long,
                    'Long DTE':        dte_long,
                    'Long Mid':        mid_long,
                    'Débito Neto':     debito,
                    'Débito x Lote':   round(debito * 100, 2),
                }
                st.session_state['te_calendar_registros'].append(registro)
                st.success("✅ Calendar registrado abajo.")

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
