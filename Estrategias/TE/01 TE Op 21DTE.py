# pages/graficos.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.utils import check_password
from utils.utils_schwab import connect_to_schwab, get_current_price_schwab, get_atm_strike_schwab
import io

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def get_next_friday(start_date, weeks_ahead=0):
    days_ahead = (4 - start_date.weekday()) % 7
    if days_ahead == 0 and weeks_ahead == 0:
        days_ahead = 7
    next_friday = start_date + timedelta(days=days_ahead + (weeks_ahead * 7))
    return next_friday


def create_calendar_order(strike, front_date, back_date, quantity, ticker):
    orders = []
    strike_int = int(strike)
    orders.append({
        'Action': 'SELL', 'Quantity': quantity, 'Symbol': ticker,
        'SecType': 'OPT', 'Expiry': front_date, 'Strike': strike_int,
        'Right': 'P', 'Exchange': 'SMART', 'Currency': 'USD', 'Label': 'PUT Front (SELL)'
    })
    orders.append({
        'Action': 'BUY', 'Quantity': quantity, 'Symbol': ticker,
        'SecType': 'OPT', 'Expiry': back_date, 'Strike': strike_int,
        'Right': 'P', 'Exchange': 'SMART', 'Currency': 'USD', 'Label': 'PUT Back (BUY)'
    })
    return pd.DataFrame(orders)


def initialize_session_state_te():
    state_vars = {
        'schwab_client_te': None, 'current_price_te': None,
        'strike_atm_te': None, 'dte_front_te': None, 'dte_back_te': None,
        'df_calendar_atm': None, 'order_preview_calendar': False,
        'df_adjustment': None, 'order_preview_adjustment': False
    }
    for var, default_value in state_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value


# ==============================================================================
# PUNTO DE ENTRADA PROTEGIDO
# ==============================================================================
if check_password():

    initialize_session_state_te()

    st.markdown(
        "<h1><span style='font-size: 1.5em;'>🦔</span> TE Op 21DTE - Calendar ATM</h1>",
        unsafe_allow_html=True
    )

    # ==========================================================================
    # SECCIÓN 1: GRÁFICOS
    # ==========================================================================

    st.header("1. Gráficos de Análisis Técnico")

    if 'datos_calculados' not in st.session_state:
        st.warning("⚠️ No hay datos calculados. Por favor, ve primero a la página principal (Home) para ejecutar los cálculos.")
    else:
        datos = st.session_state['datos_calculados']
        df_raw = datos['df_raw']
        spx = datos['spx']
        endog_final = datos['endog_final']
        results_k2 = datos['results_k2']
        nr_wr_series = datos['nr_wr_series']

        st.success(f"✅ Datos cargados desde memoria. ({len(spx)} días disponibles)")

        # --- CONTROLES DE FECHA ---
        st.sidebar.header("⚙️ Configuración del Gráfico")
        fecha_final = spx.index[-1].date()
        st.sidebar.info(f"📅 Última fecha disponible: {fecha_final}")
        fecha_inicio_default = fecha_final - timedelta(days=90)

        fecha_inicio = st.sidebar.date_input(
            "Fecha de inicio:",
            value=fecha_inicio_default,
            min_value=spx.index[0].date(),
            max_value=fecha_final
        )

        # --- FILTRAR DATOS ---
        fecha_inicio_dt = pd.to_datetime(fecha_inicio)
        fecha_final_dt = pd.to_datetime(fecha_final)
        spx_filtered = spx[(spx.index >= fecha_inicio_dt) & (spx.index <= fecha_final_dt)].copy()
        spx_filtered = spx_filtered[spx_filtered.index.dayofweek < 5]

        date_labels = [d.strftime('%b %d') if i % 5 == 0 else '' for i, d in enumerate(spx_filtered.index)]
        date_labels[0] = spx_filtered.index[0].strftime('%b %d')
        date_labels[-1] = spx_filtered.index[-1].strftime('%b %d')

        spx_filtered['RV_5d_pct'] = spx_filtered['RV_5d'] * 100
        UMBRAL_RV = 0.15
        spx_filtered['RV_change'] = spx_filtered['RV_5d_pct'].diff()
        is_up = spx_filtered['RV_change'] >= 0

        prob_baja_serie_k2 = results_k2['prob_baja_serie'].loc[spx_filtered.index].ffill()
        nr_wr_filtered = nr_wr_series.reindex(spx_filtered.index).fillna(0)

        # --- VIX / VIX3M ---
        tiene_vix3m = 'VIX3M' in spx_filtered.columns and spx_filtered['VIX3M'].notna().any()
        if tiene_vix3m:
            vix_filtered  = spx_filtered['VIX'].ffill()
            vix3m_filtered = spx_filtered['VIX3M'].ffill()

        UMBRAL_ALERTA       = 0.50
        UMBRAL_COMPRESION_K2 = results_k2['UMBRAL_COMPRESION']
        fechas_formateadas  = spx_filtered.index.strftime('%d-%m-%Y').tolist()
        x_range             = list(range(len(spx_filtered)))

        # --- CREAR SUBPLOTS (4 o 5 filas) ---
        n_rows      = 5 if tiene_vix3m else 4
        row_heights = [0.38, 0.14, 0.14, 0.14, 0.20] if tiene_vix3m else [0.45, 0.18, 0.19, 0.18]

        fig_combined = make_subplots(
            rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.02,
            row_heights=row_heights,
        )

        # ------------------------------------------------------------------
        # 1. VELAS
        # ------------------------------------------------------------------
        hover_text_candles = [
            f"<b>{fecha}</b><br>Open: {o:.2f}<br>High: {h:.2f}<br>Low: {l:.2f}<br>Close: {c:.2f}"
            for fecha, o, h, l, c in zip(
                fechas_formateadas, spx_filtered['Open'], spx_filtered['High'],
                spx_filtered['Low'], spx_filtered['Close']
            )
        ]
        fig_combined.add_trace(go.Candlestick(
            x=x_range, open=spx_filtered['Open'],
            high=spx_filtered['High'], low=spx_filtered['Low'], close=spx_filtered['Close'],
            name='S&P 500', text=hover_text_candles, hoverinfo='text',
            increasing=dict(line=dict(color='#00B06B')),
            decreasing=dict(line=dict(color='#F13A50'))
        ), row=1, col=1)
        fig_combined.update_yaxes(title_text='Precio', row=1, col=1)
        fig_combined.update_xaxes(showticklabels=False, row=1, col=1)

        # ------------------------------------------------------------------
        # 2. VOLATILIDAD REALIZADA
        # ------------------------------------------------------------------
        for i in range(len(spx_filtered) - 1):
            color = '#00B06B' if is_up.iloc[i+1] else '#F13A50'
            fig_combined.add_trace(go.Scatter(
                x=[i, i+1],
                y=[spx_filtered['RV_5d_pct'].iloc[i], spx_filtered['RV_5d_pct'].iloc[i+1]],
                mode='lines', line=dict(color=color, width=2),
                showlegend=False, hoverinfo='skip'
            ), row=2, col=1)

        fig_combined.add_trace(go.Scatter(
            x=x_range, y=spx_filtered['RV_5d_pct'],
            mode='markers', marker=dict(size=0.1, color='rgba(0,0,0,0)'),
            name='RV', customdata=[[fecha] for fecha in fechas_formateadas],
            hovertemplate='<b>%{customdata[0]}</b><br>RV: %{y:.2f}%<extra></extra>',
            showlegend=True
        ), row=2, col=1)

        fig_combined.add_shape(type="line", x0=0, y0=UMBRAL_RV * 100,
            x1=len(spx_filtered) - 1, y1=UMBRAL_RV * 100,
            line=dict(color="orange", width=2, dash="dot"), layer="below", row=2, col=1)
        fig_combined.add_annotation(x=0, y=1.0, text=f'Umbral RV: {UMBRAL_RV*100:.2f}%',
            showarrow=False, xref='x2', yref='y2 domain', xanchor='left', yanchor='top',
            font=dict(size=12, color="orange"), xshift=5, yshift=-5, row=2, col=1)
        fig_combined.update_yaxes(title_text='RV (%)', row=2, col=1, tickformat=".2f")
        fig_combined.update_xaxes(showticklabels=False, row=2, col=1)

        # ------------------------------------------------------------------
        # 3. MARKOV K=2
        # ------------------------------------------------------------------
        fig_combined.add_trace(go.Scatter(
            x=x_range, y=prob_baja_serie_k2,
            mode='lines', name='Prob. K=2 (Baja Vol.)',
            line=dict(color='#8A2BE2', width=2), fill='tozeroy',
            fillcolor='rgba(138, 43, 226, 0.3)',
            customdata=[[fecha] for fecha in fechas_formateadas],
            hovertemplate='<b>%{customdata[0]}</b><br>Prob. Baja K=2: %{y:.4f}<extra></extra>',
            showlegend=True
        ), row=3, col=1)

        fig_combined.add_shape(type="line", x0=0, y0=UMBRAL_COMPRESION_K2,
            x1=len(spx_filtered) - 1, y1=UMBRAL_COMPRESION_K2,
            line=dict(color="#FFD700", width=2, dash="dash"), layer="below", row=3, col=1)
        fig_combined.add_shape(type="line", x0=0, y0=UMBRAL_ALERTA,
            x1=len(spx_filtered) - 1, y1=UMBRAL_ALERTA,
            line=dict(color="#FFFFFF", width=1, dash="dot"), layer="below", row=3, col=1)
        fig_combined.add_annotation(x=0, y=UMBRAL_COMPRESION_K2,
            text=f'Compresión ({UMBRAL_COMPRESION_K2*100:.0f}%)',
            showarrow=False, xref='x3', yref='y3', xanchor='left', yanchor='bottom',
            font=dict(size=12, color="#FFD700"), xshift=5, yshift=5, row=3, col=1)
        fig_combined.add_annotation(x=0, y=UMBRAL_ALERTA, text=f'Alerta ({UMBRAL_ALERTA*100:.0f}%)',
            showarrow=False, xref='x3', yref='y3', xanchor='left', yanchor='bottom',
            font=dict(size=12, color="#FFFFFF"), xshift=5, yshift=5, row=3, col=1)
        fig_combined.update_yaxes(title_text='Prob. K=2', row=3, col=1, tickformat=".2f", range=[0, 1])
        fig_combined.update_xaxes(showticklabels=False, row=3, col=1)

        # ------------------------------------------------------------------
        # 4. NR/WR
        # ------------------------------------------------------------------
        fig_combined.add_trace(go.Bar(
            x=x_range, y=nr_wr_filtered,
            name='Señal NR/WR', marker=dict(color='#FF6B35', line=dict(width=0)),
            customdata=[[fecha, 'ACTIVA' if s > 0 else 'INACTIVA'] for fecha, s in zip(fechas_formateadas, nr_wr_filtered)],
            hovertemplate='<b>%{customdata[0]}</b><br>NR/WR: %{customdata[1]}<extra></extra>',
            showlegend=True, width=0.8
        ), row=4, col=1)

        fig_combined.add_shape(type="line", x0=-0.5, y0=0.5, x1=len(spx_filtered) - 0.5, y1=0.5,
            line=dict(color="#AAAAAA", width=1, dash="dot"), layer="below", row=4, col=1)
        fig_combined.add_annotation(x=0, y=0.9, text='COMPRESIÓN ACTIVA',
            showarrow=False, xref='x4', yref='y4', xanchor='left', yanchor='top',
            font=dict(size=11, color="#FF6B35"), xshift=5, yshift=-5, row=4, col=1)
        fig_combined.update_yaxes(title_text='NR/WR', row=4, col=1, range=[0, 1.05],
            tickvals=[0, 1], ticktext=['OFF', 'ON'])
        fig_combined.update_xaxes(
            showticklabels=False if tiene_vix3m else True,
            row=4, col=1
        )

        # ------------------------------------------------------------------
        # 5. VIX / VIX3M (solo si hay datos)
        # ------------------------------------------------------------------
        if tiene_vix3m:

            # Área de fondo: contango (verde) vs backwardation (rojo)
            # Dividimos la serie día a día para colorear correctamente
            for i in range(len(spx_filtered) - 1):
                es_contango = vix_filtered.iloc[i] <= vix3m_filtered.iloc[i]
                fill_color  = 'rgba(0, 176, 107, 0.15)' if es_contango else 'rgba(241, 58, 80, 0.15)'
                y_upper = max(vix_filtered.iloc[i], vix3m_filtered.iloc[i])
                y_lower = min(vix_filtered.iloc[i], vix3m_filtered.iloc[i])
                fig_combined.add_trace(go.Scatter(
                    x=[i, i+1, i+1, i],
                    y=[
                        vix_filtered.iloc[i],
                        vix_filtered.iloc[i+1],
                        vix3m_filtered.iloc[i+1],
                        vix3m_filtered.iloc[i]
                    ],
                    fill='toself',
                    fillcolor=fill_color,
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip',
                    mode='lines'
                ), row=5, col=1)

            # Línea VIX3M (verde)
            fig_combined.add_trace(go.Scatter(
                x=x_range, y=vix3m_filtered,
                mode='lines', name='VIX3M',
                line=dict(color='#00B06B', width=2),
                customdata=[[fecha] for fecha in fechas_formateadas],
                hovertemplate='<b>%{customdata[0]}</b><br>VIX3M: %{y:.2f}<extra></extra>',
                showlegend=True
            ), row=5, col=1)

            # Línea VIX spot (blanco)
            fig_combined.add_trace(go.Scatter(
                x=x_range, y=vix_filtered,
                mode='lines', name='VIX',
                line=dict(color='#FFFFFF', width=2),
                customdata=[[fecha] for fecha in fechas_formateadas],
                hovertemplate='<b>%{customdata[0]}</b><br>VIX: %{y:.2f}<extra></extra>',
                showlegend=True
            ), row=5, col=1)

            # Anotación del estado actual
            ultimo_estado = "CONTANGO ✅" if vix_filtered.iloc[-1] <= vix3m_filtered.iloc[-1] else "BACKWARDATION ⚠️"
            ultimo_color  = "#00B06B" if vix_filtered.iloc[-1] <= vix3m_filtered.iloc[-1] else "#F13A50"
            fig_combined.add_annotation(
                x=len(spx_filtered) - 1, y=vix_filtered.iloc[-1],
                text=ultimo_estado,
                showarrow=False,
                xref=f'x{n_rows}', yref=f'y{n_rows}',
                xanchor='right', yanchor='bottom',
                font=dict(size=11, color=ultimo_color),
                xshift=-5, yshift=5,
                row=5, col=1
            )

            fig_combined.update_yaxes(title_text='VIX / VIX3M', row=5, col=1, tickformat=".1f")
            fig_combined.update_xaxes(showticklabels=True, row=5, col=1)

        # ------------------------------------------------------------------
        # CONFIGURACIÓN FINAL
        # ------------------------------------------------------------------
        height = 1150 if tiene_vix3m else 950

        fig_combined.update_layout(
            template='plotly_dark', height=height, xaxis_rangeslider_visible=False,
            hovermode='x', plot_bgcolor='#131722', paper_bgcolor='#131722',
            font=dict(color='#AAAAAA'), margin=dict(t=50, b=100, l=60, r=40),
            showlegend=True, legend=dict(orientation="v", yanchor="top", y=1,
                xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)",
                bordercolor="rgba(255,255,255,0.1)", borderwidth=1, font=dict(size=10))
        )

        for i in range(1, n_rows + 1):
            fig_combined.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor',
                spikecolor='#AAAAAA', spikethickness=1, spikedash='dash', row=i, col=1)
            fig_combined.update_yaxes(showspikes=False, row=i, col=1)

        # Eje X con fechas en la última fila
        fig_combined.update_xaxes(
            tickmode='array', tickvals=x_range,
            ticktext=date_labels, tickangle=-45,
            row=n_rows, col=1, showgrid=False
        )

        for i in range(1, n_rows + 1):
            fig_combined.update_xaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=i, col=1)
            fig_combined.update_yaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=i, col=1)

        st.plotly_chart(fig_combined, use_container_width=True)

        # ------------------------------------------------------------------
        # MÉTRICAS
        # ------------------------------------------------------------------
        st.markdown("---")
        n_cols = 7 if tiene_vix3m else 6
        cols = st.columns(n_cols)

        with cols[0]:
            st.metric("Precio Actual", f"${spx_filtered['Close'].iloc[-1]:.2f}")
        with cols[1]:
            cambio = spx_filtered['Close'].iloc[-1] - spx_filtered['Close'].iloc[0]
            cambio_pct = (cambio / spx_filtered['Close'].iloc[0]) * 100
            st.metric(f"Cambio ({fecha_inicio} al {fecha_final})", f"${cambio:.2f}", f"{cambio_pct:.2f}%")
        with cols[2]:
            st.metric("Máximo", f"${spx_filtered['High'].max():.2f}")
        with cols[3]:
            st.metric("Mínimo", f"${spx_filtered['Low'].min():.2f}")
        with cols[4]:
            rv_latest = spx_filtered['RV_5d'].iloc[-1] * 100
            st.metric("RV_5d (Último)", f"{rv_latest:.2f}%")
        with cols[5]:
            nr_wr_status = "🟢 ACTIVA" if nr_wr_filtered.iloc[-1] > 0 else "⚪ INACTIVA"
            st.metric("Señal NR/WR", nr_wr_status)
        if tiene_vix3m:
            with cols[6]:
                ratio_actual = vix_filtered.iloc[-1] / vix3m_filtered.iloc[-1]
                estado_ts = "🟢 Contango" if ratio_actual <= 0.95 else ("🟡 Débil" if ratio_actual <= 1.0 else "🔴 Backw.")
                st.metric("VIX/VIX3M", f"{ratio_actual:.4f}", delta=estado_ts, delta_color="off")

    st.markdown("---")

    # ==========================================================================
    # SECCIÓN 2: GENERADOR DE ESTRUCTURA - CALENDAR ATM SPX
    # ==========================================================================

    st.header("2. Generador de Estructura - Calendar SPX")

    st.markdown("""
    Configura el **Calendar Spread** para SPX con control total sobre cada pata de la estrategia.
    El sistema obtendrá automáticamente el precio actual y el strike ATM desde Schwab como sugerencia.
    """)

    ticker_spx = "SPX"
    st.info(f"📊 Ticker: **{ticker_spx}** (fijo)")

    if st.session_state.schwab_client_te is None or st.session_state.current_price_te is None:
        with st.spinner(f"Conectando con Schwab y obteniendo datos de {ticker_spx}..."):
            schwab_client = connect_to_schwab()
            if schwab_client is not None:
                st.session_state.schwab_client_te = schwab_client
                current_price_spx = get_current_price_schwab(schwab_client, ticker_spx)
                if current_price_spx is not None:
                    st.session_state.current_price_te = current_price_spx
                    today = date.today()
                    temp_front_friday = get_next_friday(today, weeks_ahead=2)
                    strike_atm_spx = get_atm_strike_schwab(schwab_client, ticker_spx, current_price_spx, temp_front_friday)
                    if strike_atm_spx is not None:
                        st.session_state.strike_atm_te = strike_atm_spx

    if st.session_state.current_price_te:
        st.success(f"✅ Precio actual de {ticker_spx}: **${st.session_state.current_price_te:.2f}**")
        if st.session_state.strike_atm_te:
            st.info(f"💡 Strike ATM Sugerido: **${st.session_state.strike_atm_te:.0f}**")

    today = date.today()
    default_front_friday = get_next_friday(today, weeks_ahead=2)
    default_back_friday = get_next_friday(today, weeks_ahead=3)

    if st.session_state.strike_atm_te:
        default_strike = st.session_state.strike_atm_te
    elif st.session_state.current_price_te:
        default_strike = round(st.session_state.current_price_te / 5) * 5
    else:
        default_strike = 5900.0

    st.markdown("---")

    col_front, col_back = st.columns(2)

    with col_front:
        st.markdown("### 📅 DTE FRONT (Primera Pata)")
        st.markdown("---")

        dte_front_calendar = st.date_input(
            "Fecha de Expiración FRONT",
            value=default_front_friday,
            min_value=date.today() + timedelta(days=1),
            max_value=date.today() + timedelta(days=365),
            key='dte_front_calendar'
        )

        if dte_front_calendar.weekday() != 4:
            st.warning("⚠️ Ajustando a viernes...")
            days_to_friday = (4 - dte_front_calendar.weekday()) % 7
            if days_to_friday == 0:
                days_to_friday = 7
            dte_front_calendar = dte_front_calendar + timedelta(days=days_to_friday)
            st.info(f"✅ Ajustado a: {dte_front_calendar.strftime('%Y-%m-%d')}")

        st.markdown("#### 🎯 Configuración FRONT")

        strike_front = st.number_input(
            "Strike FRONT", min_value=0.0, value=float(default_strike),
            step=5.0, key='strike_front', help="Strike para la pata FRONT"
        )
        option_type_front = st.selectbox("Tipo de Opción FRONT", ["PUT", "CALL"], index=0, key='option_type_front')
        action_front = st.selectbox("Acción FRONT", ["SELL", "BUY"], index=0, key='action_front')

        if st.session_state.current_price_te:
            diff_front = strike_front - st.session_state.current_price_te
            diff_pct_front = (diff_front / st.session_state.current_price_te) * 100
            if option_type_front == "PUT":
                status_front = "ITM" if diff_front > 0 else ("ATM" if diff_front >= -10 else "OTM")
                color_front = "🔴" if status_front == "ITM" else ("🟡" if status_front == "ATM" else "🟢")
            else:
                status_front = "ITM" if diff_front < 0 else ("ATM" if diff_front <= 10 else "OTM")
                color_front = "🔴" if status_front == "ITM" else ("🟡" if status_front == "ATM" else "🟢")
            st.markdown(f"**📊 Análisis:**\n- Diferencia: **{diff_front:+.2f}** ({diff_pct_front:+.2f}%)\n- Estado: {color_front} **{status_front}**")

    with col_back:
        st.markdown("### 📅 DTE BACK (Segunda Pata)")
        st.markdown("---")

        dte_back_calendar = st.date_input(
            "Fecha de Expiración BACK",
            value=default_back_friday,
            min_value=dte_front_calendar + timedelta(days=1),
            max_value=date.today() + timedelta(days=365),
            key='dte_back_calendar'
        )

        if dte_back_calendar.weekday() != 4:
            st.warning("⚠️ Ajustando a viernes...")
            days_to_friday = (4 - dte_back_calendar.weekday()) % 7
            if days_to_friday == 0:
                days_to_friday = 7
            dte_back_calendar = dte_back_calendar + timedelta(days=days_to_friday)
            st.info(f"✅ Ajustado a: {dte_back_calendar.strftime('%Y-%m-%d')}")

        st.markdown("#### 🎯 Configuración BACK")

        strike_back = st.number_input(
            "Strike BACK", min_value=0.0, value=float(default_strike),
            step=5.0, key='strike_back', help="Strike para la pata BACK"
        )
        option_type_back = st.selectbox("Tipo de Opción BACK", ["PUT", "CALL"], index=0, key='option_type_back')
        action_back = st.selectbox("Acción BACK", ["BUY", "SELL"], index=0, key='action_back')

        if st.session_state.current_price_te:
            diff_back = strike_back - st.session_state.current_price_te
            diff_pct_back = (diff_back / st.session_state.current_price_te) * 100
            if option_type_back == "PUT":
                status_back = "ITM" if diff_back > 0 else ("ATM" if diff_back >= -10 else "OTM")
                color_back = "🔴" if status_back == "ITM" else ("🟡" if status_back == "ATM" else "🟢")
            else:
                status_back = "ITM" if diff_back < 0 else ("ATM" if diff_back <= 10 else "OTM")
                color_back = "🔴" if status_back == "ITM" else ("🟡" if status_back == "ATM" else "🟢")
            st.markdown(f"**📊 Análisis:**\n- Diferencia: **{diff_back:+.2f}** ({diff_pct_back:+.2f}%)\n- Estado: {color_back} **{status_back}**")

    days_diff_calendar = (dte_back_calendar - dte_front_calendar).days
    st.success(f"📅 Diferencia entre expiraciones: **{days_diff_calendar} días**")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 💰 Configuración de Orden")
        quantity_calendar = st.number_input("Cantidad de Contratos", min_value=1, value=1, step=1, key='quantity_calendar')
        limit_price_calendar = st.number_input("Precio Límite (Total)", min_value=0.0, value=1.0, step=0.05, format="%.2f", key='limit_price_calendar')
        st.info(f"💡 **Configuración:**\n- Precio Límite Total: **${limit_price_calendar:.2f}**\n- Por contrato: **${limit_price_calendar / quantity_calendar if quantity_calendar > 0 else 0:.2f}**")

    with col2:
        st.markdown("#### 📌 Configuración IBKR")
        ibkr_host_calendar = st.text_input("Host IBKR", value="127.0.0.1", key='ibkr_host_calendar')
        ibkr_port_calendar = st.number_input("Puerto IBKR", min_value=1, max_value=65535, value=5000, step=1, key='ibkr_port_calendar')
        ibkr_client_id_calendar = st.number_input("Client ID", min_value=1, value=1, step=1, key='ibkr_client_id_calendar')

    st.markdown("---")

    if st.button("📝 Generar Vista Previa de Orden", type="primary", use_container_width=True):
        st.session_state.dte_front_te = dte_front_calendar
        st.session_state.dte_back_te = dte_back_calendar
        orders = []
        orders.append({
            'Action': action_front, 'Quantity': quantity_calendar, 'Symbol': ticker_spx,
            'SecType': 'OPT', 'Expiry': dte_front_calendar.strftime("%Y-%m-%d"),
            'Strike': int(strike_front), 'Right': 'P' if option_type_front == "PUT" else 'C',
            'Exchange': 'SMART', 'Currency': 'USD', 'Label': f'{option_type_front} Front ({action_front})'
        })
        orders.append({
            'Action': action_back, 'Quantity': quantity_calendar, 'Symbol': ticker_spx,
            'SecType': 'OPT', 'Expiry': dte_back_calendar.strftime("%Y-%m-%d"),
            'Strike': int(strike_back), 'Right': 'P' if option_type_back == "PUT" else 'C',
            'Exchange': 'SMART', 'Currency': 'USD', 'Label': f'{option_type_back} Back ({action_back})'
        })
        st.session_state.df_calendar_atm = pd.DataFrame(orders)
        st.session_state.order_preview_calendar = True
        st.success("✅ Vista previa de Calendar generada exitosamente!")

    if st.session_state.order_preview_calendar and st.session_state.df_calendar_atm is not None:
        df_calendar = st.session_state.df_calendar_atm
        st.markdown("---")
        st.markdown("### 📋 Vista Previa - Calendar ATM SPX")
        st.dataframe(df_calendar, hide_index=True, use_container_width=True)
        st.markdown("---")
        st.markdown("### 📊 Resumen de la Orden")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Ticker", ticker_spx)
        with col2: st.metric("Strike ATM", f"${st.session_state.strike_atm_te:.0f}")
        with col3: st.metric("Precio Actual", f"${st.session_state.current_price_te:.2f}")
        with col4: st.metric("Contratos", quantity_calendar)
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🎯 Configuración de las Patas**")
            st.write(f"- FRONT: {action_front} {option_type_front} @ ${strike_front:.0f}")
            st.write(f"  Expiración: {dte_front_calendar.strftime('%Y-%m-%d')} ({dte_front_calendar.strftime('%A')})")
            st.write(f"- BACK: {action_back} {option_type_back} @ ${strike_back:.0f}")
            st.write(f"  Expiración: {dte_back_calendar.strftime('%Y-%m-%d')} ({dte_back_calendar.strftime('%A')})")
        with col2:
            st.markdown("**💰 Configuración**")
            st.write(f"- Precio Límite: ${limit_price_calendar:.2f}")
            st.write(f"- Contratos: {quantity_calendar}")
            st.write(f"- Total Piernas: {len(df_calendar)}")
            st.write(f"- Spread: {days_diff_calendar} días")
        st.markdown("---")
        st.markdown("**📌 Configuración de Conexión IBKR**")
        st.write(f"- Host: {ibkr_host_calendar} | Puerto: {ibkr_port_calendar} | Client ID: {ibkr_client_id_calendar}")
        st.markdown("---")
        st.warning("⚠️ **IMPORTANTE:** Asegúrate de que TWS/Gateway esté ejecutándose y la configuración sea correcta.")

        if st.button("🚀 ENVIAR ORDEN A IBKR", type="primary", use_container_width=True, key='send_calendar'):
            try:
                from utils.utils_ibkr import send_strategy_order_ibkr
            except ImportError:
                st.error("❌ Error: No se pudo importar send_strategy_order_ibkr")
                st.stop()
            if limit_price_calendar <= 0:
                st.error("❌ El precio límite debe ser mayor a 0")
                st.stop()
            with st.spinner("📡 Enviando orden a IBKR..."):
                result = send_strategy_order_ibkr(
                    df_strategy=df_calendar, limit_price=limit_price_calendar,
                    host=ibkr_host_calendar, port=int(ibkr_port_calendar),
                    client_id=int(ibkr_client_id_calendar), quantity=quantity_calendar,
                    tif='DAY', action='BUY', timeout=10
                )
            if result['success']:
                st.success(f"✅ {result['message']}")
                if result.get('order_id'): st.info(f"📋 Order ID: {result['order_id']}")
                if result.get('contracts'):
                    st.markdown("**✅ Contratos Calificados:**")
                    for i, c in enumerate(result['contracts'], 1):
                        st.write(f"{i}. {c.symbol} {c.lastTradeDateOrContractMonth} {c.strike} {c.right}")
            else:
                st.error(f"❌ {result['message']}")

    st.markdown("---")

    # ==========================================================================
    # SECCIÓN 3: AJUSTES - CALENDAR INDIVIDUAL
    # ==========================================================================

    st.header("3. Ajustes - Generador de Calendar Individual (SPX)")

    st.markdown("""
    Esta sección te permite generar un **Calendar Spread individual** adicional para ajustar tu posición.
    Configura cada pata de forma independiente con control total sobre strikes, tipos y acciones.
    """)

    if st.session_state.schwab_client_te is not None:
        schwab_client = st.session_state.schwab_client_te
        current_price_adj = get_current_price_schwab(schwab_client, ticker_spx)
        if current_price_adj is None:
            current_price_adj = st.session_state.current_price_te if st.session_state.current_price_te else 0
    else:
        current_price_adj = st.session_state.current_price_te if st.session_state.current_price_te else 0

    if current_price_adj > 0:
        st.success(f"✅ Precio actual de {ticker_spx}: **${current_price_adj:.2f}**")
        if st.session_state.strike_atm_te:
            st.info(f"💡 Strike ATM de Referencia: **${st.session_state.strike_atm_te:.0f}**")
    else:
        st.warning("⚠️ Conecta con Schwab en la Sección 2 primero para obtener el precio actual")

    if st.session_state.strike_atm_te:
        default_strike_adj = st.session_state.strike_atm_te
    else:
        default_strike_adj = round(current_price_adj / 5) * 5 if current_price_adj > 0 else 5900

    if st.session_state.dte_front_te and isinstance(st.session_state.dte_front_te, date):
        default_front_adj = st.session_state.dte_front_te
    else:
        default_front_adj = get_next_friday(today, weeks_ahead=2)

    if st.session_state.dte_back_te and isinstance(st.session_state.dte_back_te, date):
        default_back_adj = st.session_state.dte_back_te
    else:
        default_back_adj = get_next_friday(today, weeks_ahead=3)

    st.markdown("---")
    col_front_adj, col_back_adj = st.columns(2)

    with col_front_adj:
        st.markdown("### 📅 DTE FRONT (Primera Pata)")
        st.markdown("---")
        dte_front_adj = st.date_input(
            "Fecha de Expiración FRONT", value=default_front_adj,
            min_value=date.today() + timedelta(days=1),
            max_value=date.today() + timedelta(days=365), key='dte_front_adj_spx'
        )
        if dte_front_adj.weekday() != 4:
            st.warning("⚠️ Ajustando a viernes...")
            days_to_friday = (4 - dte_front_adj.weekday()) % 7
            if days_to_friday == 0: days_to_friday = 7
            dte_front_adj = dte_front_adj + timedelta(days=days_to_friday)
            st.info(f"✅ Ajustado a: {dte_front_adj.strftime('%Y-%m-%d')}")
        st.markdown("#### 🎯 Configuración FRONT")
        strike_front_adj = st.number_input("Strike FRONT", min_value=0.0, value=float(default_strike_adj), step=5.0, key='strike_front_adj')
        option_type_front_adj = st.selectbox("Tipo de Opción FRONT", ["PUT", "CALL"], index=0, key='option_type_front_adj')
        action_front_adj = st.selectbox("Acción FRONT", ["SELL", "BUY"], index=0, key='action_front_adj')
        if current_price_adj > 0:
            diff_front_adj = strike_front_adj - current_price_adj
            diff_pct_front_adj = (diff_front_adj / current_price_adj) * 100
            if option_type_front_adj == "PUT":
                status_front_adj = "ITM" if diff_front_adj > 0 else ("ATM" if diff_front_adj >= -10 else "OTM")
            else:
                status_front_adj = "ITM" if diff_front_adj < 0 else ("ATM" if diff_front_adj <= 10 else "OTM")
            color_front_adj = "🔴" if status_front_adj == "ITM" else ("🟡" if status_front_adj == "ATM" else "🟢")
            st.markdown(f"**📊 Análisis:**\n- Diferencia: **{diff_front_adj:+.2f}** ({diff_pct_front_adj:+.2f}%)\n- Estado: {color_front_adj} **{status_front_adj}**")

    with col_back_adj:
        st.markdown("### 📅 DTE BACK (Segunda Pata)")
        st.markdown("---")
        min_back_date_adj = dte_front_adj + timedelta(days=1)
        if default_back_adj <= dte_front_adj:
            default_back_adj = get_next_friday(dte_front_adj, weeks_ahead=1)
        dte_back_adj = st.date_input(
            "Fecha de Expiración BACK", value=default_back_adj,
            min_value=min_back_date_adj, max_value=date.today() + timedelta(days=365),
            key='dte_back_adj_spx'
        )
        if dte_back_adj.weekday() != 4:
            st.warning("⚠️ Ajustando a viernes...")
            days_to_friday = (4 - dte_back_adj.weekday()) % 7
            if days_to_friday == 0: days_to_friday = 7
            dte_back_adj = dte_back_adj + timedelta(days=days_to_friday)
            st.info(f"✅ Ajustado a: {dte_back_adj.strftime('%Y-%m-%d')}")
        st.markdown("#### 🎯 Configuración BACK")
        strike_back_adj = st.number_input("Strike BACK", min_value=0.0, value=float(default_strike_adj), step=5.0, key='strike_back_adj')
        option_type_back_adj = st.selectbox("Tipo de Opción BACK", ["PUT", "CALL"], index=0, key='option_type_back_adj')
        action_back_adj = st.selectbox("Acción BACK", ["BUY", "SELL"], index=0, key='action_back_adj')
        if current_price_adj > 0:
            diff_back_adj = strike_back_adj - current_price_adj
            diff_pct_back_adj = (diff_back_adj / current_price_adj) * 100
            if option_type_back_adj == "PUT":
                status_back_adj = "ITM" if diff_back_adj > 0 else ("ATM" if diff_back_adj >= -10 else "OTM")
            else:
                status_back_adj = "ITM" if diff_back_adj < 0 else ("ATM" if diff_back_adj <= 10 else "OTM")
            color_back_adj = "🔴" if status_back_adj == "ITM" else ("🟡" if status_back_adj == "ATM" else "🟢")
            st.markdown(f"**📊 Análisis:**\n- Diferencia: **{diff_back_adj:+.2f}** ({diff_pct_back_adj:+.2f}%)\n- Estado: {color_back_adj} **{status_back_adj}**")

    days_diff_adj = (dte_back_adj - dte_front_adj).days
    st.success(f"📅 Diferencia entre expiraciones: **{days_diff_adj} días**")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 💰 Configuración de Orden de Ajuste")
        quantity_adj = st.number_input("Cantidad de Contratos", min_value=1, value=1, step=1, key='quantity_adj_spx')
        limit_price_adj = st.number_input("Precio Límite (Total)", min_value=0.0, value=1.0, step=0.05, format="%.2f", key='limit_price_adj_spx')
        st.info(f"💡 **Configuración:**\n- Precio Límite Total: **${limit_price_adj:.2f}**\n- Por contrato: **${limit_price_adj / quantity_adj if quantity_adj > 0 else 0:.2f}**")
    with col2:
        st.markdown("#### 📌 Configuración IBKR")
        ibkr_host_adj = st.text_input("Host", value="127.0.0.1", key='ibkr_host_adj_spx')
        ibkr_port_adj = st.number_input("Puerto", min_value=1, max_value=65535, value=5000, step=1, key='ibkr_port_adj_spx')
        ibkr_client_id_adj = st.number_input("Client ID", min_value=1, value=1, step=1, key='ibkr_client_id_adj_spx')

    st.markdown("---")

    if st.button("📝 Generar Vista Previa de Ajuste", type="primary", use_container_width=True):
        orders_adj = []
        orders_adj.append({
            'Action': action_front_adj, 'Quantity': quantity_adj, 'Symbol': ticker_spx,
            'SecType': 'OPT', 'Expiry': dte_front_adj.strftime("%Y-%m-%d"),
            'Strike': int(strike_front_adj), 'Right': 'P' if option_type_front_adj == "PUT" else 'C',
            'Exchange': 'SMART', 'Currency': 'USD', 'Label': f'{option_type_front_adj} Front ({action_front_adj})'
        })
        orders_adj.append({
            'Action': action_back_adj, 'Quantity': quantity_adj, 'Symbol': ticker_spx,
            'SecType': 'OPT', 'Expiry': dte_back_adj.strftime("%Y-%m-%d"),
            'Strike': int(strike_back_adj), 'Right': 'P' if option_type_back_adj == "PUT" else 'C',
            'Exchange': 'SMART', 'Currency': 'USD', 'Label': f'{option_type_back_adj} Back ({action_back_adj})'
        })
        st.session_state.df_adjustment = pd.DataFrame(orders_adj)
        st.session_state.order_preview_adjustment = True
        st.success("✅ Vista previa de ajuste generada!")

    if st.session_state.order_preview_adjustment and st.session_state.df_adjustment is not None:
        df_adjustment = st.session_state.df_adjustment
        st.markdown("---")
        st.markdown("### 📋 Vista Previa - Ajuste Calendar Individual")
        st.dataframe(df_adjustment, hide_index=True, use_container_width=True)
        st.markdown("---")
        st.markdown("### 📊 Resumen del Ajuste")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**🎯 Configuración**")
            st.write(f"FRONT: {action_front_adj} {option_type_front_adj} @ ${strike_front_adj:.0f}")
            st.write(f"BACK: {action_back_adj} {option_type_back_adj} @ ${strike_back_adj:.0f}")
            st.write(f"Cantidad: {quantity_adj}")
        with col2:
            st.markdown("**📅 Fechas**")
            st.write(f"FRONT: {dte_front_adj.strftime('%Y-%m-%d')} ({dte_front_adj.strftime('%A')})")
            st.write(f"BACK: {dte_back_adj.strftime('%Y-%m-%d')} ({dte_back_adj.strftime('%A')})")
            st.write(f"Spread: {days_diff_adj} días")
        with col3:
            st.markdown("**⚙️ Datos**")
            st.write(f"Ticker: {ticker_spx}")
            st.write(f"Precio: ${limit_price_adj:.2f}")
            st.write(f"Órdenes: {len(df_adjustment)}")
        st.markdown("---")
        st.warning("⚠️ Asegúrate de que TWS/Gateway esté ejecutándose")

        if st.button("🚀 ENVIAR AJUSTE A IBKR", type="primary", use_container_width=True, key='send_adjustment'):
            try:
                from utils.utils_ibkr import send_strategy_order_ibkr
            except ImportError:
                st.error("❌ Error: No se pudo importar send_strategy_order_ibkr")
                st.stop()
            if limit_price_adj <= 0:
                st.error("❌ El precio límite debe ser mayor a 0")
                st.stop()
            with st.spinner("📡 Enviando ajuste..."):
                result = send_strategy_order_ibkr(
                    df_strategy=df_adjustment, limit_price=limit_price_adj,
                    host=ibkr_host_adj, port=int(ibkr_port_adj),
                    client_id=int(ibkr_client_id_adj), quantity=quantity_adj,
                    tif='DAY', action='BUY', timeout=10
                )
            if result['success']:
                st.success(f"✅ {result['message']}")
                if result.get('order_id'): st.info(f"📋 Order ID: {result['order_id']}")
                if result.get('contracts'):
                    st.markdown("**✅ Contratos Calificados:**")
                    for i, c in enumerate(result['contracts'], 1):
                        st.write(f"{i}. {c.symbol} {c.lastTradeDateOrContractMonth} {c.strike} {c.right}")
            else:
                st.error(f"❌ {result['message']}")

        st.markdown("---")
        st.info("""
        **📚 Sobre el Ajuste:**
        - **Propósito:** Añadir un calendar spread individual para ajustar tu posición
        - **Tipo:** LIMIT | Acción: BUY | TIF: DAY

        **🎯 Cuándo Ajustar:**
        - El precio del SPX se ha movido significativamente
        - Necesitas rebalancear tu exposición direccional
        - Quieres añadir protección en un nuevo strike
        - Deseas aprovechar cambios en la volatilidad implícita
        """)

else:
    st.title("🔒 Acceso Restringido")
    st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder.")
