# pages/graficos.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.utils import check_password

# ==============================================================================
# PUNTO DE ENTRADA PROTEGIDO
# ==============================================================================
if check_password():

    st.markdown(
        "<h1><span style='font-size: 1.5em;'>🦔</span> TE - Gráficos</h1>",
        unsafe_allow_html=True
    )

    # ==========================================================================
    # GRÁFICOS
    # ==========================================================================

    st.header("Gráficos de Análisis Técnico")

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

            # Anotación del estado actual — esquina superior derecha con fondo
            es_contango_actual = vix_filtered.iloc[-1] <= vix3m_filtered.iloc[-1]
            ultimo_estado = "CONTANGO ✅" if es_contango_actual else "BACKWARDATION ⚠️"
            ultimo_color  = "#00B06B" if es_contango_actual else "#F13A50"
            bgcolor_label = "rgba(0, 176, 107, 0.20)" if es_contango_actual else "rgba(241, 58, 80, 0.20)"
            fig_combined.add_annotation(
                x=1, y=1,
                text=f"<b>{ultimo_estado}</b>",
                showarrow=False,
                xref='x5 domain', yref='y5 domain',
                xanchor='right', yanchor='top',
                font=dict(size=12, color=ultimo_color),
                bgcolor=bgcolor_label,
                bordercolor=ultimo_color,
                borderwidth=1,
                borderpad=5,
                xshift=-8, yshift=-8,
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

else:
    st.title("🔒 Acceso Restringido")
    st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder.")
