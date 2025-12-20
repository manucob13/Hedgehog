# pages/triple_calendar.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import (
    fetch_data, 
    calculate_indicators, 
    preparar_datos_markov,
    calculate_nr_wr_signal,
    calculate_nr_wr_signal_series,
    markov_calculation_k2,
    markov_calculation_k3,
    check_password 
)

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Triple Calendar", layout="wide")

# ==============================================================================
# CONFIGURACIÓN Y VALORES POR DEFECTO PARA TRIPLE CALENDAR
# ==============================================================================

def get_default_config_df_triple_calendar(rv5d_ayer_val):
    """Genera el DataFrame de configuración de reglas con valores específicos para Triple Calendar."""
    
    default_config_data = {
        'Regla': [
            '1. Señal NR/WR Activa', 
            '2. Prob. K=2 Baja Vol.', 
            '3. Prob. K=3 Media Vol.', 
            '4. Prob. K=3 Baja Vol.', 
            '5. Prob. K=3 Consolidada', 
            '6. RV_5d Actual', 
            f'7. RV_5d HOY vs. AYER ({rv5d_ayer_val:.4f})'
        ],
        'Operador': ['==', '>=', '>=', '>=', '>=', '>=', '<'],
        # Umbrales - RV_5d cambiado a 0.15
        'Umbral': ['ON', '0.9000', '0.7500', '0.1500', '0.9500', '0.1500', 'RV_AYER'], 
        # Activación - NR/WR en OFF por defecto
        'Activa': [False, True, False, False, True, True, False], 
        'ID': ['r1_nr_wr', 'r2_k2_70', 'r3_k3_media_75', 'r4_k3_baja_15', 'r5_k3_consol_95', 'r6_rv5d_10', 'r7_rv5d_menor']
    }
    return pd.DataFrame(default_config_data)

def reset_config_callback_triple(rv5d_ayer_val):
    """Callback para el botón de reset: Restaura la configuración específica de Triple Calendar."""
    st.session_state['config_df_triple'] = get_default_config_df_triple_calendar(rv5d_ayer_val)
    # Eliminar el estado calculado del semáforo
    for key in ['df_semaforo_body_triple', 'df_semaforo_footer_triple', 'senal_color_triple']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# ==============================================================================
# LÓGICA DE CÁLCULO DEL SEMÁFORO
# ==============================================================================

def calcular_y_mostrar_semaforo_triple(df_config, metricas_actuales, rv5d_ayer):
    """Calcula el estado de cada regla y el resultado global del Semáforo."""
    
    df_config_calc = df_config.copy()
    
    def safe_float_convert(value):
        try:
            if isinstance(value, str) and value.upper() in ['ON', 'OFF', 'RV_AYER']:
                return value
            return float(value)
        except (ValueError, TypeError):
            return value

    df_config_calc['Umbral_Calc'] = df_config_calc['Umbral'].apply(safe_float_convert)
    
    # Añadir la columna 'Valor Actual'
    df_config_calc['Valor Actual'] = df_config_calc['ID'].apply(lambda id: 
        (metricas_actuales[id] and '🟢 ACTIVA' or '⚪ INACTIVA') if id == 'r1_nr_wr' else 
        f"{metricas_actuales[id]:.4f}"
    )

    senal_entrada_global_interactiva = True
    num_reglas_activas = 0
    df_config_calc['Cumple'] = 'NO'
    
    for index, row in df_config_calc.iterrows():
        rule_id = row['ID']
        metrica_actual = metricas_actuales[rule_id]
        operador = row['Operador']
        umbral_calc = row['Umbral_Calc']
        umbral_str = str(row['Umbral']).upper()
        regla_cumplida = False
        
        if row['ID'] == 'r1_nr_wr':
            if umbral_str == 'ON':
                regla_cumplida = metrica_actual 
            elif umbral_str == 'OFF':
                regla_cumplida = not metrica_actual
        
        elif row['ID'] == 'r7_rv5d_menor':
            regla_cumplida = metrica_actual < rv5d_ayer
            
        else:
            if isinstance(umbral_calc, (float, int)):
                if operador == '>=':
                    regla_cumplida = metrica_actual >= umbral_calc
                elif operador == '<=':
                    regla_cumplida = metrica_actual <= umbral_calc

        if row['Activa']:
            if regla_cumplida:
                df_config_calc.loc[index, 'Cumple'] = "SÍ"
            else:
                df_config_calc.loc[index, 'Cumple'] = "NO"
        else:
             df_config_calc.loc[index, 'Cumple'] = "INACTIVA"

        if row['Activa']:
            num_reglas_activas += 1
            if not regla_cumplida:
                senal_entrada_global_interactiva = False

    df_presentacion = df_config_calc[['Activa', 'Regla', 'Operador', 'Umbral', 'Valor Actual', 'Cumple', 'ID']].copy()
    
    if num_reglas_activas == 0:
        res_final_texto = "INACTIVA (0 Reglas Activas)"
        senal_color = "background-color: #AAAAAA; color: black"
    elif senal_entrada_global_interactiva:
        res_final_texto = ""
        senal_color = "background-color: #008000; color: white"
    else:
        res_final_texto = ""
        senal_color = "background-color: #8B0000; color: white"
        
    fila_resumen = pd.DataFrame([{
        'Regla': '🚥 SEMÁFORO TRIPLE CALENDAR 🚥', 
        'ID': 'FINAL' 
    }])
    
    st.session_state['df_semaforo_body_triple'] = df_presentacion
    st.session_state['df_semaforo_footer_triple'] = fila_resumen
    st.session_state['senal_color_triple'] = senal_color


# ==============================================================================
# FUNCIÓN PRINCIPAL - TRIPLE CALENDAR
# ==============================================================================

def main_triple_calendar():
    
    # --- TÍTULO PRINCIPAL ---
    st.markdown("<h1><span style='font-size: 1.5em;'>📅</span> Triple Calendar Strategy Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("""
    Esta herramienta analiza las condiciones óptimas para estrategias Triple Calendar en diferentes activos,
    utilizando modelos de Markov-Switching y señales de compresión de volatilidad.
    """)
    st.markdown("---")
    
    # ==============================================================================
    # SECCIÓN 1: SELECCIÓN DE TICKER Y FECHA
    # ==============================================================================
    st.header("1. Configuración Inicial")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Selector de Ticker
        ticker_options = ['SPX', 'SPY', 'QQQ']
        selected_ticker = st.selectbox(
            "Selecciona el Ticker",
            ticker_options,
            index=0,
            key='ticker_selector_triple'
        )
        st.info(f"📊 Ticker seleccionado: **{selected_ticker}**")
    
    with col2:
        # Selector de Fecha de Hoy (para los datos)
        fecha_hoy = st.date_input(
            "Fecha de Análisis (Datos)",
            value=date.today(),
            max_value=date.today(),
            key='fecha_hoy_triple'
        )
        st.info(f"📅 Fecha de Datos: **{fecha_hoy.strftime('%Y-%m-%d')}**")
    
    st.markdown("---")
    
    # ==============================================================================
    # SECCIÓN 2: CARGA Y PREPARACIÓN DE DATOS
    # ==============================================================================
    st.header("2. Carga y Preparación de Datos")
    
    # BOTÓN PARA FORZAR LA ACTUALIZACIÓN
    if st.button("🔄 Forzar Actualización (Limpiar Caché de Datos)", key='refresh_triple'):
        st.cache_data.clear()
        for key in list(st.session_state.keys()):
            if key not in ('config_df_triple', 'dte_front_days_triple', 'dte_back_days_triple', 
                          'ticker_selector_triple', 'fecha_hoy_triple', 'fecha_dte_triple', 'password_correct'): 
                del st.session_state[key]
        st.rerun()
    
    # Crear una clave única para los datos basada en el ticker
    datos_key = f'datos_calculados_{selected_ticker}'
    
    if datos_key not in st.session_state:
        
        with st.spinner(f"Descargando datos históricos de {selected_ticker} y calculando indicadores..."):
            # NOTA: Aquí necesitarías modificar fetch_data para aceptar el ticker como parámetro
            # df_raw = fetch_data(ticker=selected_ticker)
            df_raw = fetch_data()  # Por ahora usar el original
            spx = calculate_indicators(df_raw)
            endog_final, exog_tvtp_final = preparar_datos_markov(spx)

        if endog_final is None:
            st.error("❌ Error: No se pudieron preparar los datos para el análisis Markov.")
            return
        
        with st.spinner("Ejecutando modelos Markov K=2 y K=3..."):
            results_k2 = markov_calculation_k2(endog_final, exog_tvtp_final)
            results_k3 = markov_calculation_k3(endog_final, exog_tvtp_final)
        
        with st.spinner("Calculando indicador NR/WR..."):
            nr_wr_signal_on = calculate_nr_wr_signal(df_raw)
            nr_wr_series = calculate_nr_wr_signal_series(df_raw)
        
        st.session_state[datos_key] = {
            'df_raw': df_raw, 'spx': spx, 'endog_final': endog_final, 
            'exog_tvtp_final': exog_tvtp_final, 'results_k2': results_k2, 
            'results_k3': results_k3, 'nr_wr_signal_on': nr_wr_signal_on, 
            'nr_wr_series': nr_wr_series
        }
        st.success(f"✅ Cálculos completados para {selected_ticker}")
    
    else:
        st.info(f"ℹ️ Usando datos previamente calculados para {selected_ticker}")
    
    # Recuperar datos
    datos = st.session_state[datos_key]
    df_raw = datos['df_raw']
    spx = datos['spx']
    endog_final = datos['endog_final']
    results_k2 = datos['results_k2']
    results_k3 = datos['results_k3']
    nr_wr_signal_on = datos['nr_wr_signal_on']
    nr_wr_series = datos['nr_wr_series']
    
    st.dataframe(spx.tail(2))
    st.markdown("---")

    # ==============================================================================
    # SECCIÓN 3: INDICADOR NR/WR
    # ==============================================================================
    st.header("3. Indicador NR/WR (Narrow Range after Wide Range)")
    
    if nr_wr_signal_on:
        st.success("🟢 **SEÑAL NR/WR:** La compresión de volatilidad está **ACTIVA**.")
    else:
        st.info("⚪ **SEÑAL NR/WR:** La compresión de volatilidad está **INACTIVA**.")
    st.markdown("---")
    
    # ==============================================================================
    # SECCIÓN 4: MODELOS DE MARKOV
    # ==============================================================================
    st.header("4. Modelos de Markov")
    
    if 'error' in results_k2:
        st.error(f"❌ Error K=2: {results_k2['error']}")
        return
    if 'error' in results_k3:
        st.error(f"❌ Error K=3: {results_k3['error']}")
        return
    
    st.markdown(f"**Fecha del Último Cálculo:** {endog_final.index[-1].strftime('%Y-%m-%d')}")
    st.markdown("---")

    prob_k3_consolidada = results_k3['prob_baja'] + results_k3['prob_media']

    data_comparativa = {
        'Métrica': [
            'Probabilidad Baja (HOY)', 
            'Probabilidad Media (HOY)', 
            'Probabilidad Consolidada (Baja + Media)', 
            'Umbral de Señal de Entrada (70%)', 
            'Varianza Régimen Baja', 
            'Varianza Régimen Media', 
            'Varianza Régimen Alta', 
            'Umbral RV_5d Estimado'
        ],
        'K=2 (Original)': [
            f"{results_k2['prob_baja']:.4f}", 
            'N/A', 
            f"{results_k2['prob_baja']:.4f}", 
            f"{results_k2['UMBRAL_COMPRESION']:.2f}", 
            f"{results_k2['varianzas_regimen']['Baja']:.5f}", 
            'N/A', 
            f"{results_k2['varianzas_regimen']['Alta']:.5f}", 
            f"{results_k2['UMBRAL_RV5D_P_OBJETIVO']:.4f}"
        ],
        'K=3 (Propuesto)': [
            f"{results_k3['prob_baja']:.4f}", 
            f"{results_k3['prob_media']:.4f}", 
            f"**{prob_k3_consolidada:.4f}**", 
            f"{results_k3['UMBRAL_COMPRESION']:.2f}", 
            f"{results_k3['varianzas_regimen']['Baja']:.5f}", 
            f"{results_k3['varianzas_regimen']['Media']:.5f}", 
            f"{results_k3['varianzas_regimen']['Alta']:.5f}", 
            'Por Varianza'
        ]
    }

    df_comparativa = pd.DataFrame(data_comparativa)
    st.dataframe(df_comparativa, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Conclusión Operativa para Triple Calendar")

    if prob_k3_consolidada >= results_k3['UMBRAL_COMPRESION']:
        st.success(f"**✅ CONDICIONES FAVORABLES:** Probabilidad consolidada de **{prob_k3_consolidada:.4f}** (>0.70). Entorno apropiado para estrategias de venta de prima como Triple Calendar.")
    else:
        st.warning(f"**⚠️ PRECAUCIÓN:** Probabilidad consolidada de **{prob_k3_consolidada:.4f}** (<0.70). Considerar esperar mejores condiciones o ajustar la estrategia.")
    
    st.markdown("---")

    # ==============================================================================
    # SECCIÓN 5: SEMÁFORO GLOBAL
    # ==============================================================================
    st.header("5. Semáforo de Entrada - Triple Calendar 🚥")

    rv5d_ayer_val = spx["RV_5d"].iloc[-2]
    
    if 'config_df_triple' not in st.session_state:
        st.session_state['config_df_triple'] = get_default_config_df_triple_calendar(rv5d_ayer_val)

    rv5d_hoy = spx['RV_5d'].iloc[-1]
    rv5d_ayer = spx['RV_5d'].iloc[-2]
    
    metricas_actuales = {
        'r1_nr_wr': nr_wr_signal_on, 
        'r2_k2_70': results_k2['prob_baja'],
        'r3_k3_media_75': results_k3['prob_media'], 
        'r4_k3_baja_15': results_k3['prob_baja'],
        'r5_k3_consol_95': prob_k3_consolidada, 
        'r6_rv5d_10': rv5d_hoy,
        'r7_rv5d_menor': rv5d_hoy, 
    }
    
    st.markdown("##### Configuración de Reglas")

    st.button(
        "⚙️ Resetear a Valores por Defecto", 
        help="Restaura configuración optimizada para Triple Calendar", 
        on_click=reset_config_callback_triple, 
        args=(rv5d_ayer_val,),
        key='reset_triple'
    )

    df_config = st.session_state['config_df_triple'].copy()
    
    df_config['Valor Actual'] = df_config['ID'].apply(lambda id: 
        (metricas_actuales[id] and '🟢 ACTIVA' or '⚪ INACTIVA') if id == 'r1_nr_wr' else 
        f"{metricas_actuales[id]:.4f}"
    )

    col_config_all = {
        'Regla': st.column_config.TextColumn("Regla (Filtro)", disabled=True),
        'Operador': st.column_config.TextColumn("Op.", disabled=True, width="tiny"),
        'Umbral': st.column_config.TextColumn("Umbral"), 
        'Valor Actual': st.column_config.TextColumn("Valor Actual", disabled=True, width="small"),
        'Activa': st.column_config.CheckboxColumn("ON/OFF", width="small"),
        'ID': None
    }
    
    edited_df = st.data_editor(
        df_config,
        column_config=col_config_all,
        hide_index=True,
        use_container_width=True, 
        key='config_editor_triple'
    )
    
    st.session_state['config_df_triple'] = edited_df 
    
    st.markdown("---")
    
    if st.button("🚀 Recalcular Semáforo", key='calc_semaforo_triple'):
        calcular_y_mostrar_semaforo_triple(st.session_state['config_df_triple'], metricas_actuales, rv5d_ayer)
    
    st.markdown("### Tabla Consolidada de Análisis 🚦")
    
    if 'df_semaforo_body_triple' in st.session_state:
        df_body = st.session_state['df_semaforo_body_triple']
        df_footer = st.session_state['df_semaforo_footer_triple']
        senal_color = st.session_state['senal_color_triple']
        
        def color_cumple_body(row):
            styles = pd.Series('', index=row.index)
            
            if row['Cumple'] == 'SÍ':
                styles['Cumple'] = 'background-color: #008000; color: white'
            elif row['Cumple'] == 'NO':
                styles['Cumple'] = 'background-color: #8B0000; color: white'
            
            return styles

        styled_df_body = df_body.style.apply(color_cumple_body, axis=1)
        styled_df_body = styled_df_body.set_properties(**{'text-align': 'center'}, 
                                     subset=['Operador', 'Umbral', 'Valor Actual', 'Cumple'])
        
        st.dataframe(
            styled_df_body,
            hide_index=True,
            use_container_width=True,
            column_order=('Regla', 'Operador', 'Umbral', 'Valor Actual', 'Cumple'), 
            column_config={'ID': st.column_config.Column(disabled=True, width="tiny")} 
        )

        st.markdown("<br>", unsafe_allow_html=True) 

        footer_text = df_footer.iloc[0]['Regla']
        
        st.markdown(
            f"<div style='text-align: center; font-size: 1.2em; padding: 10px; border-radius: 5px; {senal_color}'>"
            f"**{footer_text}**" 
            f"</div>",
            unsafe_allow_html=True
        )

    else:
        st.info("Presione '🚀 Recalcular Semáforo' para ver el análisis completo.")

    st.markdown("---")

    # ==============================================================================
    # SECCIÓN 6: DTEs PARA TRIPLE CALENDAR (CON FECHA CONFIGURABLE)
    # ==============================================================================
    st.header("6. DTEs (Days To Expiration) - Triple Calendar")
    
    # Inicializar valores con 21 y 28 por defecto
    if 'dte_front_days_triple' not in st.session_state:
        st.session_state['dte_front_days_triple'] = 21
    if 'dte_back_days_triple' not in st.session_state:
        st.session_state['dte_back_days_triple'] = 28
    
    # Nueva fecha configurable para cálculo de DTEs (independiente de la fecha de datos)
    if 'fecha_dte_triple' not in st.session_state:
        st.session_state['fecha_dte_triple'] = date.today()
    
    col_fecha_dte = st.columns([1])[0]
    with col_fecha_dte:
        fecha_dte = st.date_input(
            "📅 Fecha Base para Cálculo de DTEs",
            value=st.session_state['fecha_dte_triple'],
            max_value=date.today() + timedelta(days=365),
            key='fecha_dte_input_triple',
            help="Esta fecha se usa como referencia para calcular las fechas de vencimiento"
        )
        st.session_state['fecha_dte_triple'] = fecha_dte
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        dte_front_days = st.number_input(
            "DTE Front (días)", 
            min_value=1, 
            max_value=365, 
            value=st.session_state['dte_front_days_triple'], 
            key='dte_front_input_triple'
        )
        st.session_state['dte_front_days_triple'] = dte_front_days

    with col2:
        dte_back_days = st.number_input(
            "DTE Back (días)", 
            min_value=1, 
            max_value=365, 
            value=st.session_state['dte_back_days_triple'], 
            key='dte_back_input_triple'
        )
        st.session_state['dte_back_days_triple'] = dte_back_days
        
    # Cálculo de fechas usando la fecha DTE seleccionada
    dte_front_date = fecha_dte + timedelta(days=dte_front_days)
    dte_back_date = fecha_dte + timedelta(days=dte_back_days)

    dte_data = {
        'Concepto': ['Fecha Base (Hoy)', 'DTE FRONT', 'DTE BACK', 'Rango de Días'],
        'Valor': [
            fecha_dte.strftime('%Y-%m-%d'), 
            dte_front_date.strftime('%Y-%m-%d'), 
            dte_back_date.strftime('%Y-%m-%d'),
            f"{dte_front_days} - {dte_back_days} días"
        ]
    }
    
    df_dte = pd.DataFrame(dte_data)
    
    st.markdown("---")
    st.dataframe(df_dte, hide_index=True, use_container_width=True)
    
    # Información adicional
    st.info(f"""
    📊 **Configuración Triple Calendar:**
    - Ticker: {selected_ticker}
    - Fecha Base: {fecha_dte.strftime('%Y-%m-%d')}
    - Front Month: ~{dte_front_days} días ({dte_front_date.strftime('%Y-%m-%d')})
    - Back Month: ~{dte_back_days} días ({dte_back_date.strftime('%Y-%m-%d')})
    - Diferencia: {dte_back_days - dte_front_days} días entre vencimientos
    """)
    
    st.markdown("---")
    
    # ==============================================================================
    # SECCIÓN 7: GRÁFICOS DE ANÁLISIS TÉCNICO
    # ==============================================================================
    st.header("7. Gráficos de Análisis Técnico Combinados")
    
    # --- CONTROLES DE FECHA PARA GRÁFICOS ---
    st.sidebar.header("⚙️ Configuración del Gráfico")
    fecha_final_grafico = spx.index[-1].date()
    st.sidebar.info(f"📅 Última fecha disponible: {fecha_final_grafico}")
    fecha_inicio_default_grafico = fecha_final_grafico - timedelta(days=90)

    fecha_inicio_grafico = st.sidebar.date_input(
        "Fecha de inicio:",
        value=fecha_inicio_default_grafico,
        min_value=spx.index[0].date(),
        max_value=fecha_final_grafico,
        key='fecha_inicio_grafico_triple'
    )

    # --- FILTRAR DATOS POR RANGO DE FECHAS ---
    fecha_inicio_dt_grafico = pd.to_datetime(fecha_inicio_grafico)
    fecha_final_dt_grafico = pd.to_datetime(fecha_final_grafico)

    spx_filtered = spx[(spx.index >= fecha_inicio_dt_grafico) & (spx.index <= fecha_final_dt_grafico)].copy()
    spx_filtered = spx_filtered[spx_filtered.index.dayofweek < 5]

    # --- PREPARACIÓN DE DATOS PARA GRÁFICO COMBINADO ---

    # Etiquetado del eje X
    date_labels = [d.strftime('%b %d') if i % 5 == 0 else '' for i, d in enumerate(spx_filtered.index)]
    date_labels[0] = spx_filtered.index[0].strftime('%b %d')
    date_labels[-1] = spx_filtered.index[-1].strftime('%b %d')

    spx_filtered['RV_5d_pct'] = spx_filtered['RV_5d'] * 100
    UMBRAL_RV = 0.10
    spx_filtered['RV_change'] = spx_filtered['RV_5d_pct'].diff()
    is_up = spx_filtered['RV_change'] >= 0

    prob_baja_serie_k2 = results_k2['prob_baja_serie'].loc[spx_filtered.index].fillna(method='ffill')

    prob_baja_serie_k3 = results_k3['prob_baja_serie'].loc[spx_filtered.index].fillna(method='ffill')
    prob_media_serie_k3 = results_k3['prob_media_serie'].loc[spx_filtered.index].fillna(method='ffill')
    prob_k3_consolidada_serie = prob_baja_serie_k3 + prob_media_serie_k3

    nr_wr_filtered = nr_wr_series.reindex(spx_filtered.index).fillna(0)

    UMBRAL_ALERTA = 0.50 
    UMBRAL_COMPRESION = results_k2['UMBRAL_COMPRESION']

    # Formato de fecha para el hover (DÍA-MES-AÑO)
    fechas_formateadas = spx_filtered.index.strftime('%d-%m-%Y').tolist()

    # --- CREAR SUBPLOTS (5 FILAS) ---
    fig_combined = make_subplots(
        rows=5, 
        cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.02, 
        row_heights=[0.45, 0.13, 0.14, 0.14, 0.14],
    )

    # ----------------------------------------------------
    # 1. GRÁFICO DE VELAS JAPONESAS (Fila 1)
    # ----------------------------------------------------

    hover_text_candles = [
        f"<b>{fecha}</b><br>Open: {o:.2f}<br>High: {h:.2f}<br>Low: {l:.2f}<br>Close: {c:.2f}"
        for fecha, o, h, l, c in zip(
            fechas_formateadas,
            spx_filtered['Open'],
            spx_filtered['High'],
            spx_filtered['Low'],
            spx_filtered['Close']
        )
    ]

    fig_combined.add_trace(go.Candlestick(
        x=list(range(len(spx_filtered))),
        open=spx_filtered['Open'],
        high=spx_filtered['High'],
        low=spx_filtered['Low'],
        close=spx_filtered['Close'],
        name=f'{selected_ticker}',
        text=hover_text_candles,
        hoverinfo='text',
        increasing=dict(line=dict(color='#00B06B')),
        decreasing=dict(line=dict(color='#F13A50'))
    ), row=1, col=1)

    fig_combined.update_yaxes(title_text='Precio', row=1, col=1)
    fig_combined.update_xaxes(showticklabels=False, row=1, col=1)

    # ----------------------------------------------------
    # 2. GRÁFICO DE VOLATILIDAD REALIZADA (RV_5d) (Fila 2)
    # ----------------------------------------------------

    for i in range(len(spx_filtered) - 1):
        color = '#00B06B' if is_up.iloc[i+1] else '#F13A50'
        
        fig_combined.add_trace(go.Scatter(
            x=[i, i+1],
            y=[spx_filtered['RV_5d_pct'].iloc[i], spx_filtered['RV_5d_pct'].iloc[i+1]],
            mode='lines',
            line=dict(color=color, width=2),
            showlegend=False,
            hoverinfo='skip'
        ), row=2, col=1)

    fig_combined.add_trace(go.Scatter(
        x=list(range(len(spx_filtered))),
        y=spx_filtered['RV_5d_pct'],
        mode='markers',
        marker=dict(size=0.1, color='rgba(0,0,0,0)'),
        name='RV',
        customdata=[[fecha] for fecha in fechas_formateadas],
        hovertemplate='<b>%{customdata[0]}</b><br>RV: %{y:.2f}%<extra></extra>',
        showlegend=True
    ), row=2, col=1)

    fig_combined.add_shape(
        type="line",
        x0=0, y0=UMBRAL_RV * 100,
        x1=len(spx_filtered) - 1, y1=UMBRAL_RV * 100,
        line=dict(color="orange", width=2, dash="dot"),
        layer="below",
        row=2, col=1
    )

    fig_combined.add_annotation(
        x=0, y=1.0, 
        text=f'Umbral RV: {UMBRAL_RV*100:.2f}%', 
        showarrow=False,
        xref='x2', yref='y2 domain', 
        xanchor='left', yanchor='top', 
        font=dict(size=12, color="orange"),
        xshift=5, yshift=-5, 
        row=2, col=1
    )

    fig_combined.update_yaxes(title_text='RV (%)', row=2, col=1, tickformat=".2f")
    fig_combined.update_xaxes(showticklabels=False, row=2, col=1) 

    # ----------------------------------------------------
    # 3. GRÁFICO DE MARKOV K=2 (Fila 3)
    # ----------------------------------------------------

    fig_combined.add_trace(go.Scatter(
        x=list(range(len(spx_filtered))),
        y=prob_baja_serie_k2,
        mode='lines',
        name='Prob. K=2 (Baja Vol.)', 
        line=dict(color='#8A2BE2', width=2),
        fill='tozeroy', 
        fillcolor='rgba(138, 43, 226, 0.3)',
        customdata=[[fecha] for fecha in fechas_formateadas],
        hovertemplate='<b>%{customdata[0]}</b><br>Prob. Baja K=2: %{y:.4f}<extra></extra>',
        showlegend=True 
    ), row=3, col=1)

    fig_combined.add_shape(
        type="line",
        x0=0, y0=UMBRAL_COMPRESION,
        x1=len(spx_filtered) - 1, y1=UMBRAL_COMPRESION,
        line=dict(color="#FFD700", width=2, dash="dash"), 
        layer="below",
        row=3, col=1
    )

    fig_combined.add_shape(
        type="line",
        x0=0, y0=UMBRAL_ALERTA,
        x1=len(spx_filtered) - 1, y1=UMBRAL_ALERTA,
        line=dict(color="#FFFFFF", width=1, dash="dot"),
        layer="below",
        row=3, col=1
    )

    fig_combined.add_annotation(
        x=0, 
        y=UMBRAL_COMPRESION, 
        text=f'Compresión Fuerte ({UMBRAL_COMPRESION*100:.0f}%)', 
        showarrow=False,
        xref='x3', yref='y3', 
        xanchor='left', 
        yanchor='bottom', 
        font=dict(size=12, color="#FFD700"),
        xshift=5, 
        yshift=5, 
        row=3, col=1
    )

    fig_combined.add_annotation(
        x=0, 
        y=UMBRAL_ALERTA, 
        text=f'Alerta ({UMBRAL_ALERTA*100:.0f}%)', 
        showarrow=False,
        xref='x3', yref='y3', 
        xanchor='left', 
        yanchor='bottom', 
        font=dict(size=12, color="#FFFFFF"), 
        xshift=5, 
        yshift=5,
        row=3, col=1
    )

    fig_combined.update_yaxes(title_text='Prob. K=2', row=3, col=1, tickformat=".2f", range=[0, 1])
    fig_combined.update_xaxes(showticklabels=False, row=3, col=1) 

    # ----------------------------------------------------
    # 4. GRÁFICO DE MARKOV K=3 (Fila 4)
    # ----------------------------------------------------

    fig_combined.add_trace(go.Scatter(
        x=list(range(len(spx_filtered))),
        y=prob_k3_consolidada_serie,
        mode='lines',
        name='Prob. K=3 (Baja+Media)', 
        line=dict(color='#00FF7F', width=2),
        fill='tozeroy', 
        fillcolor='rgba(0, 255, 127, 0.3)',
        customdata=[[fecha] for fecha in fechas_formateadas],
        hovertemplate='<b>%{customdata[0]}</b><br>Prob. Consolidada K=3: %{y:.4f}<extra></extra>',
        showlegend=True 
    ), row=4, col=1)

    fig_combined.add_shape(
        type="line",
        x0=0, y0=UMBRAL_COMPRESION,
        x1=len(spx_filtered) - 1, y1=UMBRAL_COMPRESION,
        line=dict(color="#FFD700", width=2, dash="dash"), 
        layer="below",
        row=4, col=1
    )

    fig_combined.add_shape(
        type="line",
        x0=0, y0=UMBRAL_ALERTA,
        x1=len(spx_filtered) - 1, y1=UMBRAL_ALERTA,
        line=dict(color="#FFFFFF", width=1, dash="dot"),
        layer="below",
        row=4, col=1
    )

    fig_combined.add_annotation(
        x=0, 
        y=UMBRAL_COMPRESION, 
        text=f'Compresión Fuerte ({UMBRAL_COMPRESION*100:.0f}%)', 
        showarrow=False,
        xref='x4', yref='y4', 
        xanchor='left', 
        yanchor='bottom', 
        font=dict(size=12, color="#FFD700"),
        xshift=5, 
        yshift=5, 
        row=4, col=1
    )

    fig_combined.add_annotation(
        x=0, 
        y=UMBRAL_ALERTA, 
        text=f'Alerta ({UMBRAL_ALERTA*100:.0f}%)', 
        showarrow=False,
        xref='x4', yref='y4', 
        xanchor='left', 
        yanchor='bottom', 
        font=dict(size=12, color="#FFFFFF"), 
        xshift=5, 
        yshift=5,
        row=4, col=1
    )

    fig_combined.update_yaxes(title_text='Prob. K=3', row=4, col=1, tickformat=".2f", range=[0, 1])
    fig_combined.update_xaxes(showticklabels=False, row=4, col=1)

    # ----------------------------------------------------
    # 5. GRÁFICO DE SEÑAL NR/WR (Fila 5)
    # ----------------------------------------------------

    fig_combined.add_trace(go.Bar(
        x=list(range(len(spx_filtered))),
        y=nr_wr_filtered,
        name='Señal NR/WR', 
        marker=dict(
            color='#FF6B35',
            line=dict(width=0)
        ),
        customdata=[[fecha, 'ACTIVA' if s > 0 else 'INACTIVA'] for fecha, s in zip(fechas_formateadas, nr_wr_filtered)],
        hovertemplate='<b>%{customdata[0]}</b><br>NR/WR: %{customdata[1]}<extra></extra>',
        showlegend=True,
        width=0.8
    ), row=5, col=1)

    fig_combined.add_shape(
        type="line",
        x0=-0.5, y0=0.5,
        x1=len(spx_filtered) - 0.5, y1=0.5,
        line=dict(color="#AAAAAA", width=1, dash="dot"),
        layer="below",
        row=5, col=1
    )

    fig_combined.add_annotation(
        x=0, 
        y=0.9, 
        text='COMPRESIÓN ACTIVA', 
        showarrow=False,
        xref='x5', yref='y5', 
        xanchor='left', 
        yanchor='top', 
        font=dict(size=11, color="#FF6B35"),
        xshift=5, 
        yshift=-5,
        row=5, col=1
    )

    fig_combined.update_yaxes(title_text='NR/WR', row=5, col=1, range=[0, 1.05], tickvals=[0, 1], ticktext=['OFF', 'ON'])

    # --- CONFIGURACIÓN FINAL DEL GRÁFICO COMBINADO ---
    fig_combined.update_layout(
        template='plotly_dark',
        height=1100, 
        xaxis_rangeslider_visible=False,
        hovermode='x', 
        plot_bgcolor='#131722', 
        paper_bgcolor='#131722', 
        font=dict(color='#AAAAAA'),
        margin=dict(t=50, b=100, l=60, r=40),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1, 
            xanchor="left",
            x=0.01, 
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
            font=dict(size=10)
        )
    )

    # ----------------------------------------------------------------------------------
    # CONFIGURACIÓN DE SPIKE POR DEFECTO
    # ----------------------------------------------------------------------------------

    for i in range(1, 6):
        fig_combined.update_xaxes(
            showspikes=True,
            spikemode='across', 
            spikesnap='cursor',
            spikecolor='#AAAAAA',
            spikethickness=1,
            spikedash='dash',
            row=i, 
            col=1
        )

        fig_combined.update_yaxes(
            showspikes=False,
            row=i, 
            col=1
        )

    # ----------------------------------------------------------------------------------
    # CONFIGURACIONES DE EJE X (Estética)
    # ----------------------------------------------------------------------------------

    fig_combined.update_xaxes(
        tickmode='array',
        tickvals=list(range(len(spx_filtered))),
        ticktext=date_labels,
        tickangle=-45,
        row=5, col=1, 
        showgrid=False
    )

    fig_combined.update_xaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=1, col=1)
    fig_combined.update_yaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=1, col=1)
    fig_combined.update_xaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=2, col=1)
    fig_combined.update_yaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=2, col=1)
    fig_combined.update_xaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=3, col=1)
    fig_combined.update_yaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=3, col=1)
    fig_combined.update_xaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=4, col=1)
    fig_combined.update_yaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=4, col=1)
    fig_combined.update_xaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=5, col=1)
    fig_combined.update_yaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=5, col=1)

    st.plotly_chart(fig_combined, use_container_width=True)

    # --- INFORMACIÓN ADICIONAL ---
    st.markdown("---")
    col1, col2, col3, col4, col5, col6 = st.columns(6) 

    with col1:
        st.metric("Precio Actual", f"${spx_filtered['Close'].iloc[-1]:.2f}")
    with col2:
        cambio = spx_filtered['Close'].iloc[-1] - spx_filtered['Close'].iloc[0]
        cambio_pct = (cambio / spx_filtered['Close'].iloc[0]) * 100
        st.metric(f"Cambio ({fecha_inicio_grafico} al {fecha_final_grafico})", f"${cambio:.2f}", f"{cambio_pct:.2f}%")
    with col3:
        st.metric("Máximo", f"${spx_filtered['High'].max():.2f}")
    with col4:
        st.metric("Mínimo", f"${spx_filtered['Low'].min():.2f}")
    with col5:
        rv_latest = spx_filtered['RV_5d'].iloc[-1] * 100
        st.metric("RV_5d (Último)", f"{rv_latest:.2f}%")
    with col6:
        nr_wr_status = "🟢 ACTIVA" if nr_wr_filtered.iloc[-1] > 0 else "⚪ INACTIVA"
        st.metric("Señal NR/WR", nr_wr_status)
    
# ==============================================================================
# PUNTO DE ENTRADA PROTEGIDO
# ==============================================================================

if __name__ == "__main__":
    
    if check_password():
        main_triple_calendar()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder a Triple Calendar.")
