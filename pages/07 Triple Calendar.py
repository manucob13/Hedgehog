# pages/triple_calendar.py
import streamlit as st
import pandas as pd
from datetime import date, timedelta
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
        # Selector de Fecha de Hoy
        fecha_hoy = st.date_input(
            "Fecha de Análisis",
            value=date.today(),
            max_value=date.today(),
            key='fecha_hoy_triple'
        )
        st.info(f"📅 Fecha: **{fecha_hoy.strftime('%Y-%m-%d')}**")
    
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
                          'ticker_selector_triple', 'fecha_hoy_triple', 'password_correct'): 
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
    spx = datos['spx']
    endog_final = datos['endog_final']
    results_k2 = datos['results_k2']
    results_k3 = datos['results_k3']
    nr_wr_signal_on = datos['nr_wr_signal_on']
    
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
    # SECCIÓN 6: DTEs PARA TRIPLE CALENDAR
    # ==============================================================================
    st.header("6. DTEs (Days To Expiration) - Triple Calendar")
    
    # Inicializar valores con 21 y 28 por defecto
    if 'dte_front_days_triple' not in st.session_state:
        st.session_state['dte_front_days_triple'] = 21
    if 'dte_back_days_triple' not in st.session_state:
        st.session_state['dte_back_days_triple'] = 28
    
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
        
    # Cálculo de fechas usando la fecha seleccionada
    dte_front_date = fecha_hoy + timedelta(days=dte_front_days)
    dte_back_date = fecha_hoy + timedelta(days=dte_back_days)

    dte_data = {
        'Concepto': ['Fecha de Análisis', 'DTE FRONT', 'DTE BACK', 'Rango de Días'],
        'Valor': [
            fecha_hoy.strftime('%Y-%m-%d'), 
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
    - Front Month: ~{dte_front_days} días ({dte_front_date.strftime('%Y-%m-%d')})
    - Back Month: ~{dte_back_days} días ({dte_back_date.strftime('%Y-%m-%d')})
    - Diferencia: {dte_back_days - dte_front_days} días entre vencimientos
    """)
    
    st.markdown("---")
    
# ==============================================================================
# PUNTO DE ENTRADA PROTEGIDO
# ==============================================================================

if __name__ == "__main__":
    
    if check_password():
        main_triple_calendar()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder a Triple Calendar.")
