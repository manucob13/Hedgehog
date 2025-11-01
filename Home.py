# home.py
import streamlit as st
import pandas as pd
from utils import (
    fetch_data, 
    calculate_indicators, 
    preparar_datos_markov,
    calculate_nr_wr_signal,
    calculate_nr_wr_signal_series,
    markov_calculation_k2,
    markov_calculation_k3
)

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="HEDGEHOG 1.1", layout="wide")

# --- TÍTULO PRINCIPAL CON ICONO Y TAMAÑO MODIFICADO (Erizo) ---
st.markdown("<h1><span style='font-size: 1.5em;'>🦔</span> HEDGEHOG 1.1 Modelos de Volatilidad - Markov-Switching K=2-3 - NR/WR</h1>", unsafe_allow_html=True)
st.markdown("""
Esta herramienta ejecuta y compara dos modelos de Regresión de Markov sobre la Volatilidad Realizada ($\text{RV}_{5d}$) 
del S&P 500 y añade la señal de compresión **NR/WR (Narrow Range after Wide Range)** como indicador auxiliar.
""")

# ==============================================================================
# LÓGICA DE CÁLCULO DEL SEMÁFORO (Separada para el botón)
# ==============================================================================

def calcular_y_mostrar_semaforo(df_config, metricas_actuales, rv5d_ayer):
    """Calcula el estado de cada regla y el resultado global del Semáforo."""
    
    df_config_calc = df_config.copy()
    
    # Convertir umbrales a float donde sea posible para cálculo
    def safe_float_convert(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return value

    df_config_calc['Umbral_Calc'] = df_config_calc['Umbral'].apply(safe_float_convert)
    
    # 💥 CORRECCIÓN CRUCIAL: Añadir la columna 'Valor Actual' ANTES de usarla
    df_config_calc['Valor Actual'] = df_config_calc['ID'].apply(lambda id: 
        (metricas_actuales[id] and '🟢 ACTIVA' or '⚪ INACTIVA') if id == 'r1_nr_wr' else 
        f"{metricas_actuales[id]:.4f}"
    )

    senal_entrada_global_interactiva = True
    num_reglas_activas = 0
    df_config_calc['Cumple'] = 'NO' # Inicializar columna
    
    # Itera sobre el DataFrame de configuración para calcular el cumplimiento
    for index, row in df_config_calc.iterrows():
        rule_id = row['ID']
        metrica_actual = metricas_actuales[rule_id]
        operador = row['Operador']
        umbral_calc = row['Umbral_Calc']
        umbral_str = str(row['Umbral']).upper()
        regla_cumplida = False
        
        # Lógica de Cumplimiento
        if row['ID'] == 'r1_nr_wr':
            if umbral_str == 'ON':
                regla_cumplida = metrica_actual 
            elif umbral_str == 'OFF':
                regla_cumplida = not metrica_actual
        
        elif row['ID'] == 'r7_rv5d_menor':
            regla_cumplida = metrica_actual < rv5d_ayer
            
        else: # Reglas de probabilidad y RV_5d (FLOAT)
            if isinstance(umbral_calc, float):
                if operador == '>=':
                    regla_cumplida = metrica_actual >= umbral_calc
                elif operador == '<=':
                    regla_cumplida = metrica_actual <= umbral_calc

        # Actualizar columna 'Cumple'
        if regla_cumplida:
            df_config_calc.loc[index, 'Cumple'] = "SÍ"
        else:
            df_config_calc.loc[index, 'Cumple'] = "NO"

        # Evaluación de la Señal Global
        if row['Activa']:
            num_reglas_activas += 1
            if not regla_cumplida:
                senal_entrada_global_interactiva = False

    # --- Creación de la Tabla de Presentación Final ---
    
    # Ya tenemos todas las columnas necesarias, incluyendo 'Valor Actual' y 'Cumple'
    df_presentacion = df_config_calc[['Activa', 'Regla', 'Operador', 'Umbral', 'Valor Actual', 'Cumple', 'ID']].copy()

    # Quitar el texto de la columna 'Activa' si es False
    df_presentacion['Activa'] = df_presentacion['Activa'].apply(lambda x: '✓' if x else '')
    
    # Determinar el resultado global y el color del semáforo
    if num_reglas_activas == 0:
        res_final = "INACTIVA (0 Reglas Activas)"
        senal_color = "background-color: #AAAAAA; color: black"
    elif senal_entrada_global_interactiva:
        # Simplificación solicitada: Solo "SEÑAL ACTIVA"
        res_final = "SEÑAL ACTIVA" 
        senal_color = "background-color: #008000; color: white" # Verde
    else:
        num_reglas_fallidas = num_reglas_activas - sum(df_config_calc.loc[df_config_calc['Activa'], 'Cumple'] == 'SÍ')
        res_final = f"SEÑAL DENEGADA (X {num_reglas_fallidas} de {num_reglas_activas} Fallaron)"
        senal_color = "background-color: #8B0000; color: white" # Rojo
        
    # Crear la fila de resumen (Semáforo Global)
    fila_resumen = pd.DataFrame([{
        'Activa': 'TOTAL', 
        'Regla': '🚥 SEMÁFORO GLOBAL HEDGEHOG 🚥', 
        'Operador': 'ALL', 
        'Umbral': '-', 
        'Valor Actual': '-', 
        'Cumple': res_final,
        'ID': 'FINAL' 
    }])
    
    df_final_display_con_resumen = pd.concat([df_presentacion, fila_resumen], ignore_index=True)
    
    # Guardar el DataFrame final para visualización
    st.session_state['df_semaforo_final'] = df_final_display_con_resumen
    st.session_state['senal_color'] = senal_color


# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================

def main_comparison():
    
    st.header("1. Carga y Preparación de Datos")
    
    # BOTÓN PARA FORZAR LA ACTUALIZACIÓN
    if st.button("🔄 Forzar Actualización (Limpiar Caché de Datos)"):
        st.cache_data.clear()
        for key in list(st.session_state.keys()):
            if key not in ('config_df'): 
                del st.session_state[key]
        st.rerun()
    
    # --- VERIFICAR SI YA EXISTEN LOS DATOS EN SESSION_STATE ---
    if 'datos_calculados' not in st.session_state:
        
        with st.spinner("Descargando datos históricos y calculando indicadores..."):
            df_raw = fetch_data()
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
        
        st.session_state['datos_calculados'] = {
            'df_raw': df_raw, 'spx': spx, 'endog_final': endog_final, 
            'exog_tvtp_final': exog_tvtp_final, 'results_k2': results_k2, 
            'results_k3': results_k3, 'nr_wr_signal_on': nr_wr_signal_on, 
            'nr_wr_series': nr_wr_series
        }
        st.success("✅ Todos los cálculos completados y guardados en memoria.")
    
    else:
        st.info("ℹ️ Usando datos previamente calculados (ya están en memoria).")
    
    # --- Recuperar datos de session_state ---
    datos = st.session_state['datos_calculados']
    spx = datos['spx']
    endog_final = datos['endog_final']
    results_k2 = datos['results_k2']
    results_k3 = datos['results_k3']
    nr_wr_signal_on = datos['nr_wr_signal_on']
    
    st.dataframe(spx.tail(2))
    st.markdown("---")

    st.header("2. Indicador NR/WR (Narrow Range after Wide Range)")
    
    if nr_wr_signal_on:
        st.success("🟢 **SEÑAL NR/WR:** La compresión de volatilidad está **ACTIVA**. Alta probabilidad de ruptura inminente.")
    else:
        st.info("⚪ **SEÑAL NR/WR:** La compresión de volatilidad está **INACTIVA**. La volatilidad puede ser normal o ya ha explotado.")
    st.markdown("---")
    
    st.header("3. Modelos de Markov")
    
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
        'Métrica': ['Probabilidad Baja (HOY)', 'Probabilidad Media (HOY)', 'Probabilidad Consolidada (Baja + Media)', 'Umbral de Señal de Entrada (70%)', 'Varianza Régimen Baja', 'Varianza Régimen Media', 'Varianza Régimen Alta', 'Umbral RV_5d Estimado (Para el Régimen Baja)'],
        'K=2 (Original)': [f"{results_k2['prob_baja']:.4f}", 'N/A (No existe)', f"{results_k2['prob_baja']:.4f}", f"{results_k2['UMBRAL_COMPRESION']:.2f}", f"{results_k2['varianzas_regimen']['Baja']:.5f}", 'N/A (No existe)', f"{results_k2['varianzas_regimen']['Alta']:.5f}", f"{results_k2['UMBRAL_RV5D_P_OBJETIVO']:.4f}"],
        'K=3 (Propuesto)': [f"{results_k3['prob_baja']:.4f}", f"{results_k3['prob_media']:.4f}", f"**{prob_k3_consolidada:.4f}**", f"{results_k3['UMBRAL_COMPRESION']:.2f}", f"{results_k3['varianzas_regimen']['Baja']:.5f}", f"{results_k3['varianzas_regimen']['Media']:.5f}", f"{results_k3['varianzas_regimen']['Alta']:.5f}", 'Determinado por Varianza']
    }

    df_comparativa = pd.DataFrame(data_comparativa)
    st.dataframe(df_comparativa, hide_index=True, use_container_width=True)
    st.markdown("---")

    # ----------------------------------------------------------------------
    # 4. LÓGICA HEDGEHOG Y SEMÁFORO GLOBAL 🚥 (UNIFICADO)
    # ----------------------------------------------------------------------
    st.header("4. Lógica HEDGEHOG y Semáforo Global 🚥")

    # --- 1. Inicializar la lógica de configuración en session_state ---
    rv5d_ayer_val = spx["RV_5d"].iloc[-2]
    
    default_config_data = {
        'Regla': ['1. Señal NR/WR Activa', '2. Prob. K=2 Baja Vol.', '3. Prob. K=3 Media Vol.', '4. Prob. K=3 Baja Vol.', '5. Prob. K=3 Consolidada', '6. RV_5d Actual', f'7. RV_5d HOY vs. AYER ({rv5d_ayer_val:.4f})'],
        'Operador': ['==', '>=', '>=', '>=', '>=', '<=', '<'],
        'Umbral': ['ON', 0.70, 0.75, 0.15, 0.95, 0.10, 'RV_AYER'], 
        'Activa': [True, True, True, True, True, True, True],
        'ID': ['r1_nr_wr', 'r2_k2_70', 'r3_k3_media_75', 'r4_k3_baja_15', 'r5_k3_consol_95', 'r6_rv5d_10', 'r7_rv5d_menor']
    }
    
    if 'config_df' not in st.session_state:
        st.session_state['config_df'] = pd.DataFrame(default_config_data)

    # --- 2. Extracción de Métricas Clave y Valores ---
    rv5d_hoy = spx['RV_5d'].iloc[-1]
    rv5d_ayer = spx['RV_5d'].iloc[-2]
    
    metricas_actuales = {
        'r1_nr_wr': nr_wr_signal_on, 'r2_k2_70': results_k2['prob_baja'],
        'r3_k3_media_75': results_k3['prob_media'], 'r4_k3_baja_15': results_k3['prob_baja'],
        'r5_k3_consol_95': prob_k3_consolidada, 'r6_rv5d_10': rv5d_hoy,
        'r7_rv5d_menor': rv5d_hoy, 
    }
    
    # --- 3. División de Reglas para Manipulación ---
    df_config = st.session_state['config_df'].copy()
    
    df_nr_wr = df_config[df_config['ID'] == 'r1_nr_wr'].iloc[0]
    df_reglas_editables = df_config[df_config['ID'] != 'r1_nr_wr'].reset_index(drop=True)
    
    
    # --------------------------------------------------------------------------
    # --- A. CONFIGURACIÓN DE LA REGLA 1 (SELECTBOX) ---
    # --------------------------------------------------------------------------
    st.markdown("##### Regla 1: Señal NR/WR (Selectbox)")
    
    col_r1, col_op, col_umbral, col_actual, col_activa = st.columns([4, 1, 2, 2, 1])
    
    with col_r1:
        st.markdown(f"**{df_nr_wr['Regla']}**")
    with col_op:
        st.markdown(df_nr_wr['Operador'])
    with col_umbral:
        umbral_r1 = st.selectbox(
            label='Umbral R1', options=['ON', 'OFF'],
            index=0 if df_nr_wr['Umbral'] == 'ON' else 1,
            key='umbral_r1_select', label_visibility='collapsed'
        )
    with col_actual:
        st.markdown(f"<div style='text-align: center; padding-top: 5px;'>**{metricas_actuales['r1_nr_wr'] and '🟢 ACTIVA' or '⚪ INACTIVA'}**</div>", unsafe_allow_html=True)
    with col_activa:
        activa_r1 = st.checkbox(
            label='Activa R1', value=df_nr_wr['Activa'],
            key='activa_r1_check', label_visibility='collapsed'
        )
        
    df_config.loc[df_config['ID'] == 'r1_nr_wr', 'Umbral'] = umbral_r1
    df_config.loc[df_config['ID'] == 'r1_nr_wr', 'Activa'] = activa_r1
    
    # --------------------------------------------------------------------------
    # --- B. CONFIGURACIÓN DE REGLAS 2-7 (DATA EDITOR) ---
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("##### Reglas 2-7: Volatilidad y Markov (Umbral Editable)")

    col_config_2_7 = {
        'Regla': st.column_config.TextColumn("Regla (Filtro)", disabled=True),
        'Operador': st.column_config.TextColumn("Op.", disabled=True, width="tiny"),
        'Umbral': st.column_config.NumberColumn("Umbral", format="%.4f", min_value=0.0, max_value=1.0, width="small"),
        'Valor Actual': st.column_config.TextColumn("Valor Actual", disabled=True, width="small"),
        'Activa': st.column_config.CheckboxColumn("ON/OFF", width="small"),
        'ID': None
    }
    
    # Aquí solo se añade 'Valor Actual' a la versión editable temporal
    df_reglas_editables['Valor Actual'] = df_reglas_editables['ID'].apply(lambda id: f"{metricas_actuales[id]:.4f}")
    
    edited_df_2_7 = st.data_editor(
        df_reglas_editables, 
        column_config=col_config_2_7,
        hide_index=True,
        use_container_width=False, 
        key='config_editor_2_7'
    )
    
    df_config.loc[df_config['ID'] != 'r1_nr_wr', ['Umbral', 'Activa']] = edited_df_2_7[['Umbral', 'Activa']].values
    st.session_state['config_df'] = df_config 
    
    
    # --------------------------------------------------------------------------
    # --- C. BOTÓN DE CÁLCULO Y TABLA CONSOLIDADA ---
    # --------------------------------------------------------------------------
    
    st.markdown("---")
    
    # BOTÓN DE CÁLCULO EXPLÍCITO
    if st.button("🚀 Recalcular Semáforo Consolidado"):
        calcular_y_mostrar_semaforo(st.session_state['config_df'], metricas_actuales, rv5d_ayer)
    
    st.markdown("### Tabla Consolidada de Lógica y Resultado 🚦")
    
    # Mostrar la tabla consolidada solo si ya se ha calculado
    if 'df_semaforo_final' in st.session_state:
        df_final_display_con_resumen = st.session_state['df_semaforo_final']
        senal_color = st.session_state['senal_color']
        
        # Función para dar formato de color
        def color_cumple(row):
            styles = pd.Series('', index=row.index)
            
            if row['ID'] == 'FINAL':
                styles[:] = senal_color
            elif row['Cumple'] == 'SÍ':
                styles['Cumple'] = 'background-color: #008000; color: white'
            else:
                styles['Cumple'] = 'background-color: #8B0000; color: white'
                
            return styles

        styled_df = df_final_display_con_resumen.style.apply(color_cumple, axis=1)

        # Usamos CSS para centrar el texto en las celdas
        styled_df = styled_df.set_properties(**{'text-align': 'center'}, 
                                            subset=['Activa', 'Operador', 'Umbral', 'Valor Actual', 'Cumple'])
        
        st.dataframe(
            styled_df,
            hide_index=True,
            use_container_width=True,
            column_order=('Activa', 'Regla', 'Operador', 'Umbral', 'Valor Actual', 'Cumple'),
            column_config={'ID': st.column_config.Column(disabled=True, width="tiny")} 
        )
    else:
        st.info("Presione '🚀 Recalcular Semáforo Consolidado' para ver la lógica aplicada.")

    st.markdown("---")
    # ----------------------------------------------------------------------
    
    # --- SECCIÓN DE CONCLUSIÓN K=3 ---
    st.subheader("Conclusión Operativa")

    if prob_k3_consolidada >= results_k3['UMBRAL_COMPRESION']:
        st.success(f"**SEÑAL DE ENTRADA FUERTE (K=3):** El riesgo de Alta Volatilidad es bajo. La probabilidad consolidada es **{prob_k3_consolidada:.4f}**, mayor de 0.70. Condición Favorable para estrategias de Theta.")
    else:
        st.warning(f"**RIESGO ACTIVO (K=3):** La probabilidad consolidada es **{prob_k3_consolidada:.4f}**, menor de 0.70. El Régimen de Alta Volatilidad ha tomado peso. Evitar entrar o considerar salir.")
    
    st.markdown("""
    ---
    ### Entendiendo la Diferencia Clave
    
    El **Modelo K=2** combina toda la volatilidad no-crisis en una única señal de 'Baja', lo que le hace propenso a **falsos positivos**.
    
    El **Modelo K=3** descompone la 'Baja' volatilidad en dos estados: 'Baja' (Calma Extrema) y 'Media' (Consolidación). 
    
    La **Probabilidad Consolidada (Baja + Media)** del K=3 ofrece una señal de entrada/salida más robusta: solo da luz verde cuando la suma de los dos estados favorables supera el 70%, actuando como un **filtro más estricto contra el ruido** que el K=2 ignora.
    """)


if __name__ == "__main__":
    main_comparison()
