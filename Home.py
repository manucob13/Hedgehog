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
# FUNCIÓN PRINCIPAL
# ==============================================================================

def main_comparison():
    
    st.header("1. Carga y Preparación de Datos")
    
    # BOTÓN PARA FORZAR LA ACTUALIZACIÓN
    if st.button("🔄 Forzar Actualización (Limpiar Caché de Datos)"):
        st.cache_data.clear()
        # También limpiar session_state
        for key in list(st.session_state.keys()):
            # Evitar borrar la configuración del gráfico si ya existe
            if key != 'config_senal': 
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

        st.success(f"✅ Descarga y preparación exitosa. Datos listos para el análisis ({len(endog_final)} puntos).")
        
        # --- EJECUTAR CÁLCULOS PESADOS UNA SOLA VEZ ---
        with st.spinner("Ejecutando modelos Markov K=2 y K=3..."):
            results_k2 = markov_calculation_k2(endog_final, exog_tvtp_final)
            results_k3 = markov_calculation_k3(endog_final, exog_tvtp_final)
        
        with st.spinner("Calculando indicador NR/WR..."):
            nr_wr_signal_on = calculate_nr_wr_signal(df_raw)
            nr_wr_series = calculate_nr_wr_signal_series(df_raw)
        
        # --- GUARDAR TODO EN SESSION_STATE ---
        st.session_state['datos_calculados'] = {
            'df_raw': df_raw,
            'spx': spx,
            'endog_final': endog_final,
            'exog_tvtp_final': exog_tvtp_final,
            'results_k2': results_k2,
            'results_k3': results_k3,
            'nr_wr_signal_on': nr_wr_signal_on,
            'nr_wr_series': nr_wr_series
        }
        
        st.success("✅ Todos los cálculos completados y guardados en memoria.")
    
    else:
        st.info("ℹ️ Usando datos previamente calculados (ya están en memoria).")
        # Recuperar datos
        datos = st.session_state['datos_calculados']
        df_raw = datos['df_raw']
        spx = datos['spx']
        endog_final = datos['endog_final']
        results_k2 = datos['results_k2']
        results_k3 = datos['results_k3']
        nr_wr_signal_on = datos['nr_wr_signal_on']
    
    # --- MOSTRAR VISTA PREVIA ---
    spx = st.session_state['datos_calculados']['spx']
    endog_final = st.session_state['datos_calculados']['endog_final']
    results_k2 = st.session_state['datos_calculados']['results_k2']
    results_k3 = st.session_state['datos_calculados']['results_k3']
    nr_wr_signal_on = st.session_state['datos_calculados']['nr_wr_signal_on']
    
    st.dataframe(spx.tail(2))
    st.markdown("---")

    # --- INDICADOR NR/WR ---
    st.header("2. Indicador NR/WR (Narrow Range after Wide Range)")
    
    if nr_wr_signal_on:
        st.success("🟢 **SEÑAL NR/WR:** La compresión de volatilidad está **ACTIVA**. Alta probabilidad de ruptura inminente.")
    else:
        st.info("⚪ **SEÑAL NR/WR:** La compresión de volatilidad está **INACTIVA**. La volatilidad puede ser normal o ya ha explotado.")
    st.markdown("---")
    
    st.header("3. Modelos de Markov")
    
    # Verificación de resultados
    if 'error' in results_k2:
        st.error(f"❌ Error K=2: {results_k2['error']}")
        return
    if 'error' in results_k3:
        st.error(f"❌ Error K=3: {results_k3['error']}")
        return
    
    st.markdown(f"**Fecha del Último Cálculo:** {endog_final.index[-1].strftime('%Y-%m-%d')}")
    st.markdown("---")

    # Calculo de la probabilidad consolidada K=3
    prob_k3_consolidada = results_k3['prob_baja'] + results_k3['prob_media']

    # Crear DataFrame para la tabla de comparación
    data_comparativa = {
        'Métrica': [
            'Probabilidad Baja (HOY)',
            'Probabilidad Media (HOY)',
            'Probabilidad Consolidada (Baja + Media)',
            'Umbral de Señal de Entrada (70%)',
            'Varianza Régimen Baja',
            'Varianza Régimen Media',
            'Varianza Régimen Alta',
            'Umbral RV_5d Estimado (Para el Régimen Baja)'
        ],
        'K=2 (Original)': [
            f"{results_k2['prob_baja']:.4f}",
            'N/A (No existe)',
            f"{results_k2['prob_baja']:.4f}",
            f"{results_k2['UMBRAL_COMPRESION']:.2f}",
            f"{results_k2['varianzas_regimen']['Baja']:.5f}",
            'N/A (No existe)',
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
            'Determinado por Varianza'
        ]
    }

    df_comparativa = pd.DataFrame(data_comparativa)

    st.dataframe(df_comparativa, hide_index=True, use_container_width=True)

    st.markdown("---")

    # ----------------------------------------------------------------------
    # 4. CONFIGURACIÓN DINÁMICA DE SEÑAL Y EVALUACIÓN FINAL
    # ----------------------------------------------------------------------
    st.header("4. Configuración Dinámica de Señal (HEDGEHOG)")

    # --- 1. Inicializar la lógica de configuración en session_state ---
    
    # Estructura inicial de la configuración (valores por defecto)
    default_config = {
        'r1_nr_wr': {'Activa': True, 'Umbral': 'ON', 'Tipo': 'BOOL'},
        'r2_k2_70': {'Activa': True, 'Umbral': 0.70, 'Tipo': 'FLOAT'},
        'r3_k3_media_75': {'Activa': True, 'Umbral': 0.75, 'Tipo': 'FLOAT'},
        'r4_k3_baja_15': {'Activa': True, 'Umbral': 0.15, 'Tipo': 'FLOAT'},
        'r5_k3_consol_95': {'Activa': True, 'Umbral': 0.95, 'Tipo': 'FLOAT'},
        'r6_rv5d_10': {'Activa': True, 'Umbral': 0.10, 'Tipo': 'FLOAT'},
        'r7_rv5d_menor': {'Activa': True, 'Umbral': 'RV_AYER', 'Tipo': 'COMPARISON'},
    }
    
    if 'config_senal' not in st.session_state:
        st.session_state['config_senal'] = default_config

    # --- 2. Extracción de Métricas Clave y Valores ---
    prob_k2_baja = results_k2['prob_baja']
    prob_k3_baja = results_k3['prob_baja']
    prob_k3_media = results_k3['prob_media']
    prob_k3_consolidada = prob_k3_baja + prob_k3_media

    rv5d_hoy = spx['RV_5d'].iloc[-1]
    rv5d_ayer = spx['RV_5d'].iloc[-2]
    
    # Define la métrica actual que se compara con el umbral (Valor Real)
    metricas_actuales = {
        'r1_nr_wr': nr_wr_signal_on, 
        'r2_k2_70': prob_k2_baja,
        'r3_k3_media_75': prob_k3_media,
        'r4_k3_baja_15': prob_k3_baja,
        'r5_k3_consol_95': prob_k3_consolidada,
        'r6_rv5d_10': rv5d_hoy,
        'r7_rv5d_menor': rv5d_hoy, 
    }

    # Define el operador de comparación y el valor a mostrar en la tabla
    operadores_y_descripciones = {
        'r1_nr_wr': {'Op': '==', 'Desc': '1. Señal NR/WR Activa'},
        'r2_k2_70': {'Op': '>=', 'Desc': '2. Prob. K=2 Baja Vol. (70%)'},
        'r3_k3_media_75': {'Op': '>=', 'Desc': '3. Prob. K=3 Media Vol. (75%)'},
        'r4_k3_baja_15': {'Op': '>=', 'Desc': '4. Prob. K=3 Baja Vol. (15%)'},
        'r5_k3_consol_95': {'Op': '>=', 'Desc': '5. Prob. K=3 Consolidada (95%)'},
        'r6_rv5d_10': {'Op': '<=', 'Desc': '6. RV_5d Actual (0.10)'},
        'r7_rv5d_menor': {'Op': '<', 'Desc': f'7. RV_5d HOY vs. AYER ({rv5d_ayer:.4f})'},
    }

    # --- 3. Construcción de la Tabla Interactiva (Layout de Columnas) ---
    
    col_desc, col_op, col_umbral, col_actual, col_cumple, col_activa = st.columns([3, 1.5, 2.5, 2.5, 1.5, 1.5])

    # Encabezados de la tabla
    col_desc.markdown('**Regla (Filtro)**')
    col_op.markdown('**Operador**')
    col_umbral.markdown('**Umbral**')
    col_actual.markdown('**Valor Actual**')
    col_cumple.markdown('**Cumple**')
    col_activa.markdown('**ON/OFF**')
    st.markdown("---") # Separador para los encabezados

    # Variables para la evaluación global
    senal_entrada_global_interactiva = True
    num_reglas_activas = 0

    # Iterar y crear las filas de la tabla
    for rule_id, config in st.session_state['config_senal'].items():
        # Extracción de valores
        metrica_actual = metricas_actuales[rule_id]
        operador = operadores_y_descripciones[rule_id]['Op']
        descripcion = operadores_y_descripciones[rule_id]['Desc']

        with col_desc:
            st.markdown(descripcion)

        with col_op:
            st.markdown(operador)

        # --- Campo de Umbral Editable ---
        with col_umbral:
            if config['Tipo'] == 'FLOAT':
                # Input numérico para probabilidades y RV_5d
                nuevo_umbral = st.number_input(
                    label=' ',
                    value=config['Umbral'],
                    step=0.001,
                    min_value=0.0,
                    max_value=1.0,
                    format="%.4f",
                    key=f'umbral_{rule_id}',
                    label_visibility='collapsed'
                )
                st.session_state['config_senal'][rule_id]['Umbral'] = nuevo_umbral

            elif config['Tipo'] == 'BOOL':
                # Input de texto simple para ON/OFF (no editable)
                st.text_input(
                    label=' ',
                    value='ON',
                    disabled=True,
                    key=f'umbral_{rule_id}',
                    label_visibility='collapsed'
                )
                
            elif config['Tipo'] == 'COMPARISON':
                 # Input de texto deshabilitado para comparación
                st.text_input(
                    label=' ',
                    value='RV_AYER',
                    disabled=True,
                    key=f'umbral_{rule_id}',
                    label_visibility='collapsed'
                )
        
        # --- Cálculo de la Regla (Resultado) ---
        regla_cumplida = False
        
        # Lógica de Cumplimiento
        if config['Tipo'] == 'FLOAT':
            umbral = st.session_state['config_senal'][rule_id]['Umbral']
            if operador == '>=':
                regla_cumplida = metrica_actual >= umbral
            elif operador == '<=':
                regla_cumplida = metrica_actual <= umbral
        
        elif config['Tipo'] == 'BOOL':
            regla_cumplida = metrica_actual # Ya es True/False
            
        elif config['Tipo'] == 'COMPARISON':
            regla_cumplida = metrica_actual < rv5d_ayer

        # --- Columna Valor Actual ---
        with col_actual:
            if rule_id == 'r1_nr_wr':
                st.markdown("🟢 ACTIVA" if metrica_actual else "⚪ INACTIVA")
            else:
                st.markdown(f"{metrica_actual:.4f}")

        # --- Columna Cumple ---
        with col_cumple:
            if regla_cumplida:
                st.success('SÍ')
            else:
                st.error('NO')

        # --- Checkbox ON/OFF (Activa) ---
        with col_activa:
            activo = st.checkbox(
                label=' ',
                value=config['Activa'],
                key=f'activa_{rule_id}',
                label_visibility='collapsed'
            )
            st.session_state['config_senal'][rule_id]['Activa'] = activo

        # --- Evaluación de la Señal Global ---
        if activo:
            num_reglas_activas += 1
            if not regla_cumplida:
                senal_entrada_global_interactiva = False
        
        st.markdown("---") # Separador entre reglas

    # --- 5. Conclusión Final ---
    st.subheader("Resultado Final del Sistema HEDGEHOG")
    
    if num_reglas_activas == 0:
        st.info("ℹ️ No hay reglas activas. Active al menos una regla para evaluar la señal.")
    elif senal_entrada_global_interactiva:
        st.success(f"✨ **¡SEÑAL DE ENTRADA ACTIVA!** Se cumplen las **{num_reglas_activas}** reglas activas actualmente.")
    else:
        st.error(f"🛑 **SEÑAL DE ENTRADA DENEGADA.** No se cumplen todas las **{num_reglas_activas}** reglas activas. Revise la tabla y los umbrales.")

    st.markdown("---")
    # ----------------------------------------------------------------------
    # FIN DE LA NUEVA SECCIÓN
    # ----------------------------------------------------------------------

    # Mostrar la conclusión operativa (original, ahora solo texto explicativo)
    st.subheader("Conclusión Operativa (Original)")

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
