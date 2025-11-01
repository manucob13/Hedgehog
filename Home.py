# home.py
import streamlit as st
import pandas as pd
# Asegúrate de que tu archivo utils.py esté en el mismo directorio.
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
            # Evitar borrar la configuración si existe
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
    # 4. CONFIGURACIÓN DINÁMICA DE SEÑAL Y EVALUACIÓN FINAL (ESTRUCTURADO)
    # ----------------------------------------------------------------------
    st.header("4. Configuración Dinámica de Señal (HEDGEHOG) ⚙️")

    # --- 1. Inicializar la lógica de configuración en session_state ---
    
    rv5d_ayer_val = spx["RV_5d"].iloc[-2]
    
    # Estructura inicial de la configuración (valores por defecto)
    default_config_df = pd.DataFrame({
        'Regla': [
            '1. Señal NR/WR Activa', 
            '2. Prob. K=2 Baja Vol.', 
            '3. Prob. K=3 Media Vol.', 
            '4. Prob. K=3 Baja Vol.', 
            '5. Prob. K=3 Consolidada', 
            '6. RV_5d Actual',
            f'7. RV_5d HOY vs. AYER ({rv5d_ayer_val:.4f})'
        ],
        'Operador': ['==', '>=', '>=', '>=', '>=', '<=', '<'],
        'Umbral': ['ON', 0.70, 0.75, 0.15, 0.95, 0.10, 'RV_AYER'], # Valor editable
        'Activa': [True, True, True, True, True, True, True],
        'ID': ['r1_nr_wr', 'r2_k2_70', 'r3_k3_media_75', 'r4_k3_baja_15', 'r5_k3_consol_95', 'r6_rv5d_10', 'r7_rv5d_menor']
    })
    
    if 'config_df' not in st.session_state:
        st.session_state['config_df'] = default_config_df

    # --- 2. Extracción de Métricas Clave y Valores ---
    prob_k2_baja = results_k2['prob_baja']
    prob_k3_baja = results_k3['prob_baja']
    prob_k3_media = results_k3['prob_media']
    prob_k3_consolidada = prob_k3_baja + prob_k3_media

    rv5d_hoy = spx['RV_5d'].iloc[-1]
    rv5d_ayer = spx['RV_5d'].iloc[-2]
    
    # Define la métrica actual (Valores reales para la comparación)
    metricas_actuales = {
        'r1_nr_wr': nr_wr_signal_on, 
        'r2_k2_70': prob_k2_baja,
        'r3_k3_media_75': prob_k3_media,
        'r4_k3_baja_15': prob_k3_baja,
        'r5_k3_consol_95': prob_k3_consolidada,
        'r6_rv5d_10': rv5d_hoy,
        'r7_rv5d_menor': rv5d_hoy, 
    }
    
    # --- 3. Pre-procesar el DataFrame para la Interfaz (añadir Valor Actual) ---
    df_config_display = st.session_state['config_df'].copy()
    
    df_config_display['Valor Actual'] = df_config_display['ID'].apply(
        lambda id: "🟢 ACTIVA" if id == 'r1_nr_wr' and metricas_actuales[id] else 
                   "⚪ INACTIVA" if id == 'r1_nr_wr' else 
                   f"{metricas_actuales[id]:.4f}"
    )

    # --- 4. Mostrar la tabla interactiva y capturar los cambios ---
    
    col_config = {
        'Regla': st.column_config.TextColumn("Regla (Filtro)", disabled=True),
        'Operador': st.column_config.TextColumn("Operador", disabled=True),
        'Umbral': st.column_config.NumberColumn("Umbral", format="%.4f", min_value=0.0, max_value=1.0),
        'Valor Actual': st.column_config.TextColumn("Valor Actual", disabled=True),
        'Activa': st.column_config.CheckboxColumn("ON/OFF"),
        'ID': None # Ocultar
    }
    
    edited_df = st.data_editor(
        df_config_display,
        column_config=col_config,
        hide_index=True,
        use_container_width=True,
        key='config_editor_final'
    )
    
    # Guardar los cambios de Umbral y Activa de vuelta al state
    st.session_state['config_df'] = edited_df
    
    # --- 5. Recalcular la Señal Global y la Columna 'Cumple' ---
    
    senal_entrada_global_interactiva = True
    num_reglas_activas = 0
    df_config_display['Cumple'] = "NO" # Inicializar columna

    for index, row in edited_df.iterrows():
        rule_id = row['ID']
        metrica_actual = metricas_actuales[rule_id]
        operador = row['Operador']
        # Convertir a cadena de texto para manejar 'ON', 'OFF', 'RV_AYER'
        umbral_str = str(row['Umbral']).upper()
        
        regla_cumplida = False
        
        # Lógica de Cumplimiento
        if rule_id == 'r1_nr_wr': # BOOLEAN - NR/WR
            if umbral_str == 'ON':
                regla_cumplida = metrica_actual # True (ACTIVA)
            elif umbral_str == 'OFF':
                regla_cumplida = not metrica_actual # False (INACTIVA)
            else:
                # Si el usuario cambia el Umbral a un número, lo tratamos como 'ON' si es > 0
                try:
                    umbral_val = float(row['Umbral'])
                    regla_cumplida = metrica_actual and (umbral_val > 0)
                except ValueError:
                    regla_cumplida = metrica_actual # Por defecto, si es distinto de OFF, es ON.
        
        elif rule_id == 'r7_rv5d_menor': # COMPARACIÓN RV_AYER
            regla_cumplida = metrica_actual < rv5d_ayer
            
        elif umbral_str not in ('RV_AYER', 'ON', 'OFF'): # FLOAT (Probabilidades, RV_5d)
            try:
                umbral_float = float(row['Umbral'])
                if operador == '>=':
                    regla_cumplida = metrica_actual >= umbral_float
                elif operador == '<=':
                    regla_cumplida = metrica_actual <= umbral_float
            except ValueError:
                regla_cumplida = False # Falla si el umbral no es un número y debería serlo

        # Actualizar columna 'Cumple'
        if regla_cumplida:
            df_config_display.loc[index, 'Cumple'] = "SÍ"
        else:
            df_config_display.loc[index, 'Cumple'] = "NO"

        # Evaluación de la Señal Global
        if row['Activa']:
            num_reglas_activas += 1
            if not regla_cumplida:
                senal_entrada_global_interactiva = False

    # --- 6. Mostrar la Tabla con el Resultado de Cumplimiento ---
    
    # Creamos una versión de la tabla solo para visualización con formato de color
    df_final_display = df_config_display[['Activa', 'Regla', 'Umbral', 'Valor Actual', 'Cumple']].copy()

    # Función para dar formato de color
    def color_cumple(val):
        color = 'background-color: #008000; color: white' if val == 'SÍ' else 'background-color: #8B0000; color: white'
        return color

    st.markdown("### Estado Actual de las Reglas:")
    st.dataframe(
        df_final_display.style.applymap(color_cumple, subset=['Cumple']),
        hide_index=True,
        use_container_width=True
    )
    
    # --- 7. Conclusión Final en Recuadro Destacado ---
    st.markdown("---")
    st.subheader("Resultado Final del Sistema HEDGEHOG 🎯")
    
    with st.container(border=True):
        if num_reglas_activas == 0:
            st.info("ℹ️ **NO HAY REGLAS ACTIVAS.** Active al menos una regla en la columna ON/OFF para evaluar la señal.")
        elif senal_entrada_global_interactiva:
            st.success(f"🎉 **¡SEÑAL DE ENTRADA ACTIVA!** Se cumplen todas las **{num_reglas_activas}** reglas activas actualmente.")
        else:
            st.error(f"❌ **SEÑAL DE ENTRADA DENEGADA.** No se cumplen todas las **{num_reglas_activas}** reglas activas. Revise la columna 'Cumple'.")

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
