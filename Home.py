# ----------------------------------------------------------------------
    # 4. CONFIGURACIÓN DINÁMICA DE SEÑAL Y EVALUACIÓN FINAL (ESTRUCTURADO)
    # ----------------------------------------------------------------------
    st.header("4. Configuración Dinámica de Señal (HEDGEHOG) ⚙️")

    # --- 1. Inicializar la lógica de configuración en session_state ---
    
    # Estructura inicial de la configuración (valores por defecto)
    # Se usa un DataFrame simple para data_editor, que es más limpio.
    default_config_df = pd.DataFrame({
        'Regla': [
            '1. Señal NR/WR Activa', 
            '2. Prob. K=2 Baja Vol.', 
            '3. Prob. K=3 Media Vol.', 
            '4. Prob. K=3 Baja Vol.', 
            '5. Prob. K=3 Consolidada', 
            '6. RV_5d Actual',
            f'7. RV_5d HOY vs. AYER ({spx["RV_5d"].iloc[-2]:.4f})'
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
        umbral_str = str(row['Umbral'])
        
        regla_cumplida = False
        
        # Lógica de Cumplimiento
        if rule_id == 'r1_nr_wr': # BOOLEAN - NR/WR
            if umbral_str == 'ON':
                regla_cumplida = metrica_actual # True/False del indicador
            elif umbral_str == 'OFF':
                regla_cumplida = not metrica_actual
            else:
                 # Si el usuario pone algo que no es ON/OFF
                regla_cumplida = metrica_actual # Por defecto, si es ON en Umbral se compara con True
        
        elif rule_id == 'r7_rv5d_menor': # COMPARACIÓN RV_AYER
            regla_cumplida = metrica_actual < rv5d_ayer
            
        elif umbral_str not in ('RV_AYER', 'ON', 'OFF'): # FLOAT (Probabilidades, RV_5d)
            umbral_float = float(umbral_str)
            if operador == '>=':
                regla_cumplida = metrica_actual >= umbral_float
            elif operador == '<=':
                regla_cumplida = metrica_actual <= umbral_float

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
