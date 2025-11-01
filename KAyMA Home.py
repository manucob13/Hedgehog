# home.py
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
    # === AÑADIR LA FUNCIÓN DE AUTENTICACIÓN ===
    check_password 
)

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="KAyMA 1.1", layout="wide")

# ==============================================================================
# LÓGICA DE CONFIGURACIÓN Y VALORES POR DEFECTO (NO ES NECESARIO TOCAR ESTO)
# ==============================================================================

def get_default_config_df(rv5d_ayer_val):
    """Genera el DataFrame de configuración de reglas con los valores por defecto."""
    
    # VALORES POR DEFECTO FINALES SOLICITADOS
    default_config_data = {
        'Regla': ['1. Señal NR/WR Activa', '2. Prob. K=2 Baja Vol.', '3. Prob. K=3 Media Vol.', '4. Prob. K=3 Baja Vol.', '5. Prob. K=3 Consolidada', '6. RV_5d Actual', f'7. RV_5d HOY vs. AYER ({rv5d_ayer_val:.4f})'],
        'Operador': ['==', '>=', '>=', '>=', '>=', '<=', '<'],
        # Umbrales
        'Umbral': ['ON', '0.9000', '0.7500', '0.1500', '0.9500', '0.1000', 'RV_AYER'], 
        # Activación (R7 ahora en OFF por defecto)
        'Activa': [True, True, False, False, True, True, False], 
        'ID': ['r1_nr_wr', 'r2_k2_70', 'r3_k3_media_75', 'r4_k3_baja_15', 'r5_k3_consol_95', 'r6_rv5d_10', 'r7_rv5d_menor']
    }
    return pd.DataFrame(default_config_data)

def reset_config_callback(rv5d_ayer_val):
    """Callback para el botón de reset: Restaura la configuración de reglas y elimina el semáforo calculado."""
    st.session_state['config_df'] = get_default_config_df(rv5d_ayer_val)
    # Eliminar el estado calculado del semáforo para que se fuerce un nuevo cálculo/visualización.
    for key in ['df_semaforo_body', 'df_semaforo_footer', 'senal_color']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# ==============================================================================
# LÓGICA DE CÁLCULO DEL SEMÁFORO (Separada para el botón)
# ==============================================================================

def calcular_y_mostrar_semaforo(df_config, metricas_actuales, rv5d_ayer):
    """Calcula el estado de cada regla y el resultado global del Semáforo (Lógica Conjunta AND)."""
    
    df_config_calc = df_config.copy()
    
    # Convertir umbrales a float donde sea posible para cálculo
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
            if isinstance(umbral_calc, (float, int)):
                if operador == '>=':
                    regla_cumplida = metrica_actual >= umbral_calc
                elif operador == '<=':
                    regla_cumplida = metrica_actual <= umbral_calc

        # Actualizar columna 'Cumple'
        if row['Activa']: # Solo si la regla está activa
            if regla_cumplida:
                df_config_calc.loc[index, 'Cumple'] = "SÍ"
            else:
                df_config_calc.loc[index, 'Cumple'] = "NO"
        else:
             df_config_calc.loc[index, 'Cumple'] = "INACTIVA"


        # Evaluación de la Señal Global (Lógica Conjunta: AND)
        if row['Activa']:
            num_reglas_activas += 1
            if not regla_cumplida:
                senal_entrada_global_interactiva = False

    # --- Creación de la Tabla de Presentación Final (Cuerpo) ---
    
    # Se incluye la columna 'Activa' para que la función de estilo pueda leer su estado
    df_presentacion = df_config_calc[['Activa', 'Regla', 'Operador', 'Umbral', 'Valor Actual', 'Cumple', 'ID']].copy()
    
    
    # Determinar el resultado global y el color del semáforo
    if num_reglas_activas == 0:
        res_final_texto = "INACTIVA (0 Reglas Activas)"
        senal_color = "background-color: #AAAAAA; color: black"
    elif senal_entrada_global_interactiva:
        res_final_texto = "" # Vacío para señal ACTIVA
        senal_color = "background-color: #008000; color: white" # Verde
    else:
        # Vacío también para señal DENEGADA
        res_final_texto = "" 
        senal_color = "background-color: #8B0000; color: white" # Rojo
        
    # Crear la fila de resumen (Semáforo Global)
    fila_resumen = pd.DataFrame([{
        'Regla': '🚥 SEMÁFORO GLOBAL HEDGEHOG 🚥', 
        'ID': 'FINAL' 
    }])
    
    # Guardar los DataFrames separados para visualización
    st.session_state['df_semaforo_body'] = df_presentacion # Las filas de las reglas
    st.session_state['df_semaforo_footer'] = fila_resumen # La fila final
    st.session_state['senal_color'] = senal_color


# ==============================================================================
# FUNCIÓN PRINCIPAL (CONTENIDO DE LA APP)
# ==============================================================================

def main_comparison():
    
    # --- TÍTULO PRINCIPAL CON ICONO Y TAMAÑO MODIFICADO (Erizo) ---
    st.markdown("<h1><span style='font-size: 1.5em;'>🦔</span> HEDGEHOG 1.1 Modelos de Volatilidad - Markov-Switching K=2-3 - NR/WR</h1>", unsafe_allow_html=True)
    st.markdown("""
    Esta herramienta ejecuta y compara dos modelos de Regresión de Markov sobre la Volatilidad Realizada ($\text{RV}_{5d}$) 
    del S&P 500 y añade la señal de compresión **NR/WR (Narrow Range after Wide Range)** como indicador auxiliar.
    """)
    st.markdown("---")
    
    st.header("1. Carga y Preparación de Datos")
    
    # BOTÓN PARA FORZAR LA ACTUALIZACIÓN
    if st.button("🔄 Forzar Actualización (Limpiar Caché de Datos)"):
        st.cache_data.clear()
        for key in list(st.session_state.keys()):
            # Excluir las variables de entrada del punto 5 y la configuración
            if key not in ('config_df', 'dte_front_days', 'dte_back_days', 'password_correct'): 
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
    
    # --------------------------------------------------------------------------
    # Conclusión Operativa y Entendiendo la Diferencia
    # --------------------------------------------------------------------------
    st.markdown("---")
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
    st.markdown("---")
    # --------------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # 4. LÓGICA HEDGEHOG Y SEMÁFORO GLOBAL 🚥 (UNIFICADO)
    # ----------------------------------------------------------------------
    st.header("4. Lógica HEDGEHOG y Semáforo Global 🚥")

    # --- 1. Inicializar la lógica de configuración en session_state ---
    rv5d_ayer_val = spx["RV_5d"].iloc[-2]
    
    if 'config_df' not in st.session_state:
        st.session_state['config_df'] = get_default_config_df(rv5d_ayer_val)

    # --- 2. Extracción de Métricas Clave y Valores ---
    rv5d_hoy = spx['RV_5d'].iloc[-1]
    rv5d_ayer = spx['RV_5d'].iloc[-2]
    
    metricas_actuales = {
        'r1_nr_wr': nr_wr_signal_on, 'r2_k2_70': results_k2['prob_baja'],
        'r3_k3_media_75': results_k3['prob_media'], 'r4_k3_baja_15': results_k3['prob_baja'],
        'r5_k3_consol_95': prob_k3_consolidada, 'r6_rv5d_10': rv5d_hoy,
        'r7_rv5d_menor': rv5d_hoy, 
    }
    
    # --- 3. CONFIGURACIÓN DE TODAS LAS REGLAS (DATA EDITOR UNIFICADO) ---
    st.markdown("##### Configuración de Reglas (NR/WR, Volatilidad y Markov)")

    # Botón de Reset
    st.button(
        "⚙️ Resetear a Valores por Defecto", 
        help="Restaura la configuración de reglas a los umbrales predefinidos.", 
        on_click=reset_config_callback, 
        args=(rv5d_ayer_val,)
    )

    df_config = st.session_state['config_df'].copy()
    
    # Calculate 'Valor Actual' for ALL rules before editor
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
        key='config_editor_all'
    )
    
    st.session_state['config_df'] = edited_df 
    
    
    # --------------------------------------------------------------------------
    # --- C. BOTÓN DE CÁLCULO Y TABLA CONSOLIDADA ---
    # --------------------------------------------------------------------------
    
    st.markdown("---")
    
    # BOTÓN DE CÁLCULO EXPLÍCITO
    if st.button("🚀 Recalcular Semáforo Consolidado"):
        calcular_y_mostrar_semaforo(st.session_state['config_df'], metricas_actuales, rv5d_ayer)
    
    st.markdown("### Tabla Consolidada de Lógica y Resultado 🚦")
    
    # Mostrar la tabla consolidada solo si ya se ha calculado
    if 'df_semaforo_body' in st.session_state:
        df_body = st.session_state['df_semaforo_body']
        df_footer = st.session_state['df_semaforo_footer']
        senal_color = st.session_state['senal_color']
        
        # 1. Función para dar formato de color del Cuerpo (Body)
        def color_cumple_body(row):
            styles = pd.Series('', index=row.index)
            
            # Aplica color si la regla fue calculada (SÍ o NO)
            if row['Cumple'] == 'SÍ':
                styles['Cumple'] = 'background-color: #008000; color: white'
            elif row['Cumple'] == 'NO':
                styles['Cumple'] = 'background-color: #8B0000; color: white'
            
            return styles

        # 2. Estilizar y MOSTRAR el Cuerpo de la tabla
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

        # 3. AÑADIR ESPACIO Y MOSTRAR EL PIE (FOOTER) como barra de color SIN ENCABEZADOS
        st.markdown("<br>", unsafe_allow_html=True) 

        footer_text = df_footer.iloc[0]['Regla'] # "🚥 SEMÁFORO GLOBAL HEDGEHOG 🚥"
        
        # Usamos markdown para crear una barra de color sólida y limpia
        st.markdown(
            f"<div style='text-align: center; font-size: 1.2em; padding: 10px; border-radius: 5px; {senal_color}'>"
            f"**{footer_text}**" 
            f"</div>",
            unsafe_allow_html=True
        )

    else:
        st.info("Presione '🚀 Recalcular Semáforo Consolidado' para ver la lógica aplicada.")

    st.markdown("---")
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # 5. DTEs
    # ----------------------------------------------------------------------
    st.header("5. DTEs (Days To Expiration)")
    
    # Inicializar valores de entrada en session_state con 7 y 14 por defecto
    if 'dte_front_days' not in st.session_state:
        st.session_state['dte_front_days'] = 7
    if 'dte_back_days' not in st.session_state:
        st.session_state['dte_back_days'] = 14 # Por defecto 14 días
    
    # Asegurar que los number_input reflejen los defaults correctos
    col1, col2 = st.columns(2)
    
    with col1:
        dte_front_days = st.number_input(
            "DTE Front (días)", 
            min_value=1, 
            max_value=365, 
            value=st.session_state['dte_front_days'], 
            key='dte_front_input'
        )
        st.session_state['dte_front_days'] = dte_front_days

    with col2:
        dte_back_days = st.number_input(
            "DTE Back (días)", 
            min_value=1, 
            max_value=365, 
            value=st.session_state['dte_back_days'], 
            key='dte_back_input'
        )
        st.session_state['dte_back_days'] = dte_back_days
        
    # Cálculo de fechas
    today = date.today()
    dte_front_date = today + timedelta(days=dte_front_days)
    dte_back_date = today + timedelta(days=dte_back_days)

    # Creación de la tabla
    dte_data = {
        'Métrica': ['Fecha de Hoy', 'DTE FRONT', 'DTE BACK'],
        'Valor': [
            today.strftime('%Y-%m-%d'), 
            dte_front_date.strftime('%Y-%m-%d'), 
            dte_back_date.strftime('%Y-%m-%d')
        ]
    }
    
    df_dte = pd.DataFrame(dte_data)
    
    st.markdown("---")
    st.dataframe(df_dte, hide_index=True, use_container_width=True)
    st.markdown("---")
    
# ==============================================================================
# PUNTO DE ENTRADA PROTEGIDO
# ==============================================================================

if __name__ == "__main__":
    
    # LLAMADA AL LOGIN (Muestra el formulario si es necesario)
    if check_password():
        # SI EL LOGIN ES EXITOSO, EJECUTA LA APP PRINCIPAL
        main_comparison()
    else:
        # Esto es solo si quieres que algo se muestre ANTES del login si falla, 
        # pero con check_password() el formulario de login se muestra en el sidebar
        # y el resto de la página queda vacío hasta que se loguee.
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder a la aplicación.")
