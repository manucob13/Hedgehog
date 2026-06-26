# home.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
from utils.utils import (
    fetch_data, 
    calculate_indicators, 
    preparar_datos_markov,
    calculate_nr_wr_signal,
    calculate_nr_wr_signal_series,
    markov_calculation_k2,
    check_password 
)


# ==============================================================================
# LÓGICA DE CONFIGURACIÓN Y VALORES POR DEFECTO
# ==============================================================================

def get_default_config_df(rv5d_ayer_val):
    """Genera el DataFrame de configuración de reglas con los valores por defecto."""
    
    default_config_data = {
        'Regla': [
            '1. Señal NR/WR Activa', 
            '2. Prob. K=2 Baja Vol.', 
            '3. RV_5d Actual', 
            f'4. RV_5d HOY vs. AYER ({rv5d_ayer_val:.4f})'
        ],
        'Operador': ['==', '>=', '<=', '<'],
        'Umbral': [
            'ON', 
            '0.9000', 
            '0.1500', 
            'RV_AYER'
        ], 
        'Activa': [True, True, True, False],
        'ID': ['r1_nr_wr', 'r2_k2_70', 'r6_rv5d_10', 'r7_rv5d_menor']
    }
    return pd.DataFrame(default_config_data)


def reset_config_callback(rv5d_ayer_val):
    """Callback para el botón de reset."""
    st.session_state['config_df'] = get_default_config_df(rv5d_ayer_val)
    for key in ['df_semaforo_body', 'df_semaforo_footer', 'senal_color']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


# ==============================================================================
# LÓGICA DE CÁLCULO DEL SEMÁFORO
# ==============================================================================

def calcular_y_mostrar_semaforo(df_config, metricas_actuales, rv5d_ayer):
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
        'Regla': '🚥 SEMÁFORO GLOBAL HEDGEHOG 🚥', 
        'ID': 'FINAL' 
    }])
    
    st.session_state['df_semaforo_body'] = df_presentacion
    st.session_state['df_semaforo_footer'] = fila_resumen
    st.session_state['senal_color'] = senal_color


# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================

def main_comparison():
    
    st.markdown("<h1><span style='font-size: 1.5em;'>🦔</span> Time Edge Signals v 1.1 Mod. Vola - Markov-Switching K=2 - NR/WR</h1>", unsafe_allow_html=True)
    st.markdown("""
    Esta herramienta ejecuta el modelo de Regresión de Markov K=2 sobre la Volatilidad Realizada ($\\text{RV}_{5d}$) 
    del S&P 500 y añade la señal de compresión **NR/WR (Narrow Range after Wide Range)** como indicador auxiliar.
    """)
    st.markdown("---")
    
    st.header("1.1 Carga y Preparación de Datos")
    
    if st.button("🔄 Forzar Actualización (Limpiar Caché de Datos)"):
        st.cache_data.clear()
        for key in list(st.session_state.keys()):
            if key not in ('config_df', 'password_correct'): 
                del st.session_state[key]
        st.rerun()
    
    if 'datos_calculados' not in st.session_state:
        
        with st.spinner("Descargando datos históricos y calculando indicadores..."):
            df_raw = fetch_data()
            spx = calculate_indicators(df_raw)
            endog_final, exog_tvtp_final = preparar_datos_markov(spx)

        if endog_final is None:
            st.error("❌ Error: No se pudieron preparar los datos para el análisis Markov.")
            return
        
        with st.spinner("Ejecutando modelo Markov K=2..."):
            results_k2 = markov_calculation_k2(endog_final, exog_tvtp_final)
        
        with st.spinner("Calculando indicador NR/WR..."):
            nr_wr_signal_on = calculate_nr_wr_signal(df_raw)
            nr_wr_series = calculate_nr_wr_signal_series(df_raw)
        
        st.session_state['datos_calculados'] = {
            'df_raw': df_raw, 'spx': spx, 'endog_final': endog_final, 
            'exog_tvtp_final': exog_tvtp_final, 'results_k2': results_k2,
            'nr_wr_signal_on': nr_wr_signal_on, 
            'nr_wr_series': nr_wr_series
        }
        st.success("✅ Todos los cálculos completados y guardados en memoria.")
    
    else:
        st.info("ℹ️ Usando datos previamente calculados (ya están en memoria).")
    
    datos = st.session_state['datos_calculados']
    spx = datos['spx']
    endog_final = datos['endog_final']
    results_k2 = datos['results_k2']
    nr_wr_signal_on = datos['nr_wr_signal_on']
    
    st.dataframe(spx.tail(2))
    st.markdown("---")

    st.header("1.2 Indicador NR/WR (Narrow Range after Wide Range)")
    
    if nr_wr_signal_on:
        st.success("🟢 **SEÑAL NR/WR:** La compresión de volatilidad está **ACTIVA**. Alta probabilidad de ruptura inminente.")
    else:
        st.info("⚪ **SEÑAL NR/WR:** La compresión de volatilidad está **INACTIVA**. La volatilidad puede ser normal o ya ha explotado.")
    st.markdown("---")
    
    st.header("1.3 Modelo de Markov K=2")
    
    if 'error' in results_k2:
        st.error(f"❌ Error K=2: {results_k2['error']}")
        return
    
    st.markdown(f"**Fecha del Último Cálculo:** {endog_final.index[-1].strftime('%Y-%m-%d')}")

    data_k2 = {
        'Métrica': [
            'Probabilidad Baja Vol. (HOY)', 
            'Umbral de Señal de Entrada (70%)', 
            'Varianza Régimen Baja', 
            'Varianza Régimen Alta', 
            'Umbral RV_5d Estimado (Régimen Baja)'
        ],
        'K=2': [
            f"{results_k2['prob_baja']:.4f}",
            f"{results_k2['UMBRAL_COMPRESION']:.2f}",
            f"{results_k2['varianzas_regimen']['Baja']:.5f}",
            f"{results_k2['varianzas_regimen']['Alta']:.5f}",
            f"{results_k2['UMBRAL_RV5D_P_OBJETIVO']:.4f}"
        ]
    }

    df_k2 = pd.DataFrame(data_k2)
    st.dataframe(df_k2, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Conclusión Operativa")

    if results_k2['prob_baja'] >= results_k2['UMBRAL_COMPRESION']:
        st.success(f"**SEÑAL DE ENTRADA (K=2):** El riesgo de Alta Volatilidad es bajo. La probabilidad de Baja Vol. es **{results_k2['prob_baja']:.4f}**, mayor de 0.70. Condición Favorable para estrategias de Theta.")
    else:
        st.warning(f"**RIESGO ACTIVO (K=2):** La probabilidad de Baja Vol. es **{results_k2['prob_baja']:.4f}**, menor de 0.70. El Régimen de Alta Volatilidad ha tomado peso. Evitar entrar o considerar salir.")

    st.markdown("---")
    
    # ----------------------------------------------------------------------
    # 1.4 LÓGICA HEDGEHOG Y SEMÁFORO GLOBAL
    # ----------------------------------------------------------------------
    st.header("1.4 Lógica HEDGEHOG y Semáforo Global 🚥")

    rv5d_ayer_val = spx["RV_5d"].iloc[-2]
    
    if 'config_df' not in st.session_state:
        st.session_state['config_df'] = get_default_config_df(rv5d_ayer_val)

    rv5d_hoy = spx['RV_5d'].iloc[-1]
    rv5d_ayer = spx['RV_5d'].iloc[-2]
    
    metricas_actuales = {
        'r1_nr_wr': nr_wr_signal_on,
        'r2_k2_70': results_k2['prob_baja'],
        'r6_rv5d_10': rv5d_hoy,
        'r7_rv5d_menor': rv5d_hoy, 
    }
    
    st.markdown("##### Configuración de Reglas (NR/WR y Volatilidad Markov K=2)")

    st.button(
        "⚙️ Resetear a Valores por Defecto", 
        help="Restaura la configuración de reglas a los umbrales predefinidos.", 
        on_click=reset_config_callback, 
        args=(rv5d_ayer_val,)
    )

    df_config = st.session_state['config_df'].copy()
    
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
    
    if 'df_semaforo_body' not in st.session_state:
        calcular_y_mostrar_semaforo(st.session_state['config_df'], metricas_actuales, rv5d_ayer)
    
    st.markdown("---")
    
    if st.button("🚀 Recalcular Semáforo Consolidado"):
        calcular_y_mostrar_semaforo(st.session_state['config_df'], metricas_actuales, rv5d_ayer)
    
    st.markdown("### Tabla Consolidada de Lógica y Resultado 🚦")
    
    df_body = st.session_state['df_semaforo_body']
    df_footer = st.session_state['df_semaforo_footer']
    senal_color = st.session_state['senal_color']
    
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

    st.markdown("---")


# ==============================================================================
# PUNTO DE ENTRADA PROTEGIDO
# ==============================================================================

if __name__ == "__main__":
    
    if check_password():
        main_comparison()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder a la aplicación.")
