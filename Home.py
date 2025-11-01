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

st.set_page_config(page_title="HEDGEHOG 1.1", layout="wide")
st.title("🔬 HEDGEHOG 1.1     Modelos de Volatilidad - Markov-Switching K=2-3 - NR/WR")
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

    # Mostrar la conclusión operativa
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


if __name__ == "__main__":
    main_comparison()
