import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from sklearn.preprocessing import StandardScaler
import warnings
import math

# Ocultar advertencias de statsmodels que a menudo aparecen durante el ajuste
warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN DE LA APP ---
st.set_page_config(page_title="HEDGEHOG 1.1 - Comparación K=2 vs K=3", layout="wide")
st.title("🔬 HEDGEHOG 1.1 ")
st.markdown("""
Modelos de Regresión de Markov sobre la Volatilidad Realizada ($\text{RV}_{5d}$) 
del S&P 500. Comparativa Markov K=2 (2 states) y K=3 (3 states fx varianza consolidado) """)

# ==============================================================================
# 1. FUNCIONES DE LÓGICA PURA (CARGA Y PREPARACIÓN)
# ==============================================================================

@st.cache_data(ttl=86400)
def fetch_data():
    """Descarga datos históricos del ^GSPC (SPX) y ^VIX (VIX)."""
    # Usamos un período largo para un ajuste robusto
    start = "2010-01-01" 
    end = datetime.now()

    spx = yf.download("^GSPC", start=start, end=end, auto_adjust=False, multi_level_index=False, progress=False)
    vix = yf.download("^VIX", start=start, end=end, auto_adjust=False, multi_level_index=False, progress=False)

    spx.index = pd.to_datetime(spx.index)
    vix_series = vix['Close'].rename('VIX')
    vix_series.index = pd.to_datetime(vix_series.index)

    df_merged = spx.merge(vix_series, how='left', left_index=True, right_index=True)
    df_merged.dropna(subset=['VIX'], inplace=True)
    
    return df_merged

@st.cache_data(ttl=3600)
def calculate_indicators(df_raw: pd.DataFrame):
    """Calcula todos los indicadores técnicos necesarios (RV, ATR, NR, VIX Change)."""
    spx = df_raw.copy()

    # 1. Volatilidad Realizada (RV_5d)
    spx['log_ret'] = np.log(spx['Close'] / spx['Close'].shift(1))
    spx['RV_5d'] = spx['log_ret'].rolling(window=5).std() * np.sqrt(252) # Anualizada

    # 2. Average True Range (ATR_14)
    spx['tr1'] = spx['High'] - spx['Low']
    spx['tr2'] = (spx['High'] - spx['Close'].shift(1)).abs()
    spx['tr3'] = (spx['Low'] - spx['Close'].shift(1)).abs()
    spx['true_range'] = spx[['tr1', 'tr2', 'tr3']].max(axis=1)
    spx['ATR_14'] = spx['true_range'].rolling(window=14).mean()
    spx.drop(columns=['tr1', 'tr2', 'tr3', 'true_range'], inplace=True)

    # 3. Narrow Range (NR14 - Binario)
    window = 14
    spx['nr14_threshold'] = spx['High'].rolling(window=window).max() - spx['Low'].rolling(window=window).min()
    spx['NR14'] = (spx['High'] - spx['Low'] < spx['nr14_threshold']).astype(int)
    spx.drop(columns=['nr14_threshold'], inplace=True)
    
    # 4. Ratio de volatilidad en el VIX
    spx['VIX_pct_change'] = spx['VIX'].pct_change()
    
    return spx.dropna()


def preparar_datos_markov(spx: pd.DataFrame):
    """Estandariza los datos y alinea las series de tiempo."""
    endog_variable = 'RV_5d'
    variables_tvtp = ['VIX', 'ATR_14', 'VIX_pct_change', 'NR14']
    
    data_markov = spx.copy()
    endog = data_markov[endog_variable].dropna()
    
    # Estandarizar exógenas
    exog_tvtp_original = data_markov[variables_tvtp].copy()
    scaler_tvtp = StandardScaler()
    exog_tvtp_scaled_data = scaler_tvtp.fit_transform(exog_tvtp_original.dropna())
    
    exog_tvtp_scaled = pd.DataFrame(
        exog_tvtp_scaled_data,
        index=exog_tvtp_original.dropna().index,
        columns=variables_tvtp
    )

    # Alinear y eliminar NaNs finales
    data_final = pd.concat([endog, exog_tvtp_scaled], axis=1).dropna()
    endog_final = data_final[endog_variable]
    exog_tvtp_final = data_final[variables_tvtp]
    endog_final = endog_final.loc[exog_tvtp_final.index]
    
    if len(endog_final) < 50:
        return None, None
    
    return endog_final, exog_tvtp_final

# ==============================================================================
# 2. MODELO K=2 (ORIGINAL - CON OBJETIVO 0.10)
# ==============================================================================

@st.cache_data(ttl=3600)
def markov_calculation_k2(endog_final, exog_tvtp_final):
    """
    Modelo de 2 regímenes: Baja vs. Alta. Usa 0.10 como objetivo para encontrar
    el umbral dinámico de baja volatilidad.
    """
    VALOR_OBJETIVO_RV5D = 0.10
    UMBRAL_COMPRESION = 0.70 
    
    if endog_final is None or exog_tvtp_final is None:
        return {'error': "Datos insuficientes para el modelo K=2."}

    # --- 1. AJUSTE DEL MODELO ---
    try:
        modelo = MarkovRegression(
            endog=endog_final, k_regimes=2, trend='c', 
            switching_variance=True, switching_trend=True, exog_tvtp=exog_tvtp_final
        )
        resultado = modelo.fit(maxiter=500, disp=False)
    except Exception as e:
        return {'error': f"Error de ajuste K=2: {e}"} 

    # --- 2. IDENTIFICACIÓN DE REGÍMENES (Por Varianza) ---
    regimen_vars = resultado.params.filter(regex='sigma2|Variance')
    regimen_vars_sorted = regimen_vars.sort_values(ascending=True)
    
    # 0 = Baja Volatilidad (Lower Variance)
    # Se extrae el índice de régimen (el número entre corchetes, p. ej., '[0]' -> 0)
    def extract_regime_index(index_str):
        return int(index_str.split('[')[1].replace(']', ''))
    
    regimen_baja_vol_index = extract_regime_index(regimen_vars_sorted.index[0])
    
    # --- 3. CÁLCULO DEL UMBRAL DINÁMICO (Lógica 0.10) ---
    best_percentile = None
    min_diff = float('inf')
    rv5d_historica = endog_final.values
    
    # Buscar el percentil (entre 10% y 50%) cuyo valor esté más cerca de 0.10
    for p in np.linspace(0.10, 0.50, 41): # 10% a 50% en pasos de 1%
        percentile_val = np.percentile(rv5d_historica, p * 100)
        diff = abs(percentile_val - VALOR_OBJETIVO_RV5D)
        
        if diff < min_diff:
            min_diff = diff
            best_percentile = p * 100
            UMBRAL_RV5D_P_OBJETIVO = percentile_val

    # --- 4. EXTRACCIÓN Y CONCLUSIÓN ---
    probabilidades_filtradas = resultado.filtered_marginal_probabilities
    ultima_probabilidad = probabilidades_filtradas.iloc[-1]
    
    prob_baja = ultima_probabilidad.get(regimen_baja_vol_index, 0)
    
    return {
        'nombre': 'K=2 (Original con Objetivo 0.10)',
        'endog_final': endog_final,
        'resultado': resultado,
        'indices_regimen': {'Baja': regimen_baja_vol_index},
        'varianzas_regimen': {'Baja': regimen_vars_sorted.iloc[0], 'Alta': regimen_vars_sorted.iloc[1]},
        'probabilidades_filtradas': probabilidades_filtradas, 
        'prob_baja': prob_baja,
        'UMBRAL_RV5D_P_OBJETIVO': UMBRAL_RV5D_P_OBJETIVO,
        'P_USADO': best_percentile,
        'UMBRAL_COMPRESION': UMBRAL_COMPRESION
    }

# ==============================================================================
# 3. MODELO K=3 (3 estados y rangos en funcion varianza)
# ==============================================================================

@st.cache_data(ttl=3600)
def markov_calculation_k3(endog_final, exog_tvtp_final):
    """
    Modelo de 3 regímenes: Baja, Media, Alta. Identifica regímenes
    únicamente por las varianzas estimadas.
    """
    UMBRAL_COMPRESION = 0.70 
    
    if endog_final is None or exog_tvtp_final is None:
        return {'error': "Datos insuficientes para el modelo K=3."}
        
    # --- 1. AJUSTE DEL MODELO ---
    try:
        modelo = MarkovRegression(
            endog=endog_final, k_regimes=3, trend='c', 
            switching_variance=True, switching_trend=True, exog_tvtp=exog_tvtp_final
        )
        resultado = modelo.fit(maxiter=500, disp=False)
    except Exception as e:
        return {'error': f"Error de ajuste K=3: {e}"} 

    # --- 2. IDENTIFICACIÓN DE REGÍMENES (Por Varianza) ---
    regimen_vars = resultado.params.filter(regex='sigma2|Variance')

    if len(regimen_vars) < 3:
        return {'error': "ADVERTENCIA: No se pudieron extraer los tres parámetros de varianza."}

    # Ordenar las varianzas para asignar: 0=Baja, 1=Media, 2=Alta
    regimen_vars_sorted = regimen_vars.sort_values(ascending=True)
    
    # Extracción del índice de régimen (el número entre corchetes, p. ej., '[0]' -> 0)
    def extract_regime_index(index_str):
        return int(index_str.split('[')[1].replace(']', ''))
        
    indices_regimen = {
        'Baja': extract_regime_index(regimen_vars_sorted.index[0]),
        'Media': extract_regime_index(regimen_vars_sorted.index[1]),
        'Alta': extract_regime_index(regimen_vars_sorted.index[2])
    }
    
    varianzas_regimen = {
        'Baja': regimen_vars_sorted.iloc[0],
        'Media': regimen_vars_sorted.iloc[1],
        'Alta': regimen_vars_sorted.iloc[2]
    }
    
    # --- 3. EXTRACCIÓN Y CONCLUSIÓN ---
    probabilidades_filtradas = resultado.filtered_marginal_probabilities
    ultima_probabilidad = probabilidades_filtradas.iloc[-1]
    
    prob_baja = ultima_probabilidad.get(indices_regimen['Baja'], 0)
    prob_media = ultima_probabilidad.get(indices_regimen['Media'], 0)
    
    return {
        'nombre': 'K=3 (Varianza Objetiva)',
        'resultado': resultado,
        'indices_regimen': indices_regimen,
        'varianzas_regimen': varianzas_regimen,
        'prob_baja': prob_baja,
        'prob_media': prob_media,
        'UMBRAL_COMPRESION': UMBRAL_COMPRESION
    }

# ==============================================================================
# 4. FUNCIÓN PRINCIPAL DE EJECUCIÓN Y VISUALIZACIÓN DE TABLA
# ==============================================================================

def main_comparison():
    # --- 1. Cargar datos y calcular indicadores ---
    st.header("1. Carga y Preparación de Datos")
    with st.spinner("Descargando datos históricos y calculando indicadores..."):
        df_raw = fetch_data()
        spx = calculate_indicators(df_raw)
        endog_final, exog_tvtp_final = preparar_datos_markov(spx)

    if endog_final is None:
        st.error("❌ Error: No se pudieron preparar los datos para el análisis Markov.")
        return

    st.success(f"✅ Descarga y preparación exitosa. Datos listos para el análisis ({len(endog_final)} puntos).")

    col_k2, col_k3 = st.columns(2)
    results_k2, results_k3 = None, None

    # --- 2. Ejecutar Modelo K=2 ---
    with col_k2:
        st.subheader("Modelo K=2 (Objetivo RV=0.10)")
        with st.spinner("Ajustando Modelo K=2..."):
            results_k2 = markov_calculation_k2(endog_final, exog_tvtp_final)

    # --- 3. Ejecutar Modelo K=3 ---
    with col_k3:
        st.subheader("Modelo K=3 (Varianza Objetiva)")
        with st.spinner("Ajustando Modelo K=3..."):
            results_k3 = markov_calculation_k3(endog_final, exog_tvtp_final)

    # --- 4. Mostrar Resultados Clave y Comparación (Tabla) ---

    if 'error' in results_k2:
        st.error(f"❌ Error K=2: {results_k2['error']}")
        return
    if 'error' in results_k3:
        st.error(f"❌ Error K=3: {results_k3['error']}")
        return
    
    st.header("2. Resultados Numéricos Clave y Comparación")
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
        'K=2': [
            f"{results_k2['prob_baja']:.4f}",
            'N/A (No existe)',
            f"{results_k2['prob_baja']:.4f}",
            f"{results_k2['UMBRAL_COMPRESION']:.2f}",
            f"{results_k2['varianzas_regimen']['Baja']:.5f}",
            'N/A (No existe)',
            f"{results_k2['varianzas_regimen']['Alta']:.5f}",
            f"{results_k2['UMBRAL_RV5D_P_OBJETIVO']:.4f}"
        ],
        'K=3': [
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
    st.subheader("Conclusión Operativa para Calendar Spreads")

    if prob_k3_consolidada >= results_k3['UMBRAL_COMPRESION']:
        st.success(f"**SEÑAL DE ENTRADA FUERTE (K=3):** El riesgo de Alta Volatilidad es bajo. La probabilidad consolidada es **{prob_k3_consolidada:.4f}**, superando el umbral de 0.70. Condición Favorable para estrategias de Theta.")
    else:
        st.warning(f"**RIESGO ACTIVO (K=3):** La probabilidad consolidada es **{prob_k3_consolidada:.4f}**, por debajo del umbral de 0.70. El Régimen de Alta Volatilidad ha tomado peso. Evitar entrar o considerar salir.")
    
    st.markdown("""
    ---
    ### Entendiendo la Diferencia Clave
    
    El **Modelo K=2** combina toda la volatilidad no-crisis en una única señal de 'Baja', lo que lo hace propenso a **falsos positivos**.
    
    El **Modelo K=3** descompone la 'Baja' volatilidad en dos estados: 'Baja' (Calma Extrema) y 'Media' (Consolidación). 
    
    La **Probabilidad Consolidada (Baja + Media)** del K=3 ofrece una señal de entrada/salida más robusta: solo da luz verde cuando la suma de los dos estados favorables supera el 70%, actuando como un **filtro más estricto contra el ruido** que el K=2 ignora.
    """)

# ==============================================================================
# EJECUCIÓN DEL SCRIPT
# ==============================================================================
if __name__ == "__main__":
    main_comparison()
