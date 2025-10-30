import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from sklearn.preprocessing import StandardScaler
import warnings

# Ocultar advertencias de statsmodels que a menudo aparecen durante el ajuste
warnings.filterwarnings('ignore')

# --- Configuración de la app ---
st.set_page_config(page_title="HEDGEHOG", layout="wide")
st.title("📊 HEDGEHOG 1.0 - Análisis Básico de Volatilidad")

# ==============================================================================
# 1. FUNCIONES DE LÓGICA PURA
# (Estas funciones NO contienen llamadas directas a Streamlit, solo lógica de datos)
# ==============================================================================

@st.cache_data(ttl=86400)
def fetch_data():
    """Descarga datos históricos del ^GSPC (SPX) y ^VIX (VIX)."""
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

    # 1. Volatilidad Realizada (RV_5d y RV_21d)
    spx['log_ret'] = np.log(spx['Close'] / spx['Close'].shift(1))
    spx['RV_5d'] = spx['log_ret'].rolling(window=5).std() * np.sqrt(252)
    spx['RV_21d'] = spx['log_ret'].rolling(window=21).std() * np.sqrt(252)

    # 2. Average True Range (ATR_14)
    spx['daily_range'] = spx['High'] - spx['Low']
    spx['previous_close'] = spx['Close'].shift(1)
    spx['tr1'] = spx['High'] - spx['Low']
    spx['tr2'] = (spx['High'] - spx['previous_close']).abs()
    spx['tr3'] = (spx['Low'] - spx['previous_close']).abs()
    spx['true_range'] = spx[['tr1', 'tr2', 'tr3']].max(axis=1)
    period = 14
    spx['ATR_14'] = spx['true_range'].rolling(window=period).mean()
    spx.drop(columns=['previous_close', 'tr1', 'tr2', 'tr3', 'daily_range'], inplace=True)

    # 3. Narrow Range (NR14 - Binario)
    window = 14
    spx['nr14_threshold'] = spx['true_range'].rolling(window=window).quantile(0.14)
    spx['NR14'] = (spx['true_range'] < spx['nr14_threshold']).astype(int)
    spx.drop(columns=['true_range', 'nr14_threshold'], inplace=True)
    
    # 4. Ratio de volatilidad en el VIX
    spx['VIX_pct_change'] = spx['VIX'].pct_change()
    
    return spx

@st.cache_data(ttl=3600)
def markov_calculation(spx: pd.DataFrame):
    """
    Prepara los datos, ajusta el modelo Markov-Switching y extrae los resultados clave.
    Devuelve un diccionario con los resultados del modelo.
    """
    
    # --- 0. CONFIGURACIÓN ---
    endog_variable = 'RV_5d'
    variables_tvtp = ['VIX', 'ATR_14', 'VIX_pct_change', 'NR14']
    UMBRAL_COMPRESION = 0.70
    VALOR_OBJETIVO_RV5D = 0.10 

    data_markov = spx.copy()
    
    if 'NR14' in data_markov.columns:
        data_markov['NR14'] = data_markov['NR14'].fillna(0).astype(int) 

    # --- 1. PREPARACIÓN DE DATOS Y ESTANDARIZACIÓN ---
    endog = data_markov[endog_variable].dropna()
    
    # CÁLCULO DINÁMICO DEL UMBRAL RV_5d
    rv5d_historico = endog.copy()
    percentiles = np.linspace(0.10, 0.50, 41)
    min_diff = float('inf')
    best_percentile = 0.10

    for p in percentiles:
        p_value = rv5d_historico.quantile(p)
        diff = abs(p_value - VALOR_OBJETIVO_RV5D)
        if diff < min_diff:
            min_diff = diff
            best_percentile = p

    UMBRAL_RV5D_P_OBJETIVO = rv5d_historico.quantile(best_percentile)
    P_USADO = best_percentile * 100

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
    
    if len(endog_final) < 50:
        return {'error': f"Datos insuficientes para el modelo Markov. Solo {len(endog_final)} puntos disponibles."}

    # --- 2. AJUSTE DEL MODELO MARKOV-SWITCHING ---
    resultado = None
    try:
        modelo = MarkovRegression(
            endog=endog_final, k_regimes=2, trend='c',
            switching_variance=True, switching_trend=True, exog_tvtp=exog_tvtp_final
        )
        resultado = modelo.fit(maxiter=500, disp=False)
    except Exception:
        try:
            # Intento con un método diferente si el primero falla
            resultado = modelo.fit(maxiter=500, disp=False, method='powell')
        except Exception as e2:
            return {'error': f"ERROR CRÍTICO: Ambos métodos de ajuste fallaron: {e2}"} 

    # --- 3. EXTRACCIÓN E INTERPRETACIÓN DE RESULTADOS ---
    regimen_vars = resultado.params.filter(regex='sigma2|Variance').sort_values(ascending=True)
    if len(regimen_vars) < 2:
        return {'error': "ADVERTENCIA: No se pudieron extraer los dos parámetros de varianza."}

    # Determinar el índice del régimen de baja volatilidad (el que tiene la varianza más pequeña)
    try:
        regimen_baja_vol_param_name = regimen_vars.index[0]
        regimen_baja_vol_index = int(regimen_baja_vol_param_name.split('[')[1].replace(']', ''))
    except:
        regimen_baja_vol_index = 0

    regimen_alta_vol_index = 1 if regimen_baja_vol_index == 0 else 0
    regimen_vars_sorted = regimen_vars.sort_values(ascending=False)
    regimen_alta_vol_param_name = regimen_vars_sorted.index[0]
    
    probabilidades_filtradas = resultado.filtered_marginal_probabilities
    ultima_probabilidad = probabilidades_filtradas.iloc[-1]
    ultima_fecha = probabilidades_filtradas.index[-1].strftime('%Y-%m-%d')

    prob_baja_vol = ultima_probabilidad.get(regimen_baja_vol_index, 0)
    
    conclusion = "🟡 SEÑAL NEUTRA: Probabilidades divididas (zona de transición)."
    if prob_baja_vol >= UMBRAL_COMPRESION:
        conclusion = "🟢 SEÑAL DE TRADING: Alta probabilidad de BAJA VOLATILIDAD/COMPRESIÓN."
    elif (1 - prob_baja_vol) >= UMBRAL_COMPRESION:
        conclusion = "🔴 SEÑAL DE TRADING: Alta probabilidad de ALTA VOLATILIDAD/EXPANSIÓN."
    
    return {
        'endog_final': endog_final,
        'resultado': resultado,
        'UMBRAL_RV5D_P_OBJETIVO': UMBRAL_RV5D_P_OBJETIVO,
        'P_USADO': P_USADO,
        'regimen_baja_vol_index': regimen_baja_vol_index,
        'regimen_alta_vol_index': regimen_alta_vol_index,
        'prob_baja_vol': prob_baja_vol,
        'prob_alta_vol': 1 - prob_baja_vol,
        'conclusion': conclusion,
        'UMBRAL_COMPRESION': UMBRAL_COMPRESION,
        'var_baja_vol': resultado.params[regimen_baja_vol_param_name],
        'var_alta_vol': resultado.params[regimen_alta_vol_param_name],
        'summary': resultado.summary().as_text(),
        'ultima_fecha': ultima_fecha
    }


# ==============================================================================
# 2. FLUJO DE EJECUCIÓN PRINCIPAL Y PANTALLA DE STREAMLIT
# ==============================================================================

# 1. Cargar datos base
st.header("1. Descarga y Cálculo de Indicadores")
with st.spinner("Descargando datos históricos (^GSPC y ^VIX)..."):
    df_raw = fetch_data()
st.success(f"✅ Descarga de datos exitosa desde {df_raw.index.min().date()} - {df_raw.index.max().date()}")

# 2. Calcular indicadores
with st.spinner("Calculando indicadores de volatilidad (RV, ATR, NR, VIX Change)..."):
    spx = calculate_indicators(df_raw)
st.success("✅ Indicadores calculados.")

# Muestra de las últimas filas del DataFrame 
st.subheader("Últimas 2 Filas del DataFrame de Indicadores (spx)")
st.dataframe(spx.tail(2)) 

# 3. Análisis de Régimen de Volatilidad
st.header("2. Análisis de Régimen de Volatilidad (Modelo Markov)")

markov_results = None
with st.spinner("⚙️ Ajustando el Modelo Markov-Switching... Esto puede tardar unos segundos."):
    markov_results = markov_calculation(spx)

if markov_results and 'error' in markov_results:
    st.error(f"❌ Cálculo Fallido: {markov_results['error']}")
elif markov_results:
    st.success("✅ Ajuste del Modelo Markov exitoso.")
    st.subheader("✅ Resultados Clave del Cálculo")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f"**Varianza Baja Volatilidad (Reg. {markov_results['regimen_baja_vol_index']}):** `{markov_results['var_baja_vol']:.4f}`")
        st.markdown(f"**Varianza Alta Volatilidad (Reg. {markov_results['regimen_alta_vol_index']}):** `{markov_results['var_alta_vol']:.4f}`")
        st.markdown(f"🔥 **UMBRAL RV_5d (P{markov_results['P_USADO']:.0f} más cercano a 0.10):** `{markov_results['UMBRAL_RV5D_P_OBJETIVO']:.4f}`")

    with col2:
        # Esta línea muestra la última fecha de datos usada para el entrenamiento:
        st.markdown(f"**Fecha del ultimo dia de Entrenamiento:** `{markov_results['ultima_fecha']}`")
        st.markdown(f"**🚀 Probabilidad HOY (Baja Volatilidad):** **`{markov_results['prob_baja_vol']:.4f}`**")
        st.markdown(f"## {markov_results['conclusion']}")

    with st.expander("Ver Resumen Estadístico Completo del Modelo"):
        st.code(markov_results['summary'])

else:
    st.error("El cálculo del modelo Markov no devolvió resultados.")
