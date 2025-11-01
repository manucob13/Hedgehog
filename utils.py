import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# FUNCIONES DE CARGA Y PREPARACIÓN (COMPARTIDAS)
# ==============================================================================

def check_password():
    """
    Controla el acceso. Devuelve True si el usuario ingresa las credenciales correctas
    y False en caso contrario. Usa un botón explícito para evitar errores de renderizado.
    """
    
    # 1. Intenta obtener las credenciales de st.secrets
    try:
        credentials = st.secrets["credentials"]
    except KeyError:
        st.error("Error: Las credenciales secretas no están configuradas.")
        return False
    
    # 2. Control de Acceso (Si ya está correcto, devuelve True inmediatamente)
    if st.session_state.get("password_correct", False):
        return True

    # --- Mostrar Formulario de Login ---
    with st.sidebar:
        st.header("🔑 Iniciar Sesión")
        
        # Usamos st.empty() para controlar dónde aparecerá el error
        error_placeholder = st.empty() 

        # Campos de entrada con keys simples
        username = st.text_input("Usuario", key="login_username_input")
        password = st.text_input("Contraseña", type="password", key="login_password_input")
        
        # Botón para activar la verificación
        if st.button("Login"):
            # 3. Verificación al hacer clic en el botón
            if username == credentials["username"] and password == credentials["password"]:
                st.session_state["password_correct"] = True
                
                # Opcional: Limpiamos los campos para seguridad
                # Nota: Las keys de los inputs deben coincidir: "login_username_input" y "login_password_input"
                del st.session_state["login_username_input"]
                del st.session_state["login_password_input"]
                
                # CORRECCIÓN DE ERROR: Usamos st.rerun()
                st.rerun() 
            else:
                st.session_state["password_correct"] = False
                # Mostramos el error solo después del intento fallido
                error_placeholder.error("😕 Usuario o Contraseña incorrecta")
        
    # 4. Si el login no es correcto o no se ha intentado, el acceso es False
    return False


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
    """Calcula todos los indicadores técnicos necesarios."""
    spx = df_raw.copy()

    # 1. Volatilidad Realizada (RV_5d)
    spx['log_ret'] = np.log(spx['Close'] / spx['Close'].shift(1))
    spx['RV_5d'] = spx['log_ret'].rolling(window=5).std() * np.sqrt(252)

    # 2. Average True Range (ATR_14)
    spx['tr1'] = spx['High'] - spx['Low']
    spx['tr2'] = (spx['High'] - spx['Close'].shift(1)).abs()
    spx['tr3'] = (spx['Low'] - spx['Close'].shift(1)).abs()
    spx['true_range'] = spx[['tr1', 'tr2', 'tr3']].max(axis=1)
    spx['ATR_14'] = spx['true_range'].rolling(window=14).mean()
    spx.drop(columns=['tr1', 'tr2', 'tr3', 'true_range'], inplace=True)

    # 3. Narrow Range (NR14)
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


def check_recent_wr(wr_series, tr_series, wr_len, max_delay):
    """Verifica si hubo un WR en las últimas 'max_delay' barras."""
    wr_recent = pd.Series(False, index=wr_series.index)
    
    for i in range(1, max_delay + 1):
        condition = (tr_series.shift(i) == tr_series.rolling(window=wr_len).max().shift(i))
        wr_recent = wr_recent | condition
    
    return wr_recent


def calculate_nr_wr_signal(spx_raw: pd.DataFrame) -> bool:
    """Calcula la señal NR/WR (solo última señal)."""
    df = spx_raw.copy()

    wr4_len = 4
    nr4_len = 4
    wr7_len = 7
    nr7_len = 7
    max_delay = 3 

    high_low = df['High'] - df['Low']
    high_prev_close = np.abs(df['High'] - df['Close'].shift(1))
    low_prev_close = np.abs(df['Low'] - df['Close'].shift(1))
    df['tr_nr_wr'] = pd.DataFrame({
        'hl': high_low, 
        'hpc': high_prev_close, 
        'lpc': low_prev_close
    }).max(axis=1)

    df['wr4'] = (df['tr_nr_wr'] == df['tr_nr_wr'].rolling(window=wr4_len).max())
    df['wr7'] = (df['tr_nr_wr'] == df['tr_nr_wr'].rolling(window=wr7_len).max())
    df['nr4'] = (df['tr_nr_wr'] == df['tr_nr_wr'].rolling(window=nr4_len).min())
    df['nr7'] = (df['tr_nr_wr'] == df['tr_nr_wr'].rolling(window=nr7_len).min())
    
    df['wr4_recent'] = check_recent_wr(df['wr4'], df['tr_nr_wr'], wr4_len, max_delay)
    df['wr7_recent'] = check_recent_wr(df['wr7'], df['tr_nr_wr'], wr7_len, max_delay)

    df['signal_nr4'] = df['nr4'] & df['wr4_recent'] 
    df['signal_nr7'] = df['nr7'] & df['wr7_recent']
    df['signal_nr_final'] = df['signal_nr7'] | df['signal_nr4']

    if not df['signal_nr_final'].empty:
        return df['signal_nr_final'].iloc[-1]
    return False


def calculate_nr_wr_signal_series(spx_raw: pd.DataFrame) -> pd.Series:
    """Calcula la señal NR/WR como serie temporal completa."""
    df = spx_raw.copy()

    wr4_len = 4
    nr4_len = 4
    wr7_len = 7
    nr7_len = 7
    max_delay = 3 

    high_low = df['High'] - df['Low']
    high_prev_close = np.abs(df['High'] - df['Close'].shift(1))
    low_prev_close = np.abs(df['Low'] - df['Close'].shift(1))
    df['tr_nr_wr'] = pd.DataFrame({
        'hl': high_low, 
        'hpc': high_prev_close, 
        'lpc': low_prev_close
    }).max(axis=1)

    df['wr4'] = (df['tr_nr_wr'] == df['tr_nr_wr'].rolling(window=wr4_len).max())
    df['wr7'] = (df['tr_nr_wr'] == df['tr_nr_wr'].rolling(window=wr7_len).max())
    df['nr4'] = (df['tr_nr_wr'] == df['tr_nr_wr'].rolling(window=nr4_len).min())
    df['nr7'] = (df['tr_nr_wr'] == df['tr_nr_wr'].rolling(window=nr7_len).min())
    
    df['wr4_recent'] = check_recent_wr(df['wr4'], df['tr_nr_wr'], wr4_len, max_delay)
    df['wr7_recent'] = check_recent_wr(df['wr7'], df['tr_nr_wr'], wr7_len, max_delay)

    df['signal_nr4'] = df['nr4'] & df['wr4_recent'] 
    df['signal_nr7'] = df['nr7'] & df['wr7_recent']
    df['signal_nr_final'] = df['signal_nr7'] | df['signal_nr4']

    return df['signal_nr_final'].astype(float)


@st.cache_data(ttl=3600)
def markov_calculation_k2(endog_final, exog_tvtp_final):
    """Modelo de 2 regímenes."""
    VALOR_OBJETIVO_RV5D = 0.10
    UMBRAL_COMPRESION = 0.70 
    
    if endog_final is None or exog_tvtp_final is None:
        return {'error': "Datos insuficientes para el modelo K=2."}

    try:
        modelo = MarkovRegression(
            endog=endog_final, k_regimes=2, trend='c', 
            switching_variance=True, switching_trend=True, exog_tvtp=exog_tvtp_final
        )
        resultado = modelo.fit(maxiter=500, disp=False)
    except Exception as e:
        return {'error': f"Error de ajuste K=2: {e}"} 

    regimen_vars = resultado.params.filter(regex='sigma2|Variance')
    regimen_vars_sorted = regimen_vars.sort_values(ascending=True)
    
    def extract_regime_index(index_str):
        return int(index_str.split('[')[1].replace(']', ''))
    
    regimen_baja_vol_index = extract_regime_index(regimen_vars_sorted.index[0])
    
    best_percentile = None
    min_diff = float('inf')
    rv5d_historica = endog_final.values
    
    for p in np.linspace(0.10, 0.50, 41):
        percentile_val = np.percentile(rv5d_historica, p * 100)
        diff = abs(percentile_val - VALOR_OBJETIVO_RV5D)
        
        if diff < min_diff:
            min_diff = diff
            best_percentile = p * 100
            UMBRAL_RV5D_P_OBJETIVO = percentile_val

    probabilidades_filtradas = resultado.filtered_marginal_probabilities
    ultima_probabilidad = probabilidades_filtradas.iloc[-1]
    
    prob_baja = ultima_probabilidad.get(regimen_baja_vol_index, 0)
    
    # Para gráficos, devolvemos también la serie completa
    prob_baja_serie = probabilidades_filtradas[regimen_baja_vol_index].rename('Prob_Baja_K2')
    
    return {
        'nombre': 'K=2 (Original con Objetivo 0.10)',
        'endog_final': endog_final,
        'resultado': resultado,
        'indices_regimen': {'Baja': regimen_baja_vol_index},
        'varianzas_regimen': {'Baja': regimen_vars_sorted.iloc[0], 'Alta': regimen_vars_sorted.iloc[1]},
        'prob_baja': prob_baja,
        'prob_baja_serie': prob_baja_serie,
        'UMBRAL_RV5D_P_OBJETIVO': UMBRAL_RV5D_P_OBJETIVO,
        'P_USADO': best_percentile,
        'UMBRAL_COMPRESION': UMBRAL_COMPRESION
    }


@st.cache_data(ttl=3600)
def markov_calculation_k3(endog_final, exog_tvtp_final):
    """Modelo de 3 regímenes."""
    UMBRAL_COMPRESION = 0.70 
    
    if endog_final is None or exog_tvtp_final is None:
        return {'error': "Datos insuficientes para el modelo K=3."}
        
    try:
        modelo = MarkovRegression(
            endog=endog_final, k_regimes=3, trend='c', 
            switching_variance=True, switching_trend=True, exog_tvtp=exog_tvtp_final
        )
        resultado = modelo.fit(maxiter=500, disp=False)
    except Exception as e:
        return {'error': f"Error de ajuste K=3: {e}"} 

    regimen_vars = resultado.params.filter(regex='sigma2|Variance')

    if len(regimen_vars) < 3:
        return {'error': "ADVERTENCIA: No se pudieron extraer los tres parámetros de varianza."}

    regimen_vars_sorted = regimen_vars.sort_values(ascending=True)
    
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
    
    probabilidades_filtradas = resultado.filtered_marginal_probabilities
    ultima_probabilidad = probabilidades_filtradas.iloc[-1]
    
    prob_baja = ultima_probabilidad.get(indices_regimen['Baja'], 0)
    prob_media = ultima_probabilidad.get(indices_regimen['Media'], 0)
    
    # Para gráficos, devolvemos también las series completas
    prob_baja_serie = probabilidades_filtradas[indices_regimen['Baja']].rename('Prob_Baja_K3')
    prob_media_serie = probabilidades_filtradas[indices_regimen['Media']].rename('Prob_Media_K3')
    
    return {
        'nombre': 'K=3 (Varianza Objetiva)',
        'resultado': resultado,
        'indices_regimen': indices_regimen,
        'varianzas_regimen': varianzas_regimen,
        'prob_baja': prob_baja,
        'prob_media': prob_media,
        'prob_baja_serie': prob_baja_serie,
        'prob_media_serie': prob_media_serie,
        'UMBRAL_COMPRESION': UMBRAL_COMPRESION
    }
