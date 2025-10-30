import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import warnings
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from sklearn.preprocessing import StandardScaler

# Ocultar advertencias de statsmodels que a menudo aparecen durante el ajuste
warnings.filterwarnings('ignore')

# --- Descarga de datos históricos (Cacheado) ---
def fetch_data():
    """Descarga datos históricos del ^GSPC (SPX) y ^VIX (VIX)."""
    start = "2010-01-01"
    end = datetime.now()

    # Descarga SPX y VIX
    spx = yf.download("^GSPC", start=start, end=end, auto_adjust=False, multi_level_index=False, progress=False)
    vix = yf.download("^VIX", start=start, end=end, auto_adjust=False, multi_level_index=False, progress=False)

    # Preparación y fusión de datos
    spx.index = pd.to_datetime(spx.index)
    vix_series = vix['Close'].rename('VIX')
    vix_series.index = pd.to_datetime(vix_series.index)

    df_merged = spx.merge(vix_series, how='left', left_index=True, right_index=True)
    df_merged.dropna(subset=['VIX'], inplace=True)
    
    print(f"DEBUG: Datos descargados desde {df_merged.index.min().date()} hasta {df_merged.index.max().date()}")

    return df_merged

# --- Cálculo de Indicadores (Cacheado) ---
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
    # NR14 es 1 si el rango de hoy es estrecho
    spx['NR14'] = (spx['true_range'] < spx['nr14_threshold']).astype(int)
    spx.drop(columns=['true_range', 'nr14_threshold'], inplace=True)
    
    # 4. Ratio de volatilidad en el VIX
    spx['VIX_pct_change'] = spx['VIX'].pct_change()
    
    print("DEBUG: Indicadores calculados.")
    return spx

# --- MARKOV CALCULATION (Función de lógica pura) ---
def markov_calculation(spx: pd.DataFrame):
    """
    Prepara los datos, ajusta el modelo Markov-Switching y extrae los resultados clave.
    Devuelve un diccionario con los resultados del modelo.
    """
    
    print("DEBUG: Ajustando el modelo Markov-Switching...")

    # --- 0. CONFIGURACIÓN Y LIMPIEZA ---
    endog_variable = 'RV_5d'
    variables_tvtp = ['VIX', 'ATR_14', 'VIX_pct_change', 'NR14']
    UMBRAL_COMPRESION = 0.70
    VALOR_OBJETIVO_RV5D = 0.10 # Necesario para el cálculo del umbral

    data_markov = spx.copy()
    
    if 'NR14' in data_markov.columns:
        data_markov['NR14'] = data_markov['NR14'].fillna(0).astype(int) 

    # --- 1. PREPARACIÓN DE DATOS Y ESTANDARIZACIÓN ---
    endog = data_markov[endog_variable].dropna()

    # CÁLCULO DINÁMICO DEL UMBRAL RV_5d
    rv5d_historico = endog.copy()
    percentiles = np.linspace(0.10, 0.50, 41)
    best_percentile = 0.10
    min_diff = float('inf')

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
        print(f"ERROR: Datos insuficientes para el modelo Markov. Solo {len(endog_final)} puntos disponibles.")
        return None

    # --- 2. AJUSTE DEL MODELO MARKOV-SWITCHING ---
    resultado = None
    try:
        modelo = MarkovRegression(
            endog=endog_final, k_regimes=2, trend='c',
            switching_variance=True, switching_trend=True, exog_tvtp=exog_tvtp_final
        )
        resultado = modelo.fit(maxiter=500, disp=False)
        print("DEBUG: Ajuste exitoso del Modelo Markov-Switching.")
    except Exception as e:
        # Intento de reajuste con 'powell'
        try:
            resultado = modelo.fit(maxiter=500, disp=False, method='powell')
            print("WARNING: Ajuste inicial falló. Reajuste exitoso usando el método 'powell'.")
        except Exception as e2:
            print(f"ERROR CRÍTICO: Ambos métodos de ajuste fallaron: {e2}")
            return None 

    # --- 3. EXTRACCIÓN E INTERPRETACIÓN DE RESULTADOS ---
    
    # Identificar los Índices de Régimen (Basado en la varianza)
    regimen_vars = resultado.params.filter(regex='sigma2|Variance').sort_values(ascending=True)
    if len(regimen_vars) < 2:
        print("ERROR: ADVERTENCIA: No se pudieron extraer los dos parámetros de varianza.")
        return None

    # Extracción de índices y nombres de parámetros
    try:
        regimen_baja_vol_param_name = regimen_vars.index[0]
        regimen_baja_vol_index = int(regimen_baja_vol_param_name.split('[')[1].replace(']', ''))
    except:
        regimen_baja_vol_index = 0

    regimen_alta_vol_index = 1 if regimen_baja_vol_index == 0 else 0
    regimen_vars_sorted = regimen_vars.sort_values(ascending=False)
    regimen_alta_vol_param_name = regimen_vars_sorted.index[0]
    
    # Obtención de la Probabilidad de Régimen HOY
    probabilidades_filtradas = resultado.filtered_marginal_probabilities
    ultima_probabilidad = probabilidades_filtradas.iloc[-1]
    
    # Obtener la fecha de la última observación
    ultima_fecha = probabilidades_filtradas.index[-1].strftime('%Y-%m-%d')


    prob_baja_vol = ultima_probabilidad.get(regimen_baja_vol_index, 0)
    
    # Conclusión
    conclusion = "🟡 SEÑAL NEUTRA: Probabilidades divididas (zona de transición)."
    if prob_baja_vol >= UMBRAL_COMPRESION:
        conclusion = "🟢 SEÑAL DE TRADING: Alta probabilidad de BAJA VOLATILIDAD/COMPRESIÓN."
    elif (1 - prob_baja_vol) >= UMBRAL_COMPRESION:
        conclusion = "🔴 SEÑAL DE TRADING: Alta probabilidad de ALTA VOLATILIDAD/EXPANSIÓN."
    
    # Retornar todos los resultados necesarios para la visualización
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
        'ultima_fecha': ultima_fecha # Nuevo: Fecha del último entrenamiento
    }

# --- EJECUCIÓN DE PRUEBA (Para ver los resultados en consola) ---
if __name__ == "__main__":
    print("--- INICIANDO PRUEBA DE MÓDULO DE ANÁLISIS ---")
    
    # 1. Cargar datos base
    df_raw = fetch_data()

    # 2. Calcular indicadores
    spx_indicators = calculate_indicators(df_raw)
    
    # Nuevo: Mostrar últimas 3 filas del dataframe con indicadores
    print("\n--- ÚLTIMAS 3 FILAS DEL DATAFRAME DE INDICADORES (SPX) ---")
    print(spx_indicators.tail(3).to_string())

    # 3. Ejecutar modelo Markov
    markov_results = markov_calculation(spx_indicators)

    # 4. Imprimir resultados clave
    if markov_results:
        print("\n--- RESULTADOS CLAVE DEL MODELO MARKOV ---")
        print(f"Fecha de Entrenamiento: {markov_results['ultima_fecha']}") # Nuevo: Mostrar la fecha
        print(f"Probabilidad de Baja Volatilidad (HOY): {markov_results['prob_baja_vol']:.4f}")
        print(f"Conclusión de la Señal: {markov_results['conclusion']}")
        print(f"Varianza Régimen Baja Vol.: {markov_results['var_baja_vol']:.4f}")
        print(f"Varianza Régimen Alta Vol.: {markov_results['var_alta_vol']:.4f}")
        print("\nResumen Estadístico Completo:")
        print(markov_results['summary'])
    else:
        print("\n--- NO SE PUDO OBTENER RESULTADOS DEL MODELO MARKOV ---")
