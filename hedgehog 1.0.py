import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import math

# Ocultar advertencias de statsmodels que a menudo aparecen durante el ajuste
warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN DE LA APP ---
st.set_page_config(page_title="HEDGEHOG 1.1 - Comparación K=2 vs K=3", layout="wide")
st.title("🔬 Comparación de Modelos Markov-Switching: K=2 (Original) vs K=3 (Propuesto)")
st.markdown("""
Esta herramienta ejecuta y compara dos modelos de Regresión de Markov sobre la Volatilidad Realizada (RV_5d) 
del S&P 500 para determinar qué enfoque ofrece una señal más clara para la volatilidad Media y Baja.
""")

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
    regimen_baja_vol_index = int(regimen_vars_sorted.index[0].split('[')[1].replace(']', ''))
    
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
# 3. MODELO K=3 (PROPUESTO - SIN OBJETIVO FIJO)
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
    
    indices_regimen = {
        'Baja': int(regimen_vars_sorted.index[0].split('[')[1].replace(']', '')),
        'Media': int(regimen_vars_sorted.index[1].split('[')[1].replace(']', '')),
        'Alta': int(regimen_vars_sorted.index[2].split('[')[2].replace(']', ''))
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
        'endog_final': endog_final,
        'resultado': resultado,
        'indices_regimen': indices_regimen,
        'varianzas_regimen': varianzas_regimen,
        'probabilidades_filtradas': probabilidades_filtradas, 
        'prob_baja': prob_baja,
        'prob_media': prob_media,
        'UMBRAL_COMPRESION': UMBRAL_COMPRESION
    }

# ==============================================================================
# 4. FUNCIÓN PRINCIPAL DE VISUALIZACIÓN
# ==============================================================================

def comparison_plot(results_k2: dict, results_k3: dict, df_raw: pd.DataFrame):
    """
    Genera un gráfico Plotly comparando las probabilidades de Baja/Media Volatilidad
    de ambos modelos.
    """
    st.subheader("📊 3. Gráfico de Comparación de Probabilidades (Últimos 120 Días)")
    st.markdown("---")
    
    # Usamos los últimos 120 días para mayor claridad
    endog_final_k2 = results_k2['endog_final']
    fecha_final = endog_final_k2.index.max()
    fecha_inicio = fecha_final - pd.DateOffset(days=120)

    # DataFrame base para el gráfico (alineado con la RV_5d)
    df_plot_base = endog_final_k2[endog_final_k2.index >= fecha_inicio].rename('RV_5d').to_frame()
    df_spx_close = df_raw['Close'].loc[df_plot_base.index]

    # --- Datos K=2 ---
    prob_k2_full = results_k2['probabilidades_filtradas']
    prob_k2_baja = prob_k2_full[results_k2['indices_regimen']['Baja']].loc[df_plot_base.index]
    df_plot_base['Prob_K2_Baja'] = prob_k2_baja
    
    # --- Datos K=3 ---
    prob_k3_full = results_k3['probabilidades_filtradas']
    indices_k3 = results_k3['indices_regimen']
    
    prob_k3_baja = prob_k3_full[indices_k3['Baja']].loc[df_plot_base.index]
    prob_k3_media = prob_k3_full[indices_k3['Media']].loc[df_plot_base.index]
    df_plot_base['Prob_K3_Baja'] = prob_k3_baja
    df_plot_base['Prob_K3_Media'] = prob_k3_media

    
    # --- Creación del Gráfico ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        subplot_titles=('Volatilidad Realizada (RV_5d) y SPX', 'Probabilidad de Régimen de Calma'), 
                        vertical_spacing=0.1,
                        row_heights=[0.3, 0.7])

    # 1. Subplot Superior: RV_5d y Precio SPX
    # SPX Price (Eje Secundario)
    fig.add_trace(go.Scatter(x=df_plot_base.index, y=df_spx_close, name='SPX Close', 
                             line=dict(color='rgba(128, 128, 128, 0.5)', width=1)), row=1, col=1, secondary_y=True)
    
    # RV_5d (Eje Primario)
    fig.add_trace(go.Scatter(x=df_plot_base.index, y=df_plot_base['RV_5d'], name='RV_5d', 
                             line=dict(color='#0077b6', width=2)), row=1, col=1, secondary_y=False)
    
    # Umbral K=2 (Para referencia)
    fig.add_hline(y=results_k2['UMBRAL_RV5D_P_OBJETIVO'], row=1, col=1, 
                  line_dash="dot", line_color="orange", opacity=0.8,
                  annotation_text=f"Umbral K=2 ({results_k2['UMBRAL_RV5D_P_OBJETIVO']:.4f})", 
                  annotation_position="bottom right")


    # 2. Subplot Inferior: Comparación de Probabilidades
    
    # Probabilidad K=2 (Baja Volatilidad)
    fig.add_trace(go.Scatter(x=df_plot_base.index, y=df_plot_base['Prob_K2_Baja'], name='Prob. K=2 Baja (0.10 Obj.)', 
                             line=dict(color='darkgreen', width=2)), row=2, col=1)
    
    # Probabilidad K=3 (Baja Volatilidad)
    fig.add_trace(go.Scatter(x=df_plot_base.index, y=df_plot_base['Prob_K3_Baja'], name='Prob. K=3 Baja (Real Calma)', 
                             line=dict(color='green', width=3, dash='dot')), row=2, col=1)

    # Probabilidad K=3 (Media Volatilidad) - Clave para tu Calendar Spread
    fig.add_trace(go.Scatter(x=df_plot_base.index, y=df_plot_base['Prob_K3_Media'], name='Prob. K=3 Media (Consolidación)', 
                             line=dict(color='orange', width=2)), row=2, col=1)

    # Umbral de Señal Fuerte (70%)
    fig.add_hline(y=results_k2['UMBRAL_COMPRESION'], row=2, col=1, 
                  line_dash="dash", line_color="red", opacity=0.8,
                  annotation_text="Umbral Señal Fuerte (70%)", annotation_position="top left")
    
    # --- Configuración Final ---
    fig.update_layout(height=800, 
                      title_text="Comparación de Probabilidades de Regímenes",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                      template="plotly_white")
    
    fig.update_yaxes(title_text="RV_5d Anualizada", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Precio SPX", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Probabilidad", row=2, col=1, range=[0, 1.05])
    
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# 5. EJECUCIÓN DEL FLUJO DE TRABAJO
# ==============================================================================

# --- 1. Cargar datos y calcular indicadores ---
st.header("1. Carga y Preparación de Datos")
with st.spinner("Descargando datos históricos y calculando indicadores..."):
    df_raw = fetch_data()
    spx = calculate_indicators(df_raw)
    endog_final, exog_tvtp_final = preparar_datos_markov(spx)

if endog_final is None:
    st.error("❌ Error: No se pudieron preparar los datos para el análisis Markov.")
    st.stop()

st.success(f"✅ Descarga y preparación exitosa. Datos listos para el análisis ({len(endog_final)} puntos).")

col_k2, col_k3 = st.columns(2)
results_k2, results_k3 = None, None

# --- 2. Ejecutar Modelo K=2 ---
with col_k2:
    st.subheader("Modelo K=2 (Original, Objetivo RV=0.10)")
    with st.spinner("Ajustando Modelo K=2..."):
        results_k2 = markov_calculation_k2(endog_final, exog_tvtp_final)

# --- 3. Ejecutar Modelo K=3 ---
with col_k3:
    st.subheader("Modelo K=3 (Propuesto, Objetividad por Varianza)")
    with st.spinner("Ajustando Modelo K=3..."):
        results_k3 = markov_calculation_k3(endog_final, exog_tvtp_final)

# --- 4. Mostrar Resultados Clave ---

if 'error' in results_k2 or 'error' in results_k3:
    st.error("Uno o ambos modelos fallaron. Revise los mensajes de error.")
else:
    st.header("2. Resultados Numéricos Clave")
    
    col_k2, col_k3 = st.columns(2)
    
    with col_k2:
        st.markdown("### 🟢 K=2: Baja vs. Alta")
        st.info(f"Umbral de RV 5d usado: **{results_k2['UMBRAL_RV5D_P_OBJETIVO']:.4f}** (P{results_k2['P_USADO']:.0f} es el más cercano a 0.10)")
        st.markdown(f"**Varianza del Régimen Baja:** `{results_k2['varianzas_regimen']['Baja']:.5f}`")
        st.markdown(f"**Varianza del Régimen Alta:** `{results_k2['varianzas_regimen']['Alta']:.5f}`")
        st.markdown(f"**Probabilidad HOY (Baja Volatilidad):** **`{results_k2['prob_baja']:.4f}`**")
    
    with col_k3:
        st.markdown("### 🟡 K=3: Baja, Media, y Alta")
        st.markdown(f"**Varianza del Régimen Baja:** `{results_k3['varianzas_regimen']['Baja']:.5f}`")
        st.markdown(f"**Varianza del Régimen Media:** `{results_k3['varianzas_regimen']['Media']:.5f}`")
        st.markdown(f"**Varianza del Régimen Alta:** `{results_k3['varianzas_regimen']['Alta']:.5f}`")
        st.markdown(f"**Probabilidad HOY (Baja Volatilidad):** **`{results_k3['prob_baja']:.4f}`**")
        st.markdown(f"**Probabilidad HOY (Media Volatilidad):** **`{results_k3['prob_media']:.4f}`**")


    # --- 5. Mostrar Gráfico de Comparación ---
    comparison_plot(results_k2, results_k3, df_raw)
    
    st.markdown("""
    ---
    ### Conclusión para tu Estrategia de Calendar Spread
    
    Al comparar los gráficos, notarás que:
    
    1.  **Modelo K=2 (Línea Azul Oscura/Verde Oscuro):** Tiende a tener una probabilidad alta de Régimen Bajo incluso cuando la volatilidad es ligeramente superior al 0.10, porque el Régimen Bajo debe absorber todos los estados que no son "Crisis".
    2.  **Modelo K=3 (Línea Verde Claro y Naranja):** La probabilidad de Baja Volatilidad (Línea Verde Claro) se reduce y se vuelve más estricta. La volatilidad baja y constante (ideal para Calendar) se distingue de la Volatilidad Media (Línea Naranja), que representa el mercado lateral con movimientos mayores.

    **Recomendación:** Para tu Calendar Spread, la **suma de las probabilidades de Régimen Baja y Régimen Media del Modelo K=3** te da el mejor indicador de un mercado apto para la Theta (consolidación/calma). El régimen de Baja Volatilidad del K=3 (Verde Claro) te da la señal de máxima compresión.
    """)
