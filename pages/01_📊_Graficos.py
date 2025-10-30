import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from sklearn.preprocessing import StandardScaler 
import warnings
import math 

# Ocultar advertencias de statsmodels que a menudo aparecen durante el ajuste
warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Gráficos - HEDGEHOG", layout="wide")
st.title("📊 Gráficos de Análisis Técnico Combinados (K=2, K=3)")

# --- INSERCIÓN: BOTÓN PARA FORZAR LA ACTUALIZACIÓN DE DATOS ---
if st.button("🔄 Forzar Actualización de Datos (Limpiar Caché)", help="Esto borrará la caché de 24 horas de los datos del SPX y VIX y los descargará de nuevo."):
    st.cache_data.clear()
    st.rerun()
st.markdown("---")

# ==============================================================================
# 1. FUNCIONES DE LÓGICA PURA (ADAPTADAS DEL MAIN)
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
    spx.drop(columns=['log_ret'], inplace=True)
    
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
    return {
        'nombre': 'K=2 (Original con Objetivo 0.10)',
        'endog_final': endog_final,
        'resultado': resultado,
        'indices_regimen': {'Baja': regimen_baja_vol_index},
        'varianzas_regimen': {'Baja': regimen_vars_sorted.iloc[0], 'Alta': regimen_vars_sorted.iloc[1]},
        'prob_baja': resultado.filtered_marginal_probabilities[regimen_baja_vol_index].rename('Prob_Baja_K2'),
        'UMBRAL_COMPRESION': UMBRAL_COMPRESION
    }

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
    
    # --- 3. EXTRACCIÓN Y CONCLUSIÓN ---
    probabilidades_filtradas = resultado.filtered_marginal_probabilities
    
    prob_baja_serie = probabilidades_filtradas[indices_regimen['Baja']].rename('Prob_Baja_K3')
    prob_media_serie = probabilidades_filtradas[indices_regimen['Media']].rename('Prob_Media_K3')
    
    return {
        'nombre': 'K=3 (Varianza Objetiva)',
        'resultado': resultado,
        'indices_regimen': indices_regimen,
        'prob_baja': prob_baja_serie,
        'prob_media': prob_media_serie,
        'UMBRAL_COMPRESION': UMBRAL_COMPRESION
    }


# ==============================================================================
# 2. CARGAR Y EJECUTAR MODELOS
# ==============================================================================

with st.spinner("Cargando datos y ajustando Modelos Markov K=2 y K=3..."):
    df_raw = fetch_data()
    spx = calculate_indicators(df_raw)
    endog_final, exog_tvtp_final = preparar_datos_markov(spx)
    
    if endog_final is None:
        st.error("❌ Error: Datos insuficientes para el análisis Markov.")
        st.stop()
        
    results_k2 = markov_calculation_k2(endog_final, exog_tvtp_final)
    results_k3 = markov_calculation_k3(endog_final, exog_tvtp_final)

if 'error' in results_k2:
    st.error(f"❌ Error al ejecutar el modelo K=2: {results_k2['error']}")
    st.stop() 
if 'error' in results_k3:
    st.error(f"❌ Error al ejecutar el modelo K=3: {results_k3['error']}")
    st.stop() 

st.success(f"✅ Datos y Modelos Markov (K=2, K=3) cargados exitosamente. ({len(spx)} días)")

# --- CONTROLES DE FECHA ---
st.sidebar.header("⚙️ Configuración del Gráfico")
fecha_final = spx.index[-1].date()
st.sidebar.info(f"📅 Última fecha disponible: {fecha_final}")
fecha_inicio_default = fecha_final - timedelta(days=90) # Mostrar 3 meses por defecto

fecha_inicio = st.sidebar.date_input(
    "Fecha de inicio:",
    value=fecha_inicio_default,
    min_value=spx.index[0].date(),
    max_value=fecha_final
)

# --- FILTRAR DATOS POR RANGO DE FECHAS ---
fecha_inicio_dt = pd.to_datetime(fecha_inicio)
fecha_final_dt = pd.to_datetime(fecha_final)

spx_filtered = spx[(spx.index >= fecha_inicio_dt) & (spx.index <= fecha_final_dt)].copy()
spx_filtered = spx_filtered[spx_filtered.index.dayofweek < 5] # ELIMINAR fines de semana

# --- PREPARACIÓN DE DATOS PARA GRÁFICO COMBINADO ---

# 1. Etiquetas de Fecha para el eje X
date_labels = [d.strftime('%b %d') if i % 5 == 0 else '' for i, d in enumerate(spx_filtered.index)]
date_labels[0] = spx_filtered.index[0].strftime('%b %d')
date_labels[-1] = spx_filtered.index[-1].strftime('%b %d')

# 2. RV a porcentaje y lógica de color
spx_filtered['RV_5d_pct'] = spx_filtered['RV_5d'] * 100
UMBRAL_RV = 0.10
spx_filtered['RV_change'] = spx_filtered['RV_5d_pct'].diff()
is_up = spx_filtered['RV_change'] >= 0
rv_green_plot = spx_filtered['RV_5d_pct'].where(is_up, other=np.nan)
rv_red_plot = spx_filtered['RV_5d_pct'].where(~is_up, other=np.nan)


# 3. Datos Markov K=2 (Probabilidad Baja)
prob_baja_serie_k2 = results_k2['prob_baja'].loc[spx_filtered.index].fillna(method='ffill')

# 4. Datos Markov K=3 (Probabilidad Consolidada)
prob_baja_serie_k3 = results_k3['prob_baja'].loc[spx_filtered.index].fillna(method='ffill')
prob_media_serie_k3 = results_k3['prob_media'].loc[spx_filtered.index].fillna(method='ffill')
prob_k3_consolidada = prob_baja_serie_k3 + prob_media_serie_k3


UMBRAL_ALERTA = 0.50 
UMBRAL_COMPRESION = results_k2['UMBRAL_COMPRESION'] # 0.70

# --- CREAR SUBPLOTS (4 FILAS) ---
fig_combined = make_subplots(
    rows=4, 
    cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.02,
    row_heights=[0.5, 0.15, 0.175, 0.175], # SPX: 50%, RV: 15%, K=2: 17.5%, K=3: 17.5%
)

# ----------------------------------------------------
# 1. GRÁFICO DE VELAS JAPONESAS (Fila 1)
# ----------------------------------------------------
fig_combined.add_trace(go.Candlestick(
    x=list(range(len(spx_filtered))),
    open=spx_filtered['Open'],
    high=spx_filtered['High'],
    low=spx_filtered['Low'],
    close=spx_filtered['Close'],
    name='S&P 500', 
    increasing=dict(line=dict(color='#00B06B')),
    decreasing=dict(line=dict(color='#F13A50'))
), row=1, col=1)

# Configuraciones de la Fila 1
fig_combined.update_yaxes(title_text='Precio', row=1, col=1)
fig_combined.update_xaxes(showticklabels=False, row=1, col=1)

# ----------------------------------------------------
# 2. GRÁFICO DE VOLATILIDAD REALIZADA (RV_5d) (Fila 2)
# ----------------------------------------------------

# Dibujar segmentos individuales con color según dirección
for i in range(len(spx_filtered) - 1):
    # Determinar el color del segmento según si sube o baja
    color = '#00B06B' if is_up.iloc[i+1] else '#F13A50'
    
    # Dibujar segmento entre punto i y punto i+1
    fig_combined.add_trace(go.Scatter(
        x=[i, i+1],
        y=[spx_filtered['RV_5d_pct'].iloc[i], spx_filtered['RV_5d_pct'].iloc[i+1]],
        mode='lines',
        line=dict(color=color, width=2),
        showlegend=False,
        hoverinfo='skip'
    ), row=2, col=1)

# Añadir puntos invisibles para el hover unificado
fig_combined.add_trace(go.Scatter(
    x=list(range(len(spx_filtered))),
    y=spx_filtered['RV_5d_pct'],
    mode='markers',
    marker=dict(size=0.1, color='rgba(0,0,0,0)'),
    name='RV',
    hoverinfo='text',
    text=[f"RV: {y:.2f}%" for y in spx_filtered['RV_5d_pct']],
    showlegend=True
), row=2, col=1)

# Añadir línea horizontal discontinua del umbral (Fila 2)
fig_combined.add_shape(
    type="line",
    x0=0, y0=UMBRAL_RV * 100,
    x1=len(spx_filtered) - 1, y1=UMBRAL_RV * 100,
    line=dict(color="orange", width=2, dash="dot"),
    layer="below",
    row=2, col=1
)

# Añadir etiqueta para el umbral (Fila 2)
fig_combined.add_annotation(
    x=0, y=1.0, 
    text=f'Umbral RV: {UMBRAL_RV*100:.2f}%', 
    showarrow=False,
    xref='x2', yref='y2 domain', 
    xanchor='left', yanchor='top', 
    font=dict(size=12, color="orange"),
    xshift=5, yshift=-5, 
    row=2, col=1
)

# Configuraciones de la Fila 2
fig_combined.update_yaxes(title_text='RV (%)', row=2, col=1, tickformat=".2f")
fig_combined.update_xaxes(showticklabels=False, row=2, col=1) 

# ----------------------------------------------------
# 3. GRÁFICO DE MARKOV K=2 (PROBABILIDAD BAJA) (Fila 3)
# ----------------------------------------------------

fig_combined.add_trace(go.Scatter(
    x=list(range(len(spx_filtered))),
    y=prob_baja_serie_k2,
    mode='lines',
    name='Prob. K=2 (Baja Vol.)', 
    line=dict(color='#8A2BE2', width=2), # Púrpura
    fill='tozeroy', 
    fillcolor='rgba(138, 43, 226, 0.3)',
    hoverinfo='text',
    text=[f"Prob. Baja K=2: {p:.4f}" for p in prob_baja_serie_k2],
    showlegend=True 
), row=3, col=1)

# Umbral 1: 70% (Línea de Compresión Fuerte)
fig_combined.add_shape(
    type="line",
    x0=0, y0=UMBRAL_COMPRESION,
    x1=len(spx_filtered) - 1, y1=UMBRAL_COMPRESION,
    line=dict(color="#FFD700", width=2, dash="dash"), 
    layer="below",
    row=3, col=1
)

# Umbral 2: 50% (Línea de Alerta/Cambio de Régimen)
fig_combined.add_shape(
    type="line",
    x0=0, y0=UMBRAL_ALERTA,
    x1=len(spx_filtered) - 1, y1=UMBRAL_ALERTA,
    line=dict(color="#FFFFFF", width=1, dash="dot"), # Blanco
    layer="below",
    row=3, col=1
)

# Etiqueta para el umbral 70% - A LA IZQUIERDA
fig_combined.add_annotation(
    x=0, 
    y=UMBRAL_COMPRESION, 
    text=f'Compresión Fuerte ({UMBRAL_COMPRESION*100:.0f}%)', 
    showarrow=False,
    xref='x3', yref='y3', 
    xanchor='left', 
    yanchor='bottom', 
    font=dict(size=12, color="#FFD700"),
    xshift=5, 
    yshift=5, 
    row=3, col=1
)

# Etiqueta para el umbral 50% - A LA IZQUIERDA
fig_combined.add_annotation(
    x=0, 
    y=UMBRAL_ALERTA, 
    text=f'Alerta ({UMBRAL_ALERTA*100:.0f}%)', 
    showarrow=False,
    xref='x3', yref='y3', 
    xanchor='left', 
    yanchor='bottom', 
    font=dict(size=12, color="#FFFFFF"), 
    xshift=5, 
    yshift=5,
    row=3, col=1
)

# Configuraciones de la Fila 3
fig_combined.update_yaxes(title_text='Prob. K=2', row=3, col=1, tickformat=".2f", range=[0, 1])
fig_combined.update_xaxes(showticklabels=False, row=3, col=1) 

# ----------------------------------------------------
# 4. GRÁFICO DE MARKOV K=3 (PROBABILIDAD CONSOLIDADA) (Fila 4)
# ----------------------------------------------------

fig_combined.add_trace(go.Scatter(
    x=list(range(len(spx_filtered))),
    y=prob_k3_consolidada,
    mode='lines',
    name='Prob. K=3 (Baja+Media)', 
    line=dict(color='#00FF7F', width=2), # Verde Esmeralda
    fill='tozeroy', 
    fillcolor='rgba(0, 255, 127, 0.3)',
    hoverinfo='text',
    text=[f"Prob. Consolidada K=3: {p:.4f}" for p in prob_k3_consolidada],
    showlegend=True 
), row=4, col=1)

# Umbral 1: 70% (Línea de Compresión Fuerte)
fig_combined.add_shape(
    type="line",
    x0=0, y0=UMBRAL_COMPRESION,
    x1=len(spx_filtered) - 1, y1=UMBRAL_COMPRESION,
    line=dict(color="#FFD700", width=2, dash="dash"), 
    layer="below",
    row=4, col=1
)

# Umbral 2: 50% (Línea de Alerta/Cambio de Régimen)
fig_combined.add_shape(
    type="line",
    x0=0, y0=UMBRAL_ALERTA,
    x1=len(spx_filtered) - 1, y1=UMBRAL_ALERTA,
    line=dict(color="#FFFFFF", width=1, dash="dot"), # Blanco
    layer="below",
    row=4, col=1
)

# Etiqueta para el umbral 70% - A LA IZQUIERDA
fig_combined.add_annotation(
    x=0, 
    y=UMBRAL_COMPRESION, 
    text=f'Compresión Fuerte ({UMBRAL_COMPRESION*100:.0f}%)', 
    showarrow=False,
    xref='x4', yref='y4', 
    xanchor='left', 
    yanchor='bottom', 
    font=dict(size=12, color="#FFD700"),
    xshift=5, 
    yshift=5, 
    row=4, col=1
)

# Etiqueta para el umbral 50% - A LA IZQUIERDA
fig_combined.add_annotation(
    x=0, 
    y=UMBRAL_ALERTA, 
    text=f'Alerta ({UMBRAL_ALERTA*100:.0f}%)', 
    showarrow=False,
    xref='x4', yref='y4', 
    xanchor='left', 
    yanchor='bottom', 
    font=dict(size=12, color="#FFFFFF"), 
    xshift=5, 
    yshift=5,
    row=4, col=1
)

# Configuraciones de la Fila 4
fig_combined.update_yaxes(title_text='Prob. K=3', row=4, col=1, tickformat=".2f", range=[0, 1])

# --- CONFIGURACIÓN FINAL DEL GRÁFICO COMBINADO ---
fig_combined.update_layout(
    template='plotly_dark',
    height=1000, 
    xaxis_rangeslider_visible=False,
    hovermode='x unified',
    plot_bgcolor='#131722', 
    paper_bgcolor='#131722', 
    font=dict(color='#AAAAAA'),
    margin=dict(t=50, b=100, l=60, r=40),
    showlegend=True,
    legend=dict(
        orientation="v",
        yanchor="top",
        y=1, 
        xanchor="left",
        x=0.01, 
        bgcolor="rgba(0,0,0,0.5)",
        bordercolor="rgba(255,255,255,0.1)",
        borderwidth=1,
        font=dict(size=10)
    )
)

# Configurar el eje X compartido (solo las etiquetas inferiores, ahora en la Fila 4)
fig_combined.update_xaxes(
    tickmode='array',
    tickvals=list(range(len(spx_filtered))),
    ticktext=date_labels,
    tickangle=-45,
    row=4, col=1, 
    showgrid=False
)

# Configuraciones adicionales para ejes en tema oscuro
fig_combined.update_xaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=1, col=1)
fig_combined.update_yaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=1, col=1)
fig_combined.update_xaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=2, col=1)
fig_combined.update_yaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=2, col=1)
fig_combined.update_xaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=3, col=1)
fig_combined.update_yaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=3, col=1)
fig_combined.update_xaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=4, col=1)
fig_combined.update_yaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=4, col=1)


st.plotly_chart(fig_combined, use_container_width=True)

# --- INFORMACIÓN ADICIONAL ---
st.markdown("---")
col1, col2, col3, col4, col5 = st.columns(5) 

with col1:
    st.metric("Precio Actual", f"${spx_filtered['Close'].iloc[-1]:.2f}")
with col2:
    cambio = spx_filtered['Close'].iloc[-1] - spx_filtered['Close'].iloc[0]
    cambio_pct = (cambio / spx_filtered['Close'].iloc[0]) * 100
    st.metric(f"Cambio ({fecha_inicio} al {fecha_final})", f"${cambio:.2f}", f"{cambio_pct:.2f}%")
with col3:
    st.metric("Máximo", f"${spx_filtered['High'].max():.2f}")
with col4:
    st.metric("Mínimo", f"${spx_filtered['Low'].min():.2f}")
with col5:
    rv_latest = spx_filtered['RV_5d'].iloc[-1] * 100
    st.metric("RV_5d (Último)", f"{rv_latest:.2f}%")
