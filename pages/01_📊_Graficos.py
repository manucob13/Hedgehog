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

warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Gráficos - HEDGEHOG", layout="wide")
st.title("📊 Gráficos de Análisis Técnico Combinados")

# ==============================================================================
# 1. FUNCIONES DE LÓGICA PURA (DUPLICADAS)
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
    
    # --- 3. EXTRACCIÓN Y CONCLUSIÓN ---
    return {
        'resultado': resultado,
        'indices_regimen': {'Baja': regimen_baja_vol_index},
        'UMBRAL_COMPRESION': UMBRAL_COMPRESION
    }

# ==============================================================================
# 2. CARGAR Y EJECUTAR MODELOS
# ==============================================================================

with st.spinner("Cargando datos y ajustando Modelo Markov K=2..."):
    df_raw = fetch_data()
    spx = calculate_indicators(df_raw)
    endog_final, exog_tvtp_final = preparar_datos_markov(spx)
    results_k2 = markov_calculation_k2(endog_final, exog_tvtp_final)

if 'error' in results_k2:
    st.error(f"❌ Error al ejecutar el modelo K=2: {results_k2['error']}")
    st.stop() 

st.success(f"✅ Datos y Modelo Markov K=2 cargados exitosamente. ({len(spx)} días)")

# --- CONTROLES DE FECHA ---
st.sidebar.header("⚙️ Configuración del Gráfico")
fecha_final = spx.index[-1].date()
st.sidebar.info(f"📅 Última fecha disponible: {fecha_final}")
fecha_inicio_default = fecha_final - timedelta(days=60)

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

st.markdown(f"**Período seleccionado:** {fecha_inicio} hasta {fecha_final} ({len(spx_filtered)} días)")

# --- PREPARACIÓN DE DATOS PARA GRÁFICO COMBINADO ---

# Crear etiquetas de fecha inteligentes para el eje x
date_labels = []
prev_year = None
prev_month = None

for d in spx_filtered.index:
    if prev_year is None or d.year != prev_year:
        date_labels.append(d.strftime('%b %y'))
        prev_year = d.year
        prev_month = d.month
    elif d.month != prev_month:
        date_labels.append(d.strftime('%b %d'))
        prev_month = d.month
    elif len(date_labels) % 5 == 0:
        date_labels.append(d.strftime('%d'))
    else:
        date_labels.append('')

# Volatilidad Realizada a porcentaje
spx_filtered['RV_5d_pct'] = spx_filtered['RV_5d'] * 100

# Lógica de Color para RV
UMBRAL_RV = 0.10
spx_filtered['RV_change'] = spx_filtered['RV_5d_pct'].diff()
is_up = spx_filtered['RV_change'] >= 0
is_down = ~is_up
rv_green_mask = is_up | is_up.shift(-1).fillna(False)
rv_red_mask = is_down | is_down.shift(-1).fillna(False)
rv_green_plot = spx_filtered['RV_5d_pct'].where(rv_green_mask, other=np.nan)
rv_red_plot = spx_filtered['RV_5d_pct'].where(rv_red_mask, other=np.nan)


# Datos Markov para el período filtrado
probabilidades = results_k2['resultado'].filtered_marginal_probabilities
indice_baja = results_k2['indices_regimen']['Baja']
prob_baja_serie = probabilidades[indice_baja].loc[spx_filtered.index]
prob_baja_serie = prob_baja_serie.fillna(method='ffill')

UMBRAL_ALERTA = 0.50 
UMBRAL_COMPRESION = results_k2['UMBRAL_COMPRESION'] # 0.70

# --- CREAR SUBPLOTS (3 FILAS) ---
fig_combined = make_subplots(
    rows=3, 
    cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.02,
    row_heights=[0.6, 0.2, 0.2], # SPX: 60%, RV: 20%, Markov: 20%
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
# Traza de LÍNEA VERDE (Subida) - Volatilidad Baja
fig_combined.add_trace(go.Scatter(
    x=list(range(len(spx_filtered))),
    y=rv_green_plot,
    mode='lines+markers', 
    name='RV (Sube/Baja Vol.)', 
    line=dict(color='#00B06B', width=2),
    marker=dict(size=5, color='#00B06B'),
    hoverinfo='text',
    text=[f"RV: {y:.2f}% ({'Sube' if u else 'Baja'})" for y, u in zip(spx_filtered['RV_5d_pct'], is_up)],
    showlegend=True 
), row=2, col=1)

# Traza de LÍNEA ROJA (Bajada)
fig_combined.add_trace(go.Scatter(
    x=list(range(len(spx_filtered))),
    y=rv_red_plot,
    mode='lines+markers', 
    name='RV (Baja)', 
    line=dict(color='#F13A50', width=2),
    marker=dict(size=5, color='#F13A50'),
    hoverinfo='text',
    text=[f"RV: {y:.2f}% ({'Sube' if u else 'Baja'})" for y, u in zip(spx_filtered['RV_5d_pct'], is_up)],
    showlegend=False 
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
    y=prob_baja_serie,
    mode='lines',
    name='Prob. K=2 (Baja Vol.)', 
    line=dict(color='#8A2BE2', width=2), 
    fill='tozeroy', 
    fillcolor='rgba(138, 43, 226, 0.3)',
    hoverinfo='text',
    text=[f"Prob. Baja K=2: {p:.4f}" for p in prob_baja_serie],
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

# Etiqueta para el umbral 70% - MOVIDA A LA IZQUIERDA
fig_combined.add_annotation(
    x=0, # Primer punto del eje X (izquierda)
    y=UMBRAL_COMPRESION, 
    text=f'Compresión Fuerte ({UMBRAL_COMPRESION*100:.0f}%)', 
    showarrow=False,
    xref='x3', yref='y3', 
    xanchor='left', # Anclaje a la izquierda
    yanchor='bottom', 
    font=dict(size=12, color="#FFD700"),
    xshift=5, # Pequeño margen a la derecha
    yshift=5, 
    row=3, col=1
)

# Etiqueta para el umbral 50% - MOVIDA A LA IZQUIERDA
fig_combined.add_annotation(
    x=0, # Primer punto del eje X (izquierda)
    y=UMBRAL_ALERTA, 
    text=f'Alerta ({UMBRAL_ALERTA*100:.0f}%)', 
    showarrow=False,
    xref='x3', yref='y3', 
    xanchor='left', # Anclaje a la izquierda
    yanchor='bottom', 
    font=dict(size=12, color="#FFFFFF"), # Blanco
    xshift=5, # Pequeño margen a la derecha
    yshift=5,
    row=3, col=1
)

# Configuraciones de la Fila 3
fig_combined.update_yaxes(title_text='Prob. K=2', row=3, col=1, tickformat=".2f", range=[0, 1])

# --- CONFIGURACIÓN FINAL DEL GRÁFICO COMBINADO ---
fig_combined.update_layout(
    template='plotly_dark',
    height=900, 
    xaxis_rangeslider_visible=False,
    hovermode='x unified',
    plot_bgcolor='#131722', 
    paper_bgcolor='#131722', 
    font=dict(color='#AAAAAA'),
    margin=dict(t=50, b=100, l=60, r=40),
    # AJUSTES DE LEYENDA (ya estaban a la izquierda)
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

# Configurar el eje X compartido (solo las etiquetas inferiores, ahora en la Fila 3)
fig_combined.update_xaxes(
    tickmode='array',
    tickvals=list(range(len(spx_filtered))),
    ticktext=date_labels,
    tickangle=-45,
    row=3, col=1, 
    showgrid=False
)

# Configuraciones adicionales para ejes en tema oscuro
fig_combined.update_xaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=1, col=1)
fig_combined.update_yaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=1, col=1)
fig_combined.update_xaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=2, col=1)
fig_combined.update_yaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=2, col=1)
fig_combined.update_xaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=3, col=1)
fig_combined.update_yaxes(gridcolor='#2A2E39', linecolor='#383C44', mirror=True, row=3, col=1)

st.plotly_chart(fig_combined, use_container_width=True)

# --- INFORMACIÓN ADICIONAL ---
st.markdown("---")
# ... El resto de la sección de st.metric ...
col1, col2, col3, col4, col5 = st.columns(5) 

with col1:
    st.metric("Precio Actual", f"${spx_filtered['Close'].iloc[-1]:.2f}")
with col2:
    cambio = spx_filtered['Close'].iloc[-1] - spx_filtered['Close'].iloc[0]
    cambio_pct = (cambio / spx_filtered['Close'].iloc[0]) * 100
    st.metric("Cambio en el Período", f"${cambio:.2f}", f"{cambio_pct:.2f}%")
with col3:
    st.metric("Máximo", f"${spx_filtered['High'].max():.2f}")
with col4:
    st.metric("Mínimo", f"${spx_filtered['Low'].min():.2f}")
with col5:
    rv_latest = spx_filtered['RV_5d'].iloc[-1] * 100
    st.metric("RV_5d (Último)", f"{rv_latest:.2f}%")
