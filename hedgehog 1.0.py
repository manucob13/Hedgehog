import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import warnings
import matplotlib.pyplot as plt
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from sklearn.preprocessing import StandardScaler
from matplotlib.dates import DateFormatter
from matplotlib.lines import Line2D # Necesaria para la leyenda manual en el gráfico

# --- Configuración de la app ---
st.set_page_config(page_title="HEDGEHOG", layout="wide")
st.title("📊 HEDGEHOG 1.0")

# --- Descarga de datos históricos ---
@st.cache_data(ttl=86400)
def fetch_data():
    start = "2010-01-01"
    end = datetime.now()

    # Descarga SPX y VIX
    spx = yf.download("^GSPC", start=start, end=end, auto_adjust=False, multi_level_index=False)
    vix = yf.download("^VIX", start=start, end=end, auto_adjust=False, multi_level_index=False)

    # Convertir índices a datetime
    spx.index = pd.to_datetime(spx.index)
    vix_series = vix['Close'].rename('VIX')
    vix_series.index = pd.to_datetime(vix_series.index)

    # Fusionar SPX con VIX
    df_merged = spx.merge(vix_series, how='left', left_index=True, right_index=True)
    df_merged.dropna(subset=['VIX'], inplace=True)

    return df_merged

# Cargar datos
df = fetch_data()
st.success(f"✅ Descarga datos SPX - VIX desde {df.index.min().date()} - {df.index.max().date()}")

# --- Selección de rango para graficar SPX ---
st.sidebar.header("Rango grafico -  SPX")
min_date = df.index.min().date()
max_date = df.index.max().date()

# Valores por defecto: últimos 3 meses
default_end = max_date
default_start = (pd.Timestamp(default_end) - pd.DateOffset(months=3)).date()

# Selector de fechas
start_plot = st.sidebar.date_input("Fecha inicio", min_value=min_date, max_value=max_date, value=default_start)
end_plot = st.sidebar.date_input("Fecha fin", min_value=start_plot, max_value=max_date, value=default_end)

# --- Filtrar DataFrame según rango ---
df_plot = df.loc[(df.index.date >= start_plot) & (df.index.date <= end_plot)]

# --- Gráfico SPX con velas japonesas (sin huecos de fines de semana) ---
if not df_plot.empty:
    # Convertir índice a string para eje categórico
    df_plot_plotly = df_plot.copy()
    df_plot_plotly['Fecha_str'] = df_plot_plotly.index.strftime('%Y-%m-%d')

    fig = go.Figure(data=[go.Candlestick(
        x=df_plot_plotly['Fecha_str'],
        open=df_plot_plotly['Open'],
        high=df_plot_plotly['High'],
        low=df_plot_plotly['Low'],
        close=df_plot_plotly['Close'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])

    fig.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Precio",
        xaxis_rangeslider_visible=False,
        xaxis_type='category',  # elimina huecos de fines de semana
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No hay datos disponibles para el rango seleccionado.")

# --- Volatilidad realizada 5d vs 21d ---
spx = df.copy()
# Calcular retornos logarítmicos diarios
spx['log_ret'] = np.log(spx['Close'] / spx['Close'].shift(1))
# Calcular volatilidad realizada móvil anualizada a 5 y 21 días
spx['RV_5d'] = spx['log_ret'].rolling(window=5).std() * np.sqrt(252)
spx['RV_21d'] = spx['log_ret'].rolling(window=21).std() * np.sqrt(252)

## --- ATR ---
# Calcular rango diario como High - Low
spx['daily_range'] = spx['High'] - spx['Low']
# Calcular True Range (TR)
spx['previous_close'] = spx['Close'].shift(1)
spx['tr1'] = spx['High'] - spx['Low']
spx['tr2'] = (spx['High'] - spx['previous_close']).abs()
spx['tr3'] = (spx['Low'] - spx['previous_close']).abs()
spx['true_range'] = spx[['tr1', 'tr2', 'tr3']].max(axis=1)
# Calcular ATR como media móvil simple del True Range en ventana de 14 días (o la que prefieras)
period = 14
spx['ATR_14'] = spx['true_range'].rolling(window=period).mean()
# Limpiar columnas auxiliares si quieres
spx.drop(columns=['previous_close', 'tr1', 'tr2', 'tr3'], inplace=True)

## --- Narrow Range (NR) ---
# Calcular ventana para NR
window = 14
# Calcular percentil 14 (o cuantil 0.14) del true_range en ventana móvil
spx['nr14_threshold'] = spx['true_range'].rolling(window=window).quantile(0.14)
# Crear columna que indica si el true_range está por debajo del percentil 14% (True=NR día estrecho)
spx['NR14'] = spx['true_range'] < spx['nr14_threshold']

## --- Ratio de volatilidad en el VIX ---
spx['VIX_pct_change'] = spx['VIX'].pct_change()
# Mostrar las últimas tres filas del DataFrame spx

## --- MARKOV ---
@st.cache_data(ttl=3600) # Importante para que Streamlit no recalcule el modelo en cada interacción

# Calculo y entrenamiento MODELO de MARKOV
def markov_calculation(spx: pd.DataFrame):
    """
    Prepara los datos, ajusta el modelo Markov-Switching y extrae los resultados clave.
    Devuelve un diccionario con los resultados del modelo.
    """
    
    st.subheader("⚙️ 1. Preparación y Cálculo del Modelo Markov")
    st.info("Ajustando el modelo. Esto puede tomar unos segundos...")

    # --- 0. CONFIGURACIÓN Y LIMPIEZA ---
    endog_variable = 'RV_5d'
    variables_tvtp = ['VIX', 'ATR_14', 'VIX_pct_change', 'NR14']
    UMBRAL_COMPRESION = 0.70
    VALOR_OBJETIVO_RV5D = 0.10 # Necesario para el cálculo del umbral

    data_markov = spx.copy()
    
    # Manejar NaN y tipo de dato para NR14
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
        st.error(f"❌ Datos insuficientes para el modelo Markov. Solo {len(endog_final)} puntos disponibles.")
        return None

    # --- 2. AJUSTE DEL MODELO MARKOV-SWITCHING ---
    resultado = None
    try:
        modelo = MarkovRegression(
            endog=endog_final, k_regimes=2, trend='c',
            switching_variance=True, switching_trend=True, exog_tvtp=exog_tvtp_final
        )
        resultado = modelo.fit(maxiter=500, disp=False)
        st.success("✅ Ajuste exitoso del Modelo Markov-Switching.")
    except Exception as e:
        # Intento de reajuste con 'powell'
        try:
            resultado = modelo.fit(maxiter=500, disp=False, method='powell')
            st.warning("⚠️ Ajuste inicial falló. Reajuste exitoso usando el método 'powell'.")
        except Exception as e2:
            st.error(f"❌ ERROR CRÍTICO: Ambos métodos de ajuste fallaron: {e2}")
            return None 

    # --- 3. EXTRACCIÓN E INTERPRETACIÓN DE RESULTADOS ---
    
    # Identificar los Índices de Régimen (Basado en la varianza)
    regimen_vars = resultado.params.filter(regex='sigma2|Variance').sort_values(ascending=True)
    if len(regimen_vars) < 2:
        st.error("❌ ADVERTENCIA: No se pudieron extraer los dos parámetros de varianza.")
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
        'summary': resultado.summary().as_text()
    }

# GRAFICAR MARKOV
def markov_plot(results: dict):
    """
    Muestra los resultados del modelo Markov en Streamlit (texto y gráfico).
    """
    if results is None:
        return

    plt.style.use('seaborn-v0_8-darkgrid')
    
    st.subheader("📊 2. Visualización Detallada (Últimos 60 Días)")

    # Extracción de variables necesarias del diccionario de resultados
    endog_final = results['endog_final']
    resultado = results['resultado']
    regimen_baja_vol_index = results['regimen_baja_vol_index']
    UMBRAL_RV5D_P_OBJETIVO = results['UMBRAL_RV5D_P_OBJETIVO']
    P_USADO = results['P_USADO']
    UMBRAL_COMPRESION = results['UMBRAL_COMPRESION']

    # 4.1 Filtrado de datos para los últimos 60 días
    fecha_final = endog_final.index.max()
    fecha_inicio = fecha_final - pd.DateOffset(days=60) 

    rv_5d_filtrada = endog_final[endog_final.index >= fecha_inicio]
    prob_regimen_baja_completa = resultado.smoothed_marginal_probabilities[regimen_baja_vol_index]
    prob_filtrada = prob_regimen_baja_completa[prob_regimen_baja_completa.index >= fecha_inicio]

    # --- Generación del gráfico (Dos Subplots) ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle('Análisis de Régimen de Volatilidad (Últimos 60 Días)', fontsize=16)

    # Gráfico 1: RV_5d
    axes[0].set_title(f"1. Volatilidad Realizada (RV_5d)")
    
    # Dibujar líneas con colores dinámicos
    for i in range(1, len(rv_5d_filtrada)):
        x_vals = [rv_5d_filtrada.index[i-1], rv_5d_filtrada.index[i]]
        y_vals = [rv_5d_filtrada.iloc[i-1], rv_5d_filtrada.iloc[i]]
        color = 'green' if rv_5d_filtrada.iloc[i] > rv_5d_filtrada.iloc[i-1] else 'red'
        axes[0].plot(x_vals, y_vals, color=color, linewidth=1.5, alpha=0.8)
        
    axes[0].scatter(rv_5d_filtrada.index, rv_5d_filtrada.values, c='blue', s=20, alpha=0.6, zorder=5)
    
    # Dibujar Umbral RV_5d
    axes[0].axhline(UMBRAL_RV5D_P_OBJETIVO, color='orange', linestyle=':', linewidth=2, alpha=0.9,
                    label=f"Umbral Baja Vol. (P{P_USADO:.0f}: {UMBRAL_RV5D_P_OBJETIVO:.4f})")
    
    # Leyenda manual
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', linewidth=2, label='RV_5d Sube'),
        Line2D([0], [0], color='red', linewidth=2, label='RV_5d Baja'),
        Line2D([0], [0], color='orange', linestyle=':', linewidth=2, label=f"Umbral (P{P_USADO:.0f})")
    ]
    axes[0].legend(handles=legend_elements, loc='upper right')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Gráfico 2: Probabilidad Suavizada
    axes[1].plot(prob_filtrada.index, prob_filtrada.values, label=f'Prob. Baja Volatilidad (Reg. {regimen_baja_vol_index})', color='green', linewidth=2)
    axes[1].axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Umbral 50%')
    axes[1].axhline(UMBRAL_COMPRESION, color='orange', linestyle=':', linewidth=2, alpha=0.7, label=f"Umbral {int(UMBRAL_COMPRESION*100)}% (Señal Fuerte)")
    axes[1].set_title('2. Probabilidad de Régimen de Baja Volatilidad/Compresión')
    axes[1].fill_between(prob_filtrada.index, 0, prob_filtrada.values, where=prob_filtrada.values > 0.5, color='green', alpha=0.3)
    axes[1].legend(loc='upper right')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # Formato de Fechas en el Eje X
    axes[1].set_xticks(prob_filtrada.index[::len(prob_filtrada)//10 or 1])
    date_form = DateFormatter("%m-%d")
    axes[1].xaxis.set_major_formatter(date_form)
    plt.xticks(rotation=45, ha='right')

    plt.xlabel("Fecha")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)

st.header("Análisis de Régimen de Volatilidad")
markov_results = markov_calculation(spx)

if markov_results:
    st.subheader("✅ Resultados Clave del Cálculo")

    # Mostrar el umbral RV_5d
    st.markdown(f"🔥 **UMBRAL RV_5d (P{markov_results['P_USADO']:.0f} más cercano a 0.10):** `{markov_results['UMBRAL_RV5D_P_OBJETIVO']:.4f}`")

    # Mostrar las varianzas de los regímenes
    st.markdown("---")
    st.markdown(f"**Varianza Baja Volatilidad (Reg. {markov_results['regimen_baja_vol_index']}):** `{markov_results['var_baja_vol']:.4f}`")
    st.markdown(f"**Varianza Alta Volatilidad (Reg. {markov_results['regimen_alta_vol_index']}):** `{markov_results['var_alta_vol']:.4f}`")

    # Mostrar la probabilidad de hoy
    st.markdown("---")
    st.markdown(f"**🚀 Probabilidad HOY (Baja Volatilidad):** **`{markov_results['prob_baja_vol']:.4f}`**")
    st.markdown(f"## {markov_results['conclusion']}")

    with st.expander("Ver Resumen Estadístico Completo del Modelo"):
        st.code(markov_results['summary'])
    
    markov_plot(markov_results) 
else:
    st.error("El cálculo del modelo Markov falló. Revisar logs anteriores.")



##### POR DONDE VAMOS ####

