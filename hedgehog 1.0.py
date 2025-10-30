import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

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
st.dataframe(spx.tail(3))




##### POR DONDE VAMOS ####

import streamlit as st
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from sklearn.preprocessing import StandardScaler
from matplotlib.dates import DateFormatter

# --- 0. CONFIGURACIÓN Y LIMPIEZA (Asegúrate de tener estas en la parte superior de tu script) ---

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
# Asegúrate de que 'spx' ya está definido con los cálculos de volatilidad (RV_5d, RV_21d)
# Las variables 'df', 'spx' se asumen definidas y disponibles.

# --- 1.6 MARKOV CÁLCULO Y GRÁFICA EN STREAMLIT ---

def markov_volatility_analysis(spx):
    """
    Realiza el cálculo del modelo Markov-Switching y muestra los resultados
    y gráficos en Streamlit.
    """
    st.header("📈 Análisis de Régimen de Volatilidad (Markov-Switching)")

    # --- CÁLCULO ---
    st.subheader("⚙️ 1. Cálculo del Modelo Markov-Switching")
    
    # Variables clave
    endog_variable = 'RV_5d'
    variables_tvtp = ['VIX', 'ATR_14', 'VIX_pct_change', 'NR14']
    UMBRAL_COMPRESION = 0.70
    VALOR_OBJETIVO_RV5D = 0.10 # El valor que quieres que el umbral se aproxime

    # Crear un DataFrame limpio y convertir NR14 a entero
    data_markov = spx.copy()
    if 'NR14' in data_markov.columns:
        data_markov['NR14'] = data_markov['NR14'].fillna(0).astype(int) # Añadido fillna(0) por seguridad

    # --- 1. PREPARACIÓN DE DATOS Y ESTANDARIZACIÓN ---
    endog = data_markov[endog_variable].dropna()

    # --- CÁLCULO DINÁMICO DEL UMBRAL RV_5d (BÚSQUEDA AMPLIADA) ---
    rv5d_historico = endog.copy()
    percentiles = np.linspace(0.10, 0.50, 41)
    best_percentile = 0.10
    min_diff = float('inf')

    # Iterar para encontrar el percentil cuya cuantía es más cercana a 0.10
    for p in percentiles:
        p_value = rv5d_historico.quantile(p)
        diff = abs(p_value - VALOR_OBJETIVO_RV5D)
        if diff < min_diff:
            min_diff = diff
            best_percentile = p

    UMBRAL_RV5D_P_OBJETIVO = rv5d_historico.quantile(best_percentile)
    P_USADO = best_percentile * 100 # Para impresión
    
    # 1.1 Estandarizar las exógenas
    exog_tvtp_original = data_markov[variables_tvtp].copy()
    scaler_tvtp = StandardScaler()
    exog_tvtp_scaled_data = scaler_tvtp.fit_transform(exog_tvtp_original)
    exog_tvtp_scaled = pd.DataFrame(
        exog_tvtp_scaled_data,
        index=exog_tvtp_original.index,
        columns=variables_tvtp
    )

    # 1.2 Alinear y eliminar NaNs finales
    data_final = pd.concat([endog, exog_tvtp_scaled], axis=1).dropna()
    endog_final = data_final[endog_variable]
    exog_tvtp_final = data_final[variables_tvtp]

    st.markdown(f"🔥 **UMBRAL RV_5d (P{P_USADO:.0f} más cercano a {VALOR_OBJETIVO_RV5D}):** `{UMBRAL_RV5D_P_OBJETIVO:.4f}`")
    st.write("---")

    # --- 2. AJUSTE DEL MODELO MARKOV-SWITCHING ESTABLE ---
    st.markdown("Ajustando el **Modelo Markov-Switching con Exógenas** en las Transiciones (`exog_tvtp`)...")

    modelo = MarkovRegression(
        endog=endog_final, k_regimes=2, trend='c',
        switching_variance=True, switching_trend=True, exog_tvtp=exog_tvtp_final
    )

    try:
        resultado = modelo.fit(maxiter=500, disp=False)
        st.success("✅ Ajuste exitoso del modelo.")
    except Exception as e:
        st.warning(f"⚠️ ADVERTENCIA: El ajuste inicial falló: {e}. Intentando con 'powell'...")
        try:
            resultado = modelo.fit(maxiter=500, disp=False, method='powell')
            st.success("✅ Reajuste exitoso usando el método 'powell'.")
        except Exception as e2:
            st.error(f"❌ ERROR CRÍTICO: Ambos métodos de ajuste fallaron: {e2}")
            return # Detener la ejecución si el ajuste falla

    st.markdown("### Resumen del Modelo")
    st.code(resultado.summary().as_text())

    # --- 3. INTERPRETACIÓN Y PRONÓSTICO (Formato Limpio) ---

    # 3.1 Identificar los Índices de Régimen
    regimen_vars = resultado.params.filter(regex='sigma2|Variance').sort_values(ascending=True)
    if len(regimen_vars) < 2:
        st.error("❌ ADVERTENCIA: No se pudieron extraer los dos parámetros de varianza.")
        return

    try:
        regimen_baja_vol_param_name = regimen_vars.index[0]
        regimen_baja_vol_index = int(regimen_baja_vol_param_name.split('[')[1].replace(']', ''))
    except:
        regimen_baja_vol_index = 0

    regimen_alta_vol_index = 1 if regimen_baja_vol_index == 0 else 0
    regimen_vars_sorted = regimen_vars.sort_values(ascending=False)
    regimen_alta_vol_param_name = regimen_vars_sorted.index[0]

    st.markdown("### 🔍 Interpretación de Resultados")
    st.markdown("**✅ IDENTIFICACIÓN DEL RÉGIMEN**")
    st.write(f"&emsp;🔹 Régimen de Compresión/Baja Volatilidad: **Régimen {regimen_baja_vol_index}** (Varianza: `{resultado.params[regimen_baja_vol_param_name]:.4f}`)")
    st.write(f"&emsp;🔸 Régimen de Expansión/Alta Volatilidad: **Régimen {regimen_alta_vol_index}** (Varianza: `{resultado.params[regimen_alta_vol_param_name]:.4f}`)")

    # 3.2 Obtención de la Probabilidad de Régimen HOY (Señal de Trading)
    probabilidades_filtradas = resultado.filtered_marginal_probabilities
    ultima_probabilidad = probabilidades_filtradas.iloc[-1]

    prob_baja_vol = ultima_probabilidad.get(regimen_baja_vol_index, 0)
    prob_alta_vol = ultima_probabilidad.get(regimen_alta_vol_index, 0)

    st.markdown("**🚀 PROBABILIDAD DE RÉGIMEN EN EL ÚLTIMO DÍA DE DATOS (HOY)**")
    st.write(f"&emsp;Probabilidad de Baja Volatilidad (Reg. {regimen_baja_vol_index}): **`{prob_baja_vol:.4f}`**")
    st.write(f"&emsp;Probabilidad de Alta Volatilidad (Reg. {regimen_alta_vol_index}): **`{prob_alta_vol:.4f}`**")

    # 3.3 Conclusión de la señal
    if prob_baja_vol >= UMBRAL_COMPRESION:
        conclusion = f"🟢 **SEÑAL DE TRADING:** Alta probabilidad de **BAJA VOLATILIDAD/COMPRESIÓN**. Ideal para *calendars* o estrategias de venta de volatilidad."
    elif prob_alta_vol >= UMBRAL_COMPRESION:
        conclusion = f"🔴 **SEÑAL DE TRADING:** Alta probabilidad de **ALTA VOLATILIDAD/EXPANSIÓN**. Oportunidad para estrategias direccionales o compra de volatilidad."
    else:
        conclusion = f"🟡 **SEÑAL NEUTRA:** Probabilidades divididas (zona de transición). Se recomienda cautela."

    st.markdown(f"## {conclusion}")

    # --- GRÁFICA ---
    st.subheader("📊 2. Visualización Detallada (Últimos 60 Días)")
    
    # 4.1 Filtrado de datos para los últimos 60 días
    fecha_final = endog_final.index.max()
    fecha_inicio = fecha_final - pd.DateOffset(days=60)

    rv_5d_filtrada = endog_final[endog_final.index >= fecha_inicio]
    prob_regimen_baja_completa = resultado.smoothed_marginal_probabilities[regimen_baja_vol_index]
    prob_filtrada = prob_regimen_baja_completa[prob_regimen_baja_completa.index >= fecha_inicio]

    # 4.2 Generación del gráfico
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f'Análisis de Régimen de Volatilidad (Últimos 60 Días)', fontsize=16)

    # --- Gráfico 1: Serie de tiempo de la volatilidad (RV_5d) CON COLORES DINÁMICOS ---
    axes[0].set_title(f'1. Volatilidad Realizada ({endog_variable})')

    # Calcular diferencias para determinar subidas/bajadas
    for i in range(1, len(rv_5d_filtrada)):
        x_vals = [rv_5d_filtrada.index[i-1], rv_5d_filtrada.index[i]]
        y_vals = [rv_5d_filtrada.iloc[i-1], rv_5d_filtrada.iloc[i]]

        # Verde si sube, rojo si baja
        if rv_5d_filtrada.iloc[i] > rv_5d_filtrada.iloc[i-1]:
            axes[0].plot(x_vals, y_vals, color='green', linewidth=1.5, alpha=0.8)
        else:
            axes[0].plot(x_vals, y_vals, color='red', linewidth=1.5, alpha=0.8)

    # Añadir puntos para mejor visualización
    axes[0].scatter(rv_5d_filtrada.index, rv_5d_filtrada.values, c='blue', s=20, alpha=0.6, zorder=5)

    # --- DIBUJAR UMBRAL ENCONTRADO ---
    axes[0].axhline(UMBRAL_RV5D_P_OBJETIVO, color='orange', linestyle=':', linewidth=2, alpha=0.9,
                    label=f'Umbral Baja Vol. (P{P_USADO:.0f}: {UMBRAL_RV5D_P_OBJETIVO:.4f})')

    # Crear leyenda manual para los colores
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', linewidth=2, label='Subida'),
        Line2D([0], [0], color='red', linewidth=2, label='Bajada'),
        Line2D([0], [0], color='orange', linestyle=':', linewidth=2, label=f'Umbral (P{P_USADO:.0f}: {UMBRAL_RV5D_P_OBJETIVO:.4f})')
    ]
    axes[0].legend(handles=legend_elements, loc='upper right')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # --- Gráfico 2: Probabilidad Suavizada del Régimen de Compresión ---
    axes[1].plot(prob_filtrada.index, prob_filtrada.values, label=f'Prob. Baja Volatilidad (Reg. {regimen_baja_vol_index})', color='green', linewidth=2)
    axes[1].axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Umbral 50%')
    axes[1].axhline(UMBRAL_COMPRESION, color='orange', linestyle=':', linewidth=2, alpha=0.7, label=f'Umbral {int(UMBRAL_COMPRESION*100)}% (Señal Fuerte)')
    axes[1].set_title(f'2. Probabilidad de Régimen de Baja Volatilidad/Compresión')
    axes[1].fill_between(prob_filtrada.index, 0, prob_filtrada.values, where=prob_filtrada.values > 0.5, color='green', alpha=0.3)
    axes[1].legend(loc='upper right')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # MEJORAS CLAVE PARA VISUALIZAR TODAS LAS FECHAS EN EL EJE X
    axes[1].set_xticks(prob_filtrada.index[::len(prob_filtrada)//10 or 1]) # Mostrar suficientes fechas
    date_form = DateFormatter("%m-%d")
    axes[1].xaxis.set_major_formatter(date_form)
    plt.xticks(rotation=45, ha='right')

    plt.xlabel("Fecha")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)


# --- CÓDIGO DE INVOCACIÓN (Colócalo donde quieras que se ejecute en tu script) ---
# Asumiendo que 'spx' es tu DataFrame ya cargado y con los cálculos de RV_5d y RV_21d
# markov_volatility_analysis(spx)



