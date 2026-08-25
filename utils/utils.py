import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Modelo de Markov - SPX", layout="wide")

st.title("Análisis de Régimen de Volatilidad con Markov (SPX)")

@st.cache_data(ttl=86400)
def fetch_data():
    """Descarga datos históricos del ^GSPC (SPX), ^VIX y ^VIX3M."""
    start = "2010-01-01" 
    end = datetime.datetime.now() + timedelta(days=1) 

    spx = yf.download("^GSPC", start=start, end=end, auto_adjust=False, progress=False)
    vix = yf.download("^VIX", start=start, end=end, auto_adjust=False, progress=False)
    vix3m = yf.download("^VIX3M", start=start, end=end, auto_adjust=False, progress=False)

    # Aplanar MultiIndex de columnas si yfinance lo devuelve
    for df in [spx, vix, vix3m]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    spx.index = pd.to_datetime(spx.index)
    
    vix_series = vix['Close'].rename('VIX') if 'Close' in vix.columns else vix.iloc[:, 0].rename('VIX')
    vix_series.index = pd.to_datetime(vix_series.index)
    
    df_merged = spx.merge(vix_series, how='left', left_index=True, right_index=True)

    if not vix3m.empty and ('Close' in vix3m.columns or len(vix3m.columns) > 0):
        vix3m_series = vix3m['Close'].rename('VIX3M') if 'Close' in vix3m.columns else vix3m.iloc[:, 0].rename('VIX3M')
        vix3m_series.index = pd.to_datetime(vix3m_series.index)
        df_merged = df_merged.merge(vix3m_series, how='left', left_index=True, right_index=True)
    else:
        df_merged['VIX3M'] = np.nan

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

    # 5. VIX Term Structure Ratio (VIX/VIX3M)
    if 'VIX3M' in spx.columns and not spx['VIX3M'].isna().all():
        spx['VIX_VIX3M'] = spx['VIX'] / spx['VIX3M']
    else:
        spx['VIX_VIX3M'] = np.nan
    
    # FILTRADO SEGURO: solo se eliminan filas con nulos en las columnas esenciales
    required_cols = ['Close', 'VIX', 'RV_5d', 'ATR_14', 'VIX_pct_change', 'NR14']
    return spx.dropna(subset=required_cols)


def preparar_datos_markov(spx: pd.DataFrame):
    """Estandariza los datos y alinea las series de tiempo."""
    if spx is None or spx.empty:
        return None, None

    endog_variable = 'RV_5d'
    variables_tvtp = ['VIX', 'ATR_14', 'VIX_pct_change', 'NR14']
    
    for col in [endog_variable] + variables_tvtp:
        if col not in spx.columns:
            return None, None

    data_markov = spx.dropna(subset=[endog_variable] + variables_tvtp).copy()
    
    if len(data_markov) < 50:
        return None, None

    endog = data_markov[endog_variable]
    exog_tvtp_original = data_markov[variables_tvtp]
    
    scaler_tvtp = StandardScaler()
    exog_tvtp_scaled_data = scaler_tvtp.fit_transform(exog_tvtp_original)
    
    exog_tvtp_scaled = pd.DataFrame(
        exog_tvtp_scaled_data,
        index=exog_tvtp_original.index,
        columns=variables_tvtp
    )

    endog_final = endog.loc[exog_tvtp_scaled.index]
    exog_tvtp_final = exog_tvtp_scaled
    
    return endog_final, exog_tvtp_final


def ejecutar_app():
    with st.spinner("Cargando datos y procesando indicadores..."):
        df_raw = fetch_data()
        if df_raw.empty:
            st.error("No se pudieron obtener datos de Yahoo Finance.")
            return

        spx = calculate_indicators(df_raw)
        endog, exog = preparar_datos_markov(spx)

    if endog is None or exog is None:
        st.error("Error: No hay suficientes datos limpios para ejecutar el modelo de Markov.")
        return

    st.success(f"Datos procesados correctamente: {len(endog)} observaciones disponibles.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Vista previa de Indicadores")
        st.dataframe(spx[['Close', 'VIX', 'RV_5d', 'ATR_14', 'NR14']].tail(10))
        
    with col2:
        st.subheader("Variables Exógenas Estandarizadas")
        st.dataframe(exog.tail(10))

    if st.button("Ejecutar Modelo Markov"):
        with st.spinner("Ajustando modelo de cambio de régimen..."):
            try:
                mod = sm.tsa.MarkovRegression(
                    endog, 
                    k_regimes=2, 
                    exog=exog, 
                    switching_variance=True
                )
                res = mod.fit(disp=False)
                
                st.subheader("Probabilidades Suavizadas por Régimen")
                st.line_chart(res.smoothed_marginal_probabilities)
                
                st.subheader("Resumen del Modelo")
                st.text(str(res.summary()))
            except Exception as e:
                st.error(f"Error al entrenar el modelo de Markov: {e}")

if __name__ == "__main__":
    ejecutar_app()
