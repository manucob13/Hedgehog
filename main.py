import streamlit as st
import sys
import os
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---
# Esto permite que 'from utils.utils import check_password' funcione siempre
root_path = Path(__file__).parent.absolute()
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

# 1. CONFIGURACIÓN GLOBAL DE LA PESTAÑA Y LAYOUT
st.set_page_config(
    page_title="Hedgehog opciones",
    layout="wide",
    page_icon="🦔"
)

# 2. DEFINICIÓN DE LAS PÁGINAS
# --- Sección ESTRATEGIAS TE ---
te_signals = st.Page(
    "Estrategias/TE/00 TE Signals.py", 
    title="Signals", 
    icon="🦔", 
    default=True
)
te_calculos = st.Page(
    "Estrategias/TE/01 TE Calculos.py", 
    title="Cálculos", 
    icon="🦔"
)

# --- Sección ESTRATEGIAS TC ---
tc_signals = st.Page(
    "Estrategias/TC/00 TC Signals.py", 
    title="Signals", 
    icon="🦉"
)
tc_calculos = st.Page(
    "Estrategias/TC/01 TC Calculos.py", 
    title="Cálculos", 
    icon="🦉"
)

# --- Sección HERRAMIENTAS ---
vix_term = st.Page(
    "Herramientas/01 VIX Term Structure.py", 
    title="VIX Term Structure", 
    icon="🐣"
)
trend_donchian = st.Page(
    "Herramientas/02 Trend Wkly Donchian.py", 
    title="Trend Wkly Donchian", 
    icon="📉"
)
trend_keltner = st.Page(
    "Herramientas/03 Trend Wkly Keltner.py", 
    title="Trend Wkly Keltner", 
    icon="📉"
)
trend_Zscore = st.Page(
    "Herramientas/04 Trend Wkly Z-score.py", 
    title="Trend Wkly Zscore", 
    icon="📉"
)
gex_analyzer = st.Page(
    "Herramientas/04 GEX Analyzer.py", 
    title="GEX Analyzer", 
    icon="🧪"
)
screener_risk = st.Page(
    "Herramientas/05 Screener Ext. Risk.py", 
    title="Screener Ext. Risk", 
    icon="🚨"
)

# 3. CREACIÓN DE LA NAVEGACIÓN JERÁRQUICA
pg = st.navigation({
    "ESTRATEGIAS TE": [te_signals, te_calculos],
    "ESTRATEGIAS TC": [tc_signals, tc_calculos],
    "HERRAMIENTAS": [
        vix_term, 
        trend_donchian, 
        trend_keltner,
        trend_Zscore,
        gex_analyzer,
        screener_risk
    ]
})

# 4. EJECUCIÓN
pg.run()
