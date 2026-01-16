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
    "Estrategias/TE/TE Signals.py", 
    title="Signals", 
    icon="🦔", 
    default=True
)
te_calculos = st.Page(
    "Estrategias/TE/06 TP Calculos.py", 
    title="Cálculos", 
    icon="🦔"
)

# --- Sección ESTRATEGIAS TC ---
tc_signals = st.Page(
    "Estrategias/TC/TC Signals.py", 
    title="Signals", 
    icon="🦉"
)
tc_calculos = st.Page(
    "Estrategias/TC/TC Calculos.py", 
    title="Cálculos", 
    icon="🦉"
)

# --- Sección HERRAMIENTAS ---
vix_term = st.Page(
    "Herramientas/VIX Term Structure.py", 
    title="VIX Term Structure", 
    icon="🐣"
)
gex_analyzer = st.Page(
    "Herramientas/GEX Analyzer.py", 
    title="GEX Analyzer", 
    icon="🧪"
)
trend_donchian = st.Page(
    "Herramientas/Trend Wkly Donchian.py", 
    title="Trend Wkly Donchian", 
    icon="📊"
)
trend_keltner = st.Page(
    "Herramientas/Trend Wkly Keltner.py", 
    title="Trend Wkly Keltner", 
    icon="📉"
)
screener_risk = st.Page(
    "Herramientas/Screener Ext Risk.py", 
    title="Screener Ext. Risk", 
    icon="🚨"
)
options_ratio = st.Page(
    "Herramientas/Tools Stocks Options Ratio.py", 
    title="Tools Stocks Options Ratio", 
    icon="⚖️"
)

# 3. CREACIÓN DE LA NAVEGACIÓN JERÁRQUICA
pg = st.navigation({
    "ESTRATEGIAS TE": [te_signals, te_calculos],
    "ESTRATEGIAS TC": [tc_signals, tc_calculos],
    "HERRAMIENTAS": [
        vix_term, 
        gex_analyzer, 
        trend_donchian, 
        trend_keltner, 
        screener_risk, 
        options_ratio
    ]
})

# 4. EJECUCIÓN
pg.run()
