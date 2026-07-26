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

# --- Sección ESTRATEGIAS → Time Edge ---
te_signals = st.Page(
    "Estrategias/TE/00 TE Signals.py", 
    title="Signals", 
    icon="🦔", 
    default=True
)
te_op_21dte = st.Page(
    "Estrategias/TE/01 TE Op 21DTE.py", 
    title="Op 21DTE", 
    icon="🦔"
)

te_op_60dte = st.Page(
    "Estrategias/TE/02 TE Op 60DTE.py", 
    title="Op 60DTE", 
    icon="🦔"
)

# --- Sección ESTRATEGIAS → Triple Calendar ---
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
tc_ajustes = st.Page(
    "Estrategias/TC/02 TC Ajustes.py", 
    title="Ajustes", 
    icon="🦉"
)

# --- Sección ESTRATEGIAS → WEIC ---
weic_calculos = st.Page(
    "Estrategias/WEIC/00 WEIC Calculos.py", 
    title="Cálculos", 
    icon="🦉"

)

# --- Sección HERRAMIENTAS ---
vix_term = st.Page(
    "Herramientas/01 VIX Term Structure.py", 
    title="VIX Term Structure", 
    icon="🐣"
)
trend_Zscore = st.Page(
    "Herramientas/02 Trend Wkly Z-score.py", 
    title="Trend Wkly Zscore", 
    icon="📉"
)
gex_analyzer = st.Page(
    "Herramientas/03 GEX Analyzer.py", 
    title="GEX Analyzer", 
    icon="🧪"
)
screener_risk = st.Page(
    "Herramientas/04 Screener Ext. Risk.py", 
    title="Screener Ext. Risk", 
    icon="🚨"
)
screener_trend = st.Page(
    "Herramientas/05 Screener Trend.py", 
    title="Screener Trend", 
    icon="🚀"
)
screener_CC = st.Page(
    "Herramientas/07 CC ATM Screener.py", 
    title="Screener CC ATM", 
    icon="🧲"
)
screener_ITM = st.Page(
    "Herramientas/08 CC ITM Screener.py", 
    title="Screener CC ITM", 
    icon="📐"
)
broker_test = st.Page(
    "Herramientas/06 Broker Test.py", 
    title="Broker Test", 
    icon="🔌"
)

# 3. CREACIÓN DE LA NAVEGACIÓN JERÁRQUICA
pg = st.navigation({
    "Time Edge": [te_signals, te_op_21dte,te_op_60dte],
    "Triple Calendar": [tc_signals, tc_calculos,tc_ajustes],
    "WEIC": [weic_calculos],
    "HERRAMIENTAS": [
        vix_term, 
        trend_Zscore,
        gex_analyzer,
        screener_risk,
        screener_trend,
        screener_CC,
        screener_ITM,
        broker_test
    ]
})

# 4. EJECUCIÓN
pg.run()
