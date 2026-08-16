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
    url_path="te-signals",
    default=True
)
te_op_21dte = st.Page(
    "Estrategias/TE/01 TE Op 21DTE.py", 
    title="Op 21DTE", 
    icon="🦔",
    url_path="te-op-21dte"
)
te_op_60dte = st.Page(
    "Estrategias/TE/02 TE Op 60DTE.py", 
    title="Op 60DTE", 
    icon="🦔",
    url_path="te-op-60dte"
)
# --- Sección ESTRATEGIAS → Triple Calendar ---
tc_signals = st.Page(
    "Estrategias/TC/00 TC Signals.py", 
    title="Signals", 
    icon="🦉",
    url_path="tc-signals"
)
tc_calculos = st.Page(
    "Estrategias/TC/01 TC Calculos.py", 
    title="Cálculos", 
    icon="🦉",
    url_path="tc-calculos"
)
tc_ajustes = st.Page(
    "Estrategias/TC/02 TC Ajustes.py", 
    title="Ajustes", 
    icon="🦉",
    url_path="tc-ajustes"
)
# --- Sección ESTRATEGIAS → WEIC ---
weic_calculos = st.Page(
    "Estrategias/WEIC/00 WEIC Calculos.py", 
    title="Cálculos", 
    icon="🦉",
    url_path="weic-calculos"
)
# --- Sección CARTERA ---
cartera_panel = st.Page(
    "Cartera/00 Panel.py", 
    title="Panel", 
    icon="📊",
    url_path="cartera-panel"
)
cartera_calendario = st.Page(
    "Cartera/01 Calendario.py", 
    title="Calendario", 
    icon="🗓️",
    url_path="cartera-calendario"
)
cartera_operaciones = st.Page(
    "Cartera/02 Operaciones.py", 
    title="Operaciones", 
    icon="📋",
    url_path="cartera-operaciones"
)
cartera_importar = st.Page(
    "Cartera/03 Importar.py", 
    title="Importar", 
    icon="⬆️",
    url_path="cartera-importar"
)
# --- Sección DIVIDENDOS ---
dividendos_panel = st.Page(
    "Dividendos/00 Panel.py", 
    title="Panel", 
    icon="💰",
    url_path="dividendos-panel"
)
dividendos_asignacion = st.Page(
    "Dividendos/01 Asignacion.py", 
    title="Asignación", 
    icon="🎯",
    url_path="dividendos-asignacion"
)
dividendos_importar = st.Page(
    "Dividendos/02 Importar.py", 
    title="Importar", 
    icon="⬆️",
    url_path="dividendos-importar"
)
# --- Sección HERRAMIENTAS ---
vix_term = st.Page(
    "Herramientas/01 VIX Term Structure.py", 
    title="VIX Term Structure", 
    icon="🐣",
    url_path="vix-term"
)
trend_MACDV = st.Page(
    "Herramientas/02 Trend Wkly MACDV.py", 
    title="Trend Wkly MACD-V", 
    icon="📉",
    url_path="trend-zscore"
)
gex_analyzer = st.Page(
    "Herramientas/03 GEX Analyzer.py", 
    title="GEX Analyzer", 
    icon="🧪",
    url_path="gex-analyzer"
)
screener_risk = st.Page(
    "Herramientas/04 Screener Ext. Risk.py", 
    title="Screener Ext. Risk", 
    icon="🚨",
    url_path="screener-risk"
)
screener_trend = st.Page(
    "Herramientas/05 Screener Trend.py", 
    title="Screener Trend", 
    icon="🚀",
    url_path="screener-trend"
)
regime_hmm = st.Page(
    "Herramientas/06 Regime HMM.py", 
    title="Regime HMM", 
    icon="🧭",
    url_path="regime-hmm"
)
screener_CC = st.Page(
    "Herramientas/07 CC ATM Screener.py", 
    title="Screener CC ATM", 
    icon="🧲",
    url_path="screener-cc"
)
screener_ITM = st.Page(
    "Herramientas/08 CC ITM Screener.py", 
    title="Screener CC ITM", 
    icon="📐",
    url_path="screener-itm"
)
screener_CSP = st.Page(
    "Herramientas/09 CSP ITM Screneer.py",
    title="Screener CSP ITM",
    icon="🛡️",
    url_path="screener-csp"
)
# 3. CREACIÓN DE LA NAVEGACIÓN JERÁRQUICA
pg = st.navigation({
    "Time Edge": [te_signals, te_op_21dte,te_op_60dte],
    "Triple Calendar": [tc_signals, tc_calculos,tc_ajustes],
    "WEIC": [weic_calculos],
    "OPCIONES": [
        cartera_panel,
        cartera_calendario,
        cartera_operaciones,
        cartera_importar
    ],
    "DIVIDENDOS": [
        dividendos_panel,
        dividendos_asignacion,
        dividendos_importar
    ],
    "HERRAMIENTAS": [
        vix_term, 
        trend_MACDV,
        gex_analyzer,
        screener_risk,
        screener_trend,
        regime_hmm,
        screener_CC,
        screener_ITM,
        screener_CSP
    ]
})
# 4. EJECUCIÓN
pg.run()
