import streamlit as st
import sys
import os
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---
root_path = Path(__file__).parent.absolute()
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

# 1. CONFIGURACIÓN GLOBAL
st.set_page_config(
    page_title="Hedgehog opciones",
    layout="wide",
    page_icon="🦔"
)

# 2. DEFINICIÓN DE LAS PÁGINAS (Con el espacio detectado en el log)
# Sección TE
te_signals = st.Page(
    "Estrategias /TE/00 TE Signals.py", 
    title="Signals", 
    icon="🦔", 
    default=True
)
te_calculos = st.Page(
    "Estrategias /TE/01 TE Calculos.py", 
    title="Cálculos", 
    icon="🦔"
)

# Sección TC
tc_signals = st.Page(
    "Estrategias /TC/00 TC Signals.py", 
    title="Signals", 
    icon="🦉"
)
tc_calculos = st.Page(
    "Estrategias /TC/01 TC Calculos.py", 
    title="Cálculos", 
    icon="🦉"
)

# Sección Herramientas
vix_term = st.Page(
    "Herramientas/01 VIX Term Structure.py", 
    title="VIX Term Structure", 
    icon="🐣"
)

# 3. NAVEGACIÓN
pg = st.navigation({
    "ESTRATEGIAS TE": [te_signals, te_calculos],
    "ESTRATEGIAS TC": [tc_signals, tc_calculos],
    "HERRAMIENTAS": [vix_term]
})

# 4. EJECUCIÓN
pg.run()
