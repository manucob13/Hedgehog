import streamlit as st
import os
import sys

# --- DIAGNÓSTICO DE RUTAS ---
st.write("### 🔍 Buscando archivos en el servidor:")
base_path = os.getcwd()
st.write(f"Ruta actual: `{base_path}`")

# Listar carpetas principales
st.write("Carpetas en raíz:", os.listdir("."))

# Verificar si la carpeta existe (Sensible a mayúsculas)
folder = "Estrategias/TE"
if os.path.exists(folder):
    st.write(f"✅ Carpeta `{folder}` encontrada.")
    st.write("Archivos dentro:", os.listdir(folder))
else:
    st.error(f"❌ La carpeta `{folder}` NO existe. Revisa si es 'estrategias' o si está en otra ruta.")

# --- DEFINICIÓN DE PÁGINAS ---
try:
    te_signals = st.Page(
        "Estrategias/TE/00 TE Signals.py", 
        title="Signals", 
        icon="🦔", 
        default=True
    )
    
    pg = st.navigation({"ESTRATEGIAS TE": [te_signals]})
    pg.run()
except Exception as e:
    st.error(f"Error al crear la página: {e}")
