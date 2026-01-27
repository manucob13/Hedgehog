import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import random
from utils.utils import check_password
from utils.tickers import create_tickers_universe

warnings.filterwarnings('ignore')

# Lock global para sincronizar descargas de yfinance
_yfinance_lock = Lock()

# ============= INTERFAZ PRINCIPAL =============

def main():
    st.set_page_config(
        page_title="Trend Stocks Screener",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Trend Stocks Screener")
    st.markdown("**Detecta acciones con tendencias fuertes y sostenibles**")
    st.markdown("---")
    
    # ============= PASO 1: UNIVERSO DE TICKERS =============
    st.markdown("### 📂 PASO 1: Universo de Tickers")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if 'tickers_universe' not in st.session_state or len(st.session_state.get('tickers_universe', [])) == 0:
            st.info("🔄 Cargando universo de tickers...")
            try:
                df_tickers = create_tickers_universe()
                if isinstance(df_tickers, pd.DataFrame):
                    tickers_list = df_tickers['Ticker'].astype(str).tolist()
                else:
                    tickers_list = list(df_tickers)
                st.session_state['tickers_universe'] = tickers_list
                st.session_state['random_seed'] = random.randint(1, 10000)
                st.success(f"✅ {len(tickers_list):,} tickers cargados")
            except Exception as e:
                st.error(f"❌ Error cargando tickers: {e}")
                st.session_state['tickers_universe'] = []
        else:
            st.success(f"✅ {len(st.session_state['tickers_universe']):,} tickers disponibles")
    
    with col2:
        if st.button("🔄 Recargar Tickers", use_container_width=True):
            if 'tickers_universe' in st.session_state:
                del st.session_state['tickers_universe']
            if 'random_seed' in st.session_state:
                del st.session_state['random_seed']
            st.rerun()
    
    st.markdown("---")
    
    # ============= PASO 2: CONFIGURACIÓN (PLACEHOLDER) =============
    st.markdown("### ⚙️ PASO 2: Configuración de Parámetros")
    st.info("🚧 Configuración pendiente - proporciona los parámetros para detectar trend stocks")
    
    # Aquí irá la configuración específica para trend stocks
    # Por ejemplo: periodos de medias móviles, thresholds, etc.
    
    st.markdown("---")
    
    # ============= PASO 3: ESCANEO (PLACEHOLDER) =============
    st.markdown("### 🚀 PASO 3: Ejecutar Escaneo")
    st.info("🚧 Funcionalidad de escaneo pendiente")
    
    # Botón de escaneo (deshabilitado por ahora)
    scan_button = st.button(
        "🚀 INICIAR ESCANEO", 
        type="primary", 
        use_container_width=True,
        disabled=True  # Habilitado cuando se configure el análisis
    )
    
    if scan_button:
        st.warning("⚠️ Análisis aún no configurado")
    
    st.markdown("---")
    
    # ============= PASO 4: RESULTADOS (PLACEHOLDER) =============
    st.markdown("### 📈 PASO 4: Resultados del Escaneo")
    st.info("🚧 Los resultados aparecerán aquí una vez completado el escaneo")


if __name__ == "__main__":
    if check_password():
        main()
    else:
        st.markdown(
            """
            <div style='text-align: center; padding: 60px 20px;'>
                <h1 style='color: #FF6B6B; font-size: 48px;'>🔒 Acceso Restringido</h1>
                <p style='color: #B0B0B0; font-size: 20px; margin-top: 20px;'>
                    Introduce tus credenciales en el menú lateral.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
