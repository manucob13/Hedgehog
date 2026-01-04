import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from utils import check_password

warnings.filterwarnings('ignore')

# Configuración de página
st.set_page_config(
    page_title="MACD-V Screener",
    page_icon="🔍",
    layout="wide"
)

# ============= FUNCIONES TÉCNICAS =============
def calculate_ema(data, period):
    """Calcula EMA"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_macd_v(df, fast_len=12, slow_len=26, signal_len=9, atr_len=26):
    """Calcula MACD-V (MACD normalizado por ATR)"""
    try:
        fast_ema = calculate_ema(df['Close'], fast_len)
        slow_ema = calculate_ema(df['Close'], slow_len)
        
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=atr_len).mean()
        
        macd = ((fast_ema - slow_ema) / atr) * 100
        signal = calculate_ema(macd, signal_len)
        
        return macd, signal
    except Exception as e:
        return None, None

def analyze_ticker(ticker, period="6mo", interval="1d"):
    """Analiza un ticker y retorna sus métricas de MACD-V"""
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if data.empty or len(data) < 50:
            return None
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        macd_v, macd_v_signal = calculate_macd_v(data)
        
        if macd_v is None or macd_v.isna().all():
            return None
        
        current_macdv = macd_v.iloc[-1]
        current_signal = macd_v_signal.iloc[-1]
        current_price = data['Close'].iloc[-1]
        
        # Filtrar solo valores extremos
        if abs(current_macdv) < 150:
            return None
        
        return {
            'Ticker': ticker,
            'MACD_V': current_macdv,
            'Signal': current_signal,
            'Price': current_price,
            'Date': data.index[-1],
            'Data': data,
            'MACD_V_Series': macd_v,
            'Signal_Series': macd_v_signal
        }
    except Exception as e:
        return None

def scan_tickers(tickers_list, max_workers=10):
    """Escanea múltiples tickers en paralelo"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(tickers_list)
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_ticker, ticker): ticker 
                   for ticker in tickers_list}
        
        for future in as_completed(futures):
            completed += 1
            progress = completed / total
            progress_bar.progress(progress)
            status_text.text(f"Analizando: {completed}/{total} tickers ({progress*100:.1f}%)")
            
            result = future.result()
            if result is not None:
                results.append(result)
    
    progress_bar.empty()
    status_text.empty()
    
    return results

def plot_ticker_chart(ticker_data):
    """Genera gráfico de precio y MACD-V para un ticker"""
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), facecolor='#0E1117',
                                    gridspec_kw={'height_ratios': [3, 1]})
    
    data = ticker_data['Data']
    macd_v = ticker_data['MACD_V_Series']
    signal = ticker_data['Signal_Series']
    ticker = ticker_data['Ticker']
    
    # Gráfico de precio
    ax1.set_facecolor('#1A1D29')
    ax1.plot(data.index, data['Close'], color='#FFFFFF', linewidth=2, label='Precio')
    ax1.set_title(f'{ticker} - Precio: ${ticker_data["Price"]:.2f}', 
                  fontsize=16, fontweight='bold', color='#FFFFFF', pad=20)
    ax1.set_ylabel('Precio ($)', fontsize=12, fontweight='bold', color='#FFFFFF')
    ax1.grid(True, alpha=0.1, linestyle='-', linewidth=0.8, color='#FFFFFF')
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax1.tick_params(labelsize=10, colors='#B0B0B0')
    
    # Gráfico de MACD-V
    ax2.set_facecolor('#1A1D29')
    
    # Colorear línea MACD-V según niveles
    for i in range(1, len(macd_v)):
        if pd.notna(macd_v.iloc[i-1]) and pd.notna(macd_v.iloc[i]):
            y2 = macd_v.iloc[i]
            if y2 > 150:
                color, width = '#FF6B6B', 3
            elif y2 > 50:
                color, width = '#4ECDC4', 2.5
            elif y2 > -50:
                color, width = '#95A5A6', 2
            elif y2 > -150:
                color, width = '#EE5A6F', 2.5
            else:
                color, width = '#FF6B6B', 3
            ax2.plot([data.index[i-1], data.index[i]], 
                    [macd_v.iloc[i-1], y2], 
                    color=color, linewidth=width, alpha=0.95, zorder=5)
    
    # Línea de señal
    ax2.plot(data.index, signal, color='#FFB86C', linewidth=1.5, 
            alpha=0.6, linestyle='--', label='Signal', zorder=3)
    
    # Niveles de referencia
    ax2.axhline(y=150, color='#FF6B6B', linestyle='--', linewidth=2, alpha=0.8)
    ax2.axhline(y=50, color='#4ECDC4', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=0, color='#8E93A1', linestyle='-', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=-50, color='#EE5A6F', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=-150, color='#FF6B6B', linestyle='--', linewidth=2, alpha=0.8)
    
    # Áreas de fondo
    ax2.fill_between(data.index, 150, ax2.get_ylim()[1], alpha=0.12, color='#FF6B6B', zorder=0)
    ax2.fill_between(data.index, ax2.get_ylim()[0], -150, alpha=0.12, color='#FF6B6B', zorder=0)
    ax2.fill_between(data.index, -50, 50, alpha=0.08, color='#95A5A6', zorder=0)
    
    # Valor actual
    current_macdv = ticker_data['MACD_V']
    macd_color = '#FF6B6B' if abs(current_macdv) > 150 else '#4ECDC4' if current_macdv > 50 else '#EE5A6F' if current_macdv < -50 else '#95A5A6'
    ax2.text(0.02, 0.88, f'MACD-V: {current_macdv:.1f}', 
            transform=ax2.transAxes, fontsize=13, fontweight='bold', 
            color='white', verticalalignment='top', 
            bbox=dict(boxstyle='round,pad=0.7', facecolor=macd_color, 
                     alpha=0.95, edgecolor='white', linewidth=2.5))
    
    ax2.set_ylabel('MACD-V', fontsize=12, fontweight='bold', color='#FFFFFF')
    ax2.set_xlabel('Fecha', fontsize=12, fontweight='bold', color='#FFFFFF')
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.1, linestyle=':', linewidth=1, color='#FFFFFF')
    ax2.tick_params(labelsize=10, colors='#B0B0B0')
    
    plt.tight_layout()
    return fig

# ============= INTERFAZ STREAMLIT =============
def main_app():
    """Aplicación principal del screener"""
    st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stMetric { 
        background: linear-gradient(135deg, #1A1D29 0%, #2D3142 100%);
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #4ECDC4;
    }
    .ticker-card {
        background: linear-gradient(135deg, #1A1D29 0%, #2D3142 100%);
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #FFB86C;
        margin: 10px 0;
    }
    h1, h2, h3 { color: #FFFFFF !important; font-weight: 800 !important; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🔍 MACD-V Screener - Valores Extremos")
    st.markdown("**Busca tickers con MACD-V > 150 o < -150**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 20px; 
                    background: linear-gradient(135deg, #4ECDC4 0%, #00D9FF 100%); 
                    border-radius: 15px; margin-bottom: 20px;'>
            <h2 style='color: white; margin: 0;'>⚙️ Configuración</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Botón de reset
        if st.button("🔄 RESET COMPLETO", type="secondary", use_container_width=True):
            for key in ['scan_results', 'selected_ticker']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        st.markdown("---")
        
        period = st.selectbox("Periodo", ["3mo", "6mo", "1y", "2y"], index=1)
        interval = st.selectbox("Intervalo", ["1d", "1wk"], index=0)
        max_workers = st.slider("Threads paralelos", 5, 20, 10, 1)
        
        st.markdown("---")
        
        scan_button = st.button("🚀 ESCANEAR TODOS", type="primary", use_container_width=True)
        
        st.markdown("---")
        
        with st.expander("ℹ️ Información"):
            st.markdown("""
            **Screener MACD-V:**
            - Analiza todos los tickers del CSV
            - Filtra solo valores extremos (|MACD-V| > 150)
            - Muestra ranking ordenado
            - Permite ver gráficos individuales
            
            **Niveles MACD-V:**
            - 🔴 > 150: Sobrecompra extrema
            - 🟢 50-150: Alcista fuerte
            - ⚫ -50 a 50: Neutral
            - 🔴 < -150: Sobreventa extrema
            """)
    
    # Botón de escaneo
    if scan_button:
        st.markdown("### 🔄 Escaneando tickers...")
        
        # Cargar lista de tickers desde la carpeta raíz
        import os
        try:
            # Obtener directorio raíz (un nivel arriba de pages/)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.dirname(current_dir)
            csv_path = os.path.join(root_dir, "Tickers.csv")
            
            tickers_df = pd.read_csv(csv_path)
        except Exception as e:
            st.error(f"❌ No se pudo cargar el archivo Tickers.csv desde la raíz: {str(e)}")
            st.info("📁 Asegúrate de que Tickers.csv está en la carpeta raíz del proyecto")
            st.stop()
        
        tickers_list = tickers_df['Ticker'].tolist()
        st.info(f"📊 Analizando {len(tickers_list)} tickers...")
        
        with st.spinner("Analizando... Esto puede tardar varios minutos."):
            results = scan_tickers(tickers_list, max_workers=max_workers)
        
        if results:
            st.session_state['scan_results'] = results
            st.success(f"✅ Se encontraron {len(results)} tickers con |MACD-V| > 150")
        else:
            st.warning("⚠️ No se encontraron tickers con valores extremos")
    
    # Mostrar resultados
    if 'scan_results' in st.session_state:
        results = st.session_state['scan_results']
        
        st.markdown("---")
        st.markdown("### 📊 Resultados del Escaneo")
        
        # Crear DataFrame de resultados
        df_results = pd.DataFrame([
            {
                'Ticker': r['Ticker'],
                'MACD_V': r['MACD_V'],
                'Signal': r['Signal'],
                'Price': r['Price'],
                'Tipo': '🔴 Sobrecompra' if r['MACD_V'] > 150 else '🟢 Sobreventa'
            }
            for r in results
        ])
        
        # Ordenar por valor absoluto de MACD-V (mayor a menor)
        df_results['MACD_V_Abs'] = df_results['MACD_V'].abs()
        df_results = df_results.sort_values('MACD_V_Abs', ascending=False)
        df_results = df_results.drop('MACD_V_Abs', axis=1)
        
        # Métricas generales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Extremos", len(results))
        with col2:
            sobrecompra = len([r for r in results if r['MACD_V'] > 150])
            st.metric("🔴 Sobrecompra (>150)", sobrecompra)
        with col3:
            sobreventa = len([r for r in results if r['MACD_V'] < -150])
            st.metric("🟢 Sobreventa (<-150)", sobreventa)
        with col4:
            max_macdv = max([abs(r['MACD_V']) for r in results])
            st.metric("Max |MACD-V|", f"{max_macdv:.1f}")
        
        st.markdown("---")
        
        # Tabla de resultados
        st.dataframe(
            df_results.style.format({
                'MACD_V': '{:.1f}',
                'Signal': '{:.1f}',
                'Price': '${:.2f}'
            }),
            use_container_width=True,
            height=400
        )
        
        # Descargar resultados
        csv = df_results.to_csv(index=False)
        st.download_button(
            "📥 Descargar Resultados (CSV)",
            data=csv,
            file_name=f"macdv_screener_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.markdown("---")
        st.markdown("### 📈 Ver Gráfico Individual")
        
        # Selector de ticker
        ticker_options = [f"{r['Ticker']} (MACD-V: {r['MACD_V']:.1f})" for r in results]
        selected_option = st.selectbox("Selecciona un ticker:", ticker_options)
        
        if selected_option:
            selected_ticker = selected_option.split(" ")[0]
            ticker_data = next((r for r in results if r['Ticker'] == selected_ticker), None)
            
            if ticker_data:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Ticker", ticker_data['Ticker'])
                with col2:
                    color = "normal" if ticker_data['MACD_V'] > 0 else "inverse"
                    st.metric("MACD-V", f"{ticker_data['MACD_V']:.1f}", delta_color=color)
                with col3:
                    st.metric("Precio", f"${ticker_data['Price']:.2f}")
                
                # Mostrar gráfico
                fig = plot_ticker_chart(ticker_data)
                st.pyplot(fig)
    
    else:
        st.markdown("""
        <div style='text-align: center; padding: 60px 20px;
                    background: linear-gradient(135deg, #1A1D29 0%, #2D3142 100%);
                    border-radius: 20px; border: 3px solid #4ECDC4;
                    box-shadow: 0 8px 30px rgba(78, 205, 196, 0.3);
                    margin-top: 40px;'>
            <h2 style='color: #4ECDC4; margin: 0;'>
                👋 Bienvenido al MACD-V Screener
            </h2>
            <p style='color: #B0B0B0; font-size: 18px; margin: 20px 0;'>
                Presiona "🚀 ESCANEAR TODOS" para comenzar
            </p>
            <p style='color: #8E93A1; font-size: 14px;'>
                ✨ Detecta valores extremos (|MACD-V| > 150)<br>
                ✨ Ranking automático<br>
                ✨ Gráficos individuales
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    if check_password():
        main_app()
    else:
        st.stop()
