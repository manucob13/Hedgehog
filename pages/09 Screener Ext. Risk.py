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
    """Genera gráfico de precio y MACD-V para un ticker - MEJORADO"""
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), facecolor='#0E1117',
                                    gridspec_kw={'height_ratios': [2.5, 1], 'hspace': 0.15})
    
    data = ticker_data['Data']
    macd_v = ticker_data['MACD_V_Series']
    signal = ticker_data['Signal_Series']
    ticker = ticker_data['Ticker']
    
    # ============= GRÁFICO DE PRECIO =============
    ax1.set_facecolor('#1A1D29')
    ax1.plot(data.index, data['Close'], color='#FFFFFF', linewidth=2.5, label='Precio', zorder=3)
    
    ax1.set_title(f'{ticker} - ${ticker_data["Price"]:.2f}', 
                  fontsize=20, fontweight='bold', color='#FFFFFF', pad=15)
    ax1.set_ylabel('Precio ($)', fontsize=13, fontweight='bold', color='#FFFFFF')
    ax1.grid(True, alpha=0.1, linestyle='-', linewidth=0.8, color='#FFFFFF')
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax1.tick_params(labelsize=11, colors='#B0B0B0', labelbottom=False)
    
    for spine in ax1.spines.values():
        spine.set_color('#2D3142')
        spine.set_linewidth(1.5)
    
    # ============= GRÁFICO DE MACD-V (MEJORADO) =============
    ax2.set_facecolor('#1A1D29')
    
    # Colorear línea MACD-V según niveles
    for i in range(1, len(macd_v)):
        if pd.notna(macd_v.iloc[i-1]) and pd.notna(macd_v.iloc[i]):
            y2 = macd_v.iloc[i]
            if y2 > 150:
                color, width = '#FF6B6B', 3.5
            elif y2 > 50:
                color, width = '#4ECDC4', 3
            elif y2 > -50:
                color, width = '#95A5A6', 2.5
            elif y2 > -150:
                color, width = '#EE5A6F', 3
            else:
                color, width = '#FF6B6B', 3.5
            ax2.plot([data.index[i-1], data.index[i]], 
                    [macd_v.iloc[i-1], y2], 
                    color=color, linewidth=width, alpha=0.95, zorder=5)
    
    # Línea de señal (sin label para evitar que tape el valor)
    ax2.plot(data.index, signal, color='#FFB86C', linewidth=1.8, 
            alpha=0.5, linestyle='--', zorder=3)
    
    # Niveles de referencia
    ax2.axhline(y=150, color='#FF6B6B', linestyle='--', linewidth=2, alpha=0.8)
    ax2.axhline(y=50, color='#4ECDC4', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=0, color='#8E93A1', linestyle='-', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=-50, color='#EE5A6F', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=-150, color='#FF6B6B', linestyle='--', linewidth=2, alpha=0.8)
    
    # Áreas de fondo
    y_max = ax2.get_ylim()[1]
    y_min = ax2.get_ylim()[0]
    ax2.fill_between(data.index, 150, y_max, alpha=0.12, color='#FF6B6B', zorder=0)
    ax2.fill_between(data.index, y_min, -150, alpha=0.12, color='#FF6B6B', zorder=0)
    ax2.fill_between(data.index, -50, 50, alpha=0.08, color='#95A5A6', zorder=0)
    
    # Valor actual en esquina superior izquierda
    current_macdv = ticker_data['MACD_V']
    macd_color = '#FF6B6B' if abs(current_macdv) > 150 else '#4ECDC4' if current_macdv > 50 else '#EE5A6F' if current_macdv < -50 else '#95A5A6'
    ax2.text(0.02, 0.95, f'MACD-V: {current_macdv:.1f}', 
            transform=ax2.transAxes, fontsize=14, fontweight='bold', 
            color='white', verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.7', facecolor=macd_color, 
                     alpha=0.95, edgecolor='white', linewidth=2))
    
    # Leyenda de Signal en esquina superior derecha
    ax2.text(0.98, 0.95, 'Signal', 
            transform=ax2.transAxes, fontsize=11, 
            color='#FFB86C', verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#1A1D29', 
                     alpha=0.8, edgecolor='#FFB86C', linewidth=1))
    
    ax2.set_ylabel('MACD-V', fontsize=13, fontweight='bold', color='#FFFFFF')
    ax2.set_xlabel('Fecha', fontsize=13, fontweight='bold', color='#FFFFFF')
    ax2.grid(True, alpha=0.1, linestyle=':', linewidth=1, color='#FFFFFF')
    ax2.tick_params(labelsize=11, colors='#B0B0B0')
    
    for spine in ax2.spines.values():
        spine.set_color('#2D3142')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    return fig

# ============= INTERFAZ STREAMLIT =============
def main_app():
    """Aplicación principal del screener"""
    st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .ticker-card {
        background: linear-gradient(135deg, #1A1D29 0%, #2D3142 100%);
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid;
        margin: 8px 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .ticker-card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 15px rgba(78, 205, 196, 0.3);
    }
    .ticker-card-bullish { border-left-color: #FF6B6B; }
    .ticker-card-bearish { border-left-color: #4ECDC4; }
    .stButton>button { 
        background: linear-gradient(135deg, #4ECDC4 0%, #00D9FF 100%); 
        color: white; 
        font-weight: 700; 
        border: none; 
        padding: 10px 20px; 
        border-radius: 8px; 
        transition: all 0.3s ease; 
    }
    .stButton>button:hover { 
        transform: translateY(-2px); 
        box-shadow: 0 6px 20px rgba(78, 205, 196, 0.6); 
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
            - Click en ticker para ver gráfico
            
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
        
        st.markdown("---")
        
        # Layout: Lista de tickers (izquierda) y Gráfico (derecha)
        col_list, col_chart = st.columns([1, 3])
        
        with col_list:
            st.markdown("### 📋 Tickers Detectados")
            
            # Métricas resumen
            sobrecompra = len([r for r in results if r['MACD_V'] > 150])
            sobreventa = len([r for r in results if r['MACD_V'] < -150])
            
            st.markdown(f"""
            <div style='background: #1A1D29; padding: 15px; border-radius: 10px; margin-bottom: 15px;'>
                <div style='display: flex; justify-content: space-between;'>
                    <div style='text-align: center;'>
                        <div style='color: #FF6B6B; font-size: 24px; font-weight: bold;'>{sobrecompra}</div>
                        <div style='color: #B0B0B0; font-size: 12px;'>Sobrecompra</div>
                    </div>
                    <div style='text-align: center;'>
                        <div style='color: #4ECDC4; font-size: 24px; font-weight: bold;'>{sobreventa}</div>
                        <div style='color: #B0B0B0; font-size: 12px;'>Sobreventa</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Lista de tickers clickeable
            for idx, row in df_results.iterrows():
                ticker_data = next((r for r in results if r['Ticker'] == row['Ticker']), None)
                if ticker_data:
                    color_class = 'bullish' if row['MACD_V'] > 150 else 'bearish'
                    color_value = '#FF6B6B' if row['MACD_V'] > 150 else '#4ECDC4'
                    
                    if st.button(
                        f"{row['Ticker']} • {row['MACD_V']:.1f}",
                        key=f"btn_{row['Ticker']}",
                        use_container_width=True
                    ):
                        st.session_state['selected_ticker'] = ticker_data
                    
                    st.markdown(f"""
                    <div class='ticker-card ticker-card-{color_class}' style='border-left-color: {color_value};'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <div style='font-size: 16px; font-weight: bold; color: white;'>{row['Ticker']}</div>
                                <div style='font-size: 11px; color: #8E93A1;'>${row['Price']:.2f}</div>
                            </div>
                            <div style='text-align: right;'>
                                <div style='font-size: 18px; font-weight: bold; color: {color_value};'>{row['MACD_V']:.0f}</div>
                                <div style='font-size: 10px; color: #B0B0B0;'>MACD-V</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Botón de descarga
            st.markdown("---")
            csv = df_results.to_csv(index=False)
            st.download_button(
                "📥 Descargar CSV",
                data=csv,
                file_name=f"macdv_screener_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_chart:
            # Mostrar gráfico del ticker seleccionado
            if 'selected_ticker' in st.session_state:
                ticker_data = st.session_state['selected_ticker']
                
                st.markdown(f"### 📈 {ticker_data['Ticker']}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Precio", f"${ticker_data['Price']:.2f}")
                with col2:
                    color = "normal" if ticker_data['MACD_V'] > 0 else "inverse"
                    st.metric("MACD-V", f"{ticker_data['MACD_V']:.1f}", delta_color=color)
                with col3:
                    st.metric("Signal", f"{ticker_data['Signal']:.1f}")
                
                # Mostrar gráfico
                fig = plot_ticker_chart(ticker_data)
                st.pyplot(fig)
            else:
                st.markdown("""
                <div style='text-align: center; padding: 80px 20px;
                            background: linear-gradient(135deg, #1A1D29 0%, #2D3142 100%);
                            border-radius: 15px; border: 2px dashed #4ECDC4;'>
                    <h2 style='color: #4ECDC4; margin: 0;'>👈 Selecciona un ticker</h2>
                    <p style='color: #8E93A1; margin-top: 15px;'>
                        Click en cualquier ticker de la lista para ver su gráfico
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
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
                ✨ Gráficos interactivos
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    if check_password():
        main_app()
    else:
        st.markdown("""
        <div style='text-align: center; padding: 60px 20px;'>
            <h1 style='color: #FF6B6B; font-size: 48px;'>🔒 Acceso Restringido</h1>
            <p style='color: #B0B0B0; font-size: 20px; margin-top: 20px;'>
                Introduce tus credenciales en el menú lateral para acceder al screener.
            </p>
        </div>
        """, unsafe_allow_html=True)
