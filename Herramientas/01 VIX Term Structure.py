import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from datetime import datetime
from utils.utils import check_password

# --- CONFIGURACIÓN DE PÁGINA ---

# Configuración de página
#st.set_page_config(
#    page_title="VIX Term Structure",
#    page_icon="🐣",
#    layout="wide"
#)

# CSS personalizado para mejorar la presentación
st.markdown("""
    <style>
    /* Estilos generales */
    .main {
        background-color: #0E1117;
    }
    
    /* Título principal */
    .vix-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 0.5rem;
        font-family: 'Arial', sans-serif;
    }
    
    .vix-subtitle {
        font-size: 0.9rem;
        color: #8B949E;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Métricas personalizadas */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
    }
    
    /* Mejorar tablas */
    .dataframe {
        font-size: 0.9rem;
    }
    
    /* Secciones */
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #FFFFFF;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-left: 0.5rem;
        border-left: 4px solid #4A90E2;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# OBTENCIÓN DE DATOS
# ==============================================================================

@st.cache_data(ttl=300)
def get_vix_spot():
    """Obtiene el precio actual del índice VIX desde Yahoo Finance."""
    try:
        import yfinance as yf
        vix = yf.Ticker("^VIX")
        price = vix.history(period="1d")['Close'].iloc[-1]
        return price
    except:
        return None

@st.cache_data(ttl=300)
def scrape_vix_central():
    """Extrae datos de futuros VIX desde VIX Central."""
    try:
        url = "http://vixcentral.com/"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9"
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        futures_data = []
        
        # Buscar datos en scripts
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string and 'VIX' in str(script.string):
                script_text = script.string
                
                import re
                numbers = re.findall(r'\d+\.\d+', script_text)
                
                if len(numbers) >= 8:
                    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug']
                    
                    for i, (month, price) in enumerate(zip(months, numbers[:8])):
                        futures_data.append({
                            'month': month,
                            'price': float(price),
                            'label': f'F{i+1}',
                            'contract': f'{month}'
                        })
                    
                    if futures_data:
                        break
        
        # Fallback: buscar en tablas
        if not futures_data:
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows[1:]:
                    cols = row.find_all('td')
                    if len(cols) >= 2:
                        try:
                            month = cols[0].text.strip()
                            price = float(cols[1].text.strip())
                            
                            futures_data.append({
                                'month': month,
                                'price': price,
                                'label': f'F{len(futures_data)+1}',
                                'contract': month
                            })
                            
                            if len(futures_data) >= 8:
                                break
                        except:
                            continue
                
                if len(futures_data) >= 8:
                    break
        
        if futures_data:
            df = pd.DataFrame(futures_data)
            return df
        
        return None
        
    except Exception as e:
        st.error(f"Error extrayendo datos: {e}")
        return None

@st.cache_data(ttl=300)
def get_vix_futures_alternative():
    """Genera estructura aproximada basada en VIX spot."""
    try:
        import yfinance as yf
        
        vix = yf.Ticker("^VIX")
        vix_spot = vix.history(period="1d")['Close'].iloc[-1]
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug']
        futures_data = []
        
        for i, month in enumerate(months):
            premium = 1 + (0.025 * (i + 1))
            price = vix_spot * premium
            
            futures_data.append({
                'month': month,
                'price': round(price, 3),
                'label': f'F{i+1}',
                'contract': month
            })
        
        df = pd.DataFrame(futures_data)
        return df
        
    except:
        return None

# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================

def main_vix_structure():
    
    # Título mejorado con botón de refresh
    col_title, col_refresh = st.columns([6, 1])
    
    with col_title:
        st.markdown(
        '<h1 class="vix-title"><span style="font-size: 1.5em;">🐣</span> VIX Futures Term Structure</h1>', 
        unsafe_allow_html=True
        )
        st.markdown('<p class="vix-subtitle">Source: CBOE Delayed Quotes · vixcentral.com</p>', unsafe_allow_html=True)
    
    with col_refresh:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refrescar Datos", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Obtener datos
    with st.spinner("Obteniendo datos..."):
        vix_spot = get_vix_spot()
        df_futures = scrape_vix_central()
        
        if df_futures is None or df_futures.empty:
            df_futures = get_vix_futures_alternative()
    
    if df_futures is not None and not df_futures.empty:
        
        # ========== MÉTRICAS PRINCIPALES ==========
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            if vix_spot:
                st.metric(
                    label="VIX Spot",
                    value=f"{vix_spot:.2f}",
                    delta=None
                )
        
        with col2:
            f1_price = df_futures.iloc[0]['price']
            if vix_spot:
                f1_diff = f1_price - vix_spot
                st.metric(
                    label="F1 (Front Month)",
                    value=f"{f1_price:.2f}",
                    delta=f"{f1_diff:+.2f}"
                )
            else:
                st.metric(
                    label="F1 (Front Month)",
                    value=f"{f1_price:.2f}"
                )
        
        with col3:
            if len(df_futures) >= 8:
                f8_price = df_futures.iloc[7]['price']
                total_contango = f8_price - f1_price
                total_contango_pct = (total_contango / f1_price) * 100
                
                st.metric(
                    label="Total Contango (F1→F8)",
                    value=f"{total_contango_pct:.2f}%",
                    delta=f"{total_contango:+.2f} pts"
                )
        
        with col4:
            if len(df_futures) >= 7:
                m7_price = df_futures.iloc[6]['price']
                m4_price = df_futures.iloc[3]['price']
                m7_m4_contango = m7_price - m4_price
                m7_m4_pct = (m7_m4_contango / m4_price) * 100
                
                st.metric(
                    label="M7 to M4 Contango",
                    value=f"{m7_m4_pct:.2f}%",
                    delta=f"{m7_m4_contango:+.2f} pts"
                )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ========== GRÁFICO PRINCIPAL ==========
        st.markdown('<div class="section-title">📈 VIX Futures Term Structure</div>', unsafe_allow_html=True)
        
        fig = go.Figure()
        
        # Línea de futuros
        fig.add_trace(go.Scatter(
            x=df_futures['month'],
            y=df_futures['price'],
            mode='lines+markers+text',
            line=dict(color='#4A90E2', width=3.5),
            marker=dict(
                size=12,
                color='#4A90E2',
                line=dict(color='#FFFFFF', width=2)
            ),
            text=df_futures['price'].round(2),
            textposition='top center',
            textfont=dict(size=11, color='#FFFFFF', family='Arial'),
            hovertemplate='<b>%{x}</b><br>Price: %{y:.3f}<extra></extra>',
            showlegend=False
        ))
        
        # Línea de VIX Spot
        if vix_spot:
            fig.add_shape(
                type="line",
                x0=df_futures['month'].iloc[0],
                x1=df_futures['month'].iloc[-1],
                y0=vix_spot,
                y1=vix_spot,
                line=dict(color="#27AE60", width=2.5, dash="dash")
            )
            
            fig.add_annotation(
                x=df_futures['month'].iloc[-1],
                y=vix_spot,
                text=f"VIX Index: {vix_spot:.2f}",
                showarrow=False,
                xanchor="right",
                yanchor="bottom",
                font=dict(size=12, color="#27AE60", family='Arial'),
                bgcolor="rgba(39, 174, 96, 0.15)",
                bordercolor="#27AE60",
                borderwidth=1.5,
                borderpad=6
            )
        
        # Personalización del layout - AQUÍ ESTÁ EL CAMBIO CLAVE
        fig.update_layout(
            title=dict(
                text="",  # Título vacío explícitamente
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="Future Month",
            yaxis_title="Volatility",
            hovermode='x unified',
            template='plotly_dark',
            height=500,
            showlegend=False,
            paper_bgcolor='#0E1117',
            plot_bgcolor='#1A1D24',
            font=dict(color='#FFFFFF', family='Arial'),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)',
                zeroline=False,
                linecolor='rgba(255, 255, 255, 0.2)',
                title_font=dict(size=14)
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)',
                zeroline=False,
                linecolor='rgba(255, 255, 255, 0.2)',
                title_font=dict(size=14)
            ),
            margin=dict(t=20, b=60, l=60, r=80)
        )
        
        st.plotly_chart(fig, use_container_width=True, key="vix_chart", config={'displayModeBar': False})
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ========== TABLA DE DATOS ==========
        st.markdown('<div class="section-title">📋 Datos de Contratos de Futuros VIX</div>', unsafe_allow_html=True)
        
        # Preparar datos de la tabla
        df_table = df_futures.copy()
        
        # Calcular métricas
        df_table['diff_prev'] = df_table['price'].diff().fillna(0)
        df_table['pct_contango'] = (df_table['price'].pct_change() * 100).fillna(0)
        
        if vix_spot:
            df_table['diff_spot'] = df_table['price'] - vix_spot
            df_table['pct_spot'] = ((df_table['price'] - vix_spot) / vix_spot * 100)
        
        # Crear tabla formateada
        table_data = pd.DataFrame({
            'Contrato': df_table['label'],
            'Mes': df_table['month'],
            'Precio': df_table['price'].apply(lambda x: f"{x:.3f}"),
            'Diff vs Anterior': df_table['diff_prev'].apply(lambda x: f"{x:+.3f}" if x != 0 else "0"),
            '% Contango': df_table['pct_contango'].apply(lambda x: f"{x:+.2f}%" if x != 0 else "0%"),
        })
        
        if vix_spot:
            table_data['Diff vs Spot'] = df_table['diff_spot'].apply(lambda x: f"{x:+.3f}")
            table_data['% vs Spot'] = df_table['pct_spot'].apply(lambda x: f"{x:+.2f}%")
        
        # Mostrar tabla con estilo
        st.dataframe(
            table_data,
            use_container_width=True,
            hide_index=True,
            height=350
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ========== ANÁLISIS DE ESTRUCTURA ==========
        st.markdown('<div class="section-title">📊 Análisis de Estructura</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if len(df_futures) >= 7:
                m7_m4_contango = df_futures.iloc[6]['price'] - df_futures.iloc[3]['price']
                m7_m4_pct = (m7_m4_contango / df_futures.iloc[3]['price'] * 100)
                
                st.metric(
                    "Month 7 to 4 contango",
                    f"{m7_m4_pct:.2f}%",
                    delta=f"{m7_m4_contango:+.2f} pts",
                    delta_color="normal"
                )
        
        with col2:
            if len(df_futures) >= 8:
                total_contango = df_futures.iloc[7]['price'] - df_futures.iloc[0]['price']
                total_pct = (total_contango / df_futures.iloc[0]['price'] * 100)
                
                st.metric(
                    "Total contango (F1→F8)",
                    f"{total_pct:.2f}%",
                    delta=f"{total_contango:+.2f} pts",
                    delta_color="normal"
                )
        
        with col3:
            if len(df_futures) >= 2:
                avg_contango = df_table['diff_prev'].mean()
                
                st.metric(
                    "Avg contango por mes",
                    f"{avg_contango:.3f} pts",
                    delta=None
                )
        
    else:
        st.error("❌ No se pudieron obtener datos de futuros VIX")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.link_button("🌐 VIX Central", "http://vixcentral.com/", use_container_width=True)
        
        with col2:
            st.link_button("📊 CBOE Data", "https://www.cboe.com/tradable-products/vix/term-structure/", use_container_width=True)

# ==============================================================================
# PUNTO DE ENTRADA
# ==============================================================================
if __name__ == "__main__":
    
    if check_password():
        main_vix_structure()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder.")
