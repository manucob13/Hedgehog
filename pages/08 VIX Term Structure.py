import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from datetime import datetime
from utils import check_password

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="VIX Term Structure - VIX Central", layout="wide")

# ==============================================================================
# OBTENCIÓN DE DATOS DESDE VIX CENTRAL (WEB SCRAPING)
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
    """Extrae datos de futuros VIX desde VIX Central mediante web scraping."""
    try:
        url = "http://vixcentral.com/"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9"
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Parsear HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Buscar la tabla con los datos de futuros
        # VIX Central tiene los datos en un script o tabla específica
        # Intentar encontrar los datos en el HTML
        
        futures_data = []
        
        # Método 1: Buscar en scripts que contengan los datos
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string and 'VIX' in str(script.string):
                # Intentar extraer datos del script
                script_text = script.string
                
                # VIX Central suele tener los datos en formato JavaScript
                # Buscar patrones como: data = [16.5, 18.49, 19.68, ...]
                import re
                
                # Buscar arrays de números
                numbers = re.findall(r'\d+\.\d+', script_text)
                
                if len(numbers) >= 8:
                    # Tomar los primeros 8-9 valores que suelen ser los futuros
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
        
        # Método 2: Buscar en tabla HTML directamente
        if not futures_data:
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows[1:]:  # Saltar header
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
        st.error(f"Error extrayendo datos de VIX Central: {e}")
        return None

@st.cache_data(ttl=300)
def get_vix_futures_alternative():
    """Método alternativo: extraer de otro sitio o usar datos simulados para demo."""
    try:
        import yfinance as yf
        
        # Obtener VIX spot como referencia
        vix = yf.Ticker("^VIX")
        vix_spot = vix.history(period="1d")['Close'].iloc[-1]
        
        # Construir una curva aproximada basada en promedios históricos
        # Típicamente los futuros VIX cotizan en contango
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug']
        
        # Crear estructura de contango típica (esto es aproximado)
        # Usualmente F1 está 10-15% sobre spot, y aumenta gradualmente
        futures_data = []
        
        for i, month in enumerate(months):
            # Contango típico: aumenta ~2-3% por mes
            premium = 1 + (0.025 * (i + 1))
            price = vix_spot * premium
            
            futures_data.append({
                'month': month,
                'price': round(price, 2),
                'label': f'F{i+1}',
                'contract': month
            })
        
        df = pd.DataFrame(futures_data)
        return df
        
    except:
        return None

# ==============================================================================
# FUNCIÓN PRINCIPAL (CONTENIDO DE LA APP)
# ==============================================================================

def main_vix_structure():
    st.markdown("<h1 style='text-align: center;'>📊 VIX Futures Term Structure</h1>", unsafe_allow_html=True)
    st.caption("Fuente: VIX Central (vixcentral.com)")
    st.markdown("---")
    
    # Obtener datos
    with st.spinner("Obteniendo datos de futuros VIX..."):
        vix_spot = get_vix_spot()
        df_futures = scrape_vix_central()
        
        # Si falla el scraping, usar método alternativo
        if df_futures is None or df_futures.empty:
            st.warning("⚠️ No se pudo acceder a VIX Central. Usando estructura aproximada...")
            df_futures = get_vix_futures_alternative()
    
    # Mostrar métricas principales
    if df_futures is not None and not df_futures.empty:
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if vix_spot:
                st.metric("VIX Spot", f"{vix_spot:.2f}")
            else:
                st.metric("VIX Spot", "N/A")
        
        with col2:
            f1_price = df_futures.iloc[0]['price']
            st.metric("F1 (Front Month)", f"{f1_price:.2f}", delta=df_futures.iloc[0]['month'])
        
        with col3:
            if vix_spot:
                contango = f1_price - vix_spot
                contango_pct = (contango / vix_spot) * 100
                st.metric(
                    "Contango/Backwardation", 
                    f"{contango:.2f}",
                    delta=f"{contango_pct:.1f}%"
                )
            else:
                st.metric("Contango", "N/A")
        
        st.markdown("---")
        
        # Gráfico principal - ESTILO VIX CENTRAL
        st.subheader("📈 VIX Futures Term Structure")
        
        fig = go.Figure()
        
        # Línea de futuros (línea azul como en VIX Central)
        fig.add_trace(go.Scatter(
            x=df_futures['month'],
            y=df_futures['price'],
            mode='lines+markers+text',
            name='VIX Futures',
            line=dict(color='#4A90E2', width=3),
            marker=dict(size=10, color='#4A90E2'),
            text=df_futures['price'].round(2),
            textposition='top center',
            textfont=dict(size=11, color='#333'),
            hovertemplate='<b>%{x}</b><br>Precio: %{y:.3f}<extra></extra>'
        ))
        
        # Línea horizontal de VIX Spot (línea verde punteada como en VIX Central)
        if vix_spot:
            fig.add_hline(
                y=vix_spot,
                line_dash="dash",
                line_color="#27AE60",
                line_width=2,
                annotation_text=f"VIX Index: {vix_spot:.2f}",
                annotation_position="right",
                annotation=dict(font=dict(size=12, color="#27AE60"))
            )
        
        # Personalización para que se vea como VIX Central
        fig.update_layout(
            title={
                'text': "VIX Futures Term Structure",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#333'}
            },
            xaxis_title="Future Month",
            yaxis_title="Volatility",
            hovermode='x unified',
            template='plotly_white',
            height=550,
            showlegend=False,
            plot_bgcolor='#FAFAFA',
            xaxis=dict(
                showgrid=True,
                gridcolor='#E0E0E0',
                zeroline=False
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#E0E0E0',
                zeroline=False
            ),
            margin=dict(t=80, b=60, l=60, r=60)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabla de datos detallada
        st.subheader("📋 Datos de Contratos de Futuros VIX")
        
        # Calcular contango/backwardation para cada contrato
        df_display = df_futures.copy()
        
        # Calcular diferencias y porcentajes
        if vix_spot:
            df_display['diff_spot'] = df_display['price'] - vix_spot
            df_display['pct_spot'] = ((df_display['price'] - vix_spot) / vix_spot * 100).round(2)
        
        # Calcular diferencia entre contratos consecutivos
        df_display['diff_prev'] = df_display['price'].diff().round(3)
        df_display['pct_contango'] = (df_display['price'].pct_change() * 100).round(2)
        
        # Preparar tabla para mostrar
        table_data = pd.DataFrame({
            'Contrato': df_display['label'],
            'Mes': df_display['month'],
            'Precio': df_display['price'].round(3),
            'Diff vs Anterior': df_display['diff_prev'].fillna(0).round(3),
            '% Contango': df_display['pct_contango'].fillna(0).round(2),
        })
        
        if vix_spot:
            table_data['Diff vs Spot'] = df_display['diff_spot'].round(3)
            table_data['% vs Spot'] = df_display['pct_spot'].round(2)
        
        st.dataframe(
            table_data,
            use_container_width=True,
            hide_index=True
        )
        
        # Análisis de contango
        st.markdown("---")
        st.subheader("📊 Análisis de Estructura")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Contango M7 a M4
            if len(df_futures) >= 7:
                m7_m4_contango = df_futures.iloc[6]['price'] - df_futures.iloc[3]['price']
                m7_m4_pct = (m7_m4_contango / df_futures.iloc[3]['price'] * 100)
                
                st.metric(
                    "Month 7 to 4 contango",
                    f"{m7_m4_pct:.2f}%",
                    delta=f"{m7_m4_contango:.2f} pts"
                )
        
        with col2:
            # Contango total (M1 a M8)
            if len(df_futures) >= 8:
                total_contango = df_futures.iloc[7]['price'] - df_futures.iloc[0]['price']
                total_pct = (total_contango / df_futures.iloc[0]['price'] * 100)
                
                st.metric(
                    "Total contango (F1-F8)",
                    f"{total_pct:.2f}%",
                    delta=f"{total_contango:.2f} pts"
                )
        
        with col3:
            # Promedio de contango por mes
            if len(df_futures) >= 2:
                avg_contango = df_display['diff_prev'].mean()
                
                st.metric(
                    "Avg contango por mes",
                    f"{avg_contango:.3f} pts"
                )
        
    else:
        st.error("❌ No se pudieron obtener datos de futuros VIX")
        
        st.markdown("### 💡 Visita VIX Central directamente")
        st.link_button("🌐 VIX Central", "http://vixcentral.com/")
        
        st.info("""
        **Nota**: Para obtener datos de contratos individuales de futuros VIX necesitas:
        
        1. **VIX Central** (vixcentral.com) - Gratuito, actualización diaria
        2. **CBOE DataShop** - Comercial, datos en tiempo real
        3. **Broker con acceso a datos** - Interactive Brokers, TD Ameritrade
        4. **APIs comerciales** - Bloomberg, Quandl, IVolatility
        """)
    
    # Footer con información
    st.markdown("---")
    with st.expander("ℹ️ Sobre los Datos y Metodología"):
        st.markdown("""
        ### Fuente de Datos
        
        Los datos son extraídos de **VIX Central** (vixcentral.com), que obtiene cotizaciones 
        de futuros VIX de CBOE y las actualiza diariamente.
        
        ### Contratos de Futuros VIX
        
        - **Símbolo**: VX en CBOE
        - **Tamaño**: $1,000 × nivel del índice VIX  
        - **Expiración**: Miércoles, 30 días antes del 3er viernes del mes
        - **Meses**: Los próximos 8 meses de contratos
        
        ### Interpretación
        
        - **Contango (positivo)**: Futuros cotizan sobre el spot → mercado espera volatilidad creciente
        - **Backwardation (negativo)**: Futuros bajo el spot → mercado espera volatilidad decreciente
        - **Curva pronunciada**: Mayor incertidumbre a largo plazo
        - **Curva plana**: Expectativas estables
        
        ### Actualización
        
        Los datos se actualizan cada 5 minutos (cache de 300 segundos).
        """)

# ==============================================================================
# PUNTO DE ENTRADA PROTEGIDO
# ==============================================================================
if __name__ == "__main__":
    
    if check_password():
        main_vix_structure()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder a VIX Term Structure.")
