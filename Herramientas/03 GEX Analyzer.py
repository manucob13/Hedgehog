"""
GEX Analyzer + Max Pain - Versión Final Optimizada
Desarrollado por @Gsnchez - bquantfinance.com
"""

import streamlit as st
import json
import os
from datetime import timedelta, datetime
from typing import Tuple, Optional
import warnings
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import requests
from plotly.subplots import make_subplots
from scipy import interpolate

warnings.filterwarnings('ignore')

# ============================================================================
# AUTENTICACIÓN - Importar desde utils
# ============================================================================

try:
    from utils.utils import check_password
except ImportError:
    # Fallback si no existe el módulo utils
    def check_password():
        """Verifica si el usuario tiene credenciales correctas."""
        
        def login_form():
            """Formulario de login en el sidebar."""
            with st.sidebar:
                st.markdown("### 🔑 Iniciar Sesión")
                username = st.text_input("Usuario", key="username_input")
                password = st.text_input("Contraseña", type="password", key="password_input")
                
                if st.button("Login", use_container_width=True):
                    if username in st.secrets.get("passwords", {}) and password == st.secrets["passwords"][username]:
                        st.session_state["password_correct"] = True
                        st.session_state["username"] = username
                        st.rerun()
                    else:
                        st.error("❌ Usuario o contraseña incorrectos")
        
        # Si ya está autenticado, mostrar usuario en sidebar
        if st.session_state.get("password_correct", False):
            with st.sidebar:
                st.success(f"✅ Sesión activa: {st.session_state.get('username', 'Usuario')}")
                if st.button("🚪 Cerrar Sesión", use_container_width=True):
                    st.session_state["password_correct"] = False
                    if "username" in st.session_state:
                        del st.session_state["username"]
                    st.rerun()
            return True
        
        # Mostrar formulario de login
        login_form()
        return False

# Constantes
CONTRACT_SIZE = 100
CACHE_DIR = "data"
API_BASE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options"

# Inicializar estado de la sesión
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'ticker_data' not in st.session_state:
    st.session_state.ticker_data = {}

def ensure_cache_dir():
    """Crear directorio de caché si no existe"""
    os.makedirs(CACHE_DIR, exist_ok=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_option_data(ticker: str) -> Optional[dict]:
    """Obtener datos de opciones desde CBOE API con caché"""
    ensure_cache_dir()
    
    urls = [
        f"{API_BASE_URL}/_{ticker}.json",
        f"{API_BASE_URL}/{ticker}.json"
    ]
    
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
        except:
            continue
    return None

def parse_option_data(raw_data: dict) -> Tuple[float, pd.DataFrame]:
    """Parsear datos crudos de opciones"""
    try:
        data = pd.DataFrame.from_dict(raw_data)
        spot_price = float(data.loc["current_price", "data"])
        option_data = pd.DataFrame(data.loc["options", "data"])
        return spot_price, option_data
    except Exception as e:
        st.error(f"Error parseando datos: {e}")
        return 0, pd.DataFrame()

def process_option_data_optimized(data: pd.DataFrame) -> pd.DataFrame:
    """OPTIMIZADO: Procesar y limpiar datos de opciones usando vectorización"""
    df = data.copy()
    
    # Extracción vectorizada más eficiente
    df["type"] = df.option.str.extract(r'\d([CP])\d')
    df["strike_raw"] = df.option.str.extract(r'[CP](\d+)').astype(float)
    df["strike"] = df["strike_raw"] / 1000
    df["expiration_str"] = df.option.str.extract(r'[A-Z]+(\d{6})')
    df["expiration"] = pd.to_datetime(df["expiration_str"], format="%y%m%d", errors='coerce')
    
    # Conversión de columnas numéricas en batch
    numeric_cols = ['gamma', 'open_interest', 'volume', 'delta', 'vega', 'theta', 'iv']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Filtrado único y eficiente
    mask = (
        df['type'].notna() & 
        df['strike'].notna() & 
        df['expiration'].notna() & 
        df['gamma'].notna() & 
        df['open_interest'].notna() &
        (df['open_interest'] > 0) & 
        (df['gamma'] > 0)
    )
    
    return df[mask]

def calculate_gex_optimized(spot: float, data: pd.DataFrame, dealer_position: str = "standard") -> pd.DataFrame:
    """OPTIMIZADO: Cálculo vectorizado de GEX"""
    df = data.copy()
    
    # Cálculo vectorizado de GEX
    df["GEX"] = df["gamma"] * df["open_interest"] * CONTRACT_SIZE * (spot ** 2) * 0.01
    
    # Aplicación vectorizada del signo
    if dealer_position == "standard":
        df["GEX"] = np.where(df["type"] == "P", -df["GEX"], df["GEX"])
    elif dealer_position == "inverse":
        df["GEX"] = np.where(df["type"] == "P", df["GEX"], -df["GEX"])
    
    # Cálculos adicionales vectorizados
    total_gex = df["GEX"].sum()
    df["GEX_pct"] = (df["GEX"] / total_gex * 100) if total_gex != 0 else 0
    df["days_to_expiry"] = (df["expiration"] - pd.Timestamp.now()).dt.days
    
    return df

def calculate_gamma_profile(data: pd.DataFrame, spot: float, dealer_position: str = "standard") -> dict:
    """OPTIMIZADO: Calcular el perfil gamma usando vectorización completa"""
    
    # Obtener strikes únicos y ordenados
    strikes = np.sort(data['strike'].unique())
    
    # Crear rango de strikes para el perfil
    strike_min = strikes.min()
    strike_max = strikes.max()
    
    # Crear array de strikes para evaluación (200 puntos para suavidad)
    profile_strikes = np.linspace(strike_min, strike_max, 200)
    
    # VECTORIZACIÓN: Preparar arrays de datos
    option_strikes = data['strike'].values
    option_gammas = data['gamma'].values
    option_oi = data['open_interest'].values
    option_types = data['type'].values
    
    # Crear matriz de distancias (broadcasting)
    # Forma: [n_profile_strikes, n_options]
    distances = np.abs(profile_strikes[:, np.newaxis] - option_strikes[np.newaxis, :])
    
    # Ancho de la distribución gamma (5% del strike)
    widths = option_strikes * 0.05
    
    # Calcular contribuciones gamma usando broadcasting
    # Función gaussiana vectorizada
    gamma_contributions = option_gammas * np.exp(-(distances**2) / (2 * widths**2))
    
    # Multiplicar por open interest y factor de escala
    gamma_values = gamma_contributions * option_oi * CONTRACT_SIZE * 0.01 / 1e9
    
    # Crear máscaras para calls y puts
    is_call = option_types == 'C'
    is_put = option_types == 'P'
    
    # Aplicar convención de signos según dealer position (vectorizado)
    if dealer_position == "standard":
        # Calls positivos, puts negativos
        gamma_values[:, is_put] *= -1
    elif dealer_position == "inverse":
        # Calls negativos, puts positivos
        gamma_values[:, is_call] *= -1
    
    # Sumar contribuciones (axis=1 suma sobre las opciones)
    aggregate_gamma = np.sum(gamma_values, axis=1)
    
    # Calcular gamma separado para calls y puts
    call_gamma = np.sum(gamma_values[:, is_call], axis=1) if np.any(is_call) else np.zeros_like(profile_strikes)
    put_gamma = np.sum(gamma_values[:, is_put], axis=1) if np.any(is_put) else np.zeros_like(profile_strikes)
    
    # Calcular gamma flip usando interpolación vectorizada
    zero_crossings = np.where(np.diff(np.sign(aggregate_gamma)))[0]
    if len(zero_crossings) > 0:
        idx = zero_crossings[0]
        x1, x2 = profile_strikes[idx], profile_strikes[idx + 1]
        y1, y2 = aggregate_gamma[idx], aggregate_gamma[idx + 1]
        gamma_flip = x1 - y1 * (x2 - x1) / (y2 - y1)
    else:
        gamma_flip = spot
    
    # Calcular perfiles por vencimiento (optimizado)
    unique_expiries = sorted(data['expiration'].unique())[:3]  # Top 3 expiraciones
    expiry_profiles = {}
    
    for expiry in unique_expiries:
        # Filtrar datos para esta expiración
        expiry_mask = data['expiration'] == expiry
        if not np.any(expiry_mask):
            continue
        
        # Extraer datos relevantes
        exp_strikes = data.loc[expiry_mask, 'strike'].values
        exp_gammas = data.loc[expiry_mask, 'gamma'].values
        exp_oi = data.loc[expiry_mask, 'open_interest'].values
        exp_types = data.loc[expiry_mask, 'type'].values
        
        # Calcular distancias para esta expiración
        exp_distances = np.abs(profile_strikes[:, np.newaxis] - exp_strikes[np.newaxis, :])
        exp_widths = exp_strikes * 0.05
        
        # Contribuciones gamma
        exp_gamma_contrib = exp_gammas * np.exp(-(exp_distances**2) / (2 * exp_widths**2))
        exp_gamma_values = exp_gamma_contrib * exp_oi * CONTRACT_SIZE * 0.01 / 1e9
        
        # Aplicar signos
        if dealer_position == "standard":
            exp_gamma_values[:, exp_types == 'P'] *= -1
        elif dealer_position == "inverse":
            exp_gamma_values[:, exp_types == 'C'] *= -1
        
        # Sumar para obtener perfil de esta expiración
        expiry_profiles[expiry] = np.sum(exp_gamma_values, axis=1)
    
    return {
        'strikes': profile_strikes,
        'aggregate_gamma': aggregate_gamma,
        'call_gamma': call_gamma,
        'put_gamma': put_gamma,
        'gamma_flip': gamma_flip,
        'by_expiry': expiry_profiles
    }

def create_gamma_profile_chart(profiles: dict, spot: float, ticker: str):
    """Crear gráfico de perfil gamma"""
    fig = go.Figure()
    
    # Línea de gamma agregado (más gruesa y principal)
    fig.add_trace(go.Scatter(
        x=profiles['strikes'],
        y=profiles['aggregate_gamma'],
        mode='lines',
        name='Aggregate Gamma',
        line=dict(color='#00D9FF', width=3),
        hovertemplate='Strike: $%{x:.2f}<br>Gamma: %{y:.3f}B<br><extra></extra>'
    ))
    
    # Líneas de calls y puts
    fig.add_trace(go.Scatter(
        x=profiles['strikes'],
        y=profiles['call_gamma'],
        mode='lines',
        name='Call Gamma',
        line=dict(color='#00FF00', width=2, dash='dash'),
        opacity=0.7,
        hovertemplate='Strike: $%{x:.2f}<br>Call Gamma: %{y:.3f}B<br><extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=profiles['strikes'],
        y=profiles['put_gamma'],
        mode='lines',
        name='Put Gamma',
        line=dict(color='#FF6B6B', width=2, dash='dash'),
        opacity=0.7,
        hovertemplate='Strike: $%{x:.2f}<br>Put Gamma: %{y:.3f}B<br><extra></extra>'
    ))
    
    # Línea horizontal en cero
    fig.add_hline(y=0, line_dash="solid", line_color="rgba(255,255,255,0.3)", line_width=1)
    
    # Línea vertical para el spot
    fig.add_vline(
        x=spot,
        line_dash="dash",
        line_color="#FFD700",
        line_width=2,
        annotation_text=f"{ticker} Spot: ${spot:.2f}",
        annotation_position="top left"
    )
    
    # Línea vertical para gamma flip
    if 'gamma_flip' in profiles and profiles['gamma_flip'] != spot:
        fig.add_vline(
            x=profiles['gamma_flip'],
            line_dash="dot",
            line_color="#FE53BB",
            line_width=2,
            annotation_text=f"Gamma Flip: ${profiles['gamma_flip']:.2f}",
            annotation_position="bottom right"
        )
    
    # Sombreado de regiones positivas y negativas
    fig.add_hrect(
        y0=0, y1=profiles['aggregate_gamma'].max() * 1.1,
        fillcolor="rgba(0, 255, 0, 0.05)",
        layer="below",
        line_width=0
    )
    
    fig.add_hrect(
        y0=profiles['aggregate_gamma'].min() * 1.1, y1=0,
        fillcolor="rgba(255, 0, 0, 0.05)",
        layer="below",
        line_width=0
    )
    
    fig.update_layout(
        title={
            'text': f'📈 Gamma Exposure Profile - {ticker}',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': 'white', 'family': 'Arial Black'}
        },
        xaxis_title="Strike",
        yaxis_title="Gamma Exposure ($ billions/1% move)",
        template="plotly_dark",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified',
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            showgrid=True,
            zeroline=True,
            zerolinecolor='rgba(255,255,255,0.5)',
            zerolinewidth=2
        )
    )
    
    return fig

def create_gamma_profile_by_expiry(profiles: dict, spot: float, ticker: str):
    """Crear gráfico de perfil gamma por vencimiento"""
    fig = go.Figure()
    
    # Colores para diferentes vencimientos
    colors = ['#00D9FF', '#FE53BB', '#00FF00', '#FFD700', '#FF6B6B']
    
    # Agregar línea para gamma total
    fig.add_trace(go.Scatter(
        x=profiles['strikes'],
        y=profiles['aggregate_gamma'],
        mode='lines',
        name='All Expiries',
        line=dict(color='#00D9FF', width=3),
        hovertemplate='Strike: $%{x:.2f}<br>Total Gamma: %{y:.3f}B<br><extra></extra>'
    ))
    
    # Agregar líneas para cada vencimiento
    for i, (expiry, gamma_values) in enumerate(profiles['by_expiry'].items()):
        expiry_str = expiry.strftime('%d %b')
        color_idx = (i + 1) % len(colors)
        
        fig.add_trace(go.Scatter(
            x=profiles['strikes'],
            y=gamma_values,
            mode='lines',
            name=f'Expiry: {expiry_str}',
            line=dict(color=colors[color_idx], width=2, dash='dot'),
            opacity=0.8,
            hovertemplate=f'Strike: $%{{x:.2f}}<br>{expiry_str} Gamma: %{{y:.3f}}B<br><extra></extra>'
        ))
    
    # Línea horizontal en cero
    fig.add_hline(y=0, line_dash="solid", line_color="rgba(255,255,255,0.3)", line_width=1)
    
    # Línea vertical para el spot
    fig.add_vline(
        x=spot,
        line_dash="dash",
        line_color="#FFD700",
        line_width=2,
        annotation_text=f"{ticker} Spot: ${spot:.2f}",
        annotation_position="top left"
    )
    
    # Línea vertical para gamma flip
    if 'gamma_flip' in profiles:
        fig.add_vline(
            x=profiles['gamma_flip'],
            line_dash="dot",
            line_color="#FE53BB",
            line_width=2,
            annotation_text=f"Gamma Flip: ${profiles['gamma_flip']:.2f}",
            annotation_position="bottom right"
        )
    
    fig.update_layout(
        title={
            'text': f'📅 Gamma Profile by Expiration - {ticker}',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': 'white', 'family': 'Arial Black'}
        },
        xaxis_title="Strike",
        yaxis_title="Gamma Exposure ($ billions/1% move)",
        template="plotly_dark",
        height=500,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        hovermode='x unified',
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            showgrid=True,
            zeroline=True,
            zerolinecolor='rgba(255,255,255,0.5)'
        )
    )
    
    return fig

def calculate_max_pain_optimized(data: pd.DataFrame, spot_price: float) -> tuple:
    """OPTIMIZADO: Cálculo de Max Pain 10x más rápido usando NumPy"""
    # Obtener strikes únicos
    strikes = np.sort(data['strike'].unique())
    
    # Pre-calcular arrays para calls y puts
    calls = data[data['type'] == 'C'][['strike', 'open_interest']].values
    puts = data[data['type'] == 'P'][['strike', 'open_interest']].values
    
    pain_by_strike = {}
    
    # Cálculo vectorizado para cada precio de expiración
    for exp_price in strikes:
        # Cálculo vectorizado para calls ITM
        call_itm_mask = calls[:, 0] < exp_price
        call_pain = np.sum((exp_price - calls[call_itm_mask, 0]) * calls[call_itm_mask, 1] * CONTRACT_SIZE)
        
        # Cálculo vectorizado para puts ITM
        put_itm_mask = puts[:, 0] > exp_price
        put_pain = np.sum((puts[put_itm_mask, 0] - exp_price) * puts[put_itm_mask, 1] * CONTRACT_SIZE)
        
        pain_by_strike[exp_price] = call_pain + put_pain
    
    # Encontrar el mínimo
    if pain_by_strike:
        max_pain_strike = min(pain_by_strike, key=pain_by_strike.get)
        min_pain_value = pain_by_strike[max_pain_strike]
    else:
        max_pain_strike = spot_price
        min_pain_value = 0
    
    return max_pain_strike, pain_by_strike, min_pain_value

def calculate_all_metrics_batch(data: pd.DataFrame, spot_price: float) -> dict:
    """Calcular todas las métricas en una sola pasada para eficiencia"""
    # Pre-calcular agrupaciones comunes
    gex_by_strike = data.groupby("strike")["GEX"].sum()
    gex_by_type = data.groupby("type")["GEX"].sum()
    oi_by_type = data.groupby("type")["open_interest"].sum()
    
    # Calcular todas las métricas de una vez
    metrics = {
        'total_gex': data["GEX"].sum() / 1e9,
        'call_gex': gex_by_type.get('C', 0) / 1e9,
        'put_gex': gex_by_type.get('P', 0) / 1e9,
        'call_oi': oi_by_type.get('C', 0),
        'put_oi': oi_by_type.get('P', 0),
        'max_gex_strike': gex_by_strike.abs().idxmax() if len(gex_by_strike) > 0 else spot_price,
        'top_5_strikes': gex_by_strike.abs().nlargest(5).to_dict(),
        'nearest_expiry': data['expiration'].min(),
        'unique_call_strikes': data[data['type'] == 'C']['strike'].nunique(),
        'unique_put_strikes': data[data['type'] == 'P']['strike'].nunique(),
    }
    
    # Métricas derivadas
    metrics['put_call_ratio'] = abs(metrics['put_gex'] / metrics['call_gex']) if metrics['call_gex'] != 0 else 0
    metrics['days_to_expiry'] = max(0, (metrics['nearest_expiry'] - pd.Timestamp.now()).days) if pd.notna(metrics['nearest_expiry']) else 0
    
    return metrics

def create_max_pain_chart_optimized(pain_by_strike: dict, max_pain: float, spot: float):
    """Gráfico de Max Pain limpio sin flecha confusa"""
    if not pain_by_strike:
        return go.Figure()
    
    strikes = sorted(pain_by_strike.keys())
    
    # Reducir puntos si hay demasiados
    if len(strikes) > 100:
        step = len(strikes) // 100
        strikes_sampled = strikes[::step]
        if max_pain not in strikes_sampled:
            strikes_sampled.append(max_pain)
        if spot not in strikes_sampled:
            strikes_sampled.append(spot)
        strikes_sampled.sort()
    else:
        strikes_sampled = strikes
    
    pain_values = [pain_by_strike.get(s, 0) / 1e9 for s in strikes_sampled]
    
    fig = go.Figure()
    
    # Curva de dolor
    fig.add_trace(go.Scatter(
        x=strikes_sampled,
        y=pain_values,
        mode='lines',
        fill='tozeroy',
        line=dict(color='#FF6B6B', width=2),
        fillcolor='rgba(255, 107, 107, 0.2)',
        name='Dolor Total',
        hovertemplate='Strike: $%{x:.2f}<br>Dolor: $%{y:.2f}B<br><extra></extra>'
    ))
    
    # Punto Max Pain con label mejorado
    if max_pain in pain_by_strike:
        fig.add_trace(go.Scatter(
            x=[max_pain],
            y=[pain_by_strike[max_pain] / 1e9],
            mode='markers+text',
            marker=dict(size=15, color='#00FF00', symbol='diamond', line=dict(width=2, color='white')),
            text=['MAX PAIN'],
            textposition='top center',
            textfont=dict(size=14, color='#00FF00', family='Arial Black'),
            name='Max Pain',
            showlegend=False,
            hovertemplate='Max Pain: $%{x:.2f}<br>Mínimo Dolor<br><extra></extra>'
        ))
    
    # Línea de precio spot
    fig.add_vline(
        x=spot,
        line_dash="dash",
        line_color="#FFD700",
        line_width=2,
        annotation_text=f"Spot: ${spot:.2f}",
        annotation_position="top left"
    )
    
    # Línea vertical en Max Pain para mayor claridad
    fig.add_vline(
        x=max_pain,
        line_dash="dot",
        line_color="#00FF00",
        line_width=1,
        opacity=0.5,
        annotation_text=f"Target: ${max_pain:.2f}",
        annotation_position="bottom"
    )
    
    # Título mejorado con información clave
    distance_pct = ((max_pain - spot) / spot * 100)
    direction = "📈" if max_pain > spot else "📉"
    
    fig.update_layout(
        title={
            'text': f'🎯 MAX PAIN: ${max_pain:.2f} | Spot: ${spot:.2f} | {direction} {abs(distance_pct):.2f}%',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': 'white', 'family': 'Arial Black'}
        },
        xaxis_title="Precio Strike ($)",
        yaxis_title="Dolor Total - Payout de Dealers ($B)",
        template="plotly_dark",
        height=450,
        showlegend=False,
        hovermode='x unified',
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            showgrid=True,
            zeroline=True,
            zerolinecolor='rgba(255,255,255,0.2)'
        )
    )
    
    return fig

def calculate_pinning_probability(max_pain: float, spot: float, gex: float, days_to_expiry: int) -> dict:
    """Calcular probabilidad de alcanzar max pain"""
    distance_pct = abs(max_pain - spot) / spot * 100
    
    # Factores de probabilidad
    base_prob = 50
    
    # Factor por días hasta expiración
    if days_to_expiry == 0:
        expiry_factor = 30
    elif days_to_expiry <= 1:
        expiry_factor = 20
    elif days_to_expiry <= 7:
        expiry_factor = 10
    else:
        expiry_factor = 5
    
    # Factor por distancia
    if distance_pct < 0.5:
        distance_factor = 20
    elif distance_pct < 1:
        distance_factor = 15
    elif distance_pct < 2:
        distance_factor = 10
    elif distance_pct < 3:
        distance_factor = 5
    else:
        distance_factor = 0
    
    # Factor por GEX
    gex_factor = 10 if gex > 0 else -5
    
    # Probabilidad final
    probability = min(95, max(5, base_prob + expiry_factor + distance_factor + gex_factor))
    
    return {
        'probability': probability,
        'direction': 'UP' if max_pain > spot else 'DOWN',
        'distance': distance_pct
    }

def create_gex_by_strike_plot(spot: float, data: pd.DataFrame, strike_range: float):
    """Gráfico optimizado de GEX por strike"""
    gex_by_strike = data.groupby("strike")["GEX"].sum() / 1e9
    
    strike_min = spot * (1 - strike_range/100)
    strike_max = spot * (1 + strike_range/100)
    mask = (gex_by_strike.index >= strike_min) & (gex_by_strike.index <= strike_max)
    gex_filtered = gex_by_strike[mask]
    
    fig = go.Figure()
    
    colors = ['#00D9FF' if x > 0 else '#FE53BB' for x in gex_filtered.values]
    
    fig.add_trace(go.Bar(
        x=gex_filtered.index,
        y=gex_filtered.values,
        marker_color=colors,
        opacity=0.8,
        hovertemplate='Strike: $%{x:.2f}<br>GEX: %{y:.3f}B<br><extra></extra>'
    ))
    
    fig.add_vline(x=spot, line_dash="dash", line_color="#FFD700", line_width=2,
                  annotation_text=f"Spot: ${spot:.2f}")
    
    if not gex_filtered.empty:
        max_gex_strike = gex_filtered.abs().idxmax()
        fig.add_vline(x=max_gex_strike, line_dash="dot", line_color="#FF6B6B", 
                      line_width=1.5, opacity=0.7,
                      annotation_text=f"Max GEX: ${max_gex_strike:.2f}")
    
    fig.update_layout(
        title='📊 Exposición Gamma por Strike',
        xaxis_title="Precio Strike ($)",
        yaxis_title="Exposición Gamma ($Bn / 1% movimiento)",
        template="plotly_dark",
        height=450,
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig

def create_gex_by_expiration_plot(data: pd.DataFrame, max_days: int):
    """Gráfico optimizado de GEX por vencimiento"""
    max_date = datetime.now() + timedelta(days=max_days)
    data_filtered = data[data["expiration"] <= max_date]
    
    gex_by_exp = data_filtered.groupby("expiration")["GEX"].sum() / 1e9
    
    fig = go.Figure()
    
    colors = ['#00D9FF' if x > 0 else '#FE53BB' for x in gex_by_exp.values]
    
    fig.add_trace(go.Bar(
        x=gex_by_exp.index,
        y=gex_by_exp.values,
        marker_color=colors,
        opacity=0.8,
        hovertemplate='Fecha: %{x|%b %d}<br>GEX: %{y:.3f}B<br><extra></extra>'
    ))
    
    fig.update_layout(
        title='📅 Exposición Gamma por Vencimiento',
        xaxis_title="Fecha de Vencimiento",
        yaxis_title="Exposición Gamma ($Bn / 1% movimiento)",
        template="plotly_dark",
        height=450,
        showlegend=False,
        hovermode='x unified',
        xaxis=dict(tickformat='%b %d')
    )
    
    return fig

def create_strike_distribution_plot(spot: float, data: pd.DataFrame):
    """Gráfico de distribución Calls vs Puts"""
    calls_data = data[data['type'] == 'C'].groupby('strike')['GEX'].sum() / 1e9
    puts_data = data[data['type'] == 'P'].groupby('strike')['GEX'].sum() / 1e9
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=calls_data.index,
        y=calls_data.values,
        name='Calls',
        marker_color='#00D9FF',
        opacity=0.7,
        hovertemplate='Strike: $%{x:.2f}<br>Call GEX: %{y:.3f}B<br><extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        x=puts_data.index,
        y=puts_data.values,
        name='Puts',
        marker_color='#FE53BB',
        opacity=0.7,
        hovertemplate='Strike: $%{x:.2f}<br>Put GEX: %{y:.3f}B<br><extra></extra>'
    ))
    
    fig.add_vline(x=spot, line_dash="dash", line_color="#FFD700", line_width=2,
                  annotation_text=f"Spot: ${spot:.2f}")
    
    fig.update_layout(
        title='🎯 Distribución GEX - Calls vs Puts',
        xaxis_title="Precio Strike ($)",
        yaxis_title="Exposición Gamma ($Bn / 1% movimiento)",
        template="plotly_dark",
        height=450,
        barmode='overlay',
        legend=dict(x=0.02, y=0.98),
        hovermode='x unified'
    )
    
    return fig

def create_cumulative_gex_plot(data: pd.DataFrame):
    """Gráfico de GEX acumulativo"""
    data_copy = data.copy()
    data_copy['days_to_expiry'] = (data_copy['expiration'] - datetime.now()).dt.days
    data_copy = data_copy[data_copy['days_to_expiry'] >= 0]
    
    gex_by_days = data_copy.groupby('days_to_expiry')['GEX'].sum().sort_index() / 1e9
    gex_cumulative = gex_by_days.cumsum()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=gex_cumulative.index,
        y=gex_cumulative.values,
        mode='lines',
        fill='tozeroy',
        line=dict(color='#00D9FF', width=3),
        fillcolor='rgba(0, 217, 255, 0.2)',
        hovertemplate='Días: %{x}<br>GEX Acum: %{y:.3f}B<br><extra></extra>'
    ))
    
    # Marcadores para periodos importantes
    for exp in [30, 60, 90, 180, 365]:
        if exp in gex_cumulative.index:
            fig.add_vline(x=exp, line_dash="dot", line_color="rgba(255,255,255,0.3)",
                         line_width=1, annotation_text=f"{exp}d")
    
    fig.update_layout(
        title='📈 GEX Acumulativo por Tiempo hasta Vencimiento',
        xaxis_title="Días hasta Vencimiento",
        yaxis_title="GEX Acumulativo ($Bn)",
        template="plotly_dark",
        height=450,
        showlegend=False,
        hovermode='x'
    )
    
    return fig

def display_probability_analysis_clean(max_pain, spot_price, metrics, prob_analysis):
    """Diseño limpio y moderno para el análisis de probabilidad"""
    
    # Layout principal con 4 columnas para métricas clave
    col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
    
    with col1:
        # Probabilidad con visual mejorado
        prob = prob_analysis['probability']
        
        # Color y estado basado en probabilidad
        if prob > 70:
            color = "#00FF00"
            bg_color = "rgba(0, 255, 0, 0.1)"
            status = "SEÑAL FUERTE"
            emoji = "🟢"
        elif prob > 50:
            color = "#FFD700"
            bg_color = "rgba(255, 215, 0, 0.1)"
            status = "SEÑAL MEDIA"
            emoji = "🟡"
        else:
            color = "#FF6B6B"
            bg_color = "rgba(255, 107, 107, 0.1)"
            status = "SEÑAL DÉBIL"
            emoji = "🔴"
        
        st.markdown(f"""
        <div style='background: {bg_color}; border: 2px solid {color}; 
                    border-radius: 15px; padding: 15px; text-align: center;'>
            <div style='font-size: 36px; color: {color}; font-weight: bold;'>
                {prob}%
            </div>
            <div style='color: {color}; font-size: 14px; margin-top: 5px;'>
                {emoji} {status}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Días hasta expiración
        days = metrics['days_to_expiry']
        if days == 0:
            st.markdown("""
            <div style='text-align: center; padding: 10px;'>
                <div style='font-size: 28px; color: #00FF00;'>0DTE</div>
                <div style='color: #888; font-size: 11px;'>¡EXPIRA HOY!</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='text-align: center; padding: 10px;'>
                <div style='font-size: 28px; color: white;'>{days}d</div>
                <div style='color: #888; font-size: 11px;'>Días Restantes</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        # Distancia
        distance = prob_analysis['distance']
        dist_color = "#00FF00" if distance < 1 else "#FFD700" if distance < 3 else "#FF6B6B"
        st.markdown(f"""
        <div style='text-align: center; padding: 10px;'>
            <div style='font-size: 28px; color: {dist_color};'>{distance:.1f}%</div>
            <div style='color: #888; font-size: 11px;'>Distancia</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Dirección y velocidad
        if max_pain != spot_price:
            direction = "📈 ALCISTA" if max_pain > spot_price else "📉 BAJISTA"
            target_move = abs(max_pain - spot_price)
            speed = "Movimiento LENTO" if metrics['total_gex'] > 0 else "Movimiento RÁPIDO"
            gex_sign = "GEX +" if metrics['total_gex'] > 0 else "GEX -"
            
            st.markdown(f"""
            <div style='background: rgba(255,255,255,0.05); border-radius: 10px; padding: 10px;'>
                <div style='color: white; font-size: 16px; font-weight: bold;'>{direction}</div>
                <div style='color: #888; font-size: 12px; margin-top: 5px;'>
                    Target: ${max_pain:.2f} (${target_move:.2f})<br>
                    {speed} ({gex_sign})
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: rgba(0,255,0,0.1); border-radius: 10px; padding: 10px;'>
                <div style='color: #00FF00; font-size: 16px; font-weight: bold;'>✅ EN EQUILIBRIO</div>
                <div style='color: #888; font-size: 12px; margin-top: 5px;'>
                    Precio en Max Pain<br>
                    Baja volatilidad esperada
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Barra de progreso visual
    st.markdown("---")
    
    progress_html = f"""
    <div style='width: 100%; height: 30px; background: rgba(255,255,255,0.1); border-radius: 15px; overflow: hidden;'>
        <div style='width: {prob}%; height: 100%; background: linear-gradient(90deg, #FF6B6B, #FFD700, #00FF00); 
                    display: flex; align-items: center; justify-content: end; padding-right: 10px;
                    transition: all 0.5s ease;'>
            <span style='color: black; font-weight: bold;'>{prob}%</span>
        </div>
    </div>
    <div style='display: flex; justify-content: space-between; margin-top: 5px; color: #888; font-size: 12px;'>
        <span>0% - Muy Improbable</span>
        <span>50% - Neutral</span>
        <span>100% - Muy Probable</span>
    </div>
    """
    st.markdown(progress_html, unsafe_allow_html=True)

# INTERFAZ PRINCIPAL
def main():
    st.set_page_config(
        page_title="GEX Analyzer | bquantfinance",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🎯 GEX ANALYZER + MAX PAIN")
    st.markdown("### Análisis Profesional de Exposición Gamma y Max Pain")
    st.markdown("*Desarrollado por @Gsnchez | bquantfinance.com*")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuración")
        
        ticker = st.text_input(
            "📈 Símbolo del Activo",
            value="SPY",
            help="Ingrese el símbolo del ticker (ej: SPY, AAPL, TSLA)"
        ).upper()
        
        st.markdown("### 🎛️ Parámetros de Análisis")
        
        strike_range = st.slider(
            "Rango de Strikes (%)",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
            help="Porcentaje alrededor del precio spot"
        )
        
        max_expiration_days = st.slider(
            "Días Máximos hasta Vencimiento",
            min_value=7,
            max_value=180,
            value=60,
            step=7,
            help="Filtrar opciones por días hasta vencimiento"
        )
        
        min_open_interest = st.number_input(
            "Interés Abierto Mínimo",
            min_value=0,
            max_value=10000,
            value=500,
            step=100,
            help="Filtrar opciones con bajo interés abierto"
        )
        
        st.markdown("### 🏦 Posicionamiento de Dealers")
        dealer_position = st.selectbox(
            "Asunción de Posicionamiento",
            ["standard", "inverse", "neutral"],
            format_func=lambda x: {
                "standard": "Estándar (Short Puts, Long Calls)",
                "inverse": "Inverso (Long Puts, Short Calls)",
                "neutral": "Neutral (Sin Asunción)"
            }[x],
            help="Cómo asumir el posicionamiento de los dealers"
        )
        
        analyze_button = st.button(
            "🚀 Ejecutar Análisis",
            use_container_width=True,
            type="primary"
        )
    
    # Área principal
    if analyze_button:
        # Progress bar
        progress_bar = st.progress(0)
        status = st.empty()
        
        try:
            # Step 1: Fetch data
            status.text('🔄 Conectando con CBOE...')
            progress_bar.progress(20)
            raw_data = fetch_option_data(ticker)
            
            if raw_data:
                # Step 2: Parse data
                status.text('📊 Procesando datos de opciones...')
                progress_bar.progress(40)
                spot_price, option_data = parse_option_data(raw_data)
                
                if not option_data.empty:
                    # Step 3: Process data
                    option_data = process_option_data_optimized(option_data)
                    option_data = option_data[option_data['open_interest'] >= min_open_interest]
                    
                    if not option_data.empty:
                        # Step 4: Calculate GEX
                        status.text('📈 Calculando GEX...')
                        progress_bar.progress(60)
                        option_data = calculate_gex_optimized(spot_price, option_data, dealer_position)
                        
                        # Step 5: Calculate metrics
                        status.text('🎯 Calculando Max Pain y Gamma Profile...')
                        progress_bar.progress(80)
                        
                        # Guardar en session state
                        st.session_state.data_loaded = True
                        st.session_state.ticker_data = {
                            'ticker': ticker,
                            'spot': spot_price,
                            'data': option_data,
                            'dealer_position': dealer_position
                        }
                        
                        # Complete
                        status.text('✅ ¡Análisis completado!')
                        progress_bar.progress(100)
                        
                        # Clear progress
                        progress_bar.empty()
                        status.empty()
                        
                        # Mostrar resultados
                        display_results(ticker, spot_price, option_data, strike_range, max_expiration_days, dealer_position)
                    else:
                        progress_bar.empty()
                        status.empty()
                        st.error("❌ No hay datos válidos después del filtrado. Intente reducir el filtro de Interés Abierto.")
                else:
                    progress_bar.empty()
                    status.empty()
                    st.error("❌ No se encontraron datos de opciones")
            else:
                progress_bar.empty()
                status.empty()
                st.error(f"❌ No se pudieron obtener datos para {ticker}")
        
        except Exception as e:
            progress_bar.empty()
            status.empty()
            st.error(f"❌ Error durante el análisis: {str(e)}")
    
    # Mostrar resultados anteriores si existen
    elif st.session_state.data_loaded:
        ticker = st.session_state.ticker_data['ticker']
        spot_price = st.session_state.ticker_data['spot']
        option_data = st.session_state.ticker_data['data']
        dealer_position = st.session_state.ticker_data.get('dealer_position', 'standard')
        display_results(ticker, spot_price, option_data, strike_range, max_expiration_days, dealer_position)
    
    # Guía educativa
    else:
        show_educational_content()

def display_results(ticker, spot_price, option_data, strike_range, max_expiration_days, dealer_position='standard'):
    """Mostrar resultados del análisis"""
    
    # Calcular todas las métricas de una vez
    metrics = calculate_all_metrics_batch(option_data, spot_price)
    
    # Calcular Max Pain
    max_pain, pain_by_strike, total_pain = calculate_max_pain_optimized(option_data, spot_price)
    
    # Calcular Gamma Profile
    gamma_profiles = calculate_gamma_profile(option_data, spot_price, dealer_position)
    
    # Métricas principales
    st.markdown("### 📊 Métricas Principales")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("💰 Precio Spot", f"${spot_price:.2f}")
    
    with col2:
        st.metric("🎯 Max Pain", f"${max_pain:.2f}",
                 delta=f"{((max_pain-spot_price)/spot_price*100):+.1f}%")
    
    with col3:
        st.metric("🔄 Gamma Flip", f"${gamma_profiles['gamma_flip']:.2f}",
                 delta=f"{((gamma_profiles['gamma_flip']-spot_price)/spot_price*100):+.1f}%")
    
    with col4:
        st.metric("📊 GEX Total", f"${metrics['total_gex']:.2f}B",
                 delta="Positivo" if metrics['total_gex'] > 0 else "Negativo",
                 delta_color="normal" if metrics['total_gex'] > 0 else "inverse")
    
    with col5:
        st.metric("📈 GEX Calls", f"${metrics['call_gex']:.2f}B",
                 delta=f"{(metrics['call_gex']/metrics['total_gex']*100):.1f}%" if metrics['total_gex'] != 0 else "0%")
    
    with col6:
        st.metric("📉 GEX Puts", f"${metrics['put_gex']:.2f}B",
                 delta=f"P/C: {metrics['put_call_ratio']:.2f}")
    
    # Interpretación
    st.markdown("### 🎯 Interpretación del Mercado")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if metrics['total_gex'] > 0:
            st.success("""
            **📈 GEX POSITIVO - Dealers LARGOS en Gamma**
            - Venderán en rallies, comprarán en caídas
            - **Volatilidad reducida** - movimientos suaves
            - Soporte/Resistencia respetados
            """)
        else:
            st.warning("""
            **📉 GEX NEGATIVO - Dealers CORTOS en Gamma**
            - Comprarán en rallies, venderán en caídas
            - **Volatilidad amplificada** - movimientos bruscos
            - Posibles rupturas de niveles
            """)
    
    with col2:
        if abs(max_pain - spot_price) > 0.01:
            direction = "ALCISTA" if max_pain > spot_price else "BAJISTA"
            speed = "LENTO" if metrics['total_gex'] > 0 else "RÁPIDO"
            st.info(f"""
            **🎯 MAX PAIN SEÑAL: {direction}**
            - Target: ${max_pain:.2f} ({max_pain - spot_price:+.2f})
            - Movimiento esperado: **{speed}**
            - Distancia: {abs((max_pain-spot_price)/spot_price*100):.2f}%
            """)
        else:
            st.success("""
            **✅ PRECIO EN MAX PAIN**
            - Equilibrio alcanzado
            - Baja volatilidad esperada
            """)
    
    # Tabs con Gamma Profile añadido
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "💎 MAX PAIN",
        "📈 Gamma Profile", 
        "📊 Por Strike", 
        "📅 Por Vencimiento", 
        "🎯 Calls vs Puts", 
        "📈 GEX Acumulativo",
        "📅 Gamma por Expiry",
        "📋 Datos"
    ])
    
    with tab1:
        # Max Pain Chart
        fig = create_max_pain_chart_optimized(pain_by_strike, max_pain, spot_price)
        st.plotly_chart(fig, use_container_width=True)
        
        # Análisis de probabilidad con diseño limpio
        st.markdown("#### 🎲 Análisis de Probabilidad de Pin")
        
        # Calculate probability
        prob_analysis = calculate_pinning_probability(
            max_pain, spot_price, metrics['total_gex'], metrics['days_to_expiry']
        )
        
        # Usar el diseño limpio
        display_probability_analysis_clean(max_pain, spot_price, metrics, prob_analysis)
    
    with tab2:
        # Gamma Profile Chart
        st.markdown("#### 📈 Perfil de Exposición Gamma")
        
        # Información sobre gamma flip
        if gamma_profiles['gamma_flip'] != spot_price:
            gamma_flip_dist = ((gamma_profiles['gamma_flip'] - spot_price) / spot_price * 100)
            if gamma_flip_dist > 0:
                st.info(f"""
                **🔄 Gamma Flip @ ${gamma_profiles['gamma_flip']:.2f}** ({gamma_flip_dist:+.1f}% arriba)
                - Por encima: Dealers largos gamma (venden rallies)
                - Por debajo: Dealers cortos gamma (compran rallies)
                - Nivel crítico de cambio en comportamiento del mercado
                """)
            else:
                st.info(f"""
                **🔄 Gamma Flip @ ${gamma_profiles['gamma_flip']:.2f}** ({abs(gamma_flip_dist):.1f}% abajo)
                - Por encima: Dealers largos gamma (venden rallies)
                - Por debajo: Dealers cortos gamma (compran rallies)
                - Nivel crítico de cambio en comportamiento del mercado
                """)
        
        fig = create_gamma_profile_chart(gamma_profiles, spot_price, ticker)
        st.plotly_chart(fig, use_container_width=True)
        
        # Análisis adicional del perfil gamma
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 🎯 Zonas de Soporte/Resistencia")
            # Identificar picos en el perfil gamma
            gamma_peaks = []
            gamma_data = gamma_profiles['aggregate_gamma']
            strikes = gamma_profiles['strikes']
            
            # Encontrar máximos locales significativos
            for i in range(1, len(gamma_data)-1):
                if gamma_data[i] > gamma_data[i-1] and gamma_data[i] > gamma_data[i+1]:
                    if abs(gamma_data[i]) > 0.1:  # Umbral mínimo
                        gamma_peaks.append((strikes[i], gamma_data[i]))
            
            if gamma_peaks:
                gamma_peaks.sort(key=lambda x: abs(x[1]), reverse=True)
                for strike, gex in gamma_peaks[:3]:  # Top 3 picos
                    if gex > 0:
                        st.write(f"🟢 Soporte en ${strike:.2f} (GEX: {gex:.2f}B)")
                    else:
                        st.write(f"🔴 Resistencia en ${strike:.2f} (GEX: {abs(gex):.2f}B)")
        
        with col2:
            st.markdown("##### 📊 Interpretación del Perfil")
            max_gamma = max(gamma_profiles['aggregate_gamma'])
            min_gamma = min(gamma_profiles['aggregate_gamma'])
            
            if max_gamma > abs(min_gamma):
                st.success("""
                **Perfil Gamma Positivo Dominante**
                - Mercado tiende a estabilidad
                - Reversión a la media esperada
                - Volatilidad comprimida
                """)
            else:
                st.warning("""
                **Perfil Gamma Negativo Dominante**
                - Mercado propenso a movimientos bruscos
                - Posibles rupturas de rango
                - Volatilidad expandida
                """)
    
    with tab3:
        fig = create_gex_by_strike_plot(spot_price, option_data, strike_range)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top strikes
        st.markdown("#### 🎯 Top 5 Strikes con Mayor GEX")
        for strike, gex in metrics['top_5_strikes'].items():
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"Strike ${strike:.2f}")
            with col2:
                st.write(f"${gex/1e9:.3f}B")
            with col3:
                distance = ((strike - spot_price) / spot_price * 100)
                st.write(f"{distance:+.1f}%")
    
    with tab4:
        fig = create_gex_by_expiration_plot(option_data, max_expiration_days)
        st.plotly_chart(fig, use_container_width=True)
        
        # Próximos vencimientos
        st.markdown("#### 📅 Próximos Vencimientos Importantes")
        next_exp = option_data.groupby('expiration')['GEX'].sum().abs().nlargest(5)
        for exp_date, gex in next_exp.items():
            days = (exp_date - datetime.now()).days
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(exp_date.strftime('%d %b %Y'))
            with col2:
                st.write(f"${gex/1e9:.3f}B")
            with col3:
                st.write(f"{days} días")
    
    with tab5:
        fig = create_strike_distribution_plot(spot_price, option_data)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"""
            **📈 CALLS**
            - GEX: ${metrics['call_gex']:.3f}B
            - Strikes: {metrics['unique_call_strikes']}
            - OI: {metrics['call_oi']:,.0f}
            """)
        with col2:
            st.info(f"""
            **📉 PUTS**
            - GEX: ${metrics['put_gex']:.3f}B
            - Strikes: {metrics['unique_put_strikes']}
            - OI: {metrics['put_oi']:,.0f}
            """)
    
    with tab6:
        fig = create_cumulative_gex_plot(option_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # GEX por periodos
        st.markdown("#### ⏰ GEX por Periodo")
        
        data_temp = option_data.copy()
        data_temp['days'] = (data_temp['expiration'] - datetime.now()).dt.days
        
        periods = {
            "0-7d": (0, 7),
            "7-30d": (7, 30),
            "30-60d": (30, 60),
            "60-90d": (60, 90),
            "90+d": (90, 999)
        }
        
        cols = st.columns(5)
        for i, (period, (min_d, max_d)) in enumerate(periods.items()):
            mask = (data_temp['days'] >= min_d) & (data_temp['days'] < max_d)
            period_gex = data_temp[mask]['GEX'].sum() / 1e9
            with cols[i]:
                st.metric(period, f"${period_gex:.2f}B")
    
    with tab7:
        # Gamma Profile por Expiración
        st.markdown("#### 📅 Perfil Gamma por Fecha de Vencimiento")
        fig = create_gamma_profile_by_expiry(gamma_profiles, spot_price, ticker)
        st.plotly_chart(fig, use_container_width=True)
        
        # Análisis de vencimientos
        st.markdown("##### 📊 Análisis de Vencimientos")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **📌 Importancia de Vencimientos**
            - Los vencimientos cercanos tienen mayor impacto
            - 0DTE y weeklies generan mayor pinning
            - Monthly OPEX concentra mayor volumen
            """)
        
        with col2:
            # Calcular concentración de gamma por vencimiento
            exp_concentration = {}
            for expiry, gamma_vals in gamma_profiles['by_expiry'].items():
                exp_concentration[expiry] = sum(abs(g) for g in gamma_vals)
            
            if exp_concentration:
                st.markdown("**🎯 Concentración de Gamma**")
                for expiry, conc in sorted(exp_concentration.items(), key=lambda x: x[1], reverse=True):
                    days_to_exp = (expiry - datetime.now()).days
                    st.write(f"• {expiry.strftime('%d %b')}: {conc:.1f}B ({days_to_exp}d)")
    
    with tab8:
        st.markdown("#### 📋 Datos de Opciones")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            type_filter = st.selectbox("Tipo", ["Todas", "Calls", "Puts"])
        with col2:
            sort_by = st.selectbox("Ordenar", ["GEX", "open_interest", "volume", "strike"])
        with col3:
            n_rows = st.number_input("Filas", min_value=10, max_value=50, value=20)
        
        # Filtrar y mostrar
        display_data = option_data.copy()
        if type_filter == "Calls":
            display_data = display_data[display_data['type'] == 'C']
        elif type_filter == "Puts":
            display_data = display_data[display_data['type'] == 'P']
        
        display_data = display_data.sort_values(sort_by, ascending=False).head(n_rows)
        
        # Formatear para display
        cols = ['option', 'type', 'strike', 'expiration', 'GEX', 'open_interest', 'volume']
        df_display = display_data[cols].copy()
        df_display['GEX'] = df_display['GEX'].apply(lambda x: f"${x/1e6:.2f}M")
        df_display['expiration'] = df_display['expiration'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(df_display, use_container_width=True, height=400)
        
        # Download
        csv = display_data.to_csv(index=False)
        st.download_button(
            "📥 Descargar CSV",
            data=csv,
            file_name=f"{ticker}_gex_maxpain_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

def show_educational_content():
    """Contenido educativo"""
    st.markdown("## 📚 GEX + Max Pain + Gamma Profile: La Trinidad del Trading")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎓 Gamma Exposure (GEX)
        
        **GEX** mide la exposición de los market makers a cambios en el precio.
        
        - **GEX > 0**: Mercado estable, volatilidad reducida
        - **GEX < 0**: Mercado volátil, movimientos amplificados
        
        **Fórmula:**
        ```
        GEX = Γ × OI × S² × CS × 0.01
        ```
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Max Pain Theory
        
        **Max Pain** es el precio donde la mayoría de opciones expiran sin valor.
        
        **Por qué funciona:**
        - Market makers controlan ~85% del volumen
        - Hedging dinámico mueve el precio
        - Efecto "imán" en días de expiración
        
        **Mejor uso:** 0DTE y días de expiración
        """)
    
    with col3:
        st.markdown("""
        ### 📈 Gamma Profile
        
        **Gamma Profile** muestra cómo varía la exposición gamma por strikes.
        
        **Elementos clave:**
        - **Gamma Flip**: Punto donde gamma = 0
        - **Picos positivos**: Niveles de soporte
        - **Picos negativos**: Niveles de resistencia
        
        **Uso:** Identificar rangos y breakouts
        """)
    
    st.markdown("""
    ### 🎯 Cómo Usar Esta Herramienta
    
    1. **Ingrese un ticker** (SPY, QQQ, AAPL, etc.)
    2. **Configure los parámetros** (use valores bajos para mejor performance)
    3. **Analice los resultados**:
       - Max Pain vs Spot = Dirección esperada
       - GEX = Velocidad del movimiento
       - Gamma Profile = Niveles clave y comportamiento
       - Probabilidad = Confianza en la señal
    
    ### 📈 Mejores Prácticas
    
    - **0DTE**: Máxima efectividad de Max Pain
    - **Gamma Flip**: Nivel crítico para cambios de régimen
    - **GEX Positivo + Max Pain**: Señales más confiables
    - **Filtrar por OI**: Use mínimo 500 para datos relevantes
    
    ### 🔄 Interpretación del Gamma Profile
    
    - **Gamma agregado > 0**: Dealers venden rallies, compran caídas (estabilidad)
    - **Gamma agregado < 0**: Dealers compran rallies, venden caídas (volatilidad)
    - **Cruces por cero**: Niveles donde cambia el comportamiento del mercado
    - **Picos de gamma**: Actúan como imanes de precio
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("### 🚀 Desarrollado para la comunidad de trading cuantitativo")
    st.markdown("**[bquantfinance.com](https://bquantfinance.com)** | **[@Gsnchez](https://twitter.com/Gsnchez)**")


if __name__ == "__main__":
    if check_password():
        main()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder.")
