# pages/Market_Regime_ML.py
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

# Importar autenticación
try:
    from utils import check_password
except ImportError:
    st.error("❌ No se encontró el módulo 'utils'. Asegúrate de tener utils.py en tu directorio.")
    st.stop()

warnings.filterwarnings('ignore')

# =========================================================================
# CONFIGURACIÓN
# =========================================================================

st.set_page_config(page_title="Market Regime ML Detection", layout="wide")

# =========================================================================
# CONSTANTES DE COLORES Y ETIQUETAS
# =========================================================================

# Colores consistentes para regímenes (siempre 3 regímenes)
REGIME_COLORS = {
    'uptrend': '#00FF88',      # Verde brillante
    'sideways': '#FFD93D',     # Amarillo
    'downtrend': '#FF4444'     # Rojo
}

REGIME_LABELS = {
    'uptrend': 'Uptrend 📈',
    'sideways': 'Sideways ↔️',
    'downtrend': 'Downtrend 📉'
}

# =========================================================================
# FUNCIONES DE DESCARGA Y PREPARACIÓN DE DATOS
# =========================================================================

@st.cache_data(ttl=timedelta(hours=1))
def download_weekly_data(ticker, start_date='2010-01-01'):
    """Descarga datos SEMANALES y calcula features para clustering"""
    try:
        data = yf.download(ticker, start=start_date, interval='1wk', progress=False)
        
        if data.empty:
            return None
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        df = pd.DataFrame({
            'Close': data['Close'].squeeze(),
            'Open': data['Open'].squeeze(),
            'High': data['High'].squeeze(),
            'Low': data['Low'].squeeze(),
            'Volume': data['Volume'].squeeze()
        }, index=data.index)
        
        return df
    
    except Exception as e:
        st.error(f"Error descargando datos para {ticker}: {str(e)}")
        return None

def engineer_features(df):
    """Crea features para clustering basadas en el notebook de GitHub"""
    
    # Returns
    df['Returns'] = df['Close'].pct_change()
    df['LogReturns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Volatility (4 weeks rolling para datos semanales)
    df['Volatility'] = df['Returns'].rolling(window=4).std()
    
    # Momentum (10 weeks)
    df['Momentum10w'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)
    
    # Volume changes
    df['VolumeChange'] = df['Volume'].pct_change()
    df['VolumeMA'] = df['Volume'].rolling(window=4).mean()
    df['VolumeRatio'] = df['Volume'] / df['VolumeMA']
    
    # Price moving averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['PriceMA20Ratio'] = df['Close'] / df['MA20']
    df['PriceMA50Ratio'] = df['Close'] / df['MA50']
    
    # High-Low range
    df['HighLowRange'] = (df['High'] - df['Low']) / df['Close']
    
    # Returns skewness and kurtosis (8 weeks window para datos semanales)
    df['ReturnsSkew8w'] = df['Returns'].rolling(window=8).skew()
    df['ReturnsKurt8w'] = df['Returns'].rolling(window=8).kurt()
    
    # Lag features
    df['Returns_Lag1'] = df['Returns'].shift(1)
    df['Volatility_Lag1'] = df['Volatility'].shift(1)
    
    return df

def prepare_clustering_features(df):
    """Prepara features para clustering"""
    
    feature_columns = [
        'Returns', 'Volatility', 'Momentum10w', 
        'VolumeRatio', 'PriceMA20Ratio', 'PriceMA50Ratio',
        'HighLowRange', 'ReturnsSkew8w', 'ReturnsKurt8w',
        'Returns_Lag1', 'Volatility_Lag1'
    ]
    
    # Eliminar NaNs
    df_clean = df.dropna(subset=feature_columns)
    
    if len(df_clean) < 50:
        return None, None, None
    
    # Extraer features
    X = df_clean[feature_columns].values
    
    # Normalizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, df_clean, feature_columns

# =========================================================================
# MODELOS DE CLUSTERING
# =========================================================================

def fit_kmeans(X, n_clusters=3, random_state=42):
    """Ajusta K-Means"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    
    metrics = {
        'silhouette': silhouette_score(X, labels),
        'davies_bouldin': davies_bouldin_score(X, labels),
        'calinski_harabasz': calinski_harabasz_score(X, labels),
        'inertia': kmeans.inertia_
    }
    
    return labels, kmeans, metrics

def fit_gmm(X, n_components=3, random_state=42):
    """Ajusta Gaussian Mixture Model"""
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', 
                          random_state=random_state, n_init=10)
    gmm.fit(X)
    labels = gmm.predict(X)
    
    metrics = {
        'silhouette': silhouette_score(X, labels),
        'davies_bouldin': davies_bouldin_score(X, labels),
        'bic': gmm.bic(X),
        'aic': gmm.aic(X)
    }
    
    return labels, gmm, metrics

def fit_hmm(X, n_states=3, random_state=42):
    """Ajusta Hidden Markov Model"""
    hmm = GaussianHMM(n_components=n_states, covariance_type='full', 
                      n_iter=100, random_state=random_state)
    hmm.fit(X)
    labels = hmm.predict(X)
    
    metrics = {
        'silhouette': silhouette_score(X, labels),
        'davies_bouldin': davies_bouldin_score(X, labels),
        'log_likelihood': hmm.score(X)
    }
    
    return labels, hmm, metrics

# =========================================================================
# MAPEO DE REGÍMENES
# =========================================================================

def map_regimes_to_labels(df_clean, regime_col):
    """
    Mapea los clusters numéricos a labels consistentes (uptrend, sideways, downtrend)
    basándose en características de retorno y momentum
    """
    
    regime_stats = df_clean.groupby(regime_col).agg({
        'Returns': 'mean',
        'Momentum10w': 'mean',
        'Volatility': 'mean'
    })
    
    # Ordenar por retorno promedio
    regime_stats = regime_stats.sort_values('Returns', ascending=False)
    
    # Asignar etiquetas basadas en el ranking de retornos
    mapping = {}
    regime_ids = regime_stats.index.tolist()
    
    if len(regime_ids) == 3:
        # El mejor retorno = uptrend, medio = sideways, peor = downtrend
        mapping[regime_ids[0]] = 'uptrend'
        mapping[regime_ids[1]] = 'sideways'
        mapping[regime_ids[2]] = 'downtrend'
    elif len(regime_ids) == 2:
        # Solo dos regímenes
        mapping[regime_ids[0]] = 'uptrend'
        mapping[regime_ids[1]] = 'downtrend'
    else:
        # Más de 3 regímenes - usar percentiles
        for i, regime_id in enumerate(regime_ids):
            if i < len(regime_ids) // 3:
                mapping[regime_id] = 'uptrend'
            elif i < 2 * len(regime_ids) // 3:
                mapping[regime_id] = 'sideways'
            else:
                mapping[regime_id] = 'downtrend'
    
    return mapping, regime_stats

# =========================================================================
# VISUALIZACIÓN
# =========================================================================

def plot_regime_comparison(df_clean, ticker):
    """Visualiza los 3 modelos en comparación con colores consistentes"""
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 14), facecolor='#0E1117')
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 3, 3, 2], hspace=0.3)
    
    # =====================================================================
    # GRÁFICO 1: K-MEANS
    # =====================================================================
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#1A1D29')
    
    # Línea de precio
    ax1.plot(df_clean.index, df_clean['Close'], color='#FFFFFF', 
             linewidth=1.5, alpha=0.6, zorder=1)
    
    # Puntos coloreados por régimen
    for regime_label in ['uptrend', 'sideways', 'downtrend']:
        mask = df_clean['KMeans_Label'] == regime_label
        if mask.any():
            ax1.scatter(df_clean[mask].index, df_clean[mask]['Close'],
                       c=REGIME_COLORS[regime_label], s=35, alpha=0.8,
                       label=REGIME_LABELS[regime_label], zorder=3, 
                       edgecolors='white', linewidth=0.5)
    
    ax1.set_title(f'{ticker} - K-Means Clustering Regimes (Weekly)', 
                  fontsize=16, fontweight='bold', color='#FFFFFF', pad=20)
    ax1.set_ylabel('Price ($)', fontsize=13, color='#FFFFFF', fontweight='600')
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.9, facecolor='#1A1D29')
    ax1.grid(True, alpha=0.1, color='#FFFFFF')
    ax1.tick_params(colors='#B0B0B0', labelsize=10)
    
    for spine in ax1.spines.values():
        spine.set_color('#2D3142')
        spine.set_linewidth(1.5)
    
    # =====================================================================
    # GRÁFICO 2: GMM
    # =====================================================================
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_facecolor('#1A1D29')
    
    ax2.plot(df_clean.index, df_clean['Close'], color='#FFFFFF', 
             linewidth=1.5, alpha=0.6, zorder=1)
    
    for regime_label in ['uptrend', 'sideways', 'downtrend']:
        mask = df_clean['GMM_Label'] == regime_label
        if mask.any():
            ax2.scatter(df_clean[mask].index, df_clean[mask]['Close'],
                       c=REGIME_COLORS[regime_label], s=35, alpha=0.8,
                       label=REGIME_LABELS[regime_label], zorder=3, 
                       edgecolors='white', linewidth=0.5)
    
    ax2.set_title(f'{ticker} - Gaussian Mixture Model Regimes (Weekly)', 
                  fontsize=16, fontweight='bold', color='#FFFFFF', pad=20)
    ax2.set_ylabel('Price ($)', fontsize=13, color='#FFFFFF', fontweight='600')
    ax2.legend(loc='upper left', fontsize=11, framealpha=0.9, facecolor='#1A1D29')
    ax2.grid(True, alpha=0.1, color='#FFFFFF')
    ax2.tick_params(colors='#B0B0B0', labelsize=10)
    
    for spine in ax2.spines.values():
        spine.set_color('#2D3142')
        spine.set_linewidth(1.5)
    
    # =====================================================================
    # GRÁFICO 3: HMM
    # =====================================================================
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.set_facecolor('#1A1D29')
    
    ax3.plot(df_clean.index, df_clean['Close'], color='#FFFFFF', 
             linewidth=1.5, alpha=0.6, zorder=1)
    
    for regime_label in ['uptrend', 'sideways', 'downtrend']:
        mask = df_clean['HMM_Label'] == regime_label
        if mask.any():
            ax3.scatter(df_clean[mask].index, df_clean[mask]['Close'],
                       c=REGIME_COLORS[regime_label], s=35, alpha=0.8,
                       label=REGIME_LABELS[regime_label], zorder=3, 
                       edgecolors='white', linewidth=0.5)
    
    ax3.set_title(f'{ticker} - Hidden Markov Model Regimes (Weekly)', 
                  fontsize=16, fontweight='bold', color='#FFFFFF', pad=20)
    ax3.set_ylabel('Price ($)', fontsize=13, color='#FFFFFF', fontweight='600')
    ax3.legend(loc='upper left', fontsize=11, framealpha=0.9, facecolor='#1A1D29')
    ax3.grid(True, alpha=0.1, color='#FFFFFF')
    ax3.tick_params(colors='#B0B0B0', labelsize=10)
    
    for spine in ax3.spines.values():
        spine.set_color('#2D3142')
        spine.set_linewidth(1.5)
    
    # =====================================================================
    # GRÁFICO 4: TIMELINE COMPARACIÓN
    # =====================================================================
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.set_facecolor('#1A1D29')
    
    # Mapeo numérico para visualización
    label_to_num = {'uptrend': 2, 'sideways': 1, 'downtrend': 0}
    
    # K-Means
    kmeans_nums = df_clean['KMeans_Label'].map(label_to_num)
    for label in ['uptrend', 'sideways', 'downtrend']:
        mask = df_clean['KMeans_Label'] == label
        ax4.scatter(df_clean[mask].index, kmeans_nums[mask], 
                   c=REGIME_COLORS[label], s=25, alpha=0.8, marker='s',
                   edgecolors='white', linewidth=0.3)
    
    # GMM
    gmm_nums = df_clean['GMM_Label'].map(label_to_num) + 0.1
    for label in ['uptrend', 'sideways', 'downtrend']:
        mask = df_clean['GMM_Label'] == label
        ax4.scatter(df_clean[mask].index, gmm_nums[mask], 
                   c=REGIME_COLORS[label], s=25, alpha=0.8, marker='^',
                   edgecolors='white', linewidth=0.3)
    
    # HMM
    hmm_nums = df_clean['HMM_Label'].map(label_to_num) + 0.2
    for label in ['uptrend', 'sideways', 'downtrend']:
        mask = df_clean['HMM_Label'] == label
        ax4.scatter(df_clean[mask].index, hmm_nums[mask], 
                   c=REGIME_COLORS[label], s=25, alpha=0.8, marker='o',
                   edgecolors='white', linewidth=0.3)
    
    # Leyenda personalizada
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#888', 
               markersize=8, label='K-Means', linestyle='None'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#888', 
               markersize=8, label='GMM', linestyle='None'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#888', 
               markersize=8, label='HMM', linestyle='None')
    ]
    
    ax4.set_title('Regime Timeline Comparison', fontsize=16, 
                  fontweight='bold', color='#FFFFFF', pad=20)
    ax4.set_ylabel('Regime Type', fontsize=13, color='#FFFFFF', fontweight='600')
    ax4.set_xlabel('Date', fontsize=13, color='#FFFFFF', fontweight='600')
    ax4.set_yticks([0, 1, 2])
    ax4.set_yticklabels(['Downtrend 📉', 'Sideways ↔️', 'Uptrend 📈'])
    ax4.legend(handles=legend_elements, loc='upper left', fontsize=10, 
              framealpha=0.9, facecolor='#1A1D29')
    ax4.grid(True, alpha=0.1, color='#FFFFFF')
    ax4.tick_params(colors='#B0B0B0', labelsize=10)
    
    for spine in ax4.spines.values():
        spine.set_color('#2D3142')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    return fig

def plot_pca_visualization(X_scaled, df_clean):
    """Visualización PCA de los regímenes con colores consistentes"""
    
    # Aplicar PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), facecolor='#0E1117')
    
    # K-Means PCA
    ax1 = axes[0]
    ax1.set_facecolor('#1A1D29')
    for regime_label in ['uptrend', 'sideways', 'downtrend']:
        mask = df_clean['KMeans_Label'] == regime_label
        if mask.any():
            ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=REGIME_COLORS[regime_label], s=60, alpha=0.7,
                       label=REGIME_LABELS[regime_label], 
                       edgecolors='white', linewidth=0.5)
    ax1.set_title('K-Means - PCA Projection', fontsize=14, fontweight='bold', color='#FFFFFF')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', color='#FFFFFF')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', color='#FFFFFF')
    ax1.legend(fontsize=10, framealpha=0.9, facecolor='#1A1D29')
    ax1.grid(True, alpha=0.1, color='#FFFFFF')
    ax1.tick_params(colors='#B0B0B0')
    
    for spine in ax1.spines.values():
        spine.set_color('#2D3142')
    
    # GMM PCA
    ax2 = axes[1]
    ax2.set_facecolor('#1A1D29')
    for regime_label in ['uptrend', 'sideways', 'downtrend']:
        mask = df_clean['GMM_Label'] == regime_label
        if mask.any():
            ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=REGIME_COLORS[regime_label], s=60, alpha=0.7,
                       label=REGIME_LABELS[regime_label], 
                       edgecolors='white', linewidth=0.5)
    ax2.set_title('GMM - PCA Projection', fontsize=14, fontweight='bold', color='#FFFFFF')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', color='#FFFFFF')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', color='#FFFFFF')
    ax2.legend(fontsize=10, framealpha=0.9, facecolor='#1A1D29')
    ax2.grid(True, alpha=0.1, color='#FFFFFF')
    ax2.tick_params(colors='#B0B0B0')
    
    for spine in ax2.spines.values():
        spine.set_color('#2D3142')
    
    # HMM PCA
    ax3 = axes[2]
    ax3.set_facecolor('#1A1D29')
    for regime_label in ['uptrend', 'sideways', 'downtrend']:
        mask = df_clean['HMM_Label'] == regime_label
        if mask.any():
            ax3.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=REGIME_COLORS[regime_label], s=60, alpha=0.7,
                       label=REGIME_LABELS[regime_label], 
                       edgecolors='white', linewidth=0.5)
    ax3.set_title('HMM - PCA Projection', fontsize=14, fontweight='bold', color='#FFFFFF')
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', color='#FFFFFF')
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', color='#FFFFFF')
    ax3.legend(fontsize=10, framealpha=0.9, facecolor='#1A1D29')
    ax3.grid(True, alpha=0.1, color='#FFFFFF')
    ax3.tick_params(colors='#B0B0B0')
    
    for spine in ax3.spines.values():
        spine.set_color('#2D3142')
    
    plt.tight_layout()
    
    return fig

def plot_hmm_transition_matrix(hmm_model, regime_mapping):
    """Visualiza la matriz de transición del HMM con labels"""
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0E1117')
    ax.set_facecolor('#1A1D29')
    
    transmat = hmm_model.transmat_
    n_states = len(transmat)
    
    # Crear mapeo inverso
    inv_mapping = {v: k for k, v in regime_mapping.items()}
    state_labels = [REGIME_LABELS.get(inv_mapping.get(i, 'unknown'), f'State {i}') 
                    for i in range(n_states)]
    
    im = ax.imshow(transmat, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Añadir valores en las celdas
    for i in range(n_states):
        for j in range(n_states):
            text = ax.text(j, i, f'{transmat[i, j]:.3f}',
                          ha="center", va="center", color="black", 
                          fontsize=14, fontweight='bold')
    
    ax.set_xticks(np.arange(n_states))
    ax.set_yticks(np.arange(n_states))
    ax.set_xticklabels(state_labels, color='#FFFFFF', fontsize=11)
    ax.set_yticklabels(state_labels, color='#FFFFFF', fontsize=11)
    ax.set_xlabel('To State', fontsize=13, color='#FFFFFF', fontweight='600')
    ax.set_ylabel('From State', fontsize=13, color='#FFFFFF', fontweight='600')
    ax.set_title('HMM Transition Probability Matrix', fontsize=16, 
                 fontweight='bold', color='#FFFFFF', pad=20)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Transition Probability', color='#FFFFFF', fontsize=12)
    cbar.ax.tick_params(colors='#FFFFFF')
    
    plt.tight_layout()
    
    return fig

# =========================================================================
# ANÁLISIS DE REGÍMENES
# =========================================================================

def analyze_regime_characteristics(df_clean):
    """Analiza características de cada régimen por modelo"""
    
    results = {}
    
    for model_name, label_col in [('K-Means', 'KMeans_Label'), 
                                   ('GMM', 'GMM_Label'), 
                                   ('HMM', 'HMM_Label')]:
        
        regime_stats = df_clean.groupby(label_col).agg({
            'Returns': ['mean', 'std', 'count'],
            'Volatility': ['mean', 'std'],
            'Momentum10w': 'mean',
            'VolumeRatio': 'mean'
        }).round(4)
        
        # Renombrar columnas
        regime_stats.columns = ['_'.join(col).strip() for col in regime_stats.columns.values]
        
        # Calcular retornos anualizados (52 semanas)
        regime_stats['Annual_Return_%'] = regime_stats['Returns_mean'] * 52 * 100
        regime_stats['Annual_Vol_%'] = regime_stats['Volatility_mean'] * np.sqrt(52) * 100
        
        # Reordenar index para consistencia
        desired_order = ['uptrend', 'sideways', 'downtrend']
        existing_labels = [label for label in desired_order if label in regime_stats.index]
        regime_stats = regime_stats.reindex(existing_labels)
        
        results[model_name] = regime_stats
    
    return results

# =========================================================================
# PÁGINA PRINCIPAL
# =========================================================================

def market_regime_ml_page():
    
    # CSS personalizado
    st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stMetric {
        background: linear-gradient(135deg, #1A1D29 0%, #2D3142 100%);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #00D9FF;
        box-shadow: 0 4px 15px rgba(0, 217, 255, 0.2);
    }
    .stMetric label {
        color: #00D9FF !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 24px !important;
        font-weight: 800 !important;
    }
    .stButton>button {
        background: linear-gradient(135deg, #4ECDC4 0%, #00D9FF 100%);
        color: white;
        font-weight: 700;
        border: none;
        padding: 12px 24px;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(78, 205, 196, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(78, 205, 196, 0.6);
    }
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-weight: 800 !important;
    }
    .stAlert {
        border-radius: 12px;
        border-left: 5px solid #00D9FF;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🤖 Market Regime Detection - ML Models (Weekly)")
    st.markdown("---")
    
    # Header con diseño mejorado
    col_header1, col_header2 = st.columns([3, 1])
    with col_header1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1A1D29 0%, #2D3142 100%); 
                    padding: 20px; border-radius: 15px; border: 2px solid #4ECDC4;
                    box-shadow: 0 4px 15px rgba(78, 205, 196, 0.3);'>
            <h3 style='color: #4ECDC4; margin: 0;'>🔍 Machine Learning Regime Detection</h3>
            <p style='color: #B0B0B0; margin: 5px 0 0 0;'>
                Análisis de regímenes con K-Means, GMM y HMM en timeframe semanal
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 20px; 
                    background: linear-gradient(135deg, #4ECDC4 0%, #00D9FF 100%); 
                    border-radius: 15px; margin-bottom: 20px;'>
            <h2 style='color: white; margin: 0;'>⚙️ Configuration</h2>
        </div>
        """, unsafe_allow_html=True)
        
        ticker = st.text_input(
            "🎯 Ticker Symbol",
            value="SPY",
            help="Ingresa el símbolo del ticker (ej: SPY, QQQ, AAPL)"
        ).upper()
        
        st.markdown("---")
        
        # Fecha por defecto: 5 años atrás
        default_start = datetime.now() - timedelta(days=365*5)
        
        start_date = st.date_input(
            "📆 Start Date",
            value=default_start,
            help="Fecha de inicio para descargar datos (por defecto: 5 años)"
        )
        
        st.markdown("---")
        
        # Fijo en 3 regímenes para consistencia
        st.info("🔢 **Regímenes:** 3 (Uptrend, Sideways, Downtrend)")
        n_regimes = 3
        
        st.markdown("---")
        
        analyze_btn = st.button(
            "🚀 ANALYZE REGIMES",
            type="primary",
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Información de modelos
        st.markdown("""
        ### 📚 Models Used
        
        **K-Means Clustering**
        - Fast partitioning algorithm
        - Hard cluster assignments
        - Good for clear separations
        
        **GMM (Gaussian Mixture)**
        - Probabilistic clustering
        - Soft assignments
        - Flexible covariances
        
        **HMM (Hidden Markov)**
        - Temporal dependencies
        - Transition probabilities
        - Sequential patterns
        
        ### 📊 Features Used
        - Returns & Volatility
        - Momentum (10 weeks)
        - Volume ratios
        - Price/MA ratios
        - High-Low range
        - Returns skewness/kurtosis
        
        ### 🎨 Regime Colors
        - 🟢 **Uptrend**: Verde
        - 🟡 **Sideways**: Amarillo
        - 🔴 **Downtrend**: Rojo
        """)
    
    if analyze_btn:
        with st.spinner(f"📥 Downloading weekly data for {ticker}..."):
            df = download_weekly_data(ticker, start_date.strftime('%Y-%m-%d'))
            
            if df is None or df.empty:
                st.error(f"❌ No data available for {ticker}")
                st.stop()
            
            st.success(f"✅ Downloaded {len(df)} weekly candles")
        
        with st.spinner("🔧 Engineering features..."):
            df = engineer_features(df)
            X_scaled, df_clean, feature_cols = prepare_clustering_features(df)
            
            if X_scaled is None:
                st.error("❌ Not enough data after feature engineering")
                st.stop()
            
            st.success(f"✅ Prepared {len(df_clean)} observations with {len(feature_cols)} features")
        
        with st.spinner("🤖 Training models..."):
            
            # K-Means
            kmeans_labels, kmeans_model, kmeans_metrics = fit_kmeans(X_scaled, n_clusters=n_regimes)
            df_clean['KMeans_Regime'] = kmeans_labels
            kmeans_mapping, kmeans_stats = map_regimes_to_labels(df_clean, 'KMeans_Regime')
            df_clean['KMeans_Label'] = df_clean['KMeans_Regime'].map(kmeans_mapping)
            
            # GMM
            gmm_labels, gmm_model, gmm_metrics = fit_gmm(X_scaled, n_components=n_regimes)
            df_clean['GMM_Regime'] = gmm_labels
            gmm_mapping, gmm_stats = map_regimes_to_labels(df_clean, 'GMM_Regime')
            df_clean['GMM_Label'] = df_clean['GMM_Regime'].map(gmm_mapping)
            
            # HMM
            hmm_labels, hmm_model, hmm_metrics = fit_hmm(X_scaled, n_states=n_regimes)
            df_clean['HMM_Regime'] = hmm_labels
            hmm_mapping, hmm_stats = map_regimes_to_labels(df_clean, 'HMM_Regime')
            df_clean['HMM_Label'] = df_clean['HMM_Regime'].map(hmm_mapping)
            
            st.success("✅ All models trained successfully!")
        
        # Guardar en session state
        st.session_state.df_clean = df_clean
        st.session_state.X_scaled = X_scaled
        st.session_state.kmeans_metrics = kmeans_metrics
        st.session_state.gmm_metrics = gmm_metrics
        st.session_state.hmm_metrics = hmm_metrics
        st.session_state.hmm_model = hmm_model
        st.session_state.hmm_mapping = hmm_mapping
        st.session_state.n_regimes = n_regimes
        st.session_state.ticker = ticker
    
    # Mostrar resultados si existen en session state
    if 'df_clean' in st.session_state:
        df_clean = st.session_state.df_clean
        X_scaled = st.session_state.X_scaled
        kmeans_metrics = st.session_state.kmeans_metrics
        gmm_metrics = st.session_state.gmm_metrics
        hmm_metrics = st.session_state.hmm_metrics
        hmm_model = st.session_state.hmm_model
        hmm_mapping = st.session_state.hmm_mapping
        n_regimes = st.session_state.n_regimes
        ticker = st.session_state.ticker
        
        st.markdown("---")
        
        # Métricas de los modelos
        st.markdown("## 📊 Model Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### K-Means")
            st.metric("Silhouette Score", f"{kmeans_metrics['silhouette']:.4f}")
            st.metric("Davies-Bouldin", f"{kmeans_metrics['davies_bouldin']:.4f}")
            st.metric("Calinski-Harabasz", f"{kmeans_metrics['calinski_harabasz']:.1f}")
        
        with col2:
            st.markdown("### GMM")
            st.metric("Silhouette Score", f"{gmm_metrics['silhouette']:.4f}")
            st.metric("BIC", f"{gmm_metrics['bic']:.0f}")
            st.metric("AIC", f"{gmm_metrics['aic']:.0f}")
        
        with col3:
            st.markdown("### HMM")
            st.metric("Silhouette Score", f"{hmm_metrics['silhouette']:.4f}")
            st.metric("Davies-Bouldin", f"{hmm_metrics['davies_bouldin']:.4f}")
            st.metric("Log-Likelihood", f"{hmm_metrics['log_likelihood']:.2f}")
        
        st.markdown("---")
        
        # Régimen actual
        st.markdown("## 🎯 Current Market Regime")
        
        current = df_clean.iloc[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Date", current.name.strftime('%Y-%m-%d'))
        
        with col2:
            kmeans_label = current['KMeans_Label']
            st.metric("K-Means", REGIME_LABELS.get(kmeans_label, kmeans_label))
        
        with col3:
            gmm_label = current['GMM_Label']
            st.metric("GMM", REGIME_LABELS.get(gmm_label, gmm_label))
        
        with col4:
            hmm_label = current['HMM_Label']
            st.metric("HMM", REGIME_LABELS.get(hmm_label, hmm_label))
        
        st.markdown("---")
        
        # Gráficos principales
        st.markdown("## 📈 Regime Visualization")
        
        with st.spinner("Creating visualizations..."):
            fig1 = plot_regime_comparison(df_clean, ticker)
            st.pyplot(fig1)
        
        st.markdown("---")
        
        # PCA Visualization
        st.markdown("## 🔍 PCA Projection - Cluster Separability")
        
        with st.spinner("Computing PCA..."):
            fig2 = plot_pca_visualization(X_scaled, df_clean)
            st.pyplot(fig2)
        
        st.markdown("---")
        
        # HMM Transition Matrix
        st.markdown("## 🔄 HMM Transition Probability Matrix")
        
        with st.spinner("Plotting transition matrix..."):
            fig3 = plot_hmm_transition_matrix(hmm_model, hmm_mapping)
            st.pyplot(fig3)
        
        st.markdown("---")
        
        # Análisis de características
        st.markdown("## 📋 Regime Characteristics Analysis")
        
        regime_analysis = analyze_regime_characteristics(df_clean)
        
        for model_name, stats_df in regime_analysis.items():
            st.markdown(f"### {model_name}")
            
            # Aplicar estilo con colores
            def style_regime_rows(row):
                if row.name == 'uptrend':
                    return ['background-color: rgba(0, 255, 136, 0.15)'] * len(row)
                elif row.name == 'sideways':
                    return ['background-color: rgba(255, 217, 61, 0.15)'] * len(row)
                elif row.name == 'downtrend':
                    return ['background-color: rgba(255, 68, 68, 0.15)'] * len(row)
                return [''] * len(row)
            
            styled_df = stats_df.style.apply(style_regime_rows, axis=1)
            st.dataframe(styled_df, use_container_width=True)
        
        st.markdown("---")
        
        # Distribución de regímenes
        st.markdown("## 📊 Regime Distribution")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### K-Means")
            kmeans_dist = df_clean['KMeans_Label'].value_counts()
            for regime_label in ['uptrend', 'sideways', 'downtrend']:
                if regime_label in kmeans_dist.index:
                    count = kmeans_dist[regime_label]
                    pct = (count / len(df_clean)) * 100
                    st.write(f"**{REGIME_LABELS[regime_label]}**: {count} weeks ({pct:.1f}%)")
        
        with col2:
            st.markdown("### GMM")
            gmm_dist = df_clean['GMM_Label'].value_counts()
            for regime_label in ['uptrend', 'sideways', 'downtrend']:
                if regime_label in gmm_dist.index:
                    count = gmm_dist[regime_label]
                    pct = (count / len(df_clean)) * 100
                    st.write(f"**{REGIME_LABELS[regime_label]}**: {count} weeks ({pct:.1f}%)")
        
        with col3:
            st.markdown("### HMM")
            hmm_dist = df_clean['HMM_Label'].value_counts()
            for regime_label in ['uptrend', 'sideways', 'downtrend']:
                if regime_label in hmm_dist.index:
                    count = hmm_dist[regime_label]
                    pct = (count / len(df_clean)) * 100
                    st.write(f"**{REGIME_LABELS[regime_label]}**: {count} weeks ({pct:.1f}%)")
        
        st.markdown("---")
        
        # Exportar datos
        st.markdown("## 💾 Export Data")
        
        export_cols = ['Close', 'Returns', 'Volatility', 
                       'KMeans_Label', 'GMM_Label', 'HMM_Label']
        available_cols = [col for col in export_cols if col in df_clean.columns]
        
        csv = df_clean[available_cols].to_csv()
        st.download_button(
            label="📥 Download Full Dataset (CSV)",
            data=csv,
            file_name=f"regime_detection_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    else:
        # Mensaje de bienvenida
        st.markdown("""
        <div style='text-align: center; padding: 40px; 
                    background: linear-gradient(135deg, #1A1D29 0%, #2D3142 100%); 
                    border-radius: 20px; border: 3px solid #4ECDC4; margin-top: 30px;
                    box-shadow: 0 8px 30px rgba(78, 205, 196, 0.3);'>
            <h2 style='color: #4ECDC4; margin: 0;'>🤖 Bienvenido al Market Regime Detection ML</h2>
            <p style='color: #B0B0B0; font-size: 18px; margin: 20px 0;'>
                Configura los parámetros en el panel lateral y presiona 
                <strong style='color: #00D9FF;'>🚀 ANALYZE REGIMES</strong> 
                para comenzar el análisis con Machine Learning.
            </p>
            <p style='color: #8E93A1; font-size: 14px; margin: 10px 0 0 0;'>
                📊 Análisis con K-Means, GMM y HMM | ⏱️ Timeframe: Weekly (1w) | 📅 Default: 5 años
            </p>
        </div>
        """, unsafe_allow_html=True)

# =========================================================================
# MAIN
# =========================================================================

if __name__ == "__main__":
    # Verificar autenticación
    if check_password():
        market_regime_ml_page()
    else:
        # Mensaje de acceso restringido
        st.markdown("""
        <div style='text-align: center; padding: 60px 20px;'>
            <h1 style='color: #FF6B6B; font-size: 48px;'>🔒 Acceso Restringido</h1>
            <p style='color: #B0B0B0; font-size: 20px; margin-top: 20px;'>
                Introduce tus credenciales en el menú lateral para acceder al análisis.
            </p>
        </div>
        """, unsafe_allow_html=True)
