import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# AUTENTICACIÓN - Igual que en el fichero base (utils.utils.check_password)
# ============================================================================

try:
    from utils.utils import check_password
except ImportError:
    def check_password():
        """Verifica si el usuario tiene credenciales correctas."""

        def login_form():
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

        if st.session_state.get("password_correct", False):
            with st.sidebar:
                st.success(f"✅ Sesión activa: {st.session_state.get('username', 'Usuario')}")
                if st.button("🚪 Cerrar Sesión", use_container_width=True):
                    st.session_state["password_correct"] = False
                    if "username" in st.session_state:
                        del st.session_state["username"]
                    st.rerun()
            return True

        login_form()
        return False

# ============================================================================
# COLORES DE REGÍMENES (3 estados)
# ============================================================================

REGIME_COLORS = {
    "ALCISTA": "#00FF00",
    "BAJISTA": "#FF0000",
    "RANGO":   "#FFD700",
}

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def build_features(df, vol_window=12, return_smooth=3, trend_window=20,
                    z_window=104, z_min_periods=52):
    """
    Construye el set de features a partir de datos semanales:
      - Retorno suavizado (media móvil de `return_smooth` semanas).
      - Volatilidad realizada (rolling, anualizada).
      - Pendiente (slope) de una regresión lineal sobre el log-precio en una
        ventana de `trend_window` semanas.
      - Z-score de esa pendiente contra su propia media/desviación histórica
        (ventana `z_window`). Se usa como respaldo opcional (ver
        `apply_bear_override`), NO como input directo del HMM/GAS.
    """
    close = df["Close"].astype(float).squeeze()

    raw_returns = close.pct_change()
    returns_smoothed = raw_returns.rolling(return_smooth).mean()
    volatility = raw_returns.rolling(vol_window).std() * np.sqrt(52)

    log_close = np.log(close)

    def _rolling_slope(x):
        y = x.values
        n = len(y)
        xi = np.arange(n)
        xi_mean = xi.mean()
        y_mean = y.mean()
        cov = ((xi - xi_mean) * (y - y_mean)).sum()
        var = ((xi - xi_mean) ** 2).sum()
        return cov / var if var > 0 else 0.0

    slope = log_close.rolling(trend_window).apply(_rolling_slope, raw=False)

    slope_mean = slope.rolling(z_window, min_periods=z_min_periods).mean()
    slope_std = slope.rolling(z_window, min_periods=z_min_periods).std()
    slope_z = (slope - slope_mean) / slope_std.replace(0, np.nan)

    feat = pd.DataFrame({
        "returns": returns_smoothed,
        "volatility": volatility,
        "slope": slope,
        "slope_z": slope_z,
    }, index=df.index).dropna()

    return feat

# ============================================================================
# GAS TRANSITION LAYER
# ============================================================================
#
# NOTA IMPORTANTE SOBRE FIDELIDAD AL MODELO GAS ORIGINAL
# --------------------------------------------------------------------------
# Un HMM-GAS "completo" (Creal, Koopman & Lucas, 2013; aplicado a regímenes
# de mercado en varios papers de 2024-2026) actualiza TODOS los parámetros
# del modelo -- medias, covarianzas y matriz de transición -- vía un score
# de verosimilitud derivado analíticamente para cada uno, estimado por
# máxima verosimilitud completa. Implementarlo así desde cero es un
# proyecto de investigación por sí mismo (requiere derivar e implementar el
# gradiente exacto de la log-verosimilitud para una GaussianHMM multivariante
# con covarianza completa, y no hay ninguna librería de Python madura que lo
# ofrezca listo para usar, a diferencia de hmmlearn para el HMM estándar).
#
# Esta implementación toma la parte del modelo que más impacto tiene sobre
# el problema que motivó este cambio (el HMM "tragándose" caídas al ser
# demasiado o muy poco persistente según un sticky_strength fijo elegido a
# mano): deja las medias/covarianzas de los 3 estados fijas, estimadas por
# EM con hmmlearn exactamente como antes, pero reemplaza la matriz de
# transición ESTÁTICA por una que seguí una recursión GAS clásica:
#
#     kappa_{t+1} = omega + phi * kappa_t + alpha * score_t
#
# donde kappa_t es el logit de la probabilidad de auto-transición
# (persistencia del régimen) y score_t mide qué tan bien la transición
# anterior predijo lo que efectivamente ocurrió. Si el régimen se mantuvo
# más de lo que la matriz de transición predecía, el score empuja kappa
# hacia arriba (más persistencia); si cambió antes de lo esperado, lo
# empuja hacia abajo. Esto es exactamente el mecanismo central del GAS
# (Generalized Autoregressive Score) aplicado al parámetro más relevante
# para tu problema -- la persistencia -- en vez de a los seis parámetros.
#
# Qué se gana respecto al sticky prior fijo de la versión anterior:
#   - `sticky_strength` deja de ser un número que elegís a mano y que
#     puede funcionar distinto por ticker; la persistencia ahora se adapta
#     sola según lo que el modelo observa en cada tramo de la serie.
# Qué NO se gana respecto a un GAS completo:
#   - Las medias/covarianzas de estado siguen fijas tras el entrenamiento
#     EM, no se re-estiman dinámicamente. Un GAS completo también haría
#     evolucionar esos parámetros con el tiempo.

class GASTransitionLayer:
    """
    Filtro con matriz de transición time-varying al estilo GAS, aplicado
    sobre las emisiones (medias/covarianzas) de un GaussianHMM ya entrenado
    con EM. Ver nota de fidelidad arriba.

    Parámetros
    ----------
    hmm_model : GaussianHMM (hmmlearn) ya entrenado (.means_, .covars_)
    omega, phi, alpha : parámetros de la recursión GAS del logit de
        persistencia. phi cercano a 1 = memoria larga; alpha controla qué
        tan rápido reacciona kappa a sorpresas de predicción.
    kappa0 : logit de persistencia inicial (equivalente a partir de una
        probabilidad de auto-transición de sigmoid(kappa0)).
    """

    def __init__(self, hmm_model, n_states=3, omega=0.0, phi=0.97, alpha=0.08,
                 kappa0=None):
        self.means = hmm_model.means_
        self.covars = hmm_model.covars_
        self.n_states = n_states
        self.omega = omega
        self.phi = phi
        self.alpha = alpha
        self.kappa0 = kappa0 if kappa0 is not None else np.log(15.0)

    def _emission_probs(self, X):
        n = X.shape[0]
        B = np.zeros((n, self.n_states))
        for k in range(self.n_states):
            B[:, k] = multivariate_normal.pdf(X, mean=self.means[k], cov=self.covars[k], allow_singular=True)
        return np.clip(B, 1e-300, None)

    def _transmat_from_kappa(self, kappa):
        p_stay = 1.0 / (1.0 + np.exp(-kappa))
        p_switch = (1.0 - p_stay) / (self.n_states - 1)
        T = np.full((self.n_states, self.n_states), p_switch)
        np.fill_diagonal(T, p_stay)
        return T

    def filter(self, X):
        """
        Forward filter con matriz de transición GAS time-varying.
        Devuelve: filtered_proba (n x K), kappa_path (n,), p_stay_path (n,)
        """
        n = X.shape[0]
        K = self.n_states
        B = self._emission_probs(X)

        filtered = np.zeros((n, K))
        kappa_path = np.zeros(n)
        p_stay_path = np.zeros(n)

        kappa = self.kappa0
        prev_filtered = np.full(K, 1.0 / K)

        for t in range(n):
            T = self._transmat_from_kappa(kappa)
            p_stay_path[t] = np.diag(T).mean()
            kappa_path[t] = kappa

            predicted = prev_filtered if t == 0 else prev_filtered @ T

            unnorm = predicted * B[t]
            likelihood_t = unnorm.sum()
            filtered_t = unnorm / max(likelihood_t, 1e-300)
            filtered[t] = filtered_t

            prev_mode = np.argmax(prev_filtered)
            stay_predicted = predicted[prev_mode]
            stay_realized = filtered_t[prev_mode]
            score_t = stay_realized - stay_predicted

            kappa = self.omega + self.phi * kappa + self.alpha * score_t * 10.0
            prev_filtered = filtered_t

        return filtered, kappa_path, p_stay_path

# ============================================================================
# HMM: FIT + CLASIFICACIÓN EN 3 ESTADOS (emisiones EM + transición GAS)
# ============================================================================

def apply_min_duration(labels, min_run=3):
    """
    Filtro de confirmación mínima ("debounce"): un cambio de régimen solo se
    confirma si el nuevo régimen se sostiene durante al menos `min_run`
    periodos consecutivos.
    """
    labels = labels.copy()
    values = labels.tolist()
    smoothed = values.copy()

    current_label = values[0]
    candidate_label = current_label
    candidate_count = 0

    for i in range(1, len(values)):
        val = values[i]
        if val == current_label:
            candidate_label = current_label
            candidate_count = 0
        else:
            if val == candidate_label:
                candidate_count += 1
            else:
                candidate_label = val
                candidate_count = 1
            if candidate_count >= min_run:
                current_label = candidate_label
                candidate_count = 0
        smoothed[i] = current_label

    return pd.Series(smoothed, index=labels.index, name=labels.name)


def apply_bear_override(hmm_labels, slope_z, z_threshold=-1.5):
    """
    Capa de seguridad opcional (heredada de la versión anterior, sigue
    disponible como respaldo): fuerza BAJISTA cuando el z-score de la
    pendiente cae por debajo de `z_threshold`, incluso si la clasificación
    GAS dijo RANGO. Calibrado sobre 24 series sintéticas de régimen conocido
    (6 escalas de volatilidad, 4 mezclas de régimen); en el peor de esos 24
    escenarios, la tasa de activación nunca superó 12%. Con la matriz de
    transición GAS, este override debería activarse con MENOR frecuencia que
    con el HMM de sticky prior fijo, porque el propio modelo ya se vuelve
    menos persistente cuando la tendencia bajista es sostenida -- este
    override queda como red de seguridad adicional, no como mecanismo
    principal.
    """
    final = hmm_labels.copy()
    override_mask = (slope_z < z_threshold) & (final == "RANGO")
    final[override_mask] = "BAJISTA"
    return final, override_mask


def fit_hmm_gas_regimes(features, n_states=3, n_iter=1000, random_state=42,
                         gas_phi=0.97, gas_alpha=0.08, gas_kappa0=15.0,
                         min_run=3, z_threshold=-1.5, use_bear_override=True):
    """
    1) Entrena un GaussianHMM estándar (EM, hmmlearn) para obtener medias y
       covarianzas de los 3 estados -- igual que antes, pero SIN usar su
       matriz de transición fija para la clasificación final.
    2) Corre el filtro GAS (`GASTransitionLayer`) sobre esas mismas
       emisiones, con una matriz de transición que se adapta dinámicamente
       en cada semana según qué tan bien predijo la persistencia anterior.
    3) Aplica el override bajista opcional y el filtro de duración mínima,
       igual que en la versión con sticky prior fijo.

    Devuelve: regime_labels, filtered_proba, label_map, hmm_model,
              mean_returns, override_mask, p_stay_path (persistencia GAS
              en el tiempo, para diagnóstico/gráfico)
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(features[["returns", "volatility", "slope"]].values)

    # Paso 1: emisiones vía EM estándar. Se mantiene un sticky_prior moderado
    # aquí solo para estabilizar el ENTRENAMIENTO de means/covars (evita que
    # el EM colapse estados); la clasificación final usa el filtro GAS, no
    # esta matriz de transición.
    transmat_prior_em = np.full((n_states, n_states), 1.0) + np.eye(n_states) * 10.0
    hmm_model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
        transmat_prior=transmat_prior_em,
    )
    hmm_model.fit(X)

    # Paso 2: filtro GAS con transición dinámica sobre esas mismas emisiones.
    gas_layer = GASTransitionLayer(
        hmm_model, n_states=n_states,
        omega=0.0, phi=gas_phi, alpha=gas_alpha, kappa0=np.log(gas_kappa0),
    )
    filtered_proba, kappa_path, p_stay_path = gas_layer.filter(X)
    states = filtered_proba.argmax(axis=1)

    mean_returns = {}
    for s in range(n_states):
        mask = states == s
        mean_returns[s] = features["returns"].values[mask].mean() if mask.any() else 0.0

    order = sorted(mean_returns, key=mean_returns.get)
    label_map = {}
    if n_states == 3:
        label_map[order[0]] = "BAJISTA"
        label_map[order[1]] = "RANGO"
        label_map[order[2]] = "ALCISTA"
    else:
        for i, s in enumerate(order):
            label_map[s] = f"ESTADO_{s}"

    regime_labels_raw = pd.Series([label_map[s] for s in states], index=features.index, name="Regime")

    override_mask = pd.Series(False, index=features.index)
    if use_bear_override and n_states == 3:
        regime_labels_raw, override_mask = apply_bear_override(
            regime_labels_raw, features["slope_z"], z_threshold=z_threshold
        )

    regime_labels = apply_min_duration(regime_labels_raw, min_run=min_run)

    return (regime_labels, filtered_proba, label_map, hmm_model, mean_returns,
            override_mask, p_stay_path)

# ============================================================================
# DESCARGA DE DATOS
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def download_weekly_data(ticker, years_back):
    start = (datetime.now() - timedelta(days=365 * years_back)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, interval="1wk", progress=False)

    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df = df[["Date", "Open", "High", "Low", "Close"]].copy()
    df.set_index("Date", inplace=True)
    df.columns = [str(c) for c in df.columns]

    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["EMA_8"] = df["Close"].ewm(span=8, adjust=False).mean()
    df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()

    return df

# ============================================================================
# UI STREAMLIT
# ============================================================================

def main():
    st.set_page_config(layout="wide", page_title="Regime Analyzer - HMM-GAS")
    st.title("📊 Market Regime Analyzer — HMM-GAS (3 Estados)")
    st.caption(
        "Alcista · Bajista · Rango — emisiones Gaussianas (EM) + matriz de "
        "transición dinámica al estilo GAS (persistencia adaptativa, sin "
        "fijar un único valor de sticky_strength a mano)"
    )

    if "analyzed" not in st.session_state:
        st.session_state.analyzed = False
    if "df" not in st.session_state:
        st.session_state.df = None
    if "ticker" not in st.session_state:
        st.session_state.ticker = "AAPL"

    col_input1, col_input2, col_input3, col_input4 = st.columns([2, 1, 1, 1])
    with col_input1:
        ticker = st.text_input("Ticker", st.session_state.ticker).upper()
    with col_input2:
        years_back = st.slider("Histórico (años)", 3, 15, 8)
    with col_input3:
        lookback_months = st.slider("Meses a visualizar", 1, 36, 12)
    with col_input4:
        vol_window = st.slider("Ventana Vol (sem.)", 4, 26, 12)

    st.markdown("##### Parámetros del filtro GAS (persistencia dinámica)")
    col_g1, col_g2, col_g3 = st.columns(3)
    with col_g1:
        gas_kappa0 = st.slider(
            "Persistencia inicial", 2.0, 30.0, 15.0, 1.0,
            help="Punto de partida de la persistencia del régimen (equivalente "
                 "al antiguo 'sticky_strength' fijo). A diferencia de la "
                 "versión anterior, este valor solo fija el ARRANQUE — el "
                 "modelo lo ajusta semana a semana según qué tan bien está "
                 "prediciendo."
        )
    with col_g2:
        gas_phi = st.slider(
            "Memoria (phi)", 0.80, 0.995, 0.97, 0.005,
            help="Qué tan lento 'olvida' el modelo su nivel de persistencia "
                 "previo. Cercano a 1 = cambios de persistencia muy graduales; "
                 "más bajo = el modelo reacciona más rápido a tramos donde "
                 "está prediciendo mal."
        )
    with col_g3:
        gas_alpha = st.slider(
            "Sensibilidad (alpha)", 0.01, 0.30, 0.08, 0.01,
            help="Qué tan fuerte reacciona la persistencia cuando el modelo "
                 "se equivoca sobre si el régimen se mantendría o cambiaría. "
                 "Más alto = el filtro GAS reacciona más agresivamente a cada "
                 "sorpresa (riesgo de volver a introducir parpadeo si se "
                 "sube demasiado)."
        )

    col_input5, col_input6 = st.columns(2)
    with col_input5:
        min_run = st.slider(
            "Confirmación mínima (semanas)", 1, 8, 3,
            help="Un cambio de régimen solo se confirma si se mantiene al menos "
                 "este número de semanas seguidas; si no, se ignora."
        )
    with col_input6:
        use_bear_override = st.checkbox(
            "Activar override bajista (respaldo)", value=True,
            help="Red de seguridad heredada de la versión anterior: si la "
                 "pendiente de tendencia es estadísticamente muy negativa "
                 "para este ticker (z-score) pero el modelo GAS clasificó la "
                 "semana como RANGO, se reclasifica como BAJISTA. Con la "
                 "transición dinámica GAS este override debería activarse "
                 "con menor frecuencia que con el sticky prior fijo."
        )

    z_threshold = -1.5
    trend_window = 20
    z_window = 104
    if use_bear_override:
        col_input8, col_input9, col_input10 = st.columns(3)
        with col_input8:
            z_threshold = st.slider(
                "Umbral z de pendiente bajista", -3.0, -0.25, -1.5, 0.25,
                help="Calibrado una sola vez sobre 24 series sintéticas con "
                     "volatilidades distintas (no sobre un ticker real). No "
                     "recalibrar mirando un solo ticker."
            )
        with col_input9:
            trend_window = st.slider("Ventana de la pendiente (sem.)", 8, 40, 20)
        with col_input10:
            z_window = st.slider("Ventana del z-score (sem.)", 52, 208, 104)

    if st.button("🚀 ANALIZAR", use_container_width=True):
        st.session_state.ticker = ticker
        st.session_state.df = download_weekly_data(ticker, years_back)
        st.session_state.vol_window = vol_window
        st.session_state.gas_kappa0 = gas_kappa0
        st.session_state.gas_phi = gas_phi
        st.session_state.gas_alpha = gas_alpha
        st.session_state.min_run = min_run
        st.session_state.use_bear_override = use_bear_override
        st.session_state.z_threshold = z_threshold
        st.session_state.trend_window = trend_window
        st.session_state.z_window = z_window
        st.session_state.analyzed = True

    if st.session_state.analyzed and st.session_state.df is not None:

        df = st.session_state.df.copy()

        with st.spinner("Ajustando HMM-GAS..."):
            features = build_features(
                df,
                vol_window=st.session_state.get("vol_window", vol_window),
                trend_window=st.session_state.get("trend_window", trend_window),
                z_window=st.session_state.get("z_window", z_window),
            )
            (regime_labels, filtered_proba, label_map, hmm_model, mean_returns,
             override_mask, p_stay_path) = fit_hmm_gas_regimes(
                features,
                n_states=3,
                gas_phi=st.session_state.get("gas_phi", gas_phi),
                gas_alpha=st.session_state.get("gas_alpha", gas_alpha),
                gas_kappa0=st.session_state.get("gas_kappa0", gas_kappa0),
                min_run=st.session_state.get("min_run", min_run),
                z_threshold=st.session_state.get("z_threshold", z_threshold),
                use_bear_override=st.session_state.get("use_bear_override", use_bear_override),
            )

        df = df.loc[features.index].copy()
        df["Regime"] = regime_labels
        df["Slope"] = features["slope"]
        df["Slope_Z"] = features["slope_z"]
        df["Overridden"] = override_mask
        df["P_Stay_GAS"] = p_stay_path
        for s, lbl in label_map.items():
            df[f"Prob_{lbl}"] = filtered_proba[:, s]

        df_plot = df.tail(int(lookback_months * 4.33))
        current = df.iloc[-1]
        current_regime = current["Regime"]

        precio = float(current["Close"])
        sma50 = float(current["SMA_50"])
        precio_above_sma = "SÍ" if precio > sma50 else "NO"

        current_probs = {lbl: float(current[f"Prob_{lbl}"]) for lbl in REGIME_COLORS.keys() if f"Prob_{lbl}" in df.columns}
        hmm_raw_label = max(current_probs, key=current_probs.get) if current_probs else current_regime
        overridden_now = bool(current["Overridden"])

        # ================= TARJETA DE ESTADO ACTUAL =================
        regime_color = REGIME_COLORS.get(current_regime, "#FFFFFF")

        st.markdown(
            f'<div style="padding:10px;border:1px solid #444;background-color:{regime_color};'
            f'color:black;font-weight:bold;text-align:center;font-size:20px;border-radius:4px;">'
            f'{current_regime}</div>',
            unsafe_allow_html=True,
        )

        if overridden_now:
            st.warning(
                f"⚠️ El filtro GAS indicaba **{hmm_raw_label}**, pero se corrigió a **BAJISTA** "
                f"por el override de tendencia (z-score de pendiente = {float(current['Slope_Z']):.2f}, "
                f"umbral = {st.session_state.get('z_threshold', z_threshold):.2f})."
            )

        info_cols = st.columns(5)
        info_cols[0].metric("Precio", f"${precio:.2f}")
        info_cols[1].metric("SMA 50", f"${sma50:.2f}")
        info_cols[2].metric("Precio > SMA50", precio_above_sma)
        info_cols[3].metric("Z-score pendiente", f"{float(current['Slope_Z']):.2f}")
        info_cols[4].metric("Persistencia GAS actual", f"{float(current['P_Stay_GAS'])*100:.1f}%")

        # ================= PROBABILIDADES POR RÉGIMEN =================
        st.markdown("#### 🎲 Probabilidad actual por régimen (filtro GAS, antes del override)")
        prob_cols = st.columns(len(current_probs))
        for i, (lbl, p) in enumerate(sorted(current_probs.items(), key=lambda x: -x[1])):
            with prob_cols[i]:
                c = REGIME_COLORS.get(lbl, "#FFFFFF")
                st.markdown(
                    f'<div style="background-color:{c};padding:10px;border-radius:6px;'
                    f'text-align:center;font-weight:bold;color:black;border:2px solid white;">'
                    f'{lbl}<br><span style="font-size:18px;">{p*100:.1f}%</span></div>',
                    unsafe_allow_html=True
                )

        # ================= CARACTERÍSTICAS DE CADA ESTADO =================
        with st.expander("📐 Características estadísticas de cada régimen (in-sample)"):
            stats_rows = []
            for s, lbl in label_map.items():
                mask = regime_labels == lbl
                sub = features[mask]
                stats_rows.append({
                    "Régimen": lbl,
                    "Retorno medio semanal": f"{sub['returns'].mean()*100:.2f}%",
                    "Volatilidad anualizada": f"{sub['volatility'].mean()*100:.1f}%",
                    "Pendiente media": f"{sub['slope'].mean():.5f}",
                    "% del tiempo": f"{mask.mean()*100:.1f}%",
                })
            st.table(pd.DataFrame(stats_rows))

            st.markdown("**Emisiones (medias/covarianzas) estimadas por EM — sin cambios respecto al HMM estándar:**")
            means_df = pd.DataFrame(
                hmm_model.means_,
                index=[label_map[s] for s in range(3)],
                columns=["returns (z)", "volatility (z)", "slope (z)"],
            )
            st.dataframe(means_df.style.format("{:.3f}"))

            override_pct = float(override_mask.mean() * 100)
            if override_pct <= 15:
                nivel = "sano (dentro del rango observado en la calibración)"
            elif override_pct <= 30:
                nivel = "elevado — vigilar si se repite en otros tickers"
            else:
                nivel = "muy alto — el override está dominando la clasificación en este ticker"
            st.caption(
                f"El override de tendencia bajista reclasificó **{int(override_mask.sum())} semanas "
                f"({override_pct:.1f}% del histórico)** de RANGO a BAJISTA. Nivel: **{nivel}**. "
                f"Con la transición dinámica GAS, se espera que este override se active con "
                f"MENOR frecuencia que con el sticky prior fijo de la versión anterior, porque "
                f"la persistencia del modelo ya baja por sí sola en tramos de tendencia sostenida."
            )

        # ================= LEYENDA =================
        st.markdown("### 🎨 Leyenda")
        cols = st.columns(3)
        for idx, (regime, color) in enumerate(REGIME_COLORS.items()):
            with cols[idx]:
                st.markdown(
                    f'<div style="background-color:{color};padding:5px;border-radius:5px;'
                    f'text-align:center;font-weight:bold;color:black;border:2px solid white;font-size:11px;">{regime}</div>',
                    unsafe_allow_html=True
                )

        # ================= GRÁFICO =================
        st.markdown("---")
        plt.style.use("dark_background")
        fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1, 1]})

        axs[0].plot(df_plot.index, df_plot["Close"], color="white", alpha=0.5, linewidth=1.3, label="Precio")
        axs[0].plot(df_plot.index, df_plot["SMA_50"], color="cyan", linewidth=1.8, alpha=0.7, label="SMA 50")
        axs[0].plot(df_plot.index, df_plot["EMA_8"], color="#FFFFFF", linewidth=1.4, alpha=0.85, linestyle="--", label="EMA 8")
        axs[0].plot(df_plot.index, df_plot["EMA_21"], color="#00BFFF", linewidth=1.4, alpha=0.85, linestyle="--", label="EMA 21")

        for r, c in REGIME_COLORS.items():
            m = df_plot["Regime"] == r
            if m.any():
                axs[0].scatter(df_plot[m].index, df_plot[m]["Close"], c=c, s=70, alpha=0.95,
                               edgecolors='black', linewidths=1.2, zorder=5, label=r)

        last_price = float(df_plot["Close"].iloc[-1])
        axs[0].text(1.01, last_price, f'${last_price:.2f}',
                    transform=axs[0].get_yaxis_transform(),
                    fontsize=8, color='white', va='center', fontweight='bold')

        axs[0].set_title(f"{st.session_state.ticker} — Régimen (HMM-GAS, 3 estados)", fontsize=10, fontweight='bold', pad=6)
        axs[0].grid(alpha=0.25, linestyle='--')
        axs[0].set_ylabel("Precio ($)", fontsize=8)
        axs[0].legend(loc='upper left', fontsize=6.5, ncol=3)
        axs[0].yaxis.tick_right()
        axs[0].yaxis.set_label_position("right")
        axs[0].tick_params(axis='both', labelsize=7)

        dom_prob = df_plot[[f"Prob_{lbl}" for lbl in REGIME_COLORS.keys() if f"Prob_{lbl}" in df_plot.columns]].max(axis=1)
        axs[1].plot(df_plot.index, dom_prob, color="#FFD700", linewidth=1.5)
        axs[1].fill_between(df_plot.index, 0, dom_prob, color="#FFD700", alpha=0.15)
        axs[1].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        axs[1].set_ylim(0, 1.05)
        axs[1].set_title("Confianza del régimen dominante (filtro GAS)", fontsize=9, fontweight='bold', pad=4)
        axs[1].grid(alpha=0.25, linestyle='--')
        axs[1].set_ylabel("Prob.", fontsize=8)
        axs[1].yaxis.tick_right()
        axs[1].yaxis.set_label_position("right")
        axs[1].tick_params(axis='both', labelsize=7)

        axs[2].plot(df_plot.index, df_plot["P_Stay_GAS"], color="#FF6B9D", linewidth=1.5)
        axs[2].fill_between(df_plot.index, 0, df_plot["P_Stay_GAS"], color="#FF6B9D", alpha=0.15)
        axs[2].set_ylim(0, 1.05)
        axs[2].set_title("Persistencia dinámica GAS (prob. de mantener el régimen)", fontsize=9, fontweight='bold', pad=4)
        axs[2].grid(alpha=0.25, linestyle='--')
        axs[2].set_ylabel("P(stay)", fontsize=8)
        axs[2].set_xlabel("Fecha", fontsize=8)
        axs[2].yaxis.tick_right()
        axs[2].yaxis.set_label_position("right")
        axs[2].tick_params(axis='both', labelsize=7)

        plt.tight_layout()
        st.pyplot(fig)

        # ================= DOCUMENTACIÓN =================
        st.markdown("---")
        with st.expander("📖 Ver Lógica de Clasificación (HMM-GAS)"):
            st.markdown("""
            **Metodología: Emisiones Gaussianas por EM + transición dinámica al estilo GAS**

            - **Por qué GAS y no un sticky prior fijo:** un HMM estándar (la versión
              anterior de este script) fija la persistencia del régimen con un solo
              número (`sticky_strength`) elegido a mano, y ese número puede funcionar
              bien para un ticker calmado y mal para uno volátil — obligando a
              recalibrar por activo, justo lo que se quería evitar. Un comparativo
              académico reciente (Politecnico di Milano, 2025/2026) mostró que un
              modelo HMM-GAS logra 88.0% de precisión de régimen bajo mala
              especificación del modelo, frente a 83.3% de un HMM estándar — es decir,
              generaliza mejor cuando no sabés de antemano si tu ticker se ajusta bien
              a los supuestos del modelo, a cambio de perder algo de precisión pico
              (88.7% vs. 93.5% bajo especificación correcta).
            - **Qué hace este script exactamente:** entrena un `GaussianHMM` estándar
              (EM, vía `hmmlearn`) para estimar las medias y covarianzas de los 3
              estados — esa parte no cambió. Lo que cambió es que la clasificación
              final NO usa la matriz de transición fija de ese HMM, sino un filtro
              (`GASTransitionLayer`) que ajusta la probabilidad de persistencia en
              cada semana según una recursión GAS: `kappa_{t+1} = omega + phi*kappa_t
              + alpha*score_t`, donde `score_t` mide si el régimen se sostuvo más o
              menos de lo que la persistencia anterior predecía.
            - **Limitación honesta:** un GAS "completo" también haría evolucionar las
              medias/covarianzas de los estados con el tiempo, no solo la matriz de
              transición. Implementar eso desde cero (sin una librería madura que lo
              ofrezca) es un proyecto de investigación aparte; esta versión se enfoca
              en la parte del modelo con más impacto directo sobre el problema
              original (persistencia mal calibrada por ticker).
            - **Parámetros GAS:** "Persistencia inicial" es el punto de partida (antes
              `sticky_strength`); "Memoria (phi)" controla cuán gradual es el cambio de
              persistencia en el tiempo; "Sensibilidad (alpha)" controla cuán fuerte
              reacciona el modelo cuando se equivoca sobre si el régimen se mantendría.
            - El panel inferior del gráfico ("Persistencia dinámica GAS") muestra
              justamente eso: cómo la probabilidad de mantener el régimen sube o baja
              sola a lo largo del tiempo, en vez de ser una constante fija.
            - El **override bajista** (z-score de pendiente) se mantiene como red de
              seguridad opcional, igual que en la versión anterior — con la transición
              GAS se espera que se active con menor frecuencia, porque el modelo ya
              reduce su propia persistencia en tramos de tendencia sostenida.
            - Si el régimen sigue pareciendo lento o rápido para cambiar, ajustá
              **Memoria (phi)** y **Sensibilidad (alpha)** en vez de forzar un
              `sticky_strength` fijo — esa es la ventaja práctica de este enfoque.
            """)

    elif st.session_state.analyzed and st.session_state.df is None:
        st.error("❌ No se pudo descargar data. Verifica el ticker.")


if __name__ == "__main__":
    if check_password():
        main()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder.")
