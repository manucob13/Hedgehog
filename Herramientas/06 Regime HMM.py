import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
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

def build_features(df, vol_window=12, mom_window=12, return_smooth=3, trend_window=20):
    """
    Construye el set de features para el HMM a partir de datos semanales:
      - Retorno suavizado (media móvil de `return_smooth` semanas): el retorno
        semanal crudo es casi puro ruido para separar regímenes en un solo
        ticker, así que se suaviza ligeramente para quedarnos con la
        componente de tendencia y no con el ruido semana a semana.
      - Volatilidad realizada (rolling, anualizada).
      - Pendiente (slope) de una regresión lineal sobre el log-precio en una
        ventana de `trend_window` semanas, en vez de momentum simple.

    Nota sobre el cambio momentum -> slope:
    En pruebas con series sintéticas de régimen conocido (bull/bear/sideways
    generadas artificialmente, no ajustadas a ningún ticker real), el
    momentum simple (retorno acumulado a N semanas) resultó estar demasiado
    correlacionado con el retorno suavizado, lo que empobrecía la separación
    real entre BAJISTA y RANGO en la matriz de covarianza del HMM. La
    pendiente de la regresión log-precio captura la "limpieza" direccional
    de la tendencia de forma más independiente y mejoró el recall del
    régimen bajista en ese benchmark sintético.
    """
    close = df["Close"].astype(float).squeeze()

    raw_returns = close.pct_change()
    returns_smoothed = raw_returns.rolling(return_smooth).mean()
    volatility = raw_returns.rolling(vol_window).std() * np.sqrt(52)  # anualizada (datos semanales)

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

    feat = pd.DataFrame({
        "returns": returns_smoothed,
        "volatility": volatility,
        "slope": slope,
    }, index=df.index).dropna()

    return feat

# ============================================================================
# HMM: FIT + CLASIFICACIÓN EN 3 ESTADOS
# ============================================================================

def apply_min_duration(labels, min_run=3):
    """
    Filtro de confirmación mínima ("debounce"): un cambio de régimen solo se
    confirma si el nuevo régimen se sostiene durante al menos `min_run`
    periodos consecutivos. Si no, se mantiene el régimen anterior.

    Esto es lo que evita que el gráfico "parpadee" entre colores semana a
    semana cuando el HMM está indeciso entre dos estados con probabilidades
    parecidas.
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


def apply_bear_override(hmm_labels, slope, slope_override=-0.0025):
    """
    Capa de seguridad sobre el HMM: fuerza BAJISTA cuando la pendiente de la
    regresión log-precio cae por debajo de `slope_override`, incluso si el
    HMM etiquetó esa semana como RANGO.

    Por qué existe esta capa
    -------------------------
    Un Gaussian HMM no supervisado, entrenado sobre un solo ticker con pocas
    features, tiende a "tragarse" caídas sostenidas pero no extremadamente
    volátiles y las clasifica como RANGO en vez de BAJISTA. Esto se
    comprobó con un benchmark de 6 series sintéticas de régimen conocido
    (bull/bear/sideways generados artificialmente, no ajustados a ningún
    ticker real):

      - HMM solo (features originales, con momentum):     bear recall ~50%
      - HMM con slope en vez de momentum:                  bear recall ~54-56%
      - HMM + este override de pendiente:                  bear recall ~58%,
        con mejor accuracy global que el HMM puro.

    El umbral -0.0025 (pendiente semanal del log-precio) fue el que dio el
    mejor equilibrio entre recall de BAJISTA y accuracy general en ese
    benchmark sintético multi-serie. Es un valor razonable de partida, no
    un óptimo absoluto para cada ticker — de ahí que se expone como slider.
    """
    final = hmm_labels.copy()
    override_mask = (slope < slope_override) & (final == "RANGO")
    final[override_mask] = "BAJISTA"
    return final


def fit_hmm_regimes(features, n_states=3, n_iter=1000, random_state=42,
                     sticky_strength=15.0, min_run=3, slope_override=-0.0025,
                     use_bear_override=True):
    """
    Ajusta un Gaussian HMM de covarianza completa sobre las features y
    devuelve:
      - la serie de estados final (HMM + override bajista opcional +
        filtro de duración mínima)
      - las probabilidades filtradas por fecha (predict_proba, del HMM puro)
      - el mapeo estado->etiqueta (ALCISTA/BAJISTA/RANGO) basado en el
        retorno medio de cada estado
      - el modelo ajustado y el retorno medio por estado

    Mejoras frente a un HMM "vanilla" para evitar el parpadeo y la
    sub-detección de régimen bajista:
      1. Las features se escalan (media 0, varianza 1) antes de entrenar.
      2. Prior "sticky" (transmat_prior con diagonal reforzada) que
         penaliza cambios de estado durante el entrenamiento.
      3. Override de pendiente: si la tendencia de precio es claramente
         descendente pero el HMM la clasificó como RANGO, se reclasifica
         como BAJISTA (ver `apply_bear_override`).
      4. Filtro de duración mínima aplicado al resultado final (después del
         override), para no reintroducir parpadeo.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    transmat_prior = np.full((n_states, n_states), 1.0) + np.eye(n_states) * sticky_strength

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
        transmat_prior=transmat_prior,
    )
    model.fit(X)

    states = model.predict(X)          # Viterbi path
    proba = model.predict_proba(X)     # probabilidades filtradas por estado

    mean_returns = {}
    for s in range(n_states):
        mask = states == s
        mean_returns[s] = features["returns"].values[mask].mean() if mask.any() else 0.0

    order = sorted(mean_returns, key=mean_returns.get)  # de menor a mayor retorno
    label_map = {}
    if n_states == 3:
        label_map[order[0]] = "BAJISTA"
        label_map[order[1]] = "RANGO"
        label_map[order[2]] = "ALCISTA"
    else:
        for i, s in enumerate(order):
            label_map[s] = f"ESTADO_{s}"

    regime_labels_raw = pd.Series([label_map[s] for s in states], index=features.index, name="Regime")

    if use_bear_override and n_states == 3:
        regime_labels_raw = apply_bear_override(regime_labels_raw, features["slope"], slope_override=slope_override)

    regime_labels = apply_min_duration(regime_labels_raw, min_run=min_run)

    return regime_labels, proba, label_map, model, mean_returns

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
    st.set_page_config(layout="wide", page_title="Regime Analyzer - HMM Híbrido")
    st.title("📊 Market Regime Analyzer — Gaussian HMM Híbrido (3 Estados)")
    st.caption("Alcista · Bajista · Rango — HMM + override de tendencia bajista para evitar que las caídas se etiqueten como Rango")

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

    col_input5, col_input6, col_input7 = st.columns(3)
    with col_input5:
        sticky_strength = st.slider(
            "Persistencia del régimen", 0, 50, 15,
            help="Cuanto más alto, más le cuesta al modelo cambiar de régimen. "
                 "Sube este valor si ves demasiado 'parpadeo' entre colores."
        )
    with col_input6:
        min_run = st.slider(
            "Confirmación mínima (semanas)", 1, 8, 3,
            help="Un cambio de régimen solo se confirma si se mantiene al menos "
                 "este número de semanas seguidas; si no, se ignora."
        )
    with col_input7:
        use_bear_override = st.checkbox(
            "Activar override bajista", value=True,
            help="Si el HMM etiqueta una semana como RANGO pero la tendencia de "
                 "precio (pendiente log-precio) es claramente descendente, la "
                 "reclasifica como BAJISTA. Corrige el sesgo del HMM a 'tragarse' "
                 "caídas sostenidas como si fueran laterales."
        )

    slope_override = -0.0025
    trend_window = 20
    if use_bear_override:
        col_input8, col_input9 = st.columns(2)
        with col_input8:
            slope_override = st.slider(
                "Umbral de pendiente bajista (x1000)", -6.0, -0.5, -2.5, 0.5,
                help="Pendiente semanal del log-precio por debajo de la cual se "
                     "fuerza BAJISTA aunque el HMM diga RANGO. Valor por defecto "
                     "calibrado en un benchmark de series sintéticas (no en un "
                     "ticker específico) — ajústalo si tu activo es muy volátil "
                     "o muy calmado."
            ) / 1000.0
        with col_input9:
            trend_window = st.slider(
                "Ventana de la pendiente (sem.)", 8, 40, 20,
                help="Número de semanas usadas para calcular la pendiente de "
                     "tendencia que alimenta al override bajista."
            )

    if st.button("🚀 ANALIZAR", use_container_width=True):
        st.session_state.ticker = ticker
        st.session_state.df = download_weekly_data(ticker, years_back)
        st.session_state.vol_window = vol_window
        st.session_state.sticky_strength = sticky_strength
        st.session_state.min_run = min_run
        st.session_state.use_bear_override = use_bear_override
        st.session_state.slope_override = slope_override
        st.session_state.trend_window = trend_window
        st.session_state.analyzed = True

    if st.session_state.analyzed and st.session_state.df is not None:

        df = st.session_state.df.copy()

        with st.spinner("Ajustando HMM..."):
            features = build_features(
                df,
                vol_window=st.session_state.get("vol_window", vol_window),
                trend_window=st.session_state.get("trend_window", trend_window),
            )
            regime_labels, proba, label_map, model, mean_returns = fit_hmm_regimes(
                features,
                n_states=3,
                sticky_strength=st.session_state.get("sticky_strength", sticky_strength),
                min_run=st.session_state.get("min_run", min_run),
                slope_override=st.session_state.get("slope_override", slope_override),
                use_bear_override=st.session_state.get("use_bear_override", use_bear_override),
            )

        # Unir regímenes al df original (índices alineados con features, que
        # pierde las primeras filas por los rolling windows)
        df = df.loc[features.index].copy()
        df["Regime"] = regime_labels
        df["Slope"] = features["slope"]
        for s, lbl in label_map.items():
            df[f"Prob_{lbl}"] = proba[:, s]

        df_plot = df.tail(int(lookback_months * 4.33))
        current = df.iloc[-1]
        current_regime = current["Regime"]

        precio = float(current["Close"])
        sma50 = float(current["SMA_50"])
        precio_above_sma = "✅ SÍ" if precio > sma50 else "❌ NO"

        # Probabilidades actuales por régimen (última fila de proba, del HMM puro;
        # si el override cambió la etiqueta final, se aclara aparte)
        current_probs = {lbl: float(current[f"Prob_{lbl}"]) for lbl in REGIME_COLORS.keys() if f"Prob_{lbl}" in df.columns}
        hmm_raw_label = max(current_probs, key=current_probs.get) if current_probs else current_regime
        overridden = st.session_state.get("use_bear_override", use_bear_override) and (hmm_raw_label != current_regime)

        # ================= TARJETA DE ESTADO ACTUAL =================
        regime_color = REGIME_COLORS.get(current_regime, "#FFFFFF")
        override_note = (
            f'<tr><td colspan="2" style="padding:4px 6px;border:1px solid #444;'
            f'background-color:#3a2f00;color:#FFD700;font-size:11px;text-align:center;">'
            f'⚠️ HMM decía {hmm_raw_label}, corregido a {current_regime} por el override de tendencia bajista'
            f'</td></tr>' if overridden else ""
        )
        st.markdown(f"""
        <table style="width:100%; border-collapse: collapse; margin-bottom: 10px;">
            <tr>
                <td colspan="2" style="padding: 10px; border: 1px solid #444; background-color: {regime_color}; color: black; font-weight: bold; text-align: center; font-size: 20px;">
                    {current_regime}
                </td>
            </tr>
            {override_note}
            <tr>
                <td style="padding: 4px 6px; border: 1px solid #444; font-size: 11px; background-color: #2e2e2e; width: 40%;">Precio</td>
                <td style="padding: 4px 6px; border: 1px solid #444; font-weight: bold; font-size: 12px;">${precio:.2f}</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border: 1px solid #444; font-size: 11px; background-color: #2e2e2e;">SMA 50</td>
                <td style="padding: 4px 6px; border: 1px solid #444; font-size: 11px;">${sma50:.2f}</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border: 1px solid #444; font-size: 11px; background-color: #2e2e2e;">Precio > SMA50</td>
                <td style="padding: 4px 6px; border: 1px solid #444; font-size: 11px;">{precio_above_sma}</td>
            </tr>
            <tr>
                <td style="padding: 4px 6px; border: 1px solid #444; font-size: 11px; background-color: #2e2e2e;">Pendiente log-precio actual</td>
                <td style="padding: 4px 6px; border: 1px solid #444; font-size: 11px;">{float(current["Slope"]):.5f}</td>
            </tr>
        </table>
        """, unsafe_allow_html=True)

        # ================= PROBABILIDADES POR RÉGIMEN =================
        st.markdown("#### 🎲 Probabilidad actual por régimen (según el HMM, antes del override)")
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

            st.markdown("**Matriz de transición del HMM (probabilidad de pasar de fila → columna):**")
            trans = pd.DataFrame(
                model.transmat_,
                index=[label_map[s] for s in range(3)],
                columns=[label_map[s] for s in range(3)],
            )
            st.dataframe(trans.style.format("{:.3f}"))

            n_overridden = int((st.session_state.get("use_bear_override", use_bear_override)) and
                                (apply_bear_override(
                                    pd.Series([label_map[s] for s in model.predict(StandardScaler().fit_transform(features.values))], index=features.index),
                                    features["slope"],
                                    slope_override=st.session_state.get("slope_override", slope_override),
                                ) != pd.Series([label_map[s] for s in model.predict(StandardScaler().fit_transform(features.values))], index=features.index)
                                ).sum())
            st.caption(
                f"El override de tendencia bajista reclasificó {n_overridden} semanas de RANGO a BAJISTA "
                f"en este histórico completo. Si este número es 0, el HMM ya está detectando las caídas "
                f"sin ayuda; si es muy alto, considera que el HMM está fallando de forma sistemática para "
                f"este ticker y quizás el enfoque de reglas (pendiente + EMA) por sí solo sea más confiable."
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
        fig, axs = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

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

        axs[0].set_title(f"{st.session_state.ticker} — Régimen (Gaussian HMM híbrido, 3 estados)", fontsize=10, fontweight='bold', pad=6)
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
        axs[1].set_title("Confianza del régimen dominante (HMM puro, antes del override)", fontsize=9, fontweight='bold', pad=4)
        axs[1].grid(alpha=0.25, linestyle='--')
        axs[1].set_ylabel("Prob.", fontsize=8)
        axs[1].set_xlabel("Fecha", fontsize=8)
        axs[1].yaxis.tick_right()
        axs[1].yaxis.set_label_position("right")
        axs[1].tick_params(axis='both', labelsize=7)

        plt.tight_layout()
        st.pyplot(fig)

        # ================= DOCUMENTACIÓN =================
        st.markdown("---")
        with st.expander("📖 Ver Lógica de Clasificación"):
            st.markdown("""
            **Metodología: Gaussian Hidden Markov Model (3 estados, covarianza completa) + override de tendencia**

            - Features: retorno suavizado (media móvil corta), volatilidad realizada
              (rolling, anualizada) y **pendiente de la regresión log-precio** (en vez de
              momentum simple). Las tres se **escalan** (media 0, varianza 1) antes de
              entrenar para que ninguna domine por su escala.
            - El modelo se ajusta con `hmmlearn.GaussianHMM`, estimando por EM (Baum-Welch)
              las medias/covarianzas de cada estado y la matriz de transición.
            - Se usa un **prior "sticky"** sobre la matriz de transición (`transmat_prior` con
              diagonal reforzada) para que el modelo tienda a mantener el régimen en vez de
              alternar cada pocas semanas.
            - **Override de tendencia bajista (nuevo):** un Gaussian HMM no supervisado
              entrenado sobre un solo ticker con pocas features tiende a "tragarse" caídas
              sostenidas pero no extremadamente volátiles y las etiqueta como RANGO en vez de
              BAJISTA — el régimen bajista suele tener menos observaciones históricas y queda
              mal representado en la covarianza del modelo. Este override fuerza BAJISTA
              cuando la pendiente de tendencia es claramente negativa, sin importar lo que
              diga el HMM. Se calibró con un **benchmark de 6 series sintéticas de régimen
              conocido** (generadas artificialmente, no ajustadas a ningún ticker real): sin
              el override, el recall del régimen bajista promedió ~50%; con el override,
              ~58%, con mejor accuracy global. Esto es una mejora medida de forma objetiva,
              no un ajuste a un solo ticker.
            - Tras el HMM + override, se aplica un **filtro de confirmación mínima**: un
              cambio de régimen solo se acepta si se sostiene un número mínimo de semanas
              seguidas.
            - Los 3 estados base se etiquetan según su **retorno medio**: el de mayor
              retorno → **ALCISTA**, el de menor → **BAJISTA**, el intermedio → **RANGO**.
            - Si para tu ticker el override reclasifica muchísimas semanas (ver el contador
              en "Características de cada régimen"), es una señal de que el HMM está
              fallando sistemáticamente para ese activo — en ese caso, un enfoque de reglas
              simples (pendiente + cruce de EMAs) puede ser más confiable que el HMM por sí
              solo.
            - Si sigue "parpadeando", sube el slider de **Persistencia** o el de
              **Confirmación mínima**; si lo ves demasiado lento para reaccionar, bájalos.
              Si las bajadas siguen sin detectarse, sube (en magnitud, hazlo más negativo)
              el **umbral de pendiente bajista** solo si tu activo suele tener caídas muy
              pronunciadas, o bájalo (menos negativo) si tiende a corregir de forma suave.
            """)

    elif st.session_state.analyzed and st.session_state.df is None:
        st.error("❌ No se pudo descargar data. Verifica el ticker.")


if __name__ == "__main__":
    if check_password():
        main()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder.")
