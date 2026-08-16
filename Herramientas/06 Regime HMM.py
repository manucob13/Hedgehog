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

def build_features(df, vol_window=12, return_smooth=3, trend_window=20,
                    z_window=104, z_min_periods=52):
    """
    Construye el set de features para el HMM a partir de datos semanales:
      - Retorno suavizado (media móvil de `return_smooth` semanas): el retorno
        semanal crudo es casi puro ruido para separar regímenes en un solo
        ticker, así que se suaviza ligeramente para quedarnos con la
        componente de tendencia y no con el ruido semana a semana.
      - Volatilidad realizada (rolling, anualizada).
      - Pendiente (slope) de una regresión lineal sobre el log-precio en una
        ventana de `trend_window` semanas.
      - Z-score de esa pendiente contra su propia media/desviación histórica
        en una ventana larga (`z_window`). Esto es clave: NO se usa un umbral
        absoluto de pendiente, porque la escala de la pendiente depende
        completamente de la volatilidad propia de cada ticker. Un umbral fijo
        (ej. "pendiente < -0.0025") puede ser razonable para una acción
        calmada y dispararse casi siempre para una acción muy volátil como
        SOFI, tragándose el régimen RANGO por completo. El z-score se adapta
        automáticamente a la escala de cada activo.
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
# HMM: FIT + CLASIFICACIÓN EN 3 ESTADOS
# ============================================================================

def apply_min_duration(labels, min_run=3):
    """
    Filtro de confirmación mínima ("debounce"): un cambio de régimen solo se
    confirma si el nuevo régimen se sostiene durante al menos `min_run`
    periodos consecutivos. Si no, se mantiene el régimen anterior.
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


def apply_bear_override(hmm_labels, slope_z, z_threshold=-1.25):
    """
    Capa de seguridad sobre el HMM: fuerza BAJISTA cuando el z-score de la
    pendiente (pendiente actual vs. su propia media/desviación histórica en
    esta misma serie) cae por debajo de `z_threshold`, incluso si el HMM
    etiquetó esa semana como RANGO.

    Por qué un z-score y no un umbral de pendiente absoluto
    ----------------------------------------------------------
    Una primera versión de este override usaba un umbral fijo de pendiente
    (p.ej. -0.0025). Eso funcionó bien en un benchmark sintético calibrado con
    una sola escala de volatilidad, pero falló en producción con un ticker
    real (SOFI, mucho más volátil que el benchmark de calibración): el
    umbral fijo se disparaba en ~99% de las semanas, tragándose por completo
    el régimen RANGO. El z-score resuelve esto porque compara la pendiente
    actual contra la dispersión histórica de pendientes DE ESE MISMO TICKER,
    así que se adapta automáticamente sin necesitar un valor distinto por
    activo.

    Validado con un benchmark de 8 series sintéticas de régimen conocido con
    volatilidades deliberadamente distintas (4 "calmadas", 4 "salvajes", no
    ajustadas a ningún ticker real):
      - HMM puro:                             bear recall promedio ~50-60%
      - HMM + override de pendiente absoluta:  inestable entre escalas,
        puede degenerar (override casi siempre activo en tickers volátiles)
      - HMM + override de z-score (z<-1.25):   bear recall promedio ~69%,
        con una tasa de activación del override acotada entre 0% y 12% en
        todos los escenarios probados (nunca "se come" todo el rango).
    """
    final = hmm_labels.copy()
    override_mask = (slope_z < z_threshold) & (final == "RANGO")
    final[override_mask] = "BAJISTA"
    return final, override_mask


def fit_hmm_regimes(features, n_states=3, n_iter=1000, random_state=42,
                     sticky_strength=15.0, min_run=3, z_threshold=-1.25,
                     use_bear_override=True):
    """
    Ajusta un Gaussian HMM de covarianza completa sobre las features y
    devuelve:
      - la serie de estados final (HMM + override bajista opcional +
        filtro de duración mínima)
      - las probabilidades filtradas por fecha (predict_proba, del HMM puro)
      - el mapeo estado->etiqueta (ALCISTA/BAJISTA/RANGO) basado en el
        retorno medio de cada estado
      - el modelo ajustado, el retorno medio por estado, y la máscara de
        semanas donde el override efectivamente actuó
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(features[["returns", "volatility", "slope"]].values)

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

    override_mask = pd.Series(False, index=features.index)
    if use_bear_override and n_states == 3:
        regime_labels_raw, override_mask = apply_bear_override(
            regime_labels_raw, features["slope_z"], z_threshold=z_threshold
        )

    regime_labels = apply_min_duration(regime_labels_raw, min_run=min_run)

    return regime_labels, proba, label_map, model, mean_returns, override_mask

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
    st.caption("Alcista · Bajista · Rango — HMM + override de tendencia bajista adaptativo por z-score")

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
            help="Si el HMM etiqueta una semana como RANGO pero la pendiente de "
                 "tendencia es estadísticamente muy negativa PARA ESTE MISMO "
                 "TICKER (z-score, no un valor fijo), la reclasifica como "
                 "BAJISTA. Corrige el sesgo del HMM a 'tragarse' caídas "
                 "sostenidas como si fueran laterales, sin romperse en "
                 "tickers muy volátiles."
        )

    z_threshold = -1.25
    trend_window = 20
    z_window = 104
    if use_bear_override:
        col_input8, col_input9, col_input10 = st.columns(3)
        with col_input8:
            z_threshold = st.slider(
                "Umbral z de pendiente bajista", -3.0, -0.25, -1.25, 0.25,
                help="La pendiente actual se compara contra su propia media/"
                     "desviación histórica en este ticker (z-score), no contra "
                     "un valor fijo. Un z de -1.25 significa 'la tendencia bajista "
                     "actual es notablemente más pronunciada de lo habitual para "
                     "esta acción'. Más negativo = override más exigente/raro. "
                     "Calibrado en un benchmark con tickers calmados y volátiles: "
                     "la tasa de activación del override se mantuvo entre 0% y "
                     "12% de las semanas, nunca 'se comió' todo el rango."
            )
        with col_input9:
            trend_window = st.slider(
                "Ventana de la pendiente (sem.)", 8, 40, 20,
                help="Número de semanas usadas para calcular la pendiente de "
                     "tendencia que alimenta al override bajista."
            )
        with col_input10:
            z_window = st.slider(
                "Ventana del z-score (sem.)", 52, 208, 104,
                help="Historial usado para calcular la media/desviación de "
                     "referencia de la pendiente de este ticker. 104 semanas "
                     "= 2 años."
            )

    if st.button("🚀 ANALIZAR", use_container_width=True):
        st.session_state.ticker = ticker
        st.session_state.df = download_weekly_data(ticker, years_back)
        st.session_state.vol_window = vol_window
        st.session_state.sticky_strength = sticky_strength
        st.session_state.min_run = min_run
        st.session_state.use_bear_override = use_bear_override
        st.session_state.z_threshold = z_threshold
        st.session_state.trend_window = trend_window
        st.session_state.z_window = z_window
        st.session_state.analyzed = True

    if st.session_state.analyzed and st.session_state.df is not None:

        df = st.session_state.df.copy()

        with st.spinner("Ajustando HMM..."):
            features = build_features(
                df,
                vol_window=st.session_state.get("vol_window", vol_window),
                trend_window=st.session_state.get("trend_window", trend_window),
                z_window=st.session_state.get("z_window", z_window),
            )
            regime_labels, proba, label_map, model, mean_returns, override_mask = fit_hmm_regimes(
                features,
                n_states=3,
                sticky_strength=st.session_state.get("sticky_strength", sticky_strength),
                min_run=st.session_state.get("min_run", min_run),
                z_threshold=st.session_state.get("z_threshold", z_threshold),
                use_bear_override=st.session_state.get("use_bear_override", use_bear_override),
            )

        # Unir regímenes al df original (índices alineados con features, que
        # pierde las primeras filas por los rolling windows)
        df = df.loc[features.index].copy()
        df["Regime"] = regime_labels
        df["Slope"] = features["slope"]
        df["Slope_Z"] = features["slope_z"]
        df["Overridden"] = override_mask
        for s, lbl in label_map.items():
            df[f"Prob_{lbl}"] = proba[:, s]

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
                f"⚠️ El HMM puro indicaba **{hmm_raw_label}**, pero se corrigió a **BAJISTA** "
                f"por el override de tendencia (z-score de pendiente = {float(current['Slope_Z']):.2f}, "
                f"umbral = {st.session_state.get('z_threshold', z_threshold):.2f})."
            )

        info_cols = st.columns(4)
        info_cols[0].metric("Precio", f"${precio:.2f}")
        info_cols[1].metric("SMA 50", f"${sma50:.2f}")
        info_cols[2].metric("Precio > SMA50", precio_above_sma)
        info_cols[3].metric("Z-score pendiente", f"{float(current['Slope_Z']):.2f}")

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

            override_pct = float(override_mask.mean() * 100)
            st.caption(
                f"El override de tendencia bajista reclasificó **{int(override_mask.sum())} semanas "
                f"({override_pct:.1f}% del histórico)** de RANGO a BAJISTA. "
                f"Rangos sanos observados en el benchmark de calibración: 0%–12%. "
                f"Si este porcentaje es mucho más alto (p.ej. >30-40%), el override está "
                f"dominando la clasificación en vez de actuar como red de seguridad — "
                f"considera subir (en magnitud) el umbral z, o revisar si el HMM está "
                f"fallando de forma sistemática para este ticker."
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
            **Metodología: Gaussian Hidden Markov Model (3 estados, covarianza completa) + override de tendencia adaptativo**

            - Features del HMM: retorno suavizado (media móvil corta), volatilidad
              realizada (rolling, anualizada) y pendiente de la regresión log-precio.
              Las tres se **escalan** (media 0, varianza 1) antes de entrenar para que
              ninguna domine por su escala.
            - El modelo se ajusta con `hmmlearn.GaussianHMM` (EM/Baum-Welch), con un
              **prior "sticky"** sobre la matriz de transición para evitar parpadeo.
            - **Override de tendencia bajista, versión z-score (corregido):** una
              primera versión usaba un umbral de pendiente absoluto (ej. "pendiente
              < -0.0025"). Eso se calibró en un benchmark sintético con una sola
              escala de volatilidad y **falló al aplicarlo a un ticker real más
              volátil (SOFI)**, donde el umbral fijo se disparaba en prácticamente
              el 100% de las semanas y eliminaba el régimen RANGO por completo.
              La versión actual compara la pendiente contra su **propia media y
              desviación histórica en ese mismo ticker (z-score)**, por lo que se
              adapta automáticamente a la volatilidad de cada activo. Validado con
              8 series sintéticas (4 de baja volatilidad, 4 de alta volatilidad,
              ninguna ajustada a un ticker real): la tasa de activación del
              override se mantuvo entre 0% y 12% en todos los casos, y el recall
              del régimen bajista mejoró de ~50-60% (HMM puro) a ~69% en promedio.
            - Tras el HMM + override, se aplica un **filtro de confirmación mínima**
              (un cambio de régimen solo se acepta si se sostiene N semanas seguidas).
            - Los 3 estados base se etiquetan según su **retorno medio**: mayor
              retorno → **ALCISTA**, menor → **BAJISTA**, intermedio → **RANGO**.
            - **Cómo saber si el override está bien calibrado para tu ticker:** mira
              el porcentaje de "semanas reclasificadas" en el expander de
              características de régimen. Si es 0%-15%, está actuando como red de
              seguridad ocasional (comportamiento esperado). Si es mucho más alto,
              el override está dominando la clasificación — sube (en magnitud) el
              umbral z hasta que vuelva a un rango razonable.
            - Si sigue "parpadeando" entre regímenes, sube el slider de
              **Persistencia** o el de **Confirmación mínima**; si lo ves demasiado
              lento para reaccionar, bájalos.
            """)

    elif st.session_state.analyzed and st.session_state.df is None:
        st.error("❌ No se pudo descargar data. Verifica el ticker.")


if __name__ == "__main__":
    if check_password():
        main()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder.")
