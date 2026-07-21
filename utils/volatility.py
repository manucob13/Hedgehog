"""
volatility.py
=============
Cálculo del rango esperado del SPX para Iron Condor semanal, con salida
siempre en Viernes y origen configurable (Lunes / Martes / Miercoles),
basado en el régimen de VIX actual.

Metodología
-----------
- Datos SEMANALES de SPX (^GSPC), VIX (^VIX) y VIX3M (^VIX3M) vía yfinance,
  con sesión curl_cffi (impersonate Chrome) para evitar bloqueos.
- Segmentación histórica en 4 regímenes de VIX: 10-15, 15-18, 18-20, 20-25.
- Para cada régimen se calcula el std de los log-returns SEMANALES
  (Lunes -> Viernes, base 5 días hábiles).
- Para orígenes distintos de Lunes (Martes, Miercoles), ese sigma semanal
  se escala por sqrt(dias_habiles / 5) — Opción A: aproximación por raíz
  del tiempo, sin reconstruir el histórico a granularidad diaria.
- El precio spot (ancla de las bandas) SIEMPRE se toma del último cierre
  DIARIO disponible (no del cierre semanal), para que coincida con el día
  de entrada real:
    * Origen Lunes     -> último cierre diario = Viernes anterior
    * Origen Martes    -> último cierre diario = Lunes de esa semana
    * Origen Miercoles -> último cierre diario = Martes de esa semana
  Esto asume que la herramienta se corre el día hábil antes de la entrada
  (p.ej. la noche del lunes o la mañana del martes antes de abrir, para
  preparar la entrada del martes).
- El régimen activo (para elegir el sigma) se determina con el VIX de la
  semana YA CERRADA (shift), igual que antes — no cambia con el origen.
- Banda = Spot_diario * (1 ± stdn * sigma_aplicado)

Uso desde la página principal
------------------------------
from utils.volatility import calcular_rango_esperado, DIAS_HABILES_POR_ORIGEN

resultado = calcular_rango_esperado(stdn=1.0, origen="Martes")

resultado.last_close      # spot diario usado como ancla
resultado.spot_date       # fecha real de ese cierre diario
resultado.band_dw / band_up
resultado.sigma_regimen   # sigma semanal base (Lunes->Viernes), sin escalar
resultado.sigma_aplicado  # sigma ya escalado según origen/dias_habiles
resultado.origen
resultado.dias_habiles
resultado.regime_label
resultado.hist_df         # histórico semanal completo, para graficar

Autor  : Manuel Izquierdo (notebook)
Adaptado a Streamlit — v2.1 (origen configurable + spot diario)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from curl_cffi import requests as curl_requests
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ==============================================================================
# CONFIGURACION
# ==============================================================================
YEARS_BACK = 15

WMA_SP500_WEEKS       = 5    # WMA del SP500 (equiv. 30d diarios)
VOL_WINDOW_WEEKS      = 4    # ventana volatilidad (equiv. 21d diarios)
VOL_MEAN_WINDOW_WEEKS = 50   # media movil de la volatilidad (equiv. 252d diarios)
VIX_WMA_WEEKS         = 52   # WMA del VIX (equiv. 252d aprox)

VIX_RANGES = [
    (10, 15),
    (15, 18),
    (18, 20),
    (20, 25),
]

# Días hábiles restantes hasta el viernes, según el día de entrada.
# Se usa tanto para escalar sigma (sqrt(dias_habiles/5)) como para mostrar
# el DTE en la interfaz.
DIAS_HABILES_POR_ORIGEN = {
    "Lunes":     5,
    "Martes":    4,
    "Miercoles": 3,
}

# Offset en días de calendario desde el Lunes de referencia, para calcular
# la fecha real de entrada según el origen elegido.
OFFSET_DIAS_POR_ORIGEN = {
    "Lunes":     0,
    "Martes":    1,
    "Miercoles": 2,
}

TICKER_SPX   = "^GSPC"
TICKER_VIX   = "^VIX"
TICKER_VIX3M = "^VIX3M"


# ==============================================================================
# HELPERS
# ==============================================================================
def _wma(series: pd.Series, window: int) -> pd.Series:
    weights = np.arange(1, window + 1, dtype=float)
    return series.rolling(window).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def _week_ending_friday(ts: pd.Timestamp) -> pd.Timestamp:
    """
    Normaliza cualquier timestamp semanal a su viernes de cierre.
    Si ya es viernes, lo deja igual; si no, lo lleva al próximo viernes.
    """
    ts = pd.Timestamp(ts)
    if ts.weekday() == 4:
        return ts.normalize()
    return (ts + pd.offsets.Week(weekday=4)).normalize()


def _obtener_ultimo_cierre_diario() -> tuple:
    """
    Descarga los últimos días de datos DIARIOS del SPX y devuelve el cierre
    más reciente disponible junto con su fecha.

    Se usa como precio spot para las bandas, en vez del cierre semanal (que
    solo se actualiza los viernes), para que el spot coincida con el día
    de entrada real sea cual sea el origen elegido (Lunes/Martes/Miercoles).
    """
    session = curl_requests.Session(impersonate="chrome")
    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=10)

    df_daily = yf.download(
        TICKER_SPX,
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d"),
        auto_adjust=False,
        multi_level_index=False,
        session=session,
        interval="1d",
        progress=False,
    )

    if df_daily.empty:
        raise ValueError(
            "No se pudo descargar el cierre diario reciente del SPX."
        )

    ultimo_cierre = float(df_daily["Close"].iloc[-1])
    ultima_fecha  = pd.Timestamp(df_daily.index[-1])
    return ultimo_cierre, ultima_fecha


# ==============================================================================
# RESULTADO
# ==============================================================================
@dataclass
class SenalEntrada:
    """
    Señal binaria de entrada para el Iron Condor, calculada con los valores
    YA CONOCIDOS de la última semana cerrada (sin shift, porque nos paramos
    justo después del cierre del viernes, antes de la apertura del lunes).
    Replica el bloque "PREDICCION DEL MERCADO" del notebook.

    Nota: por ahora la señal se calcula una única vez por semana (con el
    cierre del viernes anterior) y se reutiliza para los tres orígenes
    (Lunes/Martes/Miercoles) — no se recalcula con datos más frescos para
    Martes/Miercoles. Es una simplificación consciente, no un descuido.
    """
    new_date:               str     # próximo día hábil (lunes) al que aplica la señal
    signal:                 int     # 1 = abrir IC, 0 = no operar
    last_close:             float
    last_sp500_wma30:       float
    tendencia:              str     # "Alcista" / "Bajista"
    last_vix:               float
    last_vix_wma21:         float
    vix_en_rango:           bool    # VIX dentro de algún rango definido (10-25)
    vix_lt_wma21:           bool    # VIX < VIX_WMA_21 (relajándose)
    term_structure:         str     # "Contango" / "Backwardation" (sin shift)
    vix_wma21_bajando:      bool    # informativo, no entra en la señal
    realized_vol_anual_pct: float   # informativo
    vrp_positive:           bool    # informativo: VIX > vol realizada
    cond_tendencia:         bool
    cond_vix_rango:         bool
    cond_vix_wma:           bool
    cond_contango:          bool


@dataclass
class RangoEsperadoResult:
    last_date:      str
    last_close:     float           # spot DIARIO usado como ancla de las bandas
    spot_date:      str             # fecha real de ese cierre diario
    last_open:      float
    current_vix:    float              # VIX de la última semana cerrada (shifted, sin look-ahead)
    regime_label:   Optional[str]
    regime_rows:    int
    sigma_regimen:  float              # Std_logret semanal del régimen, SIN escalar (base Lunes->Viernes)
    sigma_aplicado: float              # sigma ya escalado según origen/dias_habiles, usado en las bandas
    stdn:           float
    origen:         str                # "Lunes" / "Martes" / "Miercoles"
    dias_habiles:   int                # días hábiles hasta el viernes, según origen
    band_dw:        float
    band_up:        float
    move_pct:       float              # amplitud total de la banda, en %
    trend:          str                # "Alcista" / "Bajista"
    term_structure: str                # "Contango" / "Backwardation"
    stats_df:       pd.DataFrame       # tabla resumen de todos los regímenes VIX
    hist_df:        pd.DataFrame       # histórico semanal completo (para graficar)
    senal:          SenalEntrada       # señal de entrada del Iron Condor


# ==============================================================================
# DESCARGA + PROCESAMIENTO SEMANAL COMPLETO
# ==============================================================================
def _descargar_y_procesar(years_back: int = YEARS_BACK) -> pd.DataFrame:
    session = curl_requests.Session(impersonate="chrome")

    end_dt     = datetime.today()
    start_date = (end_dt - timedelta(days=365 * years_back)).strftime("%Y-%m-%d")
    end_date   = end_dt.strftime("%Y-%m-%d")

    df_spy = yf.download(
        TICKER_SPX, start=start_date, end=end_date, auto_adjust=False,
        multi_level_index=False, session=session, interval="1wk", progress=False,
    )
    df_vix = yf.download(
        TICKER_VIX, start=start_date, end=end_date, auto_adjust=False,
        multi_level_index=False, session=session, interval="1wk", progress=False,
    )
    df_vix3m = yf.download(
        TICKER_VIX3M, start=start_date, end=end_date, auto_adjust=False,
        multi_level_index=False, session=session, interval="1wk", progress=False,
    )

    if df_spy.empty or df_vix.empty or df_vix3m.empty:
        raise ValueError(
            "No se pudieron descargar datos semanales de SPX/VIX/VIX3M. "
            "Verificá la conexión a internet."
        )

    df_spy["WMA_30"]          = _wma(df_spy["Close"], WMA_SP500_WEEKS)
    df_spy["log_return"]      = np.log(df_spy["Close"] / df_spy["Close"].shift(1))
    df_spy["vol_21"]          = df_spy["log_return"].rolling(window=VOL_WINDOW_WEEKS).std()
    df_spy["mean_vol_21_252"] = df_spy["vol_21"].rolling(window=VOL_MEAN_WINDOW_WEEKS).mean()

    df2 = pd.DataFrame({
        "Open":          df_spy["Open"],
        "Close":         df_spy["Close"],
        "log_return":    df_spy["log_return"],
        "Vol21":         df_spy["vol_21"],
        "Avg_252_Vol21": df_spy["mean_vol_21_252"],
        "SP500_WMA_30":  df_spy["WMA_30"],
    })

    df_v = df_vix[["Close"]].rename(columns={"Close": "VIX"})
    df_v["VIX_WMA_21"] = _wma(df_v["VIX"], VIX_WMA_WEEKS)

    df_v3 = df_vix3m[["Close"]].rename(columns={"Close": "VIX3M"})

    df = df2.join(df_v, how="left").join(df_v3, how="left")
    df = df.dropna()

    # --- SHIFT: solo usamos lo conocido ANTES de que empiece la semana a operar ---
    df["Close_y"]         = df["Close"].shift(1)
    df["Avg_252_Vol21_y"] = df["Avg_252_Vol21"].shift(1)
    df["SP500_WMA_30_y"]  = df["SP500_WMA_30"].shift(1)
    df["VIX_y"]           = df["VIX"].shift(1)
    df["VIX_WMA_21_y"]    = df["VIX_WMA_21"].shift(1)
    df["VIX_WMA_21_2y"]   = df["VIX_WMA_21"].shift(2)
    df["VIX3M_y"]         = df["VIX3M"].shift(1)

    df = df.dropna()

    df["TREND"] = np.where(df["Close_y"] > df["SP500_WMA_30_y"], "Alcista", "Bajista")

    df["VIX_WMA_DOWN"] = (
        (df["VIX_y"] < df["VIX_WMA_21_y"]) &
        (df["VIX_WMA_21_y"] < df["VIX_WMA_21_2y"])
    )

    df["VIX_VIX3M_Ratio"] = df["VIX_y"] / df["VIX3M_y"]
    df["Term_Structure"]  = np.where(df["VIX_VIX3M_Ratio"] < 1, "Contango", "Backwardation")

    return df


# ==============================================================================
# SEÑAL DE ENTRADA (bloque "PREDICCION DEL MERCADO" del notebook)
# ==============================================================================
def _calcular_senal(df: pd.DataFrame) -> SenalEntrada:
    """
    Calcula la señal binaria de apertura del Iron Condor con los valores
    YA CONOCIDOS de la última semana cerrada (sin shift): estamos parados
    justo después del cierre del viernes, antes de que abra el mercado el
    lunes, así que 'VIX', 'VIX3M', 'Close', etc. de la última fila son
    datos reales, no proyectados.

    Condiciones que forman la señal (las 4 deben cumplirse para Signal = 1):
      1. Tendencia = Alcista       -> Close > SP500_WMA_30
      2. VIX dentro de rango 10-25 -> ni muy bajo ni muy alto
      3. VIX < VIX_WMA_21          -> volatilidad implícita relajándose
      4. Term Structure = Contango -> VIX < VIX3M, mercado calmado

    Informativas (no entran en la señal): VIX_WMA_21 bajando 2 semanas
    consecutivas, y VRP (VIX > vol. realizada anualizada).
    """
    last_row = df.iloc[-1]

    # Normalizamos la fecha semanal al viernes real de cierre
    last_date = _week_ending_friday(df.index[-1])

    # Próximo lunes aplicable a la señal
    next_business_day = last_date + timedelta(days=3)  # viernes -> lunes
    while next_business_day.weekday() >= 5:
        next_business_day += timedelta(days=1)

    # --- Contango/backwardation actual (VIX y VIX3M de la última semana cerrada, sin shift) ---
    current_ratio = last_row["VIX"] / last_row["VIX3M"]
    current_term_structure = "Contango" if current_ratio < 1 else "Backwardation"

    # --- VIX dentro de alguno de los rangos definidos ---
    current_vix = last_row["VIX"]
    vix_in_range = any(low <= current_vix < high for low, high in VIX_RANGES)

    # --- VIX WMA bajando 2 semanas consecutivas (solo informativo) ---
    vix_wma_down = bool(last_row["VIX_WMA_21"] < last_row["VIX_WMA_21_y"])

    # --- VRP: VIX (implícita) vs volatilidad realizada anualizada (solo informativo) ---
    realized_vol_annualized = float(last_row["Vol21"] * np.sqrt(52) * 100)
    vrp_positive = bool(current_vix > realized_vol_annualized)

    # --- Condiciones individuales que forman la señal ---
    cond_tendencia = bool(last_row["Close"] > last_row["SP500_WMA_30"])
    cond_vix_rango = bool(vix_in_range)
    cond_vix_wma   = bool(last_row["VIX"] < last_row["VIX_WMA_21"])
    cond_contango  = bool(current_term_structure == "Contango")

    opera = 1 if (cond_tendencia and cond_vix_rango and cond_vix_wma and cond_contango) else 0

    return SenalEntrada(
        new_date               = next_business_day.strftime("%Y-%m-%d"),
        signal                 = opera,
        last_close             = round(float(last_row["Close"]), 2),
        last_sp500_wma30       = round(float(last_row["SP500_WMA_30"]), 2),
        tendencia              = "Alcista" if cond_tendencia else "Bajista",
        last_vix               = round(float(current_vix), 2),
        last_vix_wma21         = round(float(last_row["VIX_WMA_21"]), 2),
        vix_en_rango            = vix_in_range,
        vix_lt_wma21           = cond_vix_wma,
        term_structure         = current_term_structure,
        vix_wma21_bajando      = vix_wma_down,
        realized_vol_anual_pct = round(realized_vol_annualized, 2),
        vrp_positive           = vrp_positive,
        cond_tendencia         = cond_tendencia,
        cond_vix_rango         = cond_vix_rango,
        cond_vix_wma           = cond_vix_wma,
        cond_contango          = cond_contango,
    )


# ==============================================================================
# SEGMENTACION POR RANGO DE VIX
# ==============================================================================
def _segmentar_por_vix(df: pd.DataFrame) -> pd.DataFrame:
    vix_stats = {}
    for low, high in VIX_RANGES:
        label = f"{low}-{high}"
        subset = df[(df["VIX_y"] >= low) & (df["VIX_y"] < high)]

        if len(subset) > 0:
            vix_stats[label] = {
                "N_semanas":   len(subset),
                "Mean_logret": subset["log_return"].mean(),
                "Std_logret":  subset["log_return"].std(),
                "P5_logret":   subset["log_return"].quantile(0.05),
                "P95_logret":  subset["log_return"].quantile(0.95),
            }
        else:
            vix_stats[label] = {
                "N_semanas": 0,
                "Mean_logret": np.nan,
                "Std_logret": np.nan,
                "P5_logret": np.nan,
                "P95_logret": np.nan,
            }

    stats_df = pd.DataFrame(vix_stats).T
    stats_df.index.name = "VIX_Range"
    return stats_df


# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================
def calcular_rango_esperado(
    stdn: float = 2.5,
    years_back: int = YEARS_BACK,
    origen: str = "Lunes",
) -> RangoEsperadoResult:
    """
    Función principal. Descarga el histórico semanal completo, segmenta por
    régimen de VIX y devuelve el rango esperado del SPX hasta el viernes,
    en base al régimen de VIX vigente y al día de entrada (origen) elegido.

    Parámetros
    ----------
    stdn       : Desviaciones estándar para las bandas (default 2.5)
    years_back : Años de histórico a descargar (default 15)
    origen     : "Lunes", "Martes" o "Miercoles" — día de entrada. La salida
                 siempre es el viernes. Determina tanto el escalado de sigma
                 (sqrt(dias_habiles/5)) como, indirectamente junto al momento
                 en que se corre la herramienta, qué cierre diario se usa
                 como spot.

    Retorna
    -------
    RangoEsperadoResult
    """
    if origen not in DIAS_HABILES_POR_ORIGEN:
        raise ValueError(
            f"Origen '{origen}' no válido. Debe ser uno de: "
            f"{list(DIAS_HABILES_POR_ORIGEN.keys())}"
        )

    df = _descargar_y_procesar(years_back)
    stats_df = _segmentar_por_vix(df)
    senal = _calcular_senal(df)

    last_row = df.iloc[-1]
    current_vix = float(last_row["VIX_y"])   # VIX de la última semana cerrada (sin look-ahead)

    current_label = None
    for low, high in VIX_RANGES:
        if low <= current_vix < high:
            current_label = f"{low}-{high}"
            break

    if current_label is None:
        raise ValueError(
            f"El VIX actual ({current_vix:.2f}) está fuera de los rangos definidos "
            f"(10-25). No se puede estimar el rango esperado."
        )

    sigma_regimen = float(stats_df.loc[current_label, "Std_logret"])
    regime_rows   = int(stats_df.loc[current_label, "N_semanas"])

    if pd.isna(sigma_regimen) or regime_rows < 10:
        raise ValueError(
            f"El régimen {current_label} tiene muy pocas observaciones "
            f"({regime_rows}) para estimar un sigma confiable."
        )

    # --- Escalado del sigma según el origen (Opción A: sqrt del tiempo) ---
    dias_habiles = DIAS_HABILES_POR_ORIGEN[origen]
    factor_escala = (dias_habiles / 5) ** 0.5
    sigma_aplicado = sigma_regimen * factor_escala

    # --- Spot: SIEMPRE el último cierre DIARIO disponible, no el semanal ---
    # Así el spot coincide con el día de entrada real (ver docstring del módulo).
    # Si por lo que sea falla la descarga diaria, se cae al cierre semanal
    # como fallback (menos preciso para Martes/Miercoles, pero no rompe la app).
    try:
        last_close, fecha_spot = _obtener_ultimo_cierre_diario()
    except Exception:
        last_close = float(last_row["Close"])
        fecha_spot = _week_ending_friday(df.index[-1])

    band_dw  = round(last_close * (1 - stdn * sigma_aplicado), 2)
    band_up  = round(last_close * (1 + stdn * sigma_aplicado), 2)
    move_pct = round((band_up - band_dw) / last_close * 100, 2)

    last_week_close_date = _week_ending_friday(df.index[-1])

    return RangoEsperadoResult(
        last_date      = last_week_close_date.strftime("%Y-%m-%d"),
        last_close     = last_close,
        spot_date      = fecha_spot.strftime("%Y-%m-%d"),
        last_open      = float(last_row["Open"]),
        current_vix    = current_vix,
        regime_label   = current_label,
        regime_rows    = regime_rows,
        sigma_regimen  = round(sigma_regimen, 6),
        sigma_aplicado = round(sigma_aplicado, 6),
        stdn           = stdn,
        origen         = origen,
        dias_habiles   = dias_habiles,
        band_dw        = band_dw,
        band_up        = band_up,
        move_pct       = move_pct,
        trend          = str(last_row["TREND"]),
        term_structure = str(last_row["Term_Structure"]),
        stats_df       = stats_df,
        hist_df        = df,
        senal          = senal,
    )
