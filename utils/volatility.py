"""
volatility.py
=============
Cálculo del rango esperado semanal del SPX — fijo a 5 DTE (Lunes → Viernes),
basado en el régimen de VIX actual. Replica la lógica del notebook
Vol_SPX_5DTE_2_0_prep.ipynb (Manuel Izquierdo).

Metodología
-----------
- Datos SEMANALES de SPX (^GSPC), VIX (^VIX) y VIX3M (^VIX3M) vía yfinance,
  con sesión curl_cffi (impersonate Chrome) para evitar bloqueos.
- Segmentación histórica en 4 regímenes de VIX: 10-15, 15-18, 18-20, 20-25.
- Para cada régimen se calcula el std de los log-returns semanales (Std_logret).
- El régimen activo se determina con el VIX de la semana YA CERRADA (shift),
  sin look-ahead: se corre asumiendo que es domingo/lunes antes de la apertura.
- Banda = Last_Close * (1 ± stdn * Std_logret_del_regimen)

Uso desde la página principal
------------------------------
from utils.volatility import calcular_rango_esperado

resultado = calcular_rango_esperado(stdn=2.5)

resultado.last_close      # último cierre semanal (viernes)
resultado.band_dw / band_up
resultado.regime_label
resultado.hist_df         # histórico semanal completo, para graficar

Autor  : Manuel Izquierdo (notebook)
Adaptado a Streamlit — v2.0
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

TICKER_SPX   = "^GSPC"
TICKER_VIX   = "^VIX"
TICKER_VIX3M = "^VIX3M"


# ==============================================================================
# WMA manual (equivalente a ta.trend.WMAIndicator, sin dependencia de 'ta')
# ==============================================================================
def _wma(series: pd.Series, window: int) -> pd.Series:
    weights = np.arange(1, window + 1, dtype=float)
    return series.rolling(window).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


# ==============================================================================
# RESULTADO
# ==============================================================================
@dataclass
class RangoEsperadoResult:
    last_date:      str
    last_close:     float
    last_open:      float
    current_vix:    float              # VIX de la última semana cerrada
    regime_label:   Optional[str]
    regime_rows:    int
    sigma_regimen:  float              # Std_logret del régimen (semanal)
    stdn:           float
    band_dw:        float
    band_up:        float
    move_pct:       float              # amplitud total de la banda, en %
    trend:          str                # "Alcista" / "Bajista"
    term_structure: str                # "Contango" / "Backwardation"
    stats_df:       pd.DataFrame       # tabla resumen de todos los regímenes VIX
    hist_df:        pd.DataFrame       # histórico semanal completo (para graficar)


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
# SEGMENTACION POR RANGO DE VIX
# ==============================================================================
def _segmentar_por_vix(df: pd.DataFrame) -> pd.DataFrame:
    vix_stats = {}
    for low, high in VIX_RANGES:
        label  = f"{low}-{high}"
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
                "N_semanas": 0, "Mean_logret": np.nan, "Std_logret": np.nan,
                "P5_logret": np.nan, "P95_logret": np.nan,
            }

    stats_df = pd.DataFrame(vix_stats).T
    stats_df.index.name = "VIX_Range"
    return stats_df


# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================
def calcular_rango_esperado(stdn: float = 2.5, years_back: int = YEARS_BACK) -> RangoEsperadoResult:
    """
    Función principal. Descarga el histórico semanal completo, segmenta por
    régimen de VIX y devuelve el rango esperado del SPX para la semana
    próxima (5 DTE, Lunes → Viernes), en base al régimen de VIX vigente.

    Parámetros
    ----------
    stdn       : Desviaciones estándar para las bandas (default 2.5)
    years_back : Años de histórico a descargar (default 15)

    Retorna
    -------
    RangoEsperadoResult
    """
    df = _descargar_y_procesar(years_back)
    stats_df = _segmentar_por_vix(df)

    last_row    = df.iloc[-1]
    current_vix = float(last_row["VIX_y"])   # VIX de la última semana cerrada (sin look-ahead)

    current_label = None
    for low, high in VIX_RANGES:
        if low <= current_vix < high:
            current_label = f"{low}-{high}"
            break

    last_close = float(last_row["Close"])

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

    band_dw  = round(last_close * (1 - stdn * sigma_regimen), 2)
    band_up  = round(last_close * (1 + stdn * sigma_regimen), 2)
    move_pct = round((band_up - band_dw) / last_close * 100, 2)

    return RangoEsperadoResult(
        last_date      = df.index[-1].strftime("%Y-%m-%d"),
        last_close     = last_close,
        last_open      = float(last_row["Open"]),
        current_vix    = current_vix,
        regime_label   = current_label,
        regime_rows    = regime_rows,
        sigma_regimen  = round(sigma_regimen, 6),
        stdn           = stdn,
        band_dw        = band_dw,
        band_up        = band_up,
        move_pct       = move_pct,
        trend          = str(last_row["TREND"]),
        term_structure = str(last_row["Term_Structure"]),
        stats_df       = stats_df,
        hist_df        = df,
    )
