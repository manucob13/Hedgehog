"""
volatility.py
=============
Calcula volatilidad histórica segmentada por régimen de VIX (Garman-Klass)
y devuelve bandas de precio para los tres tranches semanales del IC.

Uso desde la página principal
------------------------------
from utils.volatility import calcular_bandas_ic

resultado = calcular_bandas_ic(
    current_vix = 16.5,   # VIX actual obtenido en la página
    current_spx = 5580.0, # Precio SPX actual obtenido en la página
    stdn        = 2.3,    # Desviaciones estándar (ajustable)
)

# resultado es un dict con las tres tranches:
# resultado["T1"]  → lunes→viernes   (5 DTE)
# resultado["T2"]  → martes→viernes  (4 DTE)
# resultado["T3"]  → miércoles→viernes (3 DTE)
# resultado["meta"] → info del régimen, vol diaria, señal, etc.

Autor  : Manuel Izquierdo
Versión: 1.0
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# REGÍMENES DE VIX — mismos rangos que en tu notebook
# ---------------------------------------------------------------------------
VIX_REGIMES = [
    {"label": "VIX 10-15", "lo": 10, "hi": 15},
    {"label": "VIX 15-18", "lo": 15, "hi": 18},
    {"label": "VIX 18-20", "lo": 18, "hi": 20},
    {"label": "VIX 20-25", "lo": 20, "hi": 25},
]


# ---------------------------------------------------------------------------
# DATACLASS — resultado por tranche
# ---------------------------------------------------------------------------
@dataclass
class TrancheResult:
    label:          str    # "T1 - Lunes→Viernes"
    dte:            int    # días al vencimiento
    vol_daily:      float  # Avg_252_Vol21 del régimen (diaria)
    vol_scaled:     float  # vol_daily × √DTE
    band_dw:        float  # banda inferior absoluta (precio)
    band_up:        float  # banda superior absoluta (precio)
    move_pct:       float  # movimiento esperado en % del spot


@dataclass
class VolatilityResult:
    # Contexto del mercado
    current_vix:    float
    current_spx:    float
    stdn:           float
    regime_label:   str
    regime_lo:      float
    regime_hi:      float
    regime_rows:    int
    vol_daily:      float  # Avg_252_Vol21 del régimen
    
    # Señal de operación
    signal:         int    # 1 = operar, 0 = no operar
    trend:          str    # "Alcista" / "Bajista"
    vix_lt_25:      bool
    vix_lt_wma21:   bool
    wma21_falling:  bool
    last_date:      str

    # Tres tranches
    T1:             TrancheResult  # 5 DTE — lunes→viernes
    T2:             TrancheResult  # 4 DTE — martes→viernes
    T3:             TrancheResult  # 3 DTE — miércoles→viernes

    # Timestamp
    timestamp:      str


# ---------------------------------------------------------------------------
# WMA manual (sin dependencia de 'ta')
# ---------------------------------------------------------------------------
def _wma(series: pd.Series, window: int) -> pd.Series:
    weights = np.arange(1, window + 1, dtype=float)
    return series.rolling(window).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


# ---------------------------------------------------------------------------
# DESCARGA DE DATOS HISTÓRICOS
# ---------------------------------------------------------------------------
def _descargar_historico(
    ticker: str = "^GSPC",
    vix_ticker: str = "^VIX",
    start: str = "2010-01-01",
) -> pd.DataFrame:
    """
    Descarga OHLCV del ticker y VIX desde yfinance.
    Añade WMA30, WMA21 del VIX y desplazamientos de un día.
    """
    end = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    df_p = yf.download(ticker,     start=start, end=end,
                       auto_adjust=False, multi_level_index=False, progress=False)
    df_v = yf.download(vix_ticker, start=start, end=end,
                       auto_adjust=False, multi_level_index=False, progress=False)

    if df_p.empty or df_v.empty:
        raise ValueError(
            f"No se pudieron descargar datos de {ticker} o {vix_ticker}. "
            "Verifica la conexión a internet."
        )

    df_p.index = pd.to_datetime(df_p.index)
    df_v.index = pd.to_datetime(df_v.index)

    vix_close  = df_v["Close"].rename("VIX")
    vix_wma21  = _wma(vix_close, 21).rename("VIX_WMA_21")
    spx_wma30  = _wma(df_p["Close"], 30).rename("SP500_WMA_30")

    df = pd.DataFrame({
        "Open":         df_p["Open"],
        "High":         df_p["High"],
        "Low":          df_p["Low"],
        "Close":        df_p["Close"],
        "SP500_WMA_30": spx_wma30,
        "VIX":          vix_close,
        "VIX_WMA_21":   vix_wma21,
    }).dropna()

    # Valores de "ayer" — necesarios para la señal de operación
    df["Close_y"]        = df["Close"].shift(1)
    df["SP500_WMA_30_y"] = df["SP500_WMA_30"].shift(1)
    df["VIX_C_y"]        = df["VIX"].shift(1)
    df["VIX_WMA_21_y"]   = df["VIX_WMA_21"].shift(1)
    df["VIX_WMA_21_2dy"] = df["VIX_WMA_21"].shift(2)
    df["TREND"]          = np.where(
        df["Close_y"] > df["SP500_WMA_30_y"], "Alcista", "Bajista"
    )

    return df.dropna()


# ---------------------------------------------------------------------------
# GARMAN-KLASS sobre un subconjunto
# ---------------------------------------------------------------------------
def _garman_klass(df: pd.DataFrame) -> pd.DataFrame:
    """
    GK_daily    = √[ 0.5·ln(H/L)² − (2ln2−1)·ln(C/O)² ]
    Vol_21      = rolling mean 21d de GK_daily
    Avg_252_Vol21 = rolling mean 252d de Vol_21
    """
    d = df.copy()
    d["GK_daily"]       = np.sqrt(
        0.5  * np.log(d["High"] / d["Low"]) ** 2
        - (2 * np.log(2) - 1) * np.log(d["Close"] / d["Open"]) ** 2
    )
    d["Vol_21"]         = d["GK_daily"].rolling(21).mean()
    d["Avg_252_Vol21"]  = d["Vol_21"].rolling(252).mean()
    return d


# ---------------------------------------------------------------------------
# FUNCIÓN PRINCIPAL
# ---------------------------------------------------------------------------
def calcular_bandas_ic(
    current_vix: float,
    current_spx: float,
    stdn:        float = 2.3,
    ticker:      str   = "^GSPC",
    vix_ticker:  str   = "^VIX",
    start:       str   = "2010-01-01",
) -> VolatilityResult:
    """
    Función principal. Recibe el VIX y SPX actuales desde la página,
    calcula toda la volatilidad histórica por régimen en memoria y
    devuelve las bandas para los tres tranches del IC semanal.

    Parámetros
    ----------
    current_vix : VIX actual (lo pasa la página principal)
    current_spx : Precio SPX actual (lo pasa la página principal)
    stdn        : Desviaciones estándar para las bandas (default 2.3)
    ticker      : Ticker del subyacente (default '^GSPC')
    vix_ticker  : Ticker del VIX (default '^VIX')
    start       : Inicio del histórico (default '2010-01-01')

    Retorna
    -------
    VolatilityResult con T1 (5 DTE), T2 (4 DTE), T3 (3 DTE) y metadata
    """

    # 1. Descargar histórico completo
    df_raw = _descargar_historico(ticker, vix_ticker, start)

    # 2. Señal de operación — basada en el último día del histórico
    last         = df_raw.iloc[-1]
    trend        = "Alcista" if last["Close_y"] > last["SP500_WMA_30_y"] else "Bajista"
    vix_lt_25    = bool(last["VIX_C_y"] <= 25)
    vix_lt_wma21 = bool(last["VIX_C_y"] < last["VIX_WMA_21_y"])
    wma21_fall   = bool(last["VIX_WMA_21_y"] < last["VIX_WMA_21_2dy"])
    signal       = 1 if (trend == "Alcista" and vix_lt_25 and vix_lt_wma21 and wma21_fall) else 0

    # 3. Seleccionar régimen según el current_vix que llega de la página
    matched = None
    for r in VIX_REGIMES:
        if r["lo"] <= current_vix < r["hi"]:
            matched = r
            break

    # Fallback: régimen más cercano por diferencia de punto medio
    if matched is None:
        def dist(r):
            mid = (r["lo"] + r["hi"]) / 2
            return abs(current_vix - mid)
        matched = min(VIX_REGIMES, key=dist)

    # 4. Filtrar histórico por régimen y calcular GK
    df_regime = df_raw[
        (df_raw["VIX"] >= matched["lo"]) &
        (df_raw["VIX"] <  matched["hi"])
    ].copy()

    if len(df_regime) < 30:
        raise ValueError(
            f"El régimen {matched['label']} solo tiene {len(df_regime)} filas. "
            "Insuficiente para calcular Avg_252_Vol21."
        )

    df_gk = _garman_klass(df_regime)
    df_valid = df_gk.dropna(subset=["Avg_252_Vol21"])

    if df_valid.empty:
        raise ValueError(
            f"No hay suficiente historia en el régimen {matched['label']} "
            "para calcular Avg_252_Vol21 (necesita 252 + 21 días mínimo)."
        )

    vol_daily = float(df_valid["Avg_252_Vol21"].iloc[-1])

    # 5. Calcular los tres tranches
    #    vol_scaled = vol_diaria × √DTE  (regla de la raíz del tiempo)
    #    band_dw    = spx × (1 − stdn × vol_scaled)
    #    band_up    = spx × (1 + stdn × vol_scaled)

    tranches_def = [
        ("T1", "Lunes → Viernes",     5),
        ("T2", "Martes → Viernes",    4),
        ("T3", "Miércoles → Viernes", 3),
    ]

    tranches = {}
    for key, label, dte in tranches_def:
        vol_scaled = vol_daily * np.sqrt(dte)
        band_dw    = round(current_spx * (1 - stdn * vol_scaled), 2)
        band_up    = round(current_spx * (1 + stdn * vol_scaled), 2)
        move_pct   = round(stdn * vol_scaled * 100, 2)
        tranches[key] = TrancheResult(
            label      = f"{key} – {label}",
            dte        = dte,
            vol_daily  = round(vol_daily, 6),
            vol_scaled = round(vol_scaled, 6),
            band_dw    = band_dw,
            band_up    = band_up,
            move_pct   = move_pct,
        )

    return VolatilityResult(
        current_vix   = current_vix,
        current_spx   = current_spx,
        stdn          = stdn,
        regime_label  = matched["label"],
        regime_lo     = matched["lo"],
        regime_hi     = matched["hi"],
        regime_rows   = len(df_regime),
        vol_daily     = round(vol_daily, 6),
        signal        = signal,
        trend         = trend,
        vix_lt_25     = vix_lt_25,
        vix_lt_wma21  = vix_lt_wma21,
        wma21_falling = wma21_fall,
        last_date     = df_raw.index[-1].strftime("%Y-%m-%d"),
        T1            = tranches["T1"],
        T2            = tranches["T2"],
        T3            = tranches["T3"],
        timestamp     = datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
