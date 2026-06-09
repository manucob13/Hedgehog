# utils/risk_profile.py
"""
Risk Profile — Calendar Put Spread SPX
Motor: Bjerksund-Stensland 2002 + IV individual fija + T_rem exacto.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ==============================================================================
# HELPERS MATEMÁTICOS
# ==============================================================================

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ==============================================================================
# BJERKSUND-STENSLAND 2002
# ==============================================================================

def _bs2002_phi(S: float, T: float, gamma: float, H: float,
                I: float, r: float, b: float, sigma: float) -> float:
    lam = (-r + gamma * b + 0.5 * gamma * (gamma - 1) * sigma ** 2) * T
    d   = -(math.log(S / H) + (b + (gamma - 0.5) * sigma ** 2) * T) / (sigma * math.sqrt(T))
    kap = 2 * b / sigma ** 2 + (2 * gamma - 1)
    return (
        math.exp(lam) * S ** gamma *
        (_norm_cdf(d) - (I / S) ** kap *
         _norm_cdf(d - 2 * math.log(I / S) / (sigma * math.sqrt(T))))
    )


def _bs2002_call(S: float, K: float, T: float, r: float, b: float, sigma: float) -> float:
    if T <= 1e-7:
        return max(S - K, 0.0)

    if b >= r:
        d1 = (math.log(S / K) + (b + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * math.exp((b - r) * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)

    beta  = (0.5 - b / sigma ** 2) + math.sqrt((b / sigma ** 2 - 0.5) ** 2 + 2 * r / sigma ** 2)
    B_inf = beta / (beta - 1) * K
    B0    = max(K, r / (r - b) * K)
    ht    = -(b * T + 2 * sigma * math.sqrt(T)) * B0 / (B_inf - B0)
    I     = B0 + (B_inf - B0) * (1 - math.exp(ht))

    if S >= I:
        return max(S - K, 0.0)

    alpha = (I - K) * I ** (-beta)

    return (
        alpha * S ** beta
        - alpha * _bs2002_phi(S, T, beta, I, I, r, b, sigma)
        + _bs2002_phi(S, T, 1, I, I, r, b, sigma)
        - _bs2002_phi(S, T, 1, K, I, r, b, sigma)
        - K * _bs2002_phi(S, T, 0, I, I, r, b, sigma)
        + K * _bs2002_phi(S, T, 0, K, I, r, b, sigma)
    )


def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str = "put", q: float = 0.0) -> float:
    """
    Precio usando BS2002 como engine base.
    Para put usamos paridad europea, suficiente para SPX y consistente con Theo.
    """
    if T <= 1e-7 or sigma <= 1e-6:
        return max(K - S, 0.0) if option_type.lower() == "put" else max(S - K, 0.0)

    b = r - q
    call = _bs2002_call(S, K, T, r, b, sigma)

    if option_type.lower() == "call":
        return call

    d1 = (math.log(S / K) + (b + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    eu_call = S * math.exp((b - r) * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    eu_put  = eu_call - S * math.exp((b - r) * T) + K * math.exp(-r * T)

    return max(eu_put, max(K - S, 0.0))


# ==============================================================================
# GRIEGOS — bump numérico sobre BS2002
# ==============================================================================

def bs_greeks(S: float, K: float, T: float, r: float, sigma: float,
              option_type: str = "put", q: float = 0.0) -> dict:
    if T <= 1e-7 or sigma <= 1e-6:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    h_s = S * 0.001
    h_v = 0.001
    h_t = 1 / 365

    p0   = bs_price(S,       K, T,                  r, sigma,       option_type, q)
    p_su = bs_price(S + h_s, K, T,                  r, sigma,       option_type, q)
    p_sd = bs_price(S - h_s, K, T,                  r, sigma,       option_type, q)
    p_vu = bs_price(S,       K, T,                  r, sigma + h_v, option_type, q)
    p_td = bs_price(S,       K, max(T - h_t, 1e-7), r, sigma,       option_type, q)

    return {
        "delta": round((p_su - p_sd) / (2 * h_s), 4),
        "gamma": round((p_su - 2 * p0 + p_sd) / (h_s ** 2), 6),
        "vega":  round((p_vu - p0) / h_v / 100, 4),
        "theta": round((p_td - p0) / h_t, 4),
    }


# ==============================================================================
# SKEW LOCAL — se mantiene por compatibilidad/UI, pero no se usa en el profile
# ==============================================================================

def interp_iv(S_query: float,
              skew_points: Optional[list],
              fallback_iv: float) -> float:
    if not skew_points or len(skew_points) < 2:
        return fallback_iv

    strikes = np.array([p[0] for p in skew_points], dtype=float)
    ivs     = np.array([p[1] for p in skew_points], dtype=float)

    if len(skew_points) == 2:
        iv = float(np.interp(S_query, strikes, ivs))
    else:
        try:
            coeffs = np.polyfit(strikes, ivs, 2)
            iv = float(np.polyval(coeffs, S_query))
        except Exception:
            iv = float(np.interp(S_query, strikes, ivs))

    return float(np.clip(iv, 0.01, 5.0))


# ==============================================================================
# TIEMPO EXACTO
# ==============================================================================

def exact_T(exp_date_str: str) -> float:
    """T en años desde ahora hasta expiración. Asume cierre a las 16:00."""
    now      = datetime.now()
    exp_date = date.fromisoformat(exp_date_str)
    exp_dt   = datetime(exp_date.year, exp_date.month, exp_date.day, 16, 0, 0)
    seconds  = (exp_dt - now).total_seconds()
    return max(seconds / (365 * 86400), 1.0 / (365 * 24))


def exact_T_rem(short_fecha_str: str, long_fecha_str: str) -> float:
    """Tiempo residual exacto entre expiración short y expiración long."""
    short_dt = date.fromisoformat(short_fecha_str)
    long_dt  = date.fromisoformat(long_fecha_str)
    days     = (long_dt - short_dt).days
    return max(days / 365.0, 1.0 / 365)


# ==============================================================================
# INVERSIÓN DE IV
# ==============================================================================

def iv_from_mid(price: float, S: float, K: float, T: float, r: float,
                option_type: str, fallback: float = 0.17, q: float = 0.0) -> float:
    if price <= 0 or T <= 1e-7:
        return fallback

    intrinsic = max(K - S, 0) if option_type.lower() == "put" else max(S - K, 0)
    if price <= intrinsic:
        return fallback

    lo, hi = 0.001, 5.0
    for _ in range(80):
        mid_v = (lo + hi) / 2
        p = bs_price(S, K, T, r, mid_v, option_type, q)
        if abs(p - price) < 0.005:
            return mid_v
        if p < price:
            lo = mid_v
        else:
            hi = mid_v

    return (lo + hi) / 2


# ==============================================================================
# CALENDAR P/L — curva a expiración de la short
# ==============================================================================

def calendar_pl_at_short_expiry(
    S_range: np.ndarray,
    K: float,
    T_short: float,
    T_long: float,
    iv_short: float,
    iv_long: float,
    r: float,
    debit: float,
    option_type: str = "put",
    contracts: int = 1,
    q: float = 0.0,
    skew_short: Optional[list] = None,
    skew_long:  Optional[list] = None,
    short_fecha_str: str = "",
    long_fecha_str:  str = "",
) -> np.ndarray:
    """
    Curva blanca.
    Short: valorada con T→0+ (Theo muy cerca de expiración).
    Long : valorada con tiempo residual exacto entre fechas.
    IV fija por pata en toda la curva, alineado con 'Individual implied volatility'.
    """
    pl = np.zeros(len(S_range))

    T_tiny = 1.0 / (365.0 * 24 * 4)  # ~15 min
    if short_fecha_str and long_fecha_str:
        T_rem = exact_T_rem(short_fecha_str, long_fecha_str)
    else:
        T_rem = max(T_long - T_short, 1.0 / 365)

    for i, S in enumerate(S_range):
        short_val = bs_price(S, K, T_tiny, r, iv_short, option_type, q)
        long_val  = bs_price(S, K, T_rem,  r, iv_long,  option_type, q)
        pl[i] = (long_val - short_val - debit) * 100 * contracts

    return pl


# ==============================================================================
# CALENDAR P/L — curva de hoy
# ==============================================================================

def calendar_pl_today(
    S_range: np.ndarray,
    K: float,
    T_short: float,
    T_long: float,
    iv_short: float,
    iv_long: float,
    r: float,
    debit: float,
    option_type: str = "put",
    contracts: int = 1,
    q: float = 0.0,
    skew_short: Optional[list] = None,
    skew_long:  Optional[list] = None,
) -> np.ndarray:
    pl = np.zeros(len(S_range))

    for i, S in enumerate(S_range):
        short_val = bs_price(S, K, T_short, r, iv_short, option_type, q)
        long_val  = bs_price(S, K, T_long,  r, iv_long,  option_type, q)
        pl[i] = (long_val - short_val - debit) * 100 * contracts

    return pl


# ==============================================================================
# GRIEGOS AGREGADOS
# ==============================================================================

def spread_greeks_at_price(
    S: float, K: float,
    T_short: float, T_long: float,
    iv_short: float, iv_long: float,
    r: float, contracts: int,
    option_type: str = "put",
    q: float = 0.0,
) -> dict:
    g_s = bs_greeks(S, K, T_short, r, iv_short, option_type, q)
    g_l = bs_greeks(S, K, T_long,  r, iv_long,  option_type, q)

    return {
        "Delta": round((g_l["delta"] - g_s["delta"]) * contracts * 100, 2),
        "Gamma": round((g_l["gamma"] - g_s["gamma"]) * contracts * 100, 4),
        "Theta": round((g_l["theta"] - g_s["theta"]) * contracts * 100, 2),
        "Vega":  round((g_l["vega"]  - g_s["vega"])  * contracts * 100, 2),
    }


# ==============================================================================
# BUILD FIGURE
# ==============================================================================

def build_risk_profile_figure(
    S_current: float,
    K: float,
    T_short: float,
    T_long: float,
    iv_short: float,
    iv_long: float,
    r: float,
    debit: float,
    contracts: int,
    option_type: str = "put",
    margin_pct: float = 0.10,
    q: float = 0.0,
    skew_short: Optional[list] = None,
    skew_long:  Optional[list] = None,
    short_fecha_str: str = "",
    long_fecha_str:  str = "",
) -> tuple[go.Figure, list[float]]:

    width_init = max(K * 0.18, 600)
    S_init     = np.linspace(K - width_init, K + width_init, 800)

    pl_init = calendar_pl_at_short_expiry(
        S_init, K, T_short, T_long, iv_short, iv_long,
        r, debit, option_type, contracts, q,
        skew_short, skew_long, short_fecha_str, long_fecha_str,
    )

    be_preview = []
    for idx in np.where(np.diff(np.sign(pl_init)) != 0)[0]:
        x0, x1 = S_init[idx], S_init[idx + 1]
        y0, y1 = pl_init[idx], pl_init[idx + 1]
        if y1 != y0:
            be_preview.append(x0 - y0 * (x1 - x0) / (y1 - y0))

    if len(be_preview) >= 2:
        span    = max(be_preview) - min(be_preview)
        x_left  = min(be_preview) - span * margin_pct
        x_right = max(be_preview) + span * margin_pct
    elif len(be_preview) == 1:
        x_left  = be_preview[0] - width_init * 0.5
        x_right = be_preview[0] + width_init * 0.5
    else:
        x_left  = K - width_init
        x_right = K + width_init

    S_range = np.linspace(x_left, x_right, 500)

    pl_today = calendar_pl_today(
        S_range, K, T_short, T_long, iv_short, iv_long,
        r, debit, option_type, contracts, q,
        skew_short, skew_long,
    )

    pl_expiry = calendar_pl_at_short_expiry(
        S_range, K, T_short, T_long, iv_short, iv_long,
        r, debit, option_type, contracts, q,
        skew_short, skew_long, short_fecha_str, long_fecha_str,
    )

    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color="#555555", width=1, dash="dot"))

    today_label = date.today().strftime("%-m/%-d/%y")
    fig.add_trace(go.Scatter(
        x=S_range, y=pl_today,
        mode="lines",
        name=f"Hoy  {today_label}",
        line=dict(color="#00c853", width=2),
        hovertemplate="SPX: %{x:,.0f}<br>P/L: $%{y:,.0f}<extra></extra>",
    ))

    try:
        exp_label = date.fromisoformat(short_fecha_str).strftime("%-m/%-d/%y") if short_fecha_str else "Expiración"
    except Exception:
        exp_label = "Expiración"

    fig.add_trace(go.Scatter(
        x=S_range, y=pl_expiry,
        mode="lines",
        name=f"Exp  {exp_label}",
        line=dict(color="#ffffff", width=2.5),
        hovertemplate="SPX: %{x:,.0f}<br>P/L: $%{y:,.0f}<extra></extra>",
    ))

    fig.add_vline(
        x=S_current,
        line=dict(color="#ffc107", width=1.5, dash="dash"),
        annotation_text=f"{S_current:,.2f}",
        annotation_position="top",
        annotation_font=dict(color="#ffc107", size=11),
    )

    fig.add_vline(
        x=K,
        line=dict(color="#ef5350", width=1, dash="dot"),
    )

    be_points = []
    for idx in np.where(np.diff(np.sign(pl_expiry)) != 0)[0]:
        x0, x1 = S_range[idx], S_range[idx + 1]
        y0, y1 = pl_expiry[idx], pl_expiry[idx + 1]
        if y1 != y0:
            be_points.append(float(round(x0 - y0 * (x1 - x0) / (y1 - y0), 2)))

    be_points = sorted(be_points)

    for i, be in enumerate(be_points):
        fig.add_vline(
            x=be,
            line=dict(color="#ef5350", width=1, dash="longdash"),
        )

        x_label = S_range[0] if i == 0 else S_range[-1]
        xanchor = "left" if i == 0 else "right"

        fig.add_annotation(
            x=x_label,
            y=0,
            xref="x", yref="y",
            text=f"BE {be:,.0f}",
            showarrow=False,
            font=dict(color="#ef5350", size=10),
            xanchor=xanchor,
            yanchor="bottom",
            bgcolor="rgba(13,13,26,0.75)",
            borderpad=3,
        )

    max_idx = int(np.argmax(pl_expiry))
    max_pl  = pl_expiry[max_idx]

    fig.add_annotation(
        x=S_range[max_idx], y=max_pl,
        text=f"Max +${max_pl:,.0f}",
        showarrow=True, arrowhead=2, arrowcolor="#ffffff",
        font=dict(color="#ffffff", size=11),
        bgcolor="#1a1a2e", bordercolor="#ffffff", borderwidth=1,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d0d1a",
        plot_bgcolor="#0d0d1a",
        font=dict(family="monospace", color="#cccccc"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(size=11),
        ),
        margin=dict(l=60, r=30, t=40, b=50),
        hovermode="x unified",
        xaxis=dict(title="SPX Price", gridcolor="#222244", tickformat=",", zeroline=False),
        yaxis=dict(title="P/L ($)", gridcolor="#222244", zeroline=False, tickprefix="$", tickformat=","),
        height=560,
    )

    return fig, be_points


# ==============================================================================
# PRICE SLICES
# ==============================================================================

def build_price_slices(
    S_current: float, K: float,
    T_short: float, T_long: float,
    iv_short: float, iv_long: float,
    r: float, debit: float, contracts: int,
    option_type: str = "put",
    n_slices: int = 9,
    q: float = 0.0,
    skew_short: Optional[list] = None,
    skew_long:  Optional[list] = None,
    short_fecha_str: str = "",
    long_fecha_str:  str = "",
) -> pd.DataFrame:
    offsets = np.linspace(-K * 0.08, K * 0.08, n_slices)
    rows = []

    for off in offsets:
        S = S_current + off

        pl_t = float(calendar_pl_today(
            np.array([S]), K, T_short, T_long, iv_short, iv_long,
            r, debit, option_type, contracts, q, skew_short, skew_long
        )[0])

        pl_e = float(calendar_pl_at_short_expiry(
            np.array([S]), K, T_short, T_long, iv_short, iv_long,
            r, debit, option_type, contracts, q,
            skew_short, skew_long, short_fecha_str, long_fecha_str
        )[0])

        g = spread_greeks_at_price(S, K, T_short, T_long, iv_short, iv_long,
                                   r, contracts, option_type, q)

        rows.append({
            "SPX Price": round(S, 0),
            "Offset":    f"{off:+,.0f}",
            "P/L Hoy":   round(pl_t, 2),
            "P/L Exp":   round(pl_e, 2),
            "Delta":     g["Delta"],
            "Theta":     g["Theta"],
            "Vega":      g["Vega"],
        })

    return pd.DataFrame(rows)


# ==============================================================================
# RENDER
# ==============================================================================

def render_risk_profile(
    registro: dict,
    precio_spx_live: Optional[float] = None,
    idx_registro: int = 0,
    skew_short_ext: Optional[list] = None,
    skew_long_ext:  Optional[list] = None,
    iv_short_ext:   Optional[float] = None,
    iv_long_ext:    Optional[float] = None,
):
    if 'te_rp_calc' not in st.session_state:
        st.session_state['te_rp_calc'] = {}

    try:
        K           = float(registro.get("Strike", 0))
        contracts   = int(registro.get("Contratos", 1))
        short_fecha = registro.get("Short Fecha", "")
        long_fecha  = registro.get("Long Fecha",  "")
        option_type = str(registro.get("Short Tipo", "PUT")).lower()

        T_short = exact_T(short_fecha)
        T_long  = exact_T(long_fecha)
    except Exception as e:
        st.error(f"❌ Error al parsear el registro: {e}")
        return

    spx_abrir = float(str(registro.get("SPX al abrir", "5000")).replace(",", ""))
    S_current = precio_spx_live if precio_spx_live else spx_abrir

    r_rate = 0.0375
    q_rate = 0.0

    mid_short_orig = float(registro.get("Short Mid", 0) or 0)
    mid_long_orig  = float(registro.get("Long Mid",  0) or 0)

    iv_short_broker = iv_short_ext or registro.get("Short IV")
    iv_long_broker  = iv_long_ext  or registro.get("Long IV")

    iv_short_calc = iv_from_mid(mid_short_orig, spx_abrir, K, T_short, r_rate, option_type, 0.17, q_rate)
    iv_long_calc  = iv_from_mid(mid_long_orig,  spx_abrir, K, T_long,  r_rate, option_type, 0.17, q_rate)

    iv_short_base = float(iv_short_broker) / 100 if iv_short_broker else iv_short_calc
    iv_long_base  = float(iv_long_broker)  / 100 if iv_long_broker  else iv_long_calc

    tiene_skew_ext = bool(skew_short_ext and len(skew_short_ext) >= 2)

    ts = registro.get("Timestamp", "")
    dte_s = int(T_short * 365)
    dte_l = int(T_long  * 365)
    t_rem_days = (date.fromisoformat(long_fecha) - date.fromisoformat(short_fecha)).days if short_fecha and long_fecha else 0

    st.markdown(
        f"<div style='background:#0d0d1a; border:1px solid #1a3a5c; border-radius:8px; "
        f"padding:12px 18px; margin-bottom:12px;'>"
        f"<span style='font-size:1.1em; color:#ffc107; font-weight:bold;'>📊 Risk Profile — Calendar {option_type.upper()}</span>"
        f"&nbsp;&nbsp;<span style='color:#888; font-size:0.85em;'>Registrado: {ts}</span>"
        f"<br><span style='color:#aaa; font-size:0.9em;'>"
        f"Strike: <b style='color:white'>{K:,.0f}</b>"
        f" · Short: <b style='color:#ef9a9a'>{short_fecha}</b> ({dte_s}d)"
        f" · Long: <b style='color:#a5d6a7'>{long_fecha}</b> ({dte_l}d)"
        f" · T_rem exacto: <b style='color:#80cbc4'>{t_rem_days}d</b>"
        f" · Contratos: <b style='color:white'>{contracts}</b>"
        f" · SPX entrada: <b style='color:#ffc107'>{spx_abrir:,.2f}</b>"
        f"</span></div>",
        unsafe_allow_html=True,
    )

    if tiene_skew_ext:
        n_pts_s = len(skew_short_ext)
        n_pts_l = len(skew_long_ext) if skew_long_ext else 0
        st.markdown(
            f"<div style='background:#0d1a0d; border:1px solid #2e7d32; border-radius:5px; "
            f"padding:5px 12px; font-size:0.82em; margin-bottom:8px; display:inline-block;'>"
            f"ℹ️ <b style='color:#a5d6a7;'>Skew cargado desde Schwab</b> — "
            f"Short: {n_pts_s} puntos · Long: {n_pts_l} puntos · "
            f"<b style='color:white;'>no se usa en esta prueba</b></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='background:#0d1a0d; border:1px solid #1a3a5c; border-radius:5px; "
            "padding:5px 12px; font-size:0.82em; margin-bottom:8px; display:inline-block;'>"
            "ℹ️ <b style='color:#a5d6a7;'>Modo prueba TOS</b> — "
            "IV individual fija por pata en toda la curva.</div>",
            unsafe_allow_html=True,
        )

    col_s, col_l, col_net, col_spx = st.columns([1, 1, 1, 1])

    with col_s:
        mid_short = st.number_input(
            "💰 Mid Short",
            min_value=0.0, value=float(mid_short_orig),
            step=0.05, format="%.2f",
            key=f"rp_mid_short_{ts}_{K}",
        )

    with col_l:
        mid_long = st.number_input(
            "💰 Mid Long",
            min_value=0.0, value=float(mid_long_orig),
            step=0.05, format="%.2f",
            key=f"rp_mid_long_{ts}_{K}",
        )

    debit_auto = round(mid_long - mid_short, 2)

    with col_net:
        debit_manual = st.number_input(
            "📌 Net Débito (fijar)",
            value=float(debit_auto),
            step=0.05, format="%.2f",
            key=f"rp_debit_{ts}_{K}",
        )

    with col_spx:
        S_input = st.number_input(
            "📈 SPX actual",
            min_value=1000.0, value=float(S_current),
            step=1.0, format="%.2f",
            key=f"rp_spx_{ts}_{K}",
        )

    debit = debit_manual

    col_ivs, col_ivl, col_ivadj, col_info = st.columns([1, 1, 1, 2])

    with col_ivs:
        iv_short_pct = st.number_input(
            "📊 IV Short (%)",
            min_value=1.0, max_value=200.0,
            value=round(iv_short_base * 100, 2),
            step=0.5, format="%.2f",
            key=f"rp_iv_short_{ts}_{K}",
            help="IV individual de la pata short"
        )

    with col_ivl:
        iv_long_pct = st.number_input(
            "📊 IV Long (%)",
            min_value=1.0, max_value=200.0,
            value=round(iv_long_base * 100, 2),
            step=0.5, format="%.2f",
            key=f"rp_iv_long_{ts}_{K}",
            help="IV individual de la pata long"
        )

    with col_ivadj:
        iv_adj = st.slider(
            "⚙️ Ajuste IV (%)",
            min_value=-20, max_value=20, value=0, step=1,
            key=f"rp_iv_adj_{ts}_{K}",
            help="Ajuste paralelo sobre ambas patas"
        )

    with col_info:
        origen_iv = "broker individual" if (iv_short_broker or iv_long_broker) else "BS2002 estimada"
        signo_str = "DÉBITO" if debit > 0 else "CRÉDITO"
        color_d   = "#ffb74d" if debit > 0 else "#ef5350"
        st.markdown(
            f"<div style='background:#0d1a0d; border:1px solid #1a3a5c; border-radius:6px; "
            f"padding:8px 12px; margin-top:4px; font-size:0.85em;'>"
            f"<span style='color:#aaa;'>Motor: <b style='color:white;'>BS2002 + IV individual fija + T_rem exacto</b></span><br>"
            f"<span style='color:#aaa;'>IV fuente: <b style='color:white;'>{origen_iv}</b></span><br>"
            f"<span style='font-size:1.2em; font-weight:bold; color:{color_d};'>"
            f"{signo_str}: ${abs(debit):.2f} / cto &nbsp;·&nbsp; "
            f"Total: ${abs(debit) * 100 * contracts:,.2f}"
            f"</span></div>",
            unsafe_allow_html=True,
        )

    iv_s = max((iv_short_pct / 100) * (1 + iv_adj / 100), 0.01)
    iv_l = max((iv_long_pct  / 100) * (1 + iv_adj / 100), 0.01)

    skew_short_final = skew_short_ext if tiene_skew_ext else None
    skew_long_final  = skew_long_ext  if tiene_skew_ext else None

    st.markdown("<br>", unsafe_allow_html=True)

    col_ctrl, _ = st.columns([2, 3])
    with col_ctrl:
        margin_pct = st.slider(
            "↔️ Ancho del gráfico (margen más allá de los BE)",
            min_value=0, max_value=100, value=10, step=5,
            format="%d%%",
            key=f"rp_margin_{ts}_{K}",
        ) / 100.0

    fig, be_points = build_risk_profile_figure(
        S_current=S_input,
        K=K,
        T_short=T_short, T_long=T_long,
        iv_short=iv_s, iv_long=iv_l,
        r=r_rate,
        debit=debit,
        contracts=contracts,
        option_type=option_type,
        margin_pct=margin_pct,
        q=q_rate,
        skew_short=skew_short_final,
        skew_long=skew_long_final,
        short_fecha_str=short_fecha,
        long_fecha_str=long_fecha,
    )

    st.plotly_chart(fig, use_container_width=True)

    width_full = max(K * 0.18, 600)
    S_full     = np.linspace(K - width_full, K + width_full, 600)

    pl_exp_full = calendar_pl_at_short_expiry(
        S_full, K, T_short, T_long, iv_s, iv_l, r_rate, debit,
        option_type, contracts, q_rate,
        skew_short_final, skew_long_final, short_fecha, long_fecha,
    )

    max_idx      = int(np.argmax(pl_exp_full))
    max_pl_val   = float(pl_exp_full[max_idx])
    max_pl_price = float(S_full[max_idx])

    be_lower = be_points[0]  if len(be_points) >= 1 else None
    be_upper = be_points[-1] if len(be_points) >= 2 else None

    pl_now = float(calendar_pl_today(
        np.array([S_input]), K, T_short, T_long, iv_s, iv_l, r_rate, debit,
        option_type, contracts, q_rate, skew_short_final, skew_long_final,
    )[0])

    st.session_state['te_rp_calc'][idx_registro] = {
        'be_lower':         be_lower,
        'be_upper':         be_upper,
        'be_points':        be_points,
        'max_profit_price': max_pl_price,
        'max_profit_pl':    max_pl_val,
        'pl_now':           pl_now,
        'spx_usado':        S_input,
        'debit_usado':      debit,
        'K':                K,
        'T_short':          T_short,
        'T_long':           T_long,
        'contracts':        contracts,
        'option_type':      option_type,
        'tiene_skew':       tiene_skew_ext,
        'T_rem_dias':       t_rem_days,
    }

    col_be1, col_be2, col_mp, col_pl = st.columns(4)

    with col_be1:
        st.metric(
            "📉 BE Inferior",
            f"{be_lower:,.0f}" if be_lower is not None else "—",
            delta=f" ({be_lower - S_input:+,.0f} pts)" if be_lower else None,
            delta_color="inverse",
        )

    with col_be2:
        st.metric(
            "📈 BE Superior",
            f"{be_upper:,.0f}" if be_upper is not None else "—",
            delta=f" ({be_upper - S_input:+,.0f} pts)" if be_upper else None,
        )

    with col_mp:
        st.metric("🎯 Max Profit Price", f"{max_pl_price:,.0f}")

    with col_pl:
        st.metric("💰 Max P/L", f"${max_pl_val:,.0f}")

    g = spread_greeks_at_price(S_input, K, T_short, T_long, iv_s, iv_l, r_rate, contracts, option_type, q_rate)

    st.markdown(
        "<div style='background:#0d0d1a; border:1px solid #1a3a5c; border-radius:6px; "
        "padding:8px 16px; font-family:monospace; font-size:0.85em; margin-bottom:8px;'>"
        "<b style='color:#ffc107;'>▶ Griegos al precio actual del SPX</b>"
        "</div>",
        unsafe_allow_html=True,
    )

    col_g1, col_g2, col_g3, col_g4, col_g5 = st.columns(5)
    col_g1.metric("P/L Open", f"${pl_now:,.2f}")
    col_g2.metric("Delta",    f"{g['Delta']:+.2f}")
    col_g3.metric("Gamma",    f"{g['Gamma']:+.4f}")
    col_g4.metric("Theta",    f"{g['Theta']:+.2f}")
    col_g5.metric("Vega",     f"{g['Vega']:+.2f}")

    st.markdown("---")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def seccion_risk_profile(
    precio_spx_live:  Optional[float] = None,
    mids_refrescados: Optional[dict]  = None,
    idx_registro:     int             = 0,
):
    registros = st.session_state.get("te_calendar_registros", [])

    if not registros:
        st.info("ℹ️ No hay calendars registrados. Registrá uno en la sección 2.4.")
        return

    idx_sel  = min(idx_registro, len(registros) - 1)
    registro = registros[idx_sel]

    skew_short_ext = None
    skew_long_ext  = None
    iv_short_ext   = None
    iv_long_ext    = None
    spx_ref        = precio_spx_live

    if mids_refrescados:
        registro = dict(registro)
        registro['Short Mid'] = mids_refrescados.get('mid_short', registro.get('Short Mid'))
        registro['Long Mid']  = mids_refrescados.get('mid_long',  registro.get('Long Mid'))
        spx_ref        = mids_refrescados.get('spx_al_refrescar', precio_spx_live)
        skew_short_ext = mids_refrescados.get('skew_short')
        skew_long_ext  = mids_refrescados.get('skew_long')
        iv_short_ext   = mids_refrescados.get('iv_short')
        iv_long_ext    = mids_refrescados.get('iv_long')

    render_risk_profile(
        registro,
        precio_spx_live=spx_ref,
        idx_registro=idx_sel,
        skew_short_ext=skew_short_ext,
        skew_long_ext=skew_long_ext,
        iv_short_ext=iv_short_ext,
        iv_long_ext=iv_long_ext,
    )
