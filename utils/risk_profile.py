# utils/risk_profile.py
"""
Risk Profile — Calendar Put Spread SPX
Motor: Bjerksund-Stensland 2002 + skew local por vencimiento (desde Schwab API).
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

def _bs2002_phi(S, T, gamma, H, I, r, b, sigma):
    lam = (-r + gamma * b + 0.5 * gamma * (gamma - 1) * sigma ** 2) * T
    d   = -(math.log(S / H) + (b + (gamma - 0.5) * sigma ** 2) * T) / (sigma * math.sqrt(T))
    kap = 2 * b / sigma ** 2 + (2 * gamma - 1)
    return (math.exp(lam) * S ** gamma *
            (_norm_cdf(d) - (I / S) ** kap *
             _norm_cdf(d - 2 * math.log(I / S) / (sigma * math.sqrt(T)))))


def _bs2002_call(S, K, T, r, b, sigma):
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
    d1 = (math.log(S / K) + (b + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return (alpha * S ** beta
            - alpha * _bs2002_phi(S, T, beta, I, I, r, b, sigma)
            + _bs2002_phi(S, T, 1, I, I, r, b, sigma)
            - _bs2002_phi(S, T, 1, K, I, r, b, sigma)
            - K * _bs2002_phi(S, T, 0, I, I, r, b, sigma)
            + K * _bs2002_phi(S, T, 0, K, I, r, b, sigma))


def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str = "put", q: float = 0.0) -> float:
    """Precio BS2002. Para puts: put-call parity sobre call BS2002."""
    if T <= 1e-7 or sigma <= 1e-6:
        return max(K - S, 0.0) if option_type.lower() == "put" else max(S - K, 0.0)
    b    = r - q
    call = _bs2002_call(S, K, T, r, b, sigma)
    if option_type.lower() == "call":
        return call
    # Put-call parity (válido para europeas SPX cash-settled)
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
    p0   = bs_price(S,       K, T,              r, sigma,     option_type, q)
    p_su = bs_price(S + h_s, K, T,              r, sigma,     option_type, q)
    p_sd = bs_price(S - h_s, K, T,              r, sigma,     option_type, q)
    p_vu = bs_price(S,       K, T,              r, sigma + h_v, option_type, q)
    p_td = bs_price(S,       K, max(T - h_t, 1e-7), r, sigma, option_type, q)
    return {
        "delta": round((p_su - p_sd) / (2 * h_s), 4),
        "gamma": round((p_su - 2 * p0 + p_sd) / (h_s ** 2), 6),
        "vega":  round((p_vu - p0) / h_v / 100, 4),
        "theta": round((p_td - p0) / h_t, 4),
    }


# ==============================================================================
# SKEW LOCAL — interpolación cuadrática
# ==============================================================================

def interp_iv(S_query: float,
              skew_points: Optional[list],
              fallback_iv: float) -> float:
    """
    Interpola IV en función del precio del subyacente S_query.
    skew_points: [(strike, iv_decimal), ...] — viene de Schwab API vía refrescar_mids.
    Con < 2 puntos devuelve fallback_iv.
    """
    if not skew_points or len(skew_points) < 2:
        return fallback_iv
    strikes = np.array([p[0] for p in skew_points], dtype=float)
    ivs     = np.array([p[1] for p in skew_points], dtype=float)
    if len(skew_points) == 2:
        iv = float(np.interp(S_query, strikes, ivs))
    else:
        try:
            coeffs = np.polyfit(strikes, ivs, min(2, len(skew_points) - 1))
            iv = float(np.polyval(coeffs, S_query))
        except Exception:
            iv = float(np.interp(S_query, strikes, ivs))
    return float(np.clip(iv, 0.005, 5.0))


# ==============================================================================
# TIEMPO EXACTO — fracción de día con hora actual
# ==============================================================================

def exact_T(exp_date_str: str) -> float:
    """T en años desde ahora hasta expiración (cierre 16:00 del día de exp)."""
    now      = datetime.now()
    exp_date = date.fromisoformat(exp_date_str)
    exp_dt   = datetime(exp_date.year, exp_date.month, exp_date.day, 16, 0, 0)
    seconds  = (exp_dt - now).total_seconds()
    return max(seconds / (365.0 * 86400), 1.0 / (365 * 24))


# ==============================================================================
# INVERSIÓN DE IV desde mid (bisección sobre BS2002)
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
# CALENDAR P/L — a expiración de la short (curva blanca)
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
) -> np.ndarray:
    """
    Curva blanca: P/L al vencimiento de la short.
    Short → Theo con T=15min (igual que TOS "Theo at expiry", no intrínseco crudo).
    Long  → BS2002 con tiempo remanente T_long - T_short.
    Skew interpolado en cada precio del rango.
    """
    pl     = np.zeros(len(S_range))
    T_tiny = 1.0 / (365 * 24 * 4)          # ~15 min
    T_rem  = max(T_long - T_short, 1.0 / 365)

    for i, S in enumerate(S_range):
        iv_s = interp_iv(S, skew_short, iv_short)
        iv_l = interp_iv(S, skew_long,  iv_long)
        short_val = bs_price(S, K, T_tiny, r, iv_s, option_type, q)
        long_val  = bs_price(S, K, T_rem,  r, iv_l, option_type, q)
        pl[i] = (long_val - short_val - debit) * 100 * contracts

    return pl


# ==============================================================================
# CALENDAR P/L — hoy (curva verde)
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
        iv_s = interp_iv(S, skew_short, iv_short)
        iv_l = interp_iv(S, skew_long,  iv_long)
        short_val = bs_price(S, K, T_short, r, iv_s, option_type, q)
        long_val  = bs_price(S, K, T_long,  r, iv_l, option_type, q)
        pl[i] = (long_val - short_val - debit) * 100 * contracts
    return pl


# ==============================================================================
# GRIEGOS AGREGADOS DEL SPREAD
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
# BUILD PLOTLY FIGURE
# ==============================================================================

def build_risk_profile_figure(
    S_current: float, K: float,
    T_short: float, T_long: float,
    iv_short: float, iv_long: float,
    r: float, debit: float, contracts: int,
    option_type: str = "put",
    margin_pct: float = 0.10,
    q: float = 0.0,
    skew_short: Optional[list] = None,
    skew_long:  Optional[list] = None,
) -> tuple[go.Figure, list[float]]:

    # Rango amplio para detectar BEs
    width_init   = max(K * 0.18, 600)
    S_init       = np.linspace(K - width_init, K + width_init, 800)
    pl_init      = calendar_pl_at_short_expiry(
        S_init, K, T_short, T_long, iv_short, iv_long, r, debit,
        option_type, contracts, q, skew_short, skew_long,
    )
    be_preview = []
    for idx in np.where(np.diff(np.sign(pl_init)) != 0)[0]:
        x0, x1 = S_init[idx], S_init[idx + 1]
        y0, y1 = pl_init[idx], pl_init[idx + 1]
        if y1 != y0:
            be_preview.append(x0 - y0 * (x1 - x0) / (y1 - y0))

    # Ajustar rango X
    if len(be_preview) >= 2:
        span    = max(be_preview) - min(be_preview)
        x_left  = min(be_preview) - span * margin_pct
        x_right = max(be_preview) + span * margin_pct
    elif len(be_preview) == 1:
        x_left  = be_preview[0] - width_init * 0.5
        x_right = be_preview[0] + width_init * 0.5
    else:
        x_left, x_right = K - width_init, K + width_init

    S_range = np.linspace(x_left, x_right, 500)

    pl_today  = calendar_pl_today(
        S_range, K, T_short, T_long, iv_short, iv_long, r, debit,
        option_type, contracts, q, skew_short, skew_long,
    )
    pl_expiry = calendar_pl_at_short_expiry(
        S_range, K, T_short, T_long, iv_short, iv_long, r, debit,
        option_type, contracts, q, skew_short, skew_long,
    )

    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color="#555555", width=1, dash="dot"))

    today_label = date.today().strftime("%-m/%-d/%y")
    fig.add_trace(go.Scatter(
        x=S_range, y=pl_today, mode="lines",
        name=f"Hoy  {today_label}",
        line=dict(color="#00c853", width=2),
        hovertemplate="SPX: %{x:,.0f}<br>P/L: $%{y:,.0f}<extra></extra>",
    ))

    try:
        exp_label = (date.today() + pd.Timedelta(days=int(T_short * 365))).strftime("%-m/%-d/%y")
    except Exception:
        exp_label = "Expiración"

    fig.add_trace(go.Scatter(
        x=S_range, y=pl_expiry, mode="lines",
        name=f"Exp  {exp_label}",
        line=dict(color="#ffffff", width=2.5),
        hovertemplate="SPX: %{x:,.0f}<br>P/L: $%{y:,.0f}<extra></extra>",
    ))

    fig.add_vline(x=S_current,
                  line=dict(color="#ffc107", width=1.5, dash="dash"),
                  annotation_text=f"{S_current:,.2f}",
                  annotation_position="top",
                  annotation_font=dict(color="#ffc107", size=11))

    fig.add_vline(x=K, line=dict(color="#ef5350", width=1, dash="dot"))

    # Breakevens
    be_points = []
    for idx in np.where(np.diff(np.sign(pl_expiry)) != 0)[0]:
        x0, x1 = S_range[idx], S_range[idx + 1]
        y0, y1 = pl_expiry[idx], pl_expiry[idx + 1]
        if y1 != y0:
            be_points.append(float(round(x0 - y0 * (x1 - x0) / (y1 - y0), 2)))
    be_points = sorted(be_points)

    for i, be in enumerate(be_points):
        fig.add_vline(x=be, line=dict(color="#ef5350", width=1, dash="longdash"))
        fig.add_annotation(
            x=S_range[0] if i == 0 else S_range[-1], y=0,
            xref="x", yref="y",
            text=f"BE {be:,.0f}", showarrow=False,
            font=dict(color="#ef5350", size=10),
            xanchor="left" if i == 0 else "right",
            yanchor="bottom",
            bgcolor="rgba(13,13,26,0.75)", borderpad=3,
        )

    max_idx = int(np.argmax(pl_expiry))
    fig.add_annotation(
        x=S_range[max_idx], y=pl_expiry[max_idx],
        text=f"Max +${pl_expiry[max_idx]:,.0f}",
        showarrow=True, arrowhead=2, arrowcolor="#ffffff",
        font=dict(color="#ffffff", size=11),
        bgcolor="#1a1a2e", bordercolor="#ffffff", borderwidth=1,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d0d1a", plot_bgcolor="#0d0d1a",
        font=dict(family="monospace", color="#cccccc"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=60, r=30, t=40, b=50),
        hovermode="x unified",
        xaxis=dict(title="SPX Price", gridcolor="#222244", tickformat=",", zeroline=False),
        yaxis=dict(title="P/L ($)", gridcolor="#222244", zeroline=False,
                   tickprefix="$", tickformat=","),
        height=560,
    )
    return fig, be_points


# ==============================================================================
# PRICE SLICES TABLE
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
) -> pd.DataFrame:
    offsets = np.linspace(-K * 0.08, K * 0.08, n_slices)
    rows = []
    for off in offsets:
        S = S_current + off
        pl_t = float(calendar_pl_today(
            np.array([S]), K, T_short, T_long, iv_short, iv_long, r, debit,
            option_type, contracts, q, skew_short, skew_long)[0])
        pl_e = float(calendar_pl_at_short_expiry(
            np.array([S]), K, T_short, T_long, iv_short, iv_long, r, debit,
            option_type, contracts, q, skew_short, skew_long)[0])
        g = spread_greeks_at_price(S, K, T_short, T_long,
                                   interp_iv(S, skew_short, iv_short),
                                   interp_iv(S, skew_long,  iv_long),
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
# STREAMLIT COMPONENT
# ==============================================================================

def render_risk_profile(
    registro: dict,
    precio_spx_live: Optional[float] = None,
    idx_registro: int = 0,
    skew_short_ext: Optional[list] = None,   # viene de mids_refrescados['skew_short']
    skew_long_ext:  Optional[list] = None,   # viene de mids_refrescados['skew_long']
    iv_short_ext:   Optional[float] = None,  # IV ATM del broker (mids_refrescados['iv_short'])
    iv_long_ext:    Optional[float] = None,  # IV ATM del broker (mids_refrescados['iv_long'])
):
    hoy = date.today()

    if 'te_rp_calc' not in st.session_state:
        st.session_state['te_rp_calc'] = {}

    # ---- Extraer datos del registro ----
    try:
        K           = float(registro.get("Strike", 0))
        contracts   = int(registro.get("Contratos", 1))
        short_fecha = registro.get("Short Fecha", "")
        long_fecha  = registro.get("Long Fecha",  "")
        option_type = str(registro.get("Short Tipo", "PUT")).lower()
        T_short     = exact_T(short_fecha)
        T_long      = exact_T(long_fecha)
    except Exception as e:
        st.error(f"❌ Error al parsear el registro: {e}")
        return

    spx_abrir = float(str(registro.get("SPX al abrir", "5000")).replace(",", ""))
    S_current = precio_spx_live if precio_spx_live else spx_abrir

    mid_short_orig = float(registro.get("Short Mid", 0) or 0)
    mid_long_orig  = float(registro.get("Long Mid",  0) or 0)

    r_rate = 0.0375
    q_rate = 0.0

    # ---- IV base: broker > IV externo > inversión BS2002 desde mid ----
    iv_short_broker = registro.get("Short IV")
    iv_long_broker  = registro.get("Long IV")

    iv_short_calc = iv_from_mid(mid_short_orig, spx_abrir, K, T_short, r_rate, option_type, 0.17, q_rate)
    iv_long_calc  = iv_from_mid(mid_long_orig,  spx_abrir, K, T_long,  r_rate, option_type, 0.17, q_rate)

    # Prioridad: iv_short_ext (del último refresco) > iv_short_broker (guardado) > calculada
    if iv_short_ext is not None:
        iv_short_base = iv_short_ext
    elif iv_short_broker:
        iv_short_base = float(iv_short_broker) / 100
    else:
        iv_short_base = iv_short_calc

    if iv_long_ext is not None:
        iv_long_base = iv_long_ext
    elif iv_long_broker:
        iv_long_base = float(iv_long_broker) / 100
    else:
        iv_long_base = iv_long_calc

    # =========================================================================
    # HEADER
    # =========================================================================
    ts    = registro.get("Timestamp", "")
    dte_s = int(T_short * 365)
    dte_l = int(T_long  * 365)

    tiene_skew = bool(skew_short_ext or skew_long_ext)
    badge_skew = (
        "<span style='background:#1a3a1a; color:#69f0ae; padding:2px 8px; "
        "border-radius:4px; font-size:0.8em; margin-left:8px;'>✅ Skew real cargado</span>"
        if tiene_skew else
        "<span style='background:#3a1a1a; color:#ff7043; padding:2px 8px; "
        "border-radius:4px; font-size:0.8em; margin-left:8px;'>⚠️ Sin skew — IV plana</span>"
    )

    st.markdown(
        f"<div style='background:#0d0d1a; border:1px solid #1a3a5c; border-radius:8px; "
        f"padding:12px 18px; margin-bottom:12px;'>"
        f"<span style='font-size:1.1em; color:#ffc107; font-weight:bold;'>"
        f"📊 Risk Profile — Calendar {option_type.upper()}</span>"
        f"{badge_skew}"
        f"&nbsp;&nbsp;<span style='color:#888; font-size:0.85em;'>Registrado: {ts}</span>"
        f"<br><span style='color:#aaa; font-size:0.9em;'>"
        f"Strike: <b style='color:white'>{K:,.0f}</b>"
        f" · Short: <b style='color:#ef9a9a'>{short_fecha}</b> ({dte_s}d)"
        f" · Long: <b style='color:#a5d6a7'>{long_fecha}</b> ({dte_l}d)"
        f" · Contratos: <b style='color:white'>{contracts}</b>"
        f" · SPX entrada: <b style='color:#ffc107'>{spx_abrir:,.2f}</b>"
        f"</span></div>",
        unsafe_allow_html=True,
    )

    # =========================================================================
    # FILA 1: Mids + Débito + SPX
    # =========================================================================
    col_s, col_l, col_net, col_spx = st.columns([1, 1, 1, 1])
    with col_s:
        mid_short = st.number_input("💰 Mid Short", min_value=0.0, value=float(mid_short_orig),
                                    step=0.05, format="%.2f", key=f"rp_mid_short_{ts}_{K}")
    with col_l:
        mid_long  = st.number_input("💰 Mid Long",  min_value=0.0, value=float(mid_long_orig),
                                    step=0.05, format="%.2f", key=f"rp_mid_long_{ts}_{K}")
    with col_net:
        debit_manual = st.number_input("📌 Net Débito (fijar)",
                                       value=float(round(mid_long - mid_short, 2)),
                                       step=0.05, format="%.2f", key=f"rp_debit_{ts}_{K}")
    with col_spx:
        S_input = st.number_input("📈 SPX actual", min_value=1000.0, value=float(S_current),
                                  step=1.0, format="%.2f", key=f"rp_spx_{ts}_{K}")
    debit = debit_manual

    # =========================================================================
    # FILA 2: IVs + ajuste + info
    # =========================================================================
    col_ivs, col_ivl, col_ivadj, col_info = st.columns([1, 1, 1, 2])
    with col_ivs:
        iv_short_pct = st.number_input(
            "📊 IV Short (%)", min_value=1.0, max_value=200.0,
            value=round(iv_short_base * 100, 2), step=0.5, format="%.2f",
            key=f"rp_iv_short_{ts}_{K}",
            help="IV ATM pata short — del último refresco Schwab o calculada via BS2002")
    with col_ivl:
        iv_long_pct = st.number_input(
            "📊 IV Long (%)", min_value=1.0, max_value=200.0,
            value=round(iv_long_base * 100, 2), step=0.5, format="%.2f",
            key=f"rp_iv_long_{ts}_{K}",
            help="IV ATM pata long — del último refresco Schwab o calculada via BS2002")
    with col_ivadj:
        iv_adj = st.slider("⚙️ Ajuste IV (%)", min_value=-20, max_value=20, value=0, step=1,
                           key=f"rp_iv_adj_{ts}_{K}",
                           help="Escenario de expansión/contracción de vol")
    with col_info:
        origen_iv = "Schwab (refresco)" if tiene_skew else (
            "broker (guardado)" if (iv_short_broker or iv_long_broker) else "BS2002 estimada")
        signo_str = "DÉBITO" if debit > 0 else "CRÉDITO"
        color_d   = "#ffb74d" if debit > 0 else "#ef5350"
        n_pts_s   = len(skew_short_ext) if skew_short_ext else 0
        n_pts_l   = len(skew_long_ext)  if skew_long_ext  else 0
        st.markdown(
            f"<div style='background:#0d1a0d; border:1px solid #1a3a5c; border-radius:6px; "
            f"padding:8px 12px; margin-top:4px; font-size:0.85em;'>"
            f"<span style='color:#aaa;'>Motor: <b style='color:white;'>BS2002 + skew local</b></span><br>"
            f"<span style='color:#aaa;'>IV fuente: <b style='color:white;'>{origen_iv}</b>"
            f"{'  · Short '+str(n_pts_s)+' pts / Long '+str(n_pts_l)+' pts' if tiene_skew else ''}"
            f"</span><br>"
            f"<span style='font-size:1.2em; font-weight:bold; color:{color_d};'>"
            f"{signo_str}: ${abs(debit):.2f} / cto &nbsp;·&nbsp; "
            f"Total: ${abs(debit) * 100 * contracts:,.2f}</span></div>",
            unsafe_allow_html=True,
        )

    iv_s = max((iv_short_pct / 100) * (1 + iv_adj / 100), 0.01)
    iv_l = max((iv_long_pct  / 100) * (1 + iv_adj / 100), 0.01)

    # Aplicar ajuste iv_adj también al skew si existe
    skew_s = [(k, v * (1 + iv_adj / 100)) for k, v in skew_short_ext] if skew_short_ext else None
    skew_l = [(k, v * (1 + iv_adj / 100)) for k, v in skew_long_ext]  if skew_long_ext  else None

    # =========================================================================
    # GRÁFICA
    # =========================================================================
    col_chart_ctrl, _ = st.columns([2, 3])
    with col_chart_ctrl:
        margin_pct = st.slider(
            "↔️ Ancho del gráfico (margen más allá de los BE)",
            min_value=0, max_value=100, value=10, step=5,
            format="%d%%", key=f"rp_margin_{ts}_{K}",
        ) / 100.0

    fig, be_points = build_risk_profile_figure(
        S_current=S_input, K=K,
        T_short=T_short, T_long=T_long,
        iv_short=iv_s, iv_long=iv_l,
        r=r_rate, debit=debit, contracts=contracts,
        option_type=option_type, margin_pct=margin_pct,
        q=q_rate, skew_short=skew_s, skew_long=skew_l,
    )
    st.plotly_chart(fig, use_container_width=True)

    # =========================================================================
    # SESSION STATE
    # =========================================================================
    width_full   = max(K * 0.18, 600)
    S_full       = np.linspace(K - width_full, K + width_full, 600)
    pl_exp_full  = calendar_pl_at_short_expiry(
        S_full, K, T_short, T_long, iv_s, iv_l, r_rate, debit,
        option_type, contracts, q_rate, skew_s, skew_l,
    )
    max_idx      = int(np.argmax(pl_exp_full))
    max_pl_val   = float(pl_exp_full[max_idx])
    max_pl_price = float(S_full[max_idx])
    be_lower     = be_points[0]  if len(be_points) >= 1 else None
    be_upper     = be_points[-1] if len(be_points) >= 2 else None
    pl_now       = float(calendar_pl_today(
        np.array([S_input]), K, T_short, T_long, iv_s, iv_l, r_rate, debit,
        option_type, contracts, q_rate, skew_s, skew_l,
    )[0])

    st.session_state['te_rp_calc'][idx_registro] = {
        'be_lower': be_lower, 'be_upper': be_upper, 'be_points': be_points,
        'max_profit_price': max_pl_price, 'max_profit_pl': max_pl_val,
        'pl_now': pl_now, 'spx_usado': S_input, 'debit_usado': debit,
        'K': K, 'T_short': T_short, 'T_long': T_long,
        'iv_short': iv_s, 'iv_long': iv_l,
        'contracts': contracts, 'option_type': option_type,
        'tiene_skew': tiene_skew,
    }

    # =========================================================================
    # PANEL BE + MAX PROFIT
    # =========================================================================
    col_be1, col_be2, col_mp, col_pl = st.columns(4)
    with col_be1:
        be_l = f"{be_lower:,.0f}" if be_lower else "—"
        d_l  = f" ({be_lower - S_input:+,.0f} pts)" if be_lower else None
        st.metric("📉 BE Inferior", be_l, delta=d_l, delta_color="inverse")
    with col_be2:
        be_u = f"{be_upper:,.0f}" if be_upper else "—"
        d_u  = f" ({be_upper - S_input:+,.0f} pts)" if be_upper else None
        st.metric("📈 BE Superior", be_u, delta=d_u)
    with col_mp:
        st.metric("🎯 Max Profit Price", f"{max_pl_price:,.0f}")
    with col_pl:
        st.metric("💰 Max P/L", f"${max_pl_val:,.0f}")

    # =========================================================================
    # GRIEGOS
    # =========================================================================
    iv_s_atm = interp_iv(S_input, skew_s, iv_s)
    iv_l_atm = interp_iv(S_input, skew_l, iv_l)
    g = spread_greeks_at_price(S_input, K, T_short, T_long,
                                iv_s_atm, iv_l_atm, r_rate, contracts, option_type, q_rate)

    st.markdown(
        "<div style='background:#0d0d1a; border:1px solid #1a3a5c; border-radius:6px; "
        "padding:8px 16px; font-family:monospace; font-size:0.85em; margin-bottom:8px;'>"
        "<b style='color:#ffc107;'>▶ Griegos al precio actual del SPX (BS2002 + skew)</b>"
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
# SECCIÓN 2.5 — punto de entrada principal
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

    # Extraer skew e IVs del último refresco si existen
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
        skew_short_ext = mids_refrescados.get('skew_short')  # [(strike, iv_decimal), ...]
        skew_long_ext  = mids_refrescados.get('skew_long')   # [(strike, iv_decimal), ...]
        iv_short_ext   = mids_refrescados.get('iv_short')    # float decimal (ej: 0.1611)
        iv_long_ext    = mids_refrescados.get('iv_long')     # float decimal (ej: 0.1830)

    render_risk_profile(
        registro       = registro,
        precio_spx_live= spx_ref,
        idx_registro   = idx_sel,
        skew_short_ext = skew_short_ext,
        skew_long_ext  = skew_long_ext,
        iv_short_ext   = iv_short_ext,
        iv_long_ext    = iv_long_ext,
    )
