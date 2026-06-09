# utils/risk_profile.py
"""
Risk Profile — Calendar Put Spread SPX
Simula el Risk Profile de thinkorswim para calendars registrados en 2.4.
Incluye:
  · Gráfica P/L interactiva (Plotly)
  · Griegos agregados en tiempo real
  · Edición de débito/crédito post-registro
  · Prob ITM / breakevens estimados
  · Price Slices (tabla de escenarios)
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
# BLACK-SCHOLES HELPERS
# ==============================================================================

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str = "put") -> float:
    """Black-Scholes European option price."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(K - S, 0) if option_type.lower() == "put" else max(S - K, 0)
        return intrinsic
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type.lower() == "put":
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
    else:
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float,
              option_type: str = "put") -> dict:
    """Delta, Gamma, Theta, Vega for a European option."""
    if T <= 0 or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    pdf_d1 = _norm_pdf(d1)
    gamma = pdf_d1 / (S * sigma * math.sqrt(T))
    vega  = S * pdf_d1 * math.sqrt(T) / 100  # per 1% IV move
    if option_type.lower() == "put":
        delta = _norm_cdf(d1) - 1.0
        theta = (
            (-S * pdf_d1 * sigma / (2 * math.sqrt(T))
             + r * K * math.exp(-r * T) * _norm_cdf(-d2)) / 365
        )
    else:
        delta = _norm_cdf(d1)
        theta = (
            (-S * pdf_d1 * sigma / (2 * math.sqrt(T))
             - r * K * math.exp(-r * T) * _norm_cdf(d2)) / 365
        )
    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 4),
        "vega":  round(vega,  4),
    }


# ==============================================================================
# CALENDAR SPREAD P/L MODEL
# ==============================================================================

def calendar_pl_at_short_expiry(
    S_range: np.ndarray,
    K: float,
    T_short: float,   # years to short expiry (from today)
    T_long: float,    # years to long expiry  (from today)
    iv_short: float,
    iv_long: float,
    r: float,
    debit: float,     # net debit paid (positive) or credit received (negative)
    option_type: str = "put",
    contracts: int = 1,
) -> np.ndarray:
    """
    P/L of a calendar spread at short-leg expiration.
    At T_short: short leg → intrinsic value, long leg → BS with remaining T_long - T_short.
    Net P/L per contract (×100) minus original debit.
    """
    pl = np.zeros(len(S_range))
    T_rem = max(T_long - T_short, 1 / 365)  # remaining time on long leg

    for i, S in enumerate(S_range):
        # Short leg value at expiration (intrinsic)
        if option_type.lower() == "put":
            short_val = max(K - S, 0)
        else:
            short_val = max(S - K, 0)

        # Long leg value with remaining time (BS)
        long_val = bs_price(S, K, T_rem, r, iv_long, option_type)

        # Calendar P/L: long - short (we sold short, bought long)
        pl[i] = (long_val - short_val - debit) * 100 * contracts

    return pl


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
) -> np.ndarray:
    """P/L of the calendar spread TODAY (both legs still alive)."""
    pl = np.zeros(len(S_range))
    for i, S in enumerate(S_range):
        short_val = bs_price(S, K, T_short, r, iv_short, option_type)
        long_val  = bs_price(S, K, T_long,  r, iv_long,  option_type)
        pl[i] = (long_val - short_val - debit) * 100 * contracts
    return pl


# ==============================================================================
# GREEKS AGGREGATE FOR THE SPREAD AT CURRENT PRICE
# ==============================================================================

def spread_greeks_at_price(
    S: float, K: float,
    T_short: float, T_long: float,
    iv_short: float, iv_long: float,
    r: float, contracts: int,
    option_type: str = "put",
) -> dict:
    g_short = bs_greeks(S, K, T_short, r, iv_short, option_type)
    g_long  = bs_greeks(S, K, T_long,  r, iv_long,  option_type)
    # Calendar = LONG long + SHORT short
    delta = round((g_long["delta"] - g_short["delta"]) * contracts * 100, 2)
    gamma = round((g_long["gamma"] - g_short["gamma"]) * contracts * 100, 4)
    theta = round((g_long["theta"] - g_short["theta"]) * contracts * 100, 2)
    vega  = round((g_long["vega"]  - g_short["vega"])  * contracts * 100, 2)
    return {"Delta": delta, "Gamma": gamma, "Theta": theta, "Vega": vega}


# ==============================================================================
# BUILD PLOTLY FIGURE (mimics TOS Risk Profile)
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
) -> go.Figure:
    # Price range: K ± ~10%
    width = max(K * 0.12, 400)
    S_range = np.linspace(K - width, K + width, 400)

    pl_today  = calendar_pl_today(
        S_range, K, T_short, T_long, iv_short, iv_long, r, debit, option_type, contracts
    )
    pl_expiry = calendar_pl_at_short_expiry(
        S_range, K, T_short, T_long, iv_short, iv_long, r, debit, option_type, contracts
    )

    fig = go.Figure()

    # Zero line
    fig.add_hline(y=0, line=dict(color="#555555", width=1, dash="dot"))

    # Current date P/L (green)
    today_label = date.today().strftime("%-m/%-d/%y")
    fig.add_trace(go.Scatter(
        x=S_range, y=pl_today,
        mode="lines",
        name=f"Hoy  {today_label}",
        line=dict(color="#00c853", width=2),
        hovertemplate="SPX: %{x:,.0f}<br>P/L: $%{y:,.0f}<extra></extra>",
    ))

    # Expiry P/L (white)
    try:
        short_dt = date.today() + pd.Timedelta(days=int(T_short * 365))
        exp_label = short_dt.strftime("%-m/%-d/%y")
    except Exception:
        exp_label = "Expiración"

    fig.add_trace(go.Scatter(
        x=S_range, y=pl_expiry,
        mode="lines",
        name=f"Exp  {exp_label}",
        line=dict(color="#ffffff", width=2.5),
        hovertemplate="SPX: %{x:,.0f}<br>P/L: $%{y:,.0f}<extra></extra>",
    ))

    # Vertical: current price
    fig.add_vline(
        x=S_current,
        line=dict(color="#ffc107", width=1.5, dash="dash"),
        annotation_text=f"{S_current:,.2f}",
        annotation_position="top",
        annotation_font=dict(color="#ffc107", size=11),
    )

    # Vertical: strike
    fig.add_vline(
        x=K,
        line=dict(color="#ef5350", width=1, dash="dot"),
    )

    # Breakevens (where expiry line crosses zero)
    be_mask = np.diff(np.sign(pl_expiry))
    for idx in np.where(be_mask != 0)[0]:
        x0, x1 = S_range[idx], S_range[idx + 1]
        y0, y1 = pl_expiry[idx], pl_expiry[idx + 1]
        if y1 != y0:
            be = x0 - y0 * (x1 - x0) / (y1 - y0)
            fig.add_vline(
                x=be,
                line=dict(color="#ef5350", width=1, dash="longdash"),
                annotation_text=f"BE {be:,.0f}",
                annotation_position="bottom right",
                annotation_font=dict(color="#ef5350", size=10),
            )

    # Max profit annotation
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
        xaxis=dict(
            title="SPX Price",
            gridcolor="#222244",
            tickformat=",",
            zeroline=False,
        ),
        yaxis=dict(
            title="P/L ($)",
            gridcolor="#222244",
            zeroline=False,
            tickprefix="$",
            tickformat=",",
        ),
        height=420,
    )

    return fig


# ==============================================================================
# PRICE SLICES TABLE
# ==============================================================================

def build_price_slices(
    S_current: float,
    K: float,
    T_short: float, T_long: float,
    iv_short: float, iv_long: float,
    r: float, debit: float, contracts: int,
    option_type: str = "put",
    n_slices: int = 9,
) -> pd.DataFrame:
    offsets = np.linspace(-K * 0.08, K * 0.08, n_slices)
    rows = []
    for off in offsets:
        S = S_current + off
        pl_t = float(calendar_pl_today([S], K, T_short, T_long, iv_short, iv_long,
                                       r, debit, option_type, contracts)[0])
        pl_e = float(calendar_pl_at_short_expiry([S], K, T_short, T_long, iv_short,
                                                  iv_long, r, debit, option_type, contracts)[0])
        g = spread_greeks_at_price(S, K, T_short, T_long, iv_short, iv_long, r, contracts, option_type)
        rows.append({
            "SPX Price":  round(S, 0),
            "Offset":     f"{off:+,.0f}",
            "P/L Hoy":    round(pl_t, 2),
            "P/L Exp":    round(pl_e, 2),
            "Delta":      g["Delta"],
            "Theta":      g["Theta"],
            "Vega":       g["Vega"],
        })
    return pd.DataFrame(rows)


# ==============================================================================
# STREAMLIT COMPONENT — llamar desde la página principal
# ==============================================================================

def render_risk_profile(registro: dict, precio_spx_live: Optional[float] = None):
    """
    Renderiza el Risk Profile completo para un registro del 2.4.

    Parámetros esperados en `registro` (mismos que genera el 2.4):
        Timestamp, SPX al abrir, Strike,
        Short Tipo, Short Fecha, Short Mid,
        Long Tipo,  Long Fecha,  Long Mid,
        DÉBITO / CRÉDITO, Total ($), Contratos
    """
    hoy = date.today()

    # ---- Extraer datos del registro ----
    try:
        K           = float(registro.get("Strike", 0))
        contracts   = int(registro.get("Contratos", 1))
        short_fecha = registro.get("Short Fecha", "")
        long_fecha  = registro.get("Long Fecha",  "")
        option_type = str(registro.get("Short Tipo", "PUT")).lower()

        T_short = max((date.fromisoformat(short_fecha) - hoy).days, 1) / 365
        T_long  = max((date.fromisoformat(long_fecha)  - hoy).days, 1) / 365
    except Exception as e:
        st.error(f"❌ Error al parsear el registro: {e}")
        return

    # Precio SPX de referencia
    spx_abrir = float(str(registro.get("SPX al abrir", "5000")).replace(",", ""))
    S_current = precio_spx_live if precio_spx_live else spx_abrir

    # ---- Mids originales ----
    mid_short_orig = float(registro.get("Short Mid", 0) or 0)
    mid_long_orig  = float(registro.get("Long Mid",  0) or 0)
    debit_orig     = round(mid_long_orig - mid_short_orig, 2)

    # ---- IV implícita: derivada del mid via BS (aproximación) ----
    def iv_from_mid(price: float, S: float, K: float, T: float,
                    r: float, otype: str, fallback: float = 0.17) -> float:
        if price <= 0 or T <= 0:
            return fallback
        lo, hi = 0.001, 5.0
        for _ in range(60):
            mid_v = (lo + hi) / 2
            p = bs_price(S, K, T, r, mid_v, otype)
            if abs(p - price) < 0.01:
                return mid_v
            if p < price:
                lo = mid_v
            else:
                hi = mid_v
        return (lo + hi) / 2

    r_rate = 0.0375

    iv_short = iv_from_mid(mid_short_orig, spx_abrir, K, T_short, r_rate, option_type, fallback=0.17)
    iv_long  = iv_from_mid(mid_long_orig,  spx_abrir, K, T_long,  r_rate, option_type, fallback=0.17)

    # =========================================================================
    # HEADER
    # =========================================================================
    ts = registro.get("Timestamp", "")
    st.markdown(
        f"<div style='background:#0d0d1a; border:1px solid #1a3a5c; border-radius:8px; "
        f"padding:12px 18px; margin-bottom:12px;'>"
        f"<span style='font-size:1.1em; color:#ffc107; font-weight:bold;'>📊 Risk Profile — Calendar {option_type.upper()}</span>"
        f"&nbsp;&nbsp;<span style='color:#888; font-size:0.85em;'>Registrado: {ts}</span>"
        f"<br><span style='color:#aaa; font-size:0.9em;'>Strike: <b style='color:white'>{K:,.0f}</b>"
        f" · Short: <b style='color:#ef9a9a'>{short_fecha}</b>"
        f" · Long: <b style='color:#a5d6a7'>{long_fecha}</b>"
        f" · Contratos: <b style='color:white'>{contracts}</b></span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # =========================================================================
    # CONTROLES EDITABLES
    # =========================================================================
    col_s, col_l, col_spx, col_iv = st.columns([1, 1, 1, 1])

    with col_s:
        mid_short = st.number_input(
            "Mid Short (editable)",
            min_value=0.0,
            value=float(mid_short_orig),
            step=0.05, format="%.2f",
            key=f"rp_mid_short_{ts}_{K}",
            help="Modificá el mid del short para re-simular el P/L"
        )
    with col_l:
        mid_long = st.number_input(
            "Mid Long (editable)",
            min_value=0.0,
            value=float(mid_long_orig),
            step=0.05, format="%.2f",
            key=f"rp_mid_long_{ts}_{K}",
        )
    with col_spx:
        S_input = st.number_input(
            "SPX precio actual",
            min_value=1000.0,
            value=float(S_current),
            step=1.0, format="%.2f",
            key=f"rp_spx_{ts}_{K}",
        )
    with col_iv:
        iv_adj = st.slider(
            "Ajuste IV (%)",
            min_value=-10, max_value=10, value=0, step=1,
            key=f"rp_iv_{ts}_{K}",
            help="Ajuste relativo de IV para escenarios de vol"
        )

    debit = round(mid_long - mid_short, 2)
    iv_s  = max(iv_short * (1 + iv_adj / 100), 0.01)
    iv_l  = max(iv_long  * (1 + iv_adj / 100), 0.01)

    # Débito / Crédito display
    signo_str = "DÉBITO" if debit > 0 else "CRÉDITO"
    color_d   = "#ffb74d" if debit > 0 else "#ef5350"
    col_d1, col_d2, col_d3 = st.columns([1, 1, 2])
    with col_d1:
        st.metric("Net " + signo_str, f"${abs(debit):.2f} / cto")
    with col_d2:
        st.metric("Total", f"${abs(debit) * 100 * contracts:,.2f}")
    with col_d3:
        iv_pct_s = round(iv_s * 100, 1)
        iv_pct_l = round(iv_l * 100, 1)
        st.caption(
            f"IV implícita estimada — Short: **{iv_pct_s}%** · Long: **{iv_pct_l}%** · r: 3.75%"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================================================================
    # GRÁFICA RISK PROFILE
    # =========================================================================
    fig = build_risk_profile_figure(
        S_current=S_input,
        K=K,
        T_short=T_short, T_long=T_long,
        iv_short=iv_s, iv_long=iv_l,
        r=r_rate,
        debit=debit,
        contracts=contracts,
        option_type=option_type,
    )
    st.plotly_chart(fig, use_container_width=True)

    # =========================================================================
    # GRIEGOS AGREGADOS (Price Slices header)
    # =========================================================================
    g = spread_greeks_at_price(
        S_input, K, T_short, T_long, iv_s, iv_l, r_rate, contracts, option_type
    )

    # P/L actual
    pl_now = float(calendar_pl_today(
        np.array([S_input]), K, T_short, T_long, iv_s, iv_l, r_rate, debit, option_type, contracts
    )[0])

    st.markdown(
        "<div style='background:#0d0d1a; border:1px solid #1a3a5c; border-radius:6px; "
        "padding:10px 16px; font-family:monospace; font-size:0.9em;'>"
        "<b style='color:#ffc107;'>▶ Price Slices — Griegos al precio actual</b>"
        "</div>",
        unsafe_allow_html=True,
    )

    col_g1, col_g2, col_g3, col_g4, col_g5 = st.columns(5)
    col_g1.metric("P/L Open", f"${pl_now:,.2f}")
    col_g2.metric("Delta",    f"{g['Delta']:+.2f}")
    col_g3.metric("Gamma",    f"{g['Gamma']:+.4f}")
    col_g4.metric("Theta",    f"{g['Theta']:+.2f}")
    col_g5.metric("Vega",     f"{g['Vega']:+.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================================================================
    # TABLA PRICE SLICES
    # =========================================================================
    with st.expander("📋 Price Slices — Tabla de escenarios", expanded=False):
        df_slices = build_price_slices(
            S_current=S_input,
            K=K,
            T_short=T_short, T_long=T_long,
            iv_short=iv_s, iv_long=iv_l,
            r=r_rate, debit=debit, contracts=contracts,
            option_type=option_type,
        )

        def color_pl(val):
            if isinstance(val, (int, float)):
                if val > 0:
                    return "color: #00c853; font-weight: bold"
                elif val < 0:
                    return "color: #ef5350"
            return ""

        st.dataframe(
            df_slices.style.map(color_pl, subset=["P/L Hoy", "P/L Exp"]),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("---")


# ==============================================================================
# SECCIÓN 2.5 — RISK PROFILE DE CALENDARS REGISTRADOS
# Llamar desde la página principal después del 2.4
# ==============================================================================

def seccion_risk_profile(
    precio_spx_live:  Optional[float] = None,
    mids_refrescados: Optional[dict]  = None,
    idx_registro:     int             = 0,
):
    """
    Renderiza el Risk Profile del registro seleccionado.
    Llamada desde 02 TE Operar.py — el selector de registro y el botón
    de refresco viven en el fichero principal; aquí solo se dibuja el perfil.

    mids_refrescados: dict con 'mid_short', 'mid_long', 'spx_al_refrescar'
                      obtenido desde Schwab. Si es None usa los mids del registro.
    idx_registro    : índice del registro seleccionado en te_calendar_registros.
    """
    registros = st.session_state.get("te_calendar_registros", [])

    if not registros:
        st.info("ℹ️ No hay calendars registrados. Registrá uno en la sección 2.4.")
        return

    idx_sel  = min(idx_registro, len(registros) - 1)
    registro = registros[idx_sel]

    # Si hay mids frescos del broker, inyectarlos en una copia del registro
    # para que render_risk_profile los use como punto de partida
    if mids_refrescados:
        registro = dict(registro)  # copia, no muta el original
        registro['Short Mid'] = mids_refrescados.get('mid_short', registro.get('Short Mid'))
        registro['Long Mid']  = mids_refrescados.get('mid_long',  registro.get('Long Mid'))
        spx_ref = mids_refrescados.get('spx_al_refrescar', precio_spx_live)
    else:
        spx_ref = precio_spx_live

    render_risk_profile(registro, spx_ref)
