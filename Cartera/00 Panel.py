import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from datetime import date

from utils.utils import check_password
from Cartera.core.storage.db import read_trades, read_equity_summary, read_open_positions
from Cartera.core.processing.metrics import (
    filter_closed_trades,
    profit_factor,
    win_rate,
    pnl_summary,
    account_summary,
)
from Cartera.core.storage.github_sync import download_db

DB_PATH = Path(__file__).parent / "data" / "processed" / "cartera.db"
REMOTE_DB_PATH = "cartera.db"


def _sync_down():
    """Trae la última versión de la BD desde GitHub antes de leer, por si
    el contenedor de Streamlit se reinició desde la última visita."""
    gh = st.secrets.get("github_data")
    if not gh:
        return
    token, repo, branch = gh.get("token"), gh.get("repo"), gh.get("branch", "main")
    if not token or not repo:
        return
    try:
        download_db(DB_PATH, REMOTE_DB_PATH, token, repo, branch)
    except Exception:
        pass  # si falla, seguimos con lo que haya en local sin romper la página


# ---------------------------------------------------------------------------
# Gráfico lineal de evolución del NLV
# ---------------------------------------------------------------------------

def render_nlv_line_chart(equity_df):
    df = equity_df.sort_values("reportDate")

    is_up = df["total"].iloc[-1] >= df["total"].iloc[0]
    line_color = "#2ECC71" if is_up else "#E74C3C"
    fill_color = "rgba(46,204,113,0.10)" if is_up else "rgba(231,76,60,0.10)"

    fig = go.Figure(go.Scatter(
        x=df["reportDate"],
        y=df["total"],
        mode="lines",
        line=dict(color=line_color, width=2),
        fill="tozeroy",
        fillcolor=fill_color,
        hovertemplate="%{x|%d/%m/%Y}<br>NLV: $%{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", tickprefix="$"),
    )
    return fig


# ---------------------------------------------------------------------------
# Gauges (Plotly)
# ---------------------------------------------------------------------------

def render_profit_factor_gauge(pf_value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pf_value if pf_value is not None else 0.0,
        number={"valueformat": ".2f", "font": {"size": 40}},
        gauge={
            "axis": {"range": [0, 3], "tickwidth": 1},
            "bar": {"color": "#2ECC71" if (pf_value or 0) >= 1 else "#E74C3C"},
            "steps": [
                {"range": [0, 1], "color": "rgba(231,76,60,0.25)"},
                {"range": [1, 1.5], "color": "rgba(241,196,15,0.25)"},
                {"range": [1.5, 3], "color": "rgba(46,204,113,0.25)"},
            ],
        },
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    return fig


def render_win_rate_gauge(win_rate_pct):
    display_value = 0.0 if win_rate_pct is None else win_rate_pct

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=display_value,
        number={"suffix": "%", "valueformat": ".1f", "font": {"size": 40}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "#2ECC71" if display_value >= 50 else "#E74C3C"},
            "steps": [
                {"range": [0, 40], "color": "rgba(231,76,60,0.25)"},
                {"range": [40, 60], "color": "rgba(241,196,15,0.25)"},
                {"range": [60, 100], "color": "rgba(46,204,113,0.25)"},
            ],
        },
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    return fig


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Panel - Cartera", page_icon="📊", layout="wide")
    st.title("📊 Panel")

    _sync_down()

    if not DB_PATH.exists():
        st.info(
            "Todavía no has importado ninguna operación. Ve a la página "
            "**Actualizar** para traer tu primer reporte de IBKR."
        )
        return

    trades = read_trades(DB_PATH)
    equity = read_equity_summary(DB_PATH)

    if trades.empty and equity.empty:
        st.info(
            "Todavía no hay datos guardados. Ve a la página "
            "**Actualizar** para traer tu primer reporte de IBKR."
        )
        return

    today = date.today()

    # --- Selector de año (afecta a P&L; el gráfico de NLV es siempre histórico completo) ---
    available_years = sorted(equity["reportDate"].dt.year.unique(), reverse=True) if not equity.empty else [today.year]
    default_index = available_years.index(today.year) if today.year in available_years else 0
    selected_year = st.selectbox("Año", options=available_years, index=default_index, key="panel_year")

    # --- Resumen de cuenta (siempre el estado ACTUAL, no depende del selector) ---
    acc = account_summary(equity) if not equity.empty else None

    # --- Operaciones cerradas del año seleccionado ---
    closed_selected = filter_closed_trades(
        trades, date_from=date(selected_year, 1, 1), date_to=date(selected_year, 12, 31)
    ) if not trades.empty else trades

    pf = profit_factor(closed_selected)
    wr = win_rate(closed_selected)
    pnl = pnl_summary(closed_selected)

    st.caption(f"Resumen de cuenta a fecha de hoy · P&L del año {selected_year}")
    st.markdown("---")

    # --- Fila de tarjetas: Resumen de cuenta ---
    st.subheader("Resumen de cuenta")
    col1, col2, col3, col4 = st.columns(4)
    if acc:
        col1.metric("NLV", f"${acc['nlv']:,.2f}")
        col2.metric("Caja", f"${acc['cash']:,.2f}")
        col3.metric("Invertido", f"${acc['invested']:,.2f}")
        ytd = acc["ytd_return_pct"]
        col4.metric("Rentab. YTD", f"{ytd:.2f}%" if ytd is not None else "—")
    else:
        for c in (col1, col2, col3, col4):
            c.metric("—", "—")

    st.markdown("---")

    # --- Gráfico lineal: evolución del valor de cartera (NLV, histórico completo) ---
    st.subheader("Evolución del valor de la cartera (NLV)")
    if equity.empty:
        st.caption("Sin datos suficientes para el gráfico")
    else:
        st.plotly_chart(render_nlv_line_chart(equity), use_container_width=True)

    st.markdown("---")

    # --- Fila: Gauges + P&L acumulado (del año seleccionado) ---
    gcol1, gcol2, gcol3 = st.columns(3)

    with gcol1:
        st.subheader("Factor de beneficio")
        st.plotly_chart(render_profit_factor_gauge(pf), use_container_width=True)
        sub1, sub2 = st.columns(2)
        sub1.markdown(f":green[**+${pnl['gross_profit']:,.0f}**]")
        sub2.markdown(f":red[**-${abs(pnl['gross_loss']):,.0f}**]")

    with gcol2:
        st.subheader("Tasa de acierto")
        st.plotly_chart(render_win_rate_gauge(wr["win_rate_pct"]), use_container_width=True)
        sub1, sub2 = st.columns(2)
        sub1.markdown(f":green[**{wr['wins']}W**]")
        sub2.markdown(f":red[**{wr['losses']}L**]")

    with gcol3:
        st.subheader(f"P&L acumulado {selected_year}")
        net = pnl["net_pnl"]
        color = "green" if net >= 0 else "red"
        sign = "+" if net >= 0 else ""
        st.markdown(f"### :{color}[{sign}${net:,.2f}]")
        if closed_selected.empty:
            st.caption("Sin datos para este período")
        else:
            st.caption(f"{wr['total_closed']} operaciones cerradas")

    st.markdown("---")

    # --- Crédito abierto (siempre estado actual) ---
    st.subheader("Crédito abierto")
    positions = read_open_positions(DB_PATH)
    if positions.empty:
        st.caption("Sin posiciones abiertas")
    else:
        cols_to_show = [
            c for c in [
                "symbol", "underlyingSymbol", "assetCategory", "putCall",
                "strike", "expiry", "position", "markPrice", "positionValue",
                "fifoPnlUnrealized",
            ] if c in positions.columns
        ]
        st.dataframe(positions[cols_to_show], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    if check_password():
        main()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder.")
