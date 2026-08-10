import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from datetime import date

from utils.utils import check_password
from Cartera.core.storage.db import read_trades, read_equity_summary
from Cartera.core.processing.calendar import daily_pnl, monthly_calendar, yearly_summary, yearly_total
from Cartera.core.processing.metrics import tickers_traded
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


MESES_LARGO = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
]

DIAS_SEMANA = ["LUN", "MAR", "MIÉ", "JUE", "VIE", "SÁB"]


# ---------------------------------------------------------------------------
# Gráfico de barras: P&L por mes (ENE-DIC) del año seleccionado
# ---------------------------------------------------------------------------

def render_monthly_bar_chart(yearly_df):
    colors = [
        "#2ECC71" if v > 0 else ("#E74C3C" if v < 0 else "rgba(255,255,255,0.15)")
        for v in yearly_df["pnl"]
    ]

    fig = go.Figure(go.Bar(
        x=yearly_df["month_label"],
        y=yearly_df["pnl"],
        marker_color=colors,
        hovertemplate="%{x}<br>P&amp;L: $%{y:,.2f}<extra></extra>",
        text=[f"${v:,.0f}" if v != 0 else "" for v in yearly_df["pnl"]],
        textposition="outside",
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", tickprefix="$"),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# CSS de las tarjetas del calendario
# ---------------------------------------------------------------------------

CALENDAR_CSS = """
<style>
.day-card {
    border-radius: 8px;
    padding: 8px;
    min-height: 70px;
    background-color: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
}
.day-card.empty {
    background-color: transparent;
    border: none;
}
.day-card.positive {
    background-color: rgba(46,204,113,0.12);
    border: 1px solid rgba(46,204,113,0.35);
}
.day-card.negative {
    background-color: rgba(231,76,60,0.12);
    border: 1px solid rgba(231,76,60,0.35);
}
.day-num {
    font-size: 0.85em;
    opacity: 0.7;
    margin-bottom: 4px;
}
.pnl {
    font-weight: 600;
    font-size: 0.95em;
}
.pnl.positive { color: #2ECC71; }
.pnl.negative { color: #E74C3C; }
.pnl.neutral { color: #999; }
.week-total-card {
    border-radius: 8px;
    padding: 8px;
    min-height: 70px;
    background-color: rgba(255,255,255,0.05);
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.week-total-label {
    font-size: 0.75em;
    opacity: 0.7;
}
.week-total-value {
    font-weight: 700;
    font-size: 1.05em;
}
</style>
"""


def _day_card_html(day, pnl):
    if day is None:
        return "<div class='day-card empty'></div>"

    if pnl is None:
        return f"<div class='day-card'><div class='day-num'>{day}</div></div>"

    if pnl > 0:
        return (
            f"<div class='day-card positive'><div class='day-num'>{day}</div>"
            f"<div class='pnl positive'>+${pnl:,.2f}</div></div>"
        )
    if pnl < 0:
        return (
            f"<div class='day-card negative'><div class='day-num'>{day}</div>"
            f"<div class='pnl negative'>-${abs(pnl):,.2f}</div></div>"
        )
    return (
        f"<div class='day-card'><div class='day-num'>{day}</div>"
        f"<div class='pnl neutral'>$0.00</div></div>"
    )


def _week_total_html(label, total):
    color = "#2ECC71" if total > 0 else ("#E74C3C" if total < 0 else "#999")
    sign = "+" if total > 0 else ""
    return (
        f"<div class='week-total-card'>"
        f"<div class='week-total-label'>{label}</div>"
        f"<div class='week-total-value' style='color:{color}'>{sign}${total:,.2f}</div>"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Calendario - Cartera", page_icon="🗓️", layout="wide")
    st.title("🗓️ Calendario")

    _sync_down()

    if not DB_PATH.exists():
        st.info(
            "Todavía no has importado ninguna operación. Ve a la página "
            "**Actualizar** para traer tu primer reporte de IBKR."
        )
        return

    equity = read_equity_summary(DB_PATH)
    trades = read_trades(DB_PATH)

    if equity.empty:
        st.info(
            "Todavía no hay datos guardados. Ve a la página "
            "**Actualizar** para traer tu primer reporte de IBKR."
        )
        return

    daily = daily_pnl(equity)

    # --- Estado: año/mes actualmente mostrados ---
    today = date.today()
    if "cal_year" not in st.session_state:
        st.session_state.cal_year = today.year
    if "cal_month" not in st.session_state:
        st.session_state.cal_month = today.month

    # --- Navegación de mes + desplegable de año ---
    nav1, nav2, nav3, nav4, nav5 = st.columns([1, 3, 1, 1, 1.3])
    with nav1:
        if st.button("◀", use_container_width=True):
            m, y = st.session_state.cal_month - 1, st.session_state.cal_year
            if m == 0:
                m, y = 12, y - 1
            st.session_state.cal_month, st.session_state.cal_year = m, y
            st.rerun()
    with nav2:
        mes_nombre = MESES_LARGO[st.session_state.cal_month - 1]
        st.markdown(f"### {mes_nombre} {st.session_state.cal_year}")
    with nav3:
        if st.button("▶", use_container_width=True):
            m, y = st.session_state.cal_month + 1, st.session_state.cal_year
            if m == 13:
                m, y = 1, y + 1
            st.session_state.cal_month, st.session_state.cal_year = m, y
            st.rerun()
    with nav4:
        if st.button("Este mes", use_container_width=True):
            st.session_state.cal_month = today.month
            st.session_state.cal_year = today.year
            st.rerun()
    with nav5:
        available_years = sorted(daily["reportDate"].dt.year.unique(), reverse=True) if not daily.empty else [today.year]
        if st.session_state.cal_year not in available_years:
            available_years = sorted(set(available_years + [st.session_state.cal_year]), reverse=True)
        year_index = available_years.index(st.session_state.cal_year)
        picked_year = st.selectbox("Ir a año", options=available_years, index=year_index, key="cal_year_picker")
        if picked_year != st.session_state.cal_year:
            st.session_state.cal_year = picked_year
            st.rerun()

    year = st.session_state.cal_year
    month = st.session_state.cal_month

    # --- Resumen anual: gráfico de barras ENE..DIC (del año seleccionado) ---
    ys = yearly_summary(daily, year)
    total_year = yearly_total(daily, year)

    color_total = "green" if total_year > 0 else ("red" if total_year < 0 else "gray")
    sign_total = "+" if total_year > 0 else ""
    st.subheader(f"Resumen anual {year}")
    st.markdown(f"**Total del año:** :{color_total}[{sign_total}${total_year:,.2f}]")
    st.plotly_chart(render_monthly_bar_chart(ys), use_container_width=True)

    st.markdown("---")

    mc = monthly_calendar(daily, year, month)
    month_total = mc["month_total"]
    color = "green" if month_total > 0 else ("red" if month_total < 0 else "gray")
    sign = "+" if month_total > 0 else ""
    st.markdown(f"**Total del mes:** :{color}[{sign}${month_total:,.2f}]")

    st.markdown(CALENDAR_CSS, unsafe_allow_html=True)

    main_col, side_col = st.columns([3, 1])

    with main_col:
        # Cabecera de días
        header_cols = st.columns(7)
        for i, dia in enumerate(DIAS_SEMANA):
            header_cols[i].markdown(f"**{dia}**")
        header_cols[6].markdown("**SEM**")

        # Filas de semanas
        for week in mc["weeks"]:
            row_cols = st.columns(7)
            for i, d in enumerate(week["days"]):
                row_cols[i].markdown(_day_card_html(d["day"], d["pnl"]), unsafe_allow_html=True)
            row_cols[6].markdown(
                _week_total_html(week["week_label"], week["week_total"]),
                unsafe_allow_html=True,
            )

    with side_col:
        st.markdown("#### Tickers operados")
        tickers_df = tickers_traded(trades, year=year, month=month) if not trades.empty else None
        if tickers_df is None or tickers_df.empty:
            st.caption("Sin operaciones este mes")
        else:
            st.dataframe(
                tickers_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "underlyingSymbol": "Ticker",
                    "n_trades": "Ops",
                    "net_pnl": st.column_config.NumberColumn("P&L", format="$%.2f"),
                },
            )


if __name__ == "__main__":
    if check_password():
        main()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder.")
