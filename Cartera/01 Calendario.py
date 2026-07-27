import streamlit as st
from pathlib import Path
from datetime import date

from utils.utils import check_password
from Cartera.core.storage.db import read_trades, read_equity_summary
from Cartera.core.processing.calendar import daily_pnl, monthly_calendar, yearly_summary, yearly_total
from Cartera.core.processing.metrics import tickers_traded

DB_PATH = Path(__file__).parent / "data" / "processed" / "cartera.db"

MESES_LARGO = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
]

DIAS_SEMANA = ["LUN", "MAR", "MIÉ", "JUE", "VIE", "SÁB"]


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

    if not DB_PATH.exists():
        st.info(
            "Todavía no has importado ninguna operación. Ve a la página "
            "**Importar** para subir tu primer Flex Query de IBKR."
        )
        return

    equity = read_equity_summary(DB_PATH)
    trades = read_trades(DB_PATH)

    if equity.empty:
        st.info(
            "Todavía no hay datos guardados. Ve a la página "
            "**Importar** para subir tu primer Flex Query de IBKR."
        )
        return

    daily = daily_pnl(equity)

    # --- Estado: año/mes actualmente mostrados ---
    today = date.today()
    if "cal_year" not in st.session_state:
        st.session_state.cal_year = today.year
    if "cal_month" not in st.session_state:
        st.session_state.cal_month = today.month

    # --- Fila resumen anual (ENE..DIC + TOTAL) ---
    st.subheader(f"Resumen anual {st.session_state.cal_year}")
    ys = yearly_summary(daily, st.session_state.cal_year)
    total_year = yearly_total(daily, st.session_state.cal_year)

    year_cols = st.columns(13)
    for i, row in ys.iterrows():
        with year_cols[i]:
            color = "green" if row["pnl"] > 0 else ("red" if row["pnl"] < 0 else "gray")
            st.caption(row["month_label"])
            st.markdown(f":{color}[**${row['pnl']:,.0f}**]")
    with year_cols[12]:
        color = "green" if total_year > 0 else ("red" if total_year < 0 else "gray")
        st.caption("TOTAL")
        st.markdown(f":{color}[**${total_year:,.0f}**]")

    st.markdown("---")

    # --- Navegación de mes ---
    nav1, nav2, nav3, nav4 = st.columns([1, 3, 1, 1])
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

    year = st.session_state.cal_year
    month = st.session_state.cal_month

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
