import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from datetime import date

from utils.utils import check_password
from Dividendos.core.storage.db import read_trades, read_equity_summary, read_open_positions, read_cash_transactions
from Dividendos.core.processing.metrics import (
    filter_closed_trades,
    profit_factor,
    win_rate,
    pnl_summary,
    account_summary,
    dividends_by_month,
    dividends_total,
    account_growth_by_month,
)

DB_PATH = Path(__file__).parent / "data" / "processed" / "dividendos.db"


# ---------------------------------------------------------------------------
# Gráfico de barras: dividendos netos por mes + línea de media
# ---------------------------------------------------------------------------

def render_dividends_chart(monthly_df):
    avg = float(monthly_df["avg_net"].iloc[0]) if not monthly_df.empty else 0.0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly_df["month_label"],
        y=monthly_df["net"],
        marker_color="#3498DB",
        hovertemplate="%{x}<br>Dividendo neto: $%{y:,.2f}<extra></extra>",
        text=[f"${v:,.0f}" if v != 0 else "" for v in monthly_df["net"]],
        textposition="outside",
        name="Dividendo neto",
    ))
    fig.add_trace(go.Scatter(
        x=monthly_df["month_label"],
        y=[avg] * len(monthly_df),
        mode="lines",
        line=dict(color="#F39C12", width=2, dash="dash"),
        hovertemplate=f"Media: ${avg:,.2f}<extra></extra>",
        name="Media mensual",
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", tickprefix="$"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Panel - Dividendos", page_icon="💰", layout="wide")
    st.title("💰 Panel de Dividendos")

    if not DB_PATH.exists():
        st.info(
            "Todavía no has importado ninguna operación. Ve a la página "
            "**Importar** para traer tu primer reporte de IBKR."
        )
        return

    trades = read_trades(DB_PATH)
    equity = read_equity_summary(DB_PATH)
    cash_tx = read_cash_transactions(DB_PATH)

    if equity.empty:
        st.info(
            "Todavía no hay datos guardados. Ve a la página "
            "**Importar** para traer tu primer reporte de IBKR."
        )
        return

    today = date.today()
    year = today.year

    # --- Resumen de cuenta ---
    acc = account_summary(equity)

    st.subheader("Resumen de cuenta")
    col1, col2, col3, col4 = st.columns(4)
    if acc:
        col1.metric("NLV", f"${acc['nlv']:,.2f}")
        col2.metric("Caja", f"${acc['cash']:,.2f}")
        col3.metric("Invertido", f"${acc['invested']:,.2f}")
        ytd = acc["ytd_return_pct"]
        col4.metric("Rentab. YTD", f"{ytd:.2f}%" if ytd is not None else "—")

    st.markdown("---")

    # --- P&L (trading, si lo hay) ---
    closed_ytd = filter_closed_trades(
        trades, date_from=date(year, 1, 1), date_to=today
    ) if not trades.empty else trades
    pnl = pnl_summary(closed_ytd)
    wr = win_rate(closed_ytd)

    st.subheader("P&L de trading (año en curso)")
    p1, p2, p3, p4 = st.columns(4)
    net = pnl["net_pnl"]
    color = "green" if net >= 0 else "red"
    sign = "+" if net >= 0 else ""
    p1.markdown(f"**P&L neto**")
    p1.markdown(f"### :{color}[{sign}${net:,.2f}]")
    p2.metric("Ganancias brutas", f"${pnl['gross_profit']:,.2f}")
    p3.metric("Pérdidas brutas", f"${pnl['gross_loss']:,.2f}")
    p4.metric(
        "Tasa de acierto",
        f"{wr['win_rate_pct']:.1f}%" if wr["win_rate_pct"] is not None else "—",
        help=f"{wr['wins']}W / {wr['losses']}L",
    )

    st.markdown("---")

    # --- Tarjetas de crecimiento mensual de la cuenta ---
    st.subheader(f"Crecimiento mensual de la cuenta {year}")
    growth = account_growth_by_month(equity, year)
    months_with_data = growth[growth["end_nlv"].notna()]

    if months_with_data.empty:
        st.caption("Sin datos suficientes todavía")
    else:
        cards_per_row = 6
        rows_needed = [months_with_data[i:i + cards_per_row] for i in range(0, len(months_with_data), cards_per_row)]
        for chunk in rows_needed:
            cols = st.columns(len(chunk))
            for col, (_, row) in zip(cols, chunk.iterrows()):
                growth_pct = row["growth_pct"]
                color = "green" if (growth_pct or 0) >= 0 else "red"
                sign = "+" if (growth_pct or 0) >= 0 else ""
                with col:
                    st.markdown(f"**{row['month_label']}**")
                    st.markdown(f":{color}[{sign}{growth_pct:.2f}%]" if growth_pct is not None else "—")
                    st.caption(f"${row['growth_abs']:,.0f}" if row["growth_abs"] is not None else "")

    st.markdown("---")

    # --- Dividendos mensuales + media ---
    st.subheader(f"Dividendos mensuales {year}")
    div_monthly = dividends_by_month(cash_tx, year)
    div_total = dividends_total(cash_tx, year)

    d1, d2, d3 = st.columns(3)
    d1.metric("Dividendo bruto (año)", f"${div_total['gross']:,.2f}")
    d2.metric("Retenciones (año)", f"${div_total['withholding_tax']:,.2f}")
    d3.metric("Dividendo neto (año)", f"${div_total['net']:,.2f}")

    st.plotly_chart(render_dividends_chart(div_monthly), use_container_width=True)

    st.markdown("---")

    # --- Instrumentos / holdings ---
    st.subheader("Instrumentos y acciones que tienes")
    positions = read_open_positions(DB_PATH)
    if positions.empty:
        st.caption("Sin posiciones abiertas")
    else:
        cols_to_show = [
            c for c in [
                "symbol", "assetCategory", "position", "markPrice",
                "positionValue", "percentOfNAV", "fifoPnlUnrealized",
            ] if c in positions.columns
        ]
        st.dataframe(
            positions[cols_to_show].sort_values("positionValue", ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={
                "symbol": "Ticker",
                "assetCategory": "Tipo",
                "position": "Acciones",
                "markPrice": st.column_config.NumberColumn("Precio", format="$%.2f"),
                "positionValue": st.column_config.NumberColumn("Valor", format="$%.2f"),
                "percentOfNAV": st.column_config.NumberColumn("% NLV", format="%.2f%%"),
                "fifoPnlUnrealized": st.column_config.NumberColumn("P&L no realizado", format="$%.2f"),
            },
        )


if __name__ == "__main__":
    if check_password():
        main()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder.")
