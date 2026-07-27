import streamlit as st
from pathlib import Path
from datetime import date, timedelta

from utils.utils import check_password
from Cartera.core.storage.db import read_trades
from Cartera.core.processing.metrics import filter_closed_trades, pnl_summary, win_rate

DB_PATH = Path(__file__).parent / "data" / "processed" / "cartera.db"


def main():
    st.set_page_config(page_title="Operaciones - Cartera", page_icon="📋", layout="wide")
    st.title("📋 Operaciones")

    if not DB_PATH.exists():
        st.info(
            "Todavía no has importado ninguna operación. Ve a la página "
            "**Importar** para subir tu primer Flex Query de IBKR."
        )
        return

    trades = read_trades(DB_PATH)

    if trades.empty:
        st.info(
            "Todavía no hay operaciones guardadas. Ve a la página "
            "**Importar** para subir tu primer Flex Query de IBKR."
        )
        return

    # -----------------------------------------------------------------
    # Filtros
    # -----------------------------------------------------------------
    with st.container(border=True):
        f1, f2, f3, f4 = st.columns([2, 2, 2, 1])

        min_date = trades["tradeDate"].min().date()
        max_date = trades["tradeDate"].max().date()

        with f1:
            date_range = st.date_input(
                "Rango de fechas",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )

        tickers_disponibles = sorted(trades["underlyingSymbol"].dropna().unique().tolist())
        with f2:
            tickers_sel = st.multiselect("Ticker", options=tickers_disponibles)

        categorias_disponibles = sorted(trades["assetCategory"].dropna().unique().tolist())
        with f3:
            categorias_sel = st.multiselect("Tipo de activo", options=categorias_disponibles)

        with f4:
            vista = st.radio("Vista", options=["P&L", "Operaciones"], horizontal=False)

    # Aplicar filtros
    df = trades.copy()

    if isinstance(date_range, tuple) and len(date_range) == 2:
        date_from, date_to = date_range
        df = df[
            (df["tradeDate"].dt.date >= date_from) & (df["tradeDate"].dt.date <= date_to)
        ]

    if tickers_sel:
        df = df[df["underlyingSymbol"].isin(tickers_sel)]

    if categorias_sel:
        df = df[df["assetCategory"].isin(categorias_sel)]

    st.markdown("---")

    # -----------------------------------------------------------------
    # Vista P&L (solo cierres, con métricas)
    # -----------------------------------------------------------------
    if vista == "P&L":
        closed = filter_closed_trades(df)

        pnl = pnl_summary(closed)
        wr = win_rate(closed)

        m1, m2, m3, m4 = st.columns(4)
        net = pnl["net_pnl"]
        m1.metric("P&L neto", f"${net:,.2f}")
        m2.metric("Ganancias brutas", f"${pnl['gross_profit']:,.2f}")
        m3.metric("Pérdidas brutas", f"${pnl['gross_loss']:,.2f}")
        m4.metric(
            "Tasa de acierto",
            f"{wr['win_rate_pct']:.1f}%" if wr["win_rate_pct"] is not None else "—",
            help=f"{wr['wins']}W / {wr['losses']}L",
        )

        st.markdown("---")

        if closed.empty:
            st.caption("Sin operaciones cerradas para este filtro")
        else:
            cols = [
                "tradeDate", "underlyingSymbol", "assetCategory", "putCall",
                "strike", "expiry", "buySell", "quantity", "tradePrice",
                "ibCommission", "fifoPnlRealized",
            ]
            cols = [c for c in cols if c in closed.columns]
            closed_sorted = closed.sort_values("tradeDate", ascending=False)

            st.dataframe(
                closed_sorted[cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "tradeDate": st.column_config.DateColumn("Fecha", format="DD/MM/YYYY"),
                    "underlyingSymbol": "Subyacente",
                    "assetCategory": "Tipo",
                    "putCall": "Put/Call",
                    "strike": st.column_config.NumberColumn("Strike", format="%.2f"),
                    "expiry": st.column_config.DateColumn("Vencimiento", format="DD/MM/YYYY"),
                    "buySell": "Compra/Venta",
                    "quantity": "Cantidad",
                    "tradePrice": st.column_config.NumberColumn("Precio", format="%.2f"),
                    "ibCommission": st.column_config.NumberColumn("Comisión", format="%.2f"),
                    "fifoPnlRealized": st.column_config.NumberColumn("P&L", format="$%.2f"),
                },
            )

    # -----------------------------------------------------------------
    # Vista Operaciones (todas las ejecuciones, aperturas y cierres)
    # -----------------------------------------------------------------
    else:
        st.caption(f"{len(df)} operaciones (aperturas y cierres)")

        cols = [
            "tradeDate", "underlyingSymbol", "assetCategory", "putCall",
            "strike", "expiry", "buySell", "openCloseIndicator", "quantity",
            "tradePrice", "ibCommission", "fifoPnlRealized",
        ]
        cols = [c for c in cols if c in df.columns]
        df_sorted = df.sort_values("tradeDate", ascending=False)

        st.dataframe(
            df_sorted[cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "tradeDate": st.column_config.DateColumn("Fecha", format="DD/MM/YYYY"),
                "underlyingSymbol": "Subyacente",
                "assetCategory": "Tipo",
                "putCall": "Put/Call",
                "strike": st.column_config.NumberColumn("Strike", format="%.2f"),
                "expiry": st.column_config.DateColumn("Vencimiento", format="DD/MM/YYYY"),
                "buySell": "Compra/Venta",
                "openCloseIndicator": "Apert./Cierre",
                "quantity": "Cantidad",
                "tradePrice": st.column_config.NumberColumn("Precio", format="%.2f"),
                "ibCommission": st.column_config.NumberColumn("Comisión", format="%.2f"),
                "fifoPnlRealized": st.column_config.NumberColumn("P&L", format="$%.2f"),
            },
        )

    # -----------------------------------------------------------------
    # Descarga CSV del resultado filtrado
    # -----------------------------------------------------------------
    st.download_button(
        "⬇️ Descargar CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="operaciones.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    if check_password():
        main()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder.")
