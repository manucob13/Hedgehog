import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

from utils.utils import check_password
from Dividendos.core.storage.db import read_open_positions, read_equity_summary, read_target_allocation
from Dividendos.core.processing.allocation import compare_allocation

DB_PATH = Path(__file__).parent / "data" / "processed" / "dividendos.db"


# ---------------------------------------------------------------------------
# Gráfico de desviaciones: current_pct - target_pct por ticker
# ---------------------------------------------------------------------------

def render_deviation_chart(result_df):
    df = result_df.copy()
    df["deviation_pct"] = df["current_pct"] - df["target_pct"]
    df = df.sort_values("deviation_pct")

    colors = ["#E74C3C" if v > 0 else "#3498DB" for v in df["deviation_pct"]]

    fig = go.Figure(go.Bar(
        y=df["ticker"],
        x=df["deviation_pct"],
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}<br>Desviación: %{x:+.2f} pp<extra></extra>",
        text=[f"{v:+.2f} pp" for v in df["deviation_pct"]],
        textposition="outside",
    ))
    fig.update_layout(
        height=max(300, 28 * len(df)),
        margin=dict(l=10, r=40, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        xaxis=dict(
            showgrid=True, gridcolor="rgba(255,255,255,0.08)",
            zeroline=True, zerolinecolor="rgba(255,255,255,0.3)",
            title="Desviación (puntos porcentuales) — azul = por debajo, rojo = por encima",
        ),
        yaxis=dict(showgrid=False),
    )
    return fig


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Asignación - Dividendos", page_icon="🎯", layout="wide")
    st.title("🎯 Asignación: holdings vs. objetivo")

    if not DB_PATH.exists():
        st.info(
            "Todavía no has importado ninguna operación. Ve a la página "
            "**Importar** para traer tu primer reporte de IBKR."
        )
        return

    target = read_target_allocation(DB_PATH)
    if target.empty:
        st.warning(
            "Todavía no has subido tu tabla de asignación objetivo. "
            "Ve a la página **Importar** y súbela (columnas Ticker + Target_%)."
        )
        return

    positions = read_open_positions(DB_PATH)
    equity = read_equity_summary(DB_PATH)

    if equity.empty:
        st.info("Todavía no hay datos de cuenta guardados.")
        return

    last_row = equity.sort_values("reportDate").iloc[-1]
    nlv = float(last_row["total"])
    cash = float(last_row["cash"])

    result = compare_allocation(positions, target, nlv, cash_value=cash)

    if result.empty:
        st.info("No hay datos suficientes para comparar la asignación.")
        return

    # --- Resumen rápido ---
    n_por_debajo = int((result["status"] == "Por debajo").sum())
    n_por_encima = int((result["status"] == "Por encima").sum())
    n_en_objetivo = int((result["status"] == "En objetivo").sum())
    n_sin_objetivo = int((result["status"] == "Sin objetivo definido").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("En objetivo", n_en_objetivo)
    c2.metric("Por debajo", n_por_debajo)
    c3.metric("Por encima", n_por_encima)
    c4.metric("Sin objetivo definido", n_sin_objetivo)

    st.markdown("---")

    # --- Gráfico de desviaciones ---
    st.subheader("Desviación respecto al % objetivo")
    chart_df = result[result["target_pct"] > 0]  # solo tickers con objetivo definido
    if not chart_df.empty:
        st.plotly_chart(render_deviation_chart(chart_df), use_container_width=True)
    else:
        st.caption("Sin tickers con objetivo definido para graficar")

    st.markdown("---")

    # --- Tabla detallada ---
    st.subheader("Detalle por instrumento")

    display_df = result.copy()
    display_df["deviation_pct"] = display_df["current_pct"] - display_df["target_pct"]

    cols = [
        "ticker", "shares_held", "price", "current_value", "current_pct",
        "target_pct", "deviation_pct", "diff_value", "shares_diff", "status",
    ]

    def _highlight_status(val):
        colors = {
            "Por debajo": "color: #3498DB",
            "Por encima": "color: #E74C3C",
            "En objetivo": "color: #2ECC71",
            "Sin objetivo definido": "color: #999999",
        }
        return colors.get(val, "")

    styled = display_df[cols].style.map(_highlight_status, subset=["status"])

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ticker": "Ticker",
            "shares_held": st.column_config.NumberColumn("Acciones", format="%.2f"),
            "price": st.column_config.NumberColumn("Precio", format="$%.2f"),
            "current_value": st.column_config.NumberColumn("Valor actual", format="$%.2f"),
            "current_pct": st.column_config.NumberColumn("% actual", format="%.2f%%"),
            "target_pct": st.column_config.NumberColumn("% objetivo", format="%.2f%%"),
            "deviation_pct": st.column_config.NumberColumn("Desviación (pp)", format="%+.2f"),
            "diff_value": st.column_config.NumberColumn("Diferencia $", format="$%+.2f"),
            "shares_diff": st.column_config.NumberColumn("Acciones de diferencia", format="%+.2f"),
            "status": "Estado",
        },
    )

    st.caption(
        "**Acciones de diferencia**: positivo = te faltan esas acciones para llegar "
        "al objetivo; negativo = tienes ese exceso sobre el objetivo. Para la fila "
        "CASH no aplica el concepto de acciones, usa la columna 'Diferencia $'."
    )

    with st.expander("📋 Tu tabla de asignación objetivo actual"):
        st.dataframe(
            target,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ticker": "Ticker",
                "target_pct": st.column_config.NumberColumn("% objetivo", format="%.2f%%"),
                "updated_at": "Actualizado",
            },
        )


if __name__ == "__main__":
    if check_password():
        main()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder.")
