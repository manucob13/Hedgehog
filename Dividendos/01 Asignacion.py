import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

from utils.utils import check_password
from Dividendos.core.storage.db import (
    read_open_positions,
    read_equity_summary,
    read_target_allocation,
    save_target_allocation,
)
from Dividendos.core.processing.allocation import compare_allocation

DB_PATH = Path(__file__).parent / "data" / "processed" / "dividendos.db"

# Datos iniciales (tu tabla real). Solo se usan para precargar el editor
# la primera vez, si todavía no hay nada guardado en la base de datos.
DEFAULT_ALLOCATION = pd.DataFrame([
    ("SCHD", "Núcleo calidad USA (dividendos crecientes)", 17.37),
    ("JEPQ", "Covered calls Nasdaq (yield alto)", 8.76),
    ("VYM", "Alto dividendo diversificado USA", 8.22),
    ("SPYI", "Covered calls SPY (yield alto)", 7.79),
    ("O", "REIT defensivo, \"Monthly Dividend Company\"", 6.82),
    ("IDVO", "Internacional + covered calls", 6.61),
    ("VYMI", "ETF International High Dividend Yield ETF", 6.13),
    ("SMDV", "SmallCap Dividend Aristocrats", 6.13),
    ("UTG", "Utilities defensivo, yield alto", 5.64),
    ("BIT", "Renta Fija (Descuento)", 4.48),
    ("PDI", "Renta Fija (Max Yield)", 4.48),
    ("XLV", "Healthcare defensivo", 3.88),
    ("BALI", "Commodities, diversificación", 3.17),
    ("AMDW", "AMD Covered Call Strategy ETF", 2.92),
    ("VNQ", "REIT diversificado USA", 2.10),
    ("LTC", "REIT de salud con dividendos mensuales", 2.00),
    ("UTF", "Infraestructura global con dividendos mensuales y yield elevado", 2.00),
    ("CASH", "Liquidez", 1.50),
], columns=["ticker", "description", "target_pct"])


# ---------------------------------------------------------------------------
# Gráfico de desviaciones: current_pct - target_pct por ticker
# ---------------------------------------------------------------------------

def render_deviation_chart(result_df):
    df = result_df.copy()
    df["deviation_pct"] = df["current_pct"] - df["target_pct"]
    df = df.sort_values("deviation_pct")

    colors = ["#E74C3C" if v > 0 else "#3498DB" for v in df["deviation_pct"]]

    def _detail_label(row):
        """Acciones de diferencia si aplica (no-CASH y precio conocido);
        si no, muestra la diferencia en $ (caso CASH o sin precio)."""
        if row["ticker"] == "CASH" or pd.isna(row["shares_diff"]):
            return f"{row['diff_value']:+,.0f} $"
        return f"{row['shares_diff']:+.1f} acciones"

    detail_labels = df.apply(_detail_label, axis=1)
    bar_text = [
        f"{pp:+.2f} pp ({detail})"
        for pp, detail in zip(df["deviation_pct"], detail_labels)
    ]

    fig = go.Figure(go.Bar(
        y=df["ticker"],
        x=df["deviation_pct"],
        orientation="h",
        marker_color=colors,
        customdata=detail_labels,
        hovertemplate="%{y}<br>Desviación: %{x:+.2f} pp<br>%{customdata}<extra></extra>",
        text=bar_text,
        textposition="outside",
        cliponaxis=False,  # evita que Plotly recorte el texto en el borde del eje
    ))

    # Ampliamos el rango del eje X más allá del valor máximo/mínimo real,
    # para dejar hueco visual donde quepa el texto "outside" de las barras
    # más largas (si no, ese texto se corta justo en el borde del gráfico).
    max_abs = max(abs(df["deviation_pct"].max()), abs(df["deviation_pct"].min()), 0.1)
    x_range = [-max_abs * 1.9, max_abs * 1.9]

    fig.update_layout(
        height=max(300, 32 * len(df)),
        margin=dict(l=10, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        xaxis=dict(
            showgrid=True, gridcolor="rgba(255,255,255,0.08)",
            zeroline=True, zerolinecolor="rgba(255,255,255,0.3)",
            title="Desviación (puntos porcentuales) — azul = por debajo, rojo = por encima",
            range=x_range,
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

    # -----------------------------------------------------------------
    # Editor de la tabla de asignación objetivo (siempre disponible)
    # -----------------------------------------------------------------
    target = read_target_allocation(DB_PATH) if DB_PATH.exists() else pd.DataFrame()
    using_defaults = target.empty
    if using_defaults:
        target = DEFAULT_ALLOCATION.copy()

    st.subheader("✏️ Tabla de asignación objetivo")
    if using_defaults:
        st.info(
            "Todavía no has guardado ninguna tabla: te muestro tu tabla inicial. "
            "Edítala si quieres y pulsa **Guardar cambios**."
        )
    st.caption(
        "Modifica los % objetivo (o añade/quita filas) y pulsa **Guardar cambios**. "
        "Usa el ticker especial **CASH** para fijar el % objetivo de liquidez."
    )

    edit_cols = ["ticker", "description", "target_pct"]
    editable_source = target[edit_cols] if set(edit_cols).issubset(target.columns) else target

    edited = st.data_editor(
        editable_source,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", required=True),
            "description": st.column_config.TextColumn("Descripción", width="large"),
            "target_pct": st.column_config.NumberColumn(
                "% objetivo", format="%.2f%%", min_value=0.0, max_value=100.0, required=True
            ),
        },
        key="allocation_editor",
    )

    total_pct = edited["target_pct"].sum() if not edited.empty else 0.0
    color = "green" if abs(total_pct - 100) < 0.5 else "orange"
    st.markdown(f"**Suma total:** :{color}[{total_pct:.2f}%] (idealmente 100%)")

    if st.button("💾 Guardar cambios", type="primary"):
        try:
            n = save_target_allocation(DB_PATH, edited)
            st.success(f"✅ Guardados {n} tickers.")
            st.rerun()
        except Exception as e:
            st.error(f"❌ No se pudo guardar: {e}")

    st.markdown("---")

    # -----------------------------------------------------------------
    # Comparación con holdings reales (solo si ya hay datos importados)
    # -----------------------------------------------------------------
    if not DB_PATH.exists():
        st.info(
            "Todavía no has importado ninguna operación. Ve a la página "
            "**Importar** para traer tu primer reporte de IBKR y ver aquí "
            "la comparación contra tus holdings reales."
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

    # Usamos siempre la tabla ya guardada en BD para la comparación (no la
    # versión sin guardar del editor, para evitar confusión).
    saved_target = read_target_allocation(DB_PATH)
    if saved_target.empty:
        st.info("Guarda tu tabla de asignación objetivo arriba para ver la comparación.")
        return

    result = compare_allocation(positions, saved_target, nlv, cash_value=cash)

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


if __name__ == "__main__":
    if check_password():
        main()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder.")
