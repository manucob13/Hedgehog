import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, datetime
import calendar as _calendar_module

from utils.utils import check_password
from Dividendos.core.storage.db import (
    read_trades,
    read_equity_summary,
    read_open_positions,
    read_cash_transactions,
    read_target_allocation,
    read_import_log,
    save_flex_import,
)
from Dividendos.core.processing.metrics import (
    filter_closed_trades,
    pnl_summary,
    win_rate,
    account_summary,
    dividends_by_month,
    dividends_total,
    account_growth_by_month,
)
from Dividendos.core.processing.allocation import compare_allocation
from Dividendos.core.ingestion.flex_query import import_flex_report
from Dividendos.core.ingestion.ibkr_flex_service import fetch_flex_report, save_raw_xml, FlexServiceError
from Dividendos.core.storage.github_sync import download_db, upload_db

DB_PATH = Path(__file__).parent / "data" / "processed" / "dividendos.db"
RAW_DIR = Path(__file__).parent / "data" / "raw"
REMOTE_DB_PATH = "dividendos.db"

RANGE_OPTIONS = [
    "Todo", "Año en curso", "Último año", "Últimos 6 meses",
    "Últimos 3 meses", "Último mes", "Mes en curso",
]


def _months_ago(d: date, months: int) -> date:
    """Resta 'months' meses a una fecha, ajustando el día si el mes
    destino tiene menos días (sin depender de librerías externas)."""
    m = d.month - months
    y = d.year
    while m <= 0:
        m += 12
        y -= 1
    day = min(d.day, _calendar_module.monthrange(y, m)[1])
    return date(y, m, day)


def _range_start_date(range_label: str, today: date):
    """Devuelve la fecha de inicio del rango elegido, o None si es 'Todo'."""
    if range_label == "Todo":
        return None
    if range_label == "Año en curso":
        return date(today.year, 1, 1)
    if range_label == "Último año":
        return _months_ago(today, 12)
    if range_label == "Últimos 6 meses":
        return _months_ago(today, 6)
    if range_label == "Últimos 3 meses":
        return _months_ago(today, 3)
    if range_label == "Último mes":
        return _months_ago(today, 1)
    if range_label == "Mes en curso":
        return date(today.year, today.month, 1)
    return None


def _get_github_secrets():
    gh = st.secrets.get("github_data")
    if not gh:
        return None
    token, repo = gh.get("token"), gh.get("repo")
    branch = gh.get("branch", "main")
    if not token or not repo:
        return None
    return token, repo, branch


def _sync_down():
    """Trae la última versión de la BD desde GitHub antes de leer, por si
    el contenedor de Streamlit se reinició desde la última visita."""
    creds = _get_github_secrets()
    if creds is None:
        return
    token, repo, branch = creds
    try:
        download_db(DB_PATH, REMOTE_DB_PATH, token, repo, branch)
    except Exception:
        pass  # si falla, seguimos con lo que haya en local sin romper la página


def _sync_up():
    creds = _get_github_secrets()
    if creds is None:
        return
    token, repo, branch = creds
    try:
        upload_db(
            DB_PATH, REMOTE_DB_PATH, token, repo, branch,
            commit_message=f"Dividendos: auto-actualización {datetime.now().isoformat(timespec='minutes')}",
        )
    except Exception as e:
        st.warning(f"⚠️ No se pudo guardar en GitHub tras la auto-actualización: {e}")


def _maybe_auto_update_from_ibkr():
    """
    Actualiza desde IBKR automáticamente, como máximo 1 vez al día.
    Usa session_state para no repetir la llamada en cada rerun dentro de
    la misma sesión (cambiar un selector no debe volver a llamar a IBKR).
    """
    if st.session_state.get("ibkr_auto_synced_dividendos"):
        return
    st.session_state["ibkr_auto_synced_dividendos"] = True  # evita reintentos aunque falle

    last_date = None
    if DB_PATH.exists():
        log_df = read_import_log(DB_PATH)
        if not log_df.empty:
            try:
                last_date = pd.to_datetime(log_df["imported_at"]).max().date()
            except Exception:
                last_date = None

    if last_date == date.today():
        return  # ya actualizado hoy, no hace falta llamar a IBKR

    try:
        ibkr_secrets = st.secrets["ibkr_dividendos"]
        token = ibkr_secrets["flex_token"]
        query_id = ibkr_secrets["flex_query_id"]
    except (KeyError, FileNotFoundError):
        return  # sin secrets de IBKR, seguimos con lo que haya en local

    with st.spinner("Actualizando datos desde IBKR (una vez al día)..."):
        try:
            xml_text = fetch_flex_report(token, query_id)
            raw_path = save_raw_xml(xml_text, RAW_DIR)
            parsed = import_flex_report(raw_path)
            save_flex_import(DB_PATH, parsed, source_file="IBKR (auto, Panel)")
            _sync_up()
        except FlexServiceError as e:
            st.warning(f"⚠️ No se pudo auto-actualizar desde IBKR ({e.code}): {e.message}")
        except Exception as e:
            st.warning(f"⚠️ No se pudo auto-actualizar desde IBKR: {e}")


# ---------------------------------------------------------------------------
# Gráfico lineal de evolución del NLV
# ---------------------------------------------------------------------------

def render_nlv_line_chart(equity_df, scale="linear", display_mode="dollar"):
    """
    Gráfico de evolución del NLV con línea/relleno verde cuando el valor es
    positivo y rojo cuando es negativo, cambiando de color exactamente en
    el cruce por cero (sin huecos ni cortes artificiales).
    """
    df = equity_df.sort_values("reportDate").reset_index(drop=True)

    if display_mode == "percent":
        base = df["total"].iloc[0]
        y_values = (df["total"] / base - 1) * 100 if base else df["total"] * 0
        hover_fmt = "%{y:+.2f}%"
    else:
        y_values = df["total"]
        hover_fmt = "$%{y:,.2f}"

    x_raw = df["reportDate"].to_numpy()
    y_raw = y_values.to_numpy(dtype=float)

    x_num = x_raw.astype("datetime64[ns]").astype("int64").astype(float)

    x_full = [x_num[0]]
    y_full = [y_raw[0]]
    for i in range(1, len(y_raw)):
        y0, y1 = y_raw[i - 1], y_raw[i]
        x0, x1 = x_num[i - 1], x_num[i]

        if (
            not np.isnan(y0)
            and not np.isnan(y1)
            and y0 != 0
            and y1 != 0
            and np.sign(y0) != np.sign(y1)
        ):
            t = y0 / (y0 - y1)
            x_zero = x0 + t * (x1 - x0)
            x_full.append(x_zero)
            y_full.append(0.0)

        x_full.append(x1)
        y_full.append(y1)

    x_full = pd.to_datetime(np.array(x_full).astype("int64"))
    y_full = np.array(y_full)

    use_fill = scale == "linear"

    pos_y = np.where(y_full >= 0, y_full, np.nan)
    neg_y = np.where(y_full <= 0, y_full, np.nan)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_full, y=pos_y, mode="lines",
        line=dict(color="#2ECC71", width=2),
        connectgaps=False,
        fill="tozeroy" if use_fill else None,
        fillcolor="rgba(46,204,113,0.10)" if use_fill else None,
        hovertemplate=f"%{{x|%d/%m/%Y}}<br>{hover_fmt}<extra></extra>",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=x_full, y=neg_y, mode="lines",
        line=dict(color="#E74C3C", width=2),
        connectgaps=False,
        fill="tozeroy" if use_fill else None,
        fillcolor="rgba(231,76,60,0.10)" if use_fill else None,
        hovertemplate=f"%{{x|%d/%m/%Y}}<br>{hover_fmt}<extra></extra>",
        showlegend=False,
    ))

    yaxis_config = dict(
        showgrid=True, gridcolor="rgba(255,255,255,0.08)", type=scale,
        zeroline=True, zerolinecolor="rgba(255,255,255,0.3)",
    )
    if display_mode == "percent":
        yaxis_config["ticksuffix"] = "%"
    else:
        yaxis_config["tickprefix"] = "$"
        yaxis_config["tickformat"] = ",.0f"
        yaxis_config["exponentformat"] = "none"

    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        xaxis=dict(showgrid=False),
        yaxis=yaxis_config,
    )
    return fig


# ---------------------------------------------------------------------------
# Gráfico de barras: crecimiento mensual de la cuenta (%)
# ---------------------------------------------------------------------------

def render_growth_chart(growth_df):
    df = growth_df[growth_df["end_nlv"].notna()].copy()
    if df.empty:
        return None

    colors = ["#2ECC71" if v >= 0 else "#E74C3C" for v in df["growth_pct"]]

    fig = go.Figure(go.Bar(
        x=df["month_label"],
        y=df["growth_pct"],
        marker_color=colors,
        customdata=df["growth_abs"],
        hovertemplate="%{x}<br>Crecimiento: %{y:+.2f}%<br>($%{customdata:+,.0f})<extra></extra>",
        text=[f"{v:+.2f}%" for v in df["growth_pct"]],
        textposition="outside",
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        xaxis=dict(showgrid=False),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(255,255,255,0.08)",
            zeroline=True, zerolinecolor="rgba(255,255,255,0.3)",
            ticksuffix="%",
        ),
        showlegend=False,
    )
    return fig


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

    _sync_down()
    _maybe_auto_update_from_ibkr()

    if not DB_PATH.exists():
        st.info(
            "Todavía no hay datos disponibles. Comprueba que los secrets de "
            "IBKR (`[ibkr_dividendos]`) estén configurados correctamente."
        )
        return

    trades = read_trades(DB_PATH)
    equity = read_equity_summary(DB_PATH)
    cash_tx = read_cash_transactions(DB_PATH)

    if equity.empty:
        st.info(
            "Todavía no hay datos guardados. Comprueba que los secrets de "
            "IBKR (`[ibkr_dividendos]`) estén configurados correctamente."
        )
        return

    today = date.today()

    available_years = sorted(equity["reportDate"].dt.year.unique(), reverse=True)
    default_index = available_years.index(today.year) if today.year in available_years else 0
    selected_year = st.selectbox("Año", options=available_years, index=default_index, key="panel_year")

    acc = account_summary(equity)

    log_df = read_import_log(DB_PATH)
    last_update_str = "—"
    if not log_df.empty:
        last_update_str = pd.to_datetime(log_df["imported_at"]).max().strftime("%d/%m/%Y %H:%M")

    st.caption(f"Resumen de cuenta a fecha de hoy · Detalle del año {selected_year} · Última actualización: {last_update_str}")
    st.markdown("---")

    st.subheader("Resumen de cuenta")
    col1, col2, col3, col4 = st.columns(4)
    if acc:
        col1.metric("NLV", f"${acc['nlv']:,.2f}")
        col2.metric("Caja", f"${acc['cash']:,.2f}")
        col3.metric("Invertido", f"${acc['invested']:,.2f}")
        ytd = acc["ytd_return_pct"]
        col4.metric("Rentab. YTD", f"{ytd:.2f}%" if ytd is not None else "—")

    st.markdown("---")

    st.subheader("Evolución del valor de la cartera (NLV)")
    rcol1, rcol2, rcol3 = st.columns([2, 1, 1])
    with rcol1:
        range_label = st.selectbox("Rango", options=RANGE_OPTIONS, index=0, key="nlv_range")
    with rcol2:
        display_label = st.radio("Vista", options=["$", "%"], horizontal=True, key="nlv_display")
    with rcol3:
        scale_label = st.radio(
            "Escala", options=["Lineal", "Log"], horizontal=True,
            disabled=(display_label == "%"), key="nlv_scale",
        )

    display_mode = "percent" if display_label == "%" else "dollar"
    scale = "linear" if display_mode == "percent" else ("log" if scale_label == "Log" else "linear")
    if display_mode == "percent":
        st.caption("La escala logarítmica no está disponible en vista %, ya que el % puede ser negativo.")

    range_start = _range_start_date(range_label, today)
    equity_chart = (
        equity if range_start is None
        else equity[equity["reportDate"] >= pd.Timestamp(range_start)]
    )
    if equity_chart.empty:
        st.caption("Sin datos en el rango seleccionado")
    else:
        st.plotly_chart(
            render_nlv_line_chart(equity_chart, scale=scale, display_mode=display_mode),
            use_container_width=True,
        )

        period_df = equity_chart.sort_values("reportDate")
        start_val = float(period_df["total"].iloc[0])
        end_val = float(period_df["total"].iloc[-1])
        change_abs = end_val - start_val
        change_pct = (change_abs / start_val * 100) if start_val else None

        color = "green" if change_abs >= 0 else "red"
        sign = "+" if change_abs >= 0 else ""
        pct_str = f" ({sign}{change_pct:.2f}%)" if change_pct is not None else ""
        start_date_str = period_df["reportDate"].iloc[0].strftime("%d/%m/%Y")
        end_date_str = period_df["reportDate"].iloc[-1].strftime("%d/%m/%Y")

        st.markdown(
            f"**Variación del periodo** ({start_date_str} → {end_date_str}): "
            f":{color}[{sign}${change_abs:,.2f}{pct_str}]"
        )

    st.markdown("---")

    closed_selected = filter_closed_trades(
        trades, date_from=date(selected_year, 1, 1), date_to=date(selected_year, 12, 31)
    ) if not trades.empty else trades
    pnl = pnl_summary(closed_selected)
    wr = win_rate(closed_selected)

    st.subheader(f"P&L de trading ({selected_year})")
    p1, p2, p3, p4 = st.columns(4)
    net = pnl["net_pnl"]
    color = "green" if net >= 0 else "red"
    sign = "+" if net >= 0 else ""
    p1.markdown("**P&L neto**")
    p1.markdown(f"### :{color}[{sign}${net:,.2f}]")
    p2.metric("Ganancias brutas", f"${pnl['gross_profit']:,.2f}")
    p3.metric("Pérdidas brutas", f"${pnl['gross_loss']:,.2f}")
    p4.metric(
        "Tasa de acierto",
        f"{wr['win_rate_pct']:.1f}%" if wr["win_rate_pct"] is not None else "—",
        help=f"{wr['wins']}W / {wr['losses']}L",
    )

    st.markdown("---")

    st.subheader(f"Crecimiento mensual de la cuenta {selected_year}")
    growth = account_growth_by_month(equity, selected_year)
    growth_chart = render_growth_chart(growth)

    if growth_chart is None:
        st.caption("Sin datos suficientes todavía")
    else:
        st.plotly_chart(growth_chart, use_container_width=True)

    st.markdown("---")

    st.subheader(f"Dividendos mensuales {selected_year}")
    div_monthly = dividends_by_month(cash_tx, selected_year)
    div_total = dividends_total(cash_tx, selected_year)

    d1, d2, d3 = st.columns(3)
    d1.metric("Dividendo bruto (año)", f"${div_total['gross']:,.2f}")
    d2.metric("Retenciones (año)", f"${div_total['withholding_tax']:,.2f}")
    d3.metric("Dividendo neto (año)", f"${div_total['net']:,.2f}")

    st.plotly_chart(render_dividends_chart(div_monthly), use_container_width=True)

    st.markdown("---")

    st.subheader("Instrumentos y acciones que tienes")
    positions = read_open_positions(DB_PATH)
    target = read_target_allocation(DB_PATH)

    if positions.empty:
        st.caption("Sin posiciones abiertas")
    elif not target.empty and acc:
        result = compare_allocation(positions, target, acc["nlv"], cash_value=acc["cash"])
        result["deviation_pct"] = result["current_pct"] - result["target_pct"]
        result = result.sort_values("current_value", ascending=False)

        def _color_deviation(val):
            if val > 0:
                return "color: #2ECC71"
            if val < 0:
                return "color: #E74C3C"
            return ""

        cols = [
            "ticker", "shares_held", "price", "current_value",
            "current_pct", "target_pct", "deviation_pct",
        ]
        styled = result[cols].style.map(_color_deviation, subset=["deviation_pct"])

        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ticker": "Ticker",
                "shares_held": st.column_config.NumberColumn("Acciones", format="%.2f"),
                "price": st.column_config.NumberColumn("Precio", format="$%.2f"),
                "current_value": st.column_config.NumberColumn("Valor", format="$%.2f"),
                "current_pct": st.column_config.NumberColumn("% actual", format="%.2f%%"),
                "target_pct": st.column_config.NumberColumn("% objetivo", format="%.2f%%"),
                "deviation_pct": st.column_config.NumberColumn("Diferencia (pp)", format="%+.2f"),
            },
        )
        st.caption(
            "🟢 Verde = por encima del objetivo · 🔴 Rojo = por debajo del objetivo. "
            "Detalle completo en la página **Asignación**."
        )
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
        st.caption(
            "Ve a la página **Asignación** para definir tu tabla objetivo y "
            "ver aquí la diferencia con tus holdings."
        )


if __name__ == "__main__":
    if check_password():
        main()
    else:
        st.title("🔒 Acceso Restringido")
        st.info("Por favor, introduce tus credenciales en el menú lateral (sidebar) para acceder.")
