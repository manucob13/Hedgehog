import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, datetime

from utils.utils import check_password
from Cartera.core.storage.db import (
    read_trades, read_equity_summary, read_open_positions,
    read_import_log, save_flex_import,
)
from Cartera.core.processing.metrics import (
    filter_closed_trades,
    profit_factor,
    win_rate,
    pnl_summary,
    account_summary,
)
from Cartera.core.ingestion.flex_query import import_flex_report
from Cartera.core.ingestion.ibkr_flex_service import fetch_flex_report, save_raw_xml, FlexServiceError
from Cartera.core.storage.github_sync import download_db, upload_db
import calendar as _calendar_module

DB_PATH = Path(__file__).parent / "data" / "processed" / "cartera.db"
RAW_DIR = Path(__file__).parent / "data" / "raw"
REMOTE_DB_PATH = "cartera.db"

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
            commit_message=f"Cartera: auto-actualización {datetime.now().isoformat(timespec='minutes')}",
        )
    except Exception as e:
        st.warning(f"⚠️ No se pudo guardar en GitHub tras la auto-actualización: {e}")


def _maybe_auto_update_from_ibkr():
    """
    Actualiza desde IBKR automáticamente, como máximo 1 vez al día.
    Usa session_state para no repetir la llamada en cada rerun dentro de
    la misma sesión (cambiar un selector no debe volver a llamar a IBKR).
    """
    if st.session_state.get("ibkr_auto_synced_cartera"):
        return
    st.session_state["ibkr_auto_synced_cartera"] = True  # evita reintentos aunque falle

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
        ibkr_secrets = st.secrets["ibkr"]
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
    _maybe_auto_update_from_ibkr()

    if not DB_PATH.exists():
        st.info(
            "Todavía no hay datos disponibles. Comprueba que los secrets de "
            "IBKR (`[ibkr]`) estén configurados correctamente."
        )
        return

    trades = read_trades(DB_PATH)
    equity = read_equity_summary(DB_PATH)

    if trades.empty and equity.empty:
        st.info(
            "Todavía no hay datos guardados. Comprueba que los secrets de "
            "IBKR (`[ibkr]`) estén configurados correctamente."
        )
        return

    today = date.today()

    available_years = sorted(equity["reportDate"].dt.year.unique(), reverse=True) if not equity.empty else [today.year]
    default_index = available_years.index(today.year) if today.year in available_years else 0
    selected_year = st.selectbox("Año", options=available_years, index=default_index, key="panel_year")

    acc = account_summary(equity) if not equity.empty else None

    closed_selected = filter_closed_trades(
        trades, date_from=date(selected_year, 1, 1), date_to=date(selected_year, 12, 31)
    ) if not trades.empty else trades

    pf = profit_factor(closed_selected)
    wr = win_rate(closed_selected)
    pnl = pnl_summary(closed_selected)

    log_df = read_import_log(DB_PATH)
    last_update_str = "—"
    if not log_df.empty:
        last_update_str = pd.to_datetime(log_df["imported_at"]).max().strftime("%d/%m/%Y %H:%M")

    st.caption(f"Resumen de cuenta a fecha de hoy · P&L del año {selected_year} · Última actualización: {last_update_str}")
    st.markdown("---")

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

    st.subheader("Evolución del valor de la cartera (NLV)")
    if equity.empty:
        st.caption("Sin datos suficientes para el gráfico")
    else:
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
