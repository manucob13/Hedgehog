"""
metrics.py
Métricas derivadas de los trades y del equity_summary:
    - Operaciones cerradas (closed trades)
    - Factor de beneficio (profit factor)
    - Tasa de acierto (win rate) + conteo W/L
    - Resumen de P&L (ganancias brutas, pérdidas brutas, neto)
    - Resumen de cuenta (NLV, caja, invertido, rentabilidad YTD)
    - Tickers operados en un período
    - Dividendos por mes + media
    - Crecimiento mensual de la cuenta

Nota clave: `fifoPnlRealized` en las filas de cierre (openCloseIndicator='C')
ya viene neto de comisiones de apertura y cierre (IBKR lo calcula así).
Por eso no hace falta restar `ibCommission` aparte al calcular P&L.

RENTABILIDAD (TWR):
Las rentabilidades de cuenta (YTD, crecimiento mensual, gráfico del NLV en %)
se calculan con rentabilidad ponderada por tiempo (TWR, encadenando días) y
descontando los depósitos y retiradas de efectivo. Antes se comparaba el NLV
final con el inicial, lo que cuenta los ingresos de dinero como si fueran
ganancias. Convención: el flujo de un día se considera recibido al final de
ese día (no genera rentabilidad ese día).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Tipos de cash transaction que son aportaciones/retiradas externas (no rentabilidad).
# Si en tu Flex Query aparecen con otro nombre, añádelo aquí.
FLOW_TYPES = ["Deposits/Withdrawals"]


# ---------------------------------------------------------------------------
# Operaciones cerradas
# ---------------------------------------------------------------------------

def filter_closed_trades(
    trades_df: pd.DataFrame,
    date_from=None,
    date_to=None,
) -> pd.DataFrame:
    """
    Devuelve solo las filas de cierre (openCloseIndicator == 'C'), que son
    las que tienen P&L realizado. Opcionalmente filtra por tradeDate.
    """
    if trades_df.empty:
        return trades_df

    df = trades_df[trades_df["openCloseIndicator"] == "C"].copy()

    if date_from is not None:
        df = df[df["tradeDate"] >= pd.to_datetime(date_from)]
    if date_to is not None:
        df = df[df["tradeDate"] <= pd.to_datetime(date_to)]

    return df


# ---------------------------------------------------------------------------
# Factor de beneficio y tasa de acierto
# ---------------------------------------------------------------------------

def profit_factor(closed_trades_df: pd.DataFrame) -> float | None:
    """
    Factor de beneficio = ganancias brutas / |pérdidas brutas|.
    Devuelve None si no hay pérdidas (evita división por cero) y no hay
    tampoco ganancias (no hay datos suficientes para calcularlo).
    """
    if closed_trades_df.empty:
        return None

    pnl = closed_trades_df["fifoPnlRealized"]
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = pnl[pnl < 0].sum()

    if gross_loss == 0:
        return None if gross_profit == 0 else float("inf")

    return gross_profit / abs(gross_loss)


def win_rate(closed_trades_df: pd.DataFrame) -> dict:
    """
    Tasa de acierto sobre operaciones cerradas.
    Las operaciones con P&L exactamente 0 (breakeven) no cuentan ni como
    ganadora ni como perdedora, pero se reportan aparte.
    """
    if closed_trades_df.empty:
        return {
            "win_rate_pct": None,
            "wins": 0,
            "losses": 0,
            "breakeven": 0,
            "total_closed": 0,
        }

    pnl = closed_trades_df["fifoPnlRealized"]
    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())
    breakeven = int((pnl == 0).sum())
    decided = wins + losses

    win_rate_pct = (wins / decided * 100) if decided > 0 else None

    return {
        "win_rate_pct": win_rate_pct,
        "wins": wins,
        "losses": losses,
        "breakeven": breakeven,
        "total_closed": wins + losses + breakeven,
    }


# ---------------------------------------------------------------------------
# Resumen de P&L (para la tarjeta "P&L acumulado" y el +$/-$ del gauge)
# ---------------------------------------------------------------------------

def pnl_summary(closed_trades_df: pd.DataFrame) -> dict:
    """
    Devuelve:
        {
            "gross_profit": float,  # suma de operaciones ganadoras (>=0)
            "gross_loss": float,    # suma de operaciones perdedoras (<=0, negativo)
            "net_pnl": float,       # gross_profit + gross_loss
        }
    """
    if closed_trades_df.empty:
        return {"gross_profit": 0.0, "gross_loss": 0.0, "net_pnl": 0.0}

    pnl = closed_trades_df["fifoPnlRealized"]
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(pnl[pnl < 0].sum())

    return {
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_pnl": gross_profit + gross_loss,
    }


# ---------------------------------------------------------------------------
# Rentabilidad ponderada por tiempo (TWR), descontando depósitos/retiradas
# ---------------------------------------------------------------------------

def external_flows(cash_tx_df: pd.DataFrame | None, flow_types=None) -> pd.DataFrame:
    """Filas de cash transactions que son depósitos/retiradas externas."""
    flow_types = flow_types or FLOW_TYPES
    if cash_tx_df is None or cash_tx_df.empty or "type" not in cash_tx_df.columns:
        return pd.DataFrame(columns=["reportDate", "amount", "type"])
    return cash_tx_df[cash_tx_df["type"].isin(flow_types)].copy()


def twr_series(
    equity_df: pd.DataFrame,
    cash_tx_df: pd.DataFrame | None = None,
    flow_types=None,
) -> pd.DataFrame:
    """
    Serie diaria con columnas:
        reportDate, total, flow, daily_return, twr_index
    - flow: depósitos(+)/retiradas(-) asignados al primer día de equity con
      fecha >= la del movimiento (cubre fines de semana y festivos).
    - daily_return = (NLV_t - flujo_t) / NLV_{t-1} - 1
    - twr_index: índice encadenado que empieza en 1.0 en el primer día.
    """
    eq = (
        equity_df.sort_values("reportDate")[["reportDate", "total"]]
        .reset_index(drop=True)
        .copy()
    )
    eq["total"] = pd.to_numeric(eq["total"], errors="coerce")
    eq["flow"] = 0.0

    fl = external_flows(cash_tx_df, flow_types)
    if not fl.empty and len(eq):
        dates = eq["reportDate"].to_numpy()
        fdates = pd.to_datetime(fl["reportDate"]).to_numpy()
        idx = np.searchsorted(dates, fdates, side="left")
        amounts = pd.to_numeric(fl["amount"], errors="coerce").fillna(0.0).to_numpy()
        valid = idx < len(eq)  # movimientos posteriores al último dato: aún no están en el NLV
        if valid.any():
            by_day = pd.Series(amounts[valid]).groupby(idx[valid]).sum()
            eq.loc[by_day.index, "flow"] = by_day.values

    prev = eq["total"].shift(1)
    r = (eq["total"] - eq["flow"]) / prev - 1
    r = r.where(prev.notna() & (prev != 0), 0.0).fillna(0.0)
    eq["daily_return"] = r
    eq["twr_index"] = (1 + r).cumprod()
    return eq


def twr_period(twr_df, start=None, end=None, include_start_day=False) -> dict | None:
    """
    Rentabilidad TWR entre `start` y `end`.

    - Base: el último dato ANTERIOR a `start` (así el primer día del periodo
      cuenta). Si no existe, el primer dato >= start.
    - include_start_day=True: la base es el primer dato >= start (coincide con
      un gráfico que empieza ese día en 0%).

    Devuelve dict: twr_pct, gain_abs (ganancia en $ neta de flujos),
    net_flows, start_value, end_value, start_date, end_date. None si no hay datos.
    """
    if twr_df is None or twr_df.empty:
        return None

    df = twr_df
    if end is not None:
        df = df[df["reportDate"] <= pd.to_datetime(end)]
    if df.empty:
        return None

    if start is None:
        base_pos = df.index[0]
    else:
        s = pd.to_datetime(start)
        before = df[df["reportDate"] < s]
        after = df[df["reportDate"] >= s]
        if include_start_day or before.empty:
            if after.empty:
                return None
            base_pos = after.index[0]
        else:
            base_pos = before.index[-1]

    end_pos = df.index[-1]
    base = twr_df.loc[base_pos]
    last = twr_df.loc[end_pos]

    if end_pos <= base_pos:
        pct, flows = 0.0, 0.0
    else:
        pct = (last["twr_index"] / base["twr_index"] - 1) * 100
        flows = float(twr_df.loc[base_pos + 1:end_pos, "flow"].sum())

    return {
        "twr_pct": float(pct),
        "gain_abs": float(last["total"] - base["total"] - flows),
        "net_flows": flows,
        "start_value": float(base["total"]),
        "end_value": float(last["total"]),
        "start_date": base["reportDate"],
        "end_date": last["reportDate"],
    }


# ---------------------------------------------------------------------------
# Resumen de cuenta (NLV, caja, invertido, rentabilidad YTD)
# ---------------------------------------------------------------------------

def account_summary(equity_df: pd.DataFrame, as_of=None, cash_tx_df=None) -> dict:
    """
    Calcula el resumen de cuenta a partir de equity_summary (una fila/día).

    - nlv: 'total' del último día disponible (o de `as_of` si se indica)
    - cash: 'cash' de ese mismo día
    - invested: nlv - cash (valor de mercado de acciones + opciones)
    - ytd_return_pct: rentabilidad TWR del año (descuenta depósitos/retiradas
      si se pasa `cash_tx_df`). Base: último dato del año anterior, o el
      primer dato del año si no hay historial previo.
    - ytd_simple_pct: variación simple del NLV (incluye depósitos), solo para
      comparar/diagnosticar.
    - ytd_gain: ganancia en $ del año neta de depósitos/retiradas.
    - ytd_net_flows: depósitos(+)/retiradas(-) netos del año.

    Devuelve None si equity_df está vacío.
    """
    if equity_df.empty:
        return None

    df = equity_df.sort_values("reportDate")
    if as_of is not None:
        df = df[df["reportDate"] <= pd.to_datetime(as_of)]
    if df.empty:
        return None

    last_row = df.iloc[-1]
    nlv = float(last_row["total"])
    cash = float(last_row["cash"])
    invested = nlv - cash

    year = last_row["reportDate"].year
    year_rows = df[df["reportDate"].dt.year == year]

    ytd_simple_pct = None
    if not year_rows.empty:
        start_value = float(year_rows.iloc[0]["total"])
        if start_value != 0:
            ytd_simple_pct = (nlv - start_value) / start_value * 100

    ytd_return_pct = None
    ytd_gain = None
    ytd_net_flows = None
    twr = twr_series(df, cash_tx_df)
    perf = twr_period(twr, start=pd.Timestamp(year, 1, 1), end=last_row["reportDate"])
    if perf is not None:
        ytd_return_pct = perf["twr_pct"]
        ytd_gain = perf["gain_abs"]
        ytd_net_flows = perf["net_flows"]

    return {
        "as_of_date": last_row["reportDate"],
        "nlv": nlv,
        "cash": cash,
        "invested": invested,
        "ytd_return_pct": ytd_return_pct,
        "ytd_simple_pct": ytd_simple_pct,
        "ytd_gain": ytd_gain,
        "ytd_net_flows": ytd_net_flows,
    }


# ---------------------------------------------------------------------------
# Tickers operados en un período (para la tarjeta "Tickers operados")
# ---------------------------------------------------------------------------

def tickers_traded(
    trades_df: pd.DataFrame,
    year: int | None = None,
    month: int | None = None,
) -> pd.DataFrame:
    """
    Agrupa por underlyingSymbol dentro del año/mes indicados (si se pasan)
    y devuelve nº de operaciones y P&L neto realizado por ticker,
    ordenado de mayor a menor nº de operaciones.
    """
    if trades_df.empty:
        return pd.DataFrame(columns=["underlyingSymbol", "n_trades", "net_pnl"])

    df = trades_df.copy()
    if year is not None:
        df = df[df["tradeDate"].dt.year == year]
    if month is not None:
        df = df[df["tradeDate"].dt.month == month]

    if df.empty:
        return pd.DataFrame(columns=["underlyingSymbol", "n_trades", "net_pnl"])

    grouped = df.groupby("underlyingSymbol").agg(
        n_trades=("underlyingSymbol", "count"),
        net_pnl=("fifoPnlRealized", "sum"),
    ).reset_index()

    return grouped.sort_values("n_trades", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Dividendos por mes (para el gráfico de barras + línea de media)
# ---------------------------------------------------------------------------

MESES_ES = [
    "ENE", "FEB", "MAR", "ABR", "MAY", "JUN",
    "JUL", "AGO", "SEP", "OCT", "NOV", "DIC",
]


def dividends_by_month(cash_transactions_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Agrega los dividendos cobrados por mes para un año concreto.

    Incluye como "dividendo bruto" los tipos 'Dividends' y
    'Payment In Lieu Of Dividends' (este último es el pago sustitutivo que
    IBKR abona cuando la posición estuvo prestada en la fecha ex-dividendo;
    económicamente es lo mismo que un dividendo para el titular).
    'Withholding Tax' se resta aparte para dar el neto.

    Devuelve DataFrame con columnas:
        month, month_label, gross, withholding_tax, net
    Los meses sin datos aparecen con 0.0. Incluye también una columna
    'avg_net' repetida en todas las filas = media mensual del neto
    (solo contando los meses que SÍ tienen movimientos, para no falsear
    la media con meses futuros vacíos).
    """
    base = pd.DataFrame({"month": range(1, 13), "month_label": MESES_ES})

    if cash_transactions_df.empty:
        base["gross"] = 0.0
        base["withholding_tax"] = 0.0
        base["net"] = 0.0
        base["avg_net"] = 0.0
        return base

    df = cash_transactions_df[cash_transactions_df["reportDate"].dt.year == year].copy()

    gross_types = ["Dividends", "Payment In Lieu Of Dividends"]
    gross_df = df[df["type"].isin(gross_types)]
    tax_df = df[df["type"] == "Withholding Tax"]

    gross_by_month = (
        gross_df.groupby(gross_df["reportDate"].dt.month)["amount"]
        .sum(min_count=1).reindex(range(1, 13)).fillna(0.0)
    )
    tax_by_month = (
        tax_df.groupby(tax_df["reportDate"].dt.month)["amount"]
        .sum(min_count=1).reindex(range(1, 13)).fillna(0.0)
    )

    result = base.copy()
    result["gross"] = gross_by_month.values
    result["withholding_tax"] = tax_by_month.values  # ya viene negativo en IBKR
    result["net"] = result["gross"] + result["withholding_tax"]

    months_with_data = result.loc[result["gross"] != 0, "net"]
    avg_net = float(months_with_data.mean()) if not months_with_data.empty else 0.0
    result["avg_net"] = avg_net

    return result


def dividends_total(cash_transactions_df: pd.DataFrame, year: int) -> dict:
    """Resumen del año: gross, withholding_tax, net totales."""
    monthly = dividends_by_month(cash_transactions_df, year)
    return {
        "gross": float(monthly["gross"].sum()),
        "withholding_tax": float(monthly["withholding_tax"].sum()),
        "net": float(monthly["net"].sum()),
    }


# ---------------------------------------------------------------------------
# Crecimiento mensual de la cuenta (tarjetas de crecimiento)
# ---------------------------------------------------------------------------

def account_growth_by_month(equity_df: pd.DataFrame, year: int, cash_tx_df=None) -> pd.DataFrame:
    """
    Para cada mes del año indicado calcula la rentabilidad TWR del mes
    (descontando depósitos/retiradas si se pasa `cash_tx_df`). La base de cada
    mes es el último dato del mes anterior, así no se pierde el movimiento del
    primer día del mes.

    Devuelve DataFrame con columnas:
        month, month_label, start_nlv, end_nlv, growth_abs, growth_pct
    - growth_abs: ganancia en $ del mes neta de depósitos/retiradas.
    - growth_pct: rentabilidad TWR del mes en %.
    Meses sin datos: todo None.
    """
    base = pd.DataFrame({"month": range(1, 13), "month_label": MESES_ES})

    if equity_df.empty:
        base["start_nlv"] = None
        base["end_nlv"] = None
        base["growth_abs"] = None
        base["growth_pct"] = None
        return base

    twr = twr_series(equity_df, cash_tx_df)
    year_df = twr[twr["reportDate"].dt.year == year]

    rows = []
    for month in range(1, 13):
        month_df = year_df[year_df["reportDate"].dt.month == month]
        if month_df.empty:
            rows.append({"start_nlv": None, "end_nlv": None, "growth_abs": None, "growth_pct": None})
            continue

        month_start = pd.Timestamp(year, month, 1)
        month_end = month_start + pd.offsets.MonthEnd(0)
        perf = twr_period(twr, start=month_start, end=month_end)
        if perf is None:
            rows.append({"start_nlv": None, "end_nlv": None, "growth_abs": None, "growth_pct": None})
            continue

        rows.append({
            "start_nlv": perf["start_value"],
            "end_nlv": perf["end_value"],
            "growth_abs": perf["gain_abs"],
            "growth_pct": perf["twr_pct"],
        })

    result = pd.concat([base, pd.DataFrame(rows)], axis=1)
    return result
