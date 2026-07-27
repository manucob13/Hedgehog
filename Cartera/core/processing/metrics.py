"""
metrics.py
Métricas derivadas de los trades y del equity_summary:
    - Operaciones cerradas (closed trades)
    - Factor de beneficio (profit factor)
    - Tasa de acierto (win rate) + conteo W/L
    - Resumen de P&L (ganancias brutas, pérdidas brutas, neto)
    - Resumen de cuenta (NLV, caja, invertido, rentabilidad YTD)
    - Tickers operados en un período

Nota clave: `fifoPnlRealized` en las filas de cierre (openCloseIndicator='C')
ya viene neto de comisiones de apertura y cierre (IBKR lo calcula así).
Por eso no hace falta restar `ibCommission` aparte al calcular P&L.
"""

from __future__ import annotations

import pandas as pd


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

    Devuelve:
        {
            "win_rate_pct": float,   # 0-100, o None si no hay datos
            "wins": int,
            "losses": int,
            "breakeven": int,
            "total_closed": int,
        }
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
# Resumen de cuenta (NLV, caja, invertido, rentabilidad YTD)
# ---------------------------------------------------------------------------

def account_summary(equity_df: pd.DataFrame, as_of=None) -> dict:
    """
    Calcula el resumen de cuenta a partir de equity_summary (una fila/día).

    - nlv: 'total' del último día disponible (o de `as_of` si se indica)
    - cash: 'cash' de ese mismo día
    - invested: nlv - cash (valor de mercado de acciones + opciones)
    - ytd_return_pct: variación de 'total' desde el primer día del año
      hasta el día usado, en %. None si no hay dato del inicio de año.

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

    # Primer registro del mismo año que la fecha "as_of"/última disponible
    year = last_row["reportDate"].year
    year_rows = df[df["reportDate"].dt.year == year]
    ytd_return_pct = None
    if not year_rows.empty:
        start_value = float(year_rows.iloc[0]["total"])
        if start_value != 0:
            ytd_return_pct = (nlv - start_value) / start_value * 100

    return {
        "as_of_date": last_row["reportDate"],
        "nlv": nlv,
        "cash": cash,
        "invested": invested,
        "ytd_return_pct": ytd_return_pct,
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
