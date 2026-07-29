"""
allocation.py
Compara los holdings actuales (Open Positions, solo acciones/ETFs -
assetCategory == 'STK') contra una tabla de asignación objetivo definida
por el usuario, y calcula cuántas acciones faltan/sobran por ticker para
llegar al % objetivo. También soporta una fila especial 'CASH' para el
% objetivo de liquidez.

La tabla objetivo se sube desde un CSV/XLSX con columnas:
    Ticker, Target_%
(el % se interpreta sobre el NLV total de ESTA cuenta). Puede incluir una
fila con Ticker = 'CASH' para fijar el % objetivo de liquidez.
"""

from __future__ import annotations

import pandas as pd


def compare_allocation(
    open_positions_df: pd.DataFrame,
    target_allocation_df: pd.DataFrame,
    nlv: float,
    cash_value: float | None = None,
    tolerance_pct: float = 0.5,
) -> pd.DataFrame:
    """
    Compara holdings actuales vs. objetivo.

    Args:
        open_positions_df: DataFrame de core.storage.db.read_open_positions()
        target_allocation_df: DataFrame con columnas ['ticker', 'target_pct'].
            Puede incluir una fila especial con ticker == 'CASH' para fijar
            el % objetivo de liquidez (se compara contra `cash_value`, no
            contra Open Positions).
        nlv: valor liquidativo actual de la cuenta (para calcular % y $ objetivo)
        cash_value: saldo de caja actual (columna 'cash' de equity_summary).
            Necesario solo si target_allocation_df incluye una fila 'CASH'.
        tolerance_pct: diferencia de % por debajo de la cual se considera
            "En objetivo" en vez de "Por encima"/"Por debajo"

    Devuelve DataFrame con columnas:
        ticker, shares_held, price, current_value, current_pct,
        target_pct, target_value, diff_value, shares_diff, status

    - shares_diff > 0  -> te faltan esas acciones para llegar al objetivo
    - shares_diff < 0  -> tienes ese exceso de acciones sobre el objetivo
    - Para la fila 'CASH', shares_diff siempre es None (no aplica el
      concepto de "acciones"); usa diff_value (en $) en su lugar.
    - Tickers en el objetivo que no posees aparecen con shares_held = 0
    - Tickers que posees pero no están en tu tabla objetivo aparecen con
      target_pct = 0 y status = "Sin objetivo definido"
    """
    if nlv is None or nlv == 0:
        return pd.DataFrame()

    target = target_allocation_df.rename(
        columns={c: c.lower() for c in target_allocation_df.columns}
    )[["ticker", "target_pct"]].copy()
    target["ticker"] = target["ticker"].str.upper().str.strip()

    cash_target_row = target[target["ticker"] == "CASH"]
    target = target[target["ticker"] != "CASH"]

    # Solo acciones/ETFs, no las opciones de covered calls
    holdings = open_positions_df[open_positions_df["assetCategory"] == "STK"].copy()
    holdings = holdings.rename(columns={
        "symbol": "ticker",
        "position": "shares_held",
        "markPrice": "price",
        "positionValue": "current_value",
    })[["ticker", "shares_held", "price", "current_value"]]
    holdings["ticker"] = holdings["ticker"].str.upper().str.strip()

    merged = pd.merge(holdings, target, on="ticker", how="outer")

    merged["shares_held"] = merged["shares_held"].fillna(0.0)
    merged["current_value"] = merged["current_value"].fillna(0.0)
    merged["target_pct"] = merged["target_pct"].fillna(0.0)

    merged["current_pct"] = merged["current_value"] / nlv * 100
    merged["target_value"] = merged["target_pct"] / 100 * nlv
    merged["diff_value"] = merged["target_value"] - merged["current_value"]

    # Acciones de diferencia: solo calculable si conocemos el precio.
    # Si no tenemos precio (ticker objetivo que nunca se ha comprado),
    # dejamos shares_diff como NaN -- no podemos saber cuántas acciones son
    # sin un precio de mercado.
    merged["shares_diff"] = merged.apply(
        lambda r: (r["diff_value"] / r["price"]) if pd.notna(r.get("price")) and r["price"] else None,
        axis=1,
    )

    def _status(row):
        if row["target_pct"] == 0:
            return "Sin objetivo definido"
        gap = row["current_pct"] - row["target_pct"]
        if abs(gap) <= tolerance_pct:
            return "En objetivo"
        return "Por encima" if gap > 0 else "Por debajo"

    merged["status"] = merged.apply(_status, axis=1)

    cols = [
        "ticker", "shares_held", "price", "current_value", "current_pct",
        "target_pct", "target_value", "diff_value", "shares_diff", "status",
    ]
    result = merged[cols].sort_values("target_pct", ascending=False).reset_index(drop=True)

    # Fila especial CASH, si el usuario definió un objetivo para ella
    if not cash_target_row.empty and cash_value is not None:
        cash_target_pct = float(cash_target_row.iloc[0]["target_pct"])
        cash_current_pct = cash_value / nlv * 100
        cash_target_value = cash_target_pct / 100 * nlv
        cash_diff_value = cash_target_value - cash_value

        gap = cash_current_pct - cash_target_pct
        if abs(gap) <= tolerance_pct:
            cash_status = "En objetivo"
        else:
            cash_status = "Por encima" if gap > 0 else "Por debajo"

        cash_row = pd.DataFrame([{
            "ticker": "CASH",
            "shares_held": None,
            "price": None,
            "current_value": cash_value,
            "current_pct": cash_current_pct,
            "target_pct": cash_target_pct,
            "target_value": cash_target_value,
            "diff_value": cash_diff_value,
            "shares_diff": None,
            "status": cash_status,
        }])
        result = pd.concat([cash_row, result], ignore_index=True)

    return result
