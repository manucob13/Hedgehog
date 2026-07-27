"""
calendar.py
Agregación de P&L diario -> estructura de calendario mensual (tarjetas por
día + suma semanal) y resumen anual (P&L por mes, ENE-DIC).

El P&L diario se calcula como la diferencia día a día del NLV total
(columna 'total' de equity_summary). Esta es una aproximación: si hubiera
depósitos o retiros de la cuenta en un día concreto, ese día distorsionaría
el P&L real (se vería como ganancia/pérdida cuando en realidad es solo
movimiento de caja). De momento no lo corregimos porque el Cash Report de
IBKR solo trae el total del período, no el desglose diario de depósitos;
si en el futuro Flex Query te da ese desglose diario, se puede restar aquí.

Nota de diseño: siguiendo el layout del dashboard de referencia, el
calendario NO muestra columna de domingo (los mercados no operan ese día).
"""

from __future__ import annotations

import calendar as _calendar_module
from datetime import date

import pandas as pd


MESES_ES = [
    "ENE", "FEB", "MAR", "ABR", "MAY", "JUN",
    "JUL", "AGO", "SEP", "OCT", "NOV", "DIC",
]


# ---------------------------------------------------------------------------
# P&L diario a partir de equity_summary
# ---------------------------------------------------------------------------

def daily_pnl(equity_df: pd.DataFrame) -> pd.DataFrame:
    """
    A partir de equity_summary (una fila/día con 'total' = NLV), calcula el
    P&L de cada día como la diferencia respecto al día anterior CON DATOS
    (no necesariamente el día calendario anterior, sino el anterior
    registro disponible).

    Devuelve DataFrame con columnas: reportDate, nlv, pnl
    El primer día disponible del histórico completo tendrá pnl = NaN,
    porque no hay un día previo con el que compararlo.
    """
    if equity_df.empty:
        return pd.DataFrame(columns=["reportDate", "nlv", "pnl"])

    df = equity_df.sort_values("reportDate")[["reportDate", "total"]].copy()
    df = df.rename(columns={"total": "nlv"})
    df["pnl"] = df["nlv"].diff()

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Calendario mensual (tarjetas diarias + suma semanal)
# ---------------------------------------------------------------------------

def monthly_calendar(daily_df: pd.DataFrame, year: int, month: int) -> dict:
    """
    Construye la estructura de calendario para un mes concreto.

    Devuelve:
        {
            "year": int,
            "month": int,
            "weeks": [
                {
                    "week_label": "S1",
                    "days": [
                        {"day": 1, "date": date(...), "pnl": float | None},
                        ...  # Lunes a Sábado (6 entradas); None si el día
                             # no pertenece al mes o no hay dato ese día
                    ],
                    "week_total": float,
                },
                ...
            ],
            "month_total": float,
        }
    """
    pnl_by_date = {}
    if not daily_df.empty:
        pnl_by_date = dict(zip(daily_df["reportDate"].dt.date, daily_df["pnl"]))

    cal = _calendar_module.Calendar(firstweekday=0)  # 0 = lunes
    month_weeks = cal.monthdayscalendar(year, month)  # incluye Lun..Dom (7 cols)

    weeks_out = []
    month_total = 0.0
    for i, week in enumerate(month_weeks, start=1):
        days_out = []
        week_total = 0.0
        for day_num in week[:6]:  # Lunes(0) .. Sábado(5); se descarta Domingo(6)
            if day_num == 0:
                days_out.append({"day": None, "date": None, "pnl": None})
                continue

            day_date = date(year, month, day_num)
            pnl_value = pnl_by_date.get(day_date)
            if pnl_value is not None and pd.notna(pnl_value):
                week_total += pnl_value
                month_total += pnl_value

            days_out.append({
                "day": day_num,
                "date": day_date,
                "pnl": float(pnl_value) if pnl_value is not None and pd.notna(pnl_value) else None,
            })

        weeks_out.append({
            "week_label": f"S{i}",
            "days": days_out,
            "week_total": week_total,
        })

    return {
        "year": year,
        "month": month,
        "weeks": weeks_out,
        "month_total": month_total,
    }


# ---------------------------------------------------------------------------
# Resumen anual (fila ENE..DIC + TOTAL de la vista "P&L")
# ---------------------------------------------------------------------------

def yearly_summary(daily_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    P&L total por mes para un año, en el mismo orden que la fila
    ENE FEB MAR ... DIC TOTAL del dashboard de referencia.

    Devuelve DataFrame con columnas: month (1-12), month_label, pnl
    Los meses sin datos aparecen con pnl = 0.0.
    """
    base = pd.DataFrame({"month": range(1, 13), "month_label": MESES_ES})

    if daily_df.empty:
        base["pnl"] = 0.0
        return base

    year_df = daily_df[daily_df["reportDate"].dt.year == year]
    monthly = (
        year_df.groupby(year_df["reportDate"].dt.month)["pnl"]
        .sum(min_count=1)
        .reindex(range(1, 13))
        .fillna(0.0)
        .reset_index()
    )
    monthly.columns = ["month", "pnl"]

    result = base.merge(monthly, on="month", how="left")
    result["pnl"] = result["pnl"].fillna(0.0)
    return result


def yearly_total(daily_df: pd.DataFrame, year: int) -> float:
    """Suma de P&L de todo el año (para la celda 'TOTAL' de la fila anual)."""
    summary = yearly_summary(daily_df, year)
    return float(summary["pnl"].sum())
