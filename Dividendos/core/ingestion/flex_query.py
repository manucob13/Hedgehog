"""
flex_query.py
Parseo de reportes Flex Query (Activity Flex Query, formato XML) de IBKR.

Convierte cada sección relevante del XML en un DataFrame de pandas limpio,
con los tipos de dato correctos (fechas como date, numéricos como float/int).

Secciones cubiertas:
    - Trades              -> parse_trades()
    - OpenPositions        -> parse_open_positions()
    - EquitySummaryInBase  -> parse_equity_summary()      (NLV/caja/stock por día)
    - CashTransactions     -> parse_cash_transactions()   (dividendos, retenciones...)
    - ChangeInNAV          -> parse_change_in_nav()        (resumen del período)
    - CashReport           -> parse_cash_report()          (resumen del período)
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _parse_ibkr_date(value: str):
    """Convierte 'YYYYMMDD' -> date. Devuelve None si viene vacío."""
    if not value:
        return None
    return datetime.strptime(value, "%Y%m%d").date()


def _parse_ibkr_datetime(value: str):
    """Convierte 'YYYYMMDD;HHMMSS' -> datetime. Devuelve None si viene vacío."""
    if not value:
        return None
    return datetime.strptime(value, "%Y%m%d;%H%M%S")


def _to_float(value: str):
    """Convierte a float; cadena vacía -> None (no 0, para no falsear datos)."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _get_statement(xml_path: str | Path) -> ET.Element:
    """Carga el XML y devuelve el nodo <FlexStatement>."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    stmt = root.find(".//FlexStatement")
    if stmt is None:
        raise ValueError("No se encontró <FlexStatement> en el XML. ¿Es un Activity Flex Query?")
    return stmt


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------

_TRADE_NUMERIC_FIELDS = [
    "strike", "quantity", "tradePrice", "tradeMoney", "proceeds", "taxes",
    "ibCommission", "netCash", "closePrice", "cost", "fifoPnlRealized",
    "mtmPnl", "multiplier", "fxRateToBase",
]

_TRADE_DATE_FIELDS = ["tradeDate", "reportDate", "expiry", "settleDateTarget"]
_TRADE_DATETIME_FIELDS = ["dateTime", "orderTime"]


def parse_trades(xml_path: str | Path) -> pd.DataFrame:
    """Parsea la sección <Trades> a un DataFrame, una fila por ejecución."""
    stmt = _get_statement(xml_path)
    trades_node = stmt.find("Trades")
    if trades_node is None:
        return pd.DataFrame()

    rows = [t.attrib for t in trades_node.findall("Trade")]
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in _TRADE_NUMERIC_FIELDS:
        if col in df.columns:
            df[col] = df[col].apply(_to_float)

    for col in _TRADE_DATE_FIELDS:
        if col in df.columns:
            df[col] = df[col].apply(_parse_ibkr_date)

    for col in _TRADE_DATETIME_FIELDS:
        if col in df.columns:
            df[col] = df[col].apply(_parse_ibkr_datetime)

    for col in ["underlyingSymbol", "symbol", "putCall", "strike", "expiry",
                "openCloseIndicator", "buySell", "assetCategory"]:
        if col not in df.columns:
            df[col] = None

    return df


# ---------------------------------------------------------------------------
# Open Positions
# ---------------------------------------------------------------------------

_POSITION_NUMERIC_FIELDS = [
    "strike", "position", "markPrice", "positionValue", "openPrice",
    "costBasisPrice", "costBasisMoney", "percentOfNAV", "fifoPnlUnrealized",
    "multiplier", "fxRateToBase",
]
_POSITION_DATE_FIELDS = ["reportDate", "expiry"]


def parse_open_positions(xml_path: str | Path) -> pd.DataFrame:
    """Parsea <OpenPositions> a un DataFrame, una fila por posición abierta."""
    stmt = _get_statement(xml_path)
    node = stmt.find("OpenPositions")
    if node is None:
        return pd.DataFrame()

    rows = [p.attrib for p in node.findall("OpenPosition")]
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in _POSITION_NUMERIC_FIELDS:
        if col in df.columns:
            df[col] = df[col].apply(_to_float)

    for col in _POSITION_DATE_FIELDS:
        if col in df.columns:
            df[col] = df[col].apply(_parse_ibkr_date)

    return df


# ---------------------------------------------------------------------------
# Equity Summary (NLV / caja / stock por día) -> la base del calendario
# ---------------------------------------------------------------------------

_EQUITY_NUMERIC_FIELDS = [
    "cash", "cashLong", "cashShort", "stock", "stockLong", "stockShort",
    "options", "optionsLong", "optionsShort",
    "total", "totalLong", "totalShort",
]


def parse_equity_summary(xml_path: str | Path) -> pd.DataFrame:
    """
    Parsea <EquitySummaryInBase> a un DataFrame con una fila por día.

    Esta es la fuente principal para:
      - NLV histórico (columna 'total')
      - P&L diario del calendario (diff de 'total' entre días consecutivos)
    """
    stmt = _get_statement(xml_path)
    node = stmt.find("EquitySummaryInBase")
    if node is None:
        return pd.DataFrame()

    rows = [e.attrib for e in node.findall("EquitySummaryByReportDateInBase")]
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in _EQUITY_NUMERIC_FIELDS:
        if col in df.columns:
            df[col] = df[col].apply(_to_float)

    df["reportDate"] = df["reportDate"].apply(_parse_ibkr_date)
    df = df.sort_values("reportDate").reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Cash Transactions (dividendos, retenciones de impuestos, intereses...)
# ---------------------------------------------------------------------------

_CASH_TX_NUMERIC_FIELDS = ["amount", "fxRateToBase", "multiplier"]
_CASH_TX_DATE_FIELDS = ["reportDate", "settleDate", "exDate"]
_CASH_TX_DATETIME_FIELDS = ["dateTime"]


def parse_cash_transactions(xml_path: str | Path) -> pd.DataFrame:
    """
    Parsea <CashTransactions> a un DataFrame, una fila por movimiento de caja.

    Tipos habituales en `type`:
        - "Dividends"                      -> dividendo cobrado
        - "Payment In Lieu Of Dividends"    -> pago sustitutivo (común en ETFs
          cuando la posición estuvo prestada en la fecha ex-dividendo)
        - "Withholding Tax"                 -> retención de impuestos sobre
          el dividendo anterior (viene como fila aparte, típicamente negativa)
        - "Broker Interest Received/Paid", "Deposits/Withdrawals", etc.
    """
    stmt = _get_statement(xml_path)
    node = stmt.find("CashTransactions")
    if node is None:
        return pd.DataFrame()

    rows = [t.attrib for t in node.findall("CashTransaction")]
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in _CASH_TX_NUMERIC_FIELDS:
        if col in df.columns:
            df[col] = df[col].apply(_to_float)

    for col in _CASH_TX_DATE_FIELDS:
        if col in df.columns:
            df[col] = df[col].apply(_parse_ibkr_date)

    for col in _CASH_TX_DATETIME_FIELDS:
        if col in df.columns:
            df[col] = df[col].apply(_parse_ibkr_datetime)

    for col in ["symbol", "underlyingSymbol", "type", "dividendType", "transactionID"]:
        if col not in df.columns:
            df[col] = None

    return df


# ---------------------------------------------------------------------------
# Change in NAV (resumen del período solicitado, no diario)
# ---------------------------------------------------------------------------

def parse_change_in_nav(xml_path: str | Path) -> dict | None:
    """Parsea <ChangeInNAV> (un único resumen del rango de fechas del reporte)."""
    stmt = _get_statement(xml_path)
    node = stmt.find("ChangeInNAV")
    if node is None:
        return None

    data = dict(node.attrib)
    for key in ("startingValue", "endingValue", "mtm", "realized",
                "commissions", "twr", "dividends", "interest"):
        if key in data:
            data[key] = _to_float(data[key])
    for key in ("fromDate", "toDate"):
        if key in data:
            data[key] = _parse_ibkr_date(data[key])

    return data


# ---------------------------------------------------------------------------
# Cash Report (resumen del período solicitado, no diario)
# ---------------------------------------------------------------------------

def parse_cash_report(xml_path: str | Path) -> dict | None:
    """Parsea la fila BASE_SUMMARY de <CashReport> (resumen del período)."""
    stmt = _get_statement(xml_path)
    node = stmt.find("CashReport")
    if node is None:
        return None

    base_row = None
    for currency_node in node.findall("CashReportCurrency"):
        if currency_node.attrib.get("currency") == "BASE_SUMMARY":
            base_row = currency_node.attrib
            break
    if base_row is None:
        return None

    data = dict(base_row)
    for key in ("startingCash", "endingCash", "commissions", "depositWithdrawals",
                "netTradesSales", "netTradesPurchases"):
        if key in data:
            data[key] = _to_float(data[key])
    for key in ("fromDate", "toDate"):
        if key in data:
            data[key] = _parse_ibkr_date(data[key])

    return data


# ---------------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------------

def import_flex_report(xml_path: str | Path) -> dict:
    """
    Parsea todas las secciones de un Flex Query y las devuelve en un dict:
        {
            "trades": DataFrame,
            "open_positions": DataFrame,
            "equity_summary": DataFrame,
            "cash_transactions": DataFrame,
            "change_in_nav": dict | None,
            "cash_report": dict | None,
        }
    """
    return {
        "trades": parse_trades(xml_path),
        "open_positions": parse_open_positions(xml_path),
        "equity_summary": parse_equity_summary(xml_path),
        "cash_transactions": parse_cash_transactions(xml_path),
        "change_in_nav": parse_change_in_nav(xml_path),
        "cash_report": parse_cash_report(xml_path),
    }
