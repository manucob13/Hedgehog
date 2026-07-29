"""
db.py
Persistencia en SQLite de los datos importados desde Flex Query de IBKR.

Diseño:
    - trades: una fila por ejecución. Clave primaria = transactionID (IBKR
      garantiza que es único), así reimportar un XML con fechas solapadas
      no duplica filas: se actualizan (INSERT OR REPLACE).
    - equity_summary: una fila por día (reportDate como clave primaria).
      Reimportar actualiza el valor de ese día en vez de duplicarlo.
    - open_positions: NO es una serie temporal, es una "foto" del momento
      en que se generó el Flex Query. Cada importación reemplaza el
      contenido completo de la tabla (siempre refleja el último import).
    - cash_transactions: una fila por movimiento (dividendos, retenciones,
      intereses...). Clave primaria = transactionID, igual que trades.
    - target_allocation: tabla editada por el usuario en la app (no viene
      de IBKR), siempre refleja la última versión guardada.
    - import_log: registro de cada importación realizada (auditoría).
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Columnas que persistimos de cada sección (subconjunto relevante del XML)
# ---------------------------------------------------------------------------

TRADE_COLUMNS = [
    "accountId", "currency", "assetCategory", "subCategory", "symbol",
    "description", "conid", "underlyingSymbol", "strike", "putCall", "expiry",
    "multiplier", "tradeID", "transactionID", "tradeDate", "dateTime",
    "reportDate", "settleDateTarget", "transactionType", "exchange",
    "buySell", "quantity", "tradePrice", "tradeMoney", "proceeds", "taxes",
    "ibCommission", "ibCommissionCurrency", "netCash", "closePrice", "cost",
    "fifoPnlRealized", "mtmPnl", "openCloseIndicator", "notes", "orderType",
    "levelOfDetail",
]

EQUITY_SUMMARY_COLUMNS = [
    "accountId", "currency", "reportDate", "cash", "cashLong", "cashShort",
    "stock", "stockLong", "stockShort", "options", "optionsLong",
    "optionsShort", "total", "totalLong", "totalShort",
]

OPEN_POSITION_COLUMNS = [
    "accountId", "currency", "assetCategory", "subCategory", "symbol",
    "description", "conid", "underlyingSymbol", "strike", "putCall",
    "expiry", "multiplier", "reportDate", "position", "markPrice",
    "positionValue", "openPrice", "costBasisPrice", "costBasisMoney",
    "percentOfNAV", "fifoPnlUnrealized", "side",
]

CASH_TRANSACTION_COLUMNS = [
    "accountId", "currency", "assetCategory", "symbol", "description",
    "conid", "underlyingSymbol", "type", "dividendType", "amount",
    "dateTime", "reportDate", "settleDate", "exDate", "transactionID",
]

TARGET_ALLOCATION_COLUMNS = ["ticker", "description", "target_pct", "updated_at"]


# ---------------------------------------------------------------------------
# Conexión y creación de esquema
# ---------------------------------------------------------------------------

def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Abre (o crea) la base de datos SQLite en db_path."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: str | Path) -> None:
    """Crea las tablas si no existen."""
    conn = get_connection(db_path)
    try:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS trades (
                {", ".join(f'"{c}" TEXT' for c in TRADE_COLUMNS if c != "transactionID")},
                "transactionID" TEXT PRIMARY KEY
            )
        """)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS equity_summary (
                {", ".join(f'"{c}" TEXT' for c in EQUITY_SUMMARY_COLUMNS if c != "reportDate")},
                "reportDate" TEXT PRIMARY KEY
            )
        """)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS open_positions (
                {", ".join(f'"{c}" TEXT' for c in OPEN_POSITION_COLUMNS)}
            )
        """)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS cash_transactions (
                {", ".join(f'"{c}" TEXT' for c in CASH_TRANSACTION_COLUMNS if c != "transactionID")},
                "transactionID" TEXT PRIMARY KEY
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS target_allocation (
                "ticker" TEXT PRIMARY KEY,
                "description" TEXT,
                "target_pct" TEXT,
                "updated_at" TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS import_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                imported_at TEXT NOT NULL,
                source_file TEXT,
                from_date TEXT,
                to_date TEXT,
                n_trades INTEGER,
                n_equity_rows INTEGER,
                n_open_positions INTEGER,
                n_cash_transactions INTEGER
            )
        """)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Helpers de serialización (DataFrame -> filas SQLite)
# ---------------------------------------------------------------------------

def _serialize_value(v):
    """Convierte date/datetime/NaN a algo que SQLite pueda guardar como TEXT."""
    if v is None:
        return None
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    if isinstance(v, float) and pd.isna(v):
        return None
    return v


def _df_to_rows(df: pd.DataFrame, columns: list[str]) -> list[tuple]:
    """Prepara una lista de tuplas en el orden de `columns`, rellenando con
    None las columnas que falten en el DataFrame."""
    rows = []
    for _, record in df.iterrows():
        rows.append(tuple(
            _serialize_value(record[c]) if c in df.columns else None
            for c in columns
        ))
    return rows


# ---------------------------------------------------------------------------
# Upserts por sección
# ---------------------------------------------------------------------------

def upsert_trades(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Inserta/actualiza trades. Devuelve el número de filas procesadas."""
    if df.empty:
        return 0
    rows = _df_to_rows(df, TRADE_COLUMNS)
    placeholders = ", ".join("?" for _ in TRADE_COLUMNS)
    col_list = ", ".join(f'"{c}"' for c in TRADE_COLUMNS)
    conn.executemany(
        f"INSERT OR REPLACE INTO trades ({col_list}) VALUES ({placeholders})",
        rows,
    )
    return len(rows)


def upsert_equity_summary(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Inserta/actualiza filas diarias de NLV/caja/stock. Devuelve nº de filas."""
    if df.empty:
        return 0
    rows = _df_to_rows(df, EQUITY_SUMMARY_COLUMNS)
    placeholders = ", ".join("?" for _ in EQUITY_SUMMARY_COLUMNS)
    col_list = ", ".join(f'"{c}"' for c in EQUITY_SUMMARY_COLUMNS)
    conn.executemany(
        f"INSERT OR REPLACE INTO equity_summary ({col_list}) VALUES ({placeholders})",
        rows,
    )
    return len(rows)


def replace_open_positions(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Sustituye por completo la tabla de posiciones abiertas por la última
    foto importada. Devuelve el número de posiciones guardadas."""
    conn.execute("DELETE FROM open_positions")
    if df.empty:
        return 0
    rows = _df_to_rows(df, OPEN_POSITION_COLUMNS)
    placeholders = ", ".join("?" for _ in OPEN_POSITION_COLUMNS)
    col_list = ", ".join(f'"{c}"' for c in OPEN_POSITION_COLUMNS)
    conn.executemany(
        f"INSERT INTO open_positions ({col_list}) VALUES ({placeholders})",
        rows,
    )
    return len(rows)


def upsert_cash_transactions(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Inserta/actualiza movimientos de caja (dividendos, retenciones...).
    Devuelve el número de filas procesadas."""
    if df.empty:
        return 0
    rows = _df_to_rows(df, CASH_TRANSACTION_COLUMNS)
    placeholders = ", ".join("?" for _ in CASH_TRANSACTION_COLUMNS)
    col_list = ", ".join(f'"{c}"' for c in CASH_TRANSACTION_COLUMNS)
    conn.executemany(
        f"INSERT OR REPLACE INTO cash_transactions ({col_list}) VALUES ({placeholders})",
        rows,
    )
    return len(rows)


def log_import(
    conn: sqlite3.Connection,
    source_file: str | None,
    from_date,
    to_date,
    n_trades: int,
    n_equity_rows: int,
    n_open_positions: int,
    n_cash_transactions: int = 0,
) -> None:
    """Registra en import_log que se ha realizado una importación."""
    conn.execute(
        """
        INSERT INTO import_log
            (imported_at, source_file, from_date, to_date,
             n_trades, n_equity_rows, n_open_positions, n_cash_transactions)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(timespec="seconds"),
            source_file,
            _serialize_value(from_date),
            _serialize_value(to_date),
            n_trades,
            n_equity_rows,
            n_open_positions,
            n_cash_transactions,
        ),
    )


# ---------------------------------------------------------------------------
# Orquestador: importar el resultado de flex_query.import_flex_report()
# ---------------------------------------------------------------------------

def save_flex_import(
    db_path: str | Path,
    parsed_data: dict,
    source_file: str | None = None,
) -> dict:
    """
    Guarda en SQLite el resultado de `flex_query.import_flex_report(...)`.

    Devuelve un resumen con el número de filas guardadas en cada tabla.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        n_trades = upsert_trades(conn, parsed_data.get("trades", pd.DataFrame()))
        n_equity = upsert_equity_summary(conn, parsed_data.get("equity_summary", pd.DataFrame()))
        n_positions = replace_open_positions(conn, parsed_data.get("open_positions", pd.DataFrame()))
        n_cash_tx = upsert_cash_transactions(conn, parsed_data.get("cash_transactions", pd.DataFrame()))

        change_in_nav = parsed_data.get("change_in_nav") or {}
        log_import(
            conn,
            source_file=source_file,
            from_date=change_in_nav.get("fromDate"),
            to_date=change_in_nav.get("toDate"),
            n_trades=n_trades,
            n_equity_rows=n_equity,
            n_open_positions=n_positions,
            n_cash_transactions=n_cash_tx,
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "n_trades": n_trades,
        "n_equity_rows": n_equity,
        "n_open_positions": n_positions,
        "n_cash_transactions": n_cash_tx,
    }


# ---------------------------------------------------------------------------
# Lecturas (para las páginas de Streamlit)
# ---------------------------------------------------------------------------

def read_trades(
    db_path: str | Path,
    date_from: str | None = None,
    date_to: str | None = None,
) -> pd.DataFrame:
    """Lee trades, opcionalmente filtrados por tradeDate (formato 'YYYY-MM-DD')."""
    conn = get_connection(db_path)
    try:
        query = "SELECT * FROM trades WHERE 1=1"
        params: list = []
        if date_from:
            query += " AND tradeDate >= ?"
            params.append(date_from)
        if date_to:
            query += " AND tradeDate <= ?"
            params.append(date_to)
        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()

    for col in ["tradeDate", "reportDate", "expiry", "settleDateTarget"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["strike", "quantity", "tradePrice", "tradeMoney", "proceeds",
                "taxes", "ibCommission", "netCash", "closePrice", "cost",
                "fifoPnlRealized", "mtmPnl", "multiplier"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def read_equity_summary(
    db_path: str | Path,
    date_from: str | None = None,
    date_to: str | None = None,
) -> pd.DataFrame:
    """Lee la serie diaria de NLV/caja/stock, opcionalmente filtrada por fecha."""
    conn = get_connection(db_path)
    try:
        query = "SELECT * FROM equity_summary WHERE 1=1"
        params: list = []
        if date_from:
            query += " AND reportDate >= ?"
            params.append(date_from)
        if date_to:
            query += " AND reportDate <= ?"
            params.append(date_to)
        query += " ORDER BY reportDate"
        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()

    if "reportDate" in df.columns:
        df["reportDate"] = pd.to_datetime(df["reportDate"], errors="coerce")
    for col in ["cash", "cashLong", "cashShort", "stock", "stockLong",
                "stockShort", "options", "optionsLong", "optionsShort",
                "total", "totalLong", "totalShort"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def read_open_positions(db_path: str | Path) -> pd.DataFrame:
    """Lee la última foto de posiciones abiertas."""
    conn = get_connection(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM open_positions", conn)
    finally:
        conn.close()

    for col in ["strike", "position", "markPrice", "positionValue",
                "openPrice", "costBasisPrice", "costBasisMoney",
                "percentOfNAV", "fifoPnlUnrealized", "multiplier"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def read_cash_transactions(
    db_path: str | Path,
    date_from: str | None = None,
    date_to: str | None = None,
    tx_type: str | None = None,
) -> pd.DataFrame:
    """Lee movimientos de caja, opcionalmente filtrados por reportDate y/o tipo."""
    conn = get_connection(db_path)
    try:
        query = "SELECT * FROM cash_transactions WHERE 1=1"
        params: list = []
        if date_from:
            query += " AND reportDate >= ?"
            params.append(date_from)
        if date_to:
            query += " AND reportDate <= ?"
            params.append(date_to)
        if tx_type:
            query += " AND type = ?"
            params.append(tx_type)
        query += " ORDER BY reportDate"
        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()

    for col in ["dateTime", "reportDate", "settleDate", "exDate"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df


def read_import_log(db_path: str | Path) -> pd.DataFrame:
    """Lee el historial de importaciones realizadas."""
    conn = get_connection(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM import_log ORDER BY imported_at DESC", conn
        )
    finally:
        conn.close()
    return df


# ---------------------------------------------------------------------------
# Asignación objetivo (editada por el usuario en la propia app)
# ---------------------------------------------------------------------------

def _normalize_allocation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Acepta variantes razonables de nombres de columna
    (Ticker/ticker, Target_%/target_pct/Target %..., Descr./description...)
    y las normaliza a 'ticker', 'description' y 'target_pct'.
    """
    rename_map = {}
    for col in df.columns:
        key = col.strip().lower()
        if "ticker" in key or key == "symbol":
            rename_map[col] = "ticker"
        elif "descr" in key or key == "description":
            rename_map[col] = "description"
        elif "target" in key or "%" in key or "pct" in key or "obj" in key:
            rename_map[col] = "target_pct"
    return df.rename(columns=rename_map)


def save_target_allocation(db_path: str | Path, df: pd.DataFrame) -> int:
    """
    Sustituye por completo la tabla de asignación objetivo por la que se
    acaba de guardar (mismo patrón que open_positions: siempre refleja la
    última versión, no es histórico).

    df puede traer variantes de nombres de columna (Ticker, Target_%,
    Descr., etc.), se normalizan automáticamente. La columna 'description'
    es opcional. Devuelve el número de tickers guardados.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        conn.execute("DELETE FROM target_allocation")
        if df.empty:
            conn.commit()
            return 0

        df = _normalize_allocation_columns(df)
        if "description" not in df.columns:
            df["description"] = ""
        now = datetime.now().isoformat(timespec="seconds")

        rows = [
            (
                str(row["ticker"]).upper().strip(),
                str(row.get("description", "") or ""),
                float(row["target_pct"]),
                now,
            )
            for _, row in df.iterrows()
        ]
        conn.executemany(
            'INSERT OR REPLACE INTO target_allocation '
            '("ticker", "description", "target_pct", "updated_at") '
            "VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def read_target_allocation(db_path: str | Path) -> pd.DataFrame:
    """Lee la tabla de asignación objetivo actual."""
    conn = get_connection(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM target_allocation", conn)
    finally:
        conn.close()

    if "target_pct" in df.columns:
        df["target_pct"] = pd.to_numeric(df["target_pct"], errors="coerce")
    return df
