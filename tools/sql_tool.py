"""SQLite access utilities for FinAgent-R1 financial tables."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = ROOT_DIR / "data" / "tables" / "financial_data.db"
QUESTION_TABLE_MAP_PATH = ROOT_DIR / "data" / "tables" / "question_table_map.json"
FORBIDDEN_KEYWORDS = {
    "DROP",
    "DELETE",
    "UPDATE",
    "INSERT",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "ATTACH",
    "DETACH",
    "REPLACE",
}


def _load_question_table_map() -> dict[str, str]:
    if not QUESTION_TABLE_MAP_PATH.exists():
        return {}
    with QUESTION_TABLE_MAP_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


QUESTION_TABLE_MAP = _load_question_table_map()


def _connect() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"SQLite database not found: {DB_PATH}")
    return sqlite3.connect(DB_PATH)


def _normalize_query(query: str, question_id: str | None = None) -> str:
    normalized = query.strip()
    if question_id and question_id in QUESTION_TABLE_MAP:
        table_name = QUESTION_TABLE_MAP[question_id]
        normalized = normalized.replace("{table}", table_name)
        normalized = normalized.replace("financial_table", table_name)
        normalized = normalized.replace("current_table", table_name)
    return normalized


def get_table_schema(table_name: str) -> str:
    connection = _connect()
    cursor = connection.cursor()
    try:
        cursor.execute(f'PRAGMA table_info("{table_name}")')
        columns = cursor.fetchall()
        if not columns:
            return f"Table '{table_name}' not found."

        lines = [f"Table: {table_name}", "Columns:"]
        for column in columns:
            lines.append(f"  - {column[1]} ({column[2]})")

        cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 3')
        rows = cursor.fetchall()
        if rows:
            lines.append("")
            lines.append("Sample rows:")
            for row in rows:
                lines.append("  " + " | ".join(str(value) for value in row))
        return "\n".join(lines)
    except Exception as exc:
        return f"SQL schema error: {exc}"
    finally:
        connection.close()


def get_available_tables(limit: int = 10) -> str:
    connection = _connect()
    cursor = connection.cursor()
    cursor.execute("SELECT table_name, num_rows, num_cols, headers FROM table_index LIMIT ?", (limit,))
    rows = cursor.fetchall()
    connection.close()

    if not rows:
        return "No tables available."
    output = ["Available tables:"]
    for table_name, num_rows, num_cols, headers in rows:
        output.append(f"  {table_name} ({num_rows} rows, {num_cols} cols) - {headers}")
    return "\n".join(output)


def execute_sql(query: str, question_id: str | None = None) -> str:
    query = _normalize_query(query, question_id=question_id)
    if not query:
        return "SQL Error: empty query"

    upper_query = query.upper()
    if any(keyword in upper_query for keyword in FORBIDDEN_KEYWORDS):
        return "SQL Error: only read-only SELECT/PRAGMA/DESCRIBE queries are allowed."

    describe_match = re.fullmatch(r"(?:DESCRIBE|SHOW\s+SCHEMA)\s+([A-Za-z0-9_]+)", query, re.IGNORECASE)
    if describe_match:
        return get_table_schema(describe_match.group(1))

    if re.fullmatch(r"SHOW\s+TABLES", query, re.IGNORECASE):
        return get_available_tables()

    connection = _connect()
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        if cursor.description is None:
            return "Query executed successfully."
        columns = [description[0] for description in cursor.description]
        if not rows:
            return "Query returned no results."

        lines = [f"Columns: {', '.join(columns)}", "-" * 60]
        for row in rows[:20]:
            lines.append(" | ".join(str(value) for value in row))
        if len(rows) > 20:
            lines.append(f"... ({len(rows)} total rows, showing first 20)")
        return "\n".join(lines)
    except Exception as exc:
        return f"SQL Error: {exc}\nQuery: {query}"
    finally:
        connection.close()

