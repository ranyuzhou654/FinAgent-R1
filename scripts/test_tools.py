"""Basic integration checks for the FinAgent-R1 tool stack."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.calculator_tool import execute_calculate
from tools.search_tool import execute_search
from tools.sql_tool import execute_sql, get_available_tables
from tools.tool_dispatcher import detect_tool_call

DB_PATH = ROOT_DIR / "data" / "tables" / "financial_data.db"


def main() -> None:
    print("=" * 60)
    print("Tool 1: Financial search")
    print("=" * 60)
    try:
        result = execute_search("What was the revenue growth in 2019?")
        print(result[:300] + ("\n..." if len(result) > 300 else ""))
    except Exception as exc:
        print(f"Search unavailable: {exc}")

    print("\n" + "=" * 60)
    print("Tool 2: Calculator")
    print("=" * 60)
    for expression in (
        "pct_change: 1829, 1731",
        "subtract(1829, 1731), divide(#0, 1731)",
        "cagr: 100, 150, 3",
        "(1829 - 1731) / 1731 * 100",
    ):
        print(f"{expression} -> {execute_calculate(expression)}")

    print("\n" + "=" * 60)
    print("Tool 3: SQL")
    print("=" * 60)
    try:
        print(get_available_tables())
        if DB_PATH.exists():
            connection = sqlite3.connect(DB_PATH)
            cursor = connection.cursor()
            cursor.execute("SELECT table_name FROM table_index LIMIT 1")
            row = cursor.fetchone()
            connection.close()
            if row:
                print(execute_sql(f'SELECT * FROM "{row[0]}" LIMIT 3'))
    except Exception as exc:
        print(f"SQL unavailable: {exc}")

    print("\n" + "=" * 60)
    print("Dispatcher")
    print("=" * 60)
    cases = [
        "Let me search. <search>revenue growth 2019</search>",
        "Need math. <calculate>(1829 - 1731) / 1731</calculate>",
        "<sql>SHOW TABLES</sql>",
        "Just thinking, no tool call.",
    ]
    for case in cases:
        print(f"{case!r} -> {detect_tool_call(case)}")


if __name__ == "__main__":
    main()
