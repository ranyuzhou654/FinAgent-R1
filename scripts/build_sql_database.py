"""Import FinQA tables into a SQLite database."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from datasets import load_from_disk


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATASET_PATH = ROOT_DIR / "data" / "raw" / "finqa_hf"
TABLE_DIR = ROOT_DIR / "data" / "tables"
DB_PATH = TABLE_DIR / "financial_data.db"


def sanitize_column_name(name: str, index: int) -> str:
    value = re.sub(r"[^\w\s]", "", str(name).strip())
    value = re.sub(r"\s+", "_", value).strip("_").lower()
    if not value:
        value = f"col_{index}"
    if value[0].isdigit():
        value = f"col_{value}"
    return value[:50]


def sanitize_value(value: object) -> str:
    text = str(value).strip()
    cleaned = re.sub(r"[$,%()]", "", text).replace(",", "")
    try:
        return str(float(cleaned))
    except ValueError:
        return text


def table_name_from_question(question_id: str) -> str:
    report_id = question_id.rsplit("-", 1)[0]
    return f"t_{report_id.replace('-', '_').replace('/', '_')}"


def main() -> None:
    dataset = load_from_disk(str(RAW_DATASET_PATH))
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    if DB_PATH.exists():
        DB_PATH.unlink()

    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS table_index (
            table_name TEXT PRIMARY KEY,
            report_id TEXT,
            question_id TEXT,
            num_rows INTEGER,
            num_cols INTEGER,
            headers TEXT
        )
        """
    )

    created = 0
    errors = 0

    for split in ("train", "validation", "test"):
        for example in dataset[split]:
            table = example.get("table") or []
            if len(table) < 2:
                continue

            question_id = example["id"]
            report_id = question_id.rsplit("-", 1)[0]
            table_name = table_name_from_question(question_id)

            cursor.execute("SELECT 1 FROM table_index WHERE table_name = ?", (table_name,))
            if cursor.fetchone():
                continue

            try:
                headers = table[0]
                raw_column_names = [sanitize_column_name(header, idx) for idx, header in enumerate(headers)]
                column_names: list[str] = []
                seen_names: dict[str, int] = {}
                for base_name in raw_column_names:
                    count = seen_names.get(base_name, 0)
                    seen_names[base_name] = count + 1
                    column_names.append(base_name if count == 0 else f"{base_name}_{count}")

                column_defs = ", ".join(f'"{name}" TEXT' for name in column_names)
                cursor.execute(f'CREATE TABLE "{table_name}" ({column_defs})')

                placeholders = ", ".join("?" for _ in column_names)
                for row in table[1:]:
                    if len(row) != len(column_names):
                        continue
                    values = [sanitize_value(value) for value in row]
                    cursor.execute(f'INSERT INTO "{table_name}" VALUES ({placeholders})', values)

                cursor.execute(
                    """
                    INSERT INTO table_index VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        table_name,
                        report_id,
                        question_id,
                        len(table) - 1,
                        len(column_names),
                        json.dumps(column_names),
                    ),
                )
                created += 1
            except Exception as exc:
                errors += 1
                if errors <= 5:
                    print(f"Warning: failed to build {table_name}: {exc}")

    connection.commit()
    cursor.execute("SELECT COUNT(*) FROM table_index")
    total = cursor.fetchone()[0]
    print("SQL database build complete.")
    print(f"  db_path: {DB_PATH}")
    print(f"  total_tables: {total}")
    print(f"  created: {created}")
    print(f"  errors: {errors}")

    cursor.execute("SELECT table_name, headers FROM table_index LIMIT 3")
    for table_name, headers in cursor.fetchall():
        print(f"  sample: {table_name} -> {headers}")

    connection.close()


if __name__ == "__main__":
    main()

