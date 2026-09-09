#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = PROJECT_DIR / "furniture.db"


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def print_tables(conn: sqlite3.Connection) -> None:
    print("Tables:")
    for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        print(f"- {row['name']}")


def print_schema(conn: sqlite3.Connection, table: str | None = None) -> None:
    if table:
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND name = ? ORDER BY name",
            (table,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()

    if not rows:
        print("No matching table schema found.")
        return

    for row in rows:
        print(f"\n--- {row['name']} ---")
        print(row["sql"])


def print_counts(conn: sqlite3.Connection) -> None:
    print("Row counts:")
    tables = [r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    for table in tables:
        count = conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"]
        print(f"- {table}: {count}")


def print_products(conn: sqlite3.Connection, limit: int) -> None:
    query = """
    SELECT p.id, s.name AS site, c.name AS category, p.name, p.status, p.price
    FROM product p
    JOIN site s ON p.site_id = s.id
    LEFT JOIN category c ON p.category_id = c.id
    ORDER BY p.id
    LIMIT ?
    """
    rows = conn.execute(query, (limit,)).fetchall()

    print(f"Products (limit={limit}):")
    for row in rows:
        print(dict(row))


def print_models(conn: sqlite3.Connection, limit: int) -> None:
    query = """
    SELECT
      m.product_id,
      s.name AS site,
      p.name,
      m.model_type,
      m.status,
      m.mesh_path,
      m.output_dir,
      m.updated_at
    FROM product_3d m
    JOIN product p ON p.id = m.product_id
    JOIN site s ON p.site_id = s.id
    ORDER BY m.updated_at DESC
    LIMIT ?
    """
    rows = conn.execute(query, (limit,)).fetchall()

    print(f"Product 3D models (limit={limit}):")
    for row in rows:
        print(dict(row))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect int_design SQLite database.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Path to SQLite database")
    parser.add_argument("--all", action="store_true", help="Show tables, counts, products, and 3D models")
    parser.add_argument("--tables", action="store_true", help="Show table names")
    parser.add_argument("--schema", action="store_true", help="Show table schema")
    parser.add_argument("--table", type=str, default=None, help="Optional table name for --schema")
    parser.add_argument("--counts", action="store_true", help="Show row counts")
    parser.add_argument("--products", action="store_true", help="Show sample products")
    parser.add_argument("--models", action="store_true", help="Show sample product_3d rows")
    parser.add_argument("--limit", type=int, default=10, help="Limit for row previews")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    db_path = args.db.resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = connect(db_path)
    try:
        print(f"DB: {db_path}")

        selected_any = any([args.all, args.tables, args.schema, args.counts, args.products, args.models])

        if args.all or not selected_any:
            print_tables(conn)
            print()
            print_counts(conn)
            print()
            print_products(conn, args.limit)
            print()
            print_models(conn, args.limit)
            return

        if args.tables:
            print_tables(conn)
            print()
        if args.schema:
            print_schema(conn, args.table)
            print()
        if args.counts:
            print_counts(conn)
            print()
        if args.products:
            print_products(conn, args.limit)
            print()
        if args.models:
            print_models(conn, args.limit)
            print()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
