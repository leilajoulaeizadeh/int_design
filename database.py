from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

PROJECT_DIR = Path(__file__).resolve().parent
DATABASE_PATH = PROJECT_DIR / "furniture.db"

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS site (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS category (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS product (
    id INTEGER PRIMARY KEY,
    site_id INTEGER NOT NULL,
    category_id INTEGER,
    name TEXT NOT NULL,
    url TEXT,
    image_url TEXT,
    cleaned_image_data_url TEXT,
    price TEXT,
    scraped_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(site_id, name),
    FOREIGN KEY(site_id) REFERENCES site(id) ON DELETE CASCADE,
    FOREIGN KEY(category_id) REFERENCES category(id) ON DELETE SET NULL
);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DATABASE_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize_database() -> None:
    with get_connection() as conn:
        conn.executescript(CREATE_TABLES_SQL)


def _get_or_create_id(conn: sqlite3.Connection, table: str, name: str) -> int:
    if not name:
        raise ValueError("Name must be provided")
    conn.execute(f"INSERT OR IGNORE INTO {table} (name) VALUES (?)", (name,))
    row = conn.execute(f"SELECT id FROM {table} WHERE name = ?", (name,)).fetchone()
    if row is None:
        raise RuntimeError(f"Unable to create or find {table} record for {name}")
    return row["id"]


def upsert_products(site_name: str, products: Iterable[dict[str, str]]) -> None:
    initialize_database()
    with get_connection() as conn:
        site_id = _get_or_create_id(conn, "site", site_name)
        for product in products:
            category_name = product.get("category") or "Uncategorized"
            category_id = _get_or_create_id(conn, "category", category_name)
            conn.execute(
                """
                INSERT OR REPLACE INTO product
                (id, site_id, category_id, name, url, image_url, cleaned_image_data_url, price, scraped_at)
                VALUES (
                    COALESCE((SELECT id FROM product WHERE site_id = ? AND name = ?), NULL),
                    ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP
                )
                """,
                (
                    site_id,
                    product.get("name", ""),
                    site_id,
                    category_id,
                    product.get("name", ""),
                    product.get("url", ""),
                    product.get("image_url", ""),
                    product.get("cleaned_image_data_url", ""),
                    product.get("price", ""),
                ),
            )
        conn.commit()


def has_products() -> bool:
    initialize_database()
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(*) AS count FROM product").fetchone()
        return bool(row and row["count"] > 0)


def get_sites() -> list[str]:
    initialize_database()
    with get_connection() as conn:
        return [row["name"] for row in conn.execute("SELECT name FROM site ORDER BY name").fetchall()]


def get_categories() -> list[str]:
    initialize_database()
    with get_connection() as conn:
        return [row["name"] for row in conn.execute("SELECT name FROM category ORDER BY name").fetchall()]


def get_products(site: str | None = None, category: str | None = None, limit: int | None = None) -> list[dict[str, str]]:
    initialize_database()
    sql = """
    SELECT p.name, p.url, p.image_url, p.cleaned_image_data_url, p.price, s.name AS site, c.name AS category
    FROM product p
    JOIN site s ON p.site_id = s.id
    LEFT JOIN category c ON p.category_id = c.id
    """
    filters: list[str] = []
    params: list[str] = []
    if site:
        filters.append("s.name = ?")
        params.append(site)
    if category:
        filters.append("c.name = ?")
        params.append(category)
    if filters:
        sql += " WHERE " + " AND ".join(filters)
    sql += " ORDER BY s.name, c.name, p.name"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    with get_connection() as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
    return [
        {
            "site": row["site"],
            "category": row["category"] or "Uncategorized",
            "name": row["name"],
            "url": row["url"],
            "image_url": row["image_url"],
            "cleaned_image_data_url": row["cleaned_image_data_url"],
            "price": row["price"],
        }
        for row in rows
    ]
