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
    color TEXT,
    rating TEXT,
    scraped_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(site_id, name),
    FOREIGN KEY(site_id) REFERENCES site(id) ON DELETE CASCADE,
    FOREIGN KEY(category_id) REFERENCES category(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS product_3d (
    id INTEGER PRIMARY KEY,
    product_id INTEGER NOT NULL UNIQUE,
    model_type TEXT NOT NULL,
    width_m REAL NOT NULL,
    height_m REAL NOT NULL,
    depth_m REAL NOT NULL,
    texture_data_url TEXT,
    mesh_path TEXT,
    rotated_0_path TEXT,
    rotated_90_path TEXT,
    rotated_minus90_path TEXT,
    rotated_180_path TEXT,
    model_data_json TEXT,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(product_id) REFERENCES product(id) ON DELETE CASCADE
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
        # Migrate: add color and rating columns if they don't exist yet
        existing = {row[1] for row in conn.execute("PRAGMA table_info(product)").fetchall()}
        if "color" not in existing:
            conn.execute("ALTER TABLE product ADD COLUMN color TEXT")
        if "rating" not in existing:
            conn.execute("ALTER TABLE product ADD COLUMN rating TEXT")

        existing_3d = {row[1] for row in conn.execute("PRAGMA table_info(product_3d)").fetchall()}
        if "model_data_json" not in existing_3d:
            conn.execute("ALTER TABLE product_3d ADD COLUMN model_data_json TEXT")
        if "mesh_path" not in existing_3d:
            conn.execute("ALTER TABLE product_3d ADD COLUMN mesh_path TEXT")
        if "rotated_0_path" not in existing_3d:
            conn.execute("ALTER TABLE product_3d ADD COLUMN rotated_0_path TEXT")
        if "rotated_90_path" not in existing_3d:
            conn.execute("ALTER TABLE product_3d ADD COLUMN rotated_90_path TEXT")
        if "rotated_minus90_path" not in existing_3d:
            conn.execute("ALTER TABLE product_3d ADD COLUMN rotated_minus90_path TEXT")
        if "rotated_180_path" not in existing_3d:
            conn.execute("ALTER TABLE product_3d ADD COLUMN rotated_180_path TEXT")
        conn.commit()


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
                (id, site_id, category_id, name, url, image_url, cleaned_image_data_url, price, color, rating, scraped_at)
                VALUES (
                    COALESCE((SELECT id FROM product WHERE site_id = ? AND name = ?), NULL),
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP
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
                    product.get("color", ""),
                    product.get("rating", ""),
                ),
            )
        conn.commit()


def delete_products_for_site(site_name: str) -> None:
    initialize_database()
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM product WHERE site_id IN (SELECT id FROM site WHERE name = ?)",
            (site_name,),
        )
        conn.execute("DELETE FROM site WHERE name = ?", (site_name,))
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


def get_colors() -> list[str]:
    initialize_database()
    with get_connection() as conn:
        return [
            row[0] for row in conn.execute(
                "SELECT DISTINCT color FROM product WHERE color IS NOT NULL AND color != '' ORDER BY color"
            ).fetchall()
        ]


def get_ratings() -> list[str]:
    initialize_database()
    with get_connection() as conn:
        return [
            row[0] for row in conn.execute(
                "SELECT DISTINCT rating FROM product WHERE rating IS NOT NULL AND rating != '' ORDER BY rating"
            ).fetchall()
        ]


def get_products(site: str | None = None, category: str | None = None, limit: int | None = None) -> list[dict[str, str]]:
    initialize_database()
    sql = """
    SELECT p.id, p.name, p.url, p.image_url, p.cleaned_image_data_url, p.price, p.color, p.rating,
           s.name AS site, c.name AS category
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
            "id": str(row["id"]),
            "site": row["site"],
            "category": row["category"] or "Uncategorized",
            "name": row["name"],
            "url": row["url"],
            "image_url": row["image_url"],
            "cleaned_image_data_url": row["cleaned_image_data_url"],
            "price": row["price"],
            "color": row["color"] or "",
            "rating": row["rating"] or "",
        }
        for row in rows
    ]


def clear_catalog_data() -> None:
    """Remove all catalog records for fast local development resets."""
    initialize_database()
    with get_connection() as conn:
        conn.execute("DELETE FROM product_3d")
        conn.execute("DELETE FROM product")
        conn.execute("DELETE FROM category")
        conn.execute("DELETE FROM site")
        conn.commit()


def _get_product_id(conn: sqlite3.Connection, site_name: str, product_name: str) -> int | None:
    row = conn.execute(
        """
        SELECT p.id
        FROM product p
        JOIN site s ON p.site_id = s.id
        WHERE s.name = ? AND p.name = ?
        """,
        (site_name, product_name),
    ).fetchone()
    return None if row is None else int(row["id"])


def upsert_product_3d_model(
    *,
    site_name: str,
    product_name: str,
    model_type: str,
    width_m: float,
    height_m: float,
    depth_m: float,
    texture_data_url: str,
    mesh_path: str = "",
    rotated_0_path: str = "",
    rotated_90_path: str = "",
    rotated_minus90_path: str = "",
    rotated_180_path: str = "",
    model_data_json: str = "",
) -> None:
    """Create or update a stored 3D model payload for a product."""
    initialize_database()
    with get_connection() as conn:
        product_id = _get_product_id(conn, site_name=site_name, product_name=product_name)
        if product_id is None:
            raise ValueError(f"Product not found for 3D model: {site_name} / {product_name}")

        conn.execute(
            """
            INSERT INTO product_3d (
                product_id,
                model_type,
                width_m,
                height_m,
                depth_m,
                texture_data_url,
                mesh_path,
                rotated_0_path,
                rotated_90_path,
                rotated_minus90_path,
                rotated_180_path,
                model_data_json,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(product_id) DO UPDATE SET
                model_type = excluded.model_type,
                width_m = excluded.width_m,
                height_m = excluded.height_m,
                depth_m = excluded.depth_m,
                texture_data_url = excluded.texture_data_url,
                mesh_path = excluded.mesh_path,
                rotated_0_path = excluded.rotated_0_path,
                rotated_90_path = excluded.rotated_90_path,
                rotated_minus90_path = excluded.rotated_minus90_path,
                rotated_180_path = excluded.rotated_180_path,
                model_data_json = excluded.model_data_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                product_id,
                model_type,
                width_m,
                height_m,
                depth_m,
                texture_data_url,
                mesh_path,
                rotated_0_path,
                rotated_90_path,
                rotated_minus90_path,
                rotated_180_path,
                model_data_json,
            ),
        )
        conn.commit()


def get_products_with_3d_models(limit: int | None = None) -> list[dict[str, str | float]]:
    """Return products that have a stored 3D model."""
    initialize_database()
    sql = """
    SELECT
        p.name,
        s.name AS site,
        c.name AS category,
        p.url,
        p.image_url,
        p.cleaned_image_data_url,
        p.price,
        m.model_type,
        m.width_m,
        m.height_m,
        m.depth_m,
        m.texture_data_url,
        m.mesh_path,
        m.rotated_0_path,
        m.rotated_90_path,
        m.rotated_minus90_path,
        m.rotated_180_path,
        m.model_data_json
    FROM product_3d m
    JOIN product p ON m.product_id = p.id
    JOIN site s ON p.site_id = s.id
    LEFT JOIN category c ON p.category_id = c.id
    ORDER BY s.name, c.name, p.name
    """
    params: list[int] = []
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
            "url": row["url"] or "",
            "image_url": row["image_url"] or "",
            "cleaned_image_data_url": row["cleaned_image_data_url"] or "",
            "price": row["price"] or "",
            "model_type": row["model_type"],
            "width_m": float(row["width_m"]),
            "height_m": float(row["height_m"]),
            "depth_m": float(row["depth_m"]),
            "texture_data_url": row["texture_data_url"] or "",
            "mesh_path": row["mesh_path"] or "",
            "rotated_0_path": row["rotated_0_path"] or "",
            "rotated_90_path": row["rotated_90_path"] or "",
            "rotated_minus90_path": row["rotated_minus90_path"] or "",
            "rotated_180_path": row["rotated_180_path"] or "",
            "model_data_json": row["model_data_json"] or "",
        }
        for row in rows
    ]
