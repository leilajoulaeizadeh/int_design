"""Storage for the multi-object detection feature — kept in its own SQLite
database (``data/multi_object.db``), completely separate from ``furniture.db``
so these experimental "New Developments" results never mix with the existing
scraped catalog / 3D viewer tabs.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent / "data" / "multi_object.db"

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS scene (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    image_path TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS detected_object (
    id INTEGER PRIMARY KEY,
    scene_id INTEGER NOT NULL,
    label TEXT NOT NULL,
    score REAL NOT NULL,
    crop_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'detected',
    mesh_path TEXT,
    width_m REAL,
    height_m REAL,
    depth_m REAL,
    error_message TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(scene_id) REFERENCES scene(id) ON DELETE CASCADE
);
"""


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize_database() -> None:
    with get_connection() as conn:
        conn.executescript(CREATE_TABLES_SQL)
        conn.commit()


def create_scene(source: str, image_path: str) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO scene (source, image_path) VALUES (?, ?)", (source, image_path)
        )
        conn.commit()
        return int(cur.lastrowid)


def add_detected_object(scene_id: int, label: str, score: float, crop_path: str) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO detected_object (scene_id, label, score, crop_path) VALUES (?, ?, ?, ?)",
            (scene_id, label, score, crop_path),
        )
        conn.commit()
        return int(cur.lastrowid)


def update_object_reconstructed(
    object_id: int, mesh_path: str, width_m: float, height_m: float, depth_m: float
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE detected_object
            SET status = 'reconstructed', mesh_path = ?, width_m = ?, height_m = ?, depth_m = ?, error_message = NULL
            WHERE id = ?
            """,
            (mesh_path, width_m, height_m, depth_m, object_id),
        )
        conn.commit()


def update_object_failed(object_id: int, error_message: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE detected_object SET status = 'failed', error_message = ? WHERE id = ?",
            (error_message, object_id),
        )
        conn.commit()


def list_scenes() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM scene ORDER BY id DESC").fetchall()
        return [dict(row) for row in rows]


def list_objects(scene_id: int) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM detected_object WHERE scene_id = ? ORDER BY id", (scene_id,)
        ).fetchall()
        return [dict(row) for row in rows]


def get_object(object_id: int) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM detected_object WHERE id = ?", (object_id,)
        ).fetchone()
        return dict(row) if row else None
