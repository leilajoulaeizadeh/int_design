#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

DB_PATH="${DB_PATH:-$PROJECT_DIR/furniture.db}"
INPUT_DIR="${INPUT_DIR:-$PROJECT_DIR/data/input}"
MODELS_DIR="${MODELS_DIR:-$PROJECT_DIR/data/models}"
WORK_ROOT="${WORK_ROOT:-$PROJECT_DIR/data/output/triposr_work}"
ROTATED_OUTPUT_DIR="${ROTATED_OUTPUT_DIR:-$PROJECT_DIR/data/output/rotated_products}"
STATIC_MESH_DIR="${STATIC_MESH_DIR:-$PROJECT_DIR/static/meshes}"
LIMIT=""
INSTALL_DEPS=0
NO_REMOVE_BG=0

usage() {
  cat <<'EOF'
Usage: ./process_batch.sh [--limit N] [--install-deps] [--no-remove-bg]

Options:
  --limit N         Process at most N cleaned products in this run.
  --install-deps    Pass --install-deps to triposr_cpu_pipeline.py.
  --no-remove-bg    Pass --no-remove-bg to triposr_cpu_pipeline.py.

Environment overrides:
  DB_PATH           SQLite database path (default: ./furniture.db)
  INPUT_DIR         Input image folder (default: ./data/input)
  MODELS_DIR        Per-product output folder (default: ./data/models)
  WORK_ROOT         TriPoSR work folder (default: ./data/output/triposr_work)
  ROTATED_OUTPUT_DIR Rotated views output folder (default: ./data/output/rotated_products)
  STATIC_MESH_DIR   Public mesh folder for Streamlit (default: ./static/meshes)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit)
      LIMIT="${2:-}"
      shift 2
      ;;
    --install-deps)
      INSTALL_DEPS=1
      shift
      ;;
    --no-remove-bg)
      NO_REMOVE_BG=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -n "$LIMIT" && ! "$LIMIT" =~ ^[0-9]+$ ]]; then
  echo "--limit must be a non-negative integer" >&2
  exit 1
fi

mkdir -p "$INPUT_DIR" "$MODELS_DIR" "$WORK_ROOT" "$ROTATED_OUTPUT_DIR" "$STATIC_MESH_DIR"
PYTHON_CMD=(python)
if command -v uv >/dev/null 2>&1; then
  PYTHON_CMD=(uv run python)
fi

echo "Bootstrapping DB status columns..."
DB_PATH="$DB_PATH" "${PYTHON_CMD[@]}" - <<'PY'
import os
import sqlite3

db_path = os.environ["DB_PATH"]
conn = sqlite3.connect(db_path)


def has_column(table_name: str, column_name: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(row[1] == column_name for row in rows)


if not has_column("product", "status"):
    conn.execute("ALTER TABLE product ADD COLUMN status TEXT")

if not has_column("product_3d", "status"):
    conn.execute("ALTER TABLE product_3d ADD COLUMN status TEXT")

if not has_column("product_3d", "output_dir"):
    conn.execute("ALTER TABLE product_3d ADD COLUMN output_dir TEXT")

if not has_column("product_3d", "error_message"):
    conn.execute("ALTER TABLE product_3d ADD COLUMN error_message TEXT")

if not has_column("product_3d", "mesh_path"):
  conn.execute("ALTER TABLE product_3d ADD COLUMN mesh_path TEXT")

if not has_column("product_3d", "rotated_0_path"):
  conn.execute("ALTER TABLE product_3d ADD COLUMN rotated_0_path TEXT")

if not has_column("product_3d", "rotated_90_path"):
  conn.execute("ALTER TABLE product_3d ADD COLUMN rotated_90_path TEXT")

if not has_column("product_3d", "rotated_minus90_path"):
  conn.execute("ALTER TABLE product_3d ADD COLUMN rotated_minus90_path TEXT")

if not has_column("product_3d", "rotated_180_path"):
  conn.execute("ALTER TABLE product_3d ADD COLUMN rotated_180_path TEXT")

conn.execute(
    """
    CREATE TABLE IF NOT EXISTS product_3d_png (
      id INTEGER PRIMARY KEY,
      product_id INTEGER NOT NULL,
      relative_path TEXT NOT NULL,
      file_name TEXT NOT NULL,
      image_blob BLOB NOT NULL,
      updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
      UNIQUE(product_id, relative_path),
      FOREIGN KEY(product_id) REFERENCES product(id) ON DELETE CASCADE
    )
    """
)

conn.execute(
    """
    UPDATE product
    SET status = 'cleaned'
    WHERE (status IS NULL OR TRIM(status) = '')
      AND cleaned_image_data_url LIKE 'data:image/%'
    """
)

conn.execute(
    """
    UPDATE product
    SET status = 'pending'
    WHERE status IS NULL OR TRIM(status) = ''
    """
)

conn.execute(
    """
    UPDATE product_3d
    SET status = 'completed'
    WHERE (status IS NULL OR TRIM(status) = '')
      AND model_data_json IS NOT NULL
      AND TRIM(model_data_json) <> ''
    """
)

conn.commit()
conn.close()
PY

echo "Preparing cleaned product images from DB..."

LIMIT_VALUE="${LIMIT:-}"
mapfile -t PRODUCT_IDS < <(
  DB_PATH="$DB_PATH" INPUT_DIR="$INPUT_DIR" LIMIT_VALUE="$LIMIT_VALUE" "${PYTHON_CMD[@]}" - <<'PY'
import base64
import os
import sqlite3
from pathlib import Path


def decode_data_url(data_url: str) -> tuple[str | None, bytes | None]:
    if not data_url.startswith("data:image"):
        return None, None
    try:
        header, payload = data_url.split(",", 1)
    except ValueError:
        return None, None

    mime = "image/png"
    if ";" in header:
        mime = header.split(":", 1)[1].split(";", 1)[0]

    try:
        raw = base64.b64decode(payload)
    except Exception:
        return None, None

    ext_map = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/webp": "webp",
    }
    ext = ext_map.get(mime, "png")
    return ext, raw


conn = sqlite3.connect(os.environ["DB_PATH"])
conn.row_factory = sqlite3.Row
input_dir = Path(os.environ["INPUT_DIR"])
input_dir.mkdir(parents=True, exist_ok=True)
limit_value = os.environ.get("LIMIT_VALUE", "")

query = """
SELECT id, cleaned_image_data_url
FROM product
WHERE status = 'cleaned'
ORDER BY id
"""
params: tuple[int, ...] = ()
if limit_value:
    query += " LIMIT ?"
    params = (int(limit_value),)

for row in conn.execute(query, params):
    product_id = int(row["id"])
    data_url = row["cleaned_image_data_url"] or ""
    ext, raw = decode_data_url(data_url)
    if ext is None or raw is None:
        continue

    out_path = input_dir / f"{product_id}.{ext}"
    out_path.write_bytes(raw)
    print(product_id)
PY
)

if [[ ${#PRODUCT_IDS[@]} -eq 0 ]]; then
  echo "No products with status='cleaned' were eligible for processing."
  exit 0
fi

echo "Queued ${#PRODUCT_IDS[@]} product(s)."

for product_id in "${PRODUCT_IDS[@]}"; do
  input_image=""
  for ext in png jpg jpeg webp; do
    candidate="$INPUT_DIR/${product_id}.${ext}"
    if [[ -f "$candidate" ]]; then
      input_image="$candidate"
      break
    fi
  done

  if [[ -z "$input_image" ]]; then
    echo "[$product_id] Input image not found after export, skipping." >&2
    continue
  fi

  product_out="$MODELS_DIR/$product_id"
  product_work="$WORK_ROOT/$product_id"
  rotated_out="$ROTATED_OUTPUT_DIR/$product_id"
  mkdir -p "$product_out" "$product_work" "$rotated_out"

  DB_PATH="$DB_PATH" PRODUCT_ID="$product_id" PRODUCT_OUT="$product_out" "${PYTHON_CMD[@]}" - <<'PY'
import os
import sqlite3

conn = sqlite3.connect(os.environ["DB_PATH"])
conn.execute(
    """
    INSERT INTO product_3d (
      product_id,
      model_type,
      width_m,
      height_m,
      depth_m,
      texture_data_url,
      model_data_json,
      status,
      output_dir,
      mesh_path,
      rotated_0_path,
      rotated_90_path,
      rotated_minus90_path,
      rotated_180_path,
      error_message,
      updated_at
    )
    VALUES (?, 'triposr_cpu_v1', 1.6, 0.9, 0.9, '', '', '', 'running', ?, '', '', '', '', '', CURRENT_TIMESTAMP)
    ON CONFLICT(product_id) DO UPDATE SET
      status = 'running',
      output_dir = excluded.output_dir,
      mesh_path = '',
      rotated_0_path = '',
      rotated_90_path = '',
      rotated_minus90_path = '',
      rotated_180_path = '',
      error_message = '',
      updated_at = CURRENT_TIMESTAMP
    """,
    (int(os.environ["PRODUCT_ID"]), os.environ["PRODUCT_OUT"]),
)
conn.commit()
conn.close()
PY

  cmd=("${PYTHON_CMD[@]}" triposr_cpu_pipeline.py --input-image "$input_image" --work-out "$product_work" --out-dir "$product_out")
  if [[ "$INSTALL_DEPS" == "1" ]]; then
    cmd+=(--install-deps)
  fi
  if [[ "$NO_REMOVE_BG" == "1" ]]; then
    cmd+=(--no-remove-bg)
  fi

  echo "[$product_id] Running TriPoSR..."

  if "${cmd[@]}"; then
    rotated_0_path=""
    rotated_90_path=""
    rotated_minus90_path=""
    rotated_180_path=""

    for rotated_file in triposr_0deg.png triposr_90deg.png triposr_minus90deg.png triposr_180deg.png; do
      if [[ -f "$product_out/$rotated_file" ]]; then
        cp "$product_out/$rotated_file" "$rotated_out/$rotated_file"
      fi
    done

    if [[ -f "$rotated_out/triposr_0deg.png" ]]; then
      rotated_0_path="data/output/rotated_products/$product_id/triposr_0deg.png"
    fi
    if [[ -f "$rotated_out/triposr_90deg.png" ]]; then
      rotated_90_path="data/output/rotated_products/$product_id/triposr_90deg.png"
    fi
    if [[ -f "$rotated_out/triposr_minus90deg.png" ]]; then
      rotated_minus90_path="data/output/rotated_products/$product_id/triposr_minus90deg.png"
    fi
    if [[ -f "$rotated_out/triposr_180deg.png" ]]; then
      rotated_180_path="data/output/rotated_products/$product_id/triposr_180deg.png"
    fi

    DB_PATH="$DB_PATH" PRODUCT_ID="$product_id" PRODUCT_WORK="$product_work" "${PYTHON_CMD[@]}" - <<'PY'
  import os
  import sqlite3
  from pathlib import Path

  conn = sqlite3.connect(os.environ["DB_PATH"])
  product_id = int(os.environ["PRODUCT_ID"])
  work_dir = Path(os.environ["PRODUCT_WORK"])

  exists = conn.execute("SELECT 1 FROM product WHERE id = ?", (product_id,)).fetchone()
  if exists:
    png_paths = sorted(path for path in work_dir.rglob("*.png") if path.is_file())
    conn.execute("DELETE FROM product_3d_png WHERE product_id = ?", (product_id,))
    for path in png_paths:
      rel = path.relative_to(work_dir).as_posix()
      conn.execute(
        """
        INSERT INTO product_3d_png (product_id, relative_path, file_name, image_blob, updated_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(product_id, relative_path) DO UPDATE SET
          file_name = excluded.file_name,
          image_blob = excluded.image_blob,
          updated_at = CURRENT_TIMESTAMP
        """,
        (product_id, rel, path.name, path.read_bytes()),
      )

  conn.commit()
  conn.close()
  PY

    mesh_public_path=""
    if compgen -G "$product_work/0/mesh.*" > /dev/null; then
      cp "$product_work"/0/mesh.* "$product_out"/
      if [[ -f "$product_out/mesh.obj" ]]; then
        cp "$product_out/mesh.obj" "$STATIC_MESH_DIR/${product_id}.obj"
        mesh_public_path="/app/static/meshes/${product_id}.obj"
      fi
    fi

    DB_PATH="$DB_PATH" PRODUCT_ID="$product_id" PRODUCT_OUT="$product_out" MESH_PUBLIC_PATH="$mesh_public_path" ROTATED_0_PATH="$rotated_0_path" ROTATED_90_PATH="$rotated_90_path" ROTATED_MINUS90_PATH="$rotated_minus90_path" ROTATED_180_PATH="$rotated_180_path" "${PYTHON_CMD[@]}" - <<'PY'
import os
import sqlite3

conn = sqlite3.connect(os.environ["DB_PATH"])
product_id = int(os.environ["PRODUCT_ID"])
product_out = os.environ["PRODUCT_OUT"]

conn.execute("UPDATE product SET status = 'modeled' WHERE id = ?", (product_id,))
conn.execute(
    """
    UPDATE product_3d
    SET
      status = 'completed',
      output_dir = ?,
      mesh_path = ?,
      rotated_0_path = ?,
      rotated_90_path = ?,
      rotated_minus90_path = ?,
      rotated_180_path = ?,
      error_message = '',
      model_type = 'triposr_cpu_v1',
      updated_at = CURRENT_TIMESTAMP
    WHERE product_id = ?
    """,
    (
        product_out,
        os.environ.get("MESH_PUBLIC_PATH", ""),
        os.environ.get("ROTATED_0_PATH", ""),
        os.environ.get("ROTATED_90_PATH", ""),
        os.environ.get("ROTATED_MINUS90_PATH", ""),
        os.environ.get("ROTATED_180_PATH", ""),
        product_id,
    ),
)
conn.commit()
conn.close()
PY
    echo "[$product_id] Completed."
  else
    DB_PATH="$DB_PATH" PRODUCT_ID="$product_id" PRODUCT_OUT="$product_out" "${PYTHON_CMD[@]}" - <<'PY'
import os
import sqlite3

conn = sqlite3.connect(os.environ["DB_PATH"])
conn.execute(
    """
    UPDATE product_3d
    SET
      status = 'failed',
      output_dir = ?,
      error_message = 'TriPoSR pipeline failed',
      updated_at = CURRENT_TIMESTAMP
    WHERE product_id = ?
    """,
    (os.environ["PRODUCT_OUT"], int(os.environ["PRODUCT_ID"])),
)
conn.commit()
conn.close()
PY
    echo "[$product_id] Failed."
  fi
done

echo "Batch processing finished."
