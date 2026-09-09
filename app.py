
from __future__ import annotations

import base64
import json
import re
import sqlite3
from collections import deque
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps

from database import (
    DATABASE_PATH,
  clear_catalog_data,
  get_products_with_3d_models,
    get_products,
    has_products,
    initialize_database,
  upsert_product_3d_model,
    upsert_products,
)

# "New Developments" multi-object detection feature: fully separate storage/
# pipeline (own SQLite DB, own output folder) so it never mixes with the
# scraped catalog used by the tabs above. Imports are guarded so a missing
# optional dependency (opencv/ultralytics) cannot break the rest of the app.
try:
    import multi_object_pipeline
    import multi_object_store
    MULTI_OBJECT_IMPORT_ERROR: str | None = None
except Exception as _multi_object_exc:  # noqa: BLE001
    multi_object_pipeline = None  # type: ignore[assignment]
    multi_object_store = None  # type: ignore[assignment]
    MULTI_OBJECT_IMPORT_ERROR = str(_multi_object_exc)

HEADERS = {"User-Agent": "Mozilla/5.0"}

RESAMPLING = getattr(Image, "Resampling", Image)
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "notebooks" / "data"
ROOM_IMAGE_PATH = DATA_DIR / "room.jpg"
THREE_D_DIR = DATA_DIR / "3d_models"
THREE_D_SPECS_PATH = THREE_D_DIR / "product_specs.json"
ROTATED_PRODUCTS_DIR = PROJECT_DIR / "data" / "output" / "rotated_products"
EXPORT_NAME = "designed_room.png"
DEV_SITE = "Local 3D Lab"
DEV_CATEGORY = "Prototype"
DEV_PRODUCT_LIMIT = 12
CURATED_3D_SITE = "Curated Input 3D"


def asset_label(path: Path) -> str:
    label = path.stem.replace("_", " ").replace("image", "")
    return " ".join(word.capitalize() for word in label.split())


def sample_background(image: Image.Image) -> tuple[int, int, int]:
    rgb = image.convert("RGB")
    corners = [
        rgb.getpixel((0, 0)),
        rgb.getpixel((rgb.width - 1, 0)),
        rgb.getpixel((0, rgb.height - 1)),
        rgb.getpixel((rgb.width - 1, rgb.height - 1)),
    ]
    return tuple(sum(pixel[i] for pixel in corners) // len(corners) for i in range(3))


def remove_simple_background(image: Image.Image, tolerance: int = 45) -> Image.Image:
    rgba = ImageOps.exif_transpose(image).convert("RGBA")
    rgb = np.array(rgba.convert("RGB"), dtype=np.int16)
    height, width, _ = rgb.shape

    border_pixels = np.vstack(
        [
            rgb[0, :, :],
            rgb[height - 1, :, :],
            rgb[:, 0, :],
            rgb[:, width - 1, :],
        ]
    )
    background = np.median(border_pixels, axis=0)
    diff = np.max(np.abs(rgb - background), axis=2)
    background_like = diff <= tolerance

    connected_bg = np.zeros((height, width), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for x in range(width):
        if background_like[0, x]:
            queue.append((0, x))
        if background_like[height - 1, x]:
            queue.append((height - 1, x))
    for y in range(height):
        if background_like[y, 0]:
            queue.append((y, 0))
        if background_like[y, width - 1]:
            queue.append((y, width - 1))

    while queue:
        y, x = queue.popleft()
        if y < 0 or y >= height or x < 0 or x >= width:
            continue
        if connected_bg[y, x] or not background_like[y, x]:
            continue
        connected_bg[y, x] = True
        queue.append((y - 1, x))
        queue.append((y + 1, x))
        queue.append((y, x - 1))
        queue.append((y, x + 1))

    alpha = np.array(rgba.split()[-1], dtype=np.uint8)
    alpha[connected_bg] = 0
    result = np.dstack([rgb.astype(np.uint8), alpha])
    rgba = Image.fromarray(result, mode="RGBA")
    bbox = rgba.getbbox()
    return rgba.crop(bbox) if bbox else rgba


@st.cache_data
def download_image_bytes(url: str) -> bytes | None:
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        return response.content
    except requests.RequestException:
        return None


def load_assets() -> tuple[Image.Image, dict[str, Image.Image | str], dict[str, dict[str, Any]]]:
    if not ROOM_IMAGE_PATH.exists():
        raise FileNotFoundError(f"Room image not found: {ROOM_IMAGE_PATH}")

    def normalize_price(price: str | None) -> tuple[str, float | None]:
        if not price:
            return "Unknown", None
        text = price.replace("€", "").replace("$", "").replace("EUR", "").replace(" ", "").replace(",", ".")
        import re

        match = re.search(r"(\d+(?:\.\d+)?)", text)
        if not match:
            return "Unknown", None
        try:
            value = float(match.group(1))
        except ValueError:
            return "Unknown", None
        if value < 100:
            return "< 100", value
        if value < 200:
            return "100-199", value
        if value < 400:
            return "200-399", value
        if value < 600:
            return "400-599", value
        return "600+", value

    room = Image.open(ROOM_IMAGE_PATH).convert("RGBA")
    assets: dict[str, Image.Image | str] = {}
    asset_meta: dict[str, dict[str, Any]] = {}

    def trim_product_frame(image: Image.Image) -> Image.Image:
      rgba = image.convert("RGBA")
      arr = np.array(rgba)
      alpha = arr[:, :, 3]

      if np.any(alpha > 8):
        ys, xs = np.where(alpha > 8)
      else:
        rgb = arr[:, :, :3].astype(np.int16)
        corners = np.array(
          [rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]],
          dtype=np.int16,
        )
        bg = np.median(corners, axis=0)
        diff = np.max(np.abs(rgb - bg), axis=2)
        ys, xs = np.where(diff > 10)

      if ys.size == 0 or xs.size == 0:
        return rgba

      margin = 2
      left = max(int(xs.min()) - margin, 0)
      top = max(int(ys.min()) - margin, 0)
      right = min(int(xs.max()) + 1 + margin, rgba.width)
      bottom = min(int(ys.max()) + 1 + margin, rgba.height)
      if left >= right or top >= bottom:
        return rgba
      return rgba.crop((left, top, right, bottom))

    def bytes_to_trimmed_data_url(raw: bytes) -> str:
      try:
        image = Image.open(BytesIO(raw)).convert("RGBA")
        trimmed = trim_product_frame(image)
        buffer = BytesIO()
        trimmed.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
      except Exception:
        encoded = base64.b64encode(raw).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def read_image_data_url(path_value: str) -> str:
      if not path_value:
        return ""
      if path_value.startswith("data:image"):
        image = data_url_to_image(path_value)
        if image is None:
          return path_value
        return image_to_data_url(trim_product_frame(image))
      path_obj = Path(path_value)
      if not path_obj.is_absolute():
        path_obj = PROJECT_DIR / path_obj
      if not path_obj.exists():
        return ""
      try:
        return bytes_to_trimmed_data_url(path_obj.read_bytes())
      except Exception:
        return ""

    def pick_render_for_angle(render_by_index: dict[int, bytes], angle_deg: int) -> str:
      if not render_by_index:
        return ""
      target = angle_deg % 360
      best_index = None
      best_distance = None
      for index in render_by_index:
        degrees = (index * 12) % 360
        distance = min(abs(degrees - target), 360 - abs(degrees - target))
        if best_distance is None or distance < best_distance:
          best_index = index
          best_distance = distance
      if best_index is None:
        return ""
      return bytes_to_trimmed_data_url(render_by_index[best_index])

    def load_rotated_views_from_png_blobs() -> dict[str, dict[str, str]]:
      views_by_product: dict[str, dict[str, str]] = {}
      try:
        conn = sqlite3.connect(str(DATABASE_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
          """
          SELECT product_id, relative_path, file_name, image_blob
          FROM product_3d_png
          ORDER BY product_id, relative_path
          """
        ).fetchall()
      except sqlite3.Error:
        return views_by_product
      finally:
        try:
          conn.close()  # type: ignore[name-defined]
        except Exception:
          pass

      grouped: dict[str, list[sqlite3.Row]] = {}
      for row in rows:
        product_id = str(row["product_id"])
        grouped.setdefault(product_id, []).append(row)

      for product_id, product_rows in grouped.items():
        explicit: dict[str, str] = {}
        renders: dict[int, bytes] = {}

        for row in product_rows:
          file_name = str(row["file_name"] or "")
          rel = str(row["relative_path"] or "")
          blob = row["image_blob"]
          if not blob:
            continue

          data_url = bytes_to_trimmed_data_url(blob)

          if file_name == "triposr_0deg.png":
            explicit["0"] = data_url
          elif file_name == "triposr_90deg.png":
            explicit["90"] = data_url
          elif file_name == "triposr_minus90deg.png":
            explicit["-90"] = data_url
          elif file_name == "triposr_180deg.png":
            explicit["180"] = data_url

          match = re.search(r"render_(\d{3})\.png$", rel)
          if match:
            renders[int(match.group(1))] = blob

        views = {
          "0": explicit.get("0", pick_render_for_angle(renders, 0)),
          "90": explicit.get("90", pick_render_for_angle(renders, 90)),
          "-90": explicit.get("-90", pick_render_for_angle(renders, 270)),
          "180": explicit.get("180", pick_render_for_angle(renders, 180)),
        }
        views = {key: value for key, value in views.items() if value}
        if views:
          views_by_product[product_id] = views

      return views_by_product

    model_rows = get_products_with_3d_models(limit=None)
    models_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in model_rows:
      key = (str(row.get("site") or ""), str(row.get("name") or ""))
      if key[0] and key[1]:
        models_by_key[key] = row

    rotated_views_from_blob_db = load_rotated_views_from_png_blobs()

    def read_png_as_data_url(path: Path) -> str:
      return bytes_to_trimmed_data_url(path.read_bytes())

    scraped_products = get_products()
    if not scraped_products:
        raise FileNotFoundError(
            "No products found in database. Seed the catalog first (e.g. python3 seed_catalog.py)."
        )

    for product in scraped_products:
        product_id = str(product.get("id") or "")
        cleaned_image_data_url = product.get("cleaned_image_data_url") or ""
        image_url = product.get("image_url") or ""
        label = f"{product['site']} / {product['category']} / {product['name']}"
        if label in assets:
            label = f"{label} (scraped)"

        rotated_views: dict[str, str] = {}

        if product_id and product_id in rotated_views_from_blob_db:
          rotated_views.update(rotated_views_from_blob_db[product_id])

        model_row = models_by_key.get((product.get("site") or "", product.get("name") or ""), {})
        rotated_from_db = {
          "0": str(model_row.get("rotated_0_path") or ""),
          "90": str(model_row.get("rotated_90_path") or ""),
          "-90": str(model_row.get("rotated_minus90_path") or ""),
          "180": str(model_row.get("rotated_180_path") or ""),
        }
        for angle, path_value in rotated_from_db.items():
          data_url = read_image_data_url(path_value)
          if data_url:
            rotated_views[angle] = data_url

        if product_id:
            rotated_dir = ROTATED_PRODUCTS_DIR / product_id
            rotated_candidates = {
                "0": rotated_dir / "triposr_0deg.png",
                "90": rotated_dir / "triposr_90deg.png",
                "-90": rotated_dir / "triposr_minus90deg.png",
                "180": rotated_dir / "triposr_180deg.png",
            }
            for angle, image_path in rotated_candidates.items():
              if angle in rotated_views:
                continue
              if image_path.exists():
                try:
                  rotated_views[angle] = read_png_as_data_url(image_path)
                except Exception:
                  continue

        default_rotated = rotated_views.get("0", "")

        if default_rotated:
            assets[label] = default_rotated
        elif cleaned_image_data_url:
            assets[label] = cleaned_image_data_url
        elif image_url:
            # Keep remote image URL directly to avoid long blocking startup on many downloads.
            assets[label] = image_url

        price = product.get("price") or ""
        interval, _ = normalize_price(price)
        asset_meta[label] = {
                "site": product.get("site", "Unknown"),
                "category": product.get("category", "Unknown") or "Unknown",
                "price": price,
                "price_interval": interval,
                "color": product.get("color") or "",
                "product_id": product_id,
                "rotated_views": rotated_views,
            }

    return room, assets, asset_meta


def image_to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def data_url_to_image(data_url: str) -> Image.Image | None:
  """Decode data URL payload into a Pillow image."""
  if not data_url.startswith("data:image"):
    return None
  try:
    _, encoded = data_url.split(",", 1)
    raw = base64.b64decode(encoded)
    return Image.open(BytesIO(raw)).convert("RGBA")
  except Exception:
    return None


def load_product_image_for_3d(product: dict[str, str]) -> Image.Image | None:
  """Load the best available product image for 3D reconstruction."""
  cleaned_data_url = product.get("cleaned_image_data_url") or ""
  if cleaned_data_url:
    image = data_url_to_image(cleaned_data_url)
    if image is not None:
      return image

  image_url = product.get("image_url") or ""
  if not image_url:
    return None

  payload = download_image_bytes(image_url)
  if not payload:
    return None

  try:
    return Image.open(BytesIO(payload)).convert("RGBA")
  except Exception:
    return None


def ensure_scraped_products_have_3d_models(
  *,
  max_to_process: int = 4,
  force_refresh: bool = False,
) -> dict[str, int]:
  """Convert scraped 2D furniture images into persisted closed depth-mesh models."""
  scraped_products = get_products()
  existing_models = get_products_with_3d_models(limit=None)
  existing_by_key = {
    (str(entry.get("site") or ""), str(entry.get("name") or "")): str(entry.get("model_type") or "")
    for entry in existing_models
  }

  created = 0
  skipped = 0
  failed = 0

  for product in scraped_products:
    site = product.get("site") or ""
    name = product.get("name") or ""
    category = product.get("category") or ""
    if not site or not name:
      skipped += 1
      continue

    key = (site, name)
    existing_type = existing_by_key.get(key, "")
    should_upgrade_box = existing_type in {
      "",
      "box_v1",
      "fallback_box_v1",
      "depth_mesh_midas_v1",
      "depth_mesh_midas_v2_closed",
    }
    if not force_refresh and existing_type and not should_upgrade_box:
      skipped += 1
      continue

    image = load_product_image_for_3d(product)
    if image is None:
      failed += 1
      continue

    try:
      original = ImageOps.exif_transpose(image).convert("RGBA")
      cleaned = remove_simple_background(original)
      texture_preview = original.copy()
      texture_preview.thumbnail((512, 512), RESAMPLING.LANCZOS)
      texture_data_url = image_to_data_url(texture_preview)

      model_data = build_slice_volume_model_data(cleaned)
      dims = infer_dimensions_meters(name=name, category=category)

      upsert_product_3d_model(
        site_name=site,
        product_name=name,
        model_type="depth_mesh_cpu_v3",
        width_m=float(dims["width"]),
        height_m=float(dims["height"]),
        depth_m=float(dims["depth"]),
        texture_data_url=texture_data_url,
        model_data_json=json.dumps(model_data, ensure_ascii=True),
      )
      created += 1
      if created >= max_to_process:
        break
    except Exception:
      failed += 1

  return {"created": created, "skipped": skipped, "failed": failed}


_DEPTH_PIPELINE: Any | None = None


def get_depth_pipeline() -> Any:
  """Lazily load an advanced depth-estimation pipeline."""
  global _DEPTH_PIPELINE
  if _DEPTH_PIPELINE is None:
    from transformers import pipeline

    _DEPTH_PIPELINE = pipeline(task="depth-estimation", model="Intel/dpt-hybrid-midas")
  return _DEPTH_PIPELINE


def _smooth_mesh_positions_cpu(
  *,
  positions: list[float],
  front_indices: list[int],
  back_indices: list[int],
  side_indices: list[int],
) -> list[float]:
  """Apply lightweight CPU mesh smoothing using Open3D when available."""
  try:
    import open3d as o3d  # type: ignore
  except Exception:
    return positions

  verts = np.array(positions, dtype=np.float32).reshape((-1, 3))
  tris_all = np.array(front_indices + back_indices + side_indices, dtype=np.int32)
  if verts.shape[0] < 3 or tris_all.size < 3:
    return positions

  tris = tris_all.reshape((-1, 3))
  mesh = o3d.geometry.TriangleMesh(
    o3d.utility.Vector3dVector(verts.astype(np.float64)),
    o3d.utility.Vector3iVector(tris.astype(np.int32)),
  )

  smoothed = mesh.filter_smooth_taubin(number_of_iterations=1)
  smoothed_vertices = np.asarray(smoothed.vertices, dtype=np.float32)
  if smoothed_vertices.shape[0] != verts.shape[0]:
    return positions
  return smoothed_vertices.reshape(-1).tolist()


def build_slice_volume_model_data(image: Image.Image) -> dict[str, Any]:
    """Build closed, textured depth mesh from one image (front/back + side walls)."""
    rgba = image.convert("RGBA")
    preview = rgba.copy()
    preview.thumbnail((256, 256), RESAMPLING.LANCZOS)
    width, height = preview.size
    pixels = preview.load()

    threshold = 40
    mask: list[list[bool]] = []
    for y in range(height):
        row: list[bool] = []
        for x in range(width):
            row.append(pixels[x, y][3] >= threshold)
        mask.append(row)

    total_red = 0
    total_green = 0
    total_blue = 0
    count = 0
    for y in range(height):
        for x in range(width):
            if not mask[y][x]:
                continue
            red, green, blue, _ = pixels[x, y]
            total_red += int(red)
            total_green += int(green)
            total_blue += int(blue)
            count += 1

    if count == 0:
        avg_color = [200, 170, 120]
    else:
        avg_color = [total_red // count, total_green // count, total_blue // count]

    rgb_for_depth = Image.new("RGB", preview.size, (245, 245, 245))
    rgb_for_depth.paste(preview.convert("RGB"), mask=preview.split()[-1])

    estimator = get_depth_pipeline()
    depth_result = estimator(rgb_for_depth)
    depth_img = depth_result.get("depth") if isinstance(depth_result, dict) else None
    if depth_img is None:
      depth_map = np.zeros((height, width), dtype=np.float32)
    else:
      depth_resized = depth_img.convert("L").resize((width, height), RESAMPLING.BILINEAR)
      depth_map = np.array(depth_resized, dtype=np.float32) / 255.0

    # Stabilize single-image depth to avoid noisy, wire-like triangulation artifacts.
    for _ in range(4):
      depth_map = (
        depth_map
        + np.roll(depth_map, 1, axis=0)
        + np.roll(depth_map, -1, axis=0)
        + np.roll(depth_map, 1, axis=1)
        + np.roll(depth_map, -1, axis=1)
      ) / 5.0

    alpha_map = np.array(preview.split()[-1], dtype=np.float32) / 255.0
    object_mask = alpha_map >= 0.12

    if np.any(object_mask):
      valid_depth = depth_map[object_mask]
      low = float(np.percentile(valid_depth, 5))
      high = float(np.percentile(valid_depth, 95))
      high = max(high, low + 1e-6)
      depth_map = np.clip((depth_map - low) / (high - low), 0.0, 1.0)

      # Build a low-frequency depth field to prevent slice-like tearing when rotating.
      depth_img_small = Image.fromarray((depth_map * 255.0).astype(np.uint8), mode="L")
      depth_img_small = depth_img_small.resize((48, 48), RESAMPLING.BILINEAR)
      depth_img_small = depth_img_small.resize((width, height), RESAMPLING.BILINEAR)
      depth_map = np.array(depth_img_small, dtype=np.float32) / 255.0
    front_idx_map = -np.ones((height, width), dtype=np.int32)
    back_idx_map = -np.ones((height, width), dtype=np.int32)
    front_to_back: dict[int, int] = {}

    positions: list[float] = []
    uvs: list[float] = []
    vertex_idx = 0
    z_scale = 0.16
    thickness = 0.24

    for y in range(height):
        for x in range(width):
            if alpha_map[y, x] < 0.12:
                continue

            nx = (x / max(width - 1, 1)) - 0.5
            ny = 0.5 - (y / max(height - 1, 1))
            z_front = 0.42 + (depth_map[y, x] * z_scale)
            z_back = z_front - thickness
            u = x / max(width - 1, 1)
            v = 1.0 - (y / max(height - 1, 1))

            f_idx = vertex_idx
            positions.extend([float(nx), float(ny), float(z_front)])
            uvs.extend([u, v])
            vertex_idx += 1

            b_idx = vertex_idx
            positions.extend([float(nx), float(ny), float(z_back)])
            uvs.extend([u, v])
            vertex_idx += 1

            front_idx_map[y, x] = f_idx
            back_idx_map[y, x] = b_idx
            front_to_back[f_idx] = b_idx

    front_indices: list[int] = []
    back_indices: list[int] = []
    edge_counter: dict[tuple[int, int], int] = {}
    edge_oriented: dict[tuple[int, int], tuple[int, int]] = {}

    for y in range(height - 1):
        for x in range(width - 1):
            f00 = int(front_idx_map[y, x])
            f10 = int(front_idx_map[y, x + 1])
            f01 = int(front_idx_map[y + 1, x])
            f11 = int(front_idx_map[y + 1, x + 1])
            b00 = int(back_idx_map[y, x])
            b10 = int(back_idx_map[y, x + 1])
            b01 = int(back_idx_map[y + 1, x])
            b11 = int(back_idx_map[y + 1, x + 1])

            if f00 >= 0 and f10 >= 0 and f11 >= 0:
                tri = (f00, f10, f11)
                front_indices.extend(tri)
            if f00 >= 0 and f11 >= 0 and f01 >= 0:
                tri = (f00, f11, f01)
                front_indices.extend(tri)

            if b00 >= 0 and b10 >= 0 and b11 >= 0:
                back_indices.extend([b00, b11, b10])
            if b00 >= 0 and b11 >= 0 and b01 >= 0:
                back_indices.extend([b00, b01, b11])

    for i in range(0, len(front_indices), 3):
        a, b, c = front_indices[i], front_indices[i + 1], front_indices[i + 2]
        for u_idx, v_idx in ((a, b), (b, c), (c, a)):
            key = (u_idx, v_idx) if u_idx < v_idx else (v_idx, u_idx)
            edge_counter[key] = edge_counter.get(key, 0) + 1
            if key not in edge_oriented:
                edge_oriented[key] = (u_idx, v_idx)

    side_indices: list[int] = []
    for key, count_val in edge_counter.items():
        if count_val != 1:
            continue
        u_idx, v_idx = edge_oriented[key]
        bu = front_to_back.get(u_idx)
        bv = front_to_back.get(v_idx)
        if bu is None or bv is None:
            continue
        side_indices.extend([u_idx, v_idx, bv])
        side_indices.extend([u_idx, bv, bu])

    positions = _smooth_mesh_positions_cpu(
      positions=positions,
      front_indices=front_indices,
      back_indices=back_indices,
      side_indices=side_indices,
    )

    return {
        "alpha_cutoff": 0.04,
        "avg_color": avg_color,
        "positions": positions,
        "uvs": uvs,
        "front_indices": front_indices,
        "back_indices": back_indices,
        "side_indices": side_indices,
        "depth_method": "dpt_hybrid_midas_cpu_refined_v3",
    }


def seed_local_3d_catalog(force_reset: bool = False) -> None:
  """Reset DB to a tiny local catalog and store generated 3D model records."""
  initialize_database()
  existing_3d = get_products_with_3d_models(limit=DEV_PRODUCT_LIMIT)
  if existing_3d and not force_reset:
    return

  clear_catalog_data()

  local_assets = [
    path
    for path in sorted(DATA_DIR.glob("*.jpg"))
    if path.name.lower() != "room.jpg"
  ][:DEV_PRODUCT_LIMIT]

  products: list[dict[str, str]] = []
  textures_by_name: dict[str, str] = {}
  model_data_by_name: dict[str, str] = {}
  model_type_by_name: dict[str, str] = {}
  for path in local_assets:
    name = asset_label(path)
    image = remove_simple_background(Image.open(path))
    preview = image.copy()
    preview.thumbnail((256, 256), RESAMPLING.LANCZOS)
    texture_data_url = image_to_data_url(preview)

    lower_name = name.lower()
    if "piano" in lower_name or "keyboard" in lower_name:
      model_type_by_name[name] = "procedural_piano_v1"
      model_data_by_name[name] = json.dumps(build_procedural_piano_model_data(), ensure_ascii=True)
    else:
      model_type_by_name[name] = "depth_mesh_midas_v1"
      model_data_by_name[name] = json.dumps(build_slice_volume_model_data(image), ensure_ascii=True)

    products.append(
      {
        "category": DEV_CATEGORY,
        "name": name,
        "url": "",
        "image_url": "",
        "cleaned_image_data_url": texture_data_url,
        "price": "",
        "color": "",
        "rating": "",
      }
    )
    textures_by_name[name] = texture_data_url

  upsert_products(DEV_SITE, products)

  for product in products:
    name = product["name"]
    dims = infer_dimensions_meters(name=name, category=DEV_CATEGORY)
    upsert_product_3d_model(
      site_name=DEV_SITE,
      product_name=name,
      model_type=model_type_by_name[name],
      width_m=float(dims["width"]),
      height_m=float(dims["height"]),
      depth_m=float(dims["depth"]),
      texture_data_url=textures_by_name[name],
      model_data_json=model_data_by_name[name],
    )


@st.cache_data
def build_payload(room: Image.Image, assets: dict[str, Image.Image | str], asset_meta: dict[str, dict[str, Any]]) -> str:
    room_preview = room.copy()
    room_preview.thumbnail((1100, 650), RESAMPLING.LANCZOS)
    room_data_url = image_to_data_url(room_preview)

    asset_urls: dict[str, str] = {}
    for name, image in assets.items():
        if isinstance(image, Image.Image):
            asset = image.copy()
            target_width = min(300, max(120, asset.width))
            target_height = max(80, int(asset.height * (target_width / asset.width)))
            asset = asset.resize((target_width, target_height), RESAMPLING.LANCZOS)
            asset_urls[name] = image_to_data_url(asset)
        else:
            asset_urls[name] = image

    payload = {
        "room": room_data_url,
        "assets": asset_urls,
        "assetMeta": asset_meta,
        "width": room_preview.width,
        "height": room_preview.height,
    }
    return json.dumps(payload)


def build_html(payload_json: str) -> str:
    html = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    body { margin: 0; }
    .drag-wrap {
      font-family: sans-serif;
      display: flex;
      gap: 14px;
      align-items: flex-start;
    }
    .drag-sidebar {
      width: 260px;
      min-width: 260px;
      border: 1px solid #d7d7d7;
      border-radius: 10px;
      background: #fafafa;
      padding: 12px;
      box-sizing: border-box;
    }
    .drag-sidebar h4 {
      margin: 0 0 10px 0;
      font-size: 15px;
    }
    .drag-sidebar label {
      display: block;
      margin: 8px 0 4px;
      font-size: 13px;
      color: #333;
    }
    .drag-sidebar select,
    .drag-sidebar button {
      width: 100%;
      box-sizing: border-box;
      margin-bottom: 8px;
      height: 34px;
    }
    .drag-main { flex: 1; min-width: 0; }
    .drag-toolbar {
      display: flex;
      gap: 10px;
      align-items: center;
      margin: 0 0 10px 0;
      flex-wrap: wrap;
      border: 1px solid #ddd;
      border-radius: 10px;
      padding: 8px 10px;
      background: #fff;
    }
    .drag-stage {
      position: relative;
      background-size: cover;
      background-position: center;
      border: 1px solid #bbb;
      overflow: hidden;
      touch-action: none;
      max-width: 100%;
    }
    .drag-item {
      position: absolute;
      left: 50%;
      top: 50%;
      width: 220px;
      transform: translate(-50%, -50%);
      cursor: grab;
      user-select: none;
      border: 2px solid transparent;
      transform-origin: center bottom;
    }
    .drag-item.selected { border-color: #0a84ff; }
    .drag-status { margin-top: 8px; color: #444; font-size: 13px; }
    @media (max-width: 900px) {
      .drag-wrap { flex-direction: column; }
      .drag-sidebar { width: 100%; min-width: 0; }
    }
  </style>
</head>
<body>
<div id="drag-root"></div>
<script>
(() => {
  const root = document.getElementById("drag-root");
  const DATA = __PAYLOAD_JSON__;
  root.innerHTML = `
    <div class="drag-wrap">
      <aside class="drag-sidebar">
        <h4>Catalog Controls</h4>
        <label>Source</label>
        <select id="sourceSelect"></select>
        <label>Category</label>
        <select id="categorySelect"></select>
        <label>Price</label>
        <select id="priceSelect"></select>
        <label>Color</label>
        <select id="colorSelect"></select>
        <label>Asset</label>
        <select id="assetSelect"></select>
        <button id="addBtn">Add</button>
        <button id="delBtn">Delete Selected</button>
        <button id="resetBtn">🔄 Reset Design</button>
        <button id="rot90Btn">Rotate 90°</button>
        <button id="rotMinus90Btn">Rotate -90°</button>
        <button id="rot180Btn">Rotate 180°</button>
        <button id="rotLeftBtn">↺ Rotate -15°</button>
        <button id="rotRightBtn">↻ Rotate +15°</button>
        <button id="flipBtn">⇆ Flip</button>
      </aside>
      <div class="drag-main">
        <div class="drag-toolbar">
        <label>Size</label>
        <input id="sizeRange" type="range" min="60" max="500" value="220" step="5" />
        <label>Rot X</label>
        <input id="rotXRange" type="range" min="-180" max="180" value="0" step="5" style="width:90px" />
        <label>Yaw (Y)</label>
        <input id="rotYRange" type="range" min="-180" max="180" value="0" step="1" style="width:120px" />
        <label>Rot Z</label>
        <input id="rotZRange" type="range" min="-180" max="180" value="0" step="1" style="width:120px" />
        </div>
        <div id="stage" class="drag-stage"></div>
        <div id="status" class="drag-status">Tip: Left-drag move. Right-drag rotate. Wheel rotate. Shift+wheel resize.</div>
      </div>
    </div>
  `;

  const sourceSelect = root.querySelector('#sourceSelect');
  const categorySelect = root.querySelector('#categorySelect');
  const priceSelect = root.querySelector('#priceSelect');
  const colorSelect = root.querySelector('#colorSelect');
  const assetSelect = root.querySelector('#assetSelect');
  const addBtn = root.querySelector('#addBtn');
  const delBtn = root.querySelector('#delBtn');
  const resetBtn = root.querySelector('#resetBtn');
  const rot90Btn = root.querySelector('#rot90Btn');
  const rotMinus90Btn = root.querySelector('#rotMinus90Btn');
  const rot180Btn = root.querySelector('#rot180Btn');
  const sizeRange = root.querySelector('#sizeRange');
  const rotXRange = root.querySelector('#rotXRange');
  const rotYRange = root.querySelector('#rotYRange');
  const rotZRange = root.querySelector('#rotZRange');
  const rotLeftBtn = root.querySelector('#rotLeftBtn');
  const rotRightBtn = root.querySelector('#rotRightBtn');
  const flipBtn = root.querySelector('#flipBtn');
  const stage = root.querySelector('#stage');
  const status = root.querySelector('#status');

  stage.style.width = '100%';
  stage.style.maxWidth = `${DATA.width}px`;
  let scale = 1;
  let lastStageWidth = 0;
  function refreshStageLayout() {
    const measuredWidth = stage.clientWidth;
    if (measuredWidth <= 0) {
      return;
    }
    if (measuredWidth === lastStageWidth) {
      return;
    }
    lastStageWidth = measuredWidth;
    scale = measuredWidth / DATA.width;
    stage.style.height = `${DATA.height * scale}px`;
    stage.querySelectorAll('.drag-item').forEach((item) => applyItemState(item));
  }
  // Initial paint can happen while tab is hidden (width=0), so keep trying.
  refreshStageLayout();
  setInterval(refreshStageLayout, 400);
  stage.style.backgroundImage = `url('${DATA.room}')`;

  const allNames = Object.keys(DATA.assets);
  const assetMeta = DATA.assetMeta || {};

  function addOption(select, value) {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  }

  function sortWithAllFirst(values) {
    return values
      .filter((value) => value !== 'All')
      .sort((a, b) => a.localeCompare(b, undefined, { sensitivity: 'base' }));
  }

  function populateFilterOptions() {
    const sources = new Set(['All']);
    const categories = new Set(['All']);
    const prices = new Set(['All']);
    const colors = new Set(['All']);

    allNames.forEach((name) => {
      const meta = assetMeta[name] || {};
      sources.add(meta.site || 'Unknown');
      categories.add(meta.category || 'Unknown');
      prices.add(meta.price_interval || 'Unknown');
      if (meta.color) colors.add(meta.color);
    });

    addOption(sourceSelect, 'All');
    sortWithAllFirst(Array.from(sources)).forEach((value) => addOption(sourceSelect, value));

    addOption(categorySelect, 'All');
    sortWithAllFirst(Array.from(categories)).forEach((value) => addOption(categorySelect, value));

    addOption(priceSelect, 'All');
    sortWithAllFirst(Array.from(prices)).forEach((value) => addOption(priceSelect, value));

    addOption(colorSelect, 'All');
    sortWithAllFirst(Array.from(colors)).forEach((value) => addOption(colorSelect, value));

    sourceSelect.value = 'All';
    categorySelect.value = 'All';
    priceSelect.value = 'All';
    colorSelect.value = 'All';
  }

  function filterAssetOptions() {
    const sourceValue = sourceSelect.value;
    const categoryValue = categorySelect.value;
    const priceValue = priceSelect.value;
    const colorValue = colorSelect.value;
    assetSelect.innerHTML = '';

    const filtered = allNames.filter((name) => {
      const meta = assetMeta[name] || {};
      const source = meta.site || 'Unknown';
      const category = meta.category || 'Unknown';
      const price = meta.price_interval || 'Unknown';
      const color = meta.color || '';
      if (sourceValue !== 'All' && source !== sourceValue) return false;
      if (categoryValue !== 'All' && category !== categoryValue) return false;
      if (priceValue !== 'All' && price !== priceValue) return false;
      if (colorValue !== 'All' && color !== colorValue) return false;
      return true;
    });

    filtered.forEach((name) => addOption(assetSelect, name));
    if (filtered.length && !filtered.includes(assetSelect.value)) {
      assetSelect.value = filtered[0];
    }
  }

  populateFilterOptions();
  filterAssetOptions();

  sourceSelect.addEventListener('change', filterAssetOptions);
  categorySelect.addEventListener('change', filterAssetOptions);
  priceSelect.addEventListener('change', filterAssetOptions);
  colorSelect.addEventListener('change', filterAssetOptions);

  let selected = null;

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  function normalizeYaw(yawDeg) {
    let value = Number(yawDeg || 0);
    while (value > 180) value -= 360;
    while (value <= -180) value += 360;
    return value;
  }

  function yawToViewKey(yawDeg) {
    const yaw = normalizeYaw(yawDeg);
    if (yaw > 135 || yaw <= -135) return '180';
    if (yaw > 45) return '90';
    if (yaw <= -45) return '-90';
    return '0';
  }

  function getAssetViewSrc(name, yawDeg, flipped, viewOverride) {
    const meta = assetMeta[name] || {};
    const views = meta.rotated_views || {};
    // Explicit button-driven view should always win.
    const explicitKey = viewOverride ? String(viewOverride) : '';
    if (explicitKey) {
      if (views[explicitKey]) {
        return { src: views[explicitKey], key: explicitKey };
      }
      // No fallback when user explicitly requests a fixed rotation view.
      return { src: '', key: explicitKey };
    }

    if (!views || Object.keys(views).length === 0) {
      return { src: DATA.assets[name], key: 'base' };
    }

    let key = yawToViewKey(yawDeg);
    if (flipped) {
      if (key === '90') key = '-90';
      else if (key === '-90') key = '90';
    }

    return { src: views[key] || views['0'] || DATA.assets[name], key };
  }

  function applyItemState(item) {
    const x = Number(item.dataset.x);
    const y = Number(item.dataset.y);
    const w = Number(item.dataset.w);
    const rot = Number(item.dataset.rot);
    const flipped = item.dataset.flipped === 'true';
    item.style.left = `${x * scale}px`;
    item.style.top = `${y * scale}px`;
    item.style.width = `${w * scale}px`;
    const rotX = Number(item.dataset.rotx || 0);
    const rotY = Number(item.dataset.roty || 0);
    const estimatedHeight = w * 1.2;
    const yShift = (Math.cos(rotX * Math.PI / 180) - 1) * estimatedHeight / 4;
    item.style.transform = `perspective(1400px) translate(-50%, calc(-50% + ${yShift * scale}px)) rotateX(${rotX * 0.6}deg) rotateY(${rotY}deg) rotateZ(${rot}deg) scaleX(${flipped ? -1 : 1})`;

    const view = getAssetViewSrc(item.dataset.name, rotY, flipped, item.dataset.viewOverride || '');
    if (!view.src) {
      item.remove();
      if (selected === item) {
        setSelected(null);
      }
      return;
    }
    if (item.src !== view.src) {
      item.src = view.src;
    }
    item.dataset.viewKey = view.key;
  }

  function setSelected(item) {
    if (selected) selected.classList.remove('selected');
    selected = item;
    if (selected) {
      selected.classList.add('selected');
      sizeRange.value = String(Math.round(Number(selected.dataset.w)));
      rotXRange.value = String(Math.round(Number(selected.dataset.rotx || 0)));
      rotYRange.value = String(Math.round(Number(selected.dataset.roty || 0)));
      rotZRange.value = String(Math.round(Number(selected.dataset.rot || 0)));
      status.textContent = `Selected: ${selected.dataset.name} | rotX=${Math.round(Number(selected.dataset.rotx||0))}, rotY=${Math.round(Number(selected.dataset.roty||0))}, rotZ=${Math.round(Number(selected.dataset.rot))}`;
    } else {
      status.textContent = 'Tip: Left-drag move. Right-drag rotate. Wheel rotate. Shift+wheel resize.';
    }
  }

  function createItemElement(name) {
    const img = document.createElement('img');
    img.className = 'drag-item';
    img.src = DATA.assets[name];
    img.draggable = false;
    img.dataset.name = name;
    img.dataset.x = String(Math.round(DATA.width * 0.5));
    img.dataset.y = String(Math.round(DATA.height * 0.7));
    img.dataset.w = sizeRange.value;
    img.dataset.rot = '0';
    img.dataset.rotx = '0';
    img.dataset.roty = '0';
    img.dataset.flipped = 'false';
    img.dataset.viewKey = '0';
    img.dataset.viewOverride = '0';
    return img;
  }

  function wireItemInteractions(img) {
    let dragging = false;
    let dragMode = 'move';
    let dx = 0;
    let dy = 0;
    let startX = 0;
    let startRot = 0;

    img.addEventListener('pointerdown', (event) => {
      event.preventDefault();
      setSelected(img);
      dragging = true;
      img.setPointerCapture(event.pointerId);
      if (event.button === 2) {
        dragMode = 'rotate';
        startX = event.clientX;
        startRot = Number(img.dataset.rot);
      } else {
        dragMode = 'move';
        const rect = stage.getBoundingClientRect();
        dx = event.clientX - rect.left - Number(img.dataset.x) * scale;
        dy = event.clientY - rect.top - Number(img.dataset.y) * scale;
      }
      img.style.cursor = 'grabbing';
    });

    img.addEventListener('pointermove', (event) => {
      if (!dragging) return;
      if (dragMode === 'rotate') {
        const delta = event.clientX - startX;
        const next = startRot + (delta * 0.35);
        img.dataset.rot = String(next);
        applyItemState(img);
        rotZRange.value = String(Math.round(next));
        status.textContent = `Rotating: ${img.dataset.name} | rot=${Math.round(next)}`;
      } else {
        const rect = stage.getBoundingClientRect();
        const nx = clamp((event.clientX - rect.left - dx) / scale, 0, DATA.width);
        const ny = clamp((event.clientY - rect.top - dy) / scale, 0, DATA.height);
        img.dataset.x = String(nx);
        img.dataset.y = String(ny);
        applyItemState(img);
        status.textContent = `Moving: ${img.dataset.name} | x=${Math.round(nx)}, y=${Math.round(ny)}, rot=${Math.round(Number(img.dataset.rot))}`;
      }
    });

    img.addEventListener('pointerup', () => {
      dragging = false;
      dragMode = 'move';
      img.style.cursor = 'grab';
    });

    img.addEventListener('click', (event) => {
      event.stopPropagation();
      setSelected(img);
    });

    img.addEventListener('wheel', (event) => {
      if (selected !== img) return;
      event.preventDefault();

      if (event.shiftKey) {
        const currentSize = Number(img.dataset.w);
        const nextSize = clamp(currentSize + (event.deltaY > 0 ? -10 : 10), 60, 500);
        img.dataset.w = String(nextSize);
        applyItemState(img);
        sizeRange.value = String(Math.round(nextSize));
        status.textContent = `Resized: ${img.dataset.name} | size=${Math.round(nextSize)}`;
        return;
      }

      const current = Number(img.dataset.rot);
      const next = current + (event.deltaY > 0 ? 5 : -5);
      img.dataset.rot = String(next);
      applyItemState(img);
      rotZRange.value = String(Math.round(next));
      status.textContent = `Rotated: ${img.dataset.name} | rot=${Math.round(next)}`;
    }, { passive: false });
  }

  function replaceSelectedWithExplicitView(explicitKey) {
    if (!selected) return;
    const current = selected;
    const name = current.dataset.name;
    const view = getAssetViewSrc(name, 0, current.dataset.flipped === 'true', explicitKey);
    if (!view.src) {
      current.remove();
      setSelected(null);
      status.textContent = `Missing ${explicitKey}° image for ${name}; item removed.`;
      return;
    }

    const replacement = createItemElement(name);
    replacement.dataset.x = current.dataset.x;
    replacement.dataset.y = current.dataset.y;
    replacement.dataset.w = current.dataset.w;
    replacement.dataset.rot = current.dataset.rot;
    replacement.dataset.rotx = current.dataset.rotx || '0';
    replacement.dataset.roty = '0';
    replacement.dataset.flipped = current.dataset.flipped || 'false';
    replacement.dataset.viewOverride = explicitKey;
    replacement.src = view.src;
    wireItemInteractions(replacement);

    current.replaceWith(replacement);
    applyItemState(replacement);
    setSelected(replacement);
    rotYRange.value = '0';
  }

  function addItem(name) {
    const img = createItemElement(name);
    wireItemInteractions(img);

    stage.appendChild(img);
    applyItemState(img);
    setSelected(img);
  }

  addBtn.addEventListener('click', () => {
    if (!assetSelect.value) return;
    addItem(assetSelect.value);
  });

  delBtn.addEventListener('click', () => {
    if (!selected) return;
    selected.remove();
    setSelected(null);
  });

  resetBtn.addEventListener('click', () => {
    // Remove all items from the stage
    stage.querySelectorAll('.drag-item').forEach(item => item.remove());
    setSelected(null);
  });

  rot90Btn.addEventListener('click', () => {
    if (!selected) return;
    const name = selected.dataset.name;
    replaceSelectedWithExplicitView('90');
    if (selected) {
      status.textContent = `View: ${name} | image=90°, rotY=0°`;
    }
  });

  rotMinus90Btn.addEventListener('click', () => {
    if (!selected) return;
    const name = selected.dataset.name;
    replaceSelectedWithExplicitView('-90');
    if (selected) {
      status.textContent = `View: ${name} | image=-90°, rotY=0°`;
    }
  });

  rot180Btn.addEventListener('click', () => {
    if (!selected) return;
    const name = selected.dataset.name;
    replaceSelectedWithExplicitView('180');
    if (selected) {
      status.textContent = `View: ${name} | image=180°, rotY=0°`;
    }
  });

  sizeRange.addEventListener('input', () => {
    if (!selected) return;
    selected.dataset.w = sizeRange.value;
    applyItemState(selected);
  });

  rotXRange.addEventListener('input', () => {
    if (!selected) return;
    selected.dataset.rotx = rotXRange.value;
    applyItemState(selected);
    status.textContent = `RotX: ${selected.dataset.name} | rotX=${rotXRange.value}, rotY=${selected.dataset.roty||0}, rotZ=${Math.round(Number(selected.dataset.rot))}`;
  });

  rotYRange.addEventListener('input', () => {
    if (!selected) return;
    selected.dataset.viewOverride = '';
    selected.dataset.roty = rotYRange.value;
    applyItemState(selected);
    status.textContent = `RotY: ${selected.dataset.name} | rotX=${selected.dataset.rotx||0}, rotY=${rotYRange.value}, rotZ=${Math.round(Number(selected.dataset.rot))}`;
  });

  rotZRange.addEventListener('input', () => {
    if (!selected) return;
    selected.dataset.rot = rotZRange.value;
    applyItemState(selected);
    status.textContent = `RotZ: ${selected.dataset.name} | rotX=${selected.dataset.rotx||0}, rotY=${selected.dataset.roty||0}, rotZ=${rotZRange.value}`;
  });

  rotLeftBtn.addEventListener('click', () => {
    if (!selected) return;
    const next = Number(selected.dataset.rot) - 15;
    selected.dataset.rot = String(next);
    applyItemState(selected);
    rotZRange.value = String(Math.round(next));
    status.textContent = `Rotated: ${selected.dataset.name} | rot=${Math.round(next)}`;
  });

  rotRightBtn.addEventListener('click', () => {
    if (!selected) return;
    const next = Number(selected.dataset.rot) + 15;
    selected.dataset.rot = String(next);
    applyItemState(selected);
    rotZRange.value = String(Math.round(next));
    status.textContent = `Rotated: ${selected.dataset.name} | rot=${Math.round(next)}`;
  });

  flipBtn.addEventListener('click', () => {
    if (!selected) return;
    selected.dataset.flipped = selected.dataset.flipped === 'true' ? 'false' : 'true';
    applyItemState(selected);
    status.textContent = `Flipped: ${selected.dataset.name} | flip=${selected.dataset.flipped}`;
  });

  stage.addEventListener('click', () => setSelected(null));
  stage.addEventListener('contextmenu', (event) => event.preventDefault());

  if (assetSelect.options.length > 0) addItem(assetSelect.options[0].value);
})();
</script>
</body>
</html>
"""
    return html.replace("__PAYLOAD_JSON__", payload_json)


def build_procedural_piano_model_data() -> dict[str, Any]:
    """Return a normalized multi-part piano model for stable rotation from any viewing angle."""
    return {
        "parts": [
            {"kind": "box", "size": [1.0, 0.10, 0.36], "center": [0.0, 0.93, 0.0], "color": "#1f2024"},
            {"kind": "box", "size": [0.92, 0.03, 0.14], "center": [0.0, 0.885, 0.07], "color": "#f2f2f2"},
            {"kind": "box", "size": [0.78, 0.025, 0.04], "center": [0.0, 0.872, 0.12], "color": "#1a1a1a"},
            {"kind": "box", "size": [0.20, 0.07, 0.03], "center": [0.0, 1.01, -0.12], "color": "#bfbfbf"},
            {
                "kind": "cylinder",
                "radius": 0.018,
                "height": 0.86,
                "center": [-0.44, 0.43, 0.12],
                "rotation_deg": [0.0, 0.0, 5.0],
                "color": "#44474e",
            },
            {
                "kind": "cylinder",
                "radius": 0.018,
                "height": 0.86,
                "center": [0.44, 0.43, 0.12],
                "rotation_deg": [0.0, 0.0, -5.0],
                "color": "#44474e",
            },
            {
                "kind": "cylinder",
                "radius": 0.017,
                "height": 0.84,
                "center": [-0.42, 0.42, -0.14],
                "rotation_deg": [6.0, 0.0, 6.0],
                "color": "#5b6068",
            },
            {
                "kind": "cylinder",
                "radius": 0.017,
                "height": 0.84,
                "center": [0.42, 0.42, -0.14],
                "rotation_deg": [6.0, 0.0, -6.0],
                "color": "#5b6068",
            },
        ],
        "model_note": "procedural piano for stable perspective",
    }


def infer_dimensions_meters(name: str, category: str) -> dict[str, float | str]:
    """Infer rough 3D dimensions from product metadata for fast interactive previews."""
    combined = f"{name} {category}".lower()

    if "piano" in combined or "keyboard" in combined:
        return {"width": 1.4, "height": 0.85, "depth": 0.45, "shape": "piano"}
    if "sofa" in combined or "bank" in combined or "couch" in combined:
        return {"width": 2.0, "height": 0.85, "depth": 0.95, "shape": "box"}
    if "armchair" in combined or "stoel" in combined or "chair" in combined:
        return {"width": 0.6, "height": 0.9, "depth": 0.65, "shape": "box"}
    if "table" in combined or "tafel" in combined or "desk" in combined:
        return {"width": 1.4, "height": 0.75, "depth": 0.8, "shape": "box"}
    if "bed" in combined:
        return {"width": 2.0, "height": 0.6, "depth": 1.6, "shape": "box"}
    if "cabinet" in combined or "kast" in combined or "wardrobe" in combined:
        return {"width": 1.0, "height": 1.8, "depth": 0.55, "shape": "box"}
    return {"width": 1.0, "height": 0.85, "depth": 0.7, "shape": "box"}


def ensure_3d_specs(asset_meta: dict[str, dict[str, str]]) -> dict[str, dict[str, float | str]]:
    """Persist 3D model specs for quick fetch; bootstrap with a couple of products first."""
    THREE_D_DIR.mkdir(parents=True, exist_ok=True)

    specs: dict[str, dict[str, float | str]] = {}
    if THREE_D_SPECS_PATH.exists():
        try:
            specs = json.loads(THREE_D_SPECS_PATH.read_text(encoding="utf-8"))
        except Exception:
            specs = {}

    # Bootstrap first two scraped products if specs are missing.
    scraped_names = [name for name, meta in asset_meta.items() if meta.get("site") != "Local"]
    for name in scraped_names[:2]:
        if name not in specs:
            meta = asset_meta.get(name, {})
            specs[name] = infer_dimensions_meters(name=name, category=meta.get("category", ""))

    THREE_D_SPECS_PATH.write_text(json.dumps(specs, ensure_ascii=True, indent=2), encoding="utf-8")
    return specs


def estimate_room_layout_from_photo(room_image: Image.Image) -> dict[str, Any]:
    """Estimate room anchor points (walls/floor/ceiling) from a single photo.

    The estimator combines RGB edge transitions and monocular depth transitions to
    infer a plausible back-wall rectangle and front-side bounds.
    """
    rgb = room_image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    if arr.ndim != 3 or arr.shape[0] < 32 or arr.shape[1] < 32:
      return {}

    h, w, _ = arr.shape
    gray = (arr[:, :, 0] * 0.299) + (arr[:, :, 1] * 0.587) + (arr[:, :, 2] * 0.114)

    depth_map = np.zeros((h, w), dtype=np.float32)
    has_depth = False
    try:
      estimator = get_depth_pipeline()
      depth_result = estimator(rgb)
      depth_img = depth_result.get("depth") if isinstance(depth_result, dict) else None
      if depth_img is not None:
        depth_img = depth_img.convert("L").resize((w, h), RESAMPLING.BILINEAR)
        depth_map = np.asarray(depth_img, dtype=np.float32) / 255.0
        has_depth = True
    except Exception:
      has_depth = False

    # First-order gradients for RGB and depth.
    dy_gray = np.abs(gray[1:, :] - gray[:-1, :])
    dx_gray = np.abs(gray[:, 1:] - gray[:, :-1])
    row_score = np.mean(dy_gray, axis=1)

    if has_depth:
      dy_depth = np.abs(depth_map[1:, :] - depth_map[:-1, :])
      row_score = row_score + (0.55 * np.mean(dy_depth, axis=1))

    # Smooth row-wise seam score to reduce local noise.
    kernel = np.ones(11, dtype=np.float32) / 11.0
    row_score = np.convolve(row_score, kernel, mode="same")

    floor_lo = max(1, int(h * 0.48))
    floor_hi = max(floor_lo + 2, int(h * 0.95))
    floor_slice = row_score[floor_lo:floor_hi]
    if floor_slice.size <= 0:
      return {}
    floor_y = floor_lo + int(np.argmax(floor_slice))

    ceil_lo = max(1, int(h * 0.04))
    ceil_hi = max(ceil_lo + 2, int(h * 0.45))
    ceil_slice = row_score[ceil_lo:ceil_hi]
    if ceil_slice.size <= 0:
      return {}
    ceil_y = ceil_lo + int(np.argmax(ceil_slice))

    band_half = max(4, int(h * 0.08))
    band_lo = max(1, floor_y - band_half)
    band_hi = min(h - 1, floor_y + band_half)
    band_dx = dx_gray[band_lo:band_hi, :]
    col_score = np.mean(band_dx, axis=0)
    if has_depth:
      dx_depth = np.abs(depth_map[:, 1:] - depth_map[:, :-1])
      col_score = col_score + (0.4 * np.mean(dx_depth[band_lo:band_hi, :], axis=0))

    # Left/right wall boundaries from strongest vertical transitions.
    left_lo = max(1, int(w * 0.02))
    left_hi = max(left_lo + 2, int(w * 0.48))
    right_lo = max(1, int(w * 0.52))
    right_hi = max(right_lo + 2, int(w * 0.98))
    left_x = left_lo + int(np.argmax(col_score[left_lo:left_hi]))
    right_x = right_lo + int(np.argmax(col_score[right_lo:right_hi]))
    if right_x <= left_x + 4:
      left_x = int(w * 0.22)
      right_x = int(w * 0.78)

    center_x = int((left_x + right_x) / 2)
    if has_depth:
      center_band_lo = max(left_x, int(w * 0.28))
      center_band_hi = min(right_x, int(w * 0.72))
      if center_band_hi > center_band_lo + 2:
        # Use the most distant-looking point near the floor seam as VP proxy.
        depth_row = depth_map[min(h - 1, max(0, floor_y)), center_band_lo:center_band_hi]
        center_x = center_band_lo + int(np.argmin(depth_row))

    # Infer perspective compression for back-wall span.
    floor_ratio = float(floor_y) / float(max(h - 1, 1))
    perspective = np.clip((1.0 - floor_ratio) * 1.7, 0.16, 0.44)
    front_span = max(20.0, float(right_x - left_x))
    back_span = front_span * (0.34 + 0.24 * perspective)

    back_left_x = int(np.clip(center_x - (back_span * 0.5), left_x + 4, right_x - 8))
    back_right_x = int(np.clip(center_x + (back_span * 0.5), back_left_x + 8, right_x - 4))

    # Pull front corners slightly toward image borders for side planes.
    left_front_x = int(np.clip(left_x - (0.14 * front_span), 1, left_x))
    right_front_x = int(np.clip(right_x + (0.14 * front_span), right_x, w - 2))

    # Back wall top should not cross floor seam and should remain in upper half.
    back_top_y = int(np.clip(ceil_y, int(h * 0.03), max(int(h * 0.42), floor_y - 14)))
    back_bottom_y = int(np.clip(floor_y, back_top_y + 12, int(h * 0.93)))

    def norm_x(v: int) -> float:
      return float(np.clip(v / float(max(w - 1, 1)), 0.0, 1.0))

    def norm_y(v: int) -> float:
      return float(np.clip(v / float(max(h - 1, 1)), 0.0, 1.0))

    return {
      "back_top_left": [norm_x(back_left_x), norm_y(back_top_y)],
      "back_top_right": [norm_x(back_right_x), norm_y(back_top_y)],
      "back_bottom_left": [norm_x(back_left_x), norm_y(back_bottom_y)],
      "back_bottom_right": [norm_x(back_right_x), norm_y(back_bottom_y)],
      "left_front_top": [norm_x(left_front_x), norm_y(max(1, back_top_y - int(h * 0.06)))],
      "left_front_bottom": [norm_x(left_front_x), norm_y(h - 2)],
      "right_front_top": [norm_x(right_front_x), norm_y(max(1, back_top_y - int(h * 0.06)))],
      "right_front_bottom": [norm_x(right_front_x), norm_y(h - 2)],
      "floor_front_left": [norm_x(left_front_x), norm_y(h - 2)],
      "floor_front_right": [norm_x(right_front_x), norm_y(h - 2)],
      "ceiling_front_left": [norm_x(left_front_x), norm_y(1)],
      "ceiling_front_right": [norm_x(right_front_x), norm_y(1)],
    }


def build_3d_payload(room: Image.Image, room_dimensions_cm: dict[str, float]) -> str:
    """Build payload for the interactive 3D editor from DB-stored 3D products."""
    room_preview = room.copy()
    room_preview.thumbnail((1200, 700), RESAMPLING.LANCZOS)
    room_data_url = image_to_data_url(room_preview)

    records = get_products_with_3d_models(limit=None)
    models: dict[str, dict[str, Any]] = {}

    def read_local_image_data_url(path_value: str) -> str:
      if not path_value:
        return ""
      try:
        path = Path(path_value)
        if not path.is_absolute():
          path = PROJECT_DIR / path
        if not path.exists():
          return ""
        image = Image.open(path).convert("RGBA")
        return image_to_data_url(image)
      except Exception:
        return ""

    def read_mesh_text(path_value: str) -> str:
      if not path_value:
        return ""
      try:
        path = Path(path_value)
        if path.is_absolute() and path_value.startswith("/app/static/"):
          relative = path_value.removeprefix("/app/")
          path = PROJECT_DIR / relative
        if not path.is_absolute():
          path = PROJECT_DIR / path
        if not path.exists():
          return ""
        return path.read_text(encoding="utf-8", errors="ignore")
      except Exception:
        return ""

    def is_target_local_record(record: dict[str, Any]) -> bool:
      site = str(record.get("site") or "")
      return site == CURATED_3D_SITE

    def is_renderable_3d(record: dict[str, Any], mesh_path: str, mesh_text: str, model_data: dict[str, Any]) -> bool:
      model_type = str(record.get("model_type") or "")
      if model_type == "triposr_cpu_v1":
        return bool(mesh_path.strip() or mesh_text.strip())
      if model_type in {
        "depth_mesh_cpu_v3",
        "depth_mesh_midas_v1",
        "depth_mesh_midas_v2_closed",
        "alpha_extrude_voxel",
        "procedural_piano_v1",
      }:
        return bool(model_data)
      return False

    local_records = [record for record in records if is_target_local_record(record)]
    local_records.sort(key=lambda r: (str(r.get("category") or ""), str(r.get("name") or "")))

    for record in local_records:
        display_name = f"{record['site']} / {record['category']} / {record['name']}"
        raw_model_data = str(record.get("model_data_json") or "")
        model_data: dict[str, Any] = {}
        if raw_model_data:
            try:
                model_data = json.loads(raw_model_data)
            except json.JSONDecodeError:
                model_data = {}

        mesh_path = str(record.get("mesh_path") or "")
        mesh_obj_text = read_mesh_text(mesh_path)
        if not is_renderable_3d(record, mesh_path, mesh_obj_text, model_data):
          continue

        texture_data_url = str(record["texture_data_url"] or record["cleaned_image_data_url"] or "")
        if not texture_data_url:
            texture_data_url = read_local_image_data_url(str(record.get("rotated_0_path") or ""))
        original_image_data_url = read_local_image_data_url(str(record.get("image_url") or ""))
        if not original_image_data_url:
          original_image_data_url = str(record.get("cleaned_image_data_url") or "")

        models[display_name] = {
            "model_type": record["model_type"],
            "width": float(record["width_m"]),
            "height": float(record["height_m"]),
            "depth": float(record["depth_m"]),
            "texture_data_url": texture_data_url,
          "original_image_data_url": original_image_data_url,
            "mesh_path": mesh_path,
            "mesh_obj_text": mesh_obj_text,
            "model_data": model_data,
        }

    payload = {
        "room": room_data_url,
        "models": models,
      "room_dimensions_cm": room_dimensions_cm,
      "room_layout": estimate_room_layout_from_photo(room_preview),
    }
    return json.dumps(payload)


def build_room_builder_payload(room_plan: dict[str, float | str]) -> str:
    """Build payload for a synthetic room builder without using room.jpg."""
    records = get_products_with_3d_models(limit=None)
    models: dict[str, dict[str, Any]] = {}

    def read_local_image_data_url(path_value: str) -> str:
        if not path_value:
            return ""
        try:
            path = Path(path_value)
            if not path.is_absolute():
                path = PROJECT_DIR / path
            if not path.exists():
                return ""
            image = Image.open(path).convert("RGBA")
            return image_to_data_url(image)
        except Exception:
            return ""

    def read_mesh_text(path_value: str) -> str:
        if not path_value:
            return ""
        try:
            path = Path(path_value)
            if path.is_absolute() and path_value.startswith("/app/static/"):
                relative = path_value.removeprefix("/app/")
                path = PROJECT_DIR / relative
            if not path.is_absolute():
                path = PROJECT_DIR / path
            if not path.exists():
                return ""
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""

    for record in records:
        site = str(record.get("site") or "")
        if site != CURATED_3D_SITE:
            continue

        display_name = f"{record['site']} / {record['category']} / {record['name']}"
        raw_model_data = str(record.get("model_data_json") or "")
        model_data: dict[str, Any] = {}
        if raw_model_data:
            try:
                model_data = json.loads(raw_model_data)
            except json.JSONDecodeError:
                model_data = {}

        mesh_path = str(record.get("mesh_path") or "")
        mesh_obj_text = read_mesh_text(mesh_path)
        model_type = str(record.get("model_type") or "")
        if model_type == "triposr_cpu_v1":
            if not (mesh_path.strip() or mesh_obj_text.strip()):
                continue
        elif model_type in {
            "depth_mesh_cpu_v3",
            "depth_mesh_midas_v1",
            "depth_mesh_midas_v2_closed",
            "alpha_extrude_voxel",
            "procedural_piano_v1",
        }:
            if not model_data:
                continue
        else:
            continue

        texture_data_url = str(record["texture_data_url"] or record["cleaned_image_data_url"] or "")
        if not texture_data_url:
            texture_data_url = read_local_image_data_url(str(record.get("rotated_0_path") or ""))
        original_image_data_url = read_local_image_data_url(str(record.get("image_url") or ""))
        if not original_image_data_url:
            original_image_data_url = str(record.get("cleaned_image_data_url") or "")

        models[display_name] = {
            "model_type": record["model_type"],
            "width": float(record["width_m"]),
            "height": float(record["height_m"]),
            "depth": float(record["depth_m"]),
          "price": str(record.get("price") or ""),
            "texture_data_url": texture_data_url,
            "original_image_data_url": original_image_data_url,
            "mesh_path": mesh_path,
            "mesh_obj_text": mesh_obj_text,
            "model_data": model_data,
        }

    return json.dumps({"room_plan": room_plan, "models": models})


def build_room_builder_html(payload_json: str) -> str:
    """Render a synthetic, configurable room builder with product placement."""
    html = """
<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <style>
    body { margin: 0; font-family: sans-serif; font-size: 17px; color: #1f2933; }
    .wrap {
      display: flex;
      gap: 12px;
      align-items: stretch;
      height: 760px;
    }
    .panel {
      width: 280px;
      border: 1px solid #ddd;
      border-radius: 10px;
      padding: 10px;
      box-sizing: border-box;
      background: #fafafa;
      height: 100%;
      overflow-y: auto;
    }
    .panel label { display: block; margin: 8px 0 4px; font-size: 16px; font-weight: 700; color: #1f2933; }
    .panel select, .panel button {
      width: 100%;
      height: 34px;
      margin-bottom: 8px;
      box-sizing: border-box;
      font-size: 16px;
    }
    .step-badge {
      display: inline-block;
      font-size: 15px;
      font-weight: 700;
      color: #2d3748;
      margin-bottom: 4px;
    }
    .step-title {
      font-size: 22px;
      font-weight: 700;
      line-height: 1.2;
      margin: 0 0 12px 0;
      color: #1f2730;
    }
    .step-note {
      font-size: 16px;
      color: #374151;
      margin: 0 0 12px 0;
      line-height: 1.4;
    }
    #viewport {
      flex: 1;
      height: 100%;
      border: 1px solid #bbb;
      border-radius: 10px;
      overflow: hidden;
      background: #eef2f6;
      position: relative;
    }
    .preview-box {
      border: 1px solid #ddd;
      border-radius: 8px;
      background: #fff;
      padding: 6px;
      margin: 8px 0;
    }
    .preview-box img {
      width: 100%;
      display: block;
      border-radius: 6px;
      object-fit: contain;
      background: #f7f7f7;
      min-height: 120px;
      max-height: 180px;
    }
    .note { font-size: 15px; color: #2f3b4a; margin-top: 6px; }
    .summary {
      font-size: 15px;
      color: #1f2933;
      border: 1px solid #d9dee5;
      border-radius: 8px;
      padding: 8px;
      background: #fff;
      margin-bottom: 8px;
    }
    .setup-actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin: 10px 0 8px;
    }
    .setup-primary {
      background: #111;
      color: #fff;
      border: 0;
      border-radius: 8px;
      font-weight: 700;
      cursor: pointer;
    }
    .setup-secondary {
      background: #fff;
      color: #111;
      border: 1px solid #cfd5dd;
      border-radius: 8px;
      font-weight: 700;
      cursor: pointer;
    }
    .order-panel {
      width: 210px;
      border: 1px solid #ddd;
      border-radius: 10px;
      padding: 10px;
      box-sizing: border-box;
      background: #fafafa;
      display: flex;
      flex-direction: column;
      align-self: stretch;
      height: 100%;
      overflow-y: auto;
    }
    .order-panel h4 {
      margin: 4px 0 10px 0;
    }
    .order-items {
      border: 1px solid #e0e4ea;
      background: #fff;
      border-radius: 8px;
      padding: 8px;
    }
    .order-item {
      padding: 8px 0;
      border-bottom: 1px solid #eceff4;
      font-size: 15px;
    }
    .order-item-name {
      font-weight: 700;
      color: #20262d;
      margin-bottom: 2px;
    }
    .order-item-meta {
      color: #3f4d5a;
      line-height: 1.35;
    }
    .order-item:last-child {
      border-bottom: 0;
    }
    .order-total {
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px solid #d9dee5;
      font-size: 16px;
      font-weight: 700;
    }
    .order-btn {
      width: 100%;
      height: 40px;
      margin-top: 10px;
      border: 0;
      border-radius: 8px;
      background: #111;
      color: #fff;
      font-weight: 700;
      cursor: pointer;
    }
    .dimension-overlay {
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      z-index: 5;
      display: flex;
      align-items: stretch;
      justify-content: stretch;
      background: rgba(238, 242, 246, 0.92);
    }
    .plan-editor-title {
      font-size: 15px;
      color: #444;
      margin-bottom: 8px;
      font-weight: 600;
    }
    .dimension-stage {
      width: 100%;
      height: 100%;
      display: flex;
      flex-direction: column;
      padding: 20px 24px;
      box-sizing: border-box;
    }
    .dimension-overlay svg {
      width: 100%;
      height: 100%;
      display: block;
      background: transparent;
      border-radius: 6px;
      flex: 1;
    }
    .is-hidden { display: none !important; }
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"panel\">
      <div id=\"stepBadge\" class=\"step-badge\">Step 1 of 3</div>
      <div id=\"stepTitle\" class=\"step-title\">Select room shape</div>
      <div id=\"stepNote\" class=\"step-note\">Choose the room type first, then adjust dimensions directly on the room before you start placing furniture.</div>
      <div id=\"summary\" class=\"summary\"></div>
      <label>Room size preset</label>
      <select id=\"roomPresetSelect\"></select>
      <label>Room shape</label>
      <select id=\"roomShapeSelect\"></select>
      <div class=\"setup-actions\">
        <button id=\"editSizeBtn\" class=\"setup-secondary\" type=\"button\">Adjust Size</button>
        <button id=\"startDesignBtn\" class=\"setup-primary\" type=\"button\">Start Designing</button>
      </div>
      <div id=\"designControls\" class=\"is-hidden\">
        <label>Product</label>
        <select id=\"assetSelect\"></select>
        <button id=\"addBtn\">Add Product</button>
        <button id=\"moveBtn\">Move Mode</button>
        <button id=\"rotateBtn\">Rotate Mode</button>
        <button id=\"viewBtn\">Enable Orbit</button>
        <div style=\"display:grid;grid-template-columns:1fr 1fr;gap:6px;\">
          <button id=\"frontViewBtn\">Front View</button>
          <button id=\"leftViewBtn\">Left View</button>
          <button id=\"rightViewBtn\">Right View</button>
          <button id=\"topViewBtn\">Top View</button>
        </div>
        <button id=\"fitBtn\">Fit In Room</button>
        <button id=\"alignWallBtn\">Snap Parallel To Wall</button>
        <button id=\"deleteBtn\">Delete Selected</button>
        <label>Original 2D Photo</label>
        <div class="preview-box"><img id="originalPreview" alt="Original product photo" /></div>
      </div>
      <div id=\"statusNote\" class=\"note\">Synthetic room mode: first choose shape, then adjust the dimensions directly on the room. After that you can start placing products.</div>
      <div id=\"status\" class=\"note\">Status: ready</div>
    </div>
    <div id=\"viewport\"><div id=\"dimensionOverlay\" class=\"dimension-overlay\"><div class=\"dimension-stage\"><div id=\"dimensionTitle\" class=\"plan-editor-title\">Drag the room handles to resize the selected shape</div><svg id=\"planSvg\" viewBox=\"0 0 900 620\" aria-label=\"Room plan editor\"></svg></div></div></div>
    <div class=\"order-panel\">
      <h4>Order Summary</h4>
      <div id=\"orderItems\" class=\"order-items\"></div>
      <div id=\"orderTotal\" class=\"order-total\">Total: EUR 0.00</div>
      <button id=\"orderBtn\" class=\"order-btn\" type=\"button\">Go To Order</button>
    </div>
  </div>

  <script src=\"https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js\"></script>
  <script src=\"https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js\"></script>
  <script src=\"https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/OBJLoader.js\"></script>
  <script>
    (() => {
      const DATA = __PAYLOAD_JSON__;
      const viewport = document.getElementById('viewport');
      const stepBadge = document.getElementById('stepBadge');
      const stepTitle = document.getElementById('stepTitle');
      const stepNote = document.getElementById('stepNote');
      const summary = document.getElementById('summary');
      const planSvg = document.getElementById('planSvg');
      const dimensionOverlay = document.getElementById('dimensionOverlay');
      const dimensionTitle = document.getElementById('dimensionTitle');
      const roomPresetSelect = document.getElementById('roomPresetSelect');
      const roomShapeSelect = document.getElementById('roomShapeSelect');
      const editSizeBtn = document.getElementById('editSizeBtn');
      const startDesignBtn = document.getElementById('startDesignBtn');
      const designControls = document.getElementById('designControls');
      const assetSelect = document.getElementById('assetSelect');
      const addBtn = document.getElementById('addBtn');
      const moveBtn = document.getElementById('moveBtn');
      const rotateBtn = document.getElementById('rotateBtn');
      const viewBtn = document.getElementById('viewBtn');
      const frontViewBtn = document.getElementById('frontViewBtn');
      const leftViewBtn = document.getElementById('leftViewBtn');
      const rightViewBtn = document.getElementById('rightViewBtn');
      const topViewBtn = document.getElementById('topViewBtn');
      const fitBtn = document.getElementById('fitBtn');
      const alignWallBtn = document.getElementById('alignWallBtn');
      const deleteBtn = document.getElementById('deleteBtn');
      const originalPreview = document.getElementById('originalPreview');
      const orderItems = document.getElementById('orderItems');
      const orderTotal = document.getElementById('orderTotal');
      const statusNote = document.getElementById('statusNote');
      const status = document.getElementById('status');

      const roomPlan = DATA.room_plan || {};
      const ROOM_PRESETS = {
        Compact: { width_cm: 420, depth_cm: 320, height_cm: 240 },
        Standard: { width_cm: 600, depth_cm: 480, height_cm: 245 },
        Large: { width_cm: 760, depth_cm: 560, height_cm: 260 },
        Studio: { width_cm: 900, depth_cm: 700, height_cm: 280 },
      };
      const ROOM_SHAPES = [
        { value: 'rectangle', label: 'Rectangle' },
        { value: 'l_shape', label: 'L-Shape' },
        { value: 't_shape', label: 'T-Shape' },
        { value: 'cut_corner', label: 'Cut Corner' },
      ];
      const roomState = {
        shape: String(roomPlan.shape || 'rectangle'),
        sizePreset: String(roomPlan.size_preset || 'Standard'),
        width_cm: Number(roomPlan.width_cm || 600),
        depth_cm: Number(roomPlan.depth_cm || 480),
        height_cm: Number(roomPlan.height_cm || 245),
        cut_width_cm: Number(roomPlan.cut_width_cm || 300),
        cut_depth_cm: Number(roomPlan.cut_depth_cm || 240),
        stem_width_cm: Number(roomPlan.stem_width_cm || 240),
        stem_depth_cm: Number(roomPlan.stem_depth_cm || 170),
        cut_size_cm: Number(roomPlan.cut_size_cm || 140),
      };
      let setupStep = 'shape';
      const modelNames = Object.keys(DATA.models || {});
      Object.keys(ROOM_PRESETS).forEach((name) => {
        const option = document.createElement('option');
        option.value = name;
        option.textContent = name;
        roomPresetSelect.appendChild(option);
      });
      ROOM_SHAPES.forEach((shape) => {
        const option = document.createElement('option');
        option.value = shape.value;
        option.textContent = shape.label;
        roomShapeSelect.appendChild(option);
      });
      roomPresetSelect.value = roomState.sizePreset;
      roomShapeSelect.value = roomState.shape;
      modelNames.forEach((name) => {
        const option = document.createElement('option');
        option.value = name;
        option.textContent = name;
        assetSelect.appendChild(option);
      });

      function formatShapeLabel(raw) {
        return String(raw || 'rectangle').replace(/_/g, ' ').replace(/\\b\\w/g, (ch) => ch.toUpperCase());
      }

      let ROOM_HEIGHT_M = Math.max(1.8, roomState.height_cm / 100);

      function refreshSummary() {
        summary.textContent = `${formatShapeLabel(roomState.shape)} | ${roomState.width_cm.toFixed(0)} x ${roomState.depth_cm.toFixed(0)} x ${roomState.height_cm.toFixed(0)} cm`;
      }

      function clampRoomParameters() {
        roomState.width_cm = clamp(roomState.width_cm, 200, 3000);
        roomState.depth_cm = clamp(roomState.depth_cm, 200, 3000);
        roomState.height_cm = clamp(roomState.height_cm, 180, 600);
        roomState.cut_width_cm = clamp(roomState.cut_width_cm, 80, Math.max(90, roomState.width_cm - 80));
        roomState.cut_depth_cm = clamp(roomState.cut_depth_cm, 80, Math.max(90, roomState.depth_cm - 80));
        roomState.stem_width_cm = clamp(roomState.stem_width_cm, 80, Math.max(90, roomState.width_cm - 80));
        roomState.stem_depth_cm = clamp(roomState.stem_depth_cm, 80, Math.max(90, roomState.depth_cm - 80));
        roomState.cut_size_cm = clamp(roomState.cut_size_cm, 60, Math.max(70, Math.min(roomState.width_cm, roomState.depth_cm) - 60));
      }

      function setSetupStep(nextStep) {
        setupStep = nextStep;
        if (setupStep === 'shape') {
          stepBadge.textContent = 'Step 1 of 3';
          stepTitle.textContent = 'Select room shape';
          stepNote.textContent = 'Choose a room shape and size preset. Then continue to adjust the dimensions with the mouse.';
          statusNote.textContent = 'Synthetic room mode: first choose the room shape and preset.';
          dimensionTitle.textContent = 'Preview the selected room shape';
          dimensionOverlay.classList.remove('is-hidden');
          designControls.classList.add('is-hidden');
          editSizeBtn.textContent = 'Adjust Size';
          startDesignBtn.textContent = 'Start Designing';
          orbitEnabled = false;
          orbit.enabled = false;
          orbit.enableRotate = false;
          orbit.enablePan = false;
          orbit.enableZoom = false;
          viewBtn.textContent = 'Enable Orbit';
          if (grid) {
            grid.visible = true;
          }
          setCameraPreset('top');
        } else if (setupStep === 'size') {
          stepBadge.textContent = 'Step 2 of 3';
          stepTitle.textContent = 'Adjust room dimensions';
          stepNote.textContent = 'Drag the dimension handles directly on the room to match the real space. Dimensions stay visible while you edit.';
          statusNote.textContent = 'Synthetic room mode: drag the handles directly on the room canvas to adjust dimensions.';
          dimensionTitle.textContent = 'Drag the room handles to resize the selected shape';
          dimensionOverlay.classList.remove('is-hidden');
          designControls.classList.add('is-hidden');
          editSizeBtn.textContent = 'Back To Shape';
          startDesignBtn.textContent = 'Start Designing';
          orbitEnabled = false;
          orbit.enabled = false;
          orbit.enableRotate = false;
          orbit.enablePan = false;
          orbit.enableZoom = false;
          viewBtn.textContent = 'Enable Orbit';
          if (grid) {
            grid.visible = true;
          }
          setCameraPreset('top');
        } else {
          stepBadge.textContent = 'Step 3 of 3';
          stepTitle.textContent = 'Design the room';
          stepNote.textContent = '';
          statusNote.textContent = '';
          dimensionOverlay.classList.add('is-hidden');
          designControls.classList.remove('is-hidden');
          editSizeBtn.textContent = 'Edit Size';
          startDesignBtn.textContent = 'Designing';
          orbitEnabled = true;
          orbit.enabled = true;
          orbit.enableRotate = true;
          orbit.enablePan = true;
          orbit.enableZoom = true;
          viewBtn.textContent = 'Lock Orbit';
          if (grid) {
            grid.visible = false;
          }
          setCameraPreset('design');
        }
      }

      refreshSummary();

      function buildFootprint(plan) {
        const shape = String(plan.shape || 'rectangle');
        const width = Math.max(2.0, Number(plan.width_cm || 600) / 100);
        const depth = Math.max(2.0, Number(plan.depth_cm || 480) / 100);
        const halfW = width / 2;

        if (shape === 'l_shape') {
          const cutWidth = Math.max(0.8, Math.min(width - 0.8, Number(plan.cut_width_cm || 300) / 100));
          const cutDepth = Math.max(0.8, Math.min(depth - 0.8, Number(plan.cut_depth_cm || 240) / 100));
          return [
            [-halfW, 0],
            [halfW, 0],
            [halfW, depth],
            [-halfW + cutWidth, depth],
            [-halfW + cutWidth, depth - cutDepth],
            [-halfW, depth - cutDepth],
          ];
        }

        if (shape === 't_shape') {
          const stemWidth = Math.max(0.8, Math.min(width - 0.8, Number(plan.stem_width_cm || 240) / 100));
          const stemDepth = Math.max(0.8, Math.min(depth - 0.8, Number(plan.stem_depth_cm || 170) / 100));
          const stemLeft = -stemWidth / 2;
          const stemRight = stemWidth / 2;
          return [
            [-halfW, 0],
            [halfW, 0],
            [halfW, stemDepth],
            [stemRight, stemDepth],
            [stemRight, depth],
            [stemLeft, depth],
            [stemLeft, stemDepth],
            [-halfW, stemDepth],
          ];
        }

        if (shape === 'cut_corner') {
          const cutSize = Math.max(0.6, Math.min(Math.min(width, depth) - 0.6, Number(plan.cut_size_cm || 140) / 100));
          return [
            [-halfW + cutSize, 0],
            [halfW, 0],
            [halfW, depth],
            [-halfW, depth],
            [-halfW, cutSize],
          ];
        }

        return [
          [-halfW, 0],
          [halfW, 0],
          [halfW, depth],
          [-halfW, depth],
        ];
      }

      function polygonBounds(points) {
        let minX = Infinity;
        let maxX = -Infinity;
        let minZ = Infinity;
        let maxZ = -Infinity;
        points.forEach(([x, z]) => {
          minX = Math.min(minX, x);
          maxX = Math.max(maxX, x);
          minZ = Math.min(minZ, z);
          maxZ = Math.max(maxZ, z);
        });
        return { minX, maxX, minZ, maxZ };
      }

      let footprint = [];
      let bounds = { minX: 0, maxX: 0, minZ: 0, maxZ: 0 };
      let roomCenterX = 0;
      let roomCenterZ = 0;
      let roomSpanX = 0;
      let roomSpanZ = 0;

      function updateRoomMetrics() {
        clampRoomParameters();
        ROOM_HEIGHT_M = Math.max(1.8, roomState.height_cm / 100);
        footprint = buildFootprint(roomState);
        bounds = polygonBounds(footprint);
        roomCenterX = (bounds.minX + bounds.maxX) / 2;
        roomCenterZ = (bounds.minZ + bounds.maxZ) / 2;
        roomSpanX = bounds.maxX - bounds.minX;
        roomSpanZ = bounds.maxZ - bounds.minZ;
      }

      updateRoomMetrics();

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0xd3d9e2);

      const camera = new THREE.PerspectiveCamera(36, viewport.clientWidth / viewport.clientHeight, 0.1, 200);
      camera.position.set(roomCenterX + roomSpanX * 0.15, ROOM_HEIGHT_M * 1.25, bounds.maxZ + Math.max(roomSpanX, roomSpanZ) * 0.95 + 1.4);

      const renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setPixelRatio(window.devicePixelRatio);
      renderer.setSize(viewport.clientWidth, viewport.clientHeight);
      renderer.outputEncoding = THREE.sRGBEncoding;
      viewport.appendChild(renderer.domElement);

      const orbit = new THREE.OrbitControls(camera, renderer.domElement);
      orbit.enableDamping = true;
      orbit.dampingFactor = 0.08;
      orbit.target.set(roomCenterX, ROOM_HEIGHT_M * 0.35, roomCenterZ);
      orbit.enableRotate = false;
      orbit.enablePan = false;
      orbit.enableZoom = false;
      orbit.enabled = false;
      orbit.update();

      const ambient = new THREE.AmbientLight(0xffffff, 0.82);
      const key = new THREE.DirectionalLight(0xffffff, 0.78);
      key.position.set(4, 8, 5);
      const rim = new THREE.DirectionalLight(0xe8eef7, 0.16);
      rim.position.set(-5, 5, -3);
      scene.add(ambient, key, rim);

      function makeFloorTexture() {
        const canvas = document.createElement('canvas');
        canvas.width = 512;
        canvas.height = 512;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#c8a17b';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        for (let row = 0; row < 16; row += 1) {
          for (let col = 0; col < 8; col += 1) {
            const offset = row % 2 === 0 ? 0 : 32;
            const x = (col * 64 + offset) % 512;
            const y = row * 32;
            ctx.fillStyle = (row + col) % 3 === 0 ? '#d6b48d' : ((row + col) % 3 === 1 ? '#bf9a72' : '#cda67c');
            ctx.fillRect(x, y, 64, 32);
          }
        }
        const texture = new THREE.CanvasTexture(canvas);
        texture.wrapS = THREE.RepeatWrapping;
        texture.wrapT = THREE.RepeatWrapping;
        texture.repeat.set(Math.max(2, roomSpanX), Math.max(2, roomSpanZ));
        texture.encoding = THREE.sRGBEncoding;
        return texture;
      }

      const roomGeometryGroup = new THREE.Group();
      scene.add(roomGeometryGroup);
      let grid = null;
      let roomWalls = [];

      function rebuildRoomGeometry() {
        while (roomGeometryGroup.children.length) {
          roomGeometryGroup.remove(roomGeometryGroup.children[0]);
        }
        if (grid) {
          scene.remove(grid);
        }
        roomWalls = [];

        const floorShape = new THREE.Shape();
        footprint.forEach(([x, z], index) => {
          if (index === 0) {
            floorShape.moveTo(x, z);
          } else {
            floorShape.lineTo(x, z);
          }
        });
        floorShape.closePath();

        const floorGeometry = new THREE.ShapeGeometry(floorShape);
        floorGeometry.rotateX(-Math.PI / 2);
        const floorMesh = new THREE.Mesh(
          floorGeometry,
          // Flat gray floor to blend with background and avoid highlight artifacts.
          new THREE.MeshBasicMaterial({ color: 0xd3d9e2 })
        );
        roomGeometryGroup.add(floorMesh);

        for (let index = 0; index < footprint.length; index += 1) {
          const [x1, z1] = footprint[index];
          const [x2, z2] = footprint[(index + 1) % footprint.length];
          const length = Math.hypot(x2 - x1, z2 - z1);
          const midX = (x1 + x2) / 2;
          const midZ = (z1 + z2) / 2;
          const isHorizontal = Math.abs(x2 - x1) >= Math.abs(z2 - z1);
          let wallSide = 'interior';
          if (isHorizontal) {
            wallSide = Math.abs(midZ - bounds.minZ) <= Math.abs(midZ - bounds.maxZ) ? 'back' : 'front';
          } else {
            wallSide = Math.abs(midX - bounds.minX) <= Math.abs(midX - bounds.maxX) ? 'left' : 'right';
          }
          const wall = new THREE.Mesh(
            new THREE.BoxGeometry(length, ROOM_HEIGHT_M, 0.06),
            new THREE.MeshStandardMaterial({ color: 0xf6f6f3, roughness: 0.97, metalness: 0.0 })
          );
          wall.position.set(midX, ROOM_HEIGHT_M / 2, midZ);
          wall.rotation.y = Math.atan2(z2 - z1, x2 - x1);
          roomGeometryGroup.add(wall);
          roomWalls.push({ mesh: wall, side: wallSide });
        }

        const edgeMaterial = new THREE.LineBasicMaterial({ color: 0xffffff });
        const topOutlinePoints = [];
        footprint.forEach(([x, z]) => topOutlinePoints.push(new THREE.Vector3(x, ROOM_HEIGHT_M, z)));
        topOutlinePoints.push(new THREE.Vector3(footprint[0][0], ROOM_HEIGHT_M, footprint[0][1]));
        roomGeometryGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(topOutlinePoints), edgeMaterial));

        grid = new THREE.GridHelper(Math.max(roomSpanX, roomSpanZ) * 1.3, 20, 0x94a5b9, 0xc9d2dd);
        grid.position.set(roomCenterX, 0.002, roomCenterZ);
        scene.add(grid);
      }

      rebuildRoomGeometry();

      let selected = null;
      let interactionMode = 'move';
      let orbitEnabled = false;
      let dragActive = false;
      let dragPointerId = null;
      let dragStartX = 0;
      let dragStartY = 0;
      let dragStartLift = 0;
      let dragStartYaw = 0;
      let dragStartPitch = 0;
      let dragStartRoll = 0;
      const productMeshes = [];
      const raycaster = new THREE.Raycaster();
      const pointer = new THREE.Vector2();
      const floorPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
      const dragPoint = new THREE.Vector3();
      const dragOffset = new THREE.Vector3();

      function pointInPolygon(x, z, polygon) {
        let inside = false;
        for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
          const xi = polygon[i][0];
          const zi = polygon[i][1];
          const xj = polygon[j][0];
          const zj = polygon[j][1];
          const intersect = ((zi > z) !== (zj > z)) && (x < ((xj - xi) * (z - zi)) / ((zj - zi) || 1e-6) + xi);
          if (intersect) inside = !inside;
        }
        return inside;
      }

      function clamp(value, lo, hi) {
        return Math.max(lo, Math.min(hi, value));
      }

      function deg(rad) {
        return rad * 57.2958;
      }

      function updatePointerFromEvent(event) {
        const rect = renderer.domElement.getBoundingClientRect();
        pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      }

      function getSpec(name) {
        const spec = (DATA.models && DATA.models[name]) || {
          model_type: 'depth_mesh_midas_v1',
          width: 1.0,
          height: 0.8,
          depth: 0.7,
          price: '',
          texture_data_url: '',
          mesh_path: '',
          mesh_obj_text: '',
          model_data: {},
        };
        return {
          modelType: String(spec.model_type || 'depth_mesh_midas_v1'),
          width: Number(spec.width || 1.0),
          height: Number(spec.height || 0.8),
          depth: Number(spec.depth || 0.7),
          price: String(spec.price || ''),
          textureDataUrl: String(spec.texture_data_url || ''),
          originalImageDataUrl: String(spec.original_image_data_url || ''),
          meshPath: String(spec.mesh_path || ''),
          meshObjText: String(spec.mesh_obj_text || ''),
          modelData: spec.model_data || {},
        };
      }

      function updateSelectedPreview() {
        if (!selected) {
          originalPreview.removeAttribute('src');
          return;
        }
        const src = String(selected.userData.originalImageDataUrl || '');
        if (src) {
          originalPreview.src = src;
        } else {
          originalPreview.removeAttribute('src');
        }
      }

      function parsePriceValue(priceText) {
        const normalized = String(priceText || '').replace(/,/g, '.');
        const match = normalized.match(/(\\d+(?:\\.\\d+)?)/);
        return match ? Number(match[1]) : 0;
      }

      function updateOrderSummary() {
        if (!productMeshes.length) {
          orderItems.innerHTML = '<div class="order-item">No products added yet.</div>';
          orderTotal.textContent = 'Total: EUR 0.00';
          return;
        }
        let total = 0;
        orderItems.innerHTML = productMeshes.map((mesh) => {
          const priceLabel = String(mesh.userData.priceLabel || 'Price on request');
          total += Number(mesh.userData.priceValue || 0);
          const shortName = String(mesh.userData.name || '').split('/').pop().trim();
          const sourceLabel = String(mesh.userData.sourceLabel || 'Curated product');
          const dimensions = `${Number(mesh.userData.widthM || 0).toFixed(2)}m x ${Number(mesh.userData.depthM || 0).toFixed(2)}m x ${Number(mesh.userData.heightM || 0).toFixed(2)}m`;
          return `<div class="order-item"><div class="order-item-name">${shortName}</div><div class="order-item-meta">${priceLabel}</div><div class="order-item-meta">${dimensions}</div><div class="order-item-meta">${sourceLabel}</div></div>`;
        }).join('');
        orderTotal.textContent = `Total: EUR ${total.toFixed(2)}`;
      }

      function getMeshExtents(mesh) {
        const box = new THREE.Box3().setFromObject(mesh);
        return {
          minOffsetY: box.min.y - mesh.position.y,
          maxOffsetY: box.max.y - mesh.position.y,
        };
      }

      function getFloorSnappedY(mesh) {
        const extents = getMeshExtents(mesh);
        return -extents.minOffsetY;
      }

      function getCeilingClampedY(mesh) {
        const extents = getMeshExtents(mesh);
        return ROOM_HEIGHT_M - extents.maxOffsetY;
      }

      function constrainXZ(currentX, currentZ, targetX, targetZ) {
        if (pointInPolygon(targetX, targetZ, footprint)) {
          return { x: targetX, z: targetZ };
        }
        for (let factor = 0.95; factor >= 0; factor -= 0.05) {
          const x = currentX + ((targetX - currentX) * factor);
          const z = currentZ + ((targetZ - currentZ) * factor);
          if (pointInPolygon(x, z, footprint)) {
            return { x, z };
          }
        }
        return { x: currentX, z: currentZ };
      }

      function clampPositionToRoom(mesh, x, y, z) {
        const current = mesh.userData.targetPos || mesh.position;
        const next = constrainXZ(current.x, current.z, x, z);
        return new THREE.Vector3(next.x, clamp(y, getFloorSnappedY(mesh), getCeilingClampedY(mesh)), next.z);
      }

      function snapMeshToFloor(mesh) {
        mesh.position.y = getFloorSnappedY(mesh);
      }

      function fitMeshToRoom(mesh) {
        const box = new THREE.Box3().setFromObject(mesh);
        if (box.isEmpty()) {
          return;
        }
        const size = new THREE.Vector3();
        box.getSize(size);
        const uniformScale = Math.min(
          (roomSpanX * 0.45) / Math.max(size.x, 0.001),
          (roomSpanZ * 0.45) / Math.max(size.z, 0.001),
          (ROOM_HEIGHT_M * 0.76) / Math.max(size.y, 0.001),
          1.0
        );
        if (uniformScale < 0.999) {
          mesh.scale.multiplyScalar(uniformScale);
        }
        const placed = clampPositionToRoom(mesh, roomCenterX, getFloorSnappedY(mesh), roomCenterZ + Math.min(0.4, roomSpanZ * 0.08));
        mesh.position.copy(placed);
        mesh.userData.targetPos = placed.clone();
      }

      function setCameraPreset(preset) {
        const sceneBox = new THREE.Box3(
          new THREE.Vector3(bounds.minX, 0, bounds.minZ),
          new THREE.Vector3(bounds.maxX, ROOM_HEIGHT_M, bounds.maxZ)
        );
        for (let index = 0; index < productMeshes.length; index += 1) {
          sceneBox.expandByObject(productMeshes[index]);
        }
        const focus = new THREE.Vector3();
        sceneBox.getCenter(focus);
        focus.y = Math.max(ROOM_HEIGHT_M * 0.35, focus.y);
        const sceneSize = new THREE.Vector3();
        sceneBox.getSize(sceneSize);
        const radius = Math.max(sceneSize.x, sceneSize.z) * 0.92 + Math.max(ROOM_HEIGHT_M, 1.8);
        if (preset === 'front') {
          camera.position.set(focus.x, focus.y + (ROOM_HEIGHT_M * 0.12), bounds.maxZ + radius);
        } else if (preset === 'left') {
          camera.position.set(bounds.minX - radius, focus.y + (ROOM_HEIGHT_M * 0.12), focus.z);
        } else if (preset === 'right') {
          camera.position.set(bounds.maxX + radius, focus.y + (ROOM_HEIGHT_M * 0.12), focus.z);
        } else if (preset === 'top') {
          camera.position.set(focus.x, Math.max(sceneSize.x, sceneSize.z) * 1.55, focus.z + 0.01);
        } else if (preset === 'design') {
          camera.position.set(
            focus.x - (sceneSize.x * 0.72),
            Math.max(ROOM_HEIGHT_M * 1.12, sceneSize.z * 0.48),
            bounds.maxZ + (radius * 0.42)
          );
        }
        for (let index = 0; index < roomWalls.length; index += 1) {
          roomWalls[index].mesh.visible = true;
        }
        if (preset === 'front') {
          roomWalls.filter((wall) => wall.side === 'front').forEach((wall) => { wall.mesh.visible = false; });
        } else if (preset === 'left') {
          roomWalls.filter((wall) => wall.side === 'left').forEach((wall) => { wall.mesh.visible = false; });
        } else if (preset === 'right') {
          roomWalls.filter((wall) => wall.side === 'right').forEach((wall) => { wall.mesh.visible = false; });
        }
        orbit.target.copy(focus);
        orbit.update();
      }

      function setAllWallsVisible() {
        for (let index = 0; index < roomWalls.length; index += 1) {
          roomWalls[index].mesh.visible = true;
        }
      }

      function reflowProductsToRoom() {
        for (let index = 0; index < productMeshes.length; index += 1) {
          const mesh = productMeshes[index];
          const target = mesh.userData.targetPos || mesh.position;
          const clamped = clampPositionToRoom(mesh, target.x, target.y, target.z);
          mesh.position.copy(clamped);
          mesh.userData.targetPos = clamped.clone();
        }
      }

      function renderPlanEditor() {
        const margin = 72;
        const widthPx = 900;
        const heightPx = 620;
        const availableW = widthPx - (margin * 2);
        const availableH = heightPx - (margin * 2);
        const scale = Math.min(availableW / Math.max(roomSpanX, 0.001), availableH / Math.max(roomSpanZ, 0.001));
        const offsetX = widthPx / 2;
        const offsetY = margin;

        const planPoints = footprint.map(([x, z]) => {
          const px = offsetX + (x * scale);
          const py = offsetY + (z * scale);
          return `${px.toFixed(1)},${py.toFixed(1)}`;
        }).join(' ');

        const rightHandleX = offsetX + ((bounds.maxX + 0.2) * scale);
        const rightHandleY = offsetY + ((bounds.minZ + bounds.maxZ) * 0.5 * scale);
        const bottomHandleX = offsetX + (((bounds.minX + bounds.maxX) * 0.5) * scale);
        const bottomHandleY = offsetY + ((bounds.maxZ + 0.2) * scale);

        const cutWidthCm = Math.round(roomState.cut_width_cm);
        const cutDepthCm = Math.round(roomState.cut_depth_cm);
        const stemWidthCm = Math.round(roomState.stem_width_cm);
        const stemDepthCm = Math.round(roomState.stem_depth_cm);
        const cutCornerCm = Math.round(roomState.cut_size_cm);

        let extraLabels = '';
        let extraHandles = '';
        if (roomState.shape === 'l_shape') {
          extraLabels = `
            <text x="${offsetX - ((roomSpanX * 0.18) * scale)}" y="${offsetY + ((bounds.maxZ - ((roomState.cut_depth_cm / 100) * 0.5)) * scale) - 8}" text-anchor="middle" font-size="22" fill="#626d79">${cutWidthCm} cm</text>
            <text x="${offsetX + (((bounds.maxX - (roomState.cut_width_cm / 100)) + 0.16) * scale)}" y="${offsetY + (((roomSpanZ - (roomState.cut_depth_cm / 100)) + ((roomState.cut_depth_cm / 100) * 0.5)) * scale)}" font-size="22" fill="#626d79" transform="rotate(90 ${offsetX + (((bounds.maxX - (roomState.cut_width_cm / 100)) + 0.16) * scale)} ${offsetY + (((roomSpanZ - (roomState.cut_depth_cm / 100)) + ((roomState.cut_depth_cm / 100) * 0.5)) * scale)})">${cutDepthCm} cm</text>
          `;
          extraHandles = `
            <circle id="cutWidthHandle" cx="${offsetX + ((bounds.maxX - (roomState.cut_width_cm / 100)) * scale)}" cy="${offsetY + (bounds.maxZ * scale)}" r="12" fill="#f2b705"></circle>
            <circle id="cutDepthHandle" cx="${offsetX + ((bounds.maxX - (roomState.cut_width_cm / 100)) * scale)}" cy="${offsetY + ((bounds.maxZ - (roomState.cut_depth_cm / 100)) * scale)}" r="12" fill="#f2b705"></circle>
          `;
        } else if (roomState.shape === 't_shape') {
          extraLabels = `
            <text x="${offsetX}" y="${offsetY + ((roomSpanZ * 0.34) * scale) - 10}" text-anchor="middle" font-size="22" fill="#626d79">${stemWidthCm} cm</text>
            <text x="${offsetX + ((roomSpanX * 0.34) * scale)}" y="${offsetY + ((roomSpanZ * 0.22) * scale)}" font-size="22" fill="#626d79" transform="rotate(90 ${offsetX + ((roomSpanX * 0.34) * scale)} ${offsetY + ((roomSpanZ * 0.22) * scale)})">${stemDepthCm} cm</text>
          `;
          extraHandles = `
            <circle id="stemWidthHandle" cx="${offsetX + (((roomState.stem_width_cm / 100) / 2) * scale)}" cy="${offsetY + ((roomState.stem_depth_cm / 100) * scale)}" r="12" fill="#f2b705"></circle>
            <circle id="stemDepthHandle" cx="${offsetX + (((bounds.maxX + (roomState.stem_width_cm / 200))) * scale)}" cy="${offsetY + ((roomState.stem_depth_cm / 100) * scale)}" r="12" fill="#f2b705"></circle>
          `;
        } else if (roomState.shape === 'cut_corner') {
          extraLabels = `
            <text x="${offsetX - ((roomSpanX * 0.34) * scale)}" y="${offsetY + ((roomSpanZ * 0.16) * scale)}" font-size="22" fill="#626d79">${cutCornerCm} cm</text>
          `;
          extraHandles = `
            <circle id="cutCornerHandle" cx="${offsetX + ((bounds.minX + (roomState.cut_size_cm / 100)) * scale)}" cy="${offsetY + ((roomState.cut_size_cm / 100) * scale)}" r="12" fill="#f2b705"></circle>
          `;
        }

        planSvg.innerHTML = `
          <rect x="0" y="0" width="900" height="620" rx="18" fill="#eef2f6"></rect>
          <polygon points="${planPoints}" fill="#d7be9d" stroke="#1d1f22" stroke-width="10"></polygon>
          <line x1="${offsetX + (bounds.minX * scale)}" y1="${offsetY + ((bounds.maxZ + 0.45) * scale)}" x2="${offsetX + (bounds.maxX * scale)}" y2="${offsetY + ((bounds.maxZ + 0.45) * scale)}" stroke="#9aa3af" stroke-width="3"></line>
          <text x="${offsetX}" y="${offsetY + ((bounds.maxZ + 0.45) * scale) - 10}" text-anchor="middle" font-size="24" fill="#4c5664">${roomState.width_cm.toFixed(0)} cm</text>
          <line x1="${offsetX + ((bounds.maxX + 0.45) * scale)}" y1="${offsetY + (bounds.minZ * scale)}" x2="${offsetX + ((bounds.maxX + 0.45) * scale)}" y2="${offsetY + (bounds.maxZ * scale)}" stroke="#9aa3af" stroke-width="3"></line>
          <text x="${offsetX + ((bounds.maxX + 0.45) * scale) + 18}" y="${offsetY + (((bounds.minZ + bounds.maxZ) * 0.5) * scale)}" font-size="24" fill="#4c5664" transform="rotate(90 ${offsetX + ((bounds.maxX + 0.45) * scale) + 18} ${offsetY + (((bounds.minZ + bounds.maxZ) * 0.5) * scale)})">${roomState.depth_cm.toFixed(0)} cm</text>
          ${extraLabels}
          <circle id="widthHandle" cx="${rightHandleX}" cy="${rightHandleY}" r="14" fill="#111"></circle>
          <circle id="depthHandle" cx="${bottomHandleX}" cy="${bottomHandleY}" r="14" fill="#111"></circle>
          ${extraHandles}
        `;

        const widthHandle = document.getElementById('widthHandle');
        const depthHandle = document.getElementById('depthHandle');
        const cutWidthHandle = document.getElementById('cutWidthHandle');
        const cutDepthHandle = document.getElementById('cutDepthHandle');
        const stemWidthHandle = document.getElementById('stemWidthHandle');
        const stemDepthHandle = document.getElementById('stemDepthHandle');
        const cutCornerHandle = document.getElementById('cutCornerHandle');
        function dragHandle(mode, startEvent) {
          startEvent.preventDefault();
          const startX = startEvent.clientX;
          const startY = startEvent.clientY;
          const startWidth = roomState.width_cm;
          const startDepth = roomState.depth_cm;
          const startCutWidth = roomState.cut_width_cm;
          const startCutDepth = roomState.cut_depth_cm;
          const startStemWidth = roomState.stem_width_cm;
          const startStemDepth = roomState.stem_depth_cm;
          const startCutSize = roomState.cut_size_cm;

          function onMove(moveEvent) {
            if (mode === 'width') {
              const deltaCm = (moveEvent.clientX - startX) * 4;
              roomState.width_cm = clamp(startWidth + deltaCm, 200, 3000);
            } else if (mode === 'depth') {
              const deltaCm = (moveEvent.clientY - startY) * 4;
              roomState.depth_cm = clamp(startDepth + deltaCm, 200, 3000);
            } else if (mode === 'cutWidth') {
              const deltaCm = (startX - moveEvent.clientX) * 4;
              roomState.cut_width_cm = clamp(startCutWidth + deltaCm, 80, Math.max(90, roomState.width_cm - 80));
            } else if (mode === 'cutDepth') {
              const deltaCm = (startY - moveEvent.clientY) * 4;
              roomState.cut_depth_cm = clamp(startCutDepth + deltaCm, 80, Math.max(90, roomState.depth_cm - 80));
            } else if (mode === 'stemWidth') {
              const deltaCm = (moveEvent.clientX - startX) * 4;
              roomState.stem_width_cm = clamp(startStemWidth + (deltaCm * 2), 80, Math.max(90, roomState.width_cm - 80));
            } else if (mode === 'stemDepth') {
              const deltaCm = (moveEvent.clientY - startY) * 4;
              roomState.stem_depth_cm = clamp(startStemDepth + deltaCm, 80, Math.max(90, roomState.depth_cm - 80));
            } else if (mode === 'cutCorner') {
              const deltaCm = ((moveEvent.clientX - startX) - (moveEvent.clientY - startY)) * 2;
              roomState.cut_size_cm = clamp(startCutSize + deltaCm, 60, Math.max(70, Math.min(roomState.width_cm, roomState.depth_cm) - 60));
            }
            refreshSummary();
            updateRoomMetrics();
            rebuildRoomGeometry();
            reflowProductsToRoom();
            setCameraPreset('top');
            renderPlanEditor();
          }

          function onUp() {
            window.removeEventListener('pointermove', onMove);
            window.removeEventListener('pointerup', onUp);
          }

          window.addEventListener('pointermove', onMove);
          window.addEventListener('pointerup', onUp);
        }

        if (widthHandle) {
          widthHandle.addEventListener('pointerdown', (event) => dragHandle('width', event));
        }
        if (depthHandle) {
          depthHandle.addEventListener('pointerdown', (event) => dragHandle('depth', event));
        }
        if (cutWidthHandle) {
          cutWidthHandle.addEventListener('pointerdown', (event) => dragHandle('cutWidth', event));
        }
        if (cutDepthHandle) {
          cutDepthHandle.addEventListener('pointerdown', (event) => dragHandle('cutDepth', event));
        }
        if (stemWidthHandle) {
          stemWidthHandle.addEventListener('pointerdown', (event) => dragHandle('stemWidth', event));
        }
        if (stemDepthHandle) {
          stemDepthHandle.addEventListener('pointerdown', (event) => dragHandle('stemDepth', event));
        }
        if (cutCornerHandle) {
          cutCornerHandle.addEventListener('pointerdown', (event) => dragHandle('cutCorner', event));
        }
      }

      renderPlanEditor();

      function applyPreset(presetName) {
        const preset = ROOM_PRESETS[presetName] || ROOM_PRESETS.Standard;
        roomState.sizePreset = presetName;
        roomState.width_cm = preset.width_cm;
        roomState.depth_cm = preset.depth_cm;
        roomState.height_cm = preset.height_cm;
        roomState.cut_width_cm = Math.round(preset.width_cm * 0.5);
        roomState.cut_depth_cm = Math.round(preset.depth_cm * 0.5);
        roomState.stem_width_cm = Math.round(preset.width_cm * 0.4);
        roomState.stem_depth_cm = Math.round(preset.depth_cm * 0.35);
        roomState.cut_size_cm = Math.round(Math.min(preset.width_cm, preset.depth_cm) * 0.24);
        refreshSummary();
        updateRoomMetrics();
        rebuildRoomGeometry();
        reflowProductsToRoom();
        setCameraPreset('top');
        renderPlanEditor();
      }

      function nudgeSelectedY(delta) {
        if (!selected) return;
        setAllWallsVisible();
        const currentTarget = selected.userData.targetPos || selected.position;
        selected.userData.targetPos = clampPositionToRoom(selected, currentTarget.x, currentTarget.y + delta, currentTarget.z);
        status.textContent = `Status: moving ${selected.userData.name} | y=${selected.userData.targetPos.y.toFixed(2)}`;
      }

      function createProceduralObject(spec, name) {
        const data = spec.modelData || {};
        const parts = Array.isArray(data.parts) ? data.parts : [];
        if (!parts.length) return null;
        const group = new THREE.Group();
        parts.forEach((part) => {
          const kind = String(part.kind || 'box');
          const color = new THREE.Color(String(part.color || '#b8b8b8'));
          let mesh = null;
          if (kind === 'cylinder') {
            const radius = Math.max(0.006, Number(part.radius || 0.02) * spec.width);
            const height = Math.max(0.02, Number(part.height || 0.6) * spec.height);
            mesh = new THREE.Mesh(
              new THREE.CylinderGeometry(radius, radius, height, 16),
              new THREE.MeshStandardMaterial({ color, roughness: 0.5, metalness: 0.2 })
            );
          } else {
            const size = Array.isArray(part.size) ? part.size : [0.3, 0.3, 0.3];
            mesh = new THREE.Mesh(
              new THREE.BoxGeometry(
                Math.max(0.01, Number(size[0] || 0.3) * spec.width),
                Math.max(0.01, Number(size[1] || 0.3) * spec.height),
                Math.max(0.01, Number(size[2] || 0.3) * spec.depth)
              ),
              new THREE.MeshStandardMaterial({ color, roughness: 0.42, metalness: 0.22 })
            );
          }
          const center = Array.isArray(part.center) ? part.center : [0, 0.5, 0];
          mesh.position.set(
            Number(center[0] || 0) * spec.width,
            Number(center[1] || 0.5) * spec.height,
            Number(center[2] || 0) * spec.depth
          );
          const rotationDeg = Array.isArray(part.rotation_deg) ? part.rotation_deg : [0, 0, 0];
          mesh.rotation.set(
            THREE.MathUtils.degToRad(Number(rotationDeg[0] || 0)),
            THREE.MathUtils.degToRad(Number(rotationDeg[1] || 0)),
            THREE.MathUtils.degToRad(Number(rotationDeg[2] || 0))
          );
          group.add(mesh);
        });
        status.textContent = `Status: added procedural 3D model for ${name}`;
        return group;
      }

      function createSliceVolumeObject(spec, name) {
        const source = spec.textureDataUrl;
        const data = spec.modelData || {};
        const positionsSrc = Array.isArray(data.positions) ? data.positions : [];
        const uvsSrc = Array.isArray(data.uvs) ? data.uvs : [];
        const frontIndicesSrc = Array.isArray(data.front_indices) ? data.front_indices.slice() : [];
        const alphaCutoff = Math.max(0.005, Math.min(0.2, Number(data.alpha_cutoff || 0.04)));
        if (positionsSrc.length < 9 || frontIndicesSrc.length < 3) {
          return null;
        }
        const positions = new Float32Array(positionsSrc.length);
        for (let index = 0; index < positionsSrc.length; index += 3) {
          positions[index] = Number(positionsSrc[index] || 0) * spec.width;
          positions[index + 1] = Number(positionsSrc[index + 1] || 0) * spec.height + (spec.height / 2);
          positions[index + 2] = Number(positionsSrc[index + 2] || 0) * spec.depth;
        }
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        if (uvsSrc.length === (positionsSrc.length / 3) * 2) {
          geometry.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(uvsSrc.map((value) => Number(value || 0))), 2));
        }
        for (let index = 0; index < frontIndicesSrc.length; index += 1) {
          frontIndicesSrc[index] = Number(frontIndicesSrc[index] || 0);
        }
        geometry.setIndex(frontIndicesSrc);
        geometry.computeVertexNormals();
        const material = new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide, alphaTest: alphaCutoff, depthTest: true, depthWrite: true });
        if (source && (source.startsWith('http') || source.startsWith('data:image'))) {
          new THREE.TextureLoader().load(source, (tex) => {
            tex.encoding = THREE.sRGBEncoding;
            tex.minFilter = THREE.LinearFilter;
            tex.magFilter = THREE.LinearFilter;
            material.map = tex;
            material.needsUpdate = true;
          });
        }
        status.textContent = `Status: added AI depth 3D mesh for ${name}`;
        return new THREE.Mesh(geometry, material);
      }

      function createFallbackBoxObject(spec) {
        const geom = new THREE.BoxGeometry(spec.width, spec.height, spec.depth);
        const frontBackMat = new THREE.MeshStandardMaterial({ color: 0xf0b45b, roughness: 0.8, metalness: 0.05 });
        const sideMat = new THREE.MeshStandardMaterial({ color: 0xc99347, roughness: 0.85, metalness: 0.04 });
        const materials = [sideMat, sideMat, sideMat, sideMat, frontBackMat, frontBackMat];
        const source = spec.textureDataUrl;
        if (source && (source.startsWith('http') || source.startsWith('data:image'))) {
          new THREE.TextureLoader().load(source, (tex) => {
            tex.encoding = THREE.sRGBEncoding;
            frontBackMat.map = tex;
            frontBackMat.needsUpdate = true;
          });
        }
        return new THREE.Mesh(geom, materials);
      }

      function createTriposrMeshObject(spec, name) {
        const group = new THREE.Group();
        const placeholder = createFallbackBoxObject(spec);
        group.add(placeholder);
        if (!spec.meshPath && !spec.meshObjText) {
          status.textContent = `Status: missing mesh path for ${name}, using fallback box`;
          return group;
        }
        const loader = new THREE.OBJLoader();
        function convertGeometryVertexColorsToLinear(geometry) {
          if (!geometry || !geometry.attributes || !geometry.attributes.color) {
            return;
          }
          const colorAttr = geometry.attributes.color;
          const color = new THREE.Color();
          for (let index = 0; index < colorAttr.count; index += 1) {
            color.fromBufferAttribute(colorAttr, index);
            color.convertSRGBToLinear();
            colorAttr.setXYZ(index, color.r, color.g, color.b);
          }
          colorAttr.needsUpdate = true;
        }
        function applyLoadedObject(obj) {
          obj.rotation.z = Math.PI / 2;
          obj.rotation.x = -Math.PI / 2;
          obj.traverse((child) => {
            if (!child.isMesh) {
              return;
            }
            child.geometry.computeVertexNormals();
            convertGeometryVertexColorsToLinear(child.geometry);
            child.material = new THREE.MeshBasicMaterial({ color: 0xffffff, vertexColors: true, side: THREE.DoubleSide });
          });
          const sourceBox = new THREE.Box3().setFromObject(obj);
          const sourceSize = new THREE.Vector3();
          sourceBox.getSize(sourceSize);
          const sourceHorizontal = Math.max(sourceSize.x, sourceSize.z, 0.001);
          obj.scale.setScalar(Math.max(spec.width, spec.depth, 0.001) / sourceHorizontal);
          const scaledBox = new THREE.Box3().setFromObject(obj);
          const center = new THREE.Vector3();
          scaledBox.getCenter(center);
          obj.position.sub(center);
          const alignedBox = new THREE.Box3().setFromObject(obj);
          obj.position.y += -alignedBox.min.y;
          group.clear();
          group.add(obj);
          snapMeshToFloor(group);
          status.textContent = `Status: added TriPoSR mesh for ${name}`;
        }
        try {
          if (spec.meshObjText) {
            applyLoadedObject(loader.parse(spec.meshObjText));
            return group;
          }
          loader.load(spec.meshPath, (obj) => applyLoadedObject(obj));
        } catch (_error) {
          status.textContent = `Status: error loading TriPoSR mesh for ${name}, using fallback box`;
        }
        return group;
      }

      function createVoxelObject(spec) {
        const data = spec.modelData || {};
        const points = Array.isArray(data.points) ? data.points : [];
        const depthSteps = Math.max(2, Number(data.depth_steps || 8));
        const voxelWNorm = Number(data.voxel_w_norm || 0.02);
        const voxelHNorm = Number(data.voxel_h_norm || 0.02);
        if (!points.length) {
          return null;
        }
        const voxelW = Math.max(0.004, spec.width * voxelWNorm);
        const voxelH = Math.max(0.004, spec.height * voxelHNorm);
        const voxelD = Math.max(0.004, spec.depth / depthSteps);
        const mesh = new THREE.InstancedMesh(
          new THREE.BoxGeometry(voxelW, voxelH, voxelD),
          new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.84, metalness: 0.05 }),
          points.length * depthSteps
        );
        const matrix = new THREE.Matrix4();
        const position = new THREE.Vector3();
        const quat = new THREE.Quaternion();
        const scale = new THREE.Vector3(1, 1, 1);
        const color = new THREE.Color();
        let instanceIndex = 0;
        for (let pointIdx = 0; pointIdx < points.length; pointIdx += 1) {
          const point = points[pointIdx];
          const x = Number(point[0] || 0);
          const y = Number(point[1] || 0);
          const r = Number(point[2] || 160);
          const g = Number(point[3] || 140);
          const b = Number(point[4] || 120);
          const worldX = x * spec.width;
          const worldY = (y * spec.height) + (spec.height / 2);
          color.setRGB(r / 255, g / 255, b / 255);
          for (let depthIndex = 0; depthIndex < depthSteps; depthIndex += 1) {
            const dz = ((depthIndex / (depthSteps - 1)) - 0.5) * spec.depth;
            position.set(worldX, worldY, dz);
            matrix.compose(position, quat, scale);
            mesh.setMatrixAt(instanceIndex, matrix);
            mesh.setColorAt(instanceIndex, color);
            instanceIndex += 1;
          }
        }
        mesh.instanceMatrix.needsUpdate = true;
        if (mesh.instanceColor) {
          mesh.instanceColor.needsUpdate = true;
        }
        return mesh;
      }

      function addProduct(name) {
        const spec = getSpec(name);
        let mesh = null;
        if (spec.modelType === 'procedural_piano_v1') {
          mesh = createProceduralObject(spec, name);
        } else if (spec.modelType === 'triposr_cpu_v1') {
          mesh = createTriposrMeshObject(spec, name);
        } else if (spec.modelType === 'depth_mesh_midas_v1' || spec.modelType === 'depth_mesh_midas_v2_closed' || spec.modelType === 'depth_mesh_cpu_v3') {
          mesh = createSliceVolumeObject(spec, name);
        } else if (spec.modelType === 'alpha_extrude_voxel') {
          mesh = createVoxelObject(spec);
        }
        if (!mesh) {
          mesh = createFallbackBoxObject(spec);
          status.textContent = `Status: fallback box model for ${name}`;
        }
        const placementOffset = productMeshes.length * 0.35;
        const startX = clamp(roomCenterX - 0.6 + placementOffset, bounds.minX + 0.35, bounds.maxX - 0.35);
        const startZ = clamp(roomCenterZ + Math.min(0.6, roomSpanZ * 0.12) + placementOffset, bounds.minZ + 0.35, bounds.maxZ - 0.35);
        mesh.position.set(startX, spec.height / 2, startZ);
        mesh.userData = {
          name,
          originalImageDataUrl: spec.originalImageDataUrl,
          priceLabel: spec.price || 'Price on request',
          priceValue: parsePriceValue(spec.price),
          widthM: spec.width,
          depthM: spec.depth,
          heightM: spec.height,
          sourceLabel: String(name).split('/').slice(0, 2).join(' / ').trim(),
        };
        scene.add(mesh);
        snapMeshToFloor(mesh);
        productMeshes.push(mesh);
        selected = mesh;
        mesh.userData.targetPos = mesh.position.clone();
        mesh.userData.targetYaw = mesh.rotation.y;
        mesh.userData.targetPitch = mesh.rotation.x;
        mesh.userData.targetRoll = mesh.rotation.z;
        fitMeshToRoom(mesh);
        updateSelectedPreview();
        updateOrderSummary();
      }

      renderer.domElement.addEventListener('pointerdown', (event) => {
        if (setupStep !== 'design') {
          return;
        }
        if (event.button !== 0) {
          return;
        }
        updatePointerFromEvent(event);
        raycaster.setFromCamera(pointer, camera);
        const intersects = raycaster.intersectObjects(productMeshes, true);
        if (intersects.length > 0) {
          let hit = intersects[0].object;
          while (hit && !productMeshes.includes(hit)) {
            hit = hit.parent;
          }
          selected = hit;
          if (!selected) {
            return;
          }
          if (interactionMode === 'move' || interactionMode === 'rotate') {
            setAllWallsVisible();
          }
          if (interactionMode === 'move') {
            if (!raycaster.ray.intersectPlane(floorPlane, dragPoint)) {
              return;
            }
            dragOffset.copy(selected.position).sub(dragPoint);
            dragStartLift = selected.position.y;
            selected.userData.targetPos = selected.position.clone();
          } else {
            dragStartYaw = Number(selected.userData.targetYaw ?? selected.rotation.y);
            dragStartPitch = Number(selected.userData.targetPitch ?? selected.rotation.x);
            dragStartRoll = Number(selected.userData.targetRoll ?? selected.rotation.z);
          }
          dragStartX = event.clientX;
          dragStartY = event.clientY;
          dragActive = true;
          dragPointerId = event.pointerId;
          renderer.domElement.setPointerCapture(event.pointerId);
          renderer.domElement.style.cursor = 'grabbing';
          updateSelectedPreview();
        } else {
          selected = null;
          updateSelectedPreview();
        }
      });

      renderer.domElement.addEventListener('pointermove', (event) => {
        if (setupStep !== 'design') {
          return;
        }
        if (!dragActive || !selected || dragPointerId !== event.pointerId) {
          return;
        }
        if (interactionMode === 'move') {
          updatePointerFromEvent(event);
          raycaster.setFromCamera(pointer, camera);
          const currentTarget = selected.userData.targetPos || selected.position;
          if (event.shiftKey) {
            const targetY = dragStartLift - ((event.clientY - dragStartY) * 0.01);
            selected.userData.targetPos = clampPositionToRoom(selected, currentTarget.x, targetY, currentTarget.z);
            status.textContent = `Status: moving ${selected.userData.name} | y=${selected.userData.targetPos.y.toFixed(2)}`;
            return;
          }
          if (!raycaster.ray.intersectPlane(floorPlane, dragPoint)) {
            return;
          }
          const targetX = dragPoint.x + dragOffset.x;
          const targetZ = dragPoint.z + dragOffset.z;
          selected.userData.targetPos = clampPositionToRoom(selected, targetX, currentTarget.y, targetZ);
          status.textContent = `Status: moving ${selected.userData.name} | x=${selected.userData.targetPos.x.toFixed(2)} z=${selected.userData.targetPos.z.toFixed(2)}`;
        } else {
          const deltaX = event.clientX - dragStartX;
          const deltaY = event.clientY - dragStartY;
          const speed = 0.012;
          selected.userData.targetPitch = dragStartPitch + (deltaY * speed);
          if (event.shiftKey) {
            selected.userData.targetYaw = dragStartYaw;
            selected.userData.targetRoll = dragStartRoll + (deltaX * speed);
          } else {
            selected.userData.targetYaw = dragStartYaw + (deltaX * speed);
            selected.userData.targetRoll = dragStartRoll;
          }
          status.textContent = `Status: rotating ${selected.userData.name} | x=${deg(Number(selected.userData.targetPitch)).toFixed(1)}° y=${deg(Number(selected.userData.targetYaw)).toFixed(1)}° z=${deg(Number(selected.userData.targetRoll)).toFixed(1)}°`;
        }
      });

      renderer.domElement.addEventListener('pointerup', (event) => {
        if (!dragActive || dragPointerId !== event.pointerId) {
          return;
        }
        dragActive = false;
        dragPointerId = null;
        renderer.domElement.style.cursor = 'default';
        try {
          renderer.domElement.releasePointerCapture(event.pointerId);
        } catch (_error) {
        }
      });

      renderer.domElement.addEventListener('pointercancel', () => {
        dragActive = false;
        dragPointerId = null;
        renderer.domElement.style.cursor = 'default';
      });

      renderer.domElement.addEventListener('wheel', (event) => {
        if (setupStep !== 'design') {
          return;
        }
        if (!selected) return;
        if (interactionMode === 'move' || interactionMode === 'rotate' || event.shiftKey) {
          setAllWallsVisible();
        }
        event.preventDefault();
        if (interactionMode === 'move' || event.shiftKey) {
          const liftedY = clamp(selected.position.y + (event.deltaY > 0 ? -0.02 : 0.02), getFloorSnappedY(selected), getCeilingClampedY(selected));
          const clampedTarget = clampPositionToRoom(selected, selected.position.x, liftedY, selected.position.z);
          selected.position.copy(clampedTarget);
          if (selected.userData.targetPos) {
            selected.userData.targetPos.copy(clampedTarget);
          }
          status.textContent = `Status: lift ${selected.userData.name} | y=${selected.position.y.toFixed(2)}`;
          return;
        }
        if (interactionMode === 'rotate') {
          selected.userData.targetYaw = Number(selected.userData.targetYaw ?? selected.rotation.y) + (event.deltaY > 0 ? 0.03 : -0.03);
        }
      }, { passive: false });

      addBtn.addEventListener('click', () => {
        if (!assetSelect.value) return;
        addProduct(assetSelect.value);
      });
      moveBtn.addEventListener('click', () => {
        setAllWallsVisible();
        interactionMode = 'move';
        status.textContent = 'Status: move mode. Drag to move on the floor. Hold Shift while dragging to change height.';
      });
      rotateBtn.addEventListener('click', () => {
        setAllWallsVisible();
        interactionMode = 'rotate';
        status.textContent = 'Status: rotate mode. Horizontal drag rotates around Y; vertical drag rotates around X. Hold Shift for Z.';
      });
      viewBtn.addEventListener('click', () => {
        orbitEnabled = !orbitEnabled;
        orbit.enabled = orbitEnabled;
        orbit.enableRotate = orbitEnabled;
        orbit.enablePan = orbitEnabled;
        orbit.enableZoom = orbitEnabled;
        viewBtn.textContent = orbitEnabled ? 'Lock Orbit' : 'Enable Orbit';
        status.textContent = orbitEnabled ? 'Status: orbit camera enabled' : 'Status: orbit camera locked';
      });
      frontViewBtn.addEventListener('click', () => { setCameraPreset('front'); status.textContent = 'Status: front room view'; });
      leftViewBtn.addEventListener('click', () => { setCameraPreset('left'); status.textContent = 'Status: left room view'; });
      rightViewBtn.addEventListener('click', () => { setCameraPreset('right'); status.textContent = 'Status: right room view'; });
      topViewBtn.addEventListener('click', () => { setCameraPreset('top'); });
      fitBtn.addEventListener('click', () => {
        if (!selected) return;
        setAllWallsVisible();
        fitMeshToRoom(selected);
        status.textContent = `Status: fit ${selected.userData.name} inside room bounds`;
      });
      alignWallBtn.addEventListener('click', () => {
        if (!selected) return;
        setAllWallsVisible();
        const quarter = Math.PI / 2;
        selected.userData.targetYaw = Math.round((selected.userData.targetYaw ?? selected.rotation.y) / quarter) * quarter;
        status.textContent = `Status: snapped ${selected.userData.name} parallel to wall`;
      });
      deleteBtn.addEventListener('click', () => {
        if (!selected) return;
        scene.remove(selected);
        const idx = productMeshes.indexOf(selected);
        if (idx >= 0) productMeshes.splice(idx, 1);
        selected = null;
        updateSelectedPreview();
        updateOrderSummary();
        status.textContent = 'Status: deleted selected object';
      });
      roomPresetSelect.addEventListener('change', () => {
        applyPreset(roomPresetSelect.value);
        setSetupStep('size');
      });
      roomShapeSelect.addEventListener('change', () => {
        roomState.shape = roomShapeSelect.value;
        refreshSummary();
        updateRoomMetrics();
        rebuildRoomGeometry();
        reflowProductsToRoom();
        setCameraPreset('top');
        renderPlanEditor();
        setSetupStep('size');
      });
      editSizeBtn.addEventListener('click', () => {
        if (setupStep === 'shape') {
          setSetupStep('size');
        } else {
          setSetupStep('shape');
        }
      });
      startDesignBtn.addEventListener('click', () => {
        setSetupStep('design');
      });

      function animate() {
        requestAnimationFrame(animate);
        for (let index = 0; index < productMeshes.length; index += 1) {
          const mesh = productMeshes[index];
          const targetPos = mesh.userData.targetPos;
          if (targetPos && mesh.position.distanceTo(targetPos) > 0.0005) {
            mesh.position.lerp(targetPos, 0.22);
          }
          const targetYaw = mesh.userData.targetYaw;
          if (typeof targetYaw === 'number') {
            mesh.rotation.y += (targetYaw - mesh.rotation.y) * 0.22;
          }
          const targetPitch = mesh.userData.targetPitch;
          if (typeof targetPitch === 'number') {
            mesh.rotation.x += (targetPitch - mesh.rotation.x) * 0.22;
          }
          const targetRoll = mesh.userData.targetRoll;
          if (typeof targetRoll === 'number') {
            mesh.rotation.z += (targetRoll - mesh.rotation.z) * 0.22;
          }
        }
        orbit.update();
        renderer.render(scene, camera);
      }
      animate();

      window.addEventListener('resize', () => {
        const width = viewport.clientWidth;
        const height = viewport.clientHeight;
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
      });

      setSetupStep('shape');
      updateOrderSummary();
    })();
  </script>
</body>
</html>
"""
    return html.replace("__PAYLOAD_JSON__", payload_json)


def build_3d_html(payload_json: str) -> str:
    """Render a Three.js powered 3D room with mouse move/rotate controls."""
    html = """
<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <style>
    body { margin: 0; font-family: sans-serif; font-size: 17px; color: #1f2933; }
    .wrap {
      display: flex;
      gap: 12px;
      align-items: stretch;
      height: 760px;
    }
    .panel {
      width: 280px;
      border: 1px solid #ddd;
      border-radius: 10px;
      padding: 10px;
      box-sizing: border-box;
      background: #fafafa;
      height: 100%;
      overflow-y: auto;
    }
    .panel label { display: block; margin: 8px 0 4px; font-size: 16px; font-weight: 700; color: #1f2933; }
    .panel select, .panel button {
      width: 100%;
      height: 34px;
      margin-bottom: 8px;
      box-sizing: border-box;
      font-size: 16px;
    }
    #viewport {
      flex: 1;
      height: 100%;
      border: 1px solid #bbb;
      border-radius: 10px;
      overflow: hidden;
      background: #f4f6f8;
    }
    .preview-box {
      border: 1px solid #ddd;
      border-radius: 8px;
      background: #fff;
      padding: 6px;
      margin: 8px 0;
    }
    .preview-box img {
      width: 100%;
      display: block;
      border-radius: 6px;
      object-fit: contain;
      background: #f7f7f7;
      min-height: 120px;
      max-height: 180px;
    }
    .note { font-size: 15px; color: #2f3b4a; margin-top: 6px; }
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"panel\">
      <h4 style=\"margin:4px 0 8px 0;\">3D Product Placement</h4>
      <label>Product</label>
      <select id=\"assetSelect\"></select>
      <button id=\"addBtn\">Add Product</button>
      <button id=\"moveBtn\">Move Mode</button>
      <button id=\"rotateBtn\">Rotate Mode</button>
      <button id=\"viewBtn\">Inspect View</button>
      <div style=\"display:grid;grid-template-columns:1fr 1fr;gap:6px;\">
        <button id=\"frontViewBtn\">Front View</button>
        <button id=\"leftViewBtn\">Left View</button>
        <button id=\"rightViewBtn\">Right View</button>
        <button id=\"backViewBtn\">Back View</button>
      </div>
      <button id=\"fitBtn\">Fit In Room</button>
      <button id=\"alignWallBtn\">Snap Parallel To Wall</button>
      <button id=\"deleteBtn\">Delete Selected</button>
      <label>Original 2D Photo</label>
      <div class=\"preview-box\"><img id=\"originalPreview\" alt=\"Original product photo\" /></div>
      <div class=\"note\">Move mode: drag sideways for X/Z or mostly up/down for Y. Mouse wheel also adjusts Y. Rotate mode: horizontal drag = Y, vertical drag = X, Shift = Z. Inspect View enables a 3D room shell and free camera. Front/Left/Right/Back buttons jump to room angles while preserving furniture placement.</div>
      <div id=\"status\" class=\"note\">Status: ready</div>
    </div>
    <div id=\"viewport\"></div>
  </div>

  <script src=\"https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js\"></script>
  <script src=\"https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js\"></script>
  <script src=\"https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/TransformControls.js\"></script>
  <script src=\"https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/OBJLoader.js\"></script>
  <script>
    (() => {
      const DATA = __PAYLOAD_JSON__;
      const viewport = document.getElementById('viewport');
      const assetSelect = document.getElementById('assetSelect');
      const addBtn = document.getElementById('addBtn');
      const moveBtn = document.getElementById('moveBtn');
      const rotateBtn = document.getElementById('rotateBtn');
      const viewBtn = document.getElementById('viewBtn');
      const frontViewBtn = document.getElementById('frontViewBtn');
      const leftViewBtn = document.getElementById('leftViewBtn');
      const rightViewBtn = document.getElementById('rightViewBtn');
      const backViewBtn = document.getElementById('backViewBtn');
      const fitBtn = document.getElementById('fitBtn');
      const alignWallBtn = document.getElementById('alignWallBtn');
      const deleteBtn = document.getElementById('deleteBtn');
      const originalPreview = document.getElementById('originalPreview');
      const status = document.getElementById('status');

      const modelNames = Object.keys(DATA.models || {});
      modelNames.forEach((name) => {
        const option = document.createElement('option');
        option.value = name;
        option.textContent = name;
        assetSelect.appendChild(option);
      });

      const scene = new THREE.Scene();
      const roomTexture = new THREE.TextureLoader().load(DATA.room);
      const inspectBackground = new THREE.Color(0xf2f4f7);
      scene.background = roomTexture;

      const roomDimensionsCm = DATA.room_dimensions_cm || {};
      const ROOM_WIDTH_M = Math.max(1.5, Number(roomDimensionsCm.width || 680) / 100);
      const ROOM_DEPTH_M = Math.max(1.5, Number(roomDimensionsCm.depth || 240) / 100);
      const ROOM_HEIGHT_M = Math.max(1.8, Number(roomDimensionsCm.height || 245) / 100);
      const roomScaleFactor = Math.max(ROOM_WIDTH_M / 6.8, ROOM_DEPTH_M / 2.4, ROOM_HEIGHT_M / 2.45);

      const camera = new THREE.PerspectiveCamera(34, viewport.clientWidth / viewport.clientHeight, 0.1, 200);
      camera.position.set(0, 2.7 * roomScaleFactor, 8.2 * roomScaleFactor);

      const renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setPixelRatio(window.devicePixelRatio);
      renderer.setSize(viewport.clientWidth, viewport.clientHeight);
      renderer.outputEncoding = THREE.sRGBEncoding;
      viewport.appendChild(renderer.domElement);

      const orbit = new THREE.OrbitControls(camera, renderer.domElement);
      orbit.enableDamping = true;
      orbit.dampingFactor = 0.08;
      orbit.rotateSpeed = 0.85;
      orbit.zoomSpeed = 0.95;
      orbit.panSpeed = 0.8;
      orbit.screenSpacePanning = true;
      orbit.minDistance = 1.2;
      orbit.maxDistance = 20 * roomScaleFactor;
      orbit.target.set(0, 0.9 * roomScaleFactor, 0);
      orbit.enableRotate = false;
      orbit.enablePan = false;
      orbit.enableZoom = false;
      orbit.enabled = false;
      orbit.update();

      const ambient = new THREE.AmbientLight(0xffffff, 0.62);
      const key = new THREE.DirectionalLight(0xfff6e9, 1.0);
      key.position.set(4, 7, 5);
      const rim = new THREE.DirectionalLight(0xd6f2ff, 0.36);
      rim.position.set(-5, 4, -3);
      scene.add(ambient, key, rim);

      // Keep only the real room photo as background; no extra synthetic wall/floor meshes.

      const transform = new THREE.TransformControls(camera, renderer.domElement);
      transform.visible = false;
      scene.add(transform);

      let selected = null;
      let interactionMode = 'move';
      let dragActive = false;
      let dragPointerId = null;
      let dragStartX = 0;
      let dragStartY = 0;
      let dragStartLift = 0;
      let dragStartYaw = 0;
      let dragStartPitch = 0;
      let dragStartRoll = 0;
      let dragMoveAxis = 'xz';
      const productMeshes = [];
      const raycaster = new THREE.Raycaster();
      const pointer = new THREE.Vector2();
      const floorPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
      const dragPoint = new THREE.Vector3();
      const dragOffset = new THREE.Vector3();
      const ROOM_MIN_X = -ROOM_WIDTH_M / 2;
      const ROOM_MAX_X = ROOM_WIDTH_M / 2;
      const ROOM_MIN_Y = 0;
      const ROOM_MAX_Y = ROOM_HEIGHT_M;
      const ROOM_MIN_Z = 0;
      const ROOM_MAX_Z = ROOM_DEPTH_M;
      const DRAG_AXIS_LOCK_THRESHOLD = 10;
      const ROOM_CENTER_X = (ROOM_MIN_X + ROOM_MAX_X) / 2;
      const ROOM_CENTER_Z = (ROOM_MIN_Z + ROOM_MAX_Z) / 2;
      const ROOM_SIZE_X = ROOM_MAX_X - ROOM_MIN_X;
      const ROOM_SIZE_Y = ROOM_MAX_Y - ROOM_MIN_Y;
      const ROOM_SIZE_Z = ROOM_MAX_Z - ROOM_MIN_Z;

      const roomShell = new THREE.Mesh(
        new THREE.BoxGeometry(ROOM_SIZE_X, ROOM_SIZE_Y, ROOM_SIZE_Z),
        new THREE.MeshStandardMaterial({
          color: 0xe6eaef,
          side: THREE.BackSide,
          transparent: true,
          opacity: 0.04,
          roughness: 0.95,
          metalness: 0.0,
        })
      );
      roomShell.position.set(ROOM_CENTER_X, (ROOM_MIN_Y + ROOM_MAX_Y) / 2, ROOM_CENTER_Z);
      roomShell.visible = false;
      scene.add(roomShell);

      const roomPhotoGroup = new THREE.Group();
      roomPhotoGroup.visible = false;
      scene.add(roomPhotoGroup);

      function imageUv(nx, ny) {
        return [nx, 1 - ny];
      }

      function createMappedQuad(points3d, uvPoints, material) {
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array([
          points3d[0][0], points3d[0][1], points3d[0][2],
          points3d[1][0], points3d[1][1], points3d[1][2],
          points3d[2][0], points3d[2][1], points3d[2][2],
          points3d[3][0], points3d[3][1], points3d[3][2],
        ]);
        const uvs = new Float32Array([
          uvPoints[0][0], uvPoints[0][1],
          uvPoints[1][0], uvPoints[1][1],
          uvPoints[2][0], uvPoints[2][1],
          uvPoints[3][0], uvPoints[3][1],
        ]);
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
        geometry.setIndex([0, 2, 1, 2, 3, 1]);
        geometry.computeVertexNormals();
        return new THREE.Mesh(geometry, material);
      }

      function buildPhotoRoomShell() {
        while (roomPhotoGroup.children.length) {
          roomPhotoGroup.remove(roomPhotoGroup.children[0]);
        }

        const inferredLayout = DATA.room_layout || {};
        function anchorPoint(key, fallbackX, fallbackY) {
          const value = inferredLayout[key];
          if (!Array.isArray(value) || value.length < 2) {
            return imageUv(fallbackX, fallbackY);
          }
          const x = Math.max(0, Math.min(1, Number(value[0])));
          const y = Math.max(0, Math.min(1, Number(value[1])));
          return imageUv(x, y);
        }

        const roomPhotoAnchors = {
          backTopLeft: anchorPoint('back_top_left', 0.319, 0.314),
          backTopRight: anchorPoint('back_top_right', 0.691, 0.314),
          backBottomLeft: anchorPoint('back_bottom_left', 0.319, 0.707),
          backBottomRight: anchorPoint('back_bottom_right', 0.691, 0.707),
          leftFrontTop: anchorPoint('left_front_top', 0.005, 0.075),
          leftFrontBottom: anchorPoint('left_front_bottom', 0.002, 0.998),
          rightFrontTop: anchorPoint('right_front_top', 0.995, 0.074),
          rightFrontBottom: anchorPoint('right_front_bottom', 0.998, 0.998),
          floorFrontLeft: anchorPoint('floor_front_left', 0.01, 0.998),
          floorFrontRight: anchorPoint('floor_front_right', 0.99, 0.998),
          ceilingFrontLeft: anchorPoint('ceiling_front_left', 0.005, 0.01),
          ceilingFrontRight: anchorPoint('ceiling_front_right', 0.995, 0.01),
        };

        const surfaceMaterial = new THREE.MeshBasicMaterial({
          map: roomTexture,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.98,
        });

        const backWall = createMappedQuad(
          [
            [ROOM_MIN_X, ROOM_MAX_Y, ROOM_MIN_Z],
            [ROOM_MAX_X, ROOM_MAX_Y, ROOM_MIN_Z],
            [ROOM_MIN_X, ROOM_MIN_Y, ROOM_MIN_Z],
            [ROOM_MAX_X, ROOM_MIN_Y, ROOM_MIN_Z],
          ],
          [
            roomPhotoAnchors.backTopLeft,
            roomPhotoAnchors.backTopRight,
            roomPhotoAnchors.backBottomLeft,
            roomPhotoAnchors.backBottomRight,
          ],
          surfaceMaterial.clone()
        );
        roomPhotoGroup.add(backWall);

        const floor = createMappedQuad(
          [
            [ROOM_MIN_X, ROOM_MIN_Y, ROOM_MIN_Z],
            [ROOM_MAX_X, ROOM_MIN_Y, ROOM_MIN_Z],
            [ROOM_MIN_X, ROOM_MIN_Y, ROOM_MAX_Z],
            [ROOM_MAX_X, ROOM_MIN_Y, ROOM_MAX_Z],
          ],
          [
            roomPhotoAnchors.backBottomLeft,
            roomPhotoAnchors.backBottomRight,
            roomPhotoAnchors.floorFrontLeft,
            roomPhotoAnchors.floorFrontRight,
          ],
          surfaceMaterial.clone()
        );
        roomPhotoGroup.add(floor);

        const leftWall = createMappedQuad(
          [
            [ROOM_MIN_X, ROOM_MAX_Y, ROOM_MAX_Z],
            [ROOM_MIN_X, ROOM_MAX_Y, ROOM_MIN_Z],
            [ROOM_MIN_X, ROOM_MIN_Y, ROOM_MAX_Z],
            [ROOM_MIN_X, ROOM_MIN_Y, ROOM_MIN_Z],
          ],
          [
            roomPhotoAnchors.leftFrontTop,
            roomPhotoAnchors.backTopLeft,
            roomPhotoAnchors.leftFrontBottom,
            roomPhotoAnchors.backBottomLeft,
          ],
          surfaceMaterial.clone()
        );
        roomPhotoGroup.add(leftWall);

        const rightWall = createMappedQuad(
          [
            [ROOM_MAX_X, ROOM_MAX_Y, ROOM_MIN_Z],
            [ROOM_MAX_X, ROOM_MAX_Y, ROOM_MAX_Z],
            [ROOM_MAX_X, ROOM_MIN_Y, ROOM_MIN_Z],
            [ROOM_MAX_X, ROOM_MIN_Y, ROOM_MAX_Z],
          ],
          [
            roomPhotoAnchors.backTopRight,
            roomPhotoAnchors.rightFrontTop,
            roomPhotoAnchors.backBottomRight,
            roomPhotoAnchors.rightFrontBottom,
          ],
          surfaceMaterial.clone()
        );
        roomPhotoGroup.add(rightWall);

        const ceiling = createMappedQuad(
          [
            [ROOM_MIN_X, ROOM_MAX_Y, ROOM_MAX_Z],
            [ROOM_MAX_X, ROOM_MAX_Y, ROOM_MAX_Z],
            [ROOM_MIN_X, ROOM_MAX_Y, ROOM_MIN_Z],
            [ROOM_MAX_X, ROOM_MAX_Y, ROOM_MIN_Z],
          ],
          [
            roomPhotoAnchors.ceilingFrontLeft,
            roomPhotoAnchors.ceilingFrontRight,
            roomPhotoAnchors.backTopLeft,
            roomPhotoAnchors.backTopRight,
          ],
          surfaceMaterial.clone()
        );
        roomPhotoGroup.add(ceiling);
      }

      if (roomTexture.image && roomTexture.image.complete) {
        buildPhotoRoomShell();
      } else {
        roomTexture.onUpdate = () => {
          buildPhotoRoomShell();
        };
      }

      const inspectGrid = new THREE.GridHelper(ROOM_SIZE_X, 16, 0x8ea1b5, 0xc8d2dc);
      inspectGrid.position.set(ROOM_CENTER_X, ROOM_MIN_Y + 0.001, ROOM_CENTER_Z + 0.02);
      inspectGrid.visible = false;
      scene.add(inspectGrid);

      const roomHelper = new THREE.Box3Helper(
        new THREE.Box3(
          new THREE.Vector3(ROOM_MIN_X, ROOM_MIN_Y, ROOM_MIN_Z),
          new THREE.Vector3(ROOM_MAX_X, ROOM_MAX_Y, ROOM_MAX_Z)
        ),
        0x6aa6ff
      );
      roomHelper.visible = false;
      scene.add(roomHelper);
      let inspectViewEnabled = false;

      function clamp(v, lo, hi) {
        return Math.max(lo, Math.min(hi, v));
      }

      function deg(rad) {
        return rad * 57.2958;
      }

      function getMeshExtents(mesh) {
        const box = new THREE.Box3().setFromObject(mesh);
        return {
          minOffsetX: box.min.x - mesh.position.x,
          maxOffsetX: box.max.x - mesh.position.x,
          minOffsetY: box.min.y - mesh.position.y,
          maxOffsetY: box.max.y - mesh.position.y,
          minOffsetZ: box.min.z - mesh.position.z,
          maxOffsetZ: box.max.z - mesh.position.z,
        };
      }

      function getFloorSnappedY(mesh) {
        const extents = getMeshExtents(mesh);
        return ROOM_MIN_Y - extents.minOffsetY;
      }

      function getCeilingClampedY(mesh) {
        const extents = getMeshExtents(mesh);
        return ROOM_MAX_Y - extents.maxOffsetY;
      }

      function clampPositionToRoom(mesh, x, y, z) {
        const extents = getMeshExtents(mesh);
        return new THREE.Vector3(
          clamp(x, ROOM_MIN_X - extents.minOffsetX, ROOM_MAX_X - extents.maxOffsetX),
          clamp(y, ROOM_MIN_Y - extents.minOffsetY, ROOM_MAX_Y - extents.maxOffsetY),
          clamp(z, ROOM_MIN_Z - extents.minOffsetZ, ROOM_MAX_Z - extents.maxOffsetZ)
        );
      }

      function snapMeshToFloor(mesh) {
        mesh.position.y = getFloorSnappedY(mesh);
      }

      function fitMeshToRoom(mesh) {
        const box = new THREE.Box3().setFromObject(mesh);
        if (box.isEmpty()) {
          return;
        }

        const size = new THREE.Vector3();
        box.getSize(size);
        const scaleCandidates = [
          (ROOM_SIZE_X * 0.82) / Math.max(size.x, 0.001),
          (ROOM_SIZE_Y * 0.82) / Math.max(size.y, 0.001),
          (ROOM_SIZE_Z * 0.82) / Math.max(size.z, 0.001),
        ];
        const uniformScale = Math.min(...scaleCandidates, 1.0);

        if (uniformScale < 0.999) {
          mesh.scale.multiplyScalar(uniformScale);
        }

        const centeredTarget = clampPositionToRoom(mesh, ROOM_CENTER_X, getFloorSnappedY(mesh), ROOM_CENTER_Z + 0.15);
        mesh.position.copy(centeredTarget);
        mesh.userData.targetPos = centeredTarget.clone();
      }

      function setInspectView(enabled) {
        inspectViewEnabled = enabled;
        orbit.enabled = enabled;
        orbit.enableRotate = enabled;
        orbit.enablePan = enabled;
        orbit.enableZoom = enabled;
        roomHelper.visible = false;
        roomShell.visible = false;
        roomPhotoGroup.visible = enabled;
        inspectGrid.visible = false;
        scene.background = enabled ? inspectBackground : roomTexture;
        viewBtn.textContent = enabled ? 'Lock View' : 'Inspect View';
      }

      function setCameraPreset(preset) {
        const sceneBox = new THREE.Box3(
          new THREE.Vector3(ROOM_MIN_X, ROOM_MIN_Y, ROOM_MIN_Z),
          new THREE.Vector3(ROOM_MAX_X, ROOM_MAX_Y, ROOM_MAX_Z),
        );
        for (let i = 0; i < productMeshes.length; i += 1) {
          sceneBox.expandByObject(productMeshes[i]);
        }

        const focus = new THREE.Vector3();
        sceneBox.getCenter(focus);
        focus.y = ROOM_MIN_Y + (ROOM_SIZE_Y * 0.42);

        const sceneSize = new THREE.Vector3();
        sceneBox.getSize(sceneSize);
        const radius = Math.max(sceneSize.x, sceneSize.z) * 1.25 + 1.6;

        if (preset === 'front') {
          camera.position.set(focus.x, focus.y + (ROOM_SIZE_Y * 0.1), ROOM_MAX_Z + radius);
        } else if (preset === 'back') {
          camera.position.set(focus.x, focus.y + (ROOM_SIZE_Y * 0.1), ROOM_MIN_Z - radius);
        } else if (preset === 'left') {
          camera.position.set(ROOM_MIN_X - radius, focus.y + (ROOM_SIZE_Y * 0.1), focus.z);
        } else if (preset === 'right') {
          camera.position.set(ROOM_MAX_X + radius, focus.y + (ROOM_SIZE_Y * 0.1), focus.z);
        }

        orbit.target.copy(focus);
        orbit.update();
      }

      function updatePointerFromEvent(event) {
        const rect = renderer.domElement.getBoundingClientRect();
        pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      }

      function getSpec(name) {
        const spec = (DATA.models && DATA.models[name]) || {
          model_type: 'depth_mesh_midas_v1',
          width: 1.0,
          height: 0.8,
          depth: 0.7,
          texture_data_url: '',
          mesh_path: '',
          mesh_obj_text: '',
          model_data: {},
        };
        return {
          modelType: String(spec.model_type || 'depth_mesh_midas_v1'),
          width: Number(spec.width || 1.0),
          height: Number(spec.height || 0.8),
          depth: Number(spec.depth || 0.7),
          textureDataUrl: String(spec.texture_data_url || ''),
          originalImageDataUrl: String(spec.original_image_data_url || ''),
          meshPath: String(spec.mesh_path || ''),
          meshObjText: String(spec.mesh_obj_text || ''),
          modelData: spec.model_data || {},
        };
      }

      function updateSelectedPreview() {
        if (!selected) {
          originalPreview.removeAttribute('src');
          return;
        }
        const src = String(selected.userData.originalImageDataUrl || '');
        if (src) {
          originalPreview.src = src;
        } else {
          originalPreview.removeAttribute('src');
        }
      }

      function nudgeSelectedY(delta) {
        if (!selected) return;
        const minY = getFloorSnappedY(selected);
        const maxY = getCeilingClampedY(selected);
        const currentTarget = selected.userData.targetPos || selected.position;
        const targetY = clamp(currentTarget.y + delta, minY, maxY);
        selected.userData.targetPos = clampPositionToRoom(selected, currentTarget.x, targetY, currentTarget.z);
        status.textContent = `Status: moving ${selected.userData.name} | y=${selected.userData.targetPos.y.toFixed(2)}`;
      }

      function createProceduralObject(spec, name) {
        const data = spec.modelData || {};
        const parts = Array.isArray(data.parts) ? data.parts : [];
        if (!parts.length) return null;

        const group = new THREE.Group();

        parts.forEach((part) => {
          const kind = String(part.kind || 'box');
          const color = new THREE.Color(String(part.color || '#b8b8b8'));
          let mesh = null;

          if (kind === 'cylinder') {
            const radiusNorm = Number(part.radius || 0.02);
            const heightNorm = Number(part.height || 0.6);
            const radius = Math.max(0.006, radiusNorm * spec.width);
            const height = Math.max(0.02, heightNorm * spec.height);
            const geometry = new THREE.CylinderGeometry(radius, radius, height, 16);
            const material = new THREE.MeshStandardMaterial({ color, roughness: 0.5, metalness: 0.2 });
            mesh = new THREE.Mesh(geometry, material);
          } else {
            const size = Array.isArray(part.size) ? part.size : [0.3, 0.3, 0.3];
            const sx = Math.max(0.01, Number(size[0] || 0.3) * spec.width);
            const sy = Math.max(0.01, Number(size[1] || 0.3) * spec.height);
            const sz = Math.max(0.01, Number(size[2] || 0.3) * spec.depth);
            const geometry = new THREE.BoxGeometry(sx, sy, sz);
            const material = new THREE.MeshStandardMaterial({ color, roughness: 0.42, metalness: 0.22 });
            mesh = new THREE.Mesh(geometry, material);
          }

          const center = Array.isArray(part.center) ? part.center : [0, 0.5, 0];
          mesh.position.set(
            Number(center[0] || 0) * spec.width,
            Number(center[1] || 0.5) * spec.height,
            Number(center[2] || 0) * spec.depth
          );

          const rotationDeg = Array.isArray(part.rotation_deg) ? part.rotation_deg : [0, 0, 0];
          mesh.rotation.set(
            THREE.MathUtils.degToRad(Number(rotationDeg[0] || 0)),
            THREE.MathUtils.degToRad(Number(rotationDeg[1] || 0)),
            THREE.MathUtils.degToRad(Number(rotationDeg[2] || 0))
          );

          group.add(mesh);
        });

        status.textContent = `Status: added procedural 3D model for ${name}`;
        return group;
      }

      function createSliceVolumeObject(spec, name) {
        const source = spec.textureDataUrl;
        const data = spec.modelData || {};
        const positionsSrc = Array.isArray(data.positions) ? data.positions : [];
        const uvsSrc = Array.isArray(data.uvs) ? data.uvs : [];
        const frontIndicesSrc = Array.isArray(data.front_indices) ? data.front_indices : [];
        const alphaCutoff = Math.max(0.005, Math.min(0.2, Number(data.alpha_cutoff || 0.04)));

        if (positionsSrc.length < 9 || frontIndicesSrc.length < 3) {
          return null;
        }

        const positions = new Float32Array(positionsSrc.length);
        for (let i = 0; i < positionsSrc.length; i += 3) {
          positions[i] = Number(positionsSrc[i] || 0) * spec.width;
          positions[i + 1] = Number(positionsSrc[i + 1] || 0) * spec.height + (spec.height / 2);
          positions[i + 2] = Number(positionsSrc[i + 2] || 0) * spec.depth;
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        if (uvsSrc.length === (positionsSrc.length / 3) * 2) {
          const uvs = new Float32Array(uvsSrc.map((v) => Number(v || 0)));
          geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
        }

        for (let i = 0; i < frontIndicesSrc.length; i += 1) {
          frontIndicesSrc[i] = Number(frontIndicesSrc[i] || 0);
        }

        geometry.setIndex(frontIndicesSrc);
        geometry.computeVertexNormals();

        const avg = Array.isArray(data.avg_color) ? data.avg_color : [220, 180, 120];
        const fallbackColor = new THREE.Color(
          Math.max(0, Math.min(255, Number(avg[0] || 220))) / 255,
          Math.max(0, Math.min(255, Number(avg[1] || 180))) / 255,
          Math.max(0, Math.min(255, Number(avg[2] || 120))) / 255
        );

        const frontMat = new THREE.MeshBasicMaterial({
          color: 0xffffff,
          side: THREE.DoubleSide,
          transparent: false,
          alphaTest: alphaCutoff,
          depthTest: true,
          depthWrite: true,
        });

        if (source && (source.startsWith('http') || source.startsWith('data:image'))) {
          new THREE.TextureLoader().load(
            source,
            (tex) => {
              tex.encoding = THREE.sRGBEncoding;
              tex.minFilter = THREE.LinearFilter;
              tex.magFilter = THREE.LinearFilter;
              frontMat.map = tex;
              // Do not use alphaMap from color channels; it makes dark colors appear as holes.
              // PNG texture alpha is already respected by map+transparent+alphaTest.
              frontMat.needsUpdate = true;
              status.textContent = `Status: added AI depth 3D mesh for ${name}`;
            },
            undefined,
            () => {
              status.textContent = `Status: added AI depth mesh with fallback color for ${name}`;
            }
          );
        }

        return new THREE.Mesh(geometry, frontMat);
      }

      function createTriposrMeshObject(spec, name) {
        const group = new THREE.Group();
        const placeholder = createFallbackBoxObject(spec);
        group.add(placeholder);

        if (!spec.meshPath && !spec.meshObjText) {
          status.textContent = `Status: missing mesh path for ${name}, using fallback box`;
          return group;
        }

        if (!THREE.OBJLoader) {
          status.textContent = `Status: OBJLoader unavailable for ${name}, using fallback box`;
          return group;
        }

        const loader = new THREE.OBJLoader();

        function convertGeometryVertexColorsToLinear(geometry) {
          if (!geometry || !geometry.attributes || !geometry.attributes.color) {
            return;
          }
          if (geometry.userData && geometry.userData.colorsConvertedToLinear) {
            return;
          }

          const colorAttr = geometry.attributes.color;
          const color = new THREE.Color();
          for (let i = 0; i < colorAttr.count; i += 1) {
            color.fromBufferAttribute(colorAttr, i);
            color.convertSRGBToLinear();
            colorAttr.setXYZ(i, color.r, color.g, color.b);
          }
          colorAttr.needsUpdate = true;
          if (!geometry.userData) {
            geometry.userData = {};
          }
          geometry.userData.colorsConvertedToLinear = true;
        }

        function applyLoadedObject(obj) {
              // TriPoSR OBJ exports align better to the room scene after remapping
              // axes from the export convention into the app's Y-up world.
              obj.rotation.z = Math.PI / 2;
              obj.rotation.x = -Math.PI / 2;

              obj.traverse((child) => {
                if (!child.isMesh) {
                  return;
                }
                child.geometry.computeVertexNormals();
                convertGeometryVertexColorsToLinear(child.geometry);
                // Use unlit vertex-color shading so the app matches TriPoSR output colors.
                child.material = new THREE.MeshBasicMaterial({
                  color: 0xffffff,
                  vertexColors: true,
                  side: THREE.DoubleSide,
                });
              });

              const sourceBox = new THREE.Box3().setFromObject(obj);
              const sourceSize = new THREE.Vector3();
              sourceBox.getSize(sourceSize);

              // Preserve mesh proportions, but scale by the longest horizontal span
              // so furniture width matches the DB dimensions more naturally.
              const sourceHorizontal = Math.max(sourceSize.x, sourceSize.z, 0.001);
              const uniformScale = Math.max(spec.width, spec.depth, 0.001) / sourceHorizontal;
              obj.scale.setScalar(uniformScale);

              const scaledBox = new THREE.Box3().setFromObject(obj);
              const center = new THREE.Vector3();
              scaledBox.getCenter(center);
              obj.position.sub(center);

              const alignedBox = new THREE.Box3().setFromObject(obj);
              obj.position.y += -alignedBox.min.y;

              group.clear();
              group.add(obj);
              snapMeshToFloor(group);
              status.textContent = `Status: added TriPoSR mesh for ${name}`;
        }

        try {
          if (spec.meshObjText) {
            const obj = loader.parse(spec.meshObjText);
            applyLoadedObject(obj);
            return group;
          }

          loader.load(
            spec.meshPath,
            (obj) => {
              applyLoadedObject(obj);
            },
            undefined,
            () => {
              status.textContent = `Status: failed to load TriPoSR mesh for ${name}, using fallback box`;
            }
          );
        } catch (_error) {
          status.textContent = `Status: error loading TriPoSR mesh for ${name}, using fallback box`;
        }

        return group;
      }

      function createVoxelObject(spec) {
        const data = spec.modelData || {};
        const points = Array.isArray(data.points) ? data.points : [];
        const depthSteps = Math.max(2, Number(data.depth_steps || 8));
        const voxelWNorm = Number(data.voxel_w_norm || 0.02);
        const voxelHNorm = Number(data.voxel_h_norm || 0.02);

        if (!points.length) {
          return null;
        }

        const voxelW = Math.max(0.004, spec.width * voxelWNorm);
        const voxelH = Math.max(0.004, spec.height * voxelHNorm);
        const voxelD = Math.max(0.004, spec.depth / depthSteps);
        const instanceCount = points.length * depthSteps;

        const geometry = new THREE.BoxGeometry(voxelW, voxelH, voxelD);
        const material = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.84, metalness: 0.05 });
        const mesh = new THREE.InstancedMesh(geometry, material, instanceCount);
        const matrix = new THREE.Matrix4();
        const position = new THREE.Vector3();
        const quat = new THREE.Quaternion();
        const scale = new THREE.Vector3(1, 1, 1);
        const color = new THREE.Color();

        let index = 0;
        for (let pointIdx = 0; pointIdx < points.length; pointIdx += 1) {
          const point = points[pointIdx];
          const x = Number(point[0] || 0);
          const y = Number(point[1] || 0);
          const r = Number(point[2] || 160);
          const g = Number(point[3] || 140);
          const b = Number(point[4] || 120);
          const worldX = x * spec.width;
          const worldY = (y * spec.height) + (spec.height / 2);

          color.setRGB(r / 255, g / 255, b / 255);

          for (let depthIndex = 0; depthIndex < depthSteps; depthIndex += 1) {
            const dz = ((depthIndex / (depthSteps - 1)) - 0.5) * spec.depth;
            position.set(worldX, worldY, dz);
            matrix.compose(position, quat, scale);
            mesh.setMatrixAt(index, matrix);
            mesh.setColorAt(index, color);
            index += 1;
          }
        }

        mesh.instanceMatrix.needsUpdate = true;
        if (mesh.instanceColor) {
          mesh.instanceColor.needsUpdate = true;
        }
        return mesh;
      }

      function createFallbackBoxObject(spec) {
        const geom = new THREE.BoxGeometry(spec.width, spec.height, spec.depth);
        const frontBackMat = new THREE.MeshStandardMaterial({
          color: 0xf0b45b,
          roughness: 0.8,
          metalness: 0.05,
        });
        const sideMat = new THREE.MeshStandardMaterial({
          color: 0xc99347,
          roughness: 0.85,
          metalness: 0.04,
        });
        const materials = [sideMat, sideMat, sideMat, sideMat, frontBackMat, frontBackMat];

        const source = spec.textureDataUrl;
        if (source && (source.startsWith('http') || source.startsWith('data:image'))) {
          new THREE.TextureLoader().load(
            source,
            (tex) => {
              tex.encoding = THREE.sRGBEncoding;
              frontBackMat.map = tex;
              frontBackMat.needsUpdate = true;
            },
            undefined,
            () => {}
          );
        }
        return new THREE.Mesh(geom, materials);
      }

      function addProduct(name) {
        const spec = getSpec(name);
        let mesh = null;
        if (spec.modelType === 'procedural_piano_v1') {
          mesh = createProceduralObject(spec, name);
        } else if (spec.modelType === 'triposr_cpu_v1') {
          mesh = createTriposrMeshObject(spec, name);
          if (mesh) {
            status.textContent = `Status: loading TriPoSR mesh for ${name}`;
          }
        } else if (
          spec.modelType === 'depth_mesh_midas_v1'
          || spec.modelType === 'depth_mesh_midas_v2_closed'
          || spec.modelType === 'depth_mesh_cpu_v3'
        ) {
          mesh = createSliceVolumeObject(spec, name);
          if (mesh) {
            status.textContent = `Status: added AI depth 3D mesh for ${name}`;
          }
        } else if (spec.modelType === 'alpha_extrude_voxel') {
          mesh = createVoxelObject(spec);
          if (mesh) {
            status.textContent = `Status: added legacy voxel 3D model for ${name}`;
          }
        }
        if (!mesh) {
          mesh = createFallbackBoxObject(spec);
          status.textContent = `Status: fallback box model for ${name}`;
        }

        const placementOffset = productMeshes.length * 0.45;
        mesh.position.set(-0.8 + placementOffset, spec.height / 2, 0.6);
        mesh.castShadow = false;
        mesh.receiveShadow = false;
        mesh.userData = { name, originalImageDataUrl: spec.originalImageDataUrl };
        scene.add(mesh);
        snapMeshToFloor(mesh);

        productMeshes.push(mesh);
        selected = mesh;
        mesh.userData.targetPos = mesh.position.clone();
        mesh.userData.targetYaw = mesh.rotation.y;
        mesh.userData.targetPitch = mesh.rotation.x;
        mesh.userData.targetRoll = mesh.rotation.z;

        fitMeshToRoom(mesh);

        const anchor = new THREE.Vector3();
        new THREE.Box3().setFromObject(mesh).getCenter(anchor);
        orbit.target.set(anchor.x, Math.max(0.45, anchor.y), anchor.z);
        orbit.update();
        updateSelectedPreview();
      }

      renderer.domElement.addEventListener('pointerdown', (event) => {
        if (event.button !== 0) {
          return;
        }
        updatePointerFromEvent(event);
        raycaster.setFromCamera(pointer, camera);
        const intersects = raycaster.intersectObjects(productMeshes, true);
        if (intersects.length > 0) {
          let hit = intersects[0].object;
          while (hit && !productMeshes.includes(hit)) {
            hit = hit.parent;
          }
          selected = hit;
          if (!selected) {
            return;
          }
          if (interactionMode === 'move') {
            if (interactionMode === 'move') {
              if (!raycaster.ray.intersectPlane(floorPlane, dragPoint)) {
                return;
              }
              dragOffset.copy(selected.position).sub(dragPoint);
              dragMoveAxis = 'pending';
            }
            dragStartX = event.clientX;
            selected.userData.targetPos = selected.position.clone();
            dragStartY = event.clientY;
            dragStartLift = selected.position.y;
          } else {
            dragStartX = event.clientX;
            dragStartY = event.clientY;
            dragStartYaw = Number(selected.userData.targetYaw ?? selected.rotation.y);
            dragStartPitch = Number(selected.userData.targetPitch ?? selected.rotation.x);
            dragStartRoll = Number(selected.userData.targetRoll ?? selected.rotation.z);
          }

          dragActive = true;
          dragPointerId = event.pointerId;
          renderer.domElement.setPointerCapture(event.pointerId);
          renderer.domElement.style.cursor = 'grabbing';
        } else {
          selected = null;
          updateSelectedPreview();
        }
      });

      renderer.domElement.addEventListener('pointermove', (event) => {
        if (!dragActive || !selected || dragPointerId !== event.pointerId) {
          return;
        }

        if (interactionMode === 'move') {
          if (interactionMode === 'move' && dragMoveAxis === 'pending') {
            const deltaX = event.clientX - dragStartX;
            const deltaY = event.clientY - dragStartY;
            if (Math.abs(deltaY) >= DRAG_AXIS_LOCK_THRESHOLD && Math.abs(deltaY) > Math.abs(deltaX) * 1.25) {
              dragMoveAxis = 'y';
            } else if (Math.abs(deltaX) >= DRAG_AXIS_LOCK_THRESHOLD || Math.abs(deltaY) >= DRAG_AXIS_LOCK_THRESHOLD) {
              dragMoveAxis = 'xz';
            } else {
              return;
            }
          }

          if (dragMoveAxis === 'y') {
            const deltaY = event.clientY - dragStartY;
            const minY = getFloorSnappedY(selected);
            const maxY = getCeilingClampedY(selected);
            const targetY = clamp(dragStartLift - (deltaY * 0.01), minY, maxY);
            const currentTarget = selected.userData.targetPos || selected.position;
            selected.userData.targetPos = clampPositionToRoom(selected, currentTarget.x, targetY, currentTarget.z);
            status.textContent = `Status: moving ${selected.userData.name} | y=${targetY.toFixed(2)}`;
          } else {
            updatePointerFromEvent(event);
            raycaster.setFromCamera(pointer, camera);
            if (!raycaster.ray.intersectPlane(floorPlane, dragPoint)) {
              return;
            }

            const targetX = dragPoint.x + dragOffset.x;
            const targetZ = dragPoint.z + dragOffset.z;
            const currentTarget = selected.userData.targetPos || selected.position;
            selected.userData.targetPos = clampPositionToRoom(selected, targetX, currentTarget.y, targetZ);
            status.textContent = `Status: moving ${selected.userData.name} | x=${selected.userData.targetPos.x.toFixed(2)} z=${selected.userData.targetPos.z.toFixed(2)}`;
          }
        } else {
          const deltaX = event.clientX - dragStartX;
          const deltaY = event.clientY - dragStartY;
          const speed = 0.012;

          selected.userData.targetPitch = dragStartPitch + (deltaY * speed);
          if (event.shiftKey) {
            selected.userData.targetYaw = dragStartYaw;
            selected.userData.targetRoll = dragStartRoll + (deltaX * speed);
          } else {
            selected.userData.targetYaw = dragStartYaw + (deltaX * speed);
            selected.userData.targetRoll = dragStartRoll;
          }

          status.textContent = `Status: rotating ${selected.userData.name} | x=${deg(Number(selected.userData.targetPitch)).toFixed(1)}° y=${deg(Number(selected.userData.targetYaw)).toFixed(1)}° z=${deg(Number(selected.userData.targetRoll)).toFixed(1)}°`;
        }
      });

      renderer.domElement.addEventListener('pointerup', (event) => {
        if (!dragActive || dragPointerId !== event.pointerId) {
          return;
        }
        dragActive = false;
        dragPointerId = null;
        dragMoveAxis = 'xz';
        renderer.domElement.style.cursor = 'default';
        try {
          renderer.domElement.releasePointerCapture(event.pointerId);
        } catch (_error) {
        }
      });

      renderer.domElement.addEventListener('pointercancel', () => {
        dragActive = false;
        dragPointerId = null;
        dragMoveAxis = 'xz';
        renderer.domElement.style.cursor = 'default';
      });

      renderer.domElement.addEventListener('wheel', (event) => {
        if (!selected) return;
        event.preventDefault();

        if (interactionMode === 'move' || event.shiftKey) {
          const minY = getFloorSnappedY(selected);
          const maxY = getCeilingClampedY(selected);
          const liftedY = clamp(selected.position.y + (event.deltaY > 0 ? -0.02 : 0.02), minY, maxY);
          const clampedTarget = clampPositionToRoom(selected, selected.position.x, liftedY, selected.position.z);
          selected.position.copy(clampedTarget);
          if (selected.userData.targetPos) {
            selected.userData.targetPos.copy(clampedTarget);
          }
          status.textContent = `Status: lift ${selected.userData.name} | y=${selected.position.y.toFixed(2)}`;
          return;
        }

        if (interactionMode === 'rotate') {
          if (event.altKey) {
            selected.userData.targetRoll = Number(selected.userData.targetRoll ?? selected.rotation.z) + (event.deltaY > 0 ? 0.03 : -0.03);
          } else if (event.ctrlKey || event.metaKey) {
            selected.userData.targetPitch = Number(selected.userData.targetPitch ?? selected.rotation.x) + (event.deltaY > 0 ? 0.03 : -0.03);
          } else {
            selected.userData.targetYaw = Number(selected.userData.targetYaw ?? selected.rotation.y) + (event.deltaY > 0 ? 0.03 : -0.03);
          }

          status.textContent = `Status: fine rotate ${selected.userData.name} | x=${deg(Number(selected.userData.targetPitch ?? selected.rotation.x)).toFixed(1)}° y=${deg(Number(selected.userData.targetYaw ?? selected.rotation.y)).toFixed(1)}° z=${deg(Number(selected.userData.targetRoll ?? selected.rotation.z)).toFixed(1)}°`;
        }
      }, { passive: false });

      renderer.domElement.addEventListener('contextmenu', (event) => event.preventDefault());

      addBtn.addEventListener('click', () => {
        if (!assetSelect.value) return;
        addProduct(assetSelect.value);
      });

      moveBtn.addEventListener('click', () => {
        interactionMode = 'move';
        status.textContent = 'Status: move mode. Drag mostly sideways for X/Z, or mostly up/down for Y.';
      });
      rotateBtn.addEventListener('click', () => {
        interactionMode = 'rotate';
        status.textContent = 'Status: rotate mode. Drag: horizontal=Y, vertical=X. Hold Shift while dragging for Z.';
      });
      viewBtn.addEventListener('click', () => {
        setInspectView(!inspectViewEnabled);
        status.textContent = inspectViewEnabled
          ? 'Status: inspect view enabled. Camera unlocked and room bounds visible.'
          : 'Status: inspect view disabled. Camera locked to room photo.';
      });
      frontViewBtn.addEventListener('click', () => {
        setInspectView(true);
        setCameraPreset('front');
        status.textContent = 'Status: front room view';
      });
      leftViewBtn.addEventListener('click', () => {
        setInspectView(true);
        setCameraPreset('left');
        status.textContent = 'Status: left room view';
      });
      rightViewBtn.addEventListener('click', () => {
        setInspectView(true);
        setCameraPreset('right');
        status.textContent = 'Status: right room view';
      });
      backViewBtn.addEventListener('click', () => {
        setInspectView(true);
        setCameraPreset('back');
        status.textContent = 'Status: back room view';
      });
      fitBtn.addEventListener('click', () => {
        if (!selected) return;
        fitMeshToRoom(selected);
        const anchor = new THREE.Vector3();
        new THREE.Box3().setFromObject(selected).getCenter(anchor);
        orbit.target.set(anchor.x, Math.max(ROOM_MIN_Y + 0.45, anchor.y), anchor.z);
        orbit.update();
        status.textContent = `Status: fit ${selected.userData.name} inside room bounds`;
      });

      alignWallBtn.addEventListener('click', () => {
        if (!selected) return;
        const quarter = Math.PI / 2;
        selected.userData.targetYaw = Math.round((selected.userData.targetYaw ?? selected.rotation.y) / quarter) * quarter;
        status.textContent = `Status: snapped ${selected.userData.name} parallel to wall`;
      });

      deleteBtn.addEventListener('click', () => {
        if (!selected) return;
        scene.remove(selected);
        const idx = productMeshes.indexOf(selected);
        if (idx >= 0) productMeshes.splice(idx, 1);
        status.textContent = 'Status: deleted selected object';
        selected = null;
        updateSelectedPreview();
      });

      window.addEventListener('keydown', (event) => {
        if (!selected) return;
        if (event.key === 'q' || event.key === 'Q') {
          selected.userData.targetYaw = Number(selected.userData.targetYaw ?? selected.rotation.y) + 0.12;
        } else if (event.key === 'e' || event.key === 'E') {
          selected.userData.targetYaw = Number(selected.userData.targetYaw ?? selected.rotation.y) - 0.12;
        } else if (event.key === 'f' || event.key === 'F') {
          snapMeshToFloor(selected);
          selected.userData.targetPos = selected.position.clone();
        } else {
          return;
        }
        status.textContent = `Status: adjusted ${selected.userData.name} with keyboard`;
      });

      function animate() {
        requestAnimationFrame(animate);
        for (let i = 0; i < productMeshes.length; i += 1) {
          const mesh = productMeshes[i];
          const targetPos = mesh.userData.targetPos;
          if (targetPos && mesh.position.distanceTo(targetPos) > 0.0005) {
            mesh.position.lerp(targetPos, 0.22);
          }

          const targetYaw = mesh.userData.targetYaw;
          if (typeof targetYaw === 'number') {
            const deltaYaw = targetYaw - mesh.rotation.y;
            if (Math.abs(deltaYaw) > 0.0005) {
              mesh.rotation.y += deltaYaw * 0.22;
            }
          }

          const targetPitch = mesh.userData.targetPitch;
          if (typeof targetPitch === 'number') {
            const deltaPitch = targetPitch - mesh.rotation.x;
            if (Math.abs(deltaPitch) > 0.0005) {
              mesh.rotation.x += deltaPitch * 0.22;
            }
          }

          const targetRoll = mesh.userData.targetRoll;
          if (typeof targetRoll === 'number') {
            const deltaRoll = targetRoll - mesh.rotation.z;
            if (Math.abs(deltaRoll) > 0.0005) {
              mesh.rotation.z += deltaRoll * 0.22;
            }
          }
        }
        orbit.update();
        renderer.render(scene, camera);
      }
      animate();

      window.addEventListener('resize', () => {
        const width = viewport.clientWidth;
        const height = viewport.clientHeight;
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
      });

      if (assetSelect.options.length > 0) {
        addProduct(assetSelect.options[0].value);
      } else {
        status.textContent = 'Status: no 3D products found in database';
      }
    })();
  </script>
</body>
</html>
"""
    return html.replace("__PAYLOAD_JSON__", payload_json)


# --------------------------------------------------------------------------- #
# "New Developments" — multi-object detection tab (self-contained: separate
# store/pipeline above, separate viewer below, own three.js setup using the
# locally vendored modules under static/three_test/vendor instead of the CDN
# scripts the existing tabs use, so it never touches their code paths).
# --------------------------------------------------------------------------- #

def build_multi_object_viewer_html(objects: list[dict[str, Any]]) -> str:
    """Render every reconstructed object of a scene side by side, orbit-only."""
    meshes_payload = []
    for obj in objects:
        if obj.get("status") != "reconstructed" or not obj.get("mesh_path"):
            continue
        mesh_path = Path(str(obj["mesh_path"]))
        try:
            mesh_text = mesh_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        meshes_payload.append({
            "label": obj.get("label", "object"),
            "width": float(obj.get("width_m") or 1.0),
            "height": float(obj.get("height_m") or 0.85),
            "depth": float(obj.get("depth_m") or 0.7),
            "meshObjText": mesh_text,
        })

    payload_json = json.dumps({"objects": meshes_payload}, ensure_ascii=True)
    html = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    html, body { margin: 0; height: 100%; background: #eef1f4; }
    #viewport { width: 100%; height: 100%; }
  </style>
</head>
<body>
  <div id="viewport"></div>
  <script type="importmap">
    { "imports": { "three": "/app/static/three_test/vendor/three.module.js" } }
  </script>
  <script type="module">
    import * as THREE from "three";
    import { OrbitControls } from "/app/static/three_test/vendor/jsm/controls/OrbitControls.js";
    import { OBJLoader } from "/app/static/three_test/vendor/jsm/loaders/OBJLoader.js";

    const DATA = __PAYLOAD_JSON__;
    const viewport = document.getElementById('viewport');

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xeef1f4);
    const camera = new THREE.PerspectiveCamera(45, viewport.clientWidth / viewport.clientHeight, 0.01, 100);
    camera.position.set(3, 2.2, 4);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(viewport.clientWidth, viewport.clientHeight);
    viewport.appendChild(renderer.domElement);

    const orbit = new OrbitControls(camera, renderer.domElement);
    orbit.target.set(0, 0.5, 0);

    scene.add(new THREE.HemisphereLight(0xffffff, 0x555566, 1.1));
    const sun = new THREE.DirectionalLight(0xffffff, 0.7);
    sun.position.set(4, 6, 3);
    scene.add(sun);

    const loader = new OBJLoader();
    let cursorX = 0;
    (DATA.objects || []).forEach((spec) => {
      const obj = loader.parse(spec.meshObjText);
      obj.traverse((child) => {
        if (child.isMesh) {
          child.material = new THREE.MeshBasicMaterial({ color: 0xffffff, vertexColors: true, side: THREE.DoubleSide });
        }
      });
      const box = new THREE.Box3().setFromObject(obj);
      const size = new THREE.Vector3();
      box.getSize(size);
      const horizontal = Math.max(size.x, size.z, 0.001);
      obj.scale.setScalar(Math.max(spec.width, spec.depth, 0.001) / horizontal);
      const scaledBox = new THREE.Box3().setFromObject(obj);
      const center = new THREE.Vector3();
      scaledBox.getCenter(center);
      obj.position.sub(center);
      const alignedBox = new THREE.Box3().setFromObject(obj);
      obj.position.y += -alignedBox.min.y;
      obj.position.x += cursorX;
      cursorX += Math.max(spec.width, 0.5) + 0.6;
      scene.add(obj);
    });

    function animate() {
      requestAnimationFrame(animate);
      orbit.update();
      renderer.render(scene, camera);
    }
    animate();

    window.addEventListener('resize', () => {
      camera.aspect = viewport.clientWidth / viewport.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(viewport.clientWidth, viewport.clientHeight);
    });
  </script>
</body>
</html>
"""
    return html.replace("__PAYLOAD_JSON__", payload_json)


def render_multi_object_tab() -> None:
    """Upload/fetch a scene photo with several furniture items, detect each
    instance (YOLO11n-seg, CPU-only), and reconstruct each one into its own
    mesh with a free, CPU-only local visual-hull method (no GPU, no model
    downloads). Entirely separate storage (multi_object_store.py) from the
    scraped catalog used elsewhere."""
    st.caption(
        "Experimental: detect several furniture items in one photo and turn each one into "
        "its own 3D mesh. Results are stored separately and never affect the tabs above."
    )

    if MULTI_OBJECT_IMPORT_ERROR:
        st.warning(
            "This feature needs optional dependencies that are not installed: "
            f"{MULTI_OBJECT_IMPORT_ERROR}. Install with "
            "`pip install -r requirements-detect.txt`."
        )
        return

    upload_col, url_col = st.columns(2)
    with upload_col:
        uploaded = st.file_uploader("Upload a scene photo (multiple items)", type=["jpg", "jpeg", "png"])
    with url_col:
        image_url = st.text_input("...or paste an image URL")
        fetch_clicked = st.button("Fetch image from URL")

    scene_image: Image.Image | None = None
    source = ""
    if uploaded is not None:
        scene_image = Image.open(uploaded).convert("RGB")
        source = f"upload:{uploaded.name}"
    elif fetch_clicked and image_url:
        try:
            response = requests.get(image_url, headers=HEADERS, timeout=20)
            response.raise_for_status()
            scene_image = Image.open(BytesIO(response.content)).convert("RGB")
            source = f"url:{image_url}"
        except Exception as exc:
            st.error(f"Could not fetch image: {exc}")

    if scene_image is not None:
        st.image(scene_image, caption="Scene photo", width=420)
        if st.button("Detect objects in this photo"):
            with st.spinner("Running YOLO11n-seg (CPU)..."):
                try:
                    scene_id, objects = multi_object_pipeline.detect_scene(scene_image, source)
                    st.session_state["multi_object_scene_id"] = scene_id
                    if not objects:
                        st.warning("No recognizable furniture instances found in this photo.")
                    else:
                        st.success(f"Detected {len(objects)} object(s).")
                except Exception as exc:
                    st.error(f"Detection failed: {exc}")

    scene_id = st.session_state.get("multi_object_scene_id")
    if scene_id:
        objects = multi_object_store.list_objects(scene_id)
        if objects:
            st.subheader("Detected objects")
            cols = st.columns(min(4, len(objects)) or 1)
            for i, obj in enumerate(objects):
                with cols[i % len(cols)]:
                    st.image(obj["crop_path"], caption=f"{obj['label']} ({obj['score']:.2f})", width=180)
                    st.caption(f"Status: {obj['status']}")
                    if obj["status"] != "reconstructed":
                        if st.button("Build 3D mesh", key=f"reconstruct_{obj['id']}"):
                            with st.spinner(f"Reconstructing {obj['label']} (free, CPU-only local hull)..."):
                                try:
                                    multi_object_pipeline.reconstruct_object(obj["id"])
                                    st.rerun()
                                except Exception as exc:
                                    st.error(f"Reconstruction failed: {exc}")
                    if obj.get("error_message"):
                        st.caption(f"Error: {obj['error_message']}")

            reconstructed = [o for o in objects if o["status"] == "reconstructed"]
            if reconstructed:
                st.subheader("Reconstructed 3D meshes")
                viewer_html = build_multi_object_viewer_html(reconstructed)
                components.html(viewer_html, height=520, scrolling=False)


def main() -> None:
  st.set_page_config(page_title="Room Designer", layout="wide")
  st.markdown(
    """
    <style>
      .block-container {
        padding-top: 0.8rem !important;
        padding-left: 0.7rem !important;
        padding-right: 0.7rem !important;
        padding-bottom: 1rem !important;
        max-width: 100% !important;
      }
      .hero-band {
        position: relative;
        overflow: hidden;
        border: 1px solid #d7dfe7;
        border-radius: 22px;
        background: linear-gradient(135deg, #f7f9fc 0%, #ffffff 52%, #f4f7fb 100%);
        padding: 1rem 1.2rem;
        margin-bottom: 0.55rem;
      }
      .hero-grid {
        display: grid;
        grid-template-columns: 88px minmax(220px, 1.2fr) minmax(320px, 1.4fr) 220px;
        gap: 1rem;
        align-items: center;
      }
      .hero-mark {
        width: 60px;
        height: 60px;
        border-radius: 16px;
        display: grid;
        place-items: center;
        background: #eef3f8;
        color: #2a4b6e;
        font-size: 28px;
        box-shadow: inset 0 0 0 1px rgba(98, 116, 142, 0.14);
      }
      .hero-title {
        font-size: 1.72rem;
        font-weight: 800;
        color: #1d2d46;
        margin: 0 0 0.2rem 0;
      }
      .hero-copy {
        margin: 0;
        color: #334155;
        font-size: 1.25rem;
        line-height: 1.4;
      }
      .hero-db {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        margin-top: 0.7rem;
        border: 1px solid #bfe1c1;
        background: rgba(246, 255, 247, 0.9);
        color: #2b7a3d;
        border-radius: 999px;
        padding: 0.45rem 0.8rem;
        font-size: 0.88rem;
        font-weight: 700;
      }
      .hero-features {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.55rem;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.92);
        padding: 0.55rem;
        box-shadow: inset 0 0 0 1px rgba(221, 225, 232, 0.85);
      }
      .hero-feature {
        display: flex;
        gap: 0.5rem;
        align-items: flex-start;
        padding: 0.3rem 0.45rem;
      }
      .hero-feature strong {
        display: block;
        font-size: 1.22rem;
        color: #1f2937;
      }
      .hero-feature span {
        display: block;
        font-size: 1.08rem;
        color: #475569;
      }
      .hero-journey {
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.88);
        padding: 0.8rem 0.9rem;
        box-shadow: inset 0 0 0 1px rgba(222, 226, 231, 0.95);
      }
      .hero-journey h4 {
        margin: 0 0 0.45rem 0;
        font-size: 1.2rem;
        color: #1f2f44;
      }
      .hero-step {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 1.1rem;
        color: #334155;
        padding: 0.2rem 0;
      }
      .hero-step b {
        width: 20px;
        height: 20px;
        border-radius: 999px;
        display: grid;
        place-items: center;
        background: #e7edf5;
        color: #2c4a68;
        font-size: 0.74rem;
      }
      div[data-testid="stTabs"] button {
        font-size: 1.32rem !important;
        font-weight: 700 !important;
      }
      div[data-testid="stSelectbox"] label,
      div[data-testid="stNumberInput"] label,
      div[data-testid="stMarkdownContainer"] p,
      div[data-testid="stMarkdownContainer"] li,
      div[data-testid="stCaptionContainer"] {
        font-size: 1.25rem !important;
      }
      .control-card {
        border: 1px solid #d8dee7;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        background: #fbfcfe;
        margin-bottom: 1rem;
      }
      .control-card h3 {
        font-size: 1.28rem;
        margin: 0 0 0.5rem 0;
      }
      .control-card p {
        margin: 0 0 0.4rem 0;
        color: #4f5d6b;
        font-size: 1.12rem;
      }
      @media (max-width: 1200px) {
        .hero-grid {
          grid-template-columns: 72px 1fr;
        }
        .hero-features,
        .hero-journey {
          grid-column: 1 / -1;
        }
      }
    </style>
    """,
    unsafe_allow_html=True,
  )
  st.markdown(
    f"""
    <div class="hero-band">
      <div class="hero-grid">
        <div class="hero-mark">⌂</div>
        <div>
          <h1 class="hero-title">Room Designer</h1>
          <p class="hero-copy">Design and visualize your perfect space. Drag, adjust, and create with ease.</p>
        </div>
        <div class="hero-features">
          <div class="hero-feature"><div>◈</div><div><strong>3D Product Placement</strong><span>Interactive room design</span></div></div>
          <div class="hero-feature"><div>◎</div><div><strong>Real-time Preview</strong><span>Instant visual updates</span></div></div>
          <div class="hero-feature"><div>✕</div><div><strong>Easy Customization</strong><span>Shape, size, and layout</span></div></div>
          <div class="hero-feature"><div>▣</div><div><strong>Save & Order</strong><span>Seamless purchase flow</span></div></div>
        </div>
        <div class="hero-journey">
          <h4>Your Design Journey</h4>
          <div class="hero-step"><b>1</b><span>Choose room shape</span></div>
          <div class="hero-step"><b>2</b><span>Adjust room dimensions</span></div>
          <div class="hero-step"><b>3</b><span>Place furniture in 3D</span></div>
          <div class="hero-step"><b>4</b><span>Review and order</span></div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
  )

  if not DATABASE_PATH.exists():
    st.warning("Database not found; creating and populating it now.")

  try:
    room, assets, asset_meta = load_assets()
  except FileNotFoundError as exc:
    st.error(str(exc))
    return
  except Exception as exc:
    st.error(f"Failed to load assets or access database: {exc}")
    return

  existing_tab, builder_tab, multi_object_tab = st.tabs(
    ["3D Product Placement", "New Developments", "Multi-Object Detection (Beta)"]
  )

  with existing_tab:
    st.caption("Three.js test viewer backed by DB metadata and local mesh/image assets.")
    dim_col_1, dim_col_2, dim_col_3 = st.columns(3)
    with dim_col_1:
      room_width_cm = st.number_input("Room width (cm)", min_value=150, max_value=3000, value=680, step=10, key="existing_room_width")
    with dim_col_2:
      room_depth_cm = st.number_input("Room depth (cm)", min_value=150, max_value=3000, value=240, step=10, key="existing_room_depth")
    with dim_col_3:
      room_height_cm = st.number_input("Room height (cm)", min_value=180, max_value=600, value=245, step=5, key="existing_room_height")

    st.caption("Furniture dimensions come from the database and are scaled relative to the selected room dimensions.")
    payload_json = build_3d_payload(
      room,
      {
        "width": float(room_width_cm),
        "depth": float(room_depth_cm),
        "height": float(room_height_cm),
      },
    )
    html = build_3d_html(payload_json)
    components.html(html, height=900, scrolling=True)

  with builder_tab:
    room_plan: dict[str, float | str] = {
      "shape": "l_shape",
      "size_preset": "Standard",
      "width_cm": 600.0,
      "depth_cm": 480.0,
      "height_cm": 245.0,
    }
    builder_payload_json = build_room_builder_payload(room_plan)
    builder_html = build_room_builder_html(builder_payload_json)
    components.html(builder_html, height=900, scrolling=True)

  with multi_object_tab:
    render_multi_object_tab()


if __name__ == "__main__":
    main()
