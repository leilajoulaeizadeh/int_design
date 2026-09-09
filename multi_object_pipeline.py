"""Multi-object scene -> per-object 3D reconstruction pipeline.

Self-contained "New Developments" feature: given a photo containing several
furniture items, detect each instance (``object_detector.py``, YOLO11n-seg,
CPU-only) and reconstruct each one into its own mesh using the free, CPU-only
local visual-hull method (``local_hull_reconstruct.py`` — no GPU, no model
downloads, no torch). Results are kept in their own SQLite database
(``multi_object_store.py``) and their own output folder, so nothing here
touches the existing scraped catalog or its Streamlit tabs.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from PIL import Image

import local_hull_reconstruct
import multi_object_store as store
import object_detector

PROJECT_DIR = Path(__file__).resolve().parent
WORK_DIR = PROJECT_DIR / "data" / "output" / "multi_object_work"
SCENES_DIR = WORK_DIR / "scenes"
OBJECTS_DIR = WORK_DIR / "objects"
MESHES_DIR = WORK_DIR / "meshes"

# Rough real-world sizes (meters) per COCO label, only used for a plausible
# scale in the viewer — same spirit as app.py's infer_dimensions_meters, kept
# as an independent copy so this feature has no import-time dependency on app.py.
LABEL_DIMENSIONS_M: dict[str, tuple[float, float, float]] = {
    "couch": (2.0, 0.85, 0.95),
    "chair": (0.6, 0.9, 0.65),
    "bench": (1.2, 0.45, 0.4),
    "dining table": (1.4, 0.75, 0.8),
    "bed": (2.0, 0.6, 1.6),
    "tv": (1.1, 0.65, 0.08),
    "potted plant": (0.4, 0.7, 0.4),
    "refrigerator": (0.7, 1.8, 0.7),
    "oven": (0.6, 0.6, 0.6),
    "microwave": (0.5, 0.3, 0.4),
    "sink": (0.6, 0.9, 0.55),
    "toilet": (0.4, 0.4, 0.6),
    "book": (0.2, 0.25, 0.03),
}
DEFAULT_DIMENSIONS_M = (1.0, 0.85, 0.7)


def detect_scene(image: Image.Image, source: str) -> tuple[int, list[dict]]:
    """Save the scene photo, detect furniture instances, and persist each crop.

    Returns ``(scene_id, objects)`` where ``objects`` are the freshly created
    ``detected_object`` rows (as dicts).
    """
    store.initialize_database()
    SCENES_DIR.mkdir(parents=True, exist_ok=True)
    OBJECTS_DIR.mkdir(parents=True, exist_ok=True)

    scene_uid = uuid.uuid4().hex[:12]
    scene_path = SCENES_DIR / f"{scene_uid}.png"
    image.convert("RGB").save(scene_path, format="PNG")

    scene_id = store.create_scene(source=source, image_path=str(scene_path))

    detections = object_detector.detect_objects(image)
    created: list[dict] = []
    for i, detection in enumerate(detections):
        crop_path = OBJECTS_DIR / f"{scene_uid}_{i}_{detection['label'].replace(' ', '_')}.png"
        detection["image"].convert("RGB").save(crop_path, format="PNG")
        object_id = store.add_detected_object(
            scene_id=scene_id,
            label=detection["label"],
            score=detection["score"],
            crop_path=str(crop_path),
        )
        created.append(store.get_object(object_id))

    return scene_id, created


def reconstruct_object(object_id: int) -> dict:
    """Reconstruct one detected object's crop with the free CPU local hull."""
    record = store.get_object(object_id)
    if record is None:
        raise ValueError(f"No detected object with id={object_id}")

    try:
        MESHES_DIR.mkdir(parents=True, exist_ok=True)
        crop_image = Image.open(record["crop_path"])
        obj_text = local_hull_reconstruct.reconstruct_from_crop(crop_image, record["label"])

        mesh_path = MESHES_DIR / f"{object_id}.obj"
        mesh_path.write_text(obj_text, encoding="utf-8")

        width_m, height_m, depth_m = LABEL_DIMENSIONS_M.get(record["label"], DEFAULT_DIMENSIONS_M)
        store.update_object_reconstructed(
            object_id, mesh_path=str(mesh_path), width_m=width_m, height_m=height_m, depth_m=depth_m
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not swallowed
        store.update_object_failed(object_id, str(exc))
        raise

    return store.get_object(object_id)
