"""Multi-object detection for scraped scene/lifestyle photos.

Retailer photos are sometimes a whole room or vignette with several furniture
pieces in one shot (e.g. a sofa + coffee table + lamp staged together). The
rest of the pipeline (``triposr_cpu_pipeline.py``) expects one object per
image, so this module finds each furniture instance in a scene photo and crops
it out (mask-clipped, on a plain white background) so it can be fed through
the existing single-object reconstruction pipeline unchanged.

Uses Ultralytics YOLO11n-seg: CPU-only, no GPU required, and much lighter to
install than Detectron2 (single ``pip install ultralytics``). Weights are
bundled at ``data/models/yolo11n-seg.pt`` so no network access is needed at
runtime.

Caveat: YOLO11n-seg is trained on COCO, whose 80 classes only cover a handful
of furniture types (chair, couch, bed, dining table, tv, potted plant, ...).
Items like lamps, wardrobes, bookshelves or rugs will not be detected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

# cv2/ultralytics are optional extras (see requirements-detect.txt) and are
# imported lazily inside functions so importing this module never breaks the
# rest of the app when they aren't installed yet.

WEIGHTS_PATH = Path(__file__).resolve().parent / "data" / "models" / "yolo11n-seg.pt"

# COCO classes that correspond to furniture/home items detectable out of the box.
DEFAULT_FURNITURE_LABELS = frozenset({
    "chair", "couch", "bed", "dining table", "tv", "potted plant",
    "refrigerator", "oven", "sink", "toilet", "bench", "microwave", "book",
})

_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        weights = str(WEIGHTS_PATH) if WEIGHTS_PATH.exists() else "yolo11n-seg.pt"
        _MODEL = YOLO(weights)
    return _MODEL


def _mask_from_polygon(polygon: np.ndarray, height: int, width: int) -> np.ndarray:
    import cv2
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
    return mask


def detect_objects(
    image: Image.Image,
    labels: Iterable[str] | None = DEFAULT_FURNITURE_LABELS,
    conf: float = 0.35,
    margin_px: int = 12,
) -> list[dict]:
    """Detect furniture instances in a scene photo and crop each one out.

    Returns a list of ``{"label", "score", "image"}`` where ``image`` is an
    RGBA ``PIL.Image`` cropped to the instance's bounding box (with a small
    margin) and mask-clipped onto a plain white background, mirroring the
    look of the single-product photos the rest of the pipeline expects.
    """
    import cv2

    model = _load_model()
    label_filter = set(labels) if labels is not None else None

    rgb = np.asarray(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    height, width = bgr.shape[:2]

    result = model.predict(bgr, imgsz=960, conf=conf, verbose=False)[0]
    names = result.names
    boxes = result.boxes
    polys = result.masks.xy if result.masks is not None else None

    detections: list[dict] = []
    for i in range(len(boxes)):
        label = names[int(boxes.cls[i])]
        if label_filter is not None and label not in label_filter:
            continue

        x0, y0, x1, y1 = (int(v) for v in boxes.xyxy[i].tolist())
        x0 = max(0, x0 - margin_px)
        y0 = max(0, y0 - margin_px)
        x1 = min(width, x1 + margin_px)
        y1 = min(height, y1 + margin_px)
        if x1 <= x0 or y1 <= y0:
            continue

        if polys is not None and i < len(polys) and len(polys[i]) >= 3:
            mask = _mask_from_polygon(np.asarray(polys[i]), height, width)
        else:
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[y0:y1, x0:x1] = 1

        crop_rgb = rgb[y0:y1, x0:x1]
        crop_mask = mask[y0:y1, x0:x1]

        rgba = np.dstack([crop_rgb, np.full(crop_mask.shape, 255, dtype=np.uint8)])
        rgba[..., 3] = np.where(crop_mask > 0, 255, 0)
        cropped = Image.fromarray(rgba, mode="RGBA")

        # Flatten onto white so it matches the single-object scraper output.
        on_white = Image.new("RGBA", cropped.size, (255, 255, 255, 255))
        on_white.alpha_composite(cropped)

        detections.append({
            "label": label,
            "score": round(float(boxes.conf[i]), 3),
            "image": on_white.convert("RGBA"),
        })

    return detections
