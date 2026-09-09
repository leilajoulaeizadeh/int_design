"""Free, CPU-only 3D reconstruction via shape-from-silhouette (visual-hull)
space carving + marching cubes — the "local hull" method from the reference
Detectron project's ``meshgen.py``, adapted for a single detected-object crop.

No GPU, no model downloads, no torch: only numpy/opencv/scipy/scikit-image/
trimesh. That reference implementation carves a hull from three orthogonal
views (top/side/front); we only have one view per detected object (the
crop from the scene photo), so the silhouette constrains width/height and is
extruded uniformly through depth — a "cardboard cutout" solid. It is coarser
than a neural single-image model (TripoSR) but fast, deterministic, and
completely free/CPU.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import measure

GRID = 96              # voxel resolution per axis
WORK_PX = 256           # crop is normalised to this before silhouetting
DEPTH_FRACTION = 0.6    # extrusion depth relative to the silhouette's own size
MIN_FG_FRAC = 0.004     # below this the crop is treated as empty
TARGET_FACES = 6000     # decimate down to this so the mesh stays browser-light


class ReconstructionError(RuntimeError):
    """The crop could not be turned into a usable mesh."""


def _silhouette(image: np.ndarray) -> np.ndarray:
    """Extract a GRID×GRID boolean silhouette from a crop on a white background."""
    import cv2

    img = cv2.resize(image, (WORK_PX, WORK_PX), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]

    fg = ((gray < 244) | (sat > 28)).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    fg = ndimage.binary_fill_holes(fg).astype(np.uint8)

    if fg.sum() < MIN_FG_FRAC * fg.size:
        raise ReconstructionError("no foreground object found in the crop")

    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fg = (labels == largest).astype(np.uint8)

    ys, xs = np.where(fg)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = fg[y0:y1, x0:x1]

    h, w = crop.shape
    side = max(h, w)
    square = np.zeros((side, side), np.uint8)
    square[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = crop
    return cv2.resize(square, (GRID, GRID), interpolation=cv2.INTER_NEAREST).astype(bool)


def _average_color(image: np.ndarray, silhouette: np.ndarray) -> tuple[float, float, float]:
    """Mean RGB (0-1) of the crop's foreground pixels, to tint the solid."""
    import cv2

    resized = cv2.resize(image, silhouette.shape[::-1], interpolation=cv2.INTER_AREA)
    fg_pixels = resized[silhouette]
    if fg_pixels.size == 0:
        return (0.7, 0.72, 0.78)
    mean = fg_pixels.reshape(-1, 3).mean(axis=0) / 255.0
    return (float(mean[0]), float(mean[1]), float(mean[2]))


def _extrude(silhouette: np.ndarray) -> np.ndarray:
    """Uniformly extrude a (y, x) silhouette through depth into a (x, y, z) volume."""
    depth = max(4, int(GRID * DEPTH_FRACTION))
    occ_xy = silhouette[::-1, :].T  # (x, y), world-up
    return np.repeat(occ_xy[:, :, None], depth, axis=2)


def _occupancy_to_obj_text(occ: np.ndarray, color: tuple[float, float, float]) -> str:
    if not occ.any():
        raise ReconstructionError("carved volume is empty")

    volume = np.pad(occ.astype(np.float32), 1)
    verts, faces, normals, _ = measure.marching_cubes(volume, level=0.5)

    import trimesh

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        mesh = max(parts, key=lambda m: m.volume if m.volume > 0 else len(m.faces))

    if len(mesh.faces) > TARGET_FACES:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=TARGET_FACES)
        except Exception:  # noqa: BLE001 - keep the full mesh if decimation is unavailable
            pass

    trimesh.smoothing.filter_humphrey(mesh, iterations=8)

    mesh.apply_translation(-mesh.bounding_box.centroid)
    scale = float(mesh.bounding_box.extents.max())
    if scale > 0:
        mesh.apply_scale(1.0 / scale)

    lines = ["# local visual-hull reconstruction (free, CPU-only)"]
    r, g, b = color
    for vx, vy, vz in mesh.vertices:
        lines.append(f"v {vx:.6f} {vy:.6f} {vz:.6f} {r:.4f} {g:.4f} {b:.4f}")
    for fa, fb_, fc in mesh.faces:
        lines.append(f"f {fa + 1} {fb_ + 1} {fc + 1}")
    return "\n".join(lines) + "\n"


def reconstruct_from_crop(image: Image.Image, label: str) -> str:
    """Turn one mask-clipped object crop into an OBJ mesh (text), free/CPU-only."""
    rgb = np.asarray(image.convert("RGB"))
    silhouette = _silhouette(rgb)
    color = _average_color(rgb, silhouette)
    occ = _extrude(silhouette)
    return _occupancy_to_obj_text(occ, color)
