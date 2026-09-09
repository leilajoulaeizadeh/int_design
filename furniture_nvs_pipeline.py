import argparse
import importlib.util
from importlib import metadata
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rembg import remove


@dataclass
class PoseEstimate:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    quad: np.ndarray


def run_cmd(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    """Run shell command and raise on failure."""
    print("$", " ".join(cmd))
    merged_env = None
    if env is not None:
        merged_env = dict(os.environ, **env)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=merged_env, check=True)


def ensure_triposr_repo(repo_dir: Path) -> None:
    """Clone TriPoSR repository if needed."""
    if repo_dir.exists() and (repo_dir / "run.py").exists():
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(["git", "clone", "https://github.com/VAST-AI-Research/TripoSR.git", str(repo_dir)])


def ensure_runtime_deps() -> None:
    """Install missing TriPoSR runtime dependencies in current Python environment."""
    required_modules = {
        "rembg": "rembg",
        "onnxruntime": "onnxruntime",
        "trimesh": "trimesh",
        "torch": "torch",
        "PIL": "Pillow",
        "einops": "einops",
        "transformers": "transformers",
        "omegaconf": "omegaconf",
        "xatlas": "xatlas",
        "moderngl": "moderngl",
        "imageio_ffmpeg": "imageio[ffmpeg]",
    }

    missing_packages = [
        package
        for module_name, package in required_modules.items()
        if importlib.util.find_spec(module_name) is None
    ]

    if missing_packages:
        run_cmd([sys.executable, "-m", "pip", "install", *sorted(missing_packages)])

    pinned_packages = {
        "transformers": "4.35.0",
        "einops": "0.7.0",
        "omegaconf": "2.3.0",
        "trimesh": "4.0.5",
        "xatlas": "0.0.9",
        "moderngl": "5.10.0",
    }
    packages_to_pin: list[str] = []
    for package_name, expected_version in pinned_packages.items():
        try:
            installed_version = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            installed_version = None

        if installed_version != expected_version:
            packages_to_pin.append(f"{package_name}=={expected_version}")

    if packages_to_pin:
        run_cmd([sys.executable, "-m", "pip", "install", "--upgrade", *packages_to_pin])

    if importlib.util.find_spec("torchmcubes") is None:
        cmake_prefix_path = subprocess.check_output(
            [sys.executable, "-c", "import torch; print(torch.utils.cmake_prefix_path)"],
            text=True,
        ).strip()
        run_cmd(
            [sys.executable, "-m", "pip", "install", "git+https://github.com/tatsy/torchmcubes.git"],
            env={"CMAKE_PREFIX_PATH": cmake_prefix_path},
        )


def load_rgba(image_path: Path) -> np.ndarray:
    """Load image as RGBA."""
    return np.asarray(Image.open(image_path).convert("RGBA"))


def extract_full_object_mask(rgba: np.ndarray) -> np.ndarray:
    """Extract a single coherent furniture mask for the full object."""
    alpha = rgba[..., 3]
    if np.count_nonzero(alpha < 250) > 0:
        mask = (alpha > 10).astype(np.uint8)
    else:
        rgb_pil = Image.fromarray(rgba[..., :3], mode="RGB")
        fg = np.asarray(remove(rgb_pil).convert("RGBA"))
        mask = (fg[..., 3] > 10).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        raise RuntimeError("No foreground object detected.")

    # Keep one coherent object instance: the largest connected component.
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    sofa_mask = (labels == largest).astype(np.uint8)

    kernel = np.ones((7, 7), dtype=np.uint8)
    sofa_mask = cv2.morphologyEx(sofa_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    sofa_mask = cv2.morphologyEx(sofa_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return sofa_mask


def compute_object_quad(mask: np.ndarray) -> np.ndarray:
    """Approximate furniture support quad from contour hull."""
    contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Unable to find object contour.")

    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect).astype(np.float32)

    sums = box.sum(axis=1)
    diffs = np.diff(box, axis=1).ravel()
    tl = box[np.argmin(sums)]
    br = box[np.argmax(sums)]
    tr = box[np.argmin(diffs)]
    bl = box[np.argmax(diffs)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def rotation_matrix_to_euler_xyz_deg(rotation_matrix: np.ndarray) -> tuple[float, float, float]:
    """Convert rotation matrix to Euler angles in degrees."""
    sy = float(np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2))
    singular = sy < 1e-6

    if not singular:
        pitch = float(np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
        yaw = float(np.arctan2(-rotation_matrix[2, 0], sy))
        roll = float(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
    else:
        pitch = float(np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1]))
        yaw = float(np.arctan2(-rotation_matrix[2, 0], sy))
        roll = 0.0

    return np.degrees(pitch), np.degrees(yaw), np.degrees(roll)


def estimate_pose(mask: np.ndarray, quad: np.ndarray) -> PoseEstimate:
    """Estimate camera pose from object quad; yaw is the primary target."""
    tl, tr, br, bl = quad
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl), 1.0)
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr), 1.0)

    object_points = np.array(
        [
            [-width / 2.0, -height / 2.0, 0.0],
            [width / 2.0, -height / 2.0, 0.0],
            [width / 2.0, height / 2.0, 0.0],
            [-width / 2.0, height / 2.0, 0.0],
        ],
        dtype=np.float32,
    )

    h, w = mask.shape[:2]
    focal = float(max(h, w) * 1.2)
    camera_matrix = np.array(
        [[focal, 0.0, w / 2.0], [0.0, focal, h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    ok, rvec, _ = cv2.solvePnP(object_points, quad.astype(np.float32), camera_matrix, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        ok, rvec, _ = cv2.solvePnP(
            object_points,
            quad.astype(np.float32),
            camera_matrix,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    if not ok:
        raise RuntimeError("Pose estimation failed.")

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = rotation_matrix_to_euler_xyz_deg(rotation_matrix)
    return PoseEstimate(yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll, quad=quad)


def crop_alpha(rgba: np.ndarray, margin: int = 24) -> np.ndarray:
    """Crop transparent borders but keep complete object with margin."""
    alpha = rgba[..., 3]
    ys, xs = np.where(alpha > 10)
    if ys.size == 0 or xs.size == 0:
        return rgba

    y0 = max(int(ys.min()) - margin, 0)
    y1 = min(int(ys.max()) + margin + 1, rgba.shape[0])
    x0 = max(int(xs.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin + 1, rgba.shape[1])
    return rgba[y0:y1, x0:x1]


def rectify_to_front(rgba: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Yaw-focused geometric rectification using homography to canonical front plane."""
    tl, tr, br, bl = quad
    top_w = np.linalg.norm(tr - tl)
    bottom_w = np.linalg.norm(br - bl)
    left_h = np.linalg.norm(bl - tl)
    right_h = np.linalg.norm(br - tr)

    target_w = max(int(round((top_w + bottom_w) / 2.0)), 512)
    target_h = max(int(round((left_h + right_h) / 2.0)), 320)

    dst = np.array(
        [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
        dtype=np.float32,
    )
    h_mat = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    warped = cv2.warpPerspective(
        rgba,
        h_mat,
        (target_w, target_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return crop_alpha(warped)


def save_rgba(rgba: np.ndarray, path: Path) -> None:
    """Save RGBA image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba).save(path)


def run_triposr(input_image: Path, repo_dir: Path, work_out: Path) -> Path:
    """Run TriPoSR and return rendered sample directory."""
    work_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "run.py",
        str(input_image.resolve()),
        "--device",
        "cpu",
        "--output-dir",
        str(work_out.resolve()),
        "--model-save-format",
        "obj",
        "--render",
    ]

    try:
        run_cmd(cmd, cwd=repo_dir)
    except subprocess.CalledProcessError:
        pass

    sample_dir = work_out / "0"
    if not sample_dir.exists():
        raise RuntimeError(f"Missing TriPoSR output directory: {sample_dir}")

    if not list(sample_dir.glob("render_*.png")):
        raise RuntimeError("TriPoSR did not produce rendered frames.")

    return sample_dir


def masked_mse(ref_rgba: np.ndarray, frame_rgba: np.ndarray) -> float:
    """Score frame similarity to canonical front image (lower is better)."""
    h, w = ref_rgba.shape[:2]
    resized = cv2.resize(frame_rgba, (w, h), interpolation=cv2.INTER_AREA)

    ref_a = ref_rgba[..., 3].astype(np.float32) / 255.0
    frm_a = resized[..., 3].astype(np.float32) / 255.0
    weights = np.minimum(ref_a, frm_a)
    denom = float(weights.sum())
    if denom < 1e-6:
        return float("inf")

    ref_rgb = ref_rgba[..., :3].astype(np.float32) / 255.0
    frm_rgb = resized[..., :3].astype(np.float32) / 255.0
    diff = ((ref_rgb - frm_rgb) ** 2).mean(axis=2)
    return float((diff * weights).sum() / denom)


def select_canonical_views(sample_dir: Path, canonical_front_rgba: np.ndarray) -> tuple[Path, Path, Path, Path]:
    """Select 0/90/180/270 frames from TriPoSR render ring."""
    frames = sorted(sample_dir.glob("render_*.png"))
    if not frames:
        raise RuntimeError("No render frames found.")

    scores = []
    for frame in frames:
        rgba = np.asarray(Image.open(frame).convert("RGBA"))
        scores.append(masked_mse(canonical_front_rgba, rgba))

    idx0 = int(np.argmin(np.asarray(scores, dtype=np.float32)))
    n = len(frames)
    q = max(1, int(round(n / 4)))
    h = max(1, int(round(n / 2)))

    idx90 = (idx0 + q) % n
    idx180 = (idx0 + h) % n
    idx270 = (idx0 - q) % n

    print(f"Canonical render index: {idx0} (score={scores[idx0]:.5f})")
    return frames[idx0], frames[idx90], frames[idx180], frames[idx270]


def export_views(frame0: Path, frame90: Path, frame180: Path, frame270: Path, out_dir: Path) -> tuple[Path, Path, Path, Path]:
    """Export final named outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out0 = out_dir / "sofa_front.png"
    out90 = out_dir / "sofa_right_90.png"
    out180 = out_dir / "sofa_back_180.png"
    out270 = out_dir / "sofa_left_270.png"

    shutil.copy2(frame0, out0)
    shutil.copy2(frame90, out90)
    shutil.copy2(frame180, out180)
    shutil.copy2(frame270, out270)
    return out0, out90, out180, out270


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="General detect-segment-rectify-reconstruct-render furniture pipeline.")
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/output"))
    parser.add_argument("--work-out", type=Path, default=Path("data/output/triposr_work/15"))
    parser.add_argument("--repo-dir", type=Path, default=Path(".cache/TripoSR"))
    parser.add_argument("--save-rectified", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_image.exists():
        raise FileNotFoundError(f"Input image not found: {args.input_image}")

    print("[1/6] Load and segment full object...")
    rgba = load_rgba(args.input_image)
    sofa_mask = extract_full_object_mask(rgba)

    rgba_fg = rgba.copy()
    rgba_fg[..., 3] = (sofa_mask * 255).astype(np.uint8)

    print("[2/6] Estimate pose (yaw primary)...")
    quad = compute_object_quad(sofa_mask)
    pose = estimate_pose(sofa_mask, quad)
    print(f"Estimated yaw={pose.yaw_deg:.2f} deg, pitch={pose.pitch_deg:.2f} deg, roll={pose.roll_deg:.2f} deg")

    print("[3/6] Rectify to canonical front view...")
    rectified = rectify_to_front(rgba_fg, quad)
    rectified_path = args.out_dir / "rectified_front_reference.png"
    if args.save_rectified:
        save_rgba(rectified, rectified_path)

    print("[4/6] Reconstruct full 3D object with TriPoSR...")
    ensure_triposr_repo(args.repo_dir)
    ensure_runtime_deps()
    sample_dir = run_triposr(rectified_path if args.save_rectified else args.input_image, args.repo_dir, args.work_out)

    print("[5/6] Select canonical 0/90/180/270 novel views...")
    frame0, frame90, frame180, frame270 = select_canonical_views(sample_dir, rectified)

    print("[6/6] Save outputs...")
    out0, out90, out180, out270 = export_views(frame0, frame90, frame180, frame270, args.out_dir)

    print("Done.")
    print(f"0 deg   : {out0}")
    print(f"90 deg  : {out90}")
    print(f"180 deg : {out180}")
    print(f"270 deg : {out270}")
    if args.save_rectified:
        print(f"rectified reference: {rectified_path}")


if __name__ == "__main__":
    main()
