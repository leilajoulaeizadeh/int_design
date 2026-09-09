import argparse
import importlib.util
from importlib import metadata
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

orig_image = "data/input/OPHS.jpg"  # --- IGNORE ---

def run_cmd(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    """Run a shell command and raise on failure."""
    print("$", " ".join(cmd))
    merged_env = None
    if env is not None:
        merged_env = dict(os.environ, **env)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=merged_env, check=True)


def ensure_triposr_repo(repo_dir: Path) -> None:
    """Clone TripoSR repo if it does not exist."""
    if repo_dir.exists() and (repo_dir / "run.py").exists():
        return

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    run_cmd([
        "git",
        "clone",
        "https://github.com/VAST-AI-Research/TripoSR.git",
        str(repo_dir),
    ])


def maybe_install_deps(repo_dir: Path, install: bool) -> None:
    """Optionally install dependencies for CPU execution."""
    if not install:
        return

    # Install CPU PyTorch first so TripoSR dependencies can build against it.
    run_cmd([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pip",
        "setuptools",
        "wheel",
    ])
    run_cmd([
        sys.executable,
        "-m",
        "pip",
        "install",
        "torch",
        "--index-url",
        "https://download.pytorch.org/whl/cpu",
    ])
    run_cmd([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        "requirements.txt",
    ], cwd=repo_dir)


def ensure_runtime_deps() -> None:
    """Install missing minimal runtime dependencies for TripoSR execution."""
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

    if not missing_packages:
        missing_packages = []

    if missing_packages:
        print(f"Installing missing runtime dependencies: {', '.join(sorted(missing_packages))}")
        run_cmd([
            sys.executable,
            "-m",
            "pip",
            "install",
            *sorted(missing_packages),
        ])

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
        print(f"Installing TripoSR-compatible versions: {', '.join(packages_to_pin)}")
        run_cmd([
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            *packages_to_pin,
        ])

    # torchmcubes is a required compiled dependency for mesh extraction.
    if importlib.util.find_spec("torchmcubes") is None:
        print("Installing missing runtime dependency: torchmcubes")
        cmake_prefix_path = subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import torch; print(torch.utils.cmake_prefix_path)",
            ],
            text=True,
        ).strip()
        run_cmd(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "git+https://github.com/tatsy/torchmcubes.git",
            ],
            env={"CMAKE_PREFIX_PATH": cmake_prefix_path},
        )


def run_triposr(
    input_image: Path,
    repo_dir: Path,
    work_out: Path,
    no_remove_bg: bool,
    render: bool,
) -> Path:
    """Run TripoSR inference on CPU and return sample output directory (index 0)."""
    work_out.mkdir(parents=True, exist_ok=True)
    sample_dir = work_out / "0"
    sample_dir.mkdir(parents=True, exist_ok=True)

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
    ]
    if render:
        cmd.append("--render")
    if no_remove_bg:
        cmd.append("--no-remove-bg")

    try:
        run_cmd(cmd, cwd=repo_dir)
    except subprocess.CalledProcessError:
        # TripoSR occasionally raises even when mesh artifacts are still produced.
        pass

    if not sample_dir.exists():
        raise FileNotFoundError(f"Expected TripoSR output at {sample_dir}")

    return sample_dir


def _front_view_score(frame_path: Path) -> float:
    """Score how likely a render is front-facing using symmetry (horizontal + vertical).

    Considers:
    - Left/right symmetry (horizontal mirror): detects front-facing
    - Top/bottom symmetry (vertical mirror): detects levelness (legs not tilted)

    Returns higher score for frames that are both front-facing and level.
    """
    rgba = np.asarray(Image.open(frame_path).convert("RGBA"), dtype=np.float32) / 255.0
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]

    gray = rgb.mean(axis=2)
    height, width = gray.shape

    # Horizontal symmetry (left/right mirror check)
    half_w = width // 2
    if half_w == 0:
        return float("-inf")

    left = gray[:, :half_w]
    right = gray[:, width - half_w :]
    left_alpha = alpha[:, :half_w]
    right_alpha = alpha[:, width - half_w :]

    right_flipped = np.fliplr(right)
    right_alpha_flipped = np.fliplr(right_alpha)

    h_weights = np.minimum(left_alpha, right_alpha_flipped)
    h_denom = float(h_weights.sum())
    if h_denom < 1e-6:
        return float("-inf")

    h_diff = np.abs(left - right_flipped)
    h_symmetry_error = float((h_diff * h_weights).sum() / h_denom)

    # Vertical symmetry (top/bottom mirror check for levelness)
    half_h = height // 2
    if half_h == 0:
        return float("-inf")

    top = gray[:half_h, :]
    bottom = gray[height - half_h :, :]
    top_alpha = alpha[:half_h, :]
    bottom_alpha = alpha[height - half_h :, :]

    bottom_flipped = np.flipud(bottom)
    bottom_alpha_flipped = np.flipud(bottom_alpha)

    v_weights = np.minimum(top_alpha, bottom_alpha_flipped)
    v_denom = float(v_weights.sum())
    if v_denom < 1e-6:
        v_symmetry_error = 0.5  # No visibility; neutral penalty
    else:
        v_diff = np.abs(top - bottom_flipped)
        v_symmetry_error = float((v_diff * v_weights).sum() / v_denom)

    # Combine scores: h_symmetry (primary) + v_symmetry for levelness (equal weight)
    combined_error = h_symmetry_error + v_symmetry_error
    return -combined_error


def select_view_frames(sample_dir: Path) -> tuple[Path, Path, Path, Path]:
    """Pick canonical 0°, 90°, -90°, and 180° rendered frames.

    We estimate the front-facing render (canonical 0°) by maximizing both
    horizontal symmetry (front-facing) and vertical symmetry (levelness).
    Then we derive the remaining angles by stepping around the render ring.
    """
    render_frames = sorted(sample_dir.glob("render_*.png"))
    if not render_frames:
        raise FileNotFoundError("No render_*.png frames found. Ensure --render is enabled.")

    n = len(render_frames)

    scores = [_front_view_score(frame) for frame in render_frames]

    # Debug: show top 5 scores
    scored_frames = sorted(zip(scores, render_frames), key=lambda x: x[0], reverse=True)
    print("[DEBUG] Top 5 frame scores (higher is better):")
    for i, (score, frame) in enumerate(scored_frames[:5]):
        print(f"  {i+1}. {frame.name}: {score:.4f}")

    idx0 = int(np.argmax(np.asarray(scores, dtype=np.float32)))

    quarter_turn = max(1, int(round(n / 4)))
    half_turn = max(1, int(round(n / 2)))
    idx90 = (idx0 + quarter_turn) % n
    idx_minus90 = (idx0 - quarter_turn) % n
    idx180 = (idx0 + half_turn) % n

    print(f"Using render_{idx0:03d}.png as canonical 0° front reference (score: {scores[idx0]:.4f})")
    return render_frames[idx0], render_frames[idx90], render_frames[idx_minus90], render_frames[idx180]


def export_views(frame0: Path, frame90: Path, frame_minus90: Path, frame180: Path, out_dir: Path) -> tuple[Path, Path, Path, Path]:
    """Copy selected views to stable filenames."""
    out_dir.mkdir(parents=True, exist_ok=True)

    out0 = out_dir / "triposr_0deg.png"
    out90 = out_dir / "triposr_90deg.png"
    out_minus90 = out_dir / "triposr_minus90deg.png"
    out180 = out_dir / "triposr_180deg.png"

    shutil.copy2(frame0, out0)
    shutil.copy2(frame90, out90)
    shutil.copy2(frame_minus90, out_minus90)
    shutil.copy2(frame180, out180)
    return out0, out90, out_minus90, out180


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-only TripoSR pipeline: image -> mesh -> 0deg/90deg/-90deg/180deg renders"
    )
    parser.add_argument(
        "--input-image",
        type=Path,
        default=Path(f"data/input/{orig_image}"),
        help="Input image path",
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path(".cache/TripoSR"),
        help="Local TripoSR repository path",
    )
    parser.add_argument(
        "--work-out",
        type=Path,
        default=Path("data/output/triposr_work"),
        help="Intermediate TripoSR output folder",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/output"),
        help="Final export folder",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install TripoSR dependencies (CPU)",
    )
    parser.add_argument(
        "--no-remove-bg",
        action="store_true",
        help="Pass --no-remove-bg to TripoSR run.py",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable TriPoSR rendered turntable outputs (render_*.png and render.mp4)",
    )
    return parser.parse_args()


def resolve_input_image(input_image: Path) -> Path:
    """Resolve input image path, defaulting to data/input for bare filenames."""
    if input_image.exists():
        return input_image

    if not input_image.is_absolute() and input_image.parent == Path("."):
        candidate = Path("data/input") / input_image
        if candidate.exists():
            return candidate

    return input_image


def main() -> None:
    args = parse_args()
    args.input_image = resolve_input_image(args.input_image)

    if not args.input_image.exists():
        raise FileNotFoundError(f"Input image not found: {args.input_image}")

    print("[1/5] Ensuring TripoSR repo...")
    ensure_triposr_repo(args.repo_dir)

    print("[2/5] Installing dependencies (optional)...")
    maybe_install_deps(args.repo_dir, args.install_deps)

    print("[2b/5] Ensuring runtime dependencies...")
    ensure_runtime_deps()

    print("[3/5] Running TripoSR on CPU...")
    sample_dir = run_triposr(
        input_image=args.input_image,
        repo_dir=args.repo_dir,
        work_out=args.work_out,
        no_remove_bg=args.no_remove_bg,
        render=args.render,
    )

    mesh_candidates = sorted(sample_dir.glob("mesh.*"))
    mesh_path = mesh_candidates[0] if mesh_candidates else sample_dir / "mesh.glb"
    print("Done.")
    print(f"Mesh: {mesh_path}")
    if args.render:
        render_mp4 = sample_dir / "render.mp4"
        render_frames = sorted(sample_dir.glob("render_*.png"))
        print(f"Render movie: {render_mp4}")
        print(f"Render frames: {len(render_frames)}")


if __name__ == "__main__":
    main()
