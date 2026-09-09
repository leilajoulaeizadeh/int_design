---
title: Room Designer
emoji: "🛋️"
colorFrom: blue
colorTo: teal
sdk: streamlit
sdk_version: 1.37.1
app_file: app.py
pinned: false
---

# int_design — Furniture Catalog & Room Designer

A Streamlit app for browsing a scraped furniture catalog, reconstructing 3D
models of individual products, detecting multiple objects in a scene photo,
and placing furniture into a room for interactive room design.

## Project layout

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit app: catalog browsing, 3D room designer, multi-object detection tab. |
| `database.py` | SQLite schema and access layer for the scraped product catalog (`furniture.db`). |
| `scraper.py` | Scrapes furniture listings (IKEA, Furn, Meubels.com) and normalizes product images. |
| `seed_catalog.py` | CLI to populate/refresh the catalog database from the scrapers. |
| `db_inspect.py` | CLI to inspect tables/rows in `furniture.db`. |
| `triposr_cpu_pipeline.py` | Per-product 3D reconstruction via TripoSR (CPU-only). |
| `local_hull_reconstruct.py` | Free, CPU-only 3D reconstruction via shape-from-silhouette (visual hull). |
| `object_detector.py` | Multi-object detection (YOLO/ultralytics) for scraped scene/lifestyle photos. |
| `multi_object_pipeline.py` | Scene → per-object 3D reconstruction pipeline, built on `object_detector.py`. |
| `multi_object_store.py` | Storage for the multi-object detection feature (its own SQLite DB). |
| `furniture_nvs_pipeline.py` | Novel-view synthesis pipeline (pose estimation + background removal) for a single product photo. |
| `openai_practices.py` | Generates rotated catalog views of a product via Amazon Bedrock (Nova Canvas image variation). |
| `process_batch.sh` | Batch-runs `triposr_cpu_pipeline.py` over cleaned catalog images. |
| `src/int_design/` | Minimal installable package (`pip install -e .`) exposing project utilities. |
| `static/three_test/` | Standalone Three.js viewer for testing a generated mesh outside the Streamlit app. |

## Setup

### Local (virtualenv)

1. Create and activate a virtual environment:

	python3 -m venv .venv
	source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1

2. Install the base app dependencies:

	pip install -r requirements.txt

3. Optional extras, install only what you need:

	pip install -r requirements-detect.txt    # Multi-object detection tab (YOLO)
	pip install -r requirements-triposr.txt   # TripoSR 3D reconstruction

4. Start the app:

	streamlit run app.py

5. Open the URL shown in your terminal (defaults to http://localhost:8501).

### Docker

	docker compose build
	docker compose up -d      # app available at http://localhost:8501
	docker compose down

`data/`, `notebooks/`, and `furniture.db` are mounted into the container for
persistence (see `docker-compose.yml`).

## Environment variables

Copy `.env.example` to `.env` and fill in the values you need. Only required
for the Bedrock novel-view generation (`openai_practices.py`); the rest of
the app runs without any credentials.

	cp .env.example .env

`.env` is git-ignored — never commit real credentials.

## Building the catalog

	python seed_catalog.py --help

Populates `furniture.db` from the configured scrapers. Inspect the result with:

	python db_inspect.py

## 3D reconstruction

Batch reconstruct cleaned catalog images with TripoSR:

	./process_batch.sh --limit 10

Or run a single-scene, multi-object reconstruction (detection + per-object
mesh via visual hull) through `multi_object_pipeline.py`, used by the
"Multi-Object Detection" tab in the app.

## Tests

	pip install -e ".[dev]"
	pytest

## Standalone 3D interaction test UI

A separate Three.js viewer for testing interactive furniture placement and
rotation without touching the Streamlit app.

1. Ensure you already have a generated mesh, for example:

	data/output/triposr_work/0/mesh.obj

2. From the repository root, run a static file server:

	python -m http.server 8080

3. Open the test viewer in your browser:

	http://localhost:8080/static/three_test/

4. In the viewer, keep the default path or provide another OBJ path, click
   **Load Model**, then use **Move Mode** / **Rotate Mode** to manipulate it.

This viewer is intentionally isolated from the Streamlit UI.

## Deploy to Hugging Face Spaces

1. Create a new Space on Hugging Face, choose `Streamlit` as the SDK.
2. Push this repository to the Space's Git remote.
3. Ensure your images are present in `data/`.
4. Wait for the build to finish, then open your Space URL.
