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

# Room Designer

This repository is ready to deploy to Hugging Face Spaces using Streamlit.

## What is included

- `app.py`: Streamlit app entry point.
- `requirements.txt`: Python dependencies for Spaces.
- `data/room.jpg`: Room background image.
- `data/*.jpg`: Furniture assets.

## Run locally

1. Create and activate a virtual environment:

	/usr/local/bin/python3 -m venv .venv
	# macOS / Linux
	source .venv/bin/activate
	# Windows PowerShell
	.\.venv\Scripts\Activate.ps1

2. Install dependencies:

	pip install -r requirements.txt

3. Start the app:

	streamlit run app.py

4. Open the URL shown in your terminal.

## Run with Docker

Build and run using Docker Compose:

1. Build the image:

	docker compose build

2. Start the app:

	docker compose up -d

3. Open the app:

	http://localhost:8501

4. Stop the app:

	docker compose down

Notes:

- `data/`, `notebooks/`, and `furniture.db` are mounted into the container for persistence.
- The Docker image includes system libraries required for image processing and TripoSR-related optional runtime dependencies.

## Deploy to Hugging Face Spaces

1. Create a new Space on Hugging Face.
2. Choose `Streamlit` as SDK.
3. Push this repository to the Space Git remote.
4. Ensure your images are present in `data/`.
5. Wait for the build to finish, then open your Space URL.

## Notes

- The app uses local files from `data/`; no upload step is required.
- Export is available through the download button in the app UI.

## Standalone 3D Interaction Test UI

This repository includes a separate Three.js viewer for testing interactive furniture placement and rotation without touching the Streamlit app.

1. Ensure you already have a generated mesh, for example:

	data/output/triposr_work/0/mesh.obj

2. From the repository root, run a static file server:

	python -m http.server 8080

3. Open the test viewer in your browser:

	http://localhost:8080/static/three_test/

4. In the viewer:

	- Keep the default path `data/output/triposr_work/0/mesh.obj`, or provide another OBJ path.
	- Click **Load Model**.
	- Use **Move Mode** and **Rotate Mode** to manipulate the object.

Notes:

- This is intentionally isolated from the existing Streamlit UI.
- The viewer is in `static/three_test/index.html`.
