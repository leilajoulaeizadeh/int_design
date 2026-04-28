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

## Deploy to Hugging Face Spaces

1. Create a new Space on Hugging Face.
2. Choose `Streamlit` as SDK.
3. Push this repository to the Space Git remote.
4. Ensure your images are present in `data/`.
5. Wait for the build to finish, then open your Space URL.

## Notes

- The app uses local files from `data/`; no upload step is required.
- Export is available through the download button in the app UI.
