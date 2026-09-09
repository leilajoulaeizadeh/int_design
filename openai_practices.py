import base64
import json
import boto3
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from io import BytesIO

from PIL import Image

aws_region = os.getenv("aws_region", "eu-west-1")
aws_access_key_id = os.getenv("aws_access_key_id")
aws_secret_access_key = os.getenv("aws_secret_access_key")
aws_session_token = os.getenv("aws_session_token")

# # Get credentials from a session (useful for assumed roles)
# session = boto3.Session(region_name=aws_region)
# credentials = session.get_credentials().get_frozen_credentials()

# Chained angle pipeline: each entry uses the previous result as reference.
# The canonical front view (0°) is generated first from the original image,
# then every subsequent angle is generated from the immediately preceding one.
ANGLES = [
    {
        "name": "side_90",
        "degrees": 90,
        "view_prompt": """
The reference image shows the sofa from EXACTLY the front (0 degrees).

Rotate the camera exactly 90 degrees to the right from that front position.

RESULT: TRUE LEFT PROFILE VIEW.
- The camera is now perpendicular to the sofa's left side.
- Only the left side panel is visible.
- The front face is completely hidden — not visible at all.
- The back face is completely hidden — not visible at all.
- The legs visible from the left side profile.

Orthographic side elevation. Furniture catalog photography.
""",
    },
]

input_image = Path("data/input/OPHS.jpg")


def _open_image_with_fallback(path: Path) -> Image.Image:
    """Open image with Pillow, fallback to ImageMagick conversion for AVIF/HEIC."""
    try:
        return Image.open(path)
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_png = Path(tmp.name)
        try:
            subprocess.run(["convert", str(path), str(temp_png)], check=True, capture_output=True)
            with Image.open(temp_png) as converted:
                return converted.copy()
        finally:
            if temp_png.exists():
                temp_png.unlink()


def image_to_base64(path: Path) -> str:
    """Encode image to base64 and ensure minimum size for Bedrock image variation."""
    with _open_image_with_fallback(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        if width < 320 or height < 320:
            scale = max(320 / width, 320 / height)
            image = image.resize((int(width * scale), int(height * scale)))

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")



def negative_prompt() -> str:
    """Negative text applied to every generation call to prevent identity drift."""
    return (
        "different sofa, new sofa, redesigned sofa, reinterpreted sofa, "
        "different color, different fabric, different upholstery, different material, "
        "different armrests, different legs, different leg style, different leg shape, "
        "different leg color, changed legs, modified legs, replaced legs, "
        "different cushions, different proportions, "
        "different model, different brand, different shape, different style, "
        "modified furniture, changed furniture, replaced furniture, "
        "accessories, pillows, room, environment, wall, floor texture"
    )


def make_canonical_front_prompt() -> str:
    """Prompt to normalise any tilted input photo into a 0° front elevation."""
    return """
Reference image contains a sofa photographed from an unknown angle.

TASK:
Create a canonical TRUE FRONT ELEVATION furniture catalog image.

Move the camera so it is centered directly in front of the sofa.

The sofa must face the camera squarely.

TRUE FRONT ELEVATION:
- No left side visible.
- No right side visible.
- Camera at seat-back height, centered on the sofa mid-line.

Treat the sofa as a fixed rigid 3D object. Move ONLY the camera.

Preserve EXACTLY:
- dimensions and proportions
- armrests shape and size
- seat cushions and back cushions
- upholstery, stitching, seams, tufting
- legs: exact same shape, height, material, color and spacing
- color and material

White seamless studio background.
Furniture catalog photography.
Entire sofa visible and centered.
No accessories. No pillows. No environment.
"""


def make_prompt_for_angle(angle_info: dict) -> str:
    """Prompt to rotate the camera by the given angle from the reference image."""
    return f"""
Reference image shows the EXACT sofa that must appear in the output.

TASK:
Generate the SAME sofa from a new camera position.

{angle_info["view_prompt"]}

Treat the sofa as a fixed rigid 3D object. Move ONLY the camera.

Preserve EXACTLY:
- dimensions and proportions
- silhouette
- armrests shape and size
- seat cushions and back cushions
- upholstery, stitching, seams, tufting
- legs: exact same shape, height, material, color and spacing
- material, texture, color

Do NOT:
- redesign or reinterpret
- modify any part of the sofa
- generate a different sofa

White seamless studio background.
Product catalog photography.
Entire sofa visible and centered.
No accessories. No pillows. No environment.
No perspective exaggeration.
High consistency with the reference image.
"""

try:
    runtime = boto3.client(
        "bedrock-runtime",
        region_name=aws_region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
    )
    bedrock = boto3.client(
        "bedrock",
        region_name=aws_region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
    )

    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    models_response = bedrock.list_foundation_models(byOutputModality="IMAGE")
    model_summaries = models_response.get("modelSummaries", [])
    if not model_summaries:
        raise ValueError(f"No IMAGE models available in region {aws_region}")
    model_id = "amazon.nova-canvas-v1:0"
    if all(model.get("modelId") != model_id for model in model_summaries):
        model_id = model_summaries[0]["modelId"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_reference_b64 = image_to_base64(input_image)

    # ------------------------------------------------------------------ #
    # Step 1: Generate canonical front view (0°) from the original image. #
    # This corrects any tilt in the source photo and establishes a stable  #
    # reference frame for all subsequent rotations.                        #
    # ------------------------------------------------------------------ #
    print("\nGenerating canonical front view (0°) from original image...")
    front_prompt = make_canonical_front_prompt().strip()[:1024]

    front_body = {
        "taskType": "IMAGE_VARIATION",
        "imageVariationParams": {
            "images": [original_reference_b64],
            "text": front_prompt,
            "negativeText": negative_prompt(),
            # High similarity: only allow the perspective shift, not a redesign.
            "similarityStrength": 0.97,
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "quality": "standard",
            "height": 1024,
            "width": 1024,
            # Higher cfgScale makes the model follow prompts more strictly.
            "cfgScale": 7,
            "seed": 0,
        },
    }

    front_response = runtime.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(front_body),
    )
    front_payload = json.loads(front_response["body"].read())
    front_images = front_payload.get("images", [])

    if not front_images:
        raise ValueError("No images returned for canonical front view")

    canonical_front_b64 = front_images[0]
    front_path = output_dir / f"front_0deg_{timestamp}.png"
    front_path.write_bytes(base64.b64decode(canonical_front_b64))
    print(f"✓ Saved: {front_path.resolve()}")

    # ------------------------------------------------------------------ #
    # Step 2: Chain angle generation.                                      #
    # Each angle uses the PREVIOUS result as its reference image so the    #
    # model only needs to rotate by a small increment each time.           #
    # ------------------------------------------------------------------ #
    current_reference_b64 = canonical_front_b64

    for angle_info in ANGLES:
        print(f"\nGenerating {angle_info['name']} view ({angle_info['degrees']}°)...")
        prompt_text = make_prompt_for_angle(angle_info).strip()[:1024]

        body = {
            "taskType": "IMAGE_VARIATION",
            "imageVariationParams": {
                # Use ONLY the canonical front view as reference.
                # Passing the original (unknown-angle) image alongside would
                # confuse the model about which angle to rotate FROM and causes
                # identity drift (legs, armrests change).
                "images": [canonical_front_b64],
                "text": prompt_text,
                "negativeText": negative_prompt(),
                # Very high similarity: change only the camera angle.
                "similarityStrength": 0.97,
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "quality": "standard",
                "height": 1024,
                "width": 1024,
                "cfgScale": 7,
                "seed": angle_info["degrees"],
            },
        }

        response = runtime.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        payload = json.loads(response["body"].read())
        images = payload.get("images", [])

        if not images:
            print(f"  ⚠ No images returned for {angle_info['name']}, skipping.")
            continue

        # The first returned image becomes the reference for the next angle.
        current_reference_b64 = images[0]

        for idx, img_b64 in enumerate(images):
            output_path = output_dir / f"{angle_info['name']}_{idx + 1}_{timestamp}.png"
            output_path.write_bytes(base64.b64decode(img_b64))
            print(f"✓ Saved: {output_path.resolve()}")

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")




