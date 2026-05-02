from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import Iterable

import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageOps

HEADERS = {"User-Agent": "Mozilla/5.0"}


def sample_background(image: Image.Image) -> tuple[int, int, int]:
    rgb = image.convert("RGB")
    # Sample more points around the border for better background detection
    border_pixels = []
    width, height = rgb.size

    # Sample top and bottom borders
    for x in range(0, width, max(1, width // 20)):  # Sample every 5% across
        border_pixels.append(rgb.getpixel((x, 0)))
        border_pixels.append(rgb.getpixel((x, height - 1)))

    # Sample left and right borders
    for y in range(0, height, max(1, height // 20)):  # Sample every 5% down
        border_pixels.append(rgb.getpixel((0, y)))
        border_pixels.append(rgb.getpixel((width - 1, y)))

    # Remove outliers and get average of most common colors
    if not border_pixels:
        return (255, 255, 255)  # Default to white

    # Group similar colors
    color_groups = {}
    for pixel in border_pixels:
        key = (pixel[0] // 10, pixel[1] // 10, pixel[2] // 10)  # Group by 10-unit bins
        if key not in color_groups:
            color_groups[key] = []
        color_groups[key].append(pixel)

    # Find the most common color group (likely the background)
    if not color_groups:
        return (255, 255, 255)

    most_common_group = max(color_groups.values(), key=len)
    return tuple(sum(pixel[i] for pixel in most_common_group) // len(most_common_group) for i in range(3))


def remove_simple_background(image: Image.Image, tolerance: int = 30) -> Image.Image:
    rgba = ImageOps.exif_transpose(image).convert("RGBA")
    background = sample_background(rgba)

    # Determine if background is light or dark to adjust tolerance
    brightness = sum(background) / 3
    if brightness > 200:  # Light background
        tolerance = 25
    elif brightness > 150:  # Medium background
        tolerance = 35
    else:  # Dark background
        tolerance = 45

    pixels = []
    for red, green, blue, alpha in rgba.getdata():
        # Check distance from background color
        distance = ((red - background[0]) ** 2 + (green - background[1]) ** 2 + (blue - background[2]) ** 2) ** 0.5
        near_background = distance <= tolerance

        # Also remove very light pixels (likely background highlights)
        very_light = red >= 240 and green >= 240 and blue >= 240

        # For dark furniture, be more conservative with removal
        if brightness < 100:  # Dark background
            # Only remove if very close to background AND not too dark (to preserve dark furniture)
            near_background = near_background and (red + green + blue) > 100

        pixels.append((red, green, blue, 0 if near_background or very_light else alpha))

    rgba.putdata(pixels)

    # Clean up isolated pixels and smooth edges
    rgba = _clean_isolated_pixels(rgba)

    bbox = rgba.getbbox()
    return rgba.crop(bbox) if bbox else rgba


def _clean_isolated_pixels(image: Image.Image) -> Image.Image:
    """Remove isolated transparent pixels and smooth edges."""
    width, height = image.size
    pixels = list(image.getdata())
    new_pixels = []

    for i, (r, g, b, a) in enumerate(pixels):
        if a == 0:  # Already transparent
            new_pixels.append((r, g, b, a))
            continue

        # Check neighboring pixels
        y, x = divmod(i, width)
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    neighbor_idx = ny * width + nx
                    if neighbor_idx < len(pixels):
                        nr, ng, nb, na = pixels[neighbor_idx]
                        if na > 0:  # Opaque neighbor
                            neighbors.append((nr, ng, nb))

        # If mostly surrounded by transparent pixels, make this one transparent too
        if len(neighbors) < 3:  # Less than 3 opaque neighbors
            new_pixels.append((r, g, b, 0))
        else:
            new_pixels.append((r, g, b, a))

    result = Image.new('RGBA', (width, height))
    result.putdata(new_pixels)
    return result


def image_to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def download_and_clean_image(image_url: str) -> str | None:
    try:
        response = requests.get(image_url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGBA")
        cleaned = remove_simple_background(image)
        return image_to_data_url(cleaned)
    except Exception:
        return None


def _normalize_url(url: str, base: str) -> str:
    if not url:
        return ""
    if url.startswith("http"):
        return url
    if url.startswith("//"):
        return f"https:{url}"
    if url.startswith("/"):
        return f"{base.rstrip('/')}{url}"
    return f"{base.rstrip('/')}/{url.lstrip('/')}"


def fetch_ikea_products(limit: int = 10) -> list[dict[str, str]]:
    url = "https://www.ikea.com/nl/nl/cat/sofas-armchairs-fu003/"
    try:
        response = requests.get(url, headers=HEADERS, timeout=20)
        response.raise_for_status()
    except requests.RequestException:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    script = soup.find("script", type="application/ld+json")
    if not script or not script.string:
        return []

    try:
        data = json.loads(script.string)
    except json.JSONDecodeError:
        return []

    items = data.get("itemListElement", []) or []
    products: list[dict[str, str]] = []
    for item in items[:limit]:
        item_data = item.get("item", item) if isinstance(item, dict) else item
        name = item_data.get("name") or item.get("name") or ""
        product_url = _normalize_url(item_data.get("url") or item.get("url") or "", "https://www.ikea.com")
        image_data = item_data.get("image")
        image_url = ""
        if isinstance(image_data, dict):
            image_url = _normalize_url(image_data.get("url", ""), "https://www.ikea.com")
        elif isinstance(image_data, str):
            image_url = _normalize_url(image_data, "https://www.ikea.com")

        cleaned_image_data_url = download_and_clean_image(image_url) if image_url else ""

        products.append(
            {
                "name": name,
                "url": product_url,
                "image_url": image_url,
                "cleaned_image_data_url": cleaned_image_data_url,
                "price": "",
                "category": "Banken",
            }
        )

    return products


def fetch_meubella_products(limit: int = 10) -> list[dict[str, str]]:
    url = "https://www.meubella.nl/collections/woonkamer/products.json"
    try:
        response = requests.get(url, headers=HEADERS, timeout=20)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, json.JSONDecodeError):
        return []

    products: list[dict[str, str]] = []
    for item in data.get("products", [])[:limit]:
        name = item.get("title", "")
        handle = item.get("handle", "")
        product_url = _normalize_url(f"/products/{handle}", "https://www.meubella.nl") if handle else ""
        image_url = ""
        images = item.get("images", [])
        if images:
            image_url = _normalize_url(images[0].get("src", ""), "https://www.meubella.nl")

        price = ""
        variants = item.get("variants", [])
        if variants:
            price = variants[0].get("price", "")

        cleaned_image_data_url = download_and_clean_image(image_url) if image_url else ""

        category = item.get("product_type") or "Woonkamer"
        products.append(
            {
                "name": name,
                "url": product_url,
                "image_url": image_url,
                "cleaned_image_data_url": cleaned_image_data_url,
                "price": price,
                "category": category,
            }
        )

    return products
