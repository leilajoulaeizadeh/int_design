from __future__ import annotations

import base64
import json
from io import BytesIO
import re
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


def _get_ikea_product_image(product_url: str) -> str:
    """
    Fetch the product page and find a clean product image (ideally without background).
    IKEA product images typically include several variants - this function looks for
    the product shot image rather than room setting images.
    """
    try:
        response = requests.get(product_url, headers=HEADERS, timeout=20)
        response.raise_for_status()
    except requests.RequestException:
        return ""

    soup = BeautifulSoup(response.text, "html.parser")

    # Look for product images in the page structure
    # IKEA stores product images in various places - try to find clean product shots
    image_url = ""

    # Try to find images in structured data (JSON-LD)
    scripts = soup.find_all("script", type="application/ld+json")
    for script in scripts:
        if not script.string:
            continue
        try:
            data = json.loads(script.string)
            if isinstance(data, dict):
                # Look for Product schema with images
                if data.get("@type") == "Product" and "image" in data:
                    images = data.get("image", [])
                    if isinstance(images, list) and len(images) > 0:
                        # Prefer images with explicit product shots (earlier images are usually better)
                        for img in images:
                            if isinstance(img, dict):
                                img_url = img.get("url") or img.get("contentUrl", "")
                            else:
                                img_url = str(img)
                            if img_url:
                                image_url = _normalize_url(img_url, "https://www.ikea.com")
                                break
                    elif isinstance(images, str):
                        image_url = _normalize_url(images, "https://www.ikea.com")
                    if image_url:
                        break
        except json.JSONDecodeError:
            continue

    # If no image found in JSON-LD, look for product images in the page
    if not image_url:
        # Look for main product image in common IKEA selectors
        img_tags = soup.find_all("img")
        for img in img_tags:
            src = img.get("src") or img.get("data-src") or ""
            # Prefer images that look like product images (common IKEA patterns)
            if src and ("product" in src.lower() or "pub" in src.lower() or ".jpg" in src.lower()):
                image_url = _normalize_url(src, "https://www.ikea.com")
                break

    return image_url


def fetch_ikea_products(limit: int = 10) -> list[dict[str, str]]:
    """Fetch products from all IKEA categories."""
    all_products: list[dict[str, str]] = []
    for category_url, category_name in _IKEA_CATEGORIES:
        all_products.extend(fetch_ikea_products_from_url(category_url, category_name, limit))
    return all_products[:limit]


def fetch_ikea_products_from_url(url: str, category_name: str, limit: int = 10) -> list[dict[str, str]]:
    """Fetch IKEA products from a specific category URL."""
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
        # Handle both dict items and string URLs
        if isinstance(item, str):
            continue

        item_data = item.get("item", item) if isinstance(item, dict) else item
        if not isinstance(item_data, dict):
            continue

        name = item_data.get("name") or item.get("name") or ""
        product_url = _normalize_url(item_data.get("url") or item.get("url") or "", "https://www.ikea.com")

        # Fetch the product page to get a clean product image
        image_url = _get_ikea_product_image(product_url) if product_url else ""

        # Fallback to structured data image if product page fetch didn't yield results
        if not image_url:
            image_data = item_data.get("image")
            if isinstance(image_data, dict):
                image_url = _normalize_url(image_data.get("url", ""), "https://www.ikea.com")
            elif isinstance(image_data, str):
                image_url = _normalize_url(image_data, "https://www.ikea.com")

        cleaned_image_data_url = download_and_clean_image(image_url) if image_url else ""

        # Only add product if we have a successfully cleaned image (solo product image)
        if not cleaned_image_data_url:
            continue

        products.append(
            {
                "name": name,
                "url": product_url,
                "image_url": image_url,
                "cleaned_image_data_url": cleaned_image_data_url,
                "price": "",
                "category": category_name,
                "color": "",
            }
        )

    return products


def _extract_color_from_name(name: str) -> str:
    """Extract color hint from Dutch product names (e.g. 'kleur Zand', 'Antraciet')."""
    import re
    # Match 'kleur <Color>' pattern
    m = re.search(r"kleur\s+([A-Z][a-zA-Z\s]+?)(?:\s*[-,|]|$)", name)
    if m:
        return m.group(1).strip()
    return ""


def _get_furn_product_details(product_url: str) -> tuple[str, str]:
    """Fetch a furn.nl product page and return (image_url, color)."""
    try:
        response = requests.get(product_url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.RequestException:
        return "", ""

    soup = BeautifulSoup(response.text, "html.parser")
    image_url = ""
    color = ""

    # Try og:image first (most reliable on furn.nl)
    og_image = soup.find("meta", property="og:image")
    if og_image and og_image.get("content"):
        image_url = _normalize_url(og_image["content"], "https://furn.nl")

    # Try JSON-LD Product schema for image and color
    for script in soup.find_all("script", type="application/ld+json"):
        if not script.string:
            continue
        try:
            data = json.loads(script.string)
            if not isinstance(data, dict):
                continue
            items = [data] if data.get("@type") == "Product" else data.get("@graph", [])
            for item in items:
                if item.get("@type") != "Product":
                    continue
                if not image_url:
                    img = item.get("image")
                    if isinstance(img, list) and img:
                        img = img[0]
                    if isinstance(img, dict):
                        img = img.get("url") or img.get("contentUrl", "")
                    if img:
                        image_url = _normalize_url(str(img), "https://furn.nl")
                # Color from schema
                if not color:
                    color = item.get("color") or ""
        except json.JSONDecodeError:
            continue

    # Fallback image: images.furn.nl
    if not image_url:
        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src") or ""
            if src and ("images.furn.nl" in src or "/images/" in src):
                image_url = _normalize_url(src, "https://furn.nl")
                break

    # Fallback color: look in breadcrumb/page text for 'kleur' pattern
    if not color:
        import re
        page_text = soup.get_text(" ", strip=True)
        m = re.search(r"[Kk]leur[:\s]+([A-Z][a-zA-Z\s]+?)(?:[\s,|]|$)", page_text)
        if m:
            color = m.group(1).strip()

    return image_url, color


def _get_furn_product_image(product_url: str) -> str:
    image_url, _ = _get_furn_product_details(product_url)
    return image_url


# Furn.nl categories to scrape with their display names
_FURN_CATEGORIES = [
    ("https://furn.nl/banken", "Banken"),
    ("https://furn.nl/stoelen", "Stoelen"),
    ("https://furn.nl/tafels", "Tafels"),
    ("https://furn.nl/kasten", "Kasten"),
    ("https://furn.nl/verlichting", "Verlichting"),
]

_MEUBELS_COM_CATEGORIES = [
    ("https://meubels.com/banken", "Banken"),
    ("https://meubels.com/stoelen", "Stoelen"),
    ("https://meubels.com/tafels", "Tafels"),
    ("https://meubels.com/kasten", "Kasten"),
    ("https://meubels.com/lampen", "Lampen"),
]

_IKEA_CATEGORIES = [
    ("https://www.ikea.com/nl/nl/cat/sofas-armchairs-fu003/", "Sofas/Armchairs"),
    ("https://www.ikea.com/nl/nl/cat/tables-fu001/", "Tables"),
]


def fetch_furn_products(limit: int = 50) -> list[dict[str, str]]:
    """Scrape furn.nl furniture products across multiple categories."""
    products: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for category_url, category_name in _FURN_CATEGORIES:
        if len(products) >= limit:
            break

        try:
            response = requests.get(category_url, headers=HEADERS, timeout=20)
            response.raise_for_status()
        except requests.RequestException:
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract product links and prices from the listing page
        # Products appear as <a href="/product/..."> links
        for a_tag in soup.find_all("a", href=True):
            if len(products) >= limit:
                break

            href = a_tag["href"]
            if not href.startswith("/product/") and not href.startswith("https://furn.nl/product/"):
                continue

            product_url = _normalize_url(href, "https://furn.nl")
            if product_url in seen_urls:
                continue
            seen_urls.add(product_url)

            # Name: text content of the link, de-duplicated (furn repeats name twice in links)
            raw_text = a_tag.get_text(separator=" ", strip=True)
            # Furn repeats name+price twice: "Name price Name price" → take first half
            words = raw_text.split()
            half = len(words) // 2
            if half > 0 and " ".join(words[:half]) == " ".join(words[half:2*half]):
                raw_text = " ".join(words[:half])
            # Strip trailing price (e.g. "€ 354" or "€ 1.299")
            import re
            name = re.sub(r"\s*€[\s\d.,]+$", "", raw_text).strip()
            if not name:
                continue

            # Extract price from link text
            price_match = re.search(r"€\s*([\d.,]+)", raw_text)
            price = price_match.group(0).strip() if price_match else ""

            # Get product image and color from product page
            image_url, color = _get_furn_product_details(product_url)

            # Fallback color: extract from product name
            if not color:
                color = _extract_color_from_name(name)

            cleaned_image_data_url = download_and_clean_image(image_url) if image_url else ""

            # Only add product if we have a successfully cleaned image (solo product image)
            if not cleaned_image_data_url:
                continue

            products.append({
                "name": name,
                "url": product_url,
                "image_url": image_url,
                "cleaned_image_data_url": cleaned_image_data_url,
                "price": price,
                "category": category_name,
                "color": color,
            })

    return products[:limit]


def _get_meubels_com_product_details(product_url: str) -> tuple[str, str, str]:
    """Fetch a meubels.com product page and return (image_url, color, category)."""
    try:
        response = requests.get(product_url, headers=HEADERS, timeout=20)
        response.raise_for_status()
    except requests.RequestException:
        return "", "", ""

    soup = BeautifulSoup(response.text, "html.parser")
    image_url = ""
    color = ""
    category = ""

    og_image = soup.find("meta", property="og:image")
    if og_image and og_image.get("content"):
        image_url = _normalize_url(og_image["content"], "https://meubels.com")

    for script in soup.find_all("script", type="application/ld+json"):
        if not script.string:
            continue
        try:
            data = json.loads(script.string)
        except json.JSONDecodeError:
            continue

        items: list[dict] = []
        if isinstance(data, dict):
            if data.get("@type") == "Product":
                items = [data]
            elif isinstance(data.get("@graph"), list):
                items = [item for item in data["@graph"] if isinstance(item, dict)]

        for item in items:
            if item.get("@type") != "Product":
                continue
            if not image_url:
                image = item.get("image")
                if isinstance(image, list) and image:
                    image = image[0]
                if isinstance(image, dict):
                    image = image.get("url") or image.get("contentUrl", "")
                if image:
                    image_url = _normalize_url(str(image), "https://meubels.com")

    if not image_url:
        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src") or ""
            if src and "/images/products/" in src:
                image_url = _normalize_url(src, "https://meubels.com")
                break

    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = row.find_all(["th", "td"])
            if len(cells) != 2:
                continue
            key = cells[0].get_text(" ", strip=True).lower()
            value = cells[1].get_text(" ", strip=True)
            if not value:
                continue
            if key == "kleur" and not color:
                color = value
            elif key == "categorie" and not category:
                category = value.split(",")[0].strip()

    return image_url, color, category


def fetch_meubels_com_products(limit: int = 50) -> list[dict[str, str]]:
    """Scrape meubels.com furniture products across multiple categories."""
    products: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for category_url, fallback_category in _MEUBELS_COM_CATEGORIES:
        if len(products) >= limit:
            break

        try:
            response = requests.get(category_url, headers=HEADERS, timeout=20)
            response.raise_for_status()
        except requests.RequestException:
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        for a_tag in soup.find_all("a", href=True):
            if len(products) >= limit:
                break

            href = a_tag["href"]
            if not href.startswith("/product/") and not href.startswith("https://meubels.com/product/"):
                continue

            product_url = _normalize_url(href, "https://meubels.com")
            if product_url in seen_urls:
                continue
            seen_urls.add(product_url)

            raw_text = a_tag.get_text(separator=" ", strip=True)
            if not raw_text:
                continue

            import re

            raw_text = re.sub(r"^\d+%\s+", "", raw_text).strip()
            raw_text = re.sub(r"\s+\d+\s+webshops?\b", "", raw_text, flags=re.IGNORECASE).strip()

            price_matches = re.findall(r"€\s*[\d.,-]+", raw_text)
            price = price_matches[-1].strip() if price_matches else ""
            name = raw_text
            if price:
                name = name.rsplit(price, 1)[0].strip()
            name = re.sub(r"\s+€\s*[\d.,-]+$", "", name).strip()
            if not name:
                continue

            image_url, color, category = _get_meubels_com_product_details(product_url)
            if not color:
                color = _extract_color_from_name(name)
            cleaned_image_data_url = download_and_clean_image(image_url) if image_url else ""

            # Only add product if we have a successfully cleaned image (solo product image)
            if not cleaned_image_data_url:
                continue

            products.append(
                {
                    "name": name,
                    "url": product_url,
                    "image_url": image_url,
                    "cleaned_image_data_url": cleaned_image_data_url,
                    "price": price,
                    "category": category or fallback_category,
                    "color": color,
                }
            )

    return products[:limit]


def _build_meubelo_image_url(image_id: str) -> str:
    if not image_id:
        return ""

    token = image_id.strip().lstrip("/")
    if token.startswith("http"):
        return token

    if "." not in token.rsplit("/", 1)[-1]:
        token = f"{token}.jpg"

    base_name, ext = token.rsplit(".", 1)
    # Use a concrete size variant to avoid tiny placeholders.
    if not re.search(r"-\d+x\d+$", base_name):
        token = f"{base_name}-600x600.{ext}"

    return _normalize_url(token, "https://product-images-cdn.meubelo.nl")


def fetch_meubelo_products(limit: int = 50) -> list[dict[str, str]]:
    """Fetch first products from meubelo.nl via its public Algolia proxy endpoint."""
    session = requests.Session()
    try:
        session.get("https://www.meubelo.nl/", headers=HEADERS, timeout=20)
    except requests.RequestException:
        return []

    products: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    start = 0
    page_size = 50

    while len(products) < limit and start < 500:
        payload = {
            "categoryId": "0",
            "queryParameters": {"attributesToHighlight": []},
            "source": "search",
            "traceId": "room-designer-meubelo",
            "productsRange": [start, start + page_size - 1],
        }

        try:
            response = session.post(
                "https://www.meubelo.nl/api/algolia/products",
                headers={
                    **HEADERS,
                    "Content-Type": "application/json",
                    "Origin": "https://www.meubelo.nl",
                    "Referer": "https://www.meubelo.nl/",
                },
                data=json.dumps(payload),
                timeout=25,
            )
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, json.JSONDecodeError):
            break

        hits = data.get("hits", [])
        if not hits:
            break

        for hit in hits:
            if len(products) >= limit:
                break

            title = (hit.get("title") or "").strip()
            cluster = hit.get("cluster") or {}
            cluster_id = cluster.get("id") or ""
            product_url = _normalize_url(f"/p/{cluster_id}", "https://www.meubelo.nl") if cluster_id else ""

            if not title or not product_url or product_url in seen_urls:
                continue
            seen_urls.add(product_url)

            image_ids = hit.get("images") or []
            image_url = _build_meubelo_image_url(image_ids[0]) if image_ids else ""
            if not image_url:
                continue

            cleaned_image_data_url = download_and_clean_image(image_url)
            # Keep only products where we could isolate a usable product image.
            if not cleaned_image_data_url:
                continue

            price_info = hit.get("price") or {}
            value = price_info.get("value")
            price = ""
            if isinstance(value, (int, float)):
                price = f"€ {value:.2f}"

            category_names = hit.get("categoryNames") or []
            category = category_names[0] if category_names else "Woonkamer"

            features = hit.get("features") or {}
            colors = features.get("colour") or []
            color = colors[0] if colors else ""

            products.append(
                {
                    "name": title,
                    "url": product_url,
                    "image_url": image_url,
                    "cleaned_image_data_url": cleaned_image_data_url,
                    "price": price,
                    "category": category,
                    "color": color,
                }
            )

        start += page_size

    return products[:limit]


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

        # Extract color from variant options
        color = ""
        for variant in variants:
            for opt in ["option1", "option2", "option3"]:
                val = variant.get(opt, "") or ""
                if val.lower() not in ("default title", ""):
                    color = val
                    break
            if color:
                break

        category = item.get("product_type") or "Woonkamer"
        products.append(
            {
                "name": name,
                "url": product_url,
                "image_url": image_url,
                "cleaned_image_data_url": cleaned_image_data_url,
                "price": price,
                "category": category,
                "color": color,
                "rating": "",
            }
        )

    return products
