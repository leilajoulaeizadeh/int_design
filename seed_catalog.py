from __future__ import annotations

import argparse
from collections.abc import Callable

import scraper as scraper_module
from database import clear_catalog_data, get_products, upsert_products
from scraper import (
    fetch_furn_products,
    fetch_meubels_com_products,
    fetch_ikea_products_from_url,
)

Product = dict[str, str]
Fetcher = Callable[[int], list[Product]]

SOFA_TOKENS = (
    "bank",
    "banken",
    "sofa",
    "sofas",
    "armchair",
    "armchairs",
    "seat",
    "seats",
    "stoel",
    "stoelen",
    "fauteuil",
    "fauteuils",
    "zetel",
    "zetels",
)
TABLE_TOKENS = (
    "tafel",
    "tafels",
    "table",
    "tables",
    "eettafel",
    "salontafel",
    "sidetable",
    "bijzettafel",
)


def _norm(text: str) -> str:
    return (text or "").strip().lower()


def _is_sofa_or_seat(product: Product) -> bool:
    haystack = f"{_norm(product.get('category', ''))} {_norm(product.get('name', ''))}"
    return any(token in haystack for token in SOFA_TOKENS)


def _is_table(product: Product) -> bool:
    haystack = f"{_norm(product.get('category', ''))} {_norm(product.get('name', ''))}"
    return any(token in haystack for token in TABLE_TOKENS)


def _dedupe(products: list[Product]) -> list[Product]:
    seen_names: set[str] = set()
    result: list[Product] = []
    for product in products:
        name = (product.get("name") or "").strip()
        key = name.lower()
        if not name or key in seen_names:
            continue
        seen_names.add(key)
        result.append(product)
    return result


def _select_products(products: list[Product], sofas_target: int, tables_target: int) -> tuple[list[Product], int, int]:
    products = _dedupe(products)
    sofas: list[Product] = []
    tables: list[Product] = []

    for product in products:
        if len(sofas) < sofas_target and _is_sofa_or_seat(product):
            sofas.append(product)
            continue
        if len(tables) < tables_target and _is_table(product):
            tables.append(product)

        if len(sofas) >= sofas_target and len(tables) >= tables_target:
            break

    return sofas + tables, len(sofas), len(tables)


def _fetch_with_fallback(fetcher: Fetcher, limits: list[int]) -> list[Product]:
    best: list[Product] = []
    for limit in limits:
        products = fetcher(limit)
        if len(products) > len(best):
            best = products
        if len(best) >= limit:
            break
    return best


def _fetch_furn_targeted(sofas_target: int, tables_target: int) -> list[Product]:
    original_categories = list(scraper_module._FURN_CATEGORIES)
    try:
        scraper_module._FURN_CATEGORIES = [
            ("https://furn.nl/banken", "Banken"),
            ("https://furn.nl/stoelen", "Stoelen"),
        ]
        sofa_candidates = _fetch_with_fallback(fetch_furn_products, [20, 35])

        scraper_module._FURN_CATEGORIES = [("https://furn.nl/tafels", "Tafels")]
        table_candidates = _fetch_with_fallback(fetch_furn_products, [12, 20])
    finally:
        scraper_module._FURN_CATEGORIES = original_categories

    sofas, _, _ = _select_products(sofa_candidates, sofas_target, 0)
    tables, _, _ = _select_products(table_candidates, 0, tables_target)
    return _dedupe(sofas + tables)


def _fetch_meubels_targeted(sofas_target: int, tables_target: int) -> list[Product]:
    original_categories = list(scraper_module._MEUBELS_COM_CATEGORIES)
    try:
        scraper_module._MEUBELS_COM_CATEGORIES = [
            ("https://meubels.com/banken", "Banken"),
            ("https://meubels.com/stoelen", "Stoelen"),
        ]
        sofa_candidates = _fetch_with_fallback(fetch_meubels_com_products, [20, 35])

        scraper_module._MEUBELS_COM_CATEGORIES = [("https://meubels.com/tafels", "Tafels")]
        table_candidates = _fetch_with_fallback(fetch_meubels_com_products, [12, 20])
    finally:
        scraper_module._MEUBELS_COM_CATEGORIES = original_categories

    sofas, _, _ = _select_products(sofa_candidates, sofas_target, 0)
    tables, _, _ = _select_products(table_candidates, 0, tables_target)
    return _dedupe(sofas + tables)


def _fetch_ikea_targeted(sofas_target: int, tables_target: int) -> list[Product]:
    """Fetch IKEA products for sofas and tables."""
    sofa_url = "https://www.ikea.com/nl/nl/cat/hoekbanken-47388/"
    table_url = "https://www.ikea.com/nl/nl/cat/tafels-700675/"

    sofa_candidates = _fetch_with_fallback(
        lambda limit: fetch_ikea_products_from_url(sofa_url, "Sofas", limit), [20, 35]
    )
    table_candidates = _fetch_with_fallback(
        lambda limit: fetch_ikea_products_from_url(table_url, "Tables", limit), [5, 10]
    )

    sofas, _, _ = _select_products(sofa_candidates, sofas_target, 0)
    tables, _, _ = _select_products(table_candidates, 0, tables_target)
    return _dedupe(sofas + tables)


def seed_catalog(sofas_target: int, tables_target: int, clear_first: bool) -> None:
    if clear_first:
        clear_catalog_data()

    print("\n[Furn] Fetching targeted categories...")
    furn_selected = _fetch_furn_targeted(sofas_target, tables_target)
    furn_sofas, furn_tables = _select_products(furn_selected, sofas_target, tables_target)[1:]
    upsert_products("Furn", furn_selected)
    print(
        f"[Furn] Saved {len(furn_selected)} products "
        f"(sofas/seats: {furn_sofas}/{sofas_target}, tables: {furn_tables}/{tables_target})"
    )

    print("\n[Meubels.com] Fetching targeted categories...")
    meubels_selected = _fetch_meubels_targeted(sofas_target, tables_target)
    meubels_sofas, meubels_tables = _select_products(meubels_selected, sofas_target, tables_target)[1:]
    upsert_products("Meubels.com", meubels_selected)
    print(
        f"[Meubels.com] Saved {len(meubels_selected)} products "
        f"(sofas/seats: {meubels_sofas}/{sofas_target}, tables: {meubels_tables}/{tables_target})"
    )

    print("\n[IKEA] Fetching sofas and dining tables...")
    ikea_selected = _fetch_ikea_targeted(sofas_target, tables_target)
    ikea_sofas, ikea_tables = _select_products(ikea_selected, sofas_target, tables_target)[1:]
    upsert_products("IKEA", ikea_selected)
    print(
        f"[IKEA] Saved {len(ikea_selected)} products "
        f"(sofas/seats: {ikea_sofas}/{sofas_target}, tables: {ikea_tables}/{tables_target})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed catalog with per-retailer quotas.")
    parser.add_argument("--sofas", type=int, default=5, help="Sofas/seats per retailer")
    parser.add_argument("--tables", type=int, default=2, help="Tables per retailer")
    parser.add_argument(
        "--clear-first",
        action="store_true",
        help="Clear existing catalog data before seeding",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_catalog(sofas_target=args.sofas, tables_target=args.tables, clear_first=args.clear_first)

    rows = get_products()
    print(f"\nDone. Total products in DB: {len(rows)}")


if __name__ == "__main__":
    main()
