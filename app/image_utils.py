"""
image_utils.py — Image URL validation and fallback logic.

The dataset image URLs fall into three quality tiers:
  - Good:    Direct CDN/blog URLs (Unsplash, VnExpress, tourism CDNs)
  - Risky:   Bing/Google search thumbnail proxies — work today, break without warning
  - Bad:     Blank, literal "nan", single chars, or the known 40-way shared placeholder

At load time we reject bad URLs and replace them with local SVG fallbacks
(category-appropriate icons). Risky thumbnail URLs are passed through because
they often still load; the browser onerror handler catches any runtime failures.

The specific Unsplash URL shared by 40 unrelated places is treated as a
placeholder and replaced with an SVG fallback so every card shows a distinct
visual rather than the same green mountain photo 40 times.
"""

from __future__ import annotations

from typing import Mapping
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Fallback images (local SVGs) — one per label category
# ---------------------------------------------------------------------------

FALLBACK_BY_CATEGORY: dict[str, str] = {
    "adventure":  "/static/images/fallback/adventure.svg",
    "relax":      "/static/images/fallback/relax.svg",
    "rural":      "/static/images/fallback/rural.svg",
    "urban":      "/static/images/fallback/urban.svg",
    "mountain":   "/static/images/fallback/mountain.svg",
    "historical": "/static/images/fallback/historical.svg",
    "food":       "/static/images/fallback/food.svg",
    "nature":     "/static/images/fallback/nature.svg",
    "default":    "/static/images/fallback/default.svg",
}

# Label columns in priority order — used by primary_category_from_row.
# The first matching label becomes the fallback category for the place's image.
CATEGORY_PRIORITY: tuple[str, ...] = (
    "historical", "mountain", "nature", "food",
    "adventure", "relax", "urban", "rural",
)

# ---------------------------------------------------------------------------
# Known-bad URL patterns
# ---------------------------------------------------------------------------

# Literal string values that are definitely not URLs.
BAD_LITERAL_VALUES: frozenset[str] = frozenset(
    {"", "nan", "none", "null", "n/a", "na", "q", "-", "0", "false"}
)

# A single Unsplash landscape photo used as a placeholder for 40+ unrelated
# places in the dataset. Detect it by a stable prefix of the URL so variations
# in query-string parameters are also caught.
KNOWN_PLACEHOLDER_URL_PREFIXES: tuple[str, ...] = (
    "https://images.unsplash.com/photo-1528127269322-539801943592",
)

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def primary_category_from_row(row: Mapping[str, object]) -> str:
    """
    Return the best-matching label category key for a CSV/XLSX dataset row.

    Checks the 8 binary label columns (Adventure, Relax, …) in priority order
    and returns the first one that is set to 1. Falls back to "default" if none
    are set.
    """
    for key in CATEGORY_PRIORITY:
        column = key.capitalize()
        try:
            if int(row.get(column, 0)) == 1:
                return key
        except (TypeError, ValueError):
            continue
    return "default"


def fallback_image_for_category(category: str) -> str:
    """Return the local SVG fallback path for a given category key."""
    return FALLBACK_BY_CATEGORY.get(category, FALLBACK_BY_CATEGORY["default"])


def is_suspicious_image_url(raw_url: str) -> bool:
    """
    Return True when a URL should be replaced with a local SVG fallback.

    Rejects:
    - Blank / literal placeholder values ("nan", "q", etc.)
    - URLs containing spaces or unescaped quote characters
    - Non-HTTP(S) schemes (data: URIs, relative paths, etc.)
    - Missing hostname
    - The shared 40-way Unsplash placeholder used in the dataset
    """
    value = str(raw_url or "").strip()

    if value.lower() in BAD_LITERAL_VALUES:
        return True

    if any(ch in value for ch in ('"', "'", "<", ">", " ")):
        return True

    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        return True
    if not parsed.netloc:
        return True

    # Reject the known mass-duplicated placeholder URL.
    for prefix in KNOWN_PLACEHOLDER_URL_PREFIXES:
        if value.startswith(prefix):
            return True

    return False


def clean_image_url(raw_url: str, category: str) -> str:
    """
    Return a safe image URL for a place.

    Valid URLs (including Bing/CDN thumbnails) are passed through unchanged.
    The browser-side onerror handler in index.html catches any that fail at
    runtime and swaps in the appropriate SVG fallback automatically.
    """
    if is_suspicious_image_url(raw_url):
        return fallback_image_for_category(category)
    return str(raw_url).strip()
