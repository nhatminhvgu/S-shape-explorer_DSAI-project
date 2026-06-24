"""
location_resolver.py — Location alias and nearby-place logic for Vietnam tourism.

Why this exists:
The dataset stores many famous destination names under their province. For example,
"Hoi An ancient town" has location "Quang Nam", so a raw filter for location
"Hoi An" will miss the correct rows unless we resolve aliases first.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Iterable

from app.models import Place


@dataclass(frozen=True)
class LocationIntent:
    """Resolved search terms for a user location query."""

    raw: str
    normalized: str
    primary_terms: tuple[str, ...]
    nearby_terms: tuple[str, ...] = ()


# Destination/city aliases → dataset province/location terms.
# primary_terms should be strict enough to represent the requested area.
# nearby_terms are used only as fallback/expansion after primary matches.
LOCATION_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "hoi an": {
        "primary": ("hoi an", "quang nam"),
        "nearby": ("da nang",),
    },
    "sapa": {
        "primary": ("sapa", "sa pa", "lao cai"),
        "nearby": ("ha giang", "yen bai"),
    },
    "sa pa": {
        "primary": ("sa pa", "sapa", "lao cai"),
        "nearby": ("ha giang", "yen bai"),
    },
    "ha long": {
        "primary": ("ha long", "halong", "quang ninh"),
        "nearby": ("hai phong",),
    },
    "halong": {
        "primary": ("halong", "ha long", "quang ninh"),
        "nearby": ("hai phong",),
    },
    "nha trang": {
        "primary": ("nha trang", "khanh hoa"),
        "nearby": ("ninh thuan", "phu yen"),
    },
    "da lat": {
        "primary": ("da lat", "dalat", "lam dong"),
        "nearby": ("dak lak", "binh thuan"),
    },
    "dalat": {
        "primary": ("dalat", "da lat", "lam dong"),
        "nearby": ("dak lak", "binh thuan"),
    },
    "mui ne": {
        "primary": ("mui ne", "binh thuan"),
        "nearby": ("ninh thuan",),
    },
    "phu quoc": {
        "primary": ("phu quoc", "kien giang"),
        "nearby": ("an giang", "can tho"),
    },
    "con dao": {
        "primary": ("con dao", "ba ria - vung tau", "vung tau"),
        "nearby": (),
    },
    "vung tau": {
        "primary": ("vung tau", "ba ria - vung tau"),
        "nearby": ("dong nai", "ho chi minh city", "saigon", "hcmc"),
    },
    "hcmc": {
        "primary": ("hcmc", "ho chi minh city", "saigon"),
        "nearby": ("binh duong", "dong nai", "vung tau"),
    },
    "saigon": {
        "primary": ("saigon", "ho chi minh city", "hcmc"),
        "nearby": ("binh duong", "dong nai", "vung tau"),
    },
}


def normalize_location_text(text: str) -> str:
    """Lowercase, strip accents, remove duplicate spaces."""
    without_accents = "".join(
        char for char in unicodedata.normalize("NFD", text.casefold())
        if unicodedata.category(char) != "Mn"
    )
    cleaned = re.sub(r"[^a-z0-9\s\-]", " ", without_accents)
    return re.sub(r"\s+", " ", cleaned).strip()


def _dedupe_terms(terms: Iterable[str]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for term in terms:
        norm = normalize_location_text(term)
        if norm:
            seen[norm] = None
    return tuple(seen.keys())


def resolve_location(location: str) -> LocationIntent:
    """
    Resolve a user location string into terms that can match the dataset.

    Example:
        "Hoi An" → primary terms: ("hoi an", "quang nam")
    """
    normalized = normalize_location_text(location)
    if not normalized:
        return LocationIntent(raw=location, normalized="", primary_terms=(), nearby_terms=())

    alias = LOCATION_ALIASES.get(normalized)
    if alias is None:
        return LocationIntent(
            raw=location,
            normalized=normalized,
            primary_terms=(normalized,),
            nearby_terms=(),
        )

    primary_terms = _dedupe_terms(alias["primary"])
    nearby_terms = _dedupe_terms(alias["nearby"])
    return LocationIntent(
        raw=location,
        normalized=normalized,
        primary_terms=primary_terms,
        nearby_terms=nearby_terms,
    )


def place_location_match_level(place: Place, intent: LocationIntent) -> int:
    """
    Return how strongly a place matches a location intent.

    3 = destination name match, e.g. query "Hoi An" → "Hoi An ancient town"
    2 = primary area/province match, e.g. query "Hoi An" → location "Quang Nam"
    1 = nearby area match, e.g. query "Hoi An" → location "Da Nang"
    0 = no location relationship
    """
    if not intent.normalized:
        return 0

    place_name = normalize_location_text(place.name)
    place_location = normalize_location_text(place.location)

    if intent.normalized and intent.normalized in place_name:
        return 3

    if any(term in place_location or term in place_name for term in intent.primary_terms):
        return 2

    if any(term in place_location or term in place_name for term in intent.nearby_terms):
        return 1

    return 0


def is_location_related(place: Place, location: str, include_nearby: bool = True) -> bool:
    intent = resolve_location(location)
    level = place_location_match_level(place, intent)
    if include_nearby:
        return level > 0
    return level >= 2


def is_location_only_query(raw_query: str, location: str) -> bool:
    """
    Detect queries whose intent is only a place name/location, not a category.

    Examples treated as location-only:
      - "Hoi An"
      - "in Hoi An"
      - "places near Hoi An"
      - "locations around Hoi An"

    Examples not treated as location-only:
      - "beach near Hoi An"
      - "food in Hoi An"
      - "historical places in Hoi An"
    """
    query_norm = normalize_location_text(raw_query)
    intent = resolve_location(location)
    if not query_norm or not intent.normalized:
        return False

    removable_terms = set(intent.primary_terms) | {intent.normalized}
    for term in sorted(removable_terms, key=len, reverse=True):
        query_norm = re.sub(r"\b" + re.escape(term) + r"\b", " ", query_norm)

    generic_words = {
        "in", "near", "around", "at", "to", "from", "the", "a", "an",
        "place", "places", "location", "locations", "destination", "destinations",
        "spot", "spots", "site", "sites", "area", "areas", "related",
        "recommend", "recommendation", "recommendations", "visit", "travel",
        "trip", "tour", "nearby", "close", "about",
    }
    remaining = [token for token in query_norm.split() if token not in generic_words]
    return len(remaining) == 0
