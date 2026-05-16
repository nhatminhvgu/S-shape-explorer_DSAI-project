"""
data_loader.py — Loads the mock place dataset from JSON and provides lookup helpers.
"""

import json
import os
from typing import Dict, List, Optional
from app.models import Place

# ---------------------------------------------------------------------------
# Resolve the path relative to this file so it works from any working dir
# ---------------------------------------------------------------------------
_DATA_PATH = os.path.join(os.path.dirname(__file__), "places_data.json")


def load_places() -> List[Place]:
    """
    Read all places from the JSON dataset and return as validated Pydantic models.
    """
    with open(_DATA_PATH, "r", encoding="utf-8") as fh:
        raw: List[dict] = json.load(fh)
    return [Place(**item) for item in raw]


def build_place_index(places: List[Place]) -> Dict[str, Place]:
    """
    Build a dictionary keyed by place ID for O(1) lookups.
    """
    return {p.id: p for p in places}


# Module-level singletons — loaded once at import time.
PLACES: List[Place] = load_places()
PLACE_INDEX: Dict[str, Place] = build_place_index(PLACES)


def get_place_by_id(place_id: str) -> Optional[Place]:
    """Return a single Place or None if the ID is not found."""
    return PLACE_INDEX.get(place_id)
