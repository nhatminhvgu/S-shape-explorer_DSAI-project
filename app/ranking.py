"""
ranking.py — Multi-signal weighted ranker.

Takes the raw similarity candidates from the recommender and re-ranks them
using a configurable weighted combination of:
  - semantic similarity  (embedding cosine score)
  - rating               (normalised 1-5 star rating)
  - distance             (if user lat/lon provided, else skipped)
  - popularity           (normalised 1-100 score)
  - preference boost     (learned from user feedback, per-tag weights)

The weights dict must sum to 1.0; if 'distance' has weight > 0 but no
user coordinates are provided, its weight is redistributed proportionally
to the remaining signals.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from app.models import ParsedQuery, Place, PlaceResult

# ---------------------------------------------------------------------------
# Default weights
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS: Dict[str, float] = {
    "semantic":    0.50,
    "rating":      0.20,
    "popularity":  0.20,
    "distance":    0.10,
}

# Rating scale constants
_RATING_MIN = 1.0
_RATING_MAX = 5.0

# Popularity scale constants
_POP_MIN = 0.0
_POP_MAX = 100.0

# Budget compatibility: query budget → acceptable place price levels
_BUDGET_COMPAT: Dict[str, List[str]] = {
    "cheap":     ["cheap"],
    "moderate":  ["cheap", "moderate"],
    "expensive": ["cheap", "moderate", "expensive"],
}

# How much to penalise incompatible budget (multiplied onto final score)
_BUDGET_PENALTY = 0.60


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _norm(value: float, min_val: float, max_val: float) -> float:
    """Min-max normalise value to [0, 1]."""
    span = max_val - min_val
    if span == 0:
        return 0.5
    return max(0.0, min(1.0, (value - min_val) / span))


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in km between two (lat, lon) points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _distance_score(place: Place, user_lat: Optional[float], user_lon: Optional[float]) -> float:
    """
    Return a proximity score in [0, 1] where 1 = very close, 0 = far away.
    Returns 0.5 (neutral) when no user coordinates or place coordinates are available.
    """
    if user_lat is None or user_lon is None:
        return 0.5
    if place.lat is None or place.lon is None:
        return 0.5
    dist_km = _haversine_km(user_lat, user_lon, place.lat, place.lon)
    # Assume anything ≤ 0.5 km is "very close" (score=1) and ≥ 10 km is "far" (score=0)
    return max(0.0, 1.0 - dist_km / 10.0)


def _preference_boost(
    place: Place,
    user_prefs: Dict[str, int],
) -> float:
    """
    Return an additive boost in [-0.15, +0.15] based on liked/disliked tags.

    user_prefs maps tag → cumulative weight (+N liked, -N disliked).
    """
    if not user_prefs:
        return 0.0
    total = 0.0
    for tag in place.tags:
        total += user_prefs.get(tag, 0)
    # Clamp to avoid dominating the score
    return max(-0.15, min(0.15, total * 0.02))


def _resolve_weights(
    weights: Optional[Dict[str, float]],
    has_distance: bool,
) -> Dict[str, float]:
    """
    Validate and normalise the weight dict.
    If distance weight > 0 but no user location is available, redistribute
    that weight proportionally among the other signals.
    """
    w = dict(weights) if weights else dict(DEFAULT_WEIGHTS)

    # Fill missing keys with 0
    for key in DEFAULT_WEIGHTS:
        w.setdefault(key, 0.0)

    if not has_distance and w.get("distance", 0) > 0:
        extra = w.pop("distance")
        others = {k: v for k, v in w.items() if k != "distance"}
        total_others = sum(others.values()) or 1.0
        for k in others:
            w[k] += extra * (others[k] / total_others)
        w["distance"] = 0.0

    # Normalise so weights sum to 1
    total = sum(w.values()) or 1.0
    return {k: v / total for k, v in w.items()}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rank(
    candidates: List[Tuple[Place, float]],
    parsed: ParsedQuery,
    weights: Optional[Dict[str, float]] = None,
    user_lat: Optional[float] = None,
    user_lon: Optional[float] = None,
    user_prefs: Optional[Dict[str, int]] = None,
    top_k: int = 5,
) -> List[PlaceResult]:
    """
    Re-rank similarity candidates using multi-signal weighted scoring.

    Parameters
    ----------
    candidates  : list of (Place, cosine_similarity) from the recommender
    parsed      : structured query intent from the NLP parser
    weights     : optional override for signal weights (must approximately sum to 1)
    user_lat/lon: optional user GPS coordinates for distance scoring
    user_prefs  : per-tag feedback weights accumulated from past feedback
    top_k       : number of results to return

    Returns
    -------
    List of PlaceResult sorted by descending final score, length ≤ top_k.
    """
    has_distance = user_lat is not None and user_lon is not None
    w = _resolve_weights(weights, has_distance)
    prefs = user_prefs or {}

    results: List[PlaceResult] = []

    for place, similarity in candidates:
        # --- Individual normalised signal scores ---
        sem_score  = float(similarity)                                          # already in [0,1]
        rat_score  = _norm(place.rating, _RATING_MIN, _RATING_MAX)
        pop_score  = _norm(float(place.popularity), _POP_MIN, _POP_MAX)
        dist_score = _distance_score(place, user_lat, user_lon)

        # --- Weighted combination ---
        combined = (
            w["semantic"]   * sem_score +
            w["rating"]     * rat_score +
            w["popularity"] * pop_score +
            w["distance"]   * dist_score
        )

        # --- Category hard filter (soft penalty, not hard remove) ---
        if parsed.category and place.category != parsed.category:
            combined *= 0.80

        # --- Location boost — prioritise places in the queried city/province ---
        if parsed.location:
            place_city = place.city.lower().strip()
            query_loc  = parsed.location.lower().strip()
            if place_city == query_loc:
                combined *= 1.40       # exact province/city match: big boost
            elif query_loc in place_city or place_city in query_loc:
                combined *= 1.20       # partial match: moderate boost
            else:
                combined *= 0.70       # wrong location: penalise

        # --- Budget compatibility penalty ---
        if parsed.budget:
            acceptable = _BUDGET_COMPAT.get(parsed.budget, [place.price_level])
            if place.price_level not in acceptable:
                combined *= _BUDGET_PENALTY

        # --- Preference boost from feedback history ---
        combined += _preference_boost(place, prefs)

        # Clamp final score to [0, 1]
        final_score = max(0.0, min(1.0, combined))

        results.append(PlaceResult(
            place=place,
            score=round(final_score, 4),
            similarity=round(sem_score, 4),
        ))

    # Sort descending by final score, then return top_k
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]
