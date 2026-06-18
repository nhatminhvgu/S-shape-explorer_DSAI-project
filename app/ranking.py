"""
ranking.py — Multi-signal ranker combining TF-IDF similarity, 8-label matching,
             rating, and location boost.

Scoring formula (weights vary by what the user provided):

  Case A  — query + preferences:   0.35·sem + 0.45·label + 0.20·rating
  Case B  — preferences only:      0.60·label + 0.40·rating
  Case C  — query only:            0.65·sem  + 0.35·rating
  Case D  — nothing provided:      0.50·rating + 0.50·sem  (fallback)

Location boost multipliers:
  exact match   ×1.50
  partial match ×1.25
  no match      ×0.70  (soft penalty to push off-location results down)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from app.models import Place, PlaceResult
from app.recommender import LABEL_MAP
from app.nlp_parser import REGION_MAP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label_score(place: Place, preferences: List[str]) -> Tuple[float, List[str]]:
    """
    Returns (score ∈ [0,1], matched_label_names).

    score = (# preferences matched by this place) / (# total preferences).
    """
    if not preferences:
        return 0.0, []

    matched: List[str] = []
    for pref in preferences:
        field = LABEL_MAP.get(pref)
        if field and getattr(place, field, 0) == 1:
            matched.append(pref)

    return len(matched) / len(preferences), matched


def _location_boost(place: Place, query_location: str) -> float:
    """
    Return a multiplier based on how well place.location matches the query location.
    
    Handles both specific province matches and regional matches:
    - Exact match:     ×1.50
    - Regional match:  ×1.30 (e.g., "North Vietnam" matches "Hanoi")
    - Partial match:   ×1.25 (substring match)
    - No match:        ×0.70 (soft penalty)
    - Region mismatch: ×0.50 (strong penalty when region specified but not matched)
    """
    if not query_location:
        return 1.0

    p_loc = place.location.lower().strip()
    q_loc = query_location.lower().strip()

    # Exact province match
    if p_loc == q_loc:
        return 1.50
    
    # Partial/substring match
    if q_loc in p_loc or p_loc in q_loc:
        return 1.25
    
    # Regional matching: check if query is a region (North/Central/South Vietnam)
    for region, provinces in REGION_MAP.items():
        region_query = f"{region} vietnam"
        if region_query in q_loc:
            # Check if place is in this region
            if any(prov.lower() in p_loc for prov in provinces):
                return 1.30  # Strong boost for regional match
            else:
                return 0.50  # Strong penalty for region mismatch
        # Also handle just "north" without "vietnam"
        elif q_loc == region:
            if any(prov.lower() in p_loc for prov in provinces):
                return 1.30
            else:
                return 0.50
    
    # No match - soft penalty to push off-location results down
    return 0.70


def _generate_explanation(
    place: Place,
    preferences: List[str],
    matched_labels: List[str],
    query_location: str,
) -> str:
    """
    Build a chatbot-style sentence explaining why this place is recommended.

    Example:
      "Based on your preference for Relax and Nature, I recommend Cam Ranh Long
       Beach in Khanh Hoa because it matches your Relax interest: Pristine beach
       with natural beauty, ideal for relaxing and swimming. (Rating: 4.8/5)"
    """
    # ── Opening ───────────────────────────────────────────────────────────
    if preferences and matched_labels:
        pref_str  = " and ".join(preferences)
        match_str = " and ".join(matched_labels)
        opening = (
            f"Based on your preference for {pref_str}, I recommend "
            f"{place.name} in {place.location} "
            f"because it matches your {match_str} interest"
        )
    elif preferences:
        pref_str = " and ".join(preferences)
        opening = (
            f"Based on your preference for {pref_str}, I recommend "
            f"{place.name} in {place.location}"
        )
    elif query_location and query_location.lower() in place.location.lower():
        opening = (
            f"I recommend {place.name} in {place.location}, "
            f"which matches your location preference"
        )
    else:
        opening = f"I recommend {place.name} in {place.location}"

    # ── Description ───────────────────────────────────────────────────────
    desc = place.description.strip()
    if desc:
        body = f": {desc}"
    else:
        body = ""

    # ── Rating ────────────────────────────────────────────────────────────
    rating_str = f" (Rating: {place.rating}/5)" if place.rating > 0 else ""

    return opening + body + rating_str + "."


def _filter_by_region(
    candidates: List[Tuple[Place, float]],
    query_location: str,
) -> List[Tuple[Place, float]]:
    """
    If query_location is a region (e.g., "North Vietnam"), filter candidates
    to only include places from that region. Otherwise return all candidates.
    """
    q_loc = query_location.lower().strip()
    
    for region, provinces in REGION_MAP.items():
        region_query = f"{region} vietnam"
        if region_query in q_loc or q_loc == region:
            # Filter: keep only places in this region
            filtered = [
                (place, score)
                for place, score in candidates
                if any(prov.lower() in place.location.lower() for prov in provinces)
            ]
            if filtered:
                logger.info(
                    "Filtered %d candidates to %d places in %s region",
                    len(candidates), len(filtered), region.capitalize()
                )
                return filtered
    
    # No region filter applied, return all candidates
    return candidates





# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rank(
    candidates: List[Tuple[Place, float]],
    preferences: List[str],
    query_location: str = "",
    top_k: int = 5,
    has_query: bool = True,
) -> List[PlaceResult]:
    """
    Re-rank TF-IDF candidates using label matching, rating, and location boost.

    Parameters
    ----------
    candidates     : (place, sem_score) from PlaceIndex.top_k_similar()
    preferences    : validated preference labels, e.g. ["Relax", "Nature"]
    query_location : extracted or user-provided location string
    top_k          : number of results to return
    has_query      : True if the user typed a text query (affects weights)
    """
    # Filter candidates by region if a region was specified
    candidates = _filter_by_region(candidates, query_location)
    
    results: List[PlaceResult] = []

    for place, sem_score in candidates:
        label_score, matched = _label_score(place, preferences)
        rating_score = (place.rating / 5.0) if place.rating > 0 else 0.3

        # ── Weighted combination ──────────────────────────────────────────
        has_prefs = bool(preferences)
        if has_prefs and has_query:
            combined = 0.35 * sem_score + 0.45 * label_score + 0.20 * rating_score
        elif has_prefs:
            combined = 0.60 * label_score + 0.40 * rating_score
        elif has_query:
            combined = 0.65 * sem_score + 0.35 * rating_score
        else:
            combined = 0.50 * sem_score + 0.50 * rating_score

        # ── Location boost ────────────────────────────────────────────────
        combined = min(1.0, combined * _location_boost(place, query_location))

        explanation = _generate_explanation(place, preferences, matched, query_location)

        results.append(PlaceResult(
            place=place,
            score=round(combined, 4),
            matched_labels=matched,
            explanation=explanation,
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]
