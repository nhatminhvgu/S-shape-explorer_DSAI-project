"""
ranking.py — Multi-signal ranker for S-Shape Explorer.

Scoring formula (weights depend on what information is available):

  Has query + prefs + AI model:   0.30·sem + 0.30·label + 0.20·ai + 0.20·rating
  Has query + prefs:              0.35·sem + 0.45·label + 0.20·rating
  Has prefs + AI model:           0.45·label + 0.25·ai  + 0.30·rating
  Has prefs only:                 0.60·label + 0.40·rating
  Has query + AI model:           0.45·sem   + 0.30·ai  + 0.25·rating
  Has query only:                 0.65·sem   + 0.35·rating
  Nothing (fallback):             0.50·sem   + 0.50·rating

Concrete-term boost/penalty (applied before location boost):
  - Place mentions the queried term (e.g. "beach")    → +45% combined score
  - Place does NOT mention the queried term at all     → ×0.55 penalty
    (penalty is softened to ×0.70 when a location filter is active, because
     the dataset may have limited entries matching both the term and the area)

Location boost multipliers:
  Exact destination name match    ×2.00
  Same city/province              ×2.00
  Nearby area                     ×1.00
  Regional match (N/C/S Vietnam)  ×1.30
  Off-location (with constraint)  ×0.55
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from app.models import Place, PlaceResult
from app.recommender import LABEL_MAP
from app.nlp_parser import REGION_MAP
from app.location_resolver import place_location_match_level, resolve_location

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label scoring
# ---------------------------------------------------------------------------

def _label_score(place: Place, preferences: List[str]) -> Tuple[float, List[str]]:
    """
    Return (fraction_of_prefs_matched, list_of_matched_label_names).

    score = (number of preferences the place satisfies) / (total preferences).
    Example: prefs=["Food","Nature"], place has food=1, nature=0 → score=0.5
    """
    if not preferences:
        return 0.0, []
    matched = [pref for pref in preferences if getattr(place, LABEL_MAP[pref], 0) == 1]
    return len(matched) / len(preferences), matched


def _ai_intent_score(
    place: Place,
    label_probabilities: Optional[Dict[str, float]],
) -> float:
    """
    Score a place using the trained multi-label classifier's output.

    The classifier estimates the probability that the user's query belongs to
    each of the 8 label categories. Places tagged with the highest-probability
    categories receive a higher AI intent score.
    """
    if not label_probabilities:
        return 0.0
    active_probs = [
        float(label_probabilities.get(label, 0.0))
        for label, field in LABEL_MAP.items()
        if getattr(place, field, 0) == 1
    ]
    return sum(active_probs) / len(active_probs) if active_probs else 0.0


# ---------------------------------------------------------------------------
# Surface term matching — checks whether concrete query nouns appear in the
# place's name, description, location, or keywords.
# ---------------------------------------------------------------------------

# Synonym groups for concrete query terms.
# If the user asks for "beach", places with "sea", "seaside", "coast", etc.
# in their text still receive a positive surface score.
SURFACE_TERM_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "beach":       ("beach", "sea", "seaside", "coast", "coastal", "shore", "sand", "swim"),
    "sea":         ("sea", "beach", "coast", "ocean", "seaside", "shore"),
    "coast":       ("coast", "coastal", "beach", "sea", "shore"),
    "mountain":    ("mountain", "hill", "peak", "pass", "trek", "hike", "trail", "highland"),
    "waterfall":   ("waterfall", "fall", "cascade"),
    "cave":        ("cave", "grotto"),
    "lake":        ("lake",),
    "river":       ("river",),
    "island":      ("island", "archipelago", "islet"),
    "forest":      ("forest", "jungle", "reserve", "national park"),
    "temple":      ("temple",),
    "pagoda":      ("pagoda",),
    "market":      ("market",),
    "night market": ("night market", "market"),
    "street food": ("street food", "food", "cuisine", "dish", "specialty", "restaurant"),
    "village":     ("village",),
    "rice terrace": ("rice terrace", "terraced field", "rice field"),
    "rice field":  ("rice field", "field"),
    # Food-intent group — when the user types "food", non-food places should rank below
    # places that explicitly mention food/market/cuisine in their text.
    "food":        ("food", "eat", "restaurant", "cuisine", "market", "dish",
                    "specialty", "specialties", "dining", "culinary",
                    "floating market", "night market", "street food"),
    "restaurant":  ("restaurant", "food", "dining", "cuisine", "eat", "specialty"),
    "eat":         ("eat", "food", "restaurant", "dining", "cuisine"),
    "dining":      ("dining", "food", "restaurant", "cuisine", "eat"),
    "cuisine":     ("cuisine", "food", "restaurant", "eat", "dish", "specialty"),
}


def _surface_term_score(
    place: Place,
    query_terms: Optional[List[str]],
) -> float:
    """
    Return the fraction of concrete query terms that appear in the place's text.

    A term "matches" if any of its synonyms appears in the combined text of
    the place's name, location, description, and keywords.

    This prevents a generic Relax/Nature place from outranking an actual beach
    just because both carry the "Relax" or "Nature" label.
    """
    if not query_terms:
        return 0.0

    norm_terms = list(dict.fromkeys(t.casefold().strip() for t in query_terms if t.strip()))
    if not norm_terms:
        return 0.0

    place_text = " ".join([
        place.name, place.location, place.description, " ".join(place.keywords)
    ]).casefold()

    matched = 0
    for term in norm_terms:
        synonyms = SURFACE_TERM_SYNONYMS.get(term, (term,))
        for syn in synonyms:
            syn_norm = syn.casefold()
            if " " in syn_norm:
                found = syn_norm in place_text
            else:
                found = bool(re.search(r"\b" + re.escape(syn_norm) + r"\b", place_text))
            if found:
                matched += 1
                break

    return matched / len(norm_terms)


# ---------------------------------------------------------------------------
# Location boost
# ---------------------------------------------------------------------------

def _location_boost(place: Place, query_location: str) -> float:
    """
    Return a multiplier that rewards location-matching places and penalises
    off-location ones when the user specified a location constraint.

    Returning 1.0 for no location (no constraint) leaves scores unchanged.
    """
    if not query_location:
        return 1.0

    intent = resolve_location(query_location)
    match_level = place_location_match_level(place, intent)

    if match_level >= 2:
        return 2.00   # same city, province, or destination name
    if match_level == 1:
        return 1.00   # nearby area (useful fallback, no extra boost)

    p_loc = place.location.lower().strip()
    q_loc = query_location.lower().strip()

    # Regional queries (North/Central/South Vietnam)
    for region, provinces in REGION_MAP.items():
        region_query = f"{region} vietnam"
        if region_query in q_loc or q_loc == region:
            return 1.30 if any(prov.lower() in p_loc for prov in provinces) else 0.50

    # Concrete location was requested but this place is elsewhere.
    return 0.55


# ---------------------------------------------------------------------------
# Candidate filtering
# ---------------------------------------------------------------------------

def _filter_by_region(
    candidates: List[Tuple[Place, float]],
    query_location: str,
) -> List[Tuple[Place, float]]:
    """
    Hard-filter candidates when the query names a specific region or place.

    For regional queries ("North Vietnam", "South Vietnam", "Central Vietnam")
    only places in the named provinces are kept.

    For specific city/destination queries, primary-area matches are returned
    first, followed by nearby-area matches as fallback.
    """
    q_loc = query_location.lower().strip()

    # Regional filter (North/Central/South Vietnam)
    for region, provinces in REGION_MAP.items():
        region_query = f"{region} vietnam"
        if region_query in q_loc or q_loc == region:
            filtered = [
                (p, s) for p, s in candidates
                if any(prov.lower() in p.location.lower() for prov in provinces)
            ]
            if filtered:
                logger.info(
                    "Region filter: %d → %d places in %s Vietnam",
                    len(candidates), len(filtered), region.capitalize(),
                )
                return filtered

    # City / destination filter
    if query_location:
        intent = resolve_location(query_location)
        primary = [(p, s) for p, s in candidates if place_location_match_level(p, intent) >= 2]
        nearby  = [(p, s) for p, s in candidates if place_location_match_level(p, intent) == 1]
        if primary:
            logger.info(
                "Location filter %r: %d primary + %d nearby matches",
                query_location, len(primary), len(nearby),
            )
            return primary + nearby

    return candidates


# ---------------------------------------------------------------------------
# Explanation generator
# ---------------------------------------------------------------------------

def _generate_explanation(
    place: Place,
    preferences: List[str],
    matched_labels: List[str],
    query_location: str,
    low_confidence: bool = False,
) -> str:
    """
    Generate a one-sentence explanation for why this place is recommended.

    When confidence is low (no label match found for the queried category in
    the requested location), the explanation is honest about the limitation.
    """
    if low_confidence and preferences and not matched_labels:
        pref_str = " and ".join(preferences)
        return (
            f"No exact {pref_str} match found in {query_location or 'this area'}. "
            f"{place.name} in {place.location} is the closest available result."
        )

    if preferences and matched_labels:
        pref_str  = " and ".join(preferences)
        match_str = " and ".join(matched_labels)
        opening = (
            f"Recommended for {pref_str}: {place.name} in {place.location} "
            f"matches your {match_str} interest"
        )
    elif preferences:
        opening = (
            f"Based on your interest in {' and '.join(preferences)}, "
            f"I recommend {place.name} in {place.location}"
        )
    elif query_location and query_location.lower() in place.location.lower():
        opening = (
            f"{place.name} in {place.location} is a popular destination "
            f"in your selected area"
        )
    else:
        opening = f"I recommend {place.name} in {place.location}"

    desc = place.description.strip()
    rating_str = f" (Rating: {place.rating}/5)" if place.rating > 0 else ""
    return opening + (f": {desc}" if desc else "") + rating_str + "."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rank(
    candidates: List[Tuple[Place, float]],
    preferences: List[str],
    query_location: str = "",
    top_k: int = 5,
    has_query: bool = True,
    label_probabilities: Optional[Dict[str, float]] = None,
    query_terms: Optional[List[str]] = None,
) -> List[PlaceResult]:
    """
    Re-rank TF-IDF candidates using label matching, rating, and location.

    Parameters
    ----------
    candidates        : (place, sem_score) pairs from PlaceIndex.top_k_similar()
    preferences       : validated label names, e.g. ["Relax", "Nature"]
    query_location    : extracted or user-provided location string
    top_k             : number of results to return
    has_query         : True when the user typed a text query
    label_probabilities : ML classifier output probabilities for each label
    query_terms       : concrete nouns detected in the query, e.g. ["beach"]
    """
    candidates = _filter_by_region(candidates, query_location)

    # Detect whether the filtered candidate pool is likely to produce poor matches.
    # We use this to soften the surface-term penalty (×0.55 instead of ×0.40)
    # so we still return something useful instead of near-zero scores.
    location_active = bool(query_location)

    results: List[PlaceResult] = []

    for place, sem_score in candidates:
        label_score, matched = _label_score(place, preferences)
        ai_score    = _ai_intent_score(place, label_probabilities)
        rating_score = (place.rating / 5.0) if place.rating > 0 else 0.3
        surface_score = _surface_term_score(place, query_terms)

        # ── Select weight formula ─────────────────────────────────────────
        has_prefs = bool(preferences)
        has_ai    = bool(label_probabilities) and any(label_probabilities.values())

        if has_prefs and has_query and has_ai:
            combined = 0.30*sem_score + 0.30*label_score + 0.20*ai_score + 0.20*rating_score
        elif has_prefs and has_query:
            combined = 0.35*sem_score + 0.45*label_score + 0.20*rating_score
        elif has_prefs and has_ai:
            combined = 0.45*label_score + 0.25*ai_score + 0.30*rating_score
        elif has_prefs:
            combined = 0.60*label_score + 0.40*rating_score
        elif has_query and has_ai:
            combined = 0.45*sem_score + 0.30*ai_score + 0.25*rating_score
        elif has_query:
            combined = 0.65*sem_score + 0.35*rating_score
        else:
            combined = 0.50*sem_score + 0.50*rating_score

        # ── Concrete term boost/penalty ───────────────────────────────────
        # When the user specifies a concrete noun ("beach", "food", etc.),
        # places that explicitly mention that noun in their text are boosted.
        # Places that do not mention it are penalised — less harshly when a
        # location filter is active (the dataset may simply lack entries that
        # match both the term and the area).
        if query_terms:
            if surface_score <= 0:
                # Softer penalty when location filter is active, because there
                # may be no places matching both the term and the location.
                penalty = 0.55 if location_active else 0.40
                combined *= penalty
            else:
                combined = min(1.0, combined * (1.0 + 0.45 * surface_score))

        # ── Explicit preference zero-match penalty ────────────────────────
        # When the user named a label category but no results match it (AND
        # there are no query_terms — which already handled the same case above),
        # suppress non-matching places so they don't crowd out correct results.
        if has_prefs and label_score == 0 and not query_terms:
            combined *= 0.45

        # ── Location boost ────────────────────────────────────────────────
        combined = min(1.0, combined * _location_boost(place, query_location))

        # Low confidence: place has none of the preferred labels
        low_conf = bool(preferences) and label_score == 0

        explanation = _generate_explanation(
            place, preferences, matched, query_location, low_confidence=low_conf
        )

        results.append(PlaceResult(
            place=place,
            score=round(combined, 4),
            matched_labels=matched,
            explanation=explanation,
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]
