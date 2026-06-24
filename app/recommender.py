"""
recommender.py — TF-IDF similarity retrieval using pre-built pickle files.

Pre-computed files (project root):
  tfidf_model.pkl         → fitted TfidfVectorizer (315 place descriptions)
  tfidf_matrix (1).pkl    → sparse TF-IDF matrix, shape (315, 1000)

These were created by train_label_model.py and are checked in to the repo so
the app starts without retraining. The matrix rows align 1-to-1 with the
places in the CSV dataset.
"""

from __future__ import annotations

import logging
import os
import pickle
import re
import warnings
from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.models import Place

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Canonical label name → Place model field name
LABEL_MAP: dict[str, str] = {
    "Adventure": "adventure",
    "Relax":     "relax",
    "Rural":     "rural",
    "Urban":     "urban",
    "Mountain":  "mountain",
    "Historical": "historical",
    "Food":      "food",
    "Nature":    "nature",
}

VALID_LABELS: List[str] = list(LABEL_MAP.keys())

# ---------------------------------------------------------------------------
# TF-IDF query pre-processing
# ---------------------------------------------------------------------------

# Words that appear in almost every document when searching Vietnam tourism.
# Keeping them in the TF-IDF query would boost generic results at the expense
# of specific ones. Region words (north, central, south) are intentionally
# kept because they provide real location signal.
TFIDF_QUERY_STOPWORDS: frozenset[str] = frozenset({
    "vietnam", "viet", "nam", "vn",
    "tourism", "tourist", "travel", "trip", "destination", "destinations",
    "place", "places", "location", "locations", "spot", "spots",
    "recommend", "recommendation", "recommendations",
    "in", "at", "to", "from", "for", "of", "the", "a", "an", "and",
    "want", "looking", "find", "show", "tell", "give", "suggest",
})

# Surface-level synonym expansions injected into the TF-IDF query.
# When the user types "sea", the vectoriser vocabulary may not contain that
# exact token, but "beach", "coast", and "ocean" often appear in descriptions
# of coastal destinations. Expanding the query improves recall.
QUERY_SYNONYM_EXPANSION: dict[str, tuple[str, ...]] = {
    "sea":       ("beach", "coast", "ocean", "seaside"),
    "coast":     ("beach", "sea", "ocean", "seaside"),
    "ocean":     ("sea", "beach", "coast"),
    "highland":  ("mountain", "hill", "upland", "plateau"),
    "trekking":  ("trek", "hiking", "trail", "mountain"),
    "hiking":    ("hike", "trek", "trail", "mountain"),
    "island":    ("island", "archipelago", "islet"),
    "cuisine":   ("food", "dish", "eat", "restaurant", "specialty"),
    "culinary":  ("food", "cuisine", "dish", "eat"),
    "eat":       ("food", "cuisine", "restaurant"),
    "dining":    ("food", "restaurant", "cuisine"),
}


def _normalise_search_query(query: str) -> str:
    """
    Prepare a user query for TF-IDF cosine similarity retrieval.

    Steps:
    1. Lowercase
    2. Collapse "Viet Nam" → "vietnam"
    3. Normalise typo/plural forms (beachs → beach, mountains → mountain)
    4. Remove generic stopwords that appear in all Vietnam tourism docs
    5. Inject synonym expansions for terms with limited vocabulary coverage
    """
    cleaned = query.casefold()
    cleaned = re.sub(r"\bviet\s+nam\b", "vietnam", cleaned)
    # Plural / typo normalisation
    cleaned = re.sub(r"\bbeach(?:es|s)?\b", "beach", cleaned)
    cleaned = re.sub(r"\bmountains?\b",     "mountain", cleaned)
    cleaned = re.sub(r"\bislands?\b",       "island", cleaned)
    cleaned = re.sub(r"\bwaterfalls?\b",    "waterfall", cleaned)
    cleaned = re.sub(r"\bseas?\b",          "sea", cleaned)
    cleaned = re.sub(r"\boceans?\b",        "sea", cleaned)
    cleaned = re.sub(r"\bcoasts?\b",        "coast", cleaned)
    # Strip non-alphanumeric characters
    cleaned = re.sub(r"[^a-z0-9\s\-]", " ", cleaned)

    tokens = [
        t for t in re.sub(r"\s+", " ", cleaned).strip().split()
        if t not in TFIDF_QUERY_STOPWORDS
    ]

    # Synonym expansion — add extra tokens so the vectoriser finds more relevant rows
    extra: list[str] = []
    for token in tokens:
        for synonym in QUERY_SYNONYM_EXPANSION.get(token, ()):
            if synonym not in tokens and synonym not in extra:
                extra.append(synonym)

    return " ".join(tokens + extra)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _find_exact_place_match(
    query: str,
    places: List[Place],
) -> Optional[Tuple[Place, float]]:
    """
    Return (place, 1.0) if the query is an exact or near-exact place name.

    This ensures that typing "Ha Long Bay" returns Ha Long Bay first, before
    any semantic similarity re-ordering.
    """
    cleaned_query = _normalize_text(query)
    if not cleaned_query:
        return None
    for place in places:
        place_name = _normalize_text(place.name)
        if (
            cleaned_query == place_name
            or cleaned_query in place_name
            or place_name in cleaned_query
        ):
            return place, 1.0
    return None


# ---------------------------------------------------------------------------
# Load pre-computed TF-IDF artefacts once at module import time
# ---------------------------------------------------------------------------

def _load_pkl(filename: str):
    path = os.path.join(ROOT_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    logger.info("Loading %s …", filename)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suppress sklearn version warnings
        with open(path, "rb") as fh:
            return pickle.load(fh)


logger.info("Loading TF-IDF artefacts …")
_tfidf_model  = _load_pkl("tfidf_model.pkl")
_tfidf_matrix = _load_pkl("tfidf_matrix (1).pkl")
logger.info(
    "TF-IDF ready — vocabulary: %d terms | matrix: %s",
    len(_tfidf_model.vocabulary_),
    _tfidf_matrix.shape,
)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class PlaceIndex:
    """
    Wraps the pre-computed TF-IDF matrix and provides similarity retrieval.

    One instance is created at startup and reused for every request.
    Each call to top_k_similar() transforms the query and computes cosine
    similarity against all 315 place descriptions.
    """

    def __init__(self, places: List[Place]) -> None:
        self.places = places
        logger.info("PlaceIndex initialised with %d places.", len(places))

    def top_k_similar(self, query: str, k: int = 50) -> List[Tuple[Place, float]]:
        """
        Return up to k (place, similarity_score) pairs sorted by descending score.

        - Empty query → every place gets a neutral score of 0.5 so the ranking
          stage can differentiate purely by label match and rating.
        - Exact name match → that place is pinned at position 0 with score 1.0,
          followed by semantic results.
        """
        n = len(self.places)
        k = min(k, n)

        if not query.strip():
            return [(p, 0.5) for p in self.places]

        exact = _find_exact_place_match(query, self.places)
        if exact is not None:
            exact_place, exact_score = exact
            rest = [
                (p, s)
                for p, s in self._semantic_candidates(query, k + 1)
                if p.id != exact_place.id
            ]
            return [(exact_place, exact_score)] + rest[: k - 1]

        return self._semantic_candidates(query, k)

    def _semantic_candidates(
        self,
        query: str,
        k: int,
    ) -> List[Tuple[Place, float]]:
        """Run TF-IDF cosine similarity and return top-k (place, score) pairs."""
        search_query = _normalise_search_query(query) or query.lower()
        q_vec = _tfidf_model.transform([search_query])
        sims: np.ndarray = cosine_similarity(q_vec, _tfidf_matrix)[0]
        top_indices = np.argsort(sims)[::-1][:k]
        return [
            (self.places[i], float(sims[i]))
            for i in top_indices
            if i < len(self.places)
        ]
