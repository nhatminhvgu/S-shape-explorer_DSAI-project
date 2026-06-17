"""
recommender.py — TF-IDF based similarity retrieval using pre-built pickle files.

Files used (project root):
  tfidf_model.pkl        → fitted TfidfVectorizer  (315 descriptions)
  tfidf_matrix (1).pkl   → sparse TF-IDF matrix shape (315, 1000)

Preference label mapping for the 8 tourism categories from the CSV.
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

# Canonical label names → Place model field names
LABEL_MAP: dict[str, str] = {
    "Adventure": "adventure",
    "Relax":     "relax",
    "Rural":     "rural",
    "Urban":     "urban",
    "Mountain":  "mountain",
    "Historical":"historical",
    "Food":      "food",
    "Nature":    "nature",
}

VALID_LABELS: List[str] = list(LABEL_MAP.keys())


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _find_exact_place_match(
    query: str,
    places: List[Place],
) -> Optional[Tuple[Place, float]]:
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


def _load_pkl(filename: str):
    path = os.path.join(ROOT_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required pkl file not found: {path}")
    logger.info("Loading %s …", filename)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suppress version mismatch warnings
        with open(path, "rb") as fh:
            return pickle.load(fh)


logger.info("Loading TF-IDF model and matrix …")
_tfidf_model = _load_pkl("tfidf_model.pkl")
_tfidf_matrix = _load_pkl("tfidf_matrix (1).pkl")
logger.info(
    "TF-IDF ready. Vocabulary: %d terms | Matrix shape: %s",
    len(_tfidf_model.vocabulary_),
    _tfidf_matrix.shape,
)


class PlaceIndex:
    """
    Wraps the pre-computed TF-IDF matrix and provides similarity retrieval.
    One PlaceIndex is created at startup and reused for every request.
    """

    def __init__(self, places: List[Place]) -> None:
        self.places = places
        logger.info("PlaceIndex initialised with %d places.", len(places))

    def top_k_similar(self, query: str, k: int = 50) -> List[Tuple[Place, float]]:
        """
        Return up to k (place, similarity_score) tuples ordered by descending
        TF-IDF cosine similarity to the query.

        If query is empty (preference-only request), every place gets a neutral
        score of 0.5 so the ranking stage can differentiate by label match.

        If the search text matches a destination name exactly (or nearly so),
        that place is returned first before the semantic fallback runs.
        """
        n = len(self.places)
        k = min(k, n)

        if not query.strip():
            return [(p, 0.5) for p in self.places]

        exact_match = _find_exact_place_match(query, self.places)
        if exact_match is not None:
            exact_place, exact_score = exact_match
            # Keep the exact match at the top, then fill the rest with semantic results.
            exact_results = [(exact_place, exact_score)]
            remaining = [
                (place, score)
                for place, score in self._semantic_candidates(query, k + 1)
                if place.id != exact_place.id
            ]
            return exact_results + remaining[: k - 1]

        return self._semantic_candidates(query, k)

    def _semantic_candidates(
        self,
        query: str,
        k: int,
    ) -> List[Tuple[Place, float]]:
        q_vec = _tfidf_model.transform([query.lower()])
        sims: np.ndarray = cosine_similarity(q_vec, _tfidf_matrix)[0]

        top_indices = np.argsort(sims)[::-1][:k]
        return [
            (self.places[i], float(sims[i]))
            for i in top_indices
            if i < len(self.places)
        ]
