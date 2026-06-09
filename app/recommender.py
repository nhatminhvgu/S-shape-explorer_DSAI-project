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
import warnings
from typing import List, Tuple

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
        """
        n = len(self.places)
        k = min(k, n)

        if not query.strip():
            return [(p, 0.5) for p in self.places]

        q_vec = _tfidf_model.transform([query.lower()])
        sims: np.ndarray = cosine_similarity(q_vec, _tfidf_matrix)[0]

        top_indices = np.argsort(sims)[::-1][:k]
        return [
            (self.places[i], float(sims[i]))
            for i in top_indices
            if i < n
        ]
