"""
recommender.py — Embedding + similarity core.

Responsibilities:
  1. Load the all-MiniLM-L6-v2 sentence-transformer model (once, at startup).
  2. Pre-compute embeddings for all place descriptions.
  3. Expose encode() for query embedding and top_k_similar() for retrieval.

We use cosine similarity from scikit-learn (no FAISS dependency needed for
100 places; swap in faiss for 10k+ places).
"""

from __future__ import annotations

import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple

from app.models import Place

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model — loaded once when the module is first imported
# ---------------------------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"

logger.info("Loading sentence-transformer model: %s", MODEL_NAME)
_model: SentenceTransformer = SentenceTransformer(MODEL_NAME)
logger.info("Model loaded.")


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _place_to_text(place: Place) -> str:
    """
    Build a rich text representation of a place for embedding.
    Concatenating name + category + tags + description gives the model
    maximum signal about the place's identity and qualities.
    """
    tags_str = ", ".join(place.tags)
    return (
        f"{place.name}. Category: {place.category}. "
        f"Tags: {tags_str}. {place.description} "
        f"Price: {place.price_level}. Rating: {place.rating}."
    )


def encode(texts: List[str]) -> np.ndarray:
    """
    Encode a list of strings into L2-normalised embedding vectors.

    Returns shape (N, embedding_dim) float32 array.
    """
    embeddings = _model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine sim == dot product after L2-norm
        show_progress_bar=False,
    )
    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# Place index — pre-computed at import time
# ---------------------------------------------------------------------------

class PlaceIndex:
    """
    Holds pre-computed place embeddings and supports fast cosine retrieval.
    """

    def __init__(self, places: List[Place]) -> None:
        self.places: List[Place] = places
        logger.info("Embedding %d place descriptions…", len(places))
        place_texts = [_place_to_text(p) for p in places]
        self.embeddings: np.ndarray = encode(place_texts)  # (N, D)
        logger.info("Place embeddings ready. Shape: %s", self.embeddings.shape)

    def query_embedding(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns shape (1, D)."""
        return encode([query])

    def top_k_similar(
        self, query: str, k: int = 10
    ) -> List[Tuple[Place, float]]:
        """
        Compute cosine similarity between the query and every place embedding.

        Returns the top-k (place, similarity_score) tuples ordered descending.
        """
        q_emb = self.query_embedding(query)                # (1, D)
        sims: np.ndarray = cosine_similarity(q_emb, self.embeddings)[0]  # (N,)

        # argsort descending — take top k+buffer so ranker has options
        top_indices = np.argsort(sims)[::-1][:k]

        return [(self.places[i], float(sims[i])) for i in top_indices]
