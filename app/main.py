"""
main.py — FastAPI application entry point for S-Shape Explorer.

Endpoints:
  POST /recommend  → free-text query + preferences → ranked destinations
  GET  /place/{id} → single place by ID
  POST /feedback   → thumbs up/down; updates live rating
  GET  /health     → readiness probe
  GET  /           → serves the travel discovery UI (app/static/index.html)
"""

from __future__ import annotations

import logging
import unicodedata
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.data_loader import PLACES, get_place_by_id
from app.models import Place, RecommendRequest, RecommendResponse
from app.ml_intent import infer_preferences
from app.nlp_parser import parse_query, preferences_from_parsed_query
from app.location_resolver import is_location_only_query
from app.ranking import rank
from app.recommender import PlaceIndex, VALID_LABELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rating dynamics
# ---------------------------------------------------------------------------

# Each place is pre-seeded with this many "virtual" votes from its dataset
# rating. This prevents a single real like/dislike from swinging the score.
_BASE_VOTE_COUNT = 1000
_base_ratings: Dict[str, float] = {}
_place_index: Optional[PlaceIndex] = None


def _compute_dynamic_rating(base_rating: float, likes: int, dislikes: int) -> float:
    """
    Blend the dataset rating with live user votes using a dampened average.

    The _BASE_VOTE_COUNT pre-existing votes anchor the rating so early feedback
    has a small but visible effect, while sustained feedback gradually shifts it.
    """
    total = _BASE_VOTE_COUNT + likes + dislikes
    weighted = base_rating * _BASE_VOTE_COUNT + 5.0 * likes + 1.0 * dislikes
    return round(min(5.0, max(1.0, weighted / total)), 2)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _place_index, _base_ratings
    logger.info("Dataset: %d tourism places loaded.", len(PLACES))
    _base_ratings = {p.id: p.rating for p in PLACES}
    _place_index = PlaceIndex(PLACES)
    logger.info("Recommendation engine ready.")
    yield


app = FastAPI(
    title="S-Shape Explorer — Vietnam Travel Recommender",
    description=(
        "AI-powered travel recommendation for Vietnam using "
        "TF-IDF similarity, multi-label classification, and 8 preference categories."
    ),
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["UI"])
def root():
    return FileResponse(static_dir / "index.html")


@app.get("/health", tags=["Meta"])
def health() -> dict:
    return {"status": "ok", "places_loaded": len(PLACES)}


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
def recommend(body: RecommendRequest) -> RecommendResponse:
    query = (body.query or "").strip()

    parsed_query = parse_query(query) if query else {
        "category": "", "mood": [], "budget": "",
        "purpose": [], "location": "", "tags": [],
    }

    explicit_preferences = [p for p in (body.preferences or []) if p in VALID_LABELS]
    parser_preferences   = preferences_from_parsed_query(parsed_query)

    location = (body.location or "").strip() or parsed_query.get("location", "")

    location_only = is_location_only_query(query, location) if query and location else False

    if query and not location_only:
        inferred_preferences, label_probabilities = infer_preferences(query)
    else:
        inferred_preferences, label_probabilities = [], {}

    # Rule-based parser preferences take priority over ML-inferred ones because
    # they are more reliable for clear keyword queries on a small dataset.
    ml_preferences = [] if parser_preferences else inferred_preferences
    preferences = list(dict.fromkeys(
        explicit_preferences + parser_preferences + ml_preferences
    ))
    ranking_label_probabilities = {} if parser_preferences else label_probabilities

    logger.info(
        "POST /recommend | query=%r | prefs=%s | location=%r | top_k=%d",
        query, preferences, location, body.top_k,
    )

    # Use the full dataset when a location filter is active so no candidates
    # are dropped before the location boost has a chance to act on them.
    pool_size = (
        len(PLACES)
        if (not query or location)
        else min(len(PLACES), max(body.top_k * 12, 80))
    )

    candidates = _place_index.top_k_similar(query, k=pool_size)

    ranked = rank(
        candidates=candidates,
        preferences=preferences,
        query_location=location,
        top_k=body.top_k,
        has_query=bool(query),
        label_probabilities=ranking_label_probabilities,
        query_terms=parsed_query.get("tags", []),
    )

    # Detect low-confidence situations:
    # The top result has no matching label AND the user asked for a specific category.
    low_confidence_note = ""
    if (
        preferences
        and location
        and ranked
        and ranked[0].matched_labels == []
    ):
        pref_str = " / ".join(preferences)
        low_confidence_note = (
            f"No {pref_str} places found in '{location}'. "
            f"Showing the closest available matches in the area instead."
        )
        logger.info("Low confidence — %s", low_confidence_note)

    logger.info("Returning %d recommendations.", len(ranked))

    return RecommendResponse(
        query=query,
        selected_preferences=preferences,
        inferred_preferences=inferred_preferences,
        ai_label_probabilities=label_probabilities,
        recommendations=ranked,
        low_confidence_note=low_confidence_note,
    )


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


@app.get("/place/search", response_model=Place, tags=["Places"])
def search_place_by_name(q: str) -> Place:
    """Find the best-matching place by name, ignoring diacritics and case."""
    q_norm = _strip_accents(q.casefold().strip())
    best: Optional[Place] = None
    best_score = 0
    for place in PLACES:
        name_norm = _strip_accents(place.name.casefold().strip())
        if q_norm == name_norm:
            return place
        if q_norm in name_norm or name_norm in q_norm:
            score = len(set(q_norm.split()) & set(name_norm.split()))
            if score > best_score:
                best_score = score
                best = place
    if best:
        return best
    raise HTTPException(status_code=404, detail=f"No place found for '{q}'.")


@app.get("/place/{place_id}", response_model=Place, tags=["Places"])
def get_place(place_id: str) -> Place:
    place = get_place_by_id(place_id)
    if place is None:
        raise HTTPException(status_code=404, detail=f"Place '{place_id}' not found.")
    return place


class FeedbackRequest(BaseModel):
    place_id: str
    feedback: str   # "like" | "dislike"
    user_id: Optional[str] = None


@app.post("/feedback", tags=["Feedback"])
def submit_feedback(body: FeedbackRequest) -> dict:
    """
    Record a thumbs up or thumbs down for a place.

    The vote is incorporated into the place's live rating using a dampened
    weighted average. The updated rating immediately affects future ranking
    because ranking.py reads place.rating at query time.
    """
    if body.feedback not in ("like", "dislike"):
        raise HTTPException(status_code=422, detail="feedback must be 'like' or 'dislike'.")

    place = get_place_by_id(body.place_id)
    if place is None:
        raise HTTPException(status_code=404, detail=f"Place '{body.place_id}' not found.")

    if body.feedback == "like":
        place.likes += 1
    else:
        place.dislikes += 1

    base_rating = _base_ratings.get(body.place_id, place.rating)
    place.rating = _compute_dynamic_rating(base_rating, place.likes, place.dislikes)
    vote_count = _BASE_VOTE_COUNT + place.likes + place.dislikes

    logger.info(
        "Feedback %r on %r — likes=%d, dislikes=%d, rating=%.2f",
        body.feedback, body.place_id, place.likes, place.dislikes, place.rating,
    )

    return {
        "status": "ok",
        "place_id": body.place_id,
        "feedback": body.feedback,
        "likes": place.likes,
        "dislikes": place.dislikes,
        "rating": place.rating,
        "vote_count": vote_count,
    }
