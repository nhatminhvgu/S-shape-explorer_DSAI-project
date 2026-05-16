"""
main.py — FastAPI application entry point.

Endpoints:
  POST /recommend         → NLP parse + embed + retrieve + rank
  GET  /place/{id}        → Fetch a single place by ID
  POST /feedback          → Record like/dislike, update preference vector
  GET  /health            → Health / readiness probe
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.data_loader import PLACES, get_place_by_id
from app.models import (
    FeedbackRequest,
    FeedbackResponse,
    Place,
    RecommendRequest,
    RecommendResponse,
)
from app.nlp_parser import parse_query
from app.ranking import rank
from app.recommender import PlaceIndex

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Place Recommendation Engine",
    description=(
        "NLP-powered place recommender using sentence-transformer embeddings, "
        "cosine similarity, and configurable multi-signal ranking."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup — build the place index (pre-computes embeddings)
# ---------------------------------------------------------------------------
logger.info("Initialising place index with %d places…", len(PLACES))
_place_index = PlaceIndex(PLACES)
logger.info("Place index ready.")

# ---------------------------------------------------------------------------
# In-memory user preference store
# Format: { user_id: { tag: cumulative_weight } }
# In production, replace with Redis / a database.
# ---------------------------------------------------------------------------
_user_prefs: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Meta"])
def health() -> dict:
    """Simple liveness / readiness probe."""
    return {"status": "ok", "places_loaded": len(PLACES)}


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
def recommend(body: RecommendRequest) -> RecommendResponse:
    """
    Accept a natural-language query and return ranked place recommendations.

    **Flow:**
    1. Parse query → structured intent (NLP)
    2. Embed query with all-MiniLM-L6-v2
    3. Retrieve top-N candidates via cosine similarity
    4. Re-rank using weighted multi-signal scoring
    5. Apply user preference boost (if user_id provided)

    **Sample request body:**
    ```json
    {
      "query": "I want a quiet cafe in Ho Chi Minh City for studying, cheap price, nice atmosphere.",
      "user_id": "user_42",
      "top_k": 5,
      "weights": {"semantic": 0.5, "rating": 0.2, "popularity": 0.2, "distance": 0.1}
    }
    ```
    """
    logger.info("POST /recommend | query=%r | user=%s", body.query, body.user_id)

    # 1. NLP parse
    parsed = parse_query(body.query)
    from app.nlp_parser import detect_rejection
    is_rejection = detect_rejection(body.query)
    # 1. NLP parse
    parsed = parse_query(body.query)
    if body.user_id:
        user_tag_prefs = _user_prefs[body.user_id]
        if is_rejection:
            logger.info(f"User{body.user_id} expressed rejection. Update preferences negatively.")
            for tag in parsed.tags:
                user_tag_prefs[tag] -=1
            if parsed.category:
                user_tag_prefs[parsed.category] -=1
        elif parsed.tags or parsed.category:
            for tag in parsed.tags:
                user_tag_prefs[tag]+=1
    #logger.info("Parsed query: %s", parsed.model_dump())

    # 2 & 3. Embed + retrieve (fetch more than top_k so ranker has enough candidates)
    candidate_pool = min(len(PLACES), body.top_k * 4)
    candidates = _place_index.top_k_similar(body.query, k=candidate_pool)

    # 4 & 5. Re-rank with optional user prefs
    user_prefs = dict(_user_prefs[body.user_id]) if body.user_id else {}
    ranked = rank(
        candidates=candidates,
        parsed=parsed,
        weights=body.weights,
        user_prefs=user_prefs,
        top_k=body.top_k,
    )

    # Attach parsed query to each result for transparency
    for result in ranked:
        result.parsed_query = parsed

    return RecommendResponse(query=body.query, parsed=parsed, results=ranked)


@app.get("/place/{place_id}", response_model=Place, tags=["Places"])
def get_place(place_id: str) -> Place:
    """
    Retrieve full details for a single place by its ID.

    **Example:** `GET /place/place_001`
    """
    place = get_place_by_id(place_id)
    if place is None:
        raise HTTPException(status_code=404, detail=f"Place '{place_id}' not found.")
    return place


@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
def feedback(body: FeedbackRequest) -> FeedbackResponse:
    """
    Record a like or dislike for a place and update the user's preference vector.

    Liked tags gain +1 weight, disliked tags lose -1.
    These weights accumulate across sessions and influence future /recommend calls.

    **Sample request body:**
    ```json
    { "user_id": "user_42", "place_id": "place_007", "liked": true }
    ```
    """
    logger.info(
        "POST /feedback | user=%s | place=%s | liked=%s",
        body.user_id, body.place_id, body.liked,
    )

    place = get_place_by_id(body.place_id)
    if place is None:
        raise HTTPException(status_code=404, detail=f"Place '{body.place_id}' not found.")

    delta = +1 if body.liked else -1
    user_tag_prefs = _user_prefs[body.user_id]
    for tag in place.tags:
        user_tag_prefs[tag] += delta

    logger.info("Updated prefs for %s: %s", body.user_id, dict(user_tag_prefs))

    return FeedbackResponse(
        message="Feedback recorded. Preference vector updated.",
        user_id=body.user_id,
        updated_tags=dict(user_tag_prefs),
    )
