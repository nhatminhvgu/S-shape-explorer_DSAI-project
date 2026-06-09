"""
main.py — FastAPI application entry point for S-Shape Explorer.

Endpoints:
  POST /recommend  → preferences + free-text query → ranked real destinations
  GET  /place/{id} → fetch a single place by ID
  POST /feedback   → record like/dislike for a place
  GET  /health     → readiness probe
  GET  /           → serves the travel discovery UI (app/static/index.html)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.data_loader import PLACES, get_place_by_id
from app.models import Place, RecommendRequest, RecommendResponse
from app.ranking import rank
from app.recommender import PlaceIndex, VALID_LABELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_place_index: Optional[PlaceIndex] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _place_index
    logger.info("Dataset: %d real tourism places loaded.", len(PLACES))
    _place_index = PlaceIndex(PLACES)
    logger.info("Recommendation engine ready.")
    yield


app = FastAPI(
    title="S-Shape Explorer — Vietnam Travel Recommender",
    description=(
        "Chatbot-style travel recommendation for Vietnam using "
        "TF-IDF similarity and 8-category preference labels."
    ),
    version="2.0.0",
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


@app.get("/", tags=["UI"])
def root():
    return FileResponse(static_dir / "index.html")


@app.get("/health", tags=["Meta"])
def health() -> dict:
    return {"status": "ok", "places_loaded": len(PLACES)}


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
def recommend(body: RecommendRequest) -> RecommendResponse:
    query = (body.query or "").strip()

    # Validate and normalise preference labels
    preferences = [
        p for p in (body.preferences or [])
        if p in VALID_LABELS
    ]

    # Explicit location wins; otherwise try to extract from the query text
    location = (body.location or "").strip()
    if not location and query:
        from app.nlp_parser import _extract_location, _normalise
        location = _extract_location(_normalise(query))

    logger.info(
        "POST /recommend | query=%r | preferences=%s | location=%r | top_k=%d",
        query, preferences, location, body.top_k,
    )

    # Pool size: use all places when there's no text query so label matching
    # can scan the full 315-place dataset; otherwise retrieve the top 80.
    if not query:
        pool_size = len(PLACES)
    else:
        pool_size = min(len(PLACES), max(body.top_k * 12, 80))

    candidates = _place_index.top_k_similar(query, k=pool_size)

    ranked = rank(
        candidates=candidates,
        preferences=preferences,
        query_location=location,
        top_k=body.top_k,
        has_query=bool(query),
    )

    logger.info("Returning %d recommendations.", len(ranked))

    return RecommendResponse(
        query=query,
        selected_preferences=preferences,
        recommendations=ranked,
    )


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
    logger.info(
        "POST /feedback | place_id=%r | feedback=%r | user_id=%r",
        body.place_id, body.feedback, body.user_id,
    )
    if body.feedback not in ("like", "dislike"):
        raise HTTPException(status_code=422, detail="feedback must be 'like' or 'dislike'.")
    place = get_place_by_id(body.place_id)
    if place is None:
        raise HTTPException(status_code=404, detail=f"Place '{body.place_id}' not found.")
    return {"status": "ok", "place_id": body.place_id, "feedback": body.feedback}


