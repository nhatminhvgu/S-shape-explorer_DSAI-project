"""
models.py — Pydantic data models for request/response schemas and internal data structures.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict


# ---------------------------------------------------------------------------
# Place data model (mirrors the JSON dataset)
# ---------------------------------------------------------------------------

class Place(BaseModel):
    id: str
    name: str
    category: str                    # e.g. "cafe", "restaurant", "park"
    tags: List[str]                  # descriptive tags: ["quiet", "wifi", "outdoor"]
    description: str                 # free-text used for embedding
    city: str
    price_level: str                 # "cheap" | "moderate" | "expensive"
    rating: float                    # 1.0 – 5.0
    popularity: int                  # arbitrary score 1–100
    lat: Optional[float] = None
    lon: Optional[float] = None


# ---------------------------------------------------------------------------
# NLP parser output
# ---------------------------------------------------------------------------

class ParsedQuery(BaseModel):
    category: str = ""
    mood: List[str] = []
    budget: str = ""
    purpose: List[str] = []
    location: str = ""
    tags: List[str] = []


# ---------------------------------------------------------------------------
# API request bodies
# ---------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    query: str = Field(..., example="I want a quiet cafe in Ho Chi Minh City for studying, cheap price, nice atmosphere.")
    user_id: Optional[str] = Field(None, example="user_42")
    top_k: int = Field(5, ge=1, le=20)
    weights: Optional[Dict[str, float]] = Field(
        None,
        example={"semantic": 0.5, "rating": 0.2, "popularity": 0.2, "distance": 0.1},
        description="Configurable ranking weights. Must sum to 1.0.",
    )


class FeedbackRequest(BaseModel):
    user_id: str = Field(..., example="user_42")
    place_id: str = Field(..., example="place_007")
    liked: bool = Field(..., example=True)


# ---------------------------------------------------------------------------
# API response bodies
# ---------------------------------------------------------------------------

class PlaceResult(BaseModel):
    place: Place
    score: float                     # final weighted score
    similarity: float                # raw cosine similarity
    parsed_query: Optional[ParsedQuery] = None


class RecommendResponse(BaseModel):
    query: str
    parsed: ParsedQuery
    results: List[PlaceResult]


class FeedbackResponse(BaseModel):
    message: str
    user_id: str
    updated_tags: Dict[str, int]     # tag → cumulative weight (positive or negative)
