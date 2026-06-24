"""
models.py — Pydantic data models for S-Shape Explorer.

Place maps directly to the dataset columns:
  - Vietnam_Tourism_Final_8Labels.csv  (name, location, description, 8 binary labels)
  - Full_Translated_DataSet_V2.xlsx    (rating, keywords, image_url)
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class Place(BaseModel):
    id: str
    name: str
    location: str
    description: str
    rating: float = 0.0
    likes: int = 0
    dislikes: int = 0
    keywords: List[str] = Field(default_factory=list)
    image_url: str = ""
    # 8 preference labels from the dataset (0 = no, 1 = yes)
    adventure: int = 0
    relax: int = 0
    rural: int = 0
    urban: int = 0
    mountain: int = 0
    historical: int = 0
    food: int = 0
    nature: int = 0


class RecommendRequest(BaseModel):
    query: str = Field(
        "",
        example="I want a relaxing beach in Khanh Hoa",
        description="Free-text query. Can be empty if preferences are provided.",
    )
    preferences: List[str] = Field(
        default_factory=list,
        example=["Relax", "Nature"],
        description=(
            "One or more of: Adventure, Relax, Rural, Urban, "
            "Mountain, Historical, Food, Nature"
        ),
    )
    location: str = Field(
        "",
        example="Khanh Hoa",
        description="Optional province/city/region filter.",
    )
    top_k: int = Field(5, ge=1, le=20)
    user_id: Optional[str] = None


class PlaceResult(BaseModel):
    place: Place
    score: float
    matched_labels: List[str] = Field(default_factory=list)
    explanation: str = ""


class RecommendResponse(BaseModel):
    query: str
    selected_preferences: List[str]
    inferred_preferences: List[str] = Field(default_factory=list)
    ai_label_probabilities: Dict[str, float] = Field(default_factory=dict)
    recommendations: List[PlaceResult]
    # Set when results are limited due to sparse coverage in the dataset
    # for the requested location+category combination.
    low_confidence_note: str = ""
