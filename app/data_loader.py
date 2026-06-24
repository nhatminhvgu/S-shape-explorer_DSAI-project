"""
data_loader.py — Loads the real Vietnam tourism dataset from CSV + XLSX.

Sources:
  - Vietnam_Tourism_Final_8Labels.csv  → place names, descriptions, locations, 8 labels
  - Full_Translated_DataSet_V2.xlsx    → ratings (4.8/5 format), keywords, image URLs

Both files have 315 rows aligned by row index.
"""

import logging
import os
import re
import warnings
from typing import Dict, List, Optional

import pandas as pd

from app.image_utils import clean_image_url, primary_category_from_row

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT_DIR, "Vietnam_Tourism_Final_8Labels.csv")
XLSX_PATH = os.path.join(ROOT_DIR, "Full_Translated_DataSet_V2.xlsx")


def load_places() -> list:
    from app.models import Place

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV dataset not found: {CSV_PATH}")
    if not os.path.exists(XLSX_PATH):
        raise FileNotFoundError(f"XLSX dataset not found: {XLSX_PATH}")

    logger.info("Loading CSV: %s", CSV_PATH)
    csv_df = pd.read_csv(CSV_PATH)

    logger.info("Loading XLSX: %s", XLSX_PATH)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xlsx_df = pd.read_excel(XLSX_PATH)

    logger.info("CSV rows: %d | XLSX rows: %d", len(csv_df), len(xlsx_df))

    places: List[Place] = []
    for idx in range(len(csv_df)):
        csv_row = csv_df.iloc[idx]
        xlsx_row = xlsx_df.iloc[idx] if idx < len(xlsx_df) else None

        # Rating: parse "4.8/5" → 4.8
        rating = 0.0
        if xlsx_row is not None:
            raw_rating = xlsx_row.get("Rating", "")
            if pd.notna(raw_rating):
                m = re.search(r"(\d+\.?\d*)", str(raw_rating))
                if m:
                    rating = min(5.0, float(m.group(1)))

        # Keywords: '"sea", "relax", "swim"' → ['sea', 'relax', 'swim']
        keywords: List[str] = []
        if xlsx_row is not None:
            raw_kw = xlsx_row.get("Keywords", "")
            if pd.notna(raw_kw):
                keywords = [
                    k.strip().strip('"\'')
                    for k in str(raw_kw).split(",")
                    if k.strip().strip('"\'')
                ]

        # Image URL: sanitise before sending to the UI.
        category_key = primary_category_from_row(csv_row)
        raw_image_url = ""
        if xlsx_row is not None:
            raw_img = xlsx_row.get("Image_URL", "")
            if pd.notna(raw_img):
                raw_image_url = str(raw_img).strip()
        image_url = clean_image_url(raw_image_url, category_key)

        place = Place(
            id=f"place_{idx + 1:03d}",
            name=str(csv_row["Place_Name"]).strip(),
            location=str(csv_row["Location"]).strip(),
            description=str(csv_row["Description"]).strip(),
            rating=rating,
            keywords=keywords,
            image_url=image_url,
            adventure=int(csv_row["Adventure"]),
            relax=int(csv_row["Relax"]),
            rural=int(csv_row["Rural"]),
            urban=int(csv_row["Urban"]),
            mountain=int(csv_row["Mountain"]),
            historical=int(csv_row["Historical"]),
            food=int(csv_row["Food"]),
            nature=int(csv_row["Nature"]),
        )
        places.append(place)

    logger.info("Dataset loaded: %d places.", len(places))
    return places


def build_place_index(places: list) -> Dict[str, object]:
    return {p.id: p for p in places}


# Module-level singletons — loaded once at import time.
PLACES = load_places()
PLACE_INDEX = build_place_index(PLACES)


def get_place_by_id(place_id: str) -> Optional[object]:
    return PLACE_INDEX.get(place_id)
