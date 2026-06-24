"""Load and use the trained multi-label intent classifier."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT_DIR / "app" / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "label_classifier.joblib"
METADATA_PATH = ARTIFACT_DIR / "label_classifier_metadata.json"

DEFAULT_LABELS: List[str] = [
    "Adventure",
    "Relax",
    "Rural",
    "Urban",
    "Mountain",
    "Historical",
    "Food",
    "Nature",
]

_model = None
_metadata = None


def _load_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {"labels": DEFAULT_LABELS, "decision_threshold": 0.5}
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def get_label_classifier():
    """Return the trained classifier if available; otherwise return None."""
    global _model, _metadata
    if _model is not None:
        return _model
    _metadata = _load_metadata()
    if not MODEL_PATH.exists():
        logger.warning("Label classifier is not trained yet: %s", MODEL_PATH)
        return None
    _model = joblib.load(MODEL_PATH)
    logger.info("Loaded trained label classifier from %s", MODEL_PATH)
    return _model


def predict_label_probabilities(text: str) -> Dict[str, float]:
    """Predict probability for each tourism label from free text."""
    model = get_label_classifier()
    metadata = _metadata or _load_metadata()
    labels = metadata.get("labels", DEFAULT_LABELS)
    if model is None or not text.strip():
        return {label: 0.0 for label in labels}

    probabilities = model.predict_proba([text])[0]
    return {
        label: round(float(prob), 4)
        for label, prob in zip(labels, probabilities)
    }


def infer_preferences(text: str, max_labels: int = 3, threshold: float | None = None) -> Tuple[List[str], Dict[str, float]]:
    """Return top predicted labels and the full probability dictionary."""
    metadata = _metadata or _load_metadata()
    probabilities = predict_label_probabilities(text)

    decision_thresholds = metadata.get("decision_thresholds", {})
    default_threshold = float(metadata.get("decision_threshold", 0.5))

    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    selected: List[str] = []
    for label, prob in ranked:
        label_threshold = float(threshold if threshold is not None else decision_thresholds.get(label, default_threshold))
        if prob >= label_threshold:
            selected.append(label)
        if len(selected) >= max_labels:
            break

    # Keep at least the best label for non-empty queries, even when probabilities
    # are conservative. This helps the recommender use the model signal in demos.
    if not selected and text.strip() and ranked:
        selected = [ranked[0][0]]

    return selected, probabilities
