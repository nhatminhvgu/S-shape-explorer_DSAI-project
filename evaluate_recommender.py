"""
Evaluate recommendation quality using label overlap as the ground truth signal.

For each destination, the script uses that destination's text as a query. The
recommender should return other destinations with overlapping labels. This is not
as strong as real user-click data, but it gives a measurable offline benchmark
for a course project.

Run:
    python evaluate_recommender.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Set

import numpy as np

from app.data_loader import PLACES
from app.ml_intent import infer_preferences
from app.ranking import rank
from app.recommender import LABEL_MAP, PlaceIndex

ROOT_DIR = Path(__file__).resolve().parent
REPORT_DIR = ROOT_DIR / "reports"
OUTPUT_PATH = REPORT_DIR / "recommendation_metrics.json"


def place_labels(place) -> Set[str]:
    return {label for label, field in LABEL_MAP.items() if getattr(place, field, 0) == 1}


def precision_at_k(relevant: List[int], k: int) -> float:
    if k <= 0:
        return 0.0
    return float(sum(relevant[:k]) / k)


def recall_at_k(relevant: List[int], total_relevant: int, k: int) -> float:
    if total_relevant <= 0:
        return 0.0
    return float(sum(relevant[:k]) / total_relevant)


def dcg_at_k(relevant: List[int], k: int) -> float:
    return float(sum(rel / np.log2(i + 2) for i, rel in enumerate(relevant[:k])))


def ndcg_at_k(relevant: List[int], total_relevant: int, k: int) -> float:
    ideal_hits = [1] * min(total_relevant, k)
    ideal_dcg = dcg_at_k(ideal_hits, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(relevant, k) / ideal_dcg


def mean_reciprocal_rank(relevant: List[int]) -> float:
    """MRR: inverse of the rank of the first relevant result (or 0 if none)."""
    for i, rel in enumerate(relevant):
        if rel == 1:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(k_values: Iterable[int] = (3, 5, 10)) -> Dict[str, object]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    index = PlaceIndex(PLACES)
    label_sets = {p.id: place_labels(p) for p in PLACES}

    # STRICT_JACCARD_THRESHOLD: a recommendation is "strictly relevant" when the
    # label overlap (Jaccard) is at least this value.  Using any-overlap (0.0)
    # makes Precision@K artificially high because 8 broad labels cover almost
    # every tourism destination pair.  0.33 requires at least 1/3 label overlap.
    STRICT_JACCARD_THRESHOLD = 0.33

    metric_rows = {
        k: {
            "precision": [],          # soft: any label overlap
            "strict_precision": [],   # strict: Jaccard >= threshold
            "recall": [],
            "ndcg": [],
            "label_jaccard": [],
            "mrr": [],
        }
        for k in k_values
    }

    for query_place in PLACES:
        query_labels = label_sets[query_place.id]
        if not query_labels:
            continue

        query_text = f"{query_place.name} {query_place.location} {query_place.description}"
        inferred_preferences, label_probs = infer_preferences(query_text)
        candidates = [
            (place, score)
            for place, score in index.top_k_similar(query_text, k=len(PLACES))
            if place.id != query_place.id
        ]

        ranked = rank(
            candidates=candidates,
            preferences=inferred_preferences,
            query_location="",
            top_k=max(k_values),
            has_query=True,
            label_probabilities=label_probs,
        )

        # Soft relevance: at least one shared label (existing definition)
        relevant_flags = [
            1 if query_labels & label_sets[item.place.id] else 0
            for item in ranked
        ]

        # Strict relevance: Jaccard(labels_A, labels_B) >= STRICT_JACCARD_THRESHOLD
        def jaccard(a: Set[str], b: Set[str]) -> float:
            union = a | b
            return len(a & b) / len(union) if union else 0.0

        strict_flags = [
            1 if jaccard(query_labels, label_sets[item.place.id]) >= STRICT_JACCARD_THRESHOLD else 0
            for item in ranked
        ]

        label_jaccards = [
            jaccard(query_labels, label_sets[item.place.id])
            for item in ranked
        ]

        total_relevant = sum(
            1
            for place in PLACES
            if place.id != query_place.id and query_labels & label_sets[place.id]
        )

        for k in k_values:
            metric_rows[k]["precision"].append(precision_at_k(relevant_flags, k))
            metric_rows[k]["strict_precision"].append(precision_at_k(strict_flags, k))
            metric_rows[k]["recall"].append(recall_at_k(relevant_flags, total_relevant, k))
            metric_rows[k]["ndcg"].append(ndcg_at_k(relevant_flags, total_relevant, k))
            metric_rows[k]["label_jaccard"].append(float(np.mean(label_jaccards[:k])))
            metric_rows[k]["mrr"].append(mean_reciprocal_rank(relevant_flags[:k]))

    summary: Dict[str, object] = {}

    for k in k_values:
        summary[f"precision_at_{k}"] = float(np.mean(metric_rows[k]["precision"]))
        summary[f"strict_precision_at_{k}"] = float(np.mean(metric_rows[k]["strict_precision"]))
        summary[f"recall_at_{k}"] = float(np.mean(metric_rows[k]["recall"]))
        summary[f"ndcg_at_{k}"] = float(np.mean(metric_rows[k]["ndcg"]))
        summary[f"mean_label_jaccard_at_{k}"] = float(np.mean(metric_rows[k]["label_jaccard"]))
        summary[f"mrr_at_{k}"] = float(np.mean(metric_rows[k]["mrr"]))

    summary["evaluated_queries"] = len(metric_rows[min(k_values)]["precision"])
    summary["ground_truth_soft"] = (
        "A recommendation is relevant (soft) if it shares at least one of the query "
        "destination's labels. WARNING: with 8 broad labels, this makes Precision@K "
        "appear very high (~0.98). See strict_precision_at_K for a fairer metric."
    )
    summary["ground_truth_strict"] = (
        f"A recommendation is strictly relevant if Jaccard(labels_A, labels_B) >= "
        f"{STRICT_JACCARD_THRESHOLD:.2f}. This requires non-trivial label overlap "
        f"and is a more honest signal for a course evaluation."
    )
    summary["strict_jaccard_threshold"] = STRICT_JACCARD_THRESHOLD

    OUTPUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved recommendation metrics:", OUTPUT_PATH)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    evaluate()
