"""
Train a supervised multi-label classifier for the 8 tourism categories.

This script turns the original rule/similarity-based project into a clearer
Data Science & AI project by adding:
- text preprocessing
- TF-IDF feature extraction
- supervised model training
- train/test evaluation
- saved model artifacts for the FastAPI app

Run:
    python train_label_model.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, hamming_loss, jaccard_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "Vietnam_Tourism_Final_8Labels.csv"
ARTIFACT_DIR = ROOT_DIR / "app" / "artifacts"
REPORT_DIR = ROOT_DIR / "reports"
MODEL_PATH = ARTIFACT_DIR / "label_classifier.joblib"
METADATA_PATH = ARTIFACT_DIR / "label_classifier_metadata.json"
METRICS_PATH = REPORT_DIR / "label_model_metrics.json"
CLASSIFICATION_REPORT_PATH = REPORT_DIR / "label_classification_report.csv"

LABEL_COLUMNS: List[str] = [
    "Adventure",
    "Relax",
    "Rural",
    "Urban",
    "Mountain",
    "Historical",
    "Food",
    "Nature",
]


def build_training_text(df: pd.DataFrame) -> pd.Series:
    """Combine fields that describe each destination into one text feature."""
    return (
        df["Place_Name"].fillna("")
        + " in "
        + df["Location"].fillna("")
        + ". "
        + df["Description"].fillna("")
    )


def optimize_thresholds(y_true: pd.DataFrame, y_prob: np.ndarray) -> Dict[str, float]:
    """Choose one decision threshold per label using validation F1.

    The grid now starts at 0.10 (was 0.15) to give minority classes such as
    Food and Rural a better chance of being captured at lower confidence levels.
    """
    thresholds: Dict[str, float] = {}
    grid = np.round(np.arange(0.10, 0.76, 0.05), 2)  # extended down to 0.10

    for i, label in enumerate(LABEL_COLUMNS):
        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in grid:
            pred = (y_prob[:, i] >= threshold).astype(int)
            score = f1_score(y_true.iloc[:, i], pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)
        thresholds[label] = best_threshold

    return thresholds


def build_pipeline() -> Pipeline:
    """Return a fresh, unfitted model pipeline (shared between train and CV)."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=3000,
                ),
            ),
            (
                "classifier",
                OneVsRestClassifier(
                    LogisticRegression(
                        solver="liblinear",
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=42,
                    )
                ),
            ),
        ]
    )


def cross_validate_model(x: pd.Series, y: pd.DataFrame, n_splits: int = 5) -> Dict[str, object]:
    """
    Run k-fold cross-validation to get a stable performance estimate.

    A single 80/20 split with 315 rows can produce misleading per-label scores
    (e.g. Food F1=0 on one unlucky split). Cross-validation averages over
    multiple splits and shows mean ± std for each metric.

    This function uses a fixed random seed so results are reproducible.
    """
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    cv_micro_f1: List[float] = []
    cv_macro_f1: List[float] = []
    cv_per_label: Dict[str, List[float]] = {label: [] for label in LABEL_COLUMNS}

    for fold, (train_idx, val_idx) in enumerate(kf.split(x)):
        x_tr = x.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        x_vl = x.iloc[val_idx]
        y_vl = y.iloc[val_idx]

        pipe = build_pipeline()
        pipe.fit(x_tr, y_tr)
        y_pred_fold = pipe.predict(x_vl)

        cv_micro_f1.append(float(f1_score(y_vl, y_pred_fold, average="micro", zero_division=0)))
        cv_macro_f1.append(float(f1_score(y_vl, y_pred_fold, average="macro", zero_division=0)))

        report_fold = classification_report(
            y_vl, y_pred_fold,
            target_names=LABEL_COLUMNS,
            output_dict=True,
            zero_division=0,
        )
        for label in LABEL_COLUMNS:
            cv_per_label[label].append(float(report_fold[label]["f1-score"]))

    return {
        "n_splits": n_splits,
        "cv_micro_f1_mean": float(np.mean(cv_micro_f1)),
        "cv_micro_f1_std":  float(np.std(cv_micro_f1)),
        "cv_macro_f1_mean": float(np.mean(cv_macro_f1)),
        "cv_macro_f1_std":  float(np.std(cv_macro_f1)),
        "cv_per_label_f1_mean": {label: float(np.mean(cv_per_label[label])) for label in LABEL_COLUMNS},
        "cv_per_label_f1_std":  {label: float(np.std(cv_per_label[label]))  for label in LABEL_COLUMNS},
        "note": (
            "Cross-validation is more reliable than a single test split for a 315-row dataset. "
            "Food and Rural classes have fewer than 30 examples; their per-fold F1 is volatile."
        ),
    }


def apply_thresholds(y_prob: np.ndarray, thresholds: Dict[str, float]) -> np.ndarray:
    """Convert probabilities to binary predictions using per-label thresholds."""
    threshold_array = np.array([thresholds[label] for label in LABEL_COLUMNS])
    return (y_prob >= threshold_array).astype(int)


def train() -> Dict[str, object]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    missing = [c for c in ["Place_Name", "Location", "Description", *LABEL_COLUMNS] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    x = build_training_text(df)
    y = df[LABEL_COLUMNS].astype(int)

    # Split into train/validation/test. Validation is used only for choosing
    # decision thresholds; final metrics are reported on the held-out test set.
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.25,
        random_state=42,
    )

    model = build_pipeline()
    model.fit(x_train, y_train)

    val_prob = model.predict_proba(x_val)
    thresholds = optimize_thresholds(y_val, val_prob)

    test_prob = model.predict_proba(x_test)
    y_pred = apply_thresholds(test_prob, thresholds)

    report = classification_report(
        y_test,
        y_pred,
        target_names=LABEL_COLUMNS,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "dataset_rows": int(len(df)),
        "train_rows": int(len(x_train)),
        "validation_rows": int(len(x_val)),
        "test_rows": int(len(x_test)),
        "labels": LABEL_COLUMNS,
        "label_distribution": {label: int(y[label].sum()) for label in LABEL_COLUMNS},
        "micro_f1": float(f1_score(y_test, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "samples_f1": float(f1_score(y_test, y_pred, average="samples", zero_division=0)),
        "jaccard_samples": float(jaccard_score(y_test, y_pred, average="samples", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_test, y_pred)),
        "decision_thresholds": thresholds,
        "model": "TF-IDF + OneVsRest Logistic Regression",
        "notes": [
            "This is a multi-label classifier: one destination can belong to several categories.",
            "Per-label thresholds are chosen on the validation split to handle label imbalance more honestly than a fixed 0.5 threshold.",
            "Macro F1 is sensitive to weak minority-label performance, so it is more honest than accuracy for this dataset.",
            "Food F1 on the test set can reach 0.0 on certain random splits because the Food class has only 22 examples (7% of the dataset). The cross-validation mean gives a more stable estimate of true Food performance.",
            "Rural has a similar scarcity issue (26 examples). Both classes should be mentioned as limitations in the defense.",
        ],
    }

    # ── Cross-validation for a stable performance estimate ────────────────
    print("\nRunning 5-fold cross-validation (more stable than a single split)…")
    cv_results = cross_validate_model(x, y, n_splits=5)
    metrics["cross_validation"] = cv_results

    joblib.dump(model, MODEL_PATH)

    metadata = {
        "labels": LABEL_COLUMNS,
        "decision_thresholds": thresholds,
        "model_file": str(MODEL_PATH.relative_to(ROOT_DIR)),
        "training_text": "Place_Name + Location + Description",
        "model_type": "supervised_multi_label_text_classifier",
    }

    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(report).T.to_csv(CLASSIFICATION_REPORT_PATH)

    print("Saved model:", MODEL_PATH)
    print("Saved metadata:", METADATA_PATH)
    print("Saved metrics:", METRICS_PATH)
    print("Saved classification report:", CLASSIFICATION_REPORT_PATH)
    print(json.dumps({k: metrics[k] for k in ["micro_f1", "macro_f1", "weighted_f1", "hamming_loss"]}, indent=2))
    print("Decision thresholds:", json.dumps(thresholds, indent=2))
    print("\nCross-validation summary (5-fold):")
    print(f"  Micro F1: {cv_results['cv_micro_f1_mean']:.3f} ± {cv_results['cv_micro_f1_std']:.3f}")
    print(f"  Macro F1: {cv_results['cv_macro_f1_mean']:.3f} ± {cv_results['cv_macro_f1_std']:.3f}")
    print("  Per-label F1 means (CV):")
    for label in LABEL_COLUMNS:
        mean = cv_results["cv_per_label_f1_mean"][label]
        std  = cv_results["cv_per_label_f1_std"][label]
        print(f"    {label:<12}: {mean:.3f} ± {std:.3f}")
    return metrics

if __name__ == "__main__":
    train()
