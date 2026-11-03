"""Train and evaluate the F1 podium XGBoost model."""

import json
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from preprocess import FEATURES, TARGET


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "f1_processed_dataset.csv"
MODEL_OUTPUT_DIR = BASE_DIR / "f1_model_outputs"
MODEL_PATH = MODEL_OUTPUT_DIR / "f1_podium_model.pkl"
METRICS_PATH = MODEL_OUTPUT_DIR / "evaluation_metrics.json"
FEATURE_META_PATH = MODEL_OUTPUT_DIR / "feature_meta.json"


def temporal_split(
    dataset: pd.DataFrame, test_seasons: int = 2
) -> Tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """Hold out the newest seasons so test data is later than training data."""
    if test_seasons < 1:
        raise ValueError("test_seasons must be at least 1")
    if "year" not in dataset.columns:
        raise ValueError("The processed dataset is missing the 'year' column")

    years = sorted(int(year) for year in dataset["year"].dropna().unique())
    if len(years) <= test_seasons:
        raise ValueError(
            f"Need more than {test_seasons} seasons for a temporal train/test split; "
            f"found {len(years)}"
        )

    test_years = years[-test_seasons:]
    train_dataset = dataset[~dataset["year"].isin(test_years)].copy()
    test_dataset = dataset[dataset["year"].isin(test_years)].copy()
    return train_dataset, test_dataset, test_years


def main() -> None:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"Processed dataset not found at {DATASET_PATH}. Run preprocess.py first."
        )

    dataset = pd.read_csv(DATASET_PATH)
    missing_columns = [
        column for column in [*FEATURES, TARGET, "year"] if column not in dataset.columns
    ]
    if missing_columns:
        raise ValueError(
            "Processed dataset is missing required column(s): "
            + ", ".join(missing_columns)
        )

    dataset = dataset.dropna(subset=[*FEATURES, TARGET, "year"])
    train_dataset, test_dataset, test_years = temporal_split(dataset, test_seasons=2)
    if train_dataset.empty or test_dataset.empty:
        raise ValueError("Temporal split produced an empty training or test set")

    x_train = train_dataset[FEATURES]
    y_train = train_dataset[TARGET].astype(int)
    x_test = test_dataset[FEATURES]
    y_test = test_dataset[TARGET].astype(int)

    positive_count = int((y_train == 1).sum())
    negative_count = int((y_train == 0).sum())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Training data must contain both podium and non-podium rows")

    scale_pos_weight = negative_count / positive_count
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_test, probabilities)
    matrix = confusion_matrix(y_test, predictions).tolist()

    metrics = {
        "model_used": "XGBoost",
        "train_years": [
            int(train_dataset["year"].min()),
            int(train_dataset["year"].max()),
        ],
        "test_years": test_years,
        "n_train": int(len(train_dataset)),
        "n_test": int(len(test_dataset)),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "confusion_matrix": matrix,
    }

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    with FEATURE_META_PATH.open("w", encoding="utf-8") as metadata_file:
        json.dump(
            {"features": FEATURES, "target": TARGET, "using_xgboost": True},
            metadata_file,
            indent=2,
        )
    joblib.dump(model, MODEL_PATH)

    print(f"Train seasons: {metrics['train_years']}")
    print(f"Test seasons: {test_years}")
    print(f"Positive class weight: {scale_pos_weight:.2f}")
    print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(np.asarray(matrix))
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Non-Podium", "Podium"],
            zero_division=0,
        )
    )
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
