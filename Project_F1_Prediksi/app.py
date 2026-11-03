"""Flask API for serving pre-trained F1 podium predictions."""

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS


BASE_DIR = Path(__file__).resolve().parent
MODEL_OUTPUT_DIR = BASE_DIR / "f1_model_outputs"
MODEL_PATH = MODEL_OUTPUT_DIR / "f1_podium_model.pkl"
FEATURE_META_PATH = MODEL_OUTPUT_DIR / "feature_meta.json"
DATASET_PATH = BASE_DIR / "dataset" / "f1_processed_dataset.csv"


def load_runtime_assets() -> tuple[Any, list[str], pd.DataFrame]:
    """Load the model and prediction data once when the web process starts."""
    for path in (MODEL_PATH, FEATURE_META_PATH, DATASET_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required prediction asset not found: {path}")

    with FEATURE_META_PATH.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    features = metadata.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError(f"Invalid or missing feature list in {FEATURE_META_PATH}")

    dataset = pd.read_csv(DATASET_PATH)
    missing_columns = [column for column in features if column not in dataset.columns]
    required_metadata = [
        "raceId",
        "race_name",
        "circuit_name",
        "year",
        "round",
        "driverId",
        "driver_name",
        "code",
        "constructor_name",
        "grid_position",
    ]
    missing_columns.extend(
        column for column in required_metadata if column not in dataset.columns
    )
    if missing_columns:
        raise ValueError(
            f"{DATASET_PATH} is missing required column(s): "
            + ", ".join(sorted(set(missing_columns)))
        )

    model = joblib.load(MODEL_PATH)
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"Loaded model at {MODEL_PATH} does not provide predict_proba")

    return model, features, dataset


app = Flask("f1_podium_api")
CORS(
    app,
    resources={
        r"/*": {
            "origins": os.environ.get("FRONTEND_ORIGIN", "*"),
        }
    },
)
MODEL, FEATURES, DATASET = load_runtime_assets()


def _optional_int(value: Any) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def predict_for_race(race_id: int) -> tuple[dict[str, Any], int]:
    race_dataset = DATASET[DATASET["raceId"] == race_id].copy()
    if race_dataset.empty:
        return {"error": f"raceId {race_id} tidak ditemukan."}, 404

    race_dataset = race_dataset.dropna(subset=FEATURES)
    finite_rows = np.isfinite(race_dataset[FEATURES].to_numpy(dtype=float)).all(axis=1)
    race_dataset = race_dataset.loc[finite_rows].copy()
    if race_dataset.empty:
        return {
            "error": f"raceId {race_id} tidak memiliki fitur lengkap untuk prediksi."
        }, 422

    race_dataset["podium_probability"] = MODEL.predict_proba(
        race_dataset[FEATURES]
    )[:, 1]
    race_dataset = race_dataset.sort_values(
        "podium_probability", ascending=False, kind="mergesort"
    )
    first_row = race_dataset.iloc[0]

    predictions = []
    for _, row in race_dataset.iterrows():
        predictions.append(
            {
                "driver_id": _optional_int(row["driverId"]),
                "driver_name": "" if pd.isna(row["driver_name"]) else str(row["driver_name"]).strip(),
                "code": "" if pd.isna(row["code"]) else str(row["code"]),
                "constructor_name": (
                    ""
                    if pd.isna(row["constructor_name"])
                    else str(row["constructor_name"])
                ),
                "grid_position": _optional_int(row["grid_position"]),
                "podium_probability": round(float(row["podium_probability"]), 4),
                "actual_is_podium": (
                    _optional_int(row["is_podium"])
                    if "is_podium" in race_dataset.columns
                    else None
                ),
            }
        )

    return {
        "race_id": int(race_id),
        "race_name": str(first_row["race_name"]),
        "circuit_name": str(first_row["circuit_name"]),
        "year": int(first_row["year"]),
        "round": int(first_row["round"]),
        "predictions": predictions,
    }, 200


@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "status": "ok",
            "message": "API F1 aktif. Gunakan /predict?race_id=1182.",
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": True})


@app.route("/predict", methods=["GET"])
def api_predict():
    raw_race_id = request.args.get("race_id", "1000")
    try:
        race_id = int(raw_race_id)
    except (TypeError, ValueError):
        return jsonify({"error": "race_id harus berupa bilangan bulat."}), 400

    body, status = predict_for_race(race_id)
    return jsonify(body), status


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "10000")),
        debug=False,
    )
