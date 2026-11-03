"""Build the leakage-safe, model-ready F1 dataset from the raw CSV files."""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
OUT_PATH = DATA_DIR / "f1_processed_dataset.csv"

FEATURES = [
    "grid_position",
    "quali_time_gap",
    "constructor_win_rate_at_circuit",
    "driver_recent_points_avg",
]
TARGET = "is_podium"

RAW_FILES = {
    "results": "results.csv",
    "qualifying": "qualifying.csv",
    "driver_standings": "driver_standings.csv",
    "constructors": "constructors.csv",
    "circuits": "circuits.csv",
    "races": "races.csv",
    "drivers": "drivers.csv",
}


def load_raw_data(data_dir: Path = DATA_DIR) -> Dict[str, pd.DataFrame]:
    """Load all required raw CSV files, treating Ergast's ``\\N`` as missing."""
    missing_files = [filename for filename in RAW_FILES.values() if not (data_dir / filename).is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing required CSV file(s) in {data_dir}: {', '.join(missing_files)}"
        )

    return {
        key: pd.read_csv(data_dir / filename, na_values=["\\N"])
        for key, filename in RAW_FILES.items()
    }


def time_to_seconds(value: object) -> float:
    """Convert a qualifying lap time such as ``1:26.572`` to seconds."""
    if pd.isna(value):
        return np.nan

    text = str(value).strip()
    if not text or text == "\\N":
        return np.nan

    try:
        if ":" in text:
            minutes, seconds = text.split(":")
            return int(minutes) * 60 + float(seconds)
        return float(text)
    except ValueError:
        return np.nan


def build_qualifying_features(qualifying: pd.DataFrame) -> pd.DataFrame:
    """Build the best qualifying time and gap to pole for each driver."""
    q = qualifying.copy()
    if q.empty:
        return pd.DataFrame(
            columns=["raceId", "driverId", "best_quali_time", "quali_time_gap"]
        )

    for column in ("q1", "q2", "q3"):
        q[f"{column}_sec"] = q[column].map(time_to_seconds)

    q["best_quali_time"] = q[["q1_sec", "q2_sec", "q3_sec"]].min(axis=1)
    pole_time = q.groupby("raceId")["best_quali_time"].transform("min")
    q["quali_time_gap"] = q["best_quali_time"] - pole_time

    return q[["raceId", "driverId", "best_quali_time", "quali_time_gap"]]


def build_driver_recent_points_avg(
    driver_standings: pd.DataFrame, races: pd.DataFrame
) -> pd.DataFrame:
    """Average up to three prior standings-point snapshots for each driver."""
    if driver_standings.empty or races.empty:
        return pd.DataFrame(columns=["raceId", "driverId", "driver_recent_points_avg"])

    standings = driver_standings.merge(
        races[["raceId", "year", "round"]], on="raceId", how="left"
    )
    standings = standings.sort_values(
        ["driverId", "year", "round", "raceId"], kind="mergesort"
    )
    standings["points"] = pd.to_numeric(standings["points"], errors="coerce")
    standings["driver_recent_points_avg"] = standings.groupby("driverId")[
        "points"
    ].transform(lambda points: points.shift(1).rolling(window=3, min_periods=1).mean())
    standings["driver_recent_points_avg"] = standings[
        "driver_recent_points_avg"
    ].fillna(0)

    return standings[["raceId", "driverId", "driver_recent_points_avg"]]


def build_constructor_win_rate_at_circuit(
    results: pd.DataFrame, races: pd.DataFrame
) -> pd.DataFrame:
    """Compute each constructor's prior win rate at a circuit, by race."""
    if results.empty or races.empty:
        return pd.DataFrame(
            columns=["raceId", "constructorId", "constructor_win_rate_at_circuit"]
        )

    race_results = results[
        ["raceId", "constructorId", "position"]
    ].merge(
        races[["raceId", "circuitId", "year", "round"]],
        on="raceId",
        how="left",
    )
    race_results["is_win"] = (
        pd.to_numeric(race_results["position"], errors="coerce") == 1
    ).astype(int)

    constructor_races = (
        race_results.groupby(
            ["raceId", "constructorId", "circuitId", "year", "round"],
            as_index=False,
            dropna=False,
        )["is_win"]
        .max()
        .sort_values(
            ["constructorId", "circuitId", "year", "round", "raceId"],
            kind="mergesort",
        )
    )
    prior_rate = constructor_races.groupby(
        ["constructorId", "circuitId"], sort=False
    )["is_win"].transform(
        lambda wins: wins.shift(1).expanding(min_periods=1).mean()
    )
    constructor_races["constructor_win_rate_at_circuit"] = prior_rate.fillna(0)

    return constructor_races[
        ["raceId", "constructorId", "constructor_win_rate_at_circuit"]
    ]


def _require_columns(
    frame: pd.DataFrame, columns: tuple[str, ...], source: str
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required column(s): {', '.join(missing)}")


def build_processed_dataset(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join raw data and produce the same four model features and podium target."""
    results = datasets["results"]
    qualifying = datasets["qualifying"]
    driver_standings = datasets["driver_standings"]
    constructors = datasets["constructors"]
    circuits = datasets["circuits"]
    races = datasets["races"]
    drivers = datasets["drivers"]

    if any(frame.empty for frame in datasets.values()):
        empty_sources = [name for name, frame in datasets.items() if frame.empty]
        raise ValueError(f"Required dataset(s) are empty: {', '.join(empty_sources)}")

    _require_columns(
        results,
        ("raceId", "driverId", "constructorId", "grid", "position", "points"),
        "results.csv",
    )
    _require_columns(qualifying, ("raceId", "driverId", "q1", "q2", "q3"), "qualifying.csv")
    _require_columns(driver_standings, ("raceId", "driverId", "points"), "driver_standings.csv")
    _require_columns(constructors, ("constructorId", "name"), "constructors.csv")
    _require_columns(circuits, ("circuitId", "name"), "circuits.csv")
    _require_columns(races, ("raceId", "year", "round", "circuitId", "name", "date"), "races.csv")
    _require_columns(drivers, ("driverId", "forename", "surname", "code"), "drivers.csv")

    base = results[
        ["raceId", "driverId", "constructorId", "grid", "position", "points"]
    ].copy()
    base["position_num"] = pd.to_numeric(base["position"], errors="coerce")
    base[TARGET] = (base["position_num"] <= 3).astype(int)
    base = base.rename(columns={"grid": "grid_position"})
    base["grid_position"] = pd.to_numeric(base["grid_position"], errors="coerce")
    base.loc[base["grid_position"] == 0, "grid_position"] = np.nan

    dataset = base.merge(
        build_qualifying_features(qualifying),
        on=["raceId", "driverId"],
        how="left",
    )
    dataset = dataset.merge(
        build_driver_recent_points_avg(driver_standings, races),
        on=["raceId", "driverId"],
        how="left",
    )
    dataset = dataset.merge(
        build_constructor_win_rate_at_circuit(results, races),
        on=["raceId", "constructorId"],
        how="left",
    )
    dataset = dataset.merge(
        drivers[["driverId", "forename", "surname", "code"]],
        on="driverId",
        how="left",
    )
    dataset = dataset.merge(
        constructors[["constructorId", "name"]].rename(
            columns={"name": "constructor_name"}
        ),
        on="constructorId",
        how="left",
    )
    dataset = dataset.merge(
        races[["raceId", "year", "round", "name", "circuitId", "date"]].rename(
            columns={"name": "race_name"}
        ),
        on="raceId",
        how="left",
    )
    dataset = dataset.merge(
        circuits[["circuitId", "name"]].rename(columns={"name": "circuit_name"}),
        on="circuitId",
        how="left",
    )

    for column in ("grid_position", "quali_time_gap"):
        dataset[column] = dataset.groupby("raceId")[column].transform(
            lambda values: values.fillna(values.median())
        )
        dataset[column] = dataset[column].fillna(0)

    dataset["driver_recent_points_avg"] = dataset[
        "driver_recent_points_avg"
    ].fillna(0)
    dataset["constructor_win_rate_at_circuit"] = dataset[
        "constructor_win_rate_at_circuit"
    ].fillna(0)
    dataset["driver_name"] = (
        dataset["forename"].fillna("") + " " + dataset["surname"].fillna("")
    ).str.strip()

    columns = [
        "raceId",
        "year",
        "round",
        "race_name",
        "circuit_name",
        "date",
        "driverId",
        "driver_name",
        "code",
        "constructorId",
        "constructor_name",
        *FEATURES,
        TARGET,
        "position_num",
        "points",
    ]
    return dataset[columns].sort_values(
        ["year", "round", "raceId"], kind="mergesort"
    ).reset_index(drop=True)


def main() -> None:
    datasets = load_raw_data()
    processed = build_processed_dataset(datasets)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(OUT_PATH, index=False)
    print(f"Processed dataset saved to: {OUT_PATH}")
    print(f"Rows: {len(processed):,} | Podium rate: {processed[TARGET].mean():.3%}")


if __name__ == "__main__":
    main()
