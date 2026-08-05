"""UCI Heart Disease four-site adapter with strict provenance checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

HEART_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

SITE_FILE_MAP = {
    "Cleveland": "processed.cleveland.data",
    "Hungary": "processed.hungarian.data",
    "Switzerland": "processed.switzerland.data",
    "VA Long Beach": "processed.va.data",
}

EXPECTED = {
    "Cleveland": {"n": 303, "positive": 139, "missing_cells": 6},
    "Hungary": {"n": 294, "positive": 106, "missing_cells": 782},
    "Switzerland": {"n": 123, "positive": 115, "missing_cells": 273},
    "VA Long Beach": {"n": 200, "positive": 149, "missing_cells": 698},
}

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class HeartDataset:
    frame: pd.DataFrame
    numeric_features: tuple[str, ...] = tuple(NUMERIC_FEATURES)
    categorical_features: tuple[str, ...] = tuple(CATEGORICAL_FEATURES)
    target: str = "target"
    site: str = "site"


def _read_one(path: Path, site: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing Heart Disease source file: {path}")
    frame = pd.read_csv(path, names=HEART_COLUMNS, na_values="?", dtype=float)
    if frame.shape[1] != len(HEART_COLUMNS):
        raise ValueError(f"Unexpected number of columns in {path}: {frame.shape[1]}")
    frame["target"] = (frame["num"] > 0).astype(int)
    frame["site"] = site
    frame["row_in_site"] = range(len(frame))
    return frame


def load_heart_sites(data_dir: str | Path, validate: bool = True) -> HeartDataset:
    root = Path(data_dir)
    frames = [_read_one(root / filename, site) for site, filename in SITE_FILE_MAP.items()]
    frame = pd.concat(frames, ignore_index=True)
    if validate:
        validate_heart_dataset(frame)
    return HeartDataset(frame=frame)


def validate_heart_dataset(frame: pd.DataFrame) -> None:
    required = set(HEART_COLUMNS + ["target", "site", "row_in_site"])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Heart Disease frame is missing columns: {sorted(missing)}")
    if frame["target"].isna().any() or not set(frame["target"].unique()).issubset({0, 1}):
        raise ValueError("Heart Disease binary target is invalid.")
    for site, expected in EXPECTED.items():
        subset = frame.loc[frame["site"] == site]
        observed = {
            "n": int(len(subset)),
            "positive": int(subset["target"].sum()),
            "missing_cells": int(subset[FEATURES].isna().sum().sum()),
        }
        if observed != expected:
            raise ValueError(f"Integrity check failed for {site}: {observed} != {expected}")
    if len(frame) != 920:
        raise ValueError(f"Expected 920 Heart Disease records, got {len(frame)}.")
