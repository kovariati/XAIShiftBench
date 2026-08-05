"""UCI Student Performance adapter for institutional explanation-shift analysis.

The UCI canonical ``data.csv`` contains the 649-record Portuguese-language
course table. The benchmark treats final grades below 10/20 as an adverse
outcome and removes the school identifier from all predictive representations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CATEGORICAL_BASE = [
    "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason",
    "guardian", "schoolsup", "famsup", "paid", "activities", "nursery",
    "higher", "internet", "romantic",
]
NUMERIC_BASE = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel",
    "freetime", "goout", "Dalc", "Walc", "health", "absences",
]
EARLY_FEATURES = NUMERIC_BASE + CATEGORICAL_BASE
LATE_FEATURES = NUMERIC_BASE + ["G1", "G2"] + CATEGORICAL_BASE


@dataclass(frozen=True)
class StudentPerformanceData:
    frame: pd.DataFrame

    def features(self, representation: str) -> tuple[list[str], list[str], list[str]]:
        representation = representation.lower()
        if representation == "early":
            numeric = list(NUMERIC_BASE)
            categorical = list(CATEGORICAL_BASE)
        elif representation == "late":
            numeric = list(NUMERIC_BASE) + ["G1", "G2"]
            categorical = list(CATEGORICAL_BASE)
        else:
            raise ValueError("representation must be 'early' or 'late'.")
        return numeric + categorical, numeric, categorical


def load_student_performance(path: str | Path) -> StudentPerformanceData:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source)
    expected = {
        "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
        "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime", "failures",
        "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet",
        "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health",
        "absences", "G1", "G2", "G3",
    }
    missing = sorted(expected - set(frame.columns))
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    frame = frame[list(expected)].copy()
    if frame.shape != (649, 33):
        raise ValueError(f"Expected 649 rows and 33 columns, got {frame.shape}.")
    if int(frame.isna().sum().sum()) != 0:
        raise ValueError("The canonical UCI Student Performance file should have no missing values.")
    counts = frame["school"].value_counts().to_dict()
    if counts != {"GP": 423, "MS": 226}:
        raise ValueError(f"Unexpected school counts: {counts}")
    if frame["G3"].min() < 0 or frame["G3"].max() > 20:
        raise ValueError("G3 is outside the documented 0-20 grading range.")
    # Adverse class: final grade below the Portuguese passing threshold of 10/20.
    frame["target"] = (frame["G3"] < 10).astype(int)
    frame["record_id"] = [f"student_{i:04d}" for i in range(len(frame))]
    return StudentPerformanceData(frame=frame)
