"""South German Credit data adapter with explicit semantic typing.

The corrected data file uses German column names. This adapter maps them to
stable English names, validates the published 700/300 class composition, and
uses class 1 for adverse credit risk in benchmark outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RAW_TO_ENGLISH = {
    "laufkont": "status",
    "laufzeit": "duration",
    "moral": "credit_history",
    "verw": "purpose",
    "hoehe": "amount",
    "sparkont": "savings",
    "beszeit": "employment_duration",
    "rate": "installment_rate",
    "famges": "personal_status_sex",
    "buerge": "other_debtors",
    "wohnzeit": "present_residence",
    "verm": "property",
    "alter": "age",
    "weitkred": "other_installment_plans",
    "wohn": "housing",
    "bishkred": "number_credits",
    "beruf": "job",
    "pers": "people_liable",
    "telef": "telephone",
    "gastarb": "foreign_worker",
    "kredit": "credit_good",
}

NUMERIC_FEATURES = ["duration", "amount", "age"]
CATEGORICAL_FEATURES = [
    "status",
    "credit_history",
    "purpose",
    "savings",
    "employment_duration",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "present_residence",
    "property",
    "other_installment_plans",
    "housing",
    "number_credits",
    "job",
    "people_liable",
    "telephone",
    "foreign_worker",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class SouthGermanCreditData:
    frame: pd.DataFrame
    numeric_features: tuple[str, ...] = tuple(NUMERIC_FEATURES)
    categorical_features: tuple[str, ...] = tuple(CATEGORICAL_FEATURES)


def load_south_german_credit(path: str | Path) -> SouthGermanCreditData:
    """Load and validate the corrected South German Credit ASCII file."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source, sep=r"\s+")
    missing_columns = sorted(set(RAW_TO_ENGLISH) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")
    frame = frame[list(RAW_TO_ENGLISH)].rename(columns=RAW_TO_ENGLISH)
    if frame.shape != (1000, 21):
        raise ValueError(f"Expected 1000 rows and 21 columns, got {frame.shape}.")
    if int(frame.isna().sum().sum()) != 0:
        raise ValueError("Corrected South German Credit source must contain no missing values.")
    counts = frame["credit_good"].value_counts().to_dict()
    if counts != {1: 700, 0: 300}:
        raise ValueError(f"Expected 700 good and 300 bad cases, got {counts}.")
    # Benchmark convention: positive class is adverse credit risk.
    frame["target"] = (frame["credit_good"] == 0).astype(int)
    frame["record_id"] = [f"sgc_{i:04d}" for i in range(len(frame))]
    return SouthGermanCreditData(frame=frame)
