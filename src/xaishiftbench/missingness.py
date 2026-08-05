"""Deterministic controlled missingness mechanisms for tabular stress tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .datasets.south_german_credit import FEATURES, NUMERIC_FEATURES

MAR_DRIVERS = ["status", "duration", "credit_history", "age"]
MAR_ELIGIBLE = [name for name in FEATURES if name not in MAR_DRIVERS]
MNAR_ELIGIBLE = [
    "status",
    "duration",
    "credit_history",
    "amount",
    "savings",
    "employment_duration",
    "installment_rate",
    "present_residence",
    "property",
    "age",
    "number_credits",
    "job",
    "people_liable",
    "telephone",
    "foreign_worker",
]
BLOCK_FEATURES = [
    "status",
    "savings",
    "employment_duration",
    "installment_rate",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
]


@dataclass(frozen=True)
class MissingnessResult:
    frame: pd.DataFrame
    mask: pd.DataFrame
    mechanism: str
    rate: float
    missing_cells: int
    eligible_cells: int


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    sd = values.std(ddof=0)
    if sd <= 1e-12:
        return np.zeros_like(values)
    return (values - values.mean()) / sd


def _row_mar_score(frame: pd.DataFrame) -> np.ndarray:
    status = _zscore(frame["status"].to_numpy(float))
    duration = _zscore(np.log1p(frame["duration"].to_numpy(float)))
    history = _zscore(frame["credit_history"].to_numpy(float))
    age = _zscore(frame["age"].to_numpy(float))
    score = 0.75 * status + 0.65 * duration - 0.45 * history - 0.35 * age
    return score


def _own_value_score(frame: pd.DataFrame, feature: str) -> np.ndarray:
    series = frame[feature]
    if feature in NUMERIC_FEATURES:
        pct = series.rank(method="average", pct=True).to_numpy(float)
        return 2.0 * np.abs(pct - 0.5)
    frequency = series.map(series.value_counts(normalize=True)).to_numpy(float)
    rarity = 1.0 / np.maximum(frequency, 1e-12)
    rarity /= rarity.mean()
    return rarity


def _weighted_exact_cells(
    eligible: np.ndarray,
    weights: np.ndarray,
    n_select: int,
    rng: np.random.Generator,
) -> np.ndarray:
    indices = np.flatnonzero(eligible.ravel())
    if n_select < 0 or n_select > len(indices):
        raise ValueError(f"Cannot select {n_select} cells from {len(indices)} eligible cells.")
    if n_select == 0:
        return np.zeros_like(eligible, dtype=bool)
    flat_weights = np.asarray(weights, dtype=float).ravel()[indices]
    flat_weights = np.nan_to_num(flat_weights, nan=0.0, posinf=0.0, neginf=0.0)
    flat_weights = np.maximum(flat_weights, 1e-12)
    probabilities = flat_weights / flat_weights.sum()
    selected = rng.choice(indices, size=n_select, replace=False, p=probabilities)
    out = np.zeros(eligible.size, dtype=bool)
    out[selected] = True
    return out.reshape(eligible.shape)


def inject_missingness(
    frame: pd.DataFrame,
    mechanism: str,
    rate: float,
    seed: int,
) -> MissingnessResult:
    """Inject an exact cell budget under MCAR, MAR, MNAR, or block missingness.

    ``rate`` is the fraction of all ``n_rows * 20`` predictor cells, not merely
    the fraction among eligible cells. MAR driver variables remain observed.
    MNAR probabilities depend on the value of the cell that becomes missing.
    """
    mechanism = mechanism.upper()
    if mechanism not in {"CLEAN", "MCAR", "MAR", "MNAR", "BLOCK"}:
        raise ValueError(f"Unknown mechanism: {mechanism}")
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be in [0, 1].")
    x = frame[FEATURES].copy()
    n_rows, n_features = x.shape
    target_cells = int(round(rate * n_rows * n_features))
    mask = np.zeros((n_rows, n_features), dtype=bool)
    rng = np.random.default_rng(seed)

    if mechanism == "CLEAN":
        if rate != 0:
            raise ValueError("CLEAN mechanism requires rate=0.")
        eligible_count = n_rows * n_features
    elif mechanism == "MCAR":
        eligible = np.ones_like(mask, dtype=bool)
        weights = np.ones_like(mask, dtype=float)
        mask = _weighted_exact_cells(eligible, weights, target_cells, rng)
        eligible_count = int(eligible.sum())
    elif mechanism == "MAR":
        eligible = np.zeros_like(mask, dtype=bool)
        eligible_columns = [FEATURES.index(name) for name in MAR_ELIGIBLE]
        eligible[:, eligible_columns] = True
        row_weight = np.exp(np.clip(_row_mar_score(x), -4.0, 4.0))
        feature_weight = np.linspace(0.8, 1.2, len(FEATURES))[None, :]
        weights = row_weight[:, None] * feature_weight
        mask = _weighted_exact_cells(eligible, weights, target_cells, rng)
        eligible_count = int(eligible.sum())
    elif mechanism == "MNAR":
        eligible = np.zeros_like(mask, dtype=bool)
        weights = np.zeros_like(mask, dtype=float)
        for name in MNAR_ELIGIBLE:
            j = FEATURES.index(name)
            eligible[:, j] = True
            score = _own_value_score(x, name)
            weights[:, j] = np.exp(np.clip(1.25 * score, -4.0, 4.0))
        mask = _weighted_exact_cells(eligible, weights, target_cells, rng)
        eligible_count = int(eligible.sum())
    else:  # BLOCK
        block_width = len(BLOCK_FEATURES)
        if target_cells % block_width != 0:
            raise ValueError(
                f"BLOCK target budget {target_cells} must be divisible by block width {block_width}."
            )
        n_block_rows = target_cells // block_width
        if n_block_rows > n_rows:
            raise ValueError("Requested block rate requires more rows than available.")
        row_weight = np.exp(np.clip(_row_mar_score(x), -4.0, 4.0))
        selected_rows = rng.choice(
            np.arange(n_rows), size=n_block_rows, replace=False, p=row_weight / row_weight.sum()
        )
        block_columns = [FEATURES.index(name) for name in BLOCK_FEATURES]
        mask[np.ix_(selected_rows, block_columns)] = True
        eligible_count = n_rows * block_width

    if int(mask.sum()) != target_cells:
        raise AssertionError(f"Expected {target_cells} missing cells, generated {int(mask.sum())}.")
    corrupted = x.mask(mask)
    mask_frame = pd.DataFrame(mask, columns=FEATURES, index=frame.index)
    return MissingnessResult(
        frame=corrupted,
        mask=mask_frame,
        mechanism=mechanism,
        rate=rate,
        missing_cells=int(mask.sum()),
        eligible_cells=eligible_count,
    )
