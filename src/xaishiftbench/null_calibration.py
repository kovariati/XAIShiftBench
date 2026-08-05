"""Calibration of shift-induced changes against independent no-shift refit contrasts."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _null_array(null_values: ArrayLike) -> np.ndarray:
    arr = np.asarray(null_values, dtype=float).reshape(-1)
    if arr.size < 4:
        raise ValueError("At least four independent null contrasts are required.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("null_values contains non-finite values.")
    return arr


def standardized_excess(observed: float, null_values: ArrayLike, eps: float = 1e-12) -> float:
    """Robust standardized excess above ordinary no-shift refit variability.

    SEEI = (observed - median(null)) / IQR(null). It is a descriptive effect
    scale, not a universal probability or safety threshold.
    """
    if not np.isfinite(observed):
        raise ValueError("observed must be finite.")
    arr = _null_array(null_values)
    q25, median, q75 = np.quantile(arr, [0.25, 0.50, 0.75])
    return float((observed - median) / max(q75 - q25, eps))


def empirical_exceedance_fraction(observed: float, null_values: ArrayLike) -> float:
    """Plus-one upper-tail exceedance fraction for a descriptive null reference.

    This quantity is deliberately not named a p-value because domain-shift and
    no-shift contrasts need not be exchangeable.
    """
    if not np.isfinite(observed):
        raise ValueError("observed must be finite.")
    arr = _null_array(null_values)
    return float((1 + np.count_nonzero(arr >= observed)) / (arr.size + 1))


def empirical_upper_tail_p(observed: float, null_values: ArrayLike) -> float:
    """Backward-compatible alias; prefer empirical_exceedance_fraction."""
    return empirical_exceedance_fraction(observed, null_values)


def paired_excess(shift_values: ArrayLike, null_values: ArrayLike) -> np.ndarray:
    shift = np.asarray(shift_values, dtype=float).reshape(-1)
    null = np.asarray(null_values, dtype=float).reshape(-1)
    if shift.shape != null.shape:
        raise ValueError("shift_values and null_values must have identical shapes.")
    if shift.size < 4 or not np.all(np.isfinite(shift)) or not np.all(np.isfinite(null)):
        raise ValueError("At least four finite paired contrasts are required.")
    return shift - null
