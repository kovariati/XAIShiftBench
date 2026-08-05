from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xaishiftbench.cross_explainer_audit import (
    _global_profiles,
    _row_stv,
    _safe_corr,
    _stratified_positions,
)
from xaishiftbench.datasets.south_german_credit import FEATURES
from xaishiftbench.missingness import BLOCK_FEATURES, MAR_DRIVERS, inject_missingness


def _credit_like_frame(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(2026080104)
    data: dict[str, np.ndarray] = {}
    for j, feature in enumerate(FEATURES):
        if feature in {"duration", "amount", "age"}:
            data[feature] = rng.integers(1, 100 + 20 * j, size=n)
        else:
            data[feature] = rng.integers(1, 5 + (j % 3), size=n)
    data["target"] = rng.integers(0, 2, size=n)
    return pd.DataFrame(data)


@pytest.mark.parametrize("mechanism", ["MCAR", "MAR", "MNAR", "BLOCK"])
def test_missingness_mechanisms_exact_budget_and_determinism(mechanism: str) -> None:
    frame = _credit_like_frame()
    rate = 0.15
    a = inject_missingness(frame, mechanism, rate, seed=123)
    b = inject_missingness(frame, mechanism, rate, seed=123)
    assert a.missing_cells == round(rate * len(frame) * len(FEATURES))
    pd.testing.assert_frame_equal(a.mask, b.mask)
    pd.testing.assert_frame_equal(a.frame, b.frame)
    assert int(a.frame.isna().sum().sum()) == a.missing_cells
    if mechanism == "MAR":
        assert not a.mask[MAR_DRIVERS].to_numpy().any()
    if mechanism == "BLOCK":
        affected = a.mask.any(axis=1)
        assert (a.mask.loc[affected, BLOCK_FEATURES].sum(axis=1) == len(BLOCK_FEATURES)).all()
        assert not a.mask.loc[affected, [c for c in FEATURES if c not in BLOCK_FEATURES]].to_numpy().any()


def test_clean_and_missingness_validation_errors() -> None:
    frame = _credit_like_frame(20)
    clean = inject_missingness(frame, "CLEAN", 0.0, seed=1)
    assert clean.missing_cells == 0
    assert not clean.frame.isna().to_numpy().any()
    with pytest.raises(ValueError):
        inject_missingness(frame, "CLEAN", 0.1, seed=1)
    with pytest.raises(ValueError):
        inject_missingness(frame, "UNKNOWN", 0.1, seed=1)
    with pytest.raises(ValueError):
        inject_missingness(frame, "MCAR", 1.1, seed=1)
    with pytest.raises(ValueError):
        inject_missingness(frame, "BLOCK", 0.013, seed=1)


def test_cross_explainer_helpers() -> None:
    a = np.array([[0.6, -0.4], [0.2, -0.8]])
    b = np.array([[0.5, -0.5], [0.0, -1.0]])
    np.testing.assert_allclose(_row_stv(a, b), [0.1, 0.2])
    abs_profile, signed_profile = _global_profiles(a)
    np.testing.assert_allclose(abs_profile, [0.4, 0.6])
    np.testing.assert_allclose(signed_profile, [0.4, -0.6])
    assert np.isfinite(_safe_corr(np.arange(5), np.arange(5), "spearman"))
    assert np.isnan(_safe_corr(np.ones(5), np.arange(5), "spearman"))
    y = np.array([0] * 70 + [1] * 30)
    pos = _stratified_positions(y, 20, seed=99)
    assert len(pos) == 20
    assert len(np.unique(pos)) == 20
    assert 1 <= int(y[pos].sum()) <= 19
