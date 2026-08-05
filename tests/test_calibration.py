from __future__ import annotations

import numpy as np

from xaishiftbench.calibrated_shift import (
    calibrated_attribution_shift,
    matched_composition_indices,
)
from xaishiftbench.metrics import (
    energy_u_statistic_squared,
    energy_v_statistic_squared,
    kendall_tau_b,
    top_k_jaccard,
)


def test_energy_u_is_null_centered_while_v_is_positive() -> None:
    rng = np.random.default_rng(20260731)
    u_values = []
    v_values = []
    for _ in range(250):
        x = rng.normal(size=(24, 4))
        y = rng.normal(size=(24, 4))
        u_values.append(energy_u_statistic_squared(x, y))
        v_values.append(energy_v_statistic_squared(x, y))
    assert abs(float(np.mean(u_values))) < 0.02
    assert float(np.mean(v_values)) > 0.10


def test_energy_u_detects_shift() -> None:
    rng = np.random.default_rng(12)
    x = rng.normal(size=(100, 3))
    y = rng.normal(loc=0.8, size=(100, 3))
    assert energy_u_statistic_squared(x, y) > 0.5


def test_matched_indices_are_disjoint_and_composition_matched() -> None:
    ys = np.array([0] * 90 + [1] * 60)
    yt = np.array([0] * 40 + [1] * 60)
    result = matched_composition_indices(ys, yt, cap=40, seed=7)
    assert len(result.source_shift) == len(result.source_null) == len(result.target_shift)
    assert np.intersect1d(result.source_shift, result.source_null).size == 0
    assert ys[result.source_null].sum() == yt[result.target_shift].sum()
    assert ys[result.source_shift].sum() == result.source_positive_count


def test_calibrated_shift_null_and_signal() -> None:
    rng = np.random.default_rng(9)
    ys = np.array([0] * 100 + [1] * 100)
    yt = np.array([0] * 100 + [1] * 100)
    source_a = rng.normal(size=(200, 5))
    source_b = source_a + rng.normal(scale=0.01, size=source_a.shape)
    target_null = rng.normal(size=(200, 5))
    null_result = calibrated_attribution_shift(
        source_a, target_null, source_b, ys, yt, seed=22, cap=80
    )
    target_shifted = rng.normal(loc=0.7, size=(200, 5))
    signal_result = calibrated_attribution_shift(
        source_a, target_shifted, source_b, ys, yt, seed=22, cap=80
    )
    assert signal_result.shift_u2 > null_result.shift_u2
    assert signal_result.excess_over_refit_sample_null_u2 > 0


def test_rank_undefined_and_topk_tie_policy() -> None:
    assert np.isnan(kendall_tau_b([1, 1, 1], [2, 2, 2]))
    # Ties are resolved by original index, deterministically.
    assert top_k_jaccard([1, 1, 0], [1, 1, 0], 1) == 1.0
