from __future__ import annotations

import numpy as np
import pandas as pd

from xaishiftbench.metrics import (
    energy_u_statistic_squared,
    weighted_energy_u_statistic_squared,
    matched_l1_sensitivity,
    normalize_signed_rows,
)
from xaishiftbench.calibrated_shift import matched_composition_indices


def test_weighted_energy_equal_weights_matches_unweighted_u_statistic():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(9, 4))
    y = rng.normal(size=(7, 4))
    got = weighted_energy_u_statistic_squared(x, y, np.ones(len(x)), np.ones(len(y)))
    want = energy_u_statistic_squared(x, y)
    assert np.isclose(got, want, atol=1e-12, rtol=1e-12)


def test_weighted_energy_is_symmetric_under_exchange():
    rng = np.random.default_rng(18)
    x = rng.normal(size=(8, 3))
    y = rng.normal(size=(10, 3))
    wx = rng.uniform(0.2, 3.0, size=len(x))
    wy = rng.uniform(0.2, 3.0, size=len(y))
    a = weighted_energy_u_statistic_squared(x, y, wx, wy)
    b = weighted_energy_u_statistic_squared(y, x, wy, wx)
    assert np.isclose(a, b, atol=1e-12, rtol=1e-12)


def test_matched_sampling_source_shift_and_null_have_documented_prevalences():
    ys = np.array([0] * 70 + [1] * 30)
    yt = np.array([0] * 45 + [1] * 55)
    idx = matched_composition_indices(ys, yt, cap=40, seed=123, min_per_class=3)
    assert np.isclose(ys[idx.source_shift].mean(), idx.source_positive_count / idx.sample_size)
    assert np.isclose(ys[idx.source_null].mean(), idx.target_positive_count / idx.sample_size)
    assert np.isclose(yt[idx.target_shift].mean(), idx.target_positive_count / idx.sample_size)
    assert set(idx.source_shift).isdisjoint(set(idx.source_null))


def test_threshold_sensitivity_uses_primary_normalization_convention():
    # Large enough amplitudes make the epsilon irrelevant numerically, but the
    # expected value is explicitly computed with the primary helper.
    a = np.array([[1.0, -2.0], [2.0, 1.0], [3.0, -1.0]])
    b = np.array([[0.5, -1.0], [1.5, 1.0], [2.0, -0.5]])
    t = np.array([[1.0, -1.5], [2.2, 0.8], [2.7, -0.7]])
    out = matched_l1_sensitivity(a, b, t, thresholds=(1e-12,), amplitude_strata=2)
    na, nb, nt = map(normalize_signed_rows, (a, b, t))
    want = energy_u_statistic_squared(na, nt) - energy_u_statistic_squared(na, nb)
    assert np.isclose(out['l1thr_1em12_excess_u2'], want, atol=1e-12, rtol=1e-12)


def test_scientific_full_plan_contains_parity_before_finalization_and_not_scopus():
    from types import SimpleNamespace
    import reproduce
    ns = SimpleNamespace(oulad_zip=None, acs_2018_a=None, acs_2018_b=None, acs_2024_a=None, acs_2024_b=None)
    cmds = reproduce.scientific_full_commands(ns)
    texts = [' '.join(c) for c in cmds]
    i_parity = next(i for i, s in enumerate(texts) if 'compare_raw_rerun_parity.py' in s)
    i_analysis = next(i for i, s in enumerate(texts) if 'analyze_extended_results.py' in s)
    i_finalize = next(i for i, s in enumerate(texts) if 'finalize_results.py' in s)
    assert i_parity < i_analysis < i_finalize
    assert all('--scopus' not in s for s in texts)


def test_reference_manifest_covers_four_domain_reference_tables():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    manifest = (root / 'reference_outputs/MANIFEST_SHA256_REFERENCE.txt').read_text()
    for name in [
        'heart_cross_site_refits.csv',
        'student_institution_refits.csv',
        'oulad_temporal_refits.csv',
        'acs_temporal_refits.csv',
    ]:
        assert name in manifest
