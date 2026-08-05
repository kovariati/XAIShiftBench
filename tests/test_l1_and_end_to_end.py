from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from xaishiftbench.metrics import (
    attribution_l1_diagnostics,
    energy_u_statistic_squared,
    normalize_signed_rows_with_mask,
)


def test_l1_diagnostics_exposes_near_zero_rows() -> None:
    values = np.array([[0.0, 0.0], [1e-10, -1e-10], [2.0, -1.0]])
    d = attribution_l1_diagnostics(values, thresholds=(1e-12, 1e-8))
    assert d["n_rows"] == 3
    assert d["zero_row_fraction"] == 1 / 3
    assert d["fraction_l1_le_1e-12"] == 1 / 3
    assert d["fraction_l1_le_1e-08"] == 2 / 3


def test_thresholded_normalization_is_explicit() -> None:
    values = np.array([[0.0, 0.0], [1e-10, -1e-10], [2.0, -1.0]])
    normalized, keep = normalize_signed_rows_with_mask(values, min_l1=1e-8)
    assert keep.tolist() == [False, False, True]
    np.testing.assert_allclose(normalized[2], [2 / 3, -1 / 3])
    np.testing.assert_allclose(normalized[:2], 0.0)


def test_synthetic_end_to_end_fixture() -> None:
    rng = np.random.default_rng(2026080103)
    x_source = rng.normal(size=(120, 5))
    x_target = rng.normal(loc=[0.5, 0, 0, 0, 0], size=(100, 5))
    beta = np.array([1.2, -0.8, 0.5, 0.0, 0.0])
    y = (x_source @ beta + rng.normal(scale=0.8, size=120) > 0).astype(int)
    model_a = LogisticRegression(C=1.0, max_iter=1000, random_state=1).fit(x_source, y)
    model_b = LogisticRegression(C=1.0, max_iter=1000, random_state=2).fit(x_source[::-1], y[::-1])
    center = x_source.mean(axis=0)
    phi_source_a = (x_source[:60] - center) * model_a.coef_[0]
    phi_source_b = (x_source[60:] - center) * model_b.coef_[0]
    phi_target_a = (x_target[:60] - center) * model_a.coef_[0]
    qs, keep_s = normalize_signed_rows_with_mask(phi_source_a)
    qn, keep_n = normalize_signed_rows_with_mask(phi_source_b)
    qt, keep_t = normalize_signed_rows_with_mask(phi_target_a)
    assert keep_s.all() and keep_n.all() and keep_t.all()
    shift = energy_u_statistic_squared(qs, qt)
    matched_null = energy_u_statistic_squared(qs, qn)
    assert np.isfinite(shift - matched_null)


def test_combine_domain_pair_outputs_oulad(tmp_path):
    import json
    import pandas as pd
    from combine_domain_pair_outputs import combine

    pair_dir = tmp_path / "outputs" / "oulad" / "pair_runs"
    pair_dir.mkdir(parents=True)
    for pair in range(2):
        pd.DataFrame({"refit_pair": [pair], "task_index": [0], "model": ["logistic"], "value": [pair]}).to_csv(
            pair_dir / f"pair_{pair:02d}_rows.csv", index=False
        )
        pd.DataFrame({"refit_pair": [pair], "task_index": [0], "model": ["logistic"], "feature": ["x"], "value": [pair]}).to_csv(
            pair_dir / f"pair_{pair:02d}_features.csv", index=False
        )
        (pair_dir / f"pair_{pair:02d}_metadata.json").write_text(json.dumps({"pair": pair}), encoding="utf-8")
    payload = combine(tmp_path, "oulad", 2)
    assert payload["n_row_records"] == 2
    output = tmp_path / "outputs" / "oulad" / "oulad_temporal_refits.csv"
    assert output.exists()
    assert pd.read_csv(output).refit_pair.tolist() == [0, 1]
