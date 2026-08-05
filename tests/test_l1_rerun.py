from __future__ import annotations

import numpy as np

from xaishiftbench.metrics import (
    matched_l1_sensitivity,
    prefixed_attribution_l1_diagnostics,
)


def test_prefixed_l1_diagnostics_keys() -> None:
    x = np.array([[0.0, 0.0], [1e-9, -1e-9], [1.0, -2.0]])
    out = prefixed_attribution_l1_diagnostics(x, "source_l1")
    assert out["source_l1_n_rows"] == 3
    assert out["source_l1_zero_row_fraction"] == 1 / 3
    assert "source_l1_fraction_l1_le_1e-08" in out


def test_matched_l1_sensitivity_thresholds_and_strata() -> None:
    source = np.array([[1.0, 0.0], [0.4, -0.6], [1e-10, 0.0], [0.2, 0.8]])
    null = np.array([[0.9, 0.1], [0.5, -0.5], [1e-10, 0.0], [0.3, 0.7]])
    target = np.array([[0.0, 1.0], [-0.6, 0.4], [1e-10, 0.0], [0.8, 0.2]])
    out = matched_l1_sensitivity(source, null, target, thresholds=(1e-12, 1e-8))
    assert out["l1thr_1em12_source_n"] == 4
    assert out["l1thr_1em08_source_n"] == 3
    assert np.isfinite(out["l1thr_1em08_excess_u2"])
    assert "l1stratum_q4_excess_u2" in out
