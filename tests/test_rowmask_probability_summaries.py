from __future__ import annotations

import numpy as np
import pandas as pd

from xaishiftbench.credit_prediction_invariance import (
    PROBABILITY_BIN_LABELS,
    _probability_bin_rows,
    _row_mask_jaccard,
    _summarize_scenario,
)


def test_row_mask_jaccard_handles_empty_union() -> None:
    a = np.array([[False, False], [True, False], [True, True]])
    b = np.array([[False, False], [True, False], [False, True]])
    result = _row_mask_jaccard(a, b)
    assert np.allclose(result, [1.0, 1.0, 0.5])


def test_probability_bins_partition_rows() -> None:
    frame = pd.DataFrame(
        {
            "abs_probability_shift": [0.0, 0.01, 0.02, 0.04, 0.08, 0.2],
            "explanation_stv": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            "prediction_label_unchanged": [True] * 6,
        }
    )
    rows = _probability_bin_rows(frame)
    assert [row["probability_bin"] for row in rows] == list(PROBABILITY_BIN_LABELS)
    assert sum(row["n_observations"] for row in rows) == len(frame)


def test_scenario_summary_probability_invariant_counts() -> None:
    frame = pd.DataFrame(
        {
            "abs_probability_shift": [0.0, 0.005, 0.02, 0.06],
            "explanation_stv": [0.0, 0.2, 0.3, 0.4],
            "prediction_label_unchanged": [True, True, True, False],
            "reference_row_missing": [0, 1, 2, 3],
            "target_row_missing": [0, 1, 2, 3],
            "mask_jaccard": [1.0, 0.5, 0.5, 0.0],
        }
    )
    result = _summarize_scenario(frame)
    assert result["prob_invariant_01_count"] == 2
    assert np.isclose(result["prob_invariant_01_mean_stv"], 0.1)
    assert np.isclose(result["label_invariant_rate"], 0.75)
