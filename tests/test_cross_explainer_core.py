from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xaishiftbench.cross_explainer_audit import (
    _row_faithfulness,
    _row_stv,
    _safe_corr,
    _stratified_positions,
)


def test_cross_explainer_row_stv_identity() -> None:
    a = np.array([[0.5, -0.5], [0.2, -0.8]])
    assert np.allclose(_row_stv(a, a), 0.0)


def test_cross_explainer_stratified_positions() -> None:
    y = np.array([0] * 80 + [1] * 20)
    positions = _stratified_positions(y, 20, 123)
    assert len(positions) == 20
    assert len(np.unique(positions)) == 20
    assert int(y[positions].sum()) == 4


def test_cross_explainer_safe_corr_constant() -> None:
    assert np.isnan(_safe_corr(np.ones(5), np.arange(5)))


def test_cross_explainer_faithfulness_perfect_order() -> None:
    attr = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    signed, absolute = _row_faithfulness(attr, attr.copy())
    assert np.isclose(signed, 1.0)
    assert np.isclose(absolute, 1.0)


def test_explainer_registry_names() -> None:
    from xaishiftbench.cross_explainer_audit import EXPLAINERS

    assert EXPLAINERS == ("model_specific_shap", "kernel_shap")


def test_lightgbm_native_shap_additivity() -> None:
    from pathlib import Path

    from xaishiftbench.credit_models import fit_credit_model
    from xaishiftbench.cross_explainer_audit import _model_specific_shap
    from xaishiftbench.datasets.south_german_credit import FEATURES, load_south_german_credit

    path = Path(__file__).resolve().parents[1] / "data" / "south_german_credit" / "raw" / "SouthGermanCredit.asc"
    if not path.exists():
        pytest.skip("South German Credit raw data are not redistributed in the code-only archive.")
    data = load_south_german_credit(path).frame
    train = pd.concat(
        [data[data["target"] == 0].iloc[:180], data[data["target"] == 1].iloc[:90]],
        ignore_index=True,
    )
    evaluation = data.iloc[300:312][FEATURES].reset_index(drop=True)
    background = train[FEATURES].iloc[:24].reset_index(drop=True)
    model = fit_credit_model(
        "lightgbm", "none", train, train["target"].to_numpy(int), seed=2026073066
    )
    values, _, error = _model_specific_shap(model, evaluation, background)
    assert values.shape == (len(evaluation), len(FEATURES))
    assert error < 1e-10
