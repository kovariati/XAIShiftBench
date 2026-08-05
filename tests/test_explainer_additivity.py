from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from xaishiftbench.acs_models import fit_acs_model
from xaishiftbench.explainers import (
    explain_lightgbm_native_contrib,
    explain_lightgbm_tree_shap,
    explain_logistic_centered,
)
from xaishiftbench.semantic import SemanticMap


def test_centered_logistic_explainer_additivity_and_semantic_sum() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(80, 4))
    y = (x[:, 0] - x[:, 1] + 0.2 * x[:, 2] > 0).astype(int)
    model = LogisticRegression(max_iter=1000).fit(x, y)
    semantic_map = SemanticMap(names=("a", "b"), groups=((0, 1), (2, 3)))
    result = explain_logistic_centered(model, x[:20], x[20:60], semantic_map)
    reconstructed = result.base_value + result.encoded.sum(axis=1)
    np.testing.assert_allclose(reconstructed, model.decision_function(x[:20]), atol=1e-10)
    np.testing.assert_allclose(result.semantic.sum(axis=1), result.encoded.sum(axis=1))


def _acs_fixture(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(8)
    frame = pd.DataFrame({
        "AGEP": rng.integers(18, 80, n), "COW": rng.integers(1, 9, n),
        "SCHL": rng.integers(10, 25, n), "MAR": rng.integers(1, 6, n),
        "OCCP": rng.integers(10, 80, n), "POBP": rng.integers(1, 60, n),
        "RELP_HARM": rng.choice(["reference_person", "child", "spouse_or_partner"], n),
        "WKHP": rng.integers(1, 70, n), "SEX": rng.integers(1, 3, n),
        "RAC1P": rng.integers(1, 5, n),
    })
    frame["TARGET_NOMINAL_50K"] = (frame.AGEP + frame.WKHP + frame.SCHL > 85).astype(int)
    return frame


def test_native_and_treeshap_adapters_reconstruct_raw_score() -> None:
    frame = _acs_fixture()
    fitted = fit_acs_model("lightgbm", frame, frame.TARGET_NOMINAL_50K.to_numpy(int), seed=17)
    encoded = fitted.transform(frame.iloc[:12])
    native = explain_lightgbm_native_contrib(fitted.estimator, encoded, fitted.semantic_map)
    tree = explain_lightgbm_tree_shap(fitted.estimator, encoded, fitted.semantic_map)
    raw = fitted.raw_score(frame.iloc[:12])
    np.testing.assert_allclose(native.base_value + native.encoded.sum(axis=1), raw, atol=1e-8)
    np.testing.assert_allclose(tree.base_value + tree.encoded.sum(axis=1), raw, atol=1e-6)
