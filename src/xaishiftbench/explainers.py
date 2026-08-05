"""Model-specific attribution adapters with a common semantic output."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from .semantic import SemanticMap


def _dense_encoded(values: object) -> NDArray[np.float64]:
    """Convert dense or scipy-sparse encoded matrices to a finite dense array."""
    if hasattr(values, "toarray"):
        values = values.toarray()
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"encoded values must be two-dimensional; got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("encoded values contain non-finite entries.")
    return arr


@dataclass(frozen=True)
class AttributionResult:
    encoded: NDArray[np.float64]
    semantic: NDArray[np.float64]
    base_value: float


def explain_logistic_centered(
    model: LogisticRegression,
    encoded_x: NDArray[np.float64],
    encoded_background: NDArray[np.float64],
    semantic_map: SemanticMap,
) -> AttributionResult:
    """Exact linear log-odds decomposition around the source background mean."""
    coef = np.asarray(model.coef_[0], dtype=float)
    background_arr = _dense_encoded(encoded_background)
    x_arr = _dense_encoded(encoded_x)
    center = background_arr.mean(axis=0)
    encoded = (x_arr - center[None, :]) * coef[None, :]
    base = float(model.intercept_[0] + center @ coef)
    semantic = semantic_map.aggregate(encoded)
    return AttributionResult(encoded=encoded, semantic=semantic, base_value=base)


def explain_lightgbm_tree_shap(
    model: object,
    encoded_x: NDArray[np.float64],
    semantic_map: SemanticMap,
) -> AttributionResult:
    """TreeSHAP decomposition for LightGBM binary classifiers."""
    import shap

    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="LightGBM binary classifier with TreeExplainer shap values output has changed.*",
        )
        values = explainer.shap_values(
            _dense_encoded(encoded_x), check_additivity=False
        )
    if isinstance(values, list):
        values = values[-1]
    encoded = np.asarray(values, dtype=float)
    if encoded.ndim == 3:
        encoded = encoded[:, :, -1]
    expected = explainer.expected_value
    if isinstance(expected, (list, tuple, np.ndarray)):
        expected = np.asarray(expected).reshape(-1)[-1]
    semantic = semantic_map.aggregate(encoded)
    return AttributionResult(encoded=encoded, semantic=semantic, base_value=float(expected))


def explain_lightgbm_native_contrib(
    model: object,
    encoded_x: NDArray[np.float64],
    semantic_map: SemanticMap,
) -> AttributionResult:
    """Native LightGBM contribution decomposition in raw-score space.

    LightGBM returns one contribution per encoded feature and a final expected
    value column. This path is substantially faster than repeatedly creating a
    SHAP explainer and is algebraically checked in the benchmark tests.
    """
    booster = getattr(model, "booster_", None)
    if booster is None:
        raise ValueError("model must be a fitted LightGBM scikit-learn classifier.")
    contribution = np.asarray(
        booster.predict(_dense_encoded(encoded_x), pred_contrib=True), dtype=float
    )
    encoded_width = _dense_encoded(encoded_x).shape[1]
    if contribution.ndim != 2 or contribution.shape[1] != encoded_width + 1:
        raise ValueError(f"Unexpected LightGBM contribution shape: {contribution.shape}.")
    encoded = contribution[:, :-1]
    base_values = contribution[:, -1]
    if not np.allclose(base_values, base_values[0], atol=1e-10, rtol=1e-10):
        raise AssertionError("LightGBM returned non-constant expected values.")
    semantic = semantic_map.aggregate(encoded)
    return AttributionResult(encoded=encoded, semantic=semantic, base_value=float(base_values[0]))
