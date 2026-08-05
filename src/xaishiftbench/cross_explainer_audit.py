"""Cross-explainer validation under controlled credit missingness shifts.

This module tests whether explanation-shift conclusions depend on a single
attribution implementation. It compares model-specific SHAP values with model-agnostic KernelSHAP in raw-score space.
The same semantic 20-feature representation is used for both methods. KernelSHAP
and the logistic decomposition use a fixed source background; LightGBM uses its
native tree-path-dependent SHAP decomposition to guarantee exact additivity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.model_selection import train_test_split

from .credit_mechanism_matrix import _stratified_bootstrap
from .credit_models import CreditFittedModel, fit_credit_model
from .datasets.south_german_credit import FEATURES, load_south_german_credit
from .metrics import normalize_signed_rows, top_k_jaccard, weighted_sign_consistency
from .missingness import inject_missingness

TARGET_MECHANISMS = ("CLEAN", "MCAR", "MAR", "MNAR", "BLOCK")
RATE_BY_MECHANISM = {"CLEAN": 0.0, "MCAR": 0.15, "MAR": 0.15, "MNAR": 0.15, "BLOCK": 0.15}
EXPLAINERS = ("model_specific_shap", "kernel_shap")


@dataclass(frozen=True)
class CrossExplainerMetadata:
    runtime_seconds: float
    n_pairs: int
    n_model_fits: int
    n_explanation_runs: int
    n_observation_rows: int
    n_scenario_rows: int
    background_size: int
    explanation_sample_size: int
    kernel_nsamples: int


def _raw_score(model: CreditFittedModel, frame_or_array: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(frame_or_array, pd.DataFrame):
        encoded = model.transform(frame_or_array)
    else:
        raw = pd.DataFrame(np.asarray(frame_or_array), columns=FEATURES)
        encoded = model.transform(raw)
    if model.model_name == "logistic":
        return np.asarray(model.estimator.decision_function(encoded), dtype=float)
    return np.asarray(model.estimator.booster_.predict(encoded, raw_score=True), dtype=float)


def _model_specific_shap(
    model: CreditFittedModel,
    frame: pd.DataFrame,
    background: pd.DataFrame,
) -> tuple[np.ndarray, float, float]:
    """Exact model-specific semantic SHAP contributions in raw-score space."""
    encoded_x = model.transform(frame)
    encoded_background = model.transform(background)
    if model.model_name == "logistic":
        coef = np.asarray(model.estimator.coef_[0], dtype=float)
        center = encoded_background.mean(axis=0)
        encoded = (encoded_x - center[None, :]) * coef[None, :]
        base = float(model.estimator.intercept_[0] + center @ coef)
    elif model.model_name == "lightgbm":
        # LightGBM native pred_contrib provides exact raw-score, tree-path-dependent
        # SHAP values. The final column is the expected value. This route is used
        # because the finite-background interventional TreeExplainer backend failed
        # the preregistered additivity gate for a small subset of fitted models.
        contributions = np.asarray(
            model.estimator.booster_.predict(encoded_x, pred_contrib=True, raw_score=True),
            dtype=float,
        )
        if contributions.ndim != 2 or contributions.shape[1] != encoded_x.shape[1] + 1:
            raise RuntimeError(
                f"Unexpected LightGBM contribution shape: {contributions.shape}"
            )
        encoded = contributions[:, :-1]
        base_values = contributions[:, -1]
        if not np.allclose(base_values, base_values[0], atol=1e-12, rtol=0.0):
            raise RuntimeError("LightGBM native SHAP base value is not constant.")
        base = float(base_values[0])
    else:
        raise ValueError(model.model_name)
    semantic = model.semantic_map.aggregate(encoded)
    raw_score = _raw_score(model, frame)
    additivity_error = float(np.max(np.abs(base + semantic.sum(axis=1) - raw_score)))
    if additivity_error > 1e-7:
        raise AssertionError(
            f"Model-specific SHAP additivity gate failed: {additivity_error:.3e}"
        )
    return semantic, base, additivity_error


def _kernel_shap(
    model: CreditFittedModel,
    frame: pd.DataFrame,
    background: pd.DataFrame,
    seed: int,
    nsamples: int,
) -> tuple[np.ndarray, float, float]:
    """KernelSHAP over the 20 raw semantic features with a fixed source background."""
    import shap

    background_array = background[FEATURES].to_numpy(dtype=float)
    x_array = frame[FEATURES].to_numpy(dtype=float)

    def predict_raw(values: np.ndarray) -> np.ndarray:
        return _raw_score(model, values)

    explainer = shap.KernelExplainer(predict_raw, background_array, link="identity")
    previous_state = np.random.get_state()
    np.random.seed(seed)
    try:
        values = explainer.shap_values(
            x_array,
            nsamples=nsamples,
            l1_reg=f"num_features({len(FEATURES)})",
            silent=True,
        )
    finally:
        np.random.set_state(previous_state)
    semantic = np.asarray(values, dtype=float)
    if semantic.ndim == 3:
        semantic = semantic[:, :, -1]
    base = explainer.expected_value
    if isinstance(base, (list, tuple, np.ndarray)):
        base = np.asarray(base).reshape(-1)[-1]
    base = float(base)
    raw_score = _raw_score(model, frame)
    additivity_error = float(np.max(np.abs(base + semantic.sum(axis=1) - raw_score)))
    if additivity_error > 1e-5:
        raise AssertionError(
            f"KernelSHAP additivity gate failed: {additivity_error:.3e}"
        )
    return semantic, base, additivity_error


def _row_stv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum(np.abs(a - b), axis=1)


def _safe_corr(a: np.ndarray, b: np.ndarray, method: str = "spearman") -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if len(aa) < 3 or np.allclose(aa, aa[0]) or np.allclose(bb, bb[0]):
        return float("nan")
    if method == "spearman":
        return float(spearmanr(aa, bb).statistic)
    return float(kendalltau(aa, bb, variant="b").statistic)


def _global_profiles(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.mean(np.abs(values), axis=0), np.mean(values, axis=0)


def _faithfulness_replacement_effects(
    model: CreditFittedModel,
    frame: pd.DataFrame,
    training_frame: pd.DataFrame,
) -> np.ndarray:
    """Single-feature source-baseline replacement effects in raw-score space."""
    baseline: dict[str, float] = {}
    numeric = {"duration", "amount", "age"}
    for feature in FEATURES:
        series = training_frame[feature]
        if feature in numeric:
            baseline[feature] = float(series.median())
        else:
            baseline[feature] = float(series.mode(dropna=True).iloc[0])
    original = _raw_score(model, frame)
    effects = np.empty((len(frame), len(FEATURES)), dtype=float)
    for j, feature in enumerate(FEATURES):
        replaced = frame.copy()
        replaced[feature] = baseline[feature]
        effects[:, j] = original - _raw_score(model, replaced)
    return effects


def _row_faithfulness(attribution: np.ndarray, effects: np.ndarray) -> tuple[float, float]:
    signed: list[float] = []
    absolute: list[float] = []
    for attr_row, effect_row in zip(attribution, effects, strict=True):
        signed_value = _safe_corr(attr_row, effect_row, "spearman")
        absolute_value = _safe_corr(np.abs(attr_row), np.abs(effect_row), "spearman")
        if np.isfinite(signed_value):
            signed.append(signed_value)
        if np.isfinite(absolute_value):
            absolute.append(absolute_value)
    return (
        float(np.mean(signed)) if signed else float("nan"),
        float(np.mean(absolute)) if absolute else float("nan"),
    )


def _stratified_positions(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    y = np.asarray(y, dtype=int)
    counts = {label: int(round(n * np.mean(y == label))) for label in (0, 1)}
    counts[1] = max(1, counts[1])
    counts[0] = n - counts[1]
    for label in (0, 1):
        candidates = np.flatnonzero(y == label)
        take = min(counts[label], len(candidates))
        selected.extend(rng.choice(candidates, size=take, replace=False).tolist())
    if len(selected) < n:
        remaining = np.setdiff1d(np.arange(len(y)), np.asarray(selected, dtype=int))
        selected.extend(rng.choice(remaining, size=n - len(selected), replace=False).tolist())
    return np.asarray(sorted(selected), dtype=int)


def run_cross_explainer_audit(
    data_path: str | Path,
    seed_base: int = 2026073006,
    n_pairs: int = 5,
    pair_start: int = 0,
    models: tuple[str, ...] = ("logistic", "lightgbm"),
    background_size: int = 24,
    explanation_sample_size: int = 40,
    kernel_nsamples: int = 160,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, CrossExplainerMetadata]:
    """Run a clean-trained, target-missingness cross-explainer audit."""
    if n_pairs < 1:
        raise ValueError("n_pairs must be positive.")
    if explanation_sample_size < 10:
        raise ValueError("explanation_sample_size must be at least 10.")
    start = perf_counter()
    data = load_south_german_credit(data_path).frame
    train_idx, eval_idx = train_test_split(
        np.arange(len(data)), test_size=0.30, random_state=seed_base, stratify=data["target"]
    )
    train_pool = data.iloc[train_idx].reset_index(drop=True)
    eval_clean = data.iloc[eval_idx].reset_index(drop=True)
    y_eval = eval_clean["target"].to_numpy(int)
    positions = _stratified_positions(y_eval, explanation_sample_size, seed_base + 77)
    eval_sample_clean = eval_clean.iloc[positions].reset_index(drop=True)
    y_sample = eval_sample_clean["target"].to_numpy(int)

    observation_rows: list[dict[str, object]] = []
    scenario_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    n_fits = 0
    n_explanation_runs = 0

    for pair in range(pair_start, pair_start + n_pairs):
        pair_seed = seed_base + pair * 1_000_000
        train = _stratified_bootstrap(train_pool, pair_seed + 1)
        background = train[FEATURES].sample(
            n=min(background_size, len(train)), random_state=pair_seed + 2
        ).reset_index(drop=True)
        target_frames: dict[str, pd.DataFrame] = {"CLEAN": eval_sample_clean[FEATURES].copy()}
        for mechanism_index, mechanism in enumerate(TARGET_MECHANISMS[1:], start=1):
            target_frames[mechanism] = inject_missingness(
                eval_sample_clean,
                mechanism,
                RATE_BY_MECHANISM[mechanism],
                pair_seed + 100 + mechanism_index,
            ).frame.reset_index(drop=True)

        for model_index, model_name in enumerate(models):
            model_seed = pair_seed + model_index * 10_000 + 1
            model = fit_credit_model(
                model_name, "none", train, train["target"].to_numpy(int), model_seed
            )
            n_fits += 1

            source_specific, source_specific_base, source_specific_error = (
                _model_specific_shap(model, target_frames["CLEAN"], background)
            )
            n_explanation_runs += 1
            source_kernel_a, source_kernel_base, source_kernel_error = _kernel_shap(
                model,
                target_frames["CLEAN"],
                background,
                seed=pair_seed + model_index * 1000 + 10,
                nsamples=kernel_nsamples,
            )
            source_kernel_b, _, source_kernel_b_error = _kernel_shap(
                model,
                target_frames["CLEAN"],
                background,
                seed=pair_seed + model_index * 1000 + 11,
                nsamples=kernel_nsamples,
            )
            n_explanation_runs += 2
            source_norm = {
                "model_specific_shap": normalize_signed_rows(source_specific),
                "kernel_shap": normalize_signed_rows(source_kernel_a),
            }
            kernel_algorithm_noise = _row_stv(
                normalize_signed_rows(source_kernel_a), normalize_signed_rows(source_kernel_b)
            )
            source_cross_disagreement = _row_stv(
                source_norm["model_specific_shap"], source_norm["kernel_shap"]
            )
            replacement_effects = _faithfulness_replacement_effects(
                model, target_frames["CLEAN"], train
            )
            faithfulness = {
                "model_specific_shap": _row_faithfulness(
                    source_specific, replacement_effects
                ),
                "kernel_shap": _row_faithfulness(source_kernel_a, replacement_effects),
            }

            method_source = {
                "model_specific_shap": source_specific,
                "kernel_shap": source_kernel_a,
            }
            method_additivity_source = {
                "model_specific_shap": source_specific_error,
                "kernel_shap": source_kernel_error,
            }

            for target_index, mechanism in enumerate(TARGET_MECHANISMS):
                target_frame = target_frames[mechanism]
                if mechanism == "CLEAN":
                    target_specific = source_specific
                    target_kernel = source_kernel_a
                    target_specific_error = source_specific_error
                    target_kernel_error = source_kernel_error
                else:
                    target_specific, _, target_specific_error = _model_specific_shap(
                        model, target_frame, background
                    )
                    target_kernel, _, target_kernel_error = _kernel_shap(
                        model,
                        target_frame,
                        background,
                        seed=pair_seed + model_index * 1000 + 100 + target_index,
                        nsamples=kernel_nsamples,
                    )
                    n_explanation_runs += 2
                target_method = {
                    "model_specific_shap": target_specific,
                    "kernel_shap": target_kernel,
                }
                target_norm = {
                    name: normalize_signed_rows(values) for name, values in target_method.items()
                }
                target_cross_disagreement = _row_stv(
                    target_norm["model_specific_shap"], target_norm["kernel_shap"]
                )
                raw_source = _raw_score(model, target_frames["CLEAN"])
                raw_target = _raw_score(model, target_frame)
                abs_raw_shift = np.abs(raw_target - raw_source)
                predicted_unchanged = (raw_source >= 0) == (raw_target >= 0)

                method_shift: dict[str, np.ndarray] = {}
                for explainer_name in EXPLAINERS:
                    shift_stv = _row_stv(source_norm[explainer_name], target_norm[explainer_name])
                    method_shift[explainer_name] = shift_stv
                    source_abs, source_signed = _global_profiles(method_source[explainer_name])
                    target_abs, target_signed = _global_profiles(target_method[explainer_name])
                    signed_faith, absolute_faith = faithfulness[explainer_name]
                    scenario_rows.append(
                        {
                            "pair": pair,
                            "model": model_name,
                            "target_mechanism": mechanism,
                            "target_rate": RATE_BY_MECHANISM[mechanism],
                            "explainer": explainer_name,
                            "n_observations": len(target_frame),
                            "mean_shift_stv": float(np.mean(shift_stv)),
                            "median_shift_stv": float(np.median(shift_stv)),
                            "q90_shift_stv": float(np.quantile(shift_stv, 0.90)),
                            "mean_abs_raw_score_shift": float(np.mean(abs_raw_shift)),
                            "prediction_label_invariant_rate": float(np.mean(predicted_unchanged)),
                            "mean_shift_stv_label_invariant": float(
                                np.mean(shift_stv[predicted_unchanged])
                            ) if predicted_unchanged.any() else np.nan,
                            "global_abs_rank_tau": _safe_corr(
                                source_abs, target_abs, method="kendall"
                            ),
                            "global_abs_top5_jaccard": top_k_jaccard(source_abs, target_abs, 5),
                            "global_signed_consistency": weighted_sign_consistency(
                                source_signed, target_signed
                            ),
                            "source_additivity_error": method_additivity_source[explainer_name],
                            "target_additivity_error": (
                                target_specific_error
                                if explainer_name == "model_specific_shap"
                                else target_kernel_error
                            ),
                            "source_signed_replacement_faithfulness": signed_faith,
                            "source_absolute_replacement_faithfulness": absolute_faith,
                            "mean_kernel_algorithm_noise_stv": float(
                                np.mean(kernel_algorithm_noise)
                            ),
                            "shift_to_kernel_noise_ratio": float(
                                np.mean(shift_stv) / max(np.mean(kernel_algorithm_noise), 1e-12)
                            ),
                        }
                    )
                    for feature_index, feature in enumerate(FEATURES):
                        feature_rows.append(
                            {
                                "pair": pair,
                                "model": model_name,
                                "target_mechanism": mechanism,
                                "explainer": explainer_name,
                                "feature": feature,
                                "source_mean_abs": float(source_abs[feature_index]),
                                "target_mean_abs": float(target_abs[feature_index]),
                                "source_mean_signed": float(source_signed[feature_index]),
                                "target_mean_signed": float(target_signed[feature_index]),
                            }
                        )

                shift_correlation = _safe_corr(
                    method_shift["model_specific_shap"],
                    method_shift["kernel_shap"],
                )
                for row_index in range(len(target_frame)):
                    observation_rows.append(
                        {
                            "pair": pair,
                            "model": model_name,
                            "target_mechanism": mechanism,
                            "sample_position": int(row_index),
                            "y_true": int(y_sample[row_index]),
                            "raw_score_source": float(raw_source[row_index]),
                            "raw_score_target": float(raw_target[row_index]),
                            "abs_raw_score_shift": float(abs_raw_shift[row_index]),
                            "prediction_label_unchanged": bool(predicted_unchanged[row_index]),
                            "specific_shift_stv": float(
                                method_shift["model_specific_shap"][row_index]
                            ),
                            "kernel_shift_stv": float(method_shift["kernel_shap"][row_index]),
                            "source_cross_explainer_stv": float(
                                source_cross_disagreement[row_index]
                            ),
                            "target_cross_explainer_stv": float(
                                target_cross_disagreement[row_index]
                            ),
                            "kernel_algorithm_noise_stv": float(
                                kernel_algorithm_noise[row_index]
                            ),
                            "shift_stv_method_difference": float(
                                method_shift["kernel_shap"][row_index]
                                - method_shift["model_specific_shap"][row_index]
                            ),
                            "target_minus_source_cross_explainer_stv": float(
                                target_cross_disagreement[row_index]
                                - source_cross_disagreement[row_index]
                            ),
                            "scenario_shift_stv_method_spearman": shift_correlation,
                        }
                    )

    observations = pd.DataFrame(observation_rows)
    scenarios = pd.DataFrame(scenario_rows)
    features = pd.DataFrame(feature_rows)
    metadata = CrossExplainerMetadata(
        runtime_seconds=float(perf_counter() - start),
        n_pairs=n_pairs,
        n_model_fits=n_fits,
        n_explanation_runs=n_explanation_runs,
        n_observation_rows=len(observations),
        n_scenario_rows=len(scenarios),
        background_size=background_size,
        explanation_sample_size=explanation_sample_size,
        kernel_nsamples=kernel_nsamples,
    )
    return observations, scenarios, features, metadata
