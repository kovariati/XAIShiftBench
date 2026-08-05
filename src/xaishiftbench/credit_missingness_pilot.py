"""Controlled missingness-mechanism shift pilot on South German Credit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from .credit_models import CreditFittedModel, fit_credit_model
from .datasets.south_german_credit import FEATURES, load_south_german_credit
from .metrics import (
    jensen_shannon_distance,
    kendall_tau_b,
    normalize_signed_rows,
    top_k_jaccard,
    weighted_sign_consistency,
)
from .missingness import inject_missingness


SCENARIOS: tuple[tuple[str, float], ...] = (
    ("CLEAN", 0.00),
    ("MCAR", 0.05),
    ("MCAR", 0.15),
    ("MCAR", 0.30),
    ("MAR", 0.05),
    ("MAR", 0.15),
    ("MAR", 0.30),
    ("MNAR", 0.05),
    ("MNAR", 0.15),
    ("MNAR", 0.30),
    ("BLOCK", 0.05),
    ("BLOCK", 0.15),
    ("BLOCK", 0.30),
)


@dataclass(frozen=True)
class CreditPilotMetadata:
    runtime_seconds: float
    n_pairs: int
    n_model_fits: int
    n_scenario_evaluations: int


def _stratified_bootstrap(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pieces: list[pd.DataFrame] = []
    for cls, group in frame.groupby("target", sort=True):
        chosen = rng.choice(group.index.to_numpy(), size=len(group), replace=True)
        pieces.append(frame.loc[chosen])
    result = pd.concat(pieces, ignore_index=True)
    return result.iloc[rng.permutation(len(result))].reset_index(drop=True)


def _prediction_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    clipped = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return {
        "auroc": float(roc_auc_score(y, clipped)),
        "average_precision": float(average_precision_score(y, clipped)),
        "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
        "brier": float(brier_score_loss(y, clipped)),
    }


def _global_profiles(attr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.mean(np.abs(attr), axis=0), np.mean(attr, axis=0)


def _row_stv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError(f"Paired attribution arrays must match, got {a.shape} and {b.shape}.")
    return 0.5 * np.sum(np.abs(a - b), axis=1)


def _indicator_abs_share(model: CreditFittedModel, encoded_attr: np.ndarray) -> float:
    if model.indicator_mode != "all":
        return 0.0
    output_slice = model.preprocessor.output_indices_["ind"]
    numerator = np.sum(np.abs(encoded_attr[:, output_slice]))
    denominator = np.sum(np.abs(encoded_attr))
    return float(numerator / denominator) if denominator > 0 else 0.0


def _indicator_feature_profile(model: CreditFittedModel, encoded_attr: np.ndarray) -> np.ndarray:
    if model.indicator_mode != "all":
        return np.zeros(len(FEATURES), dtype=float)
    output_slice = model.preprocessor.output_indices_["ind"]
    indicator_values = encoded_attr[:, output_slice]
    indicator = model.preprocessor.named_transformers_["ind"]
    output = np.zeros(len(FEATURES), dtype=float)
    for local_index, original_index in enumerate(np.asarray(indicator.features_, dtype=int)):
        output[int(original_index)] = float(np.mean(np.abs(indicator_values[:, local_index])))
    return output


def _max_additivity_error(
    model: CreditFittedModel,
    x: pd.DataFrame,
    encoded_attr: np.ndarray,
    base_value: float,
) -> float:
    encoded_x = model.transform(x)
    reconstructed = base_value + np.sum(encoded_attr, axis=1)
    if model.model_name == "logistic":
        raw = np.asarray(model.estimator.decision_function(encoded_x), dtype=float)
    else:
        raw = np.asarray(model.estimator.booster_.predict(encoded_x, raw_score=True), dtype=float)
    return float(np.max(np.abs(raw - reconstructed)))


def _explain_bundle(
    model: CreditFittedModel,
    x: pd.DataFrame,
) -> dict[str, object]:
    encoded, semantic, base = model.explain(x)
    additivity_error = _max_additivity_error(model, x, encoded, base)
    if additivity_error > 1e-7:
        raise AssertionError(
            f"Additivity gate failed for {model.model_name}: {additivity_error:.3e}"
        )
    normalized = normalize_signed_rows(semantic)
    abs_profile, signed_profile = _global_profiles(semantic)
    return {
        "encoded": encoded,
        "semantic": semantic,
        "normalized": normalized,
        "abs_profile": abs_profile,
        "signed_profile": signed_profile,
        "base": base,
        "indicator_share": _indicator_abs_share(model, encoded),
        "indicator_profile": _indicator_feature_profile(model, encoded),
        "additivity_error": additivity_error,
    }


def run_credit_missingness_pilot(
    data_path: str | Path,
    seed_base: int = 20260730,
    n_pairs: int = 10,
    pair_start: int = 0,
    models: tuple[str, ...] = ("logistic", "lightgbm"),
    indicator_modes: tuple[str, ...] = ("none", "all"),
) -> tuple[pd.DataFrame, pd.DataFrame, CreditPilotMetadata]:
    """Run the complete paired mechanism-shift design."""
    if n_pairs < 1:
        raise ValueError("n_pairs must be at least 1.")
    if pair_start < 0:
        raise ValueError("pair_start must be non-negative.")
    start = perf_counter()
    data = load_south_german_credit(data_path).frame
    train_idx, eval_idx = train_test_split(
        np.arange(len(data)),
        test_size=0.30,
        random_state=seed_base,
        stratify=data["target"],
    )
    train_pool = data.iloc[train_idx].reset_index(drop=True)
    eval_clean = data.iloc[eval_idx].reset_index(drop=True)
    y_eval = eval_clean["target"].to_numpy(int)

    rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    n_model_fits = 0
    n_scenario_evaluations = 0

    for pair in range(pair_start, pair_start + n_pairs):
        pair_seed = seed_base + pair * 100_000
        train_a_clean = _stratified_bootstrap(train_pool, pair_seed + 1)
        train_b_clean = _stratified_bootstrap(train_pool, pair_seed + 2)
        train_a_missing = inject_missingness(train_a_clean, "MCAR", 0.15, pair_seed + 11)
        train_b_missing = inject_missingness(train_b_clean, "MCAR", 0.15, pair_seed + 12)

        reference = inject_missingness(eval_clean, "MCAR", 0.15, pair_seed + 21)
        mask_control = inject_missingness(eval_clean, "MCAR", 0.15, pair_seed + 22)
        scenario_data = {
            f"{mechanism}_{int(round(rate * 100)):02d}": inject_missingness(
                eval_clean, mechanism, rate, pair_seed + 1000 + idx
            )
            for idx, (mechanism, rate) in enumerate(SCENARIOS)
        }

        for indicator_mode in indicator_modes:
            for model_index, model_name in enumerate(models):
                model_seed = pair_seed + 10_000 * (1 + model_index) + (500 if indicator_mode == "all" else 0)
                model_a = fit_credit_model(
                    model_name,
                    indicator_mode,
                    train_a_missing.frame,
                    train_a_clean["target"].to_numpy(int),
                    model_seed + 1,
                )
                model_b = fit_credit_model(
                    model_name,
                    indicator_mode,
                    train_b_missing.frame,
                    train_b_clean["target"].to_numpy(int),
                    model_seed + 2,
                )
                n_model_fits += 2

                ref_a = _explain_bundle(model_a, reference.frame)
                ref_b = _explain_bundle(model_b, reference.frame)
                control_a = _explain_bundle(model_a, mask_control.frame)
                p_ref_a = model_a.predict_proba(reference.frame)
                p_ref_b = model_b.predict_proba(reference.frame)
                p_control_a = model_a.predict_proba(mask_control.frame)
                metrics_ref = _prediction_metrics(y_eval, p_ref_a)
                metrics_ref_b = _prediction_metrics(y_eval, p_ref_b)
                metrics_control = _prediction_metrics(y_eval, p_control_a)

                null_stv = _row_stv(ref_a["normalized"], ref_b["normalized"])
                control_stv = _row_stv(ref_a["normalized"], control_a["normalized"])
                null_js = jensen_shannon_distance(ref_a["abs_profile"], ref_b["abs_profile"])
                control_js = jensen_shannon_distance(
                    ref_a["abs_profile"], control_a["abs_profile"]
                )

                for scenario_index, (mechanism, rate) in enumerate(SCENARIOS):
                    scenario = f"{mechanism}_{int(round(rate * 100)):02d}"
                    target_missing = scenario_data[scenario]
                    target_a = _explain_bundle(model_a, target_missing.frame)
                    p_target = model_a.predict_proba(target_missing.frame)
                    metrics_target = _prediction_metrics(y_eval, p_target)
                    shift_stv = _row_stv(ref_a["normalized"], target_a["normalized"])
                    shift_js = jensen_shannon_distance(
                        ref_a["abs_profile"], target_a["abs_profile"]
                    )
                    n_scenario_evaluations += 1

                    row = {
                        "pair": pair,
                        "model": model_name,
                        "indicator_mode": indicator_mode,
                        "scenario": scenario,
                        "mechanism": mechanism,
                        "rate": rate,
                        "n_train": len(train_pool),
                        "n_eval": len(eval_clean),
                        "eval_prevalence": float(y_eval.mean()),
                        "training_mechanism": "MCAR",
                        "training_rate": 0.15,
                        "reference_mechanism": "MCAR",
                        "reference_rate": 0.15,
                        "reference_missing_cells": reference.missing_cells,
                        "target_missing_cells": target_missing.missing_cells,
                        "ref_auroc": metrics_ref["auroc"],
                        "target_auroc": metrics_target["auroc"],
                        "delta_auroc": metrics_target["auroc"] - metrics_ref["auroc"],
                        "ref_average_precision": metrics_ref["average_precision"],
                        "target_average_precision": metrics_target["average_precision"],
                        "ref_log_loss": metrics_ref["log_loss"],
                        "target_log_loss": metrics_target["log_loss"],
                        "delta_log_loss": metrics_target["log_loss"] - metrics_ref["log_loss"],
                        "ref_brier": metrics_ref["brier"],
                        "target_brier": metrics_target["brier"],
                        "delta_brier": metrics_target["brier"] - metrics_ref["brier"],
                        "mean_abs_probability_shift": float(np.mean(np.abs(p_target - p_ref_a))),
                        "mean_abs_probability_null": float(np.mean(np.abs(p_ref_b - p_ref_a))),
                        "mean_abs_probability_mask_control": float(
                            np.mean(np.abs(p_control_a - p_ref_a))
                        ),
                        "null_abs_delta_auroc": abs(metrics_ref_b["auroc"] - metrics_ref["auroc"]),
                        "mask_control_abs_delta_auroc": abs(
                            metrics_control["auroc"] - metrics_ref["auroc"]
                        ),
                        "mean_shift_stv": float(np.mean(shift_stv)),
                        "median_shift_stv": float(np.median(shift_stv)),
                        "q90_shift_stv": float(np.quantile(shift_stv, 0.90)),
                        "mean_null_stv": float(np.mean(null_stv)),
                        "median_null_stv": float(np.median(null_stv)),
                        "q90_null_stv": float(np.quantile(null_stv, 0.90)),
                        "mean_mask_control_stv": float(np.mean(control_stv)),
                        "median_mask_control_stv": float(np.median(control_stv)),
                        "q90_mask_control_stv": float(np.quantile(control_stv, 0.90)),
                        "paired_excess_over_refit": float(np.mean(shift_stv) - np.mean(null_stv)),
                        "paired_excess_over_mask_control": float(
                            np.mean(shift_stv) - np.mean(control_stv)
                        ),
                        "shift_js": shift_js,
                        "null_js": null_js,
                        "mask_control_js": control_js,
                        "paired_excess_js_over_refit": shift_js - null_js,
                        "paired_excess_js_over_mask_control": shift_js - control_js,
                        "shift_rank_tau": kendall_tau_b(
                            ref_a["abs_profile"], target_a["abs_profile"]
                        ),
                        "null_rank_tau": kendall_tau_b(
                            ref_a["abs_profile"], ref_b["abs_profile"]
                        ),
                        "shift_top5_jaccard": top_k_jaccard(
                            ref_a["abs_profile"], target_a["abs_profile"], 5
                        ),
                        "null_top5_jaccard": top_k_jaccard(
                            ref_a["abs_profile"], ref_b["abs_profile"], 5
                        ),
                        "shift_sign_consistency": weighted_sign_consistency(
                            ref_a["signed_profile"], target_a["signed_profile"]
                        ),
                        "null_sign_consistency": weighted_sign_consistency(
                            ref_a["signed_profile"], ref_b["signed_profile"]
                        ),
                        "reference_indicator_abs_share": ref_a["indicator_share"],
                        "target_indicator_abs_share": target_a["indicator_share"],
                        "max_additivity_error": max(
                            float(ref_a["additivity_error"]),
                            float(ref_b["additivity_error"]),
                            float(control_a["additivity_error"]),
                            float(target_a["additivity_error"]),
                        ),
                    }
                    rows.append(row)

                    ref_abs = np.asarray(ref_a["abs_profile"], dtype=float)
                    target_abs = np.asarray(target_a["abs_profile"], dtype=float)
                    ref_signed = np.asarray(ref_a["signed_profile"], dtype=float)
                    target_signed = np.asarray(target_a["signed_profile"], dtype=float)
                    ref_ind = np.asarray(ref_a["indicator_profile"], dtype=float)
                    target_ind = np.asarray(target_a["indicator_profile"], dtype=float)
                    for feature_index, feature in enumerate(model_a.semantic_map.names):
                        # model semantic order and FEATURES contain the same set but not
                        # necessarily the same order.
                        original_index = FEATURES.index(feature)
                        feature_rows.append(
                            {
                                "pair": pair,
                                "model": model_name,
                                "indicator_mode": indicator_mode,
                                "scenario": scenario,
                                "mechanism": mechanism,
                                "rate": rate,
                                "feature": feature,
                                "reference_mean_abs": ref_abs[feature_index],
                                "target_mean_abs": target_abs[feature_index],
                                "delta_mean_abs": target_abs[feature_index] - ref_abs[feature_index],
                                "reference_mean_signed": ref_signed[feature_index],
                                "target_mean_signed": target_signed[feature_index],
                                "reference_indicator_mean_abs": ref_ind[original_index],
                                "target_indicator_mean_abs": target_ind[original_index],
                            }
                        )

    runtime = perf_counter() - start
    metadata = CreditPilotMetadata(
        runtime_seconds=float(runtime),
        n_pairs=n_pairs,
        n_model_fits=n_model_fits,
        n_scenario_evaluations=n_scenario_evaluations,
    )
    return pd.DataFrame(rows), pd.DataFrame(feature_rows), metadata
