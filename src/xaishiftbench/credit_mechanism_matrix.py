"""Full training-mechanism by deployment-mechanism explanation-shift matrix."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
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

MECHANISMS = ("CLEAN", "MCAR", "MAR", "MNAR", "BLOCK")
RATE_BY_MECHANISM = {"CLEAN": 0.0, "MCAR": 0.15, "MAR": 0.15, "MNAR": 0.15, "BLOCK": 0.15}


@dataclass(frozen=True)
class CreditMatrixMetadata:
    runtime_seconds: float
    n_pairs: int
    n_model_fits: int
    n_scenario_evaluations: int


def _stratified_bootstrap(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pieces: list[pd.DataFrame] = []
    for _, group in frame.groupby("target", sort=True):
        chosen = rng.choice(group.index.to_numpy(), size=len(group), replace=True)
        pieces.append(frame.loc[chosen])
    result = pd.concat(pieces, ignore_index=True)
    return result.iloc[rng.permutation(len(result))].reset_index(drop=True)


def _prediction_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return {
        "auroc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }


def _row_stv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum(np.abs(a - b), axis=1)


def _global_profiles(attr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.mean(np.abs(attr), axis=0), np.mean(attr, axis=0)


def _indicator_abs_share(model: CreditFittedModel, encoded_attr: np.ndarray) -> float:
    if model.indicator_mode != "all":
        return 0.0
    output_slice = model.preprocessor.output_indices_["ind"]
    numerator = np.sum(np.abs(encoded_attr[:, output_slice]))
    denominator = np.sum(np.abs(encoded_attr))
    return float(numerator / denominator) if denominator > 0 else 0.0


def _max_additivity_error(
    model: CreditFittedModel, x: pd.DataFrame, encoded_attr: np.ndarray, base_value: float
) -> float:
    encoded_x = model.transform(x)
    reconstructed = base_value + encoded_attr.sum(axis=1)
    if model.model_name == "logistic":
        raw = np.asarray(model.estimator.decision_function(encoded_x), dtype=float)
    else:
        raw = np.asarray(model.estimator.booster_.predict(encoded_x, raw_score=True), dtype=float)
    return float(np.max(np.abs(raw - reconstructed)))


def _bundle(model: CreditFittedModel, x: pd.DataFrame) -> dict[str, object]:
    encoded, semantic, base = model.explain(x)
    normalized = normalize_signed_rows(semantic)
    abs_profile, signed_profile = _global_profiles(semantic)
    return {
        "encoded": encoded,
        "semantic": semantic,
        "normalized": normalized,
        "abs_profile": abs_profile,
        "signed_profile": signed_profile,
        "indicator_share": _indicator_abs_share(model, encoded),
        "additivity_error": _max_additivity_error(model, x, encoded, base),
    }


def run_credit_mechanism_matrix(
    data_path: str | Path,
    seed_base: int = 2026073005,
    n_pairs: int = 10,
    pair_start: int = 0,
    models: tuple[str, ...] = ("logistic", "lightgbm"),
    indicator_modes: tuple[str, ...] = ("none", "all"),
) -> tuple[pd.DataFrame, pd.DataFrame, CreditMatrixMetadata]:
    if n_pairs < 1:
        raise ValueError("n_pairs must be at least 1.")
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
    n_fits = 0

    for pair in range(pair_start, pair_start + n_pairs):
        pair_seed = seed_base + pair * 1_000_000
        train_a_clean = _stratified_bootstrap(train_pool, pair_seed + 1)
        train_b_clean = _stratified_bootstrap(train_pool, pair_seed + 2)

        train_a = {}
        train_b = {}
        eval_primary = {}
        eval_secondary = {}
        for mechanism_index, mechanism in enumerate(MECHANISMS):
            rate = RATE_BY_MECHANISM[mechanism]
            train_a[mechanism] = inject_missingness(
                train_a_clean, mechanism, rate, pair_seed + 100 + mechanism_index
            )
            train_b[mechanism] = inject_missingness(
                train_b_clean, mechanism, rate, pair_seed + 200 + mechanism_index
            )
            eval_primary[mechanism] = inject_missingness(
                eval_clean, mechanism, rate, pair_seed + 300 + mechanism_index
            )
            eval_secondary[mechanism] = inject_missingness(
                eval_clean, mechanism, rate, pair_seed + 400 + mechanism_index
            )

        for source_index, source_mechanism in enumerate(MECHANISMS):
            source_rate = RATE_BY_MECHANISM[source_mechanism]
            reference = eval_primary[source_mechanism]
            matched_control = eval_secondary[source_mechanism]

            for indicator_mode in indicator_modes:
                for model_index, model_name in enumerate(models):
                    model_seed = (
                        pair_seed
                        + source_index * 100_000
                        + model_index * 10_000
                        + (5000 if indicator_mode == "all" else 0)
                    )
                    model_a = fit_credit_model(
                        model_name,
                        indicator_mode,
                        train_a[source_mechanism].frame,
                        train_a_clean["target"].to_numpy(int),
                        model_seed + 1,
                    )
                    model_b = fit_credit_model(
                        model_name,
                        indicator_mode,
                        train_b[source_mechanism].frame,
                        train_b_clean["target"].to_numpy(int),
                        model_seed + 2,
                    )
                    n_fits += 2

                    ref_a = _bundle(model_a, reference.frame)
                    ref_b = _bundle(model_b, reference.frame)
                    control_a = _bundle(model_a, matched_control.frame)
                    p_ref = model_a.predict_proba(reference.frame)
                    p_ref_b = model_b.predict_proba(reference.frame)
                    p_control = model_a.predict_proba(matched_control.frame)
                    metrics_ref = _prediction_metrics(y_eval, p_ref)
                    metrics_ref_b = _prediction_metrics(y_eval, p_ref_b)
                    metrics_control = _prediction_metrics(y_eval, p_control)
                    null_stv = _row_stv(ref_a["normalized"], ref_b["normalized"])
                    control_stv = _row_stv(ref_a["normalized"], control_a["normalized"])
                    null_js = jensen_shannon_distance(ref_a["abs_profile"], ref_b["abs_profile"])
                    control_js = jensen_shannon_distance(
                        ref_a["abs_profile"], control_a["abs_profile"]
                    )

                    for target_mechanism in MECHANISMS:
                        target_rate = RATE_BY_MECHANISM[target_mechanism]
                        target = eval_secondary[target_mechanism]
                        target_a = _bundle(model_a, target.frame)
                        p_target = model_a.predict_proba(target.frame)
                        metrics_target = _prediction_metrics(y_eval, p_target)
                        shift_stv = _row_stv(ref_a["normalized"], target_a["normalized"])
                        shift_js = jensen_shannon_distance(
                            ref_a["abs_profile"], target_a["abs_profile"]
                        )
                        row = {
                            "pair": pair,
                            "model": model_name,
                            "indicator_mode": indicator_mode,
                            "source_mechanism": source_mechanism,
                            "source_rate": source_rate,
                            "target_mechanism": target_mechanism,
                            "target_rate": target_rate,
                            "matched_mechanism": source_mechanism == target_mechanism,
                            "n_train": len(train_pool),
                            "n_eval": len(eval_clean),
                            "eval_prevalence": float(y_eval.mean()),
                            "source_train_missing_cells": train_a[source_mechanism].missing_cells,
                            "reference_missing_cells": reference.missing_cells,
                            "target_missing_cells": target.missing_cells,
                            "reference_auroc": metrics_ref["auroc"],
                            "target_auroc": metrics_target["auroc"],
                            "delta_auroc": metrics_target["auroc"] - metrics_ref["auroc"],
                            "reference_average_precision": metrics_ref["average_precision"],
                            "target_average_precision": metrics_target["average_precision"],
                            "reference_log_loss": metrics_ref["log_loss"],
                            "target_log_loss": metrics_target["log_loss"],
                            "delta_log_loss": metrics_target["log_loss"] - metrics_ref["log_loss"],
                            "reference_brier": metrics_ref["brier"],
                            "target_brier": metrics_target["brier"],
                            "delta_brier": metrics_target["brier"] - metrics_ref["brier"],
                            "mean_abs_probability_shift": float(np.mean(np.abs(p_target - p_ref))),
                            "mean_abs_probability_refit_null": float(
                                np.mean(np.abs(p_ref_b - p_ref))
                            ),
                            "mean_abs_probability_matched_control": float(
                                np.mean(np.abs(p_control - p_ref))
                            ),
                            "null_abs_delta_auroc": abs(
                                metrics_ref_b["auroc"] - metrics_ref["auroc"]
                            ),
                            "matched_control_abs_delta_auroc": abs(
                                metrics_control["auroc"] - metrics_ref["auroc"]
                            ),
                            "mean_shift_stv": float(np.mean(shift_stv)),
                            "mean_refit_null_stv": float(np.mean(null_stv)),
                            "mean_matched_control_stv": float(np.mean(control_stv)),
                            "mechanism_mismatch_excess_stv": float(
                                np.mean(shift_stv) - np.mean(control_stv)
                            ),
                            "paired_excess_over_refit_stv": float(
                                np.mean(shift_stv) - np.mean(null_stv)
                            ),
                            "shift_js": shift_js,
                            "refit_null_js": null_js,
                            "matched_control_js": control_js,
                            "mechanism_mismatch_excess_js": shift_js - control_js,
                            "shift_rank_tau": kendall_tau_b(
                                ref_a["abs_profile"], target_a["abs_profile"]
                            ),
                            "refit_null_rank_tau": kendall_tau_b(
                                ref_a["abs_profile"], ref_b["abs_profile"]
                            ),
                            "shift_top5_jaccard": top_k_jaccard(
                                ref_a["abs_profile"], target_a["abs_profile"], 5
                            ),
                            "refit_null_top5_jaccard": top_k_jaccard(
                                ref_a["abs_profile"], ref_b["abs_profile"], 5
                            ),
                            "shift_sign_consistency": weighted_sign_consistency(
                                ref_a["signed_profile"], target_a["signed_profile"]
                            ),
                            "refit_null_sign_consistency": weighted_sign_consistency(
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
                        row["silent_mechanism_shift"] = bool(
                            not row["matched_mechanism"]
                            and row["mechanism_mismatch_excess_stv"] > 0
                            and abs(row["delta_auroc"])
                            <= max(row["matched_control_abs_delta_auroc"], 1e-12)
                        )
                        rows.append(row)

                        ref_abs = np.asarray(ref_a["abs_profile"], dtype=float)
                        target_abs = np.asarray(target_a["abs_profile"], dtype=float)
                        ref_signed = np.asarray(ref_a["signed_profile"], dtype=float)
                        target_signed = np.asarray(target_a["signed_profile"], dtype=float)
                        for feature_index, feature in enumerate(model_a.semantic_map.names):
                            feature_rows.append(
                                {
                                    "pair": pair,
                                    "model": model_name,
                                    "indicator_mode": indicator_mode,
                                    "source_mechanism": source_mechanism,
                                    "target_mechanism": target_mechanism,
                                    "feature": feature,
                                    "reference_mean_abs": float(ref_abs[feature_index]),
                                    "target_mean_abs": float(target_abs[feature_index]),
                                    "delta_mean_abs": float(
                                        target_abs[feature_index] - ref_abs[feature_index]
                                    ),
                                    "reference_mean_signed": float(ref_signed[feature_index]),
                                    "target_mean_signed": float(target_signed[feature_index]),
                                }
                            )

    metadata = CreditMatrixMetadata(
        runtime_seconds=float(perf_counter() - start),
        n_pairs=n_pairs,
        n_model_fits=n_fits,
        n_scenario_evaluations=len(rows),
    )
    return pd.DataFrame(rows), pd.DataFrame(feature_rows), metadata
