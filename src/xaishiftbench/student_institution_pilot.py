"""Two-school institutional attribution-shift benchmark on UCI Student Performance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from .calibrated_shift import calibrated_attribution_shift
from .datasets.student_performance import load_student_performance
from .domain_shift import domain_classifier_auc
from .metrics import (
    jensen_shannon_distance,
    kendall_tau_b,
    normalize_signed_rows,
    prefixed_attribution_l1_diagnostics,
    matched_l1_sensitivity,
    repeated_explanation_subsample_sensitivity,
    top_k_jaccard,
    weighted_sign_consistency,
)
from .student_models import StudentFittedModel, fit_student_model


@dataclass(frozen=True)
class StudentPilotMetadata:
    runtime_seconds: float
    n_pairs: int
    n_splits: int
    n_model_fits: int
    n_scenario_evaluations: int


def _stratified_bootstrap(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pieces = []
    for _, group in frame.groupby("target", sort=True):
        chosen = rng.choice(group.index.to_numpy(), size=len(group), replace=True)
        pieces.append(frame.loc[chosen])
    result = pd.concat(pieces, ignore_index=True)
    return result.iloc[rng.permutation(len(result))].reset_index(drop=True)


def _source_split(source: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_idx, id_idx = train_test_split(
        np.arange(len(source)), test_size=0.30, random_state=seed, stratify=source["target"]
    )
    return source.iloc[train_idx].reset_index(drop=True), source.iloc[id_idx].reset_index(drop=True)


def _prediction_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return {
        "auroc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }


def _global_profiles(attr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.mean(np.abs(attr), axis=0), np.mean(attr, axis=0)


def _explain_bundle(
    model: StudentFittedModel, x: pd.DataFrame, tolerance: float = 1e-7
) -> dict[str, Any]:
    encoded, semantic, base = model.explain(x)
    normalized = normalize_signed_rows(semantic)
    additivity = float(np.max(np.abs(model.raw_score(x) - (base + encoded.sum(axis=1)))))
    if additivity > tolerance:
        raise AssertionError(f"Additivity gate failed for {model.model_name}: {additivity:.3e}")
    return {
        "semantic": semantic,
        "normalized": normalized,
        "additivity_error": additivity,
    }


def run_student_institution_pilot(
    data_path: str | Path,
    seed_base: int = 2026073004,
    n_pairs: int = 20,
    pair_start: int = 0,
    total_pairs: int = 20,
    n_splits: int = 5,
    explain_cap: int = 160,
    models: tuple[str, ...] = ("logistic", "lightgbm"),
    representations: tuple[str, ...] = ("early", "late"),
    sampling_repeats: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, StudentPilotMetadata]:
    if total_pairs < n_splits or total_pairs % n_splits:
        raise ValueError("total_pairs must be a positive multiple of n_splits.")
    if pair_start < 0 or n_pairs < 1 or pair_start + n_pairs > total_pairs:
        raise ValueError("Requested pair range is outside the release-defined grid.")
    refits_per_split = total_pairs // n_splits
    data = load_student_performance(data_path)
    frame = data.frame
    start = perf_counter()
    rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    n_fits = 0

    for direction_index, (source_school, target_school) in enumerate((("GP", "MS"), ("MS", "GP"))):
        source = frame.loc[frame["school"] == source_school].reset_index(drop=True)
        target = frame.loc[frame["school"] == target_school].reset_index(drop=True)
        y_target = target["target"].to_numpy(int)

        for representation_index, representation in enumerate(representations):
            features, numeric_features, categorical_features = data.features(representation)
            for model_index, model_name in enumerate(models):
                for pair in range(pair_start, pair_start + n_pairs):
                    split_index = pair // refits_per_split
                    refit_index = pair % refits_per_split
                    split_seed = (
                        seed_base + direction_index * 1_000_000 + split_index * 10_000
                    )
                    train_pool, id_eval = _source_split(source, split_seed)
                    y_id = id_eval["target"].to_numpy(int)
                    pair_seed = (
                        seed_base
                        + direction_index * 10_000_000
                        + representation_index * 1_000_000
                        + model_index * 100_000
                        + split_index * 10_000
                        + refit_index * 100
                    )
                    train_a = _stratified_bootstrap(train_pool, pair_seed + 1)
                    train_b = _stratified_bootstrap(train_pool, pair_seed + 2)
                    model_a = fit_student_model(
                        model_name,
                        train_a,
                        train_a["target"].to_numpy(int),
                        features,
                        numeric_features,
                        categorical_features,
                        pair_seed + 11,
                    )
                    model_b = fit_student_model(
                        model_name,
                        train_b,
                        train_b["target"].to_numpy(int),
                        features,
                        numeric_features,
                        categorical_features,
                        pair_seed + 12,
                    )
                    n_fits += 2

                    p_id_a = model_a.predict_proba(id_eval[features])
                    p_target_a = model_a.predict_proba(target[features])
                    p_id_b = model_b.predict_proba(id_eval[features])
                    mid = _prediction_metrics(y_id, p_id_a)
                    mt = _prediction_metrics(y_target, p_target_a)
                    midb = _prediction_metrics(y_id, p_id_b)

                    bid = _explain_bundle(model_a, id_eval[features])
                    bt = _explain_bundle(model_a, target[features])
                    bidb = _explain_bundle(model_b, id_eval[features])
                    btb = _explain_bundle(model_b, target[features])
                    calibrated = calibrated_attribution_shift(
                        bid["normalized"],
                        bt["normalized"],
                        bidb["normalized"],
                        y_id,
                        y_target,
                        normalized_target_b=btb["normalized"],
                        seed=pair_seed + 41,
                        cap=explain_cap,
                        class_cap=max(20, explain_cap // 2),
                    )
                    si = calibrated.indices.source_shift
                    sn = calibrated.indices.source_null
                    ti = calibrated.indices.target_shift
                    attr_id_a = bid["semantic"][si]
                    attr_target_a = bt["semantic"][ti]
                    attr_id_b = bidb["semantic"][sn]
                    norm_id_a = bid["normalized"][si]
                    norm_target_a = bt["normalized"][ti]
                    abs_id_a, signed_id_a = _global_profiles(attr_id_a)
                    abs_target_a, signed_target_a = _global_profiles(attr_target_a)
                    abs_id_b, signed_id_b = _global_profiles(attr_id_b)
                    shift_js = jensen_shannon_distance(abs_id_a, abs_target_a)
                    null_js = jensen_shannon_distance(abs_id_a, abs_id_b)

                    if refit_index == 0:
                        input_auc = domain_classifier_auc(
                            model_a.transform(id_eval.iloc[si][features]),
                            model_a.transform(target.iloc[ti][features]),
                            pair_seed + 51,
                        )
                        explanation_auc = domain_classifier_auc(
                            norm_id_a, norm_target_a, pair_seed + 52
                        )
                    else:
                        input_auc = explanation_auc = float("nan")

                    source_l1 = np.sum(np.abs(attr_id_a), axis=1)
                    target_l1 = np.sum(np.abs(attr_target_a), axis=1)
                    row: dict[str, Any] = {
                        "pair": pair,
                        "split_index": split_index,
                        "refit_index": refit_index,
                        "split_seed": split_seed,
                        "source_school": source_school,
                        "target_school": target_school,
                        "direction": f"{source_school}_to_{target_school}",
                        "representation": representation,
                        "model": model_name,
                        "n_source_total": len(source),
                        "n_train_pool": len(train_pool),
                        "n_id_eval": len(id_eval),
                        "n_target": len(target),
                        "n_explanation": calibrated.indices.sample_size,
                        "source_prevalence": float(source["target"].mean()),
                        "id_prevalence": float(y_id.mean()),
                        "target_prevalence": float(y_target.mean()),
                        "id_auroc": mid["auroc"],
                        "target_auroc": mt["auroc"],
                        "delta_auroc": mt["auroc"] - mid["auroc"],
                        "id_average_precision": mid["average_precision"],
                        "target_average_precision": mt["average_precision"],
                        "id_log_loss": mid["log_loss"],
                        "target_log_loss": mt["log_loss"],
                        "delta_log_loss": mt["log_loss"] - mid["log_loss"],
                        "id_brier": mid["brier"],
                        "target_brier": mt["brier"],
                        "delta_brier": mt["brier"] - mid["brier"],
                        "null_abs_delta_auroc": abs(midb["auroc"] - mid["auroc"]),
                        "shift_energy_u2": calibrated.shift_u2,
                        "same_model_sample_null_energy_u2": calibrated.same_model_sample_null_u2,
                        "refit_sample_null_energy_u2": calibrated.refit_sample_null_u2,
                        "calibrated_excess_u2": calibrated.excess_over_refit_sample_null_u2,
                        "paired_refit_l2": calibrated.paired_refit_l2,
                        "class_macro_shift_u2": calibrated.class_macro_shift_u2,
                        "class_macro_sample_null_u2": calibrated.class_macro_sample_null_u2,
                        "class_macro_refit_null_u2": calibrated.class_macro_refit_null_u2,
                        "class_macro_excess_u2": calibrated.class_macro_excess_u2,
                        "class_weighted_shift_u2": calibrated.class_weighted_shift_u2,
                        "class_weighted_refit_null_u2": calibrated.class_weighted_refit_null_u2,
                        "class_weighted_excess_u2": calibrated.class_weighted_excess_u2,
                        "class0_shift_u2": calibrated.class0_shift_u2,
                        "class1_shift_u2": calibrated.class1_shift_u2,
                        "class0_refit_null_u2": calibrated.class0_refit_null_u2,
                        "class1_refit_null_u2": calibrated.class1_refit_null_u2,
                        "class0_n": calibrated.class0_n,
                        "class1_n": calibrated.class1_n,
                        "class_weighting_prevalence": calibrated.class_weighting_prevalence,
                        "class_weighting_scheme": "target_prevalence",
                        "reverse_shift_energy_u2": calibrated.reverse_shift_u2,
                        "reverse_refit_sample_null_energy_u2": calibrated.reverse_refit_sample_null_u2,
                        "reverse_calibrated_excess_u2": calibrated.reverse_excess_u2,
                        "symmetrized_calibrated_excess_u2": calibrated.symmetrized_excess_u2,
                        "symmetry_gap_u2": calibrated.symmetry_gap_u2,
                        "shift_js": shift_js,
                        "null_js": null_js,
                        "paired_excess_js": shift_js - null_js,
                        "shift_rank_tau": kendall_tau_b(abs_id_a, abs_target_a),
                        "null_rank_tau": kendall_tau_b(abs_id_a, abs_id_b),
                        "shift_top5_jaccard": top_k_jaccard(abs_id_a, abs_target_a, min(5, len(abs_id_a))),
                        "null_top5_jaccard": top_k_jaccard(abs_id_a, abs_id_b, min(5, len(abs_id_a))),
                        "shift_sign_consistency": weighted_sign_consistency(signed_id_a, signed_target_a),
                        "null_sign_consistency": weighted_sign_consistency(signed_id_a, signed_id_b),
                        "source_mean_attribution_l1": float(source_l1.mean()),
                        "target_mean_attribution_l1": float(target_l1.mean()),
                        "delta_mean_attribution_l1": float(target_l1.mean() - source_l1.mean()),
                        "input_domain_auc": input_auc,
                        "explanation_domain_auc": explanation_auc,
                        "max_additivity_error": max(
                            bid["additivity_error"],
                            bt["additivity_error"],
                            bidb["additivity_error"],
                            btb["additivity_error"],
                        ),
                    }
                    row.update(prefixed_attribution_l1_diagnostics(attr_id_a, "source_l1"))
                    row.update(prefixed_attribution_l1_diagnostics(attr_id_b, "null_l1"))
                    row.update(prefixed_attribution_l1_diagnostics(attr_target_a, "target_l1"))
                    row.update(matched_l1_sensitivity(attr_id_a, attr_id_b, attr_target_a))
                    if sampling_repeats:
                        row.update(repeated_explanation_subsample_sensitivity(
                            norm_id_a, bidb["normalized"][sn], norm_target_a,
                            y_id[si], y_id[sn], y_target[ti],
                            seed=pair_seed + 7041, repeats=sampling_repeats,
                        ))
                    row["silent_explanation_shift"] = bool(
                        row["calibrated_excess_u2"] > 0
                        and abs(row["delta_auroc"]) <= row["null_abs_delta_auroc"]
                    )
                    rows.append(row)

                    for index, feature in enumerate(model_a.semantic_map.names):
                        feature_rows.append(
                            {
                                "pair": pair,
                                "split_index": split_index,
                                "refit_index": refit_index,
                                "direction": row["direction"],
                                "representation": representation,
                                "model": model_name,
                                "feature": feature,
                                "id_mean_abs": float(abs_id_a[index]),
                                "target_mean_abs": float(abs_target_a[index]),
                                "delta_mean_abs": float(abs_target_a[index] - abs_id_a[index]),
                                "id_mean_signed": float(signed_id_a[index]),
                                "target_mean_signed": float(signed_target_a[index]),
                            }
                        )

    metadata = StudentPilotMetadata(
        runtime_seconds=float(perf_counter() - start),
        n_pairs=n_pairs,
        n_splits=n_splits,
        n_model_fits=n_fits,
        n_scenario_evaluations=len(rows),
    )
    return pd.DataFrame(rows), pd.DataFrame(feature_rows), metadata
