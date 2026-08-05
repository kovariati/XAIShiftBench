"""OULAD 2013-to-2014 temporal attribution-transfer benchmark.

Each nominal refit index maps to one of five independent source train/ID splits
and one of four bootstrap refit pairs.  Activity-type columns are defined from
the source presentation only, preventing future target-only platform activity
categories from entering the primary feature schema.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from .calibrated_shift import calibrated_attribution_shift
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
from .oulad_models import fit_oulad_model, oulad_model_config

CAT = ["gender", "region", "highest_education", "imd_band", "age_band", "disability"]
BASE_NUM = [
    "num_of_prev_attempts",
    "studied_credits",
    "days_registered_before_start",
    "total_clicks",
    "interaction_rows",
    "active_days",
    "unique_sites",
    "clicks_per_active_day",
]


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return {
        "auroc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }


def _bootstrap(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pieces = []
    for _, group in frame.groupby("target_unsuccessful", sort=True):
        selected = rng.choice(group.index.to_numpy(), size=len(group), replace=True)
        pieces.append(frame.loc[selected])
    out = pd.concat(pieces, ignore_index=True)
    return out.iloc[rng.permutation(len(out))].reset_index(drop=True)


def _source_split(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_idx, id_idx = train_test_split(
        np.arange(len(frame)),
        test_size=0.30,
        random_state=seed,
        stratify=frame["target_unsuccessful"],
    )
    return frame.iloc[train_idx].reset_index(drop=True), frame.iloc[id_idx].reset_index(drop=True)


def _bundle(model: Any, x: pd.DataFrame, tolerance: float = 1e-7) -> dict[str, Any]:
    encoded, semantic, base = model.explain(x)
    raw = model.raw_score(x)
    additivity = float(np.max(np.abs(raw - (base + encoded.sum(axis=1)))))
    if additivity > tolerance:
        raise AssertionError(f"Additivity gate failed for {model.model_name}: {additivity:.3e}")
    return {
        "semantic": semantic,
        "normalized": normalize_signed_rows(semantic),
        "additivity_error": additivity,
    }


def _source_only_feature_space(
    source: pd.DataFrame, target: pd.DataFrame, all_click_cols: list[str]
) -> tuple[list[str], list[str]]:
    source_cols = [
        c for c in all_click_cols if float(source[c].fillna(0).abs().sum()) > 0.0
    ]
    target_only = [
        c
        for c in all_click_cols
        if c not in source_cols and float(target[c].fillna(0).abs().sum()) > 0.0
    ]
    return source_cols, target_only


def run_pair(
    model_table: Path,
    pairs_file: Path,
    pair_index: int,
    seed_base: int = 2026073007,
    explain_cap: int = 160,
    n_splits: int = 5,
    refits_per_split: int = 4,
    sampling_repeats: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not 0 <= pair_index < n_splits * refits_per_split:
        raise ValueError("pair_index is outside the release-defined split/refit grid.")
    split_index = pair_index // refits_per_split
    refit_index = pair_index % refits_per_split
    df = pd.read_csv(model_table)
    pairs = pd.read_csv(pairs_file)
    all_click_cols = sorted(
        c for c in df.columns if c.startswith("clicks_") and c not in BASE_NUM
    )
    rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    fits = 0
    t0 = perf_counter()

    for task_i, pair in pairs.iterrows():
        for horizon in (14, 56):
            source = df[
                (df.code_module == pair.code_module)
                & (df.code_presentation == pair.source_presentation)
                & (df.horizon_day == horizon)
            ].copy()
            target = df[
                (df.code_module == pair.code_module)
                & (df.code_presentation == pair.target_presentation)
                & (df.horizon_day == horizon)
            ].copy()
            split_seed = (
                seed_base + task_i * 1_000_000 + horizon * 1_000 + split_index * 10_000
            )
            pool, id_eval = _source_split(source, split_seed)
            click_cols, target_only_cols = _source_only_feature_space(
                pool, target, all_click_cols
            )
            id_only_cols = [
                column
                for column in all_click_cols
                if column not in click_cols
                and float(id_eval[column].fillna(0).abs().sum()) > 0.0
            ]
            numeric = BASE_NUM + click_cols
            features = numeric + CAT
            y_id = id_eval.target_unsuccessful.to_numpy(int)
            y_target = target.target_unsuccessful.to_numpy(int)

            for model_index, model_name in enumerate(("logistic", "lightgbm")):
                seed = (
                    seed_base
                    + task_i * 10_000_000
                    + horizon * 100_000
                    + model_index * 10_000
                    + split_index * 1_000
                    + refit_index * 100
                )
                train_a = _bootstrap(pool, seed + 1)
                train_b = _bootstrap(pool, seed + 2)
                model_a = fit_oulad_model(
                    model_name,
                    train_a,
                    train_a.target_unsuccessful.to_numpy(int),
                    features,
                    numeric,
                    CAT,
                    seed + 3,
                )
                model_b = fit_oulad_model(
                    model_name,
                    train_b,
                    train_b.target_unsuccessful.to_numpy(int),
                    features,
                    numeric,
                    CAT,
                    seed + 4,
                )
                fits += 2

                p_id = model_a.predict_proba(id_eval)
                p_target = model_a.predict_proba(target)
                p_id_b = model_b.predict_proba(id_eval)
                mid = _metrics(y_id, p_id)
                mt = _metrics(y_target, p_target)
                midb = _metrics(y_id, p_id_b)

                bid = _bundle(model_a, id_eval)
                bt = _bundle(model_a, target)
                bidb = _bundle(model_b, id_eval)
                btb = _bundle(model_b, target)
                calibrated = calibrated_attribution_shift(
                    bid["normalized"],
                    bt["normalized"],
                    bidb["normalized"],
                    y_id,
                    y_target,
                    normalized_target_b=btb["normalized"],
                    seed=seed + 41,
                    cap=explain_cap,
                    class_cap=max(20, explain_cap // 2),
                )
                si = calibrated.indices.source_shift
                sn = calibrated.indices.source_null
                ti = calibrated.indices.target_shift
                attr_id = bid["semantic"][si]
                attr_target = bt["semantic"][ti]
                attr_null = bidb["semantic"][sn]
                abs_id = np.mean(np.abs(attr_id), axis=0)
                abs_target = np.mean(np.abs(attr_target), axis=0)
                abs_null = np.mean(np.abs(attr_null), axis=0)
                signed_id = np.mean(attr_id, axis=0)
                signed_target = np.mean(attr_target, axis=0)
                signed_null = np.mean(attr_null, axis=0)
                shift_js = jensen_shannon_distance(abs_id, abs_target)
                null_js = jensen_shannon_distance(abs_id, abs_null)
                source_l1 = np.sum(np.abs(attr_id), axis=1)
                target_l1 = np.sum(np.abs(attr_target), axis=1)

                row: dict[str, Any] = {
                    "refit_pair": pair_index,
                    "split_index": split_index,
                    "refit_index": refit_index,
                    "task_index": task_i,
                    "code_module": pair.code_module,
                    "period": pair.period,
                    "source_presentation": pair.source_presentation,
                    "target_presentation": pair.target_presentation,
                    "horizon_day": horizon,
                    "model": model_name,
                    "feature_schema": "source_train_only_activity_types",
                    "n_source_activity_features": len(click_cols),
                    "n_excluded_source_id_only_activity_features": len(id_only_cols),
                    "excluded_source_id_only_activity_features": "|".join(id_only_cols),
                    "n_excluded_target_only_activity_features": len(target_only_cols),
                    "excluded_target_only_activity_features": "|".join(target_only_cols),
                    "n_source": len(source),
                    "n_target": len(target),
                    "n_source_train": len(pool),
                    "n_source_id": len(id_eval),
                    "n_explanation": calibrated.indices.sample_size,
                    "source_prevalence": float(source.target_unsuccessful.mean()),
                    "target_prevalence": float(target.target_unsuccessful.mean()),
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
                    "model_config_json": json.dumps(oulad_model_config(model_name), sort_keys=True),
                    "model_config_sha256": hashlib.sha256(
                        json.dumps(oulad_model_config(model_name), sort_keys=True).encode("utf-8")
                    ).hexdigest(),
                    "shift_js": shift_js,
                    "null_js": null_js,
                    "paired_excess_js": shift_js - null_js,
                    "shift_rank_tau": kendall_tau_b(abs_id, abs_target),
                    "null_rank_tau": kendall_tau_b(abs_id, abs_null),
                    "shift_top5_jaccard": top_k_jaccard(abs_id, abs_target, min(5, len(abs_id))),
                    "null_top5_jaccard": top_k_jaccard(abs_id, abs_null, min(5, len(abs_id))),
                    "shift_sign_consistency": weighted_sign_consistency(signed_id, signed_target),
                    "null_sign_consistency": weighted_sign_consistency(signed_id, signed_null),
                    "source_mean_attribution_l1": float(source_l1.mean()),
                    "target_mean_attribution_l1": float(target_l1.mean()),
                    "delta_mean_attribution_l1": float(target_l1.mean() - source_l1.mean()),
                    "max_additivity_error": max(
                        bid["additivity_error"],
                        bt["additivity_error"],
                        bidb["additivity_error"],
                        btb["additivity_error"],
                    ),
                }
                row.update(prefixed_attribution_l1_diagnostics(attr_id, "source_l1"))
                row.update(prefixed_attribution_l1_diagnostics(attr_null, "null_l1"))
                row.update(prefixed_attribution_l1_diagnostics(attr_target, "target_l1"))
                row.update(matched_l1_sensitivity(attr_id, attr_null, attr_target))
                if sampling_repeats:
                    row.update(repeated_explanation_subsample_sensitivity(
                        bid["normalized"][si], bidb["normalized"][sn], bt["normalized"][ti],
                        y_id[si], y_id[sn], y_target[ti],
                        seed=seed + 7041, repeats=sampling_repeats,
                    ))
                row["silent_explanation_shift"] = bool(
                    row["calibrated_excess_u2"] > 0
                    and abs(row["delta_auroc"]) <= row["null_abs_delta_auroc"]
                )
                rows.append(row)

                for j, feature in enumerate(model_a.semantic_map.names):
                    feature_rows.append(
                        {
                            "refit_pair": pair_index,
                            "split_index": split_index,
                            "refit_index": refit_index,
                            "task_index": task_i,
                            "code_module": pair.code_module,
                            "period": pair.period,
                            "horizon_day": horizon,
                            "model": model_name,
                            "feature": feature,
                            "id_mean_abs": float(abs_id[j]),
                            "target_mean_abs": float(abs_target[j]),
                            "delta_mean_abs": float(abs_target[j] - abs_id[j]),
                            "id_mean_signed": float(signed_id[j]),
                            "target_mean_signed": float(signed_target[j]),
                        }
                    )

    metadata = {
        "runtime_seconds": perf_counter() - t0,
        "n_model_fits": fits,
        "n_rows": len(rows),
        "split_index": split_index,
        "refit_index": refit_index,
        "primary_feature_schema": "source-training-partition-only activity-type columns",
        "model_configs": {
            name: oulad_model_config(name) for name in ("logistic", "lightgbm")
        },
    }
    return pd.DataFrame(rows), pd.DataFrame(feature_rows), metadata
