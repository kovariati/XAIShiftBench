"""ACSIncome 2018-to-2024 temporal attribution-transfer benchmark.

The preprocessing input must contain ADJINC-adjusted income.  The 2018 source
pool is resplit five times.  Unconditional explanation-distribution metrics are
constructed once from the nominal target because the fitted source models and
feature distributions do not change with the alternative 2024 label
threshold.  Class-conditional metrics are recalculated for each target
definition.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from .acs_models import fit_acs_model
from .calibrated_shift import (
    calibrated_shift_from_matched_arrays,
    matched_composition_indices,
)
from .metrics import (
    energy_u_statistic_squared,
    jensen_shannon_distance,
    kendall_tau_b,
    normalize_signed_rows,
    prefixed_attribution_l1_diagnostics,
    matched_l1_sensitivity,
    repeated_explanation_subsample_sensitivity,
    top_k_jaccard,
    weighted_sign_consistency,
    weighted_energy_u_statistic_squared,
)

TARGETS = {
    "nominal_50k": "TARGET_NOMINAL_50K",
    "real_2018_50k": "TARGET_REAL_2018_50K",
}


def _bootstrap(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pieces = []
    for _, group in frame.groupby("TARGET_NOMINAL_50K", sort=True):
        selected = rng.choice(group.index.to_numpy(), size=len(group), replace=True)
        pieces.append(frame.loc[selected])
    out = pd.concat(pieces, ignore_index=True)
    return out.iloc[rng.permutation(len(out))].reset_index(drop=True)


def _metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> dict[str, float]:
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, int)
    kwargs = {} if w is None else {"sample_weight": np.asarray(w, float)}
    return {
        "auroc": float(roc_auc_score(y, p, **kwargs)),
        "average_precision": float(average_precision_score(y, p, **kwargs)),
        "log_loss": float(log_loss(y, p, labels=[0, 1], **kwargs)),
        "brier": float(brier_score_loss(y, p, **kwargs)),
    }


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


def _source_split(source: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    strata = source.STATE.astype(str) + "_" + source.TARGET_NOMINAL_50K.astype(str)
    counts = strata.value_counts()
    rare = set(counts[counts < 2].index)
    if rare:
        strata = strata.where(~strata.isin(rare), source.STATE.astype(str) + "_rare")
    train_idx, id_idx = train_test_split(
        np.arange(len(source)), test_size=0.25, random_state=seed, stratify=strata
    )
    return source.iloc[train_idx].reset_index(drop=True), source.iloc[id_idx].reset_index(drop=True)


def load_acs_temporal_frames(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = root / "outputs" / "acs_temporal_pilot"
    source = pd.read_csv(out / "acs_2018_source_pool.csv.gz")
    target = pd.read_csv(out / "acs_2024_target_eval.csv.gz")
    return source, target


def _matched_explanation_bundle(
    model_a: Any,
    model_b: Any,
    id_eval: pd.DataFrame,
    target: pd.DataFrame,
    target_column: str,
    *,
    seed: int,
    explain_cap: int,
) -> dict[str, Any]:
    y_source = id_eval[target_column].to_numpy(int)
    y_target = target[target_column].to_numpy(int)
    indices = matched_composition_indices(
        y_source, y_target, cap=explain_cap, seed=seed, min_per_class=3
    )
    source_rows = pd.concat(
        [id_eval.iloc[indices.source_shift], id_eval.iloc[indices.source_null]],
        ignore_index=True,
    )
    target_rows = target.iloc[indices.target_shift].reset_index(drop=True)
    n = indices.sample_size
    bundle_source_a = _bundle(model_a, source_rows)
    bundle_source_b = _bundle(model_b, source_rows)
    bundle_target_a = _bundle(model_a, target_rows)
    bundle_target_b = _bundle(model_b, target_rows)
    source_shift_a = bundle_source_a["normalized"][:n]
    source_null_a = bundle_source_a["normalized"][n:]
    source_shift_b = bundle_source_b["normalized"][:n]
    source_null_b = bundle_source_b["normalized"][n:]
    target_a = bundle_target_a["normalized"]
    ys = y_source[indices.source_shift]
    yn = y_source[indices.source_null]
    yt = y_target[indices.target_shift]
    calibrated = calibrated_shift_from_matched_arrays(
        source_shift_a,
        source_null_a,
        target_a,
        source_shift_b,
        source_null_b,
        ys,
        yn,
        yt,
        target_b=bundle_target_b["normalized"],
    )
    return {
        "indices": indices,
        "calibrated": calibrated,
        "source_semantic_a": bundle_source_a["semantic"][:n],
        "source_null_semantic_b": bundle_source_b["semantic"][n:],
        "target_semantic_a": bundle_target_a["semantic"],
        "source_normalized_a": source_shift_a,
        "source_null_normalized_b": source_null_b,
        "target_normalized_a": target_a,
        "source_shift_rows": source_rows.iloc[:n].reset_index(drop=True),
        "source_null_rows": source_rows.iloc[n:].reset_index(drop=True),
        "source_rows": source_rows.iloc[:n].reset_index(drop=True),  # backward-compatible alias
        "target_rows": target_rows,
        "max_additivity_error": max(
            bundle_source_a["additivity_error"],
            bundle_source_b["additivity_error"],
            bundle_target_a["additivity_error"],
            bundle_target_b["additivity_error"],
        ),
    }


def run_pair(
    root: Path,
    pair_index: int,
    seed_base: int = 2026073008,
    explain_cap: int = 256,
    frames: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    n_splits: int = 5,
    refits_per_split: int = 4,
    sampling_repeats: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not 0 <= pair_index < n_splits * refits_per_split:
        raise ValueError("pair_index is outside the release-defined split/refit grid.")
    source, target = load_acs_temporal_frames(root) if frames is None else frames
    split_index = pair_index // refits_per_split
    refit_index = pair_index % refits_per_split
    split_seed = seed_base + split_index * 10_000
    train, id_eval = _source_split(source, split_seed)
    rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    subgroup_rows: list[dict[str, Any]] = []
    fits = 0
    t0 = perf_counter()

    for model_index, model_name in enumerate(("logistic", "lightgbm")):
        seed = seed_base + model_index * 100_000 + split_index * 10_000 + refit_index * 100
        train_a = _bootstrap(train, seed + 1)
        train_b = _bootstrap(train, seed + 2)
        model_a = fit_acs_model(
            model_name, train_a, train_a.TARGET_NOMINAL_50K.to_numpy(int), seed + 3
        )
        model_b = fit_acs_model(
            model_name, train_b, train_b.TARGET_NOMINAL_50K.to_numpy(int), seed + 4
        )
        fits += 2
        p_id = model_a.predict_proba(id_eval)
        p_target = model_a.predict_proba(target)
        p_id_b = model_b.predict_proba(id_eval)

        explanation_by_target = {
            name: _matched_explanation_bundle(
                model_a,
                model_b,
                id_eval,
                target,
                column,
                seed=seed + 40 + idx * 1000,
                explain_cap=explain_cap,
            )
            for idx, (name, column) in enumerate(TARGETS.items())
        }
        nominal_bundle = explanation_by_target["nominal_50k"]
        attr_id = nominal_bundle["source_semantic_a"]
        attr_target = nominal_bundle["target_semantic_a"]
        attr_null = nominal_bundle["source_null_semantic_b"]
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
        l1_audit = {}
        l1_audit.update(prefixed_attribution_l1_diagnostics(attr_id, "source_l1"))
        l1_audit.update(prefixed_attribution_l1_diagnostics(attr_null, "null_l1"))
        l1_audit.update(prefixed_attribution_l1_diagnostics(attr_target, "target_l1"))
        l1_audit.update(matched_l1_sensitivity(attr_id, attr_null, attr_target))
        unconditional = nominal_bundle["calibrated"]
        # Survey-weight sensitivity on the same release-defined matched rows.
        # This does not change the primary empirical-PUMS-record estimand and is
        # not claimed as a full design-based population estimator.
        survey_weighted_shift_u2 = weighted_energy_u_statistic_squared(
            nominal_bundle["source_normalized_a"],
            nominal_bundle["target_normalized_a"],
            nominal_bundle["source_shift_rows"].PWGTP.to_numpy(float),
            nominal_bundle["target_rows"].PWGTP.to_numpy(float),
        )
        survey_weighted_refit_null_u2 = weighted_energy_u_statistic_squared(
            nominal_bundle["source_normalized_a"],
            nominal_bundle["source_null_normalized_b"],
            nominal_bundle["source_shift_rows"].PWGTP.to_numpy(float),
            nominal_bundle["source_null_rows"].PWGTP.to_numpy(float),
        )
        survey_weighted_excess_u2 = float(survey_weighted_shift_u2 - survey_weighted_refit_null_u2)

        for target_name, target_column in TARGETS.items():
            y_id = id_eval[target_column].to_numpy(int)
            y_target = target[target_column].to_numpy(int)
            mid = _metrics(y_id, p_id)
            mt = _metrics(y_target, p_target)
            midb = _metrics(y_id, p_id_b)
            midw = _metrics(y_id, p_id, id_eval.PWGTP.to_numpy(float))
            mtw = _metrics(y_target, p_target, target.PWGTP.to_numpy(float))
            target_matched_bundle = explanation_by_target[target_name]
            conditional = target_matched_bundle["calibrated"]
            row: dict[str, Any] = {
                "refit_pair": pair_index,
                "split_index": split_index,
                "refit_index": refit_index,
                "split_seed": split_seed,
                "model": model_name,
                "target_definition": target_name,
                "target_column": target_column,
                "unconditional_explanation_construction": "identical_across_target_definitions",
                "n_train": len(train),
                "n_source_id": len(id_eval),
                "n_target": len(target),
                "n_explanation": nominal_bundle["indices"].sample_size,
                "target_matched_n_explanation": target_matched_bundle["indices"].sample_size,
                "id_prevalence": float(y_id.mean()),
                "target_prevalence": float(y_target.mean()),
                "id_weighted_prevalence": float(np.average(y_id, weights=id_eval.PWGTP)),
                "target_weighted_prevalence": float(np.average(y_target, weights=target.PWGTP)),
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
                "weighted_id_auroc": midw["auroc"],
                "weighted_target_auroc": mtw["auroc"],
                "weighted_delta_auroc": mtw["auroc"] - midw["auroc"],
                "weighted_id_log_loss": midw["log_loss"],
                "weighted_target_log_loss": mtw["log_loss"],
                "weighted_delta_log_loss": mtw["log_loss"] - midw["log_loss"],
                "null_abs_delta_auroc": abs(midb["auroc"] - mid["auroc"]),
                "shift_energy_u2": unconditional["shift_u2"],
                "same_model_sample_null_energy_u2": unconditional["same_model_sample_null_u2"],
                "refit_sample_null_energy_u2": unconditional["refit_sample_null_u2"],
                "calibrated_excess_u2": unconditional["excess_u2"],
                # the release target-definition sensitivity. Historical unconditional
                # fields above deliberately retain the nominal-50k release endpoint
                # for exact backward parity. These fields redraw the matched rows
                # under each target definition and therefore expose the effect of
                # the 2024 CPI-adjusted label definition on the explanation endpoint.
                "target_matched_shift_energy_u2": conditional["shift_u2"],
                "target_matched_same_model_sample_null_energy_u2": conditional["same_model_sample_null_u2"],
                "target_matched_refit_null_energy_u2": conditional["refit_sample_null_u2"],
                "target_matched_excess_u2": conditional["excess_u2"],
                "target_matched_symmetrized_excess_u2": conditional["symmetrized_excess_u2"],
                "paired_refit_l2": unconditional["paired_refit_l2"],
                "class_macro_shift_u2": conditional["class_macro_shift_u2"],
                "class_macro_sample_null_u2": conditional["class_macro_sample_null_u2"],
                "class_macro_refit_null_u2": conditional["class_macro_refit_null_u2"],
                "class_macro_excess_u2": conditional["class_macro_excess_u2"],
                "class_weighted_shift_u2": conditional["class_weighted_shift_u2"],
                "class_weighted_refit_null_u2": conditional["class_weighted_refit_null_u2"],
                "class_weighted_excess_u2": conditional["class_weighted_excess_u2"],
                "class0_shift_u2": conditional["class0_shift_u2"],
                "class1_shift_u2": conditional["class1_shift_u2"],
                "class0_refit_null_u2": conditional["class0_refit_null_u2"],
                "class1_refit_null_u2": conditional["class1_refit_null_u2"],
                "class0_n": conditional["class0_n"],
                "class1_n": conditional["class1_n"],
                "class_weighting_prevalence": conditional["class_weighting_prevalence"],
                "class_weighting_scheme": conditional["class_weighting_scheme"],
                "reverse_shift_energy_u2": unconditional["reverse_shift_u2"],
                "reverse_refit_sample_null_energy_u2": unconditional["reverse_refit_sample_null_u2"],
                "reverse_calibrated_excess_u2": unconditional["reverse_excess_u2"],
                "symmetrized_calibrated_excess_u2": unconditional["symmetrized_excess_u2"],
                "symmetry_gap_u2": unconditional["symmetry_gap_u2"],
                "pwgpt_weighted_shift_energy_u2": survey_weighted_shift_u2,
                "pwgpt_weighted_refit_null_energy_u2": survey_weighted_refit_null_u2,
                "pwgpt_weighted_excess_u2": survey_weighted_excess_u2,
                "pwgpt_weighting_note": "sensitivity_on_release_defined_matched_rows_not_full_design_based_estimator",
                "shift_js": shift_js,
                "null_js": null_js,
                "paired_excess_js": shift_js - null_js,
                "shift_rank_tau": kendall_tau_b(abs_id, abs_target),
                "null_rank_tau": kendall_tau_b(abs_id, abs_null),
                "shift_top5_jaccard": top_k_jaccard(abs_id, abs_target, 5),
                "null_top5_jaccard": top_k_jaccard(abs_id, abs_null, 5),
                "shift_sign_consistency": weighted_sign_consistency(signed_id, signed_target),
                "null_sign_consistency": weighted_sign_consistency(signed_id, signed_null),
                "source_mean_attribution_l1": float(source_l1.mean()),
                "target_mean_attribution_l1": float(target_l1.mean()),
                "delta_mean_attribution_l1": float(target_l1.mean() - source_l1.mean()),
                "max_additivity_error": max(
                    nominal_bundle["max_additivity_error"],
                    explanation_by_target[target_name]["max_additivity_error"],
                ),
            }
            row.update(l1_audit)
            if sampling_repeats and target_name == "nominal_50k":
                idx_nom = nominal_bundle["indices"]
                y_nom_id = id_eval["TARGET_NOMINAL_50K"].to_numpy(int)
                y_nom_target = target["TARGET_NOMINAL_50K"].to_numpy(int)
                row.update(repeated_explanation_subsample_sensitivity(
                    nominal_bundle["source_normalized_a"],
                    nominal_bundle["source_null_normalized_b"],
                    nominal_bundle["target_normalized_a"],
                    y_nom_id[idx_nom.source_shift],
                    y_nom_id[idx_nom.source_null],
                    y_nom_target[idx_nom.target_shift],
                    seed=seed + 7041, repeats=sampling_repeats,
                ))
            row["silent_explanation_shift"] = bool(
                row["calibrated_excess_u2"] > 0
                and abs(row["delta_auroc"]) <= row["null_abs_delta_auroc"]
            )
            rows.append(row)

        for feature_index, feature in enumerate(model_a.semantic_map.names):
            feature_rows.append(
                {
                    "refit_pair": pair_index,
                    "split_index": split_index,
                    "refit_index": refit_index,
                    "model": model_name,
                    "feature": feature,
                    "id_mean_abs": float(abs_id[feature_index]),
                    "target_mean_abs": float(abs_target[feature_index]),
                    "delta_mean_abs": float(abs_target[feature_index] - abs_id[feature_index]),
                    "id_mean_signed": float(signed_id[feature_index]),
                    "target_mean_signed": float(signed_target[feature_index]),
                }
            )

        source_rows = nominal_bundle["source_rows"]
        target_rows = nominal_bundle["target_rows"]
        for group in ("SEX", "RAC1P"):
            levels = sorted(set(source_rows[group].dropna()) & set(target_rows[group].dropna()))
            for level in levels:
                a = nominal_bundle["source_normalized_a"][source_rows[group].to_numpy() == level]
                b = nominal_bundle["target_normalized_a"][target_rows[group].to_numpy() == level]
                if min(len(a), len(b)) >= 2:
                    subgroup_rows.append(
                        {
                            "refit_pair": pair_index,
                            "split_index": split_index,
                            "refit_index": refit_index,
                            "model": model_name,
                            "group": group,
                            "level": level,
                            "n_source": len(a),
                            "n_target": len(b),
                            "descriptive_shift_energy_u2": energy_u_statistic_squared(a, b),
                            "interpretation": "descriptive_not_fairness_or_causal",
                        }
                    )

    metadata = {
        "runtime_seconds": perf_counter() - t0,
        "n_model_fits": fits,
        "n_rows": len(rows),
        "split_index": split_index,
        "refit_index": refit_index,
        "income_variable": "PINCP_ADJ",
        "target_definition_note": "Unconditional explanation metrics are identical across target definitions by construction; prediction and class-conditional metrics vary.",
    }
    return pd.DataFrame(rows), pd.DataFrame(feature_rows), pd.DataFrame(subgroup_rows), metadata
