"""Four-site Heart Disease attribution-shift benchmark.

The release uses repeated source train/ID splits and an off-diagonal energy
U-statistic.  The deployment contrast is calibrated against disjoint source
samples with matched sample sizes and class compositions; the second source
sample is also evaluated under an independent model refit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .calibrated_shift import calibrated_attribution_shift
from .datasets.heart_disease import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, load_heart_sites
from .domain_shift import domain_classifier_auc
from .explainers import explain_lightgbm_native_contrib, explain_logistic_centered
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
from .null_calibration import empirical_exceedance_fraction, standardized_excess
from .semantic import SemanticMap, semantic_map_from_preprocessor

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")


@dataclass(frozen=True)
class FittedModel:
    name: str
    preprocessor: ColumnTransformer
    estimator: Any
    semantic_map: SemanticMap
    background_encoded: np.ndarray

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.preprocessor.transform(x[FEATURES]), dtype=float)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(self.transform(x))[:, 1], dtype=float)

    def raw_score(self, x: pd.DataFrame) -> np.ndarray:
        z = self.transform(x)
        if self.name == "lightgbm":
            return np.asarray(self.estimator.booster_.predict(z, raw_score=True), dtype=float)
        return np.asarray(self.estimator.decision_function(z), dtype=float)

    def explain(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
        encoded = self.transform(x)
        if self.name == "logistic":
            result = explain_logistic_centered(
                self.estimator, encoded, self.background_encoded, self.semantic_map
            )
        elif self.name == "lightgbm":
            result = explain_lightgbm_native_contrib(
                self.estimator, encoded, self.semantic_map
            )
        else:
            raise ValueError(f"Unsupported model: {self.name}")
        return result.encoded, result.semantic, result.base_value


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )


def fit_model(name: str, train: pd.DataFrame, seed: int) -> FittedModel:
    preprocessor = make_preprocessor()
    encoded = np.asarray(preprocessor.fit_transform(train[FEATURES]), dtype=float)
    if name == "logistic":
        estimator: Any = LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=5000, random_state=seed
        )
    elif name == "lightgbm":
        estimator = LGBMClassifier(
            objective="binary",
            n_estimators=160,
            learning_rate=0.04,
            num_leaves=15,
            min_child_samples=15,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    estimator.fit(encoded, train["target"].to_numpy(int))
    return FittedModel(
        name=name,
        preprocessor=preprocessor,
        estimator=estimator,
        semantic_map=semantic_map_from_preprocessor(
            preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES
        ),
        background_encoded=encoded,
    )


def _bootstrap(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pieces = []
    for _, group in frame.groupby("target", sort=True):
        chosen = rng.choice(group.index.to_numpy(), size=len(group), replace=True)
        pieces.append(frame.loc[chosen])
    sampled = pd.concat(pieces, ignore_index=True)
    return sampled.iloc[rng.permutation(len(sampled))].reset_index(drop=True)


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


def _explain_bundle(model: FittedModel, x: pd.DataFrame, tolerance: float = 1e-7) -> dict[str, Any]:
    encoded, semantic, base = model.explain(x)
    error = float(np.max(np.abs(model.raw_score(x) - (base + encoded.sum(axis=1)))))
    if error > tolerance:
        raise AssertionError(f"Additivity gate failed for {model.name}: {error:.3e}")
    return {
        "encoded": encoded,
        "semantic": semantic,
        "normalized": normalize_signed_rows(semantic),
        "base": base,
        "additivity_error": error,
    }


def _semantic_permutation_profile(
    fitted: FittedModel, x: pd.DataFrame, y: np.ndarray, seed: int, repeats: int = 5
) -> np.ndarray:
    baseline = log_loss(y, fitted.predict_proba(x), labels=[0, 1])
    rng = np.random.default_rng(seed)
    values = np.zeros(len(FEATURES), dtype=float)
    for feature_index, feature in enumerate(FEATURES):
        original = x[feature].to_numpy(copy=True)
        losses = []
        for _ in range(repeats):
            permuted = x.copy()
            permuted[feature] = original[rng.permutation(len(original))]
            losses.append(log_loss(y, fitted.predict_proba(permuted), labels=[0, 1]) - baseline)
        values[feature_index] = float(np.mean(losses))
    return values


def _source_split(source: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    strata = source["site"].astype(str) + "|" + source["target"].astype(str)
    train_idx, id_idx = train_test_split(
        np.arange(len(source)), test_size=0.30, random_state=seed, stratify=strata
    )
    return source.iloc[train_idx].reset_index(drop=True), source.iloc[id_idx].reset_index(drop=True)


def run_heart_cross_site_pilot(
    data_dir: str | Path,
    seed_base: int = 20260729,
    n_pairs: int = 20,
    pair_start: int = 0,
    total_pairs: int = 20,
    n_splits: int = 5,
    explain_cap: int = 160,
    models: tuple[str, ...] = ("logistic", "lightgbm"),
    sampling_repeats: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if total_pairs < n_splits or total_pairs % n_splits:
        raise ValueError("total_pairs must be a positive multiple of n_splits.")
    if pair_start < 0 or n_pairs < 1 or pair_start + n_pairs > total_pairs:
        raise ValueError("Requested pair range is outside the release-defined grid.")
    refits_per_split = total_pairs // n_splits
    heart = load_heart_sites(data_dir).frame
    all_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []

    for target_index, target_site in enumerate(sorted(heart["site"].unique())):
        target = heart.loc[heart["site"] == target_site].reset_index(drop=True)
        source = heart.loc[heart["site"] != target_site].reset_index(drop=True)
        y_target = target["target"].to_numpy(int)

        for model_index, model_name in enumerate(models):
            for pair in range(pair_start, pair_start + n_pairs):
                split_index = pair // refits_per_split
                refit_index = pair % refits_per_split
                split_seed = seed_base + target_index * 1_000_000 + split_index * 10_000
                train_pool, id_eval = _source_split(source, split_seed)
                y_id = id_eval["target"].to_numpy(int)
                pair_seed = (
                    seed_base
                    + target_index * 10_000_000
                    + model_index * 1_000_000
                    + split_index * 10_000
                    + refit_index * 100
                )
                model_a = fit_model(model_name, _bootstrap(train_pool, pair_seed + 1), pair_seed + 11)
                model_b = fit_model(model_name, _bootstrap(train_pool, pair_seed + 2), pair_seed + 12)

                p_id_a = model_a.predict_proba(id_eval)
                p_target_a = model_a.predict_proba(target)
                p_id_b = model_b.predict_proba(id_eval)
                metrics_id = _prediction_metrics(y_id, p_id_a)
                metrics_target = _prediction_metrics(y_target, p_target_a)
                metrics_id_b = _prediction_metrics(y_id, p_id_b)

                bundle_id_a = _explain_bundle(model_a, id_eval)
                bundle_target_a = _explain_bundle(model_a, target)
                bundle_id_b = _explain_bundle(model_b, id_eval)
                bundle_target_b = _explain_bundle(model_b, target)
                calibrated = calibrated_attribution_shift(
                    bundle_id_a["normalized"],
                    bundle_target_a["normalized"],
                    bundle_id_b["normalized"],
                    y_id,
                    y_target,
                    normalized_target_b=bundle_target_b["normalized"],
                    seed=pair_seed + 41,
                    cap=explain_cap,
                    class_cap=max(20, explain_cap // 2),
                )
                si = calibrated.indices.source_shift
                sn = calibrated.indices.source_null
                ti = calibrated.indices.target_shift
                attr_id_a = bundle_id_a["semantic"][si]
                attr_target_a = bundle_target_a["semantic"][ti]
                attr_id_b = bundle_id_b["semantic"][sn]
                norm_id_a = bundle_id_a["normalized"][si]
                norm_target_a = bundle_target_a["normalized"][ti]
                abs_id_a, signed_id_a = _global_profiles(attr_id_a)
                abs_target_a, signed_target_a = _global_profiles(attr_target_a)
                abs_id_b, signed_id_b = _global_profiles(attr_id_b)
                shift_js = jensen_shannon_distance(abs_id_a, abs_target_a)
                null_js = jensen_shannon_distance(abs_id_a, abs_id_b)

                if refit_index == 0:
                    feature_domain_auc = domain_classifier_auc(
                        model_a.transform(id_eval.iloc[si]), model_a.transform(target.iloc[ti]), pair_seed + 51
                    )
                    explanation_domain_auc = domain_classifier_auc(
                        norm_id_a, norm_target_a, pair_seed + 52
                    )
                    perm_id = _semantic_permutation_profile(
                        model_a, id_eval.iloc[si], y_id[si], pair_seed + 61
                    )
                    perm_target = _semantic_permutation_profile(
                        model_a, target.iloc[ti], y_target[ti], pair_seed + 62
                    )
                    shap_perm_tau_id = kendall_tau_b(abs_id_a, perm_id)
                    shap_perm_tau_target = kendall_tau_b(abs_target_a, perm_target)
                else:
                    feature_domain_auc = explanation_domain_auc = float("nan")
                    perm_id = perm_target = np.full(len(FEATURES), np.nan)
                    shap_perm_tau_id = shap_perm_tau_target = float("nan")

                source_l1 = np.sum(np.abs(attr_id_a), axis=1)
                target_l1 = np.sum(np.abs(attr_target_a), axis=1)
                row: dict[str, Any] = {
                    "target_site": target_site,
                    "model": model_name,
                    "pair": pair,
                    "split_index": split_index,
                    "refit_index": refit_index,
                    "split_seed": split_seed,
                    "seed_a": pair_seed + 11,
                    "seed_b": pair_seed + 12,
                    "n_train_pool": len(train_pool),
                    "n_id": len(id_eval),
                    "n_target": len(target),
                    "n_explanation": calibrated.indices.sample_size,
                    "source_explanation_positive": calibrated.indices.source_positive_count,
                    "target_explanation_positive": calibrated.indices.target_positive_count,
                    "target_prevalence": float(y_target.mean()),
                    "id_auroc": metrics_id["auroc"],
                    "target_auroc": metrics_target["auroc"],
                    "delta_auroc": metrics_target["auroc"] - metrics_id["auroc"],
                    "null_abs_delta_auroc": abs(metrics_id["auroc"] - metrics_id_b["auroc"]),
                    "id_average_precision": metrics_id["average_precision"],
                    "target_average_precision": metrics_target["average_precision"],
                    "id_log_loss": metrics_id["log_loss"],
                    "target_log_loss": metrics_target["log_loss"],
                    "id_brier": metrics_id["brier"],
                    "target_brier": metrics_target["brier"],
                    "mean_abs_probability_refit_change": float(np.mean(np.abs(p_id_a - p_id_b))),
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
                    "shift_top5_jaccard": top_k_jaccard(abs_id_a, abs_target_a, 5),
                    "null_top5_jaccard": top_k_jaccard(abs_id_a, abs_id_b, 5),
                    "shift_sign_consistency": weighted_sign_consistency(signed_id_a, signed_target_a),
                    "null_sign_consistency": weighted_sign_consistency(signed_id_a, signed_id_b),
                    "source_mean_attribution_l1": float(source_l1.mean()),
                    "target_mean_attribution_l1": float(target_l1.mean()),
                    "delta_mean_attribution_l1": float(target_l1.mean() - source_l1.mean()),
                    "feature_domain_auc": feature_domain_auc,
                    "explanation_domain_auc": explanation_domain_auc,
                    "shap_permutation_tau_id": shap_perm_tau_id,
                    "shap_permutation_tau_target": shap_perm_tau_target,
                    "max_additivity_error": max(
                        bundle_id_a["additivity_error"],
                        bundle_target_a["additivity_error"],
                        bundle_id_b["additivity_error"],
                        bundle_target_b["additivity_error"],
                    ),
                }
                row.update(prefixed_attribution_l1_diagnostics(attr_id_a, "source_l1"))
                row.update(prefixed_attribution_l1_diagnostics(attr_id_b, "null_l1"))
                row.update(prefixed_attribution_l1_diagnostics(attr_target_a, "target_l1"))
                row.update(matched_l1_sensitivity(attr_id_a, attr_id_b, attr_target_a))
                if sampling_repeats:
                    row.update(repeated_explanation_subsample_sensitivity(
                        norm_id_a, bundle_id_b["normalized"][sn], norm_target_a,
                        y_id[si], y_id[sn], y_target[ti],
                        seed=pair_seed + 7041, repeats=sampling_repeats,
                    ))
                row["silent_explanation_shift"] = bool(
                    row["calibrated_excess_u2"] > 0
                    and abs(row["delta_auroc"]) <= row["null_abs_delta_auroc"]
                )
                all_rows.append(row)

                for feature_idx, feature in enumerate(model_a.semantic_map.names):
                    importance_rows.append(
                        {
                            "target_site": target_site,
                            "model": model_name,
                            "pair": pair,
                            "split_index": split_index,
                            "refit_index": refit_index,
                            "feature": feature,
                            "mean_abs_explanation_id": float(abs_id_a[feature_idx]),
                            "mean_abs_explanation_target": float(abs_target_a[feature_idx]),
                            "mean_signed_explanation_id": float(signed_id_a[feature_idx]),
                            "mean_signed_explanation_target": float(signed_target_a[feature_idx]),
                            "permutation_logloss_increase_id": float(perm_id[feature_idx]),
                            "permutation_logloss_increase_target": float(perm_target[feature_idx]),
                        }
                    )

    refits = pd.DataFrame(all_rows)
    importance = pd.DataFrame(importance_rows)
    return refits, summarize_heart_pilot(refits), importance


def _wilcoxon_p(values: pd.Series) -> float:
    arr = values.dropna().to_numpy(float)
    if len(arr) == 0 or np.allclose(arr, 0.0):
        return 1.0
    return float(wilcoxon(arr, alternative="greater", zero_method="wilcox").pvalue)


def summarize_heart_pilot(refits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (site, model), group in refits.groupby(["target_site", "model"], sort=True):
        observed = float(group["shift_energy_u2"].median())
        null = group["refit_sample_null_energy_u2"]
        q95 = float(null.quantile(0.95))
        q95_pred = float(group["null_abs_delta_auroc"].quantile(0.95))
        silent = (group["shift_energy_u2"] > q95) & (
            group["delta_auroc"].abs() <= q95_pred
        )
        rows.append(
            {
                "target_site": site,
                "model": model,
                "n_pairs": int(len(group)),
                "n_splits": int(group["split_index"].nunique()),
                "id_auroc_mean": float(group["id_auroc"].mean()),
                "target_auroc_mean": float(group["target_auroc"].mean()),
                "delta_auroc_mean": float(group["delta_auroc"].mean()),
                "shift_energy_u2_median": observed,
                "sample_null_energy_u2_median": float(group["same_model_sample_null_energy_u2"].median()),
                "refit_sample_null_energy_u2_median": float(null.median()),
                "calibrated_excess_u2_mean": float(group["calibrated_excess_u2"].mean()),
                "calibrated_excess_u2_median": float(group["calibrated_excess_u2"].median()),
                "positive_excess_fraction": float((group["calibrated_excess_u2"] > 0).mean()),
                "shift_above_refit_null_q95_fraction": float((group["shift_energy_u2"] > q95).mean()),
                "seei_u2": standardized_excess(observed, null) if len(null) >= 4 else float("nan"),
                "null_exceedance_fraction_u2": empirical_exceedance_fraction(observed, null) if len(null) >= 4 else float("nan"),
                "wilcoxon_greater_excess_p_conditional": _wilcoxon_p(group["calibrated_excess_u2"]),
                "class_macro_excess_u2_mean": float(group["class_macro_excess_u2"].mean()),
                "class_weighted_excess_u2_mean": float(group["class_weighted_excess_u2"].mean()),
                "shift_js_median": float(group["shift_js"].median()),
                "null_js_median": float(group["null_js"].median()),
                "paired_excess_js_median": float(group["paired_excess_js"].median()),
                "shift_rank_tau_median": float(group["shift_rank_tau"].median()),
                "shift_top5_jaccard_median": float(group["shift_top5_jaccard"].median()),
                "shift_sign_consistency_median": float(group["shift_sign_consistency"].median()),
                "delta_mean_attribution_l1_mean": float(group["delta_mean_attribution_l1"].mean()),
                "feature_domain_auc_mean": float(group["feature_domain_auc"].mean()),
                "explanation_domain_auc_mean": float(group["explanation_domain_auc"].mean()),
                "silent_explanation_shift_rate_diagnostic": float(silent.mean()),
                "refit_sample_null_energy_u2_q95": q95,
                "null_abs_delta_auroc_q95": q95_pred,
                "max_additivity_error": float(group["max_additivity_error"].max()),
            }
        )
    return pd.DataFrame(rows)
