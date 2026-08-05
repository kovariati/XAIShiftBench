"""Observation-level prediction-invariance audit for the credit mechanism matrix.

The module recreates the model-A branch of the the release full mechanism matrix using
identical seeds. It records paired probability and explanation changes for each
evaluation case, allowing explanation drift to be examined after conditioning on
prediction-label invariance and the magnitude of probability movement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from .credit_mechanism_matrix import (
    MECHANISMS,
    RATE_BY_MECHANISM,
    _bundle,
    _row_stv,
    _stratified_bootstrap,
)
from .credit_models import fit_credit_model
from .datasets.south_german_credit import load_south_german_credit
from .missingness import inject_missingness

PROBABILITY_THRESHOLDS = (0.01, 0.025, 0.05)
PROBABILITY_BIN_EDGES = (-np.inf, 0.01, 0.025, 0.05, 0.10, np.inf)
PROBABILITY_BIN_LABELS = (
    "[0,0.01]",
    "(0.01,0.025]",
    "(0.025,0.05]",
    "(0.05,0.10]",
    ">0.10",
)


@dataclass(frozen=True)
class PredictionInvarianceMetadata:
    runtime_seconds: float
    n_pairs: int
    n_model_fits: int
    n_scenario_evaluations: int
    n_observation_comparisons: int


def _row_mask_jaccard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    intersection = np.sum(a & b, axis=1)
    union = np.sum(a | b, axis=1)
    return np.divide(
        intersection,
        union,
        out=np.ones_like(intersection, dtype=float),
        where=union > 0,
    )


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(spearmanr(x, y).statistic)


def _summarize_scenario(rows: pd.DataFrame) -> dict[str, float | int]:
    abs_dp = rows["abs_probability_shift"].to_numpy(float)
    stv = rows["explanation_stv"].to_numpy(float)
    unchanged = rows["prediction_label_unchanged"].to_numpy(bool)
    summary: dict[str, float | int] = {
        "n_observations": int(len(rows)),
        "mean_abs_probability_shift": float(np.mean(abs_dp)),
        "median_abs_probability_shift": float(np.median(abs_dp)),
        "mean_explanation_stv": float(np.mean(stv)),
        "median_explanation_stv": float(np.median(stv)),
        "q90_explanation_stv": float(np.quantile(stv, 0.90)),
        "label_invariant_rate": float(np.mean(unchanged)),
        "label_invariant_mean_stv": float(np.mean(stv[unchanged])) if unchanged.any() else np.nan,
        "spearman_abs_probability_vs_stv": _safe_spearman(abs_dp, stv),
        "mean_reference_row_missing": float(rows["reference_row_missing"].mean()),
        "mean_target_row_missing": float(rows["target_row_missing"].mean()),
        "mean_mask_jaccard": float(rows["mask_jaccard"].mean()),
    }
    for threshold in PROBABILITY_THRESHOLDS:
        mask = abs_dp <= threshold + 1e-15
        key = str(threshold).replace("0.", "")
        summary[f"prob_invariant_{key}_count"] = int(mask.sum())
        summary[f"prob_invariant_{key}_rate"] = float(mask.mean())
        summary[f"prob_invariant_{key}_mean_stv"] = (
            float(np.mean(stv[mask])) if mask.any() else np.nan
        )
        silent = mask & (stv >= 0.10)
        summary[f"silent_stv010_prob_{key}_rate"] = float(np.mean(silent))
        summary[f"silent_stv010_within_prob_{key}_rate"] = (
            float(np.mean(stv[mask] >= 0.10)) if mask.any() else np.nan
        )
    return summary


def _probability_bin_rows(rows: pd.DataFrame) -> list[dict[str, object]]:
    bins = pd.cut(
        rows["abs_probability_shift"],
        bins=PROBABILITY_BIN_EDGES,
        labels=PROBABILITY_BIN_LABELS,
        right=True,
        include_lowest=True,
    )
    output: list[dict[str, object]] = []
    for label in PROBABILITY_BIN_LABELS:
        group = rows.loc[bins == label]
        output.append(
            {
                "probability_bin": label,
                "n_observations": int(len(group)),
                "fraction_observations": float(len(group) / len(rows)),
                "mean_abs_probability_shift": (
                    float(group["abs_probability_shift"].mean()) if len(group) else np.nan
                ),
                "mean_explanation_stv": (
                    float(group["explanation_stv"].mean()) if len(group) else np.nan
                ),
                "median_explanation_stv": (
                    float(group["explanation_stv"].median()) if len(group) else np.nan
                ),
                "q90_explanation_stv": (
                    float(group["explanation_stv"].quantile(0.90)) if len(group) else np.nan
                ),
                "label_invariant_rate": (
                    float(group["prediction_label_unchanged"].mean()) if len(group) else np.nan
                ),
            }
        )
    return output


def run_credit_prediction_invariance(
    data_path: str | Path,
    seed_base: int = 2026073005,
    n_pairs: int = 1,
    pair_start: int = 0,
    models: tuple[str, ...] = ("logistic", "lightgbm"),
    indicator_modes: tuple[str, ...] = ("none", "all"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, PredictionInvarianceMetadata]:
    """Run observation-level prediction-invariance diagnostics.

    Returns observation rows, scenario summaries, probability-bin summaries,
    and metadata. Model seeds and masks match the the release mechanism matrix.
    """
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
    original_eval_index = np.asarray(eval_idx, dtype=int)

    observation_rows: list[dict[str, object]] = []
    scenario_rows: list[dict[str, object]] = []
    bin_rows: list[dict[str, object]] = []
    n_fits = 0
    n_scenarios = 0

    for pair in range(pair_start, pair_start + n_pairs):
        pair_seed = seed_base + pair * 1_000_000
        train_a_clean = _stratified_bootstrap(train_pool, pair_seed + 1)

        train_a = {}
        eval_primary = {}
        eval_secondary = {}
        for mechanism_index, mechanism in enumerate(MECHANISMS):
            rate = RATE_BY_MECHANISM[mechanism]
            train_a[mechanism] = inject_missingness(
                train_a_clean, mechanism, rate, pair_seed + 100 + mechanism_index
            )
            eval_primary[mechanism] = inject_missingness(
                eval_clean, mechanism, rate, pair_seed + 300 + mechanism_index
            )
            eval_secondary[mechanism] = inject_missingness(
                eval_clean, mechanism, rate, pair_seed + 400 + mechanism_index
            )

        for source_index, source_mechanism in enumerate(MECHANISMS):
            reference = eval_primary[source_mechanism]
            ref_mask = reference.mask.to_numpy(bool)
            for indicator_mode in indicator_modes:
                for model_index, model_name in enumerate(models):
                    model_seed = (
                        pair_seed
                        + source_index * 100_000
                        + model_index * 10_000
                        + (5000 if indicator_mode == "all" else 0)
                    )
                    model = fit_credit_model(
                        model_name,
                        indicator_mode,
                        train_a[source_mechanism].frame,
                        train_a_clean["target"].to_numpy(int),
                        model_seed + 1,
                    )
                    n_fits += 1
                    ref_bundle = _bundle(model, reference.frame)
                    p_ref = model.predict_proba(reference.frame)
                    pred_ref = p_ref >= 0.5

                    for target_mechanism in MECHANISMS:
                        target = eval_secondary[target_mechanism]
                        target_bundle = _bundle(model, target.frame)
                        p_target = model.predict_proba(target.frame)
                        pred_target = p_target >= 0.5
                        stv = _row_stv(ref_bundle["normalized"], target_bundle["normalized"])
                        abs_dp = np.abs(p_target - p_ref)
                        target_mask = target.mask.to_numpy(bool)
                        mask_jaccard = _row_mask_jaccard(ref_mask, target_mask)
                        scenario_records = pd.DataFrame(
                            {
                                "pair": pair,
                                "model": model_name,
                                "indicator_mode": indicator_mode,
                                "source_mechanism": source_mechanism,
                                "target_mechanism": target_mechanism,
                                "matched_mechanism": source_mechanism == target_mechanism,
                                "eval_position": np.arange(len(eval_clean), dtype=int),
                                "original_row_index": original_eval_index,
                                "y_true": y_eval,
                                "p_reference": p_ref,
                                "p_target": p_target,
                                "abs_probability_shift": abs_dp,
                                "predicted_reference": pred_ref.astype(int),
                                "predicted_target": pred_target.astype(int),
                                "prediction_label_unchanged": pred_ref == pred_target,
                                "explanation_stv": stv,
                                "reference_row_missing": ref_mask.sum(axis=1),
                                "target_row_missing": target_mask.sum(axis=1),
                                "mask_jaccard": mask_jaccard,
                            }
                        )
                        observation_rows.extend(scenario_records.to_dict("records"))
                        identity = {
                            "pair": pair,
                            "model": model_name,
                            "indicator_mode": indicator_mode,
                            "source_mechanism": source_mechanism,
                            "target_mechanism": target_mechanism,
                            "matched_mechanism": source_mechanism == target_mechanism,
                        }
                        scenario_rows.append({**identity, **_summarize_scenario(scenario_records)})
                        for bin_record in _probability_bin_rows(scenario_records):
                            bin_rows.append({**identity, **bin_record})
                        n_scenarios += 1

    observations = pd.DataFrame(observation_rows)
    scenarios = pd.DataFrame(scenario_rows)
    bins = pd.DataFrame(bin_rows)
    metadata = PredictionInvarianceMetadata(
        runtime_seconds=float(perf_counter() - start),
        n_pairs=n_pairs,
        n_model_fits=n_fits,
        n_scenario_evaluations=n_scenarios,
        n_observation_comparisons=len(observations),
    )
    return observations, scenarios, bins, metadata
