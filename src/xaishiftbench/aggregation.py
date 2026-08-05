"""Explicit aggregation rules for XAIShiftBench deployment contrasts."""
from __future__ import annotations
from collections.abc import Iterable
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

ADDITIVE_COLUMNS = [
    "shift_energy_u2", "same_model_sample_null_energy_u2", "refit_sample_null_energy_u2",
    "calibrated_excess_u2", "sample_adjusted_excess_u2", "refit_increment_u2",
    "class_macro_excess_u2", "target_prevalence_weighted_class_excess_u2",
    "class0_excess_u2", "class1_excess_u2", "delta_auroc", "abs_delta_auroc",
    "paired_excess_js", "shift_rank_tau", "delta_mean_attribution_l1",
    "log_attribution_l1_ratio", "symmetrized_calibrated_excess_u2", "symmetry_gap_u2",
]

def exact_split_tests(values: Iterable[float]) -> dict[str, float | int]:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n_split_observations": 0, "positive_split_count": 0,
                "sign_test_greater_p": np.nan, "wilcoxon_split_greater_p": np.nan}
    positive = int(np.sum(x > 0))
    sign_p = float(binomtest(positive, int(x.size), .5, alternative="greater").pvalue)
    try:
        wp = float(wilcoxon(x, alternative="greater", zero_method="wilcox", method="auto").pvalue) if x.size >= 2 and not np.allclose(x, 0) else np.nan
    except ValueError:
        wp = np.nan
    return {"n_split_observations": int(x.size), "positive_split_count": positive,
            "sign_test_greater_p": sign_p, "wilcoxon_split_greater_p": wp}

def prepare_pair_rows(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    d["sample_adjusted_excess_u2"] = d["shift_energy_u2"] - d["same_model_sample_null_energy_u2"]
    d["refit_increment_u2"] = d["refit_sample_null_energy_u2"] - d["same_model_sample_null_energy_u2"]
    d["class0_excess_u2"] = d["class0_shift_u2"] - d["class0_refit_null_u2"]
    d["class1_excess_u2"] = d["class1_shift_u2"] - d["class1_refit_null_u2"]
    p = pd.to_numeric(d["target_prevalence"], errors="coerce")
    d["target_prevalence_weighted_class_excess_u2"] = (1-p)*d["class0_excess_u2"] + p*d["class1_excess_u2"]
    d["class_weighting_prevalence"] = p
    d["class_weighting_scheme"] = "target_prevalence"
    d["abs_delta_auroc"] = d["delta_auroc"].abs()
    with np.errstate(divide="ignore", invalid="ignore"):
        d["log_attribution_l1_ratio"] = np.log(pd.to_numeric(d["target_mean_attribution_l1"], errors="coerce") / pd.to_numeric(d["source_mean_attribution_l1"], errors="coerce"))
    for c in ["symmetrized_calibrated_excess_u2", "symmetry_gap_u2"]:
        if c not in d:
            d[c] = np.nan
    return d

def split_level_summary(pair_rows: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    records = []
    for keys, g in pair_rows.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        r: dict[str, Any] = dict(zip(group_columns, keys))
        r["n_pair_rows"] = len(g)
        r["n_refits"] = g["refit_index"].nunique()
        for c in ADDITIVE_COLUMNS:
            if c in g:
                r[f"mean_{c}"] = float(pd.to_numeric(g[c], errors="coerce").mean())
        r["positive_pair_fraction"] = float((g["calibrated_excess_u2"] > 0).mean())
        r["max_additivity_error"] = float(g["max_additivity_error"].max())
        records.append(r)
    return pd.DataFrame(records)

def contrast_summary_from_splits(
    split_rows: pd.DataFrame,
    group_columns: list[str],
    *,
    value_column: str = "mean_calibrated_excess_u2",
    pair_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    records = []
    for keys, g in split_rows.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        r: dict[str, Any] = dict(zip(group_columns, keys))
        v = pd.to_numeric(g[value_column], errors="coerce").dropna()
        r.update(
            n_splits=int(g["split_index"].nunique()),
            mean_calibrated_excess_u2=float(v.mean()),
            median_calibrated_excess_u2=float(v.median()),
            min_split_mean_u2=float(v.min()),
            max_split_mean_u2=float(v.max()),
            positive_split_fraction=float((v > 0).mean()),
            split_q05_u2=float(v.quantile(.05)),
            split_q95_u2=float(v.quantile(.95)),
        )
        r.update(exact_split_tests(v))
        for c in g.columns:
            if c.startswith("mean_") and c != value_column:
                r[c] = float(pd.to_numeric(g[c], errors="coerce").mean())
        r["max_additivity_error"] = float(g["max_additivity_error"].max())
        if pair_rows is not None:
            mask = np.ones(len(pair_rows), dtype=bool)
            for c, k in zip(group_columns, keys):
                mask &= pair_rows[c].astype(str).to_numpy() == str(k)
            raw = pair_rows.loc[mask]
            r["n_pair_rows"] = len(raw)
            r["positive_pair_fraction"] = float((raw["calibrated_excess_u2"] > 0).mean()) if len(raw) else np.nan
            r["pair_refit_null_q95_u2"] = float(raw["refit_sample_null_energy_u2"].quantile(.95)) if len(raw) else np.nan
        records.append(r)
    return pd.DataFrame(records)
