"""Corrected multi-domain aggregation for XAIShiftBench the release.

All uncertainty summaries use five split-level observations. Refits are nested
computational repeats and nonlinear statistics are never averaged from lower
levels.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from xaishiftbench.aggregation import (
    contrast_summary_from_splits,
    prepare_pair_rows,
    split_level_summary,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "multidomain"
FIG = ROOT / "figures" / "multidomain"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
SEED = 2026073113

def add_identifiers(frame, dataset, task_columns, family_columns):
    d = frame.copy()
    d["dataset"] = dataset
    d["task_id"] = d[task_columns].astype(str).agg("|".join, axis=1)
    d["family_id"] = d[family_columns].astype(str).agg("|".join, axis=1)
    pair_col = "pair" if "pair" in d else "refit_pair"
    d["pair_id"] = d[pair_col].astype(int)
    return prepare_pair_rows(d)

def load_rows():
    heart = add_identifiers(
        pd.read_csv(ROOT / "outputs/heart/heart_cross_site_refits.csv"),
        "Heart Disease", ["target_site"], ["target_site"])
    student = add_identifiers(
        pd.read_csv(ROOT / "outputs/student/student_institution_refits.csv"),
        "Student Performance", ["direction", "representation"], ["direction"])
    oulad = add_identifiers(
        pd.read_csv(ROOT / "outputs/oulad/oulad_temporal_refits.csv"),
        "OULAD", ["code_module", "period", "horizon_day"], ["code_module", "period"])
    acs = pd.read_csv(ROOT / "outputs/acs/acs_temporal_refits.csv")
    acs = acs.loc[acs["target_definition"] == "nominal_50k"].copy()
    acs["contrast"] = "2018_to_2024"
    acs = add_identifiers(acs, "ACSIncome", ["contrast"], ["contrast"])
    combined = pd.concat([heart, student, oulad, acs], ignore_index=True, sort=False)
    required = {
        "dataset", "task_id", "family_id", "model", "split_index",
        "refit_index", "calibrated_excess_u2",
        "target_prevalence_weighted_class_excess_u2",
    }
    missing = sorted(required - set(combined.columns))
    if missing:
        raise KeyError(f"Missing required pair-level fields: {missing}")
    return combined

def holm_split_tests(summary):
    out = summary.copy()
    for source, target in [
        ("sign_test_greater_p", "sign_test_holm_p"),
        ("wilcoxon_split_greater_p", "wilcoxon_split_holm_p"),
    ]:
        out[target] = np.nan
        valid = out[source].notna()
        if valid.any():
            out.loc[valid, target] = multipletests(
                out.loc[valid, source], method="holm")[1]
    return out

def dataset_summary(strict):
    rows = []
    for dataset, g in strict.groupby("dataset", sort=True):
        v = g["mean_calibrated_excess_u2"].to_numpy(float)
        rows.append({
            "dataset": dataset,
            "n_strict_contrasts": int(len(g)),
            "mean_excess_u2": float(v.mean()),
            "median_excess_u2": float(np.median(v)),
            "positive_contrasts": int(np.sum(v > 0)),
            "mean_target_weighted_class_excess_u2":
                float(g["mean_target_prevalence_weighted_class_excess_u2"].mean()),
            "mean_class_macro_excess_u2":
                float(g["mean_class_macro_excess_u2"].mean()),
            "mean_symmetrized_excess_u2":
                float(g["mean_symmetrized_calibrated_excess_u2"].mean()),
            "mean_log_attribution_l1_ratio":
                float(g["mean_log_attribution_l1_ratio"].mean()),
            "max_additivity_error": float(g["max_additivity_error"].max()),
        })
    return pd.DataFrame(rows)

def leave_one_dataset_out(table):
    return pd.DataFrame([
        {
            "omitted_dataset": d,
            "n_datasets_remaining": int(len(table) - 1),
            "dataset_balanced_mean_excess_u2":
                float(table.loc[table["dataset"] != d, "mean_excess_u2"].mean()),
        }
        for d in table["dataset"]
    ])

def association(strict, n_boot=5000):
    x = strict["mean_abs_delta_auroc"].to_numpy(float)
    y = strict["mean_calibrated_excess_u2"].to_numpy(float)
    rho, rho_p = spearmanr(x, y)
    r, r_p = pearsonr(x, y)
    rng = np.random.default_rng(SEED)
    datasets = strict["dataset"].unique()
    boot = []
    for _ in range(n_boot):
        selected = rng.choice(datasets, size=len(datasets), replace=True)
        pieces = []
        for i, dataset in enumerate(selected):
            block = strict.loc[strict["dataset"] == dataset].copy()
            block["cluster_draw"] = i
            pieces.append(block)
        sample = pd.concat(pieces, ignore_index=True)
        if (sample["mean_abs_delta_auroc"].nunique() > 1
                and sample["mean_calibrated_excess_u2"].nunique() > 1):
            boot.append(float(spearmanr(
                sample["mean_abs_delta_auroc"],
                sample["mean_calibrated_excess_u2"]).statistic))
    return {
        "n_strict_contrasts": int(len(strict)),
        "n_dataset_clusters": int(len(datasets)),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "pearson_r": float(r),
        "pearson_p": float(r_p),
        "cluster_bootstrap_spearman_ci_low":
            float(np.quantile(boot, 0.025)),
        "cluster_bootstrap_spearman_ci_high":
            float(np.quantile(boot, 0.975)),
        "interpretation":
            "Exploratory and severely underpowered with four dataset clusters.",
    }

def make_figures(strict_splits, strict, datasets):
    order = strict.sort_values(["dataset", "family_id"]).reset_index(drop=True)
    labels = (order["dataset"] + ": " + order["family_id"]).tolist()
    y_positions = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    for y, (_, row) in zip(y_positions, order.iterrows()):
        points = strict_splits.loc[
            (strict_splits["dataset"] == row["dataset"])
            & (strict_splits["family_id"] == row["family_id"])
        ].sort_values("split_index")
        values = points["mean_calibrated_excess_u2"].to_numpy(float)
        ax.plot([values.min(), values.max()], [y, y], linewidth=1.0)
        ax.scatter(values, np.full_like(values, y, dtype=float), s=18, zorder=3)
        ax.scatter([row["mean_calibrated_excess_u2"]], [y],
                   marker="D", s=30, zorder=4)
    ax.axvline(0, linewidth=0.8)
    ax.set_yticks(y_positions, labels)
    ax.set_xlabel(
        "Calibrated normalized signed attribution-composition excess (U-squared)")
    ax.set_title("Five split means per strict deployment contrast")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG / "Figure_1_strict_contrasts.pdf", bbox_inches="tight")
    fig.savefig(FIG / "Figure_1_strict_contrasts.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(datasets["dataset"], datasets["mean_excess_u2"])
    ax.axhline(0, linewidth=0.8)
    ax.set_ylabel("Mean strict-contrast excess (U-squared)")
    ax.set_title("Dataset-level descriptive means")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG / "Figure_2_dataset_summary.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.4))
    for dataset, group in strict.groupby("dataset"):
        ax.scatter(group["mean_abs_delta_auroc"],
                   group["mean_calibrated_excess_u2"], label=dataset)
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("Mean absolute AUROC change")
    ax.set_ylabel("Mean calibrated attribution-composition excess")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "Figure_3_prediction_explanation.pdf",
                bbox_inches="tight")
    plt.close(fig)

def main():
    pair_rows = load_rows()
    pair_rows.to_csv(OUT / "pair_level_real_deployment.csv", index=False)

    task_model_splits = split_level_summary(
        pair_rows, ["dataset", "task_id", "family_id", "model", "split_index"])
    task_model_splits.to_csv(
        OUT / "task_model_split_level.csv", index=False)

    task_splits = split_level_summary(
        pair_rows, ["dataset", "task_id", "family_id", "split_index"])
    task_splits.to_csv(
        OUT / "task_definition_split_level.csv", index=False)
    task_summary = holm_split_tests(contrast_summary_from_splits(
        task_splits, ["dataset", "task_id", "family_id"], pair_rows=pair_rows))
    task_summary.to_csv(
        OUT / "task_definition_summary_27.csv", index=False)

    strict_splits = split_level_summary(
        pair_rows, ["dataset", "family_id", "split_index"])
    strict_splits.to_csv(
        OUT / "strict_contrast_split_level.csv", index=False)
    strict = holm_split_tests(contrast_summary_from_splits(
        strict_splits, ["dataset", "family_id"], pair_rows=pair_rows))
    strict.to_csv(
        OUT / "strict_contrast_summary_16.csv", index=False)

    model_splits = split_level_summary(
        pair_rows, ["dataset", "family_id", "model", "split_index"])
    model_summary = holm_split_tests(contrast_summary_from_splits(
        model_splits, ["dataset", "family_id", "model"], pair_rows=pair_rows))
    model_summary.to_csv(
        OUT / "model_family_split_summary.csv", index=False)

    datasets = dataset_summary(strict)
    datasets.to_csv(OUT / "dataset_summary_4.csv", index=False)
    loo = leave_one_dataset_out(datasets)
    loo.to_csv(OUT / "leave_one_dataset_out.csv", index=False)


    # Model-family agreement at strict contrast level.
    model_pivot = model_summary.pivot_table(
        index=["dataset", "family_id"], columns="model",
        values="mean_calibrated_excess_u2").reset_index()
    if {"logistic", "lightgbm"}.issubset(model_pivot.columns):
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        for dataset, group in model_pivot.groupby("dataset"):
            ax.scatter(group["logistic"], group["lightgbm"], label=dataset)
        lo = float(np.nanmin(model_pivot[["logistic", "lightgbm"]].to_numpy()))
        hi = float(np.nanmax(model_pivot[["logistic", "lightgbm"]].to_numpy()))
        ax.plot([lo, hi], [lo, hi], linewidth=0.8)
        ax.axhline(0, linewidth=0.6); ax.axvline(0, linewidth=0.6)
        ax.set_xlabel("Logistic regression excess (U-squared)")
        ax.set_ylabel("LightGBM excess (U-squared)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG / "Figure_4_model_family_agreement.pdf", bbox_inches="tight")
        fig.savefig(FIG / "Figure_4_model_family_agreement.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.4))
    for dataset, group in strict.groupby("dataset"):
        ax.scatter(group["mean_calibrated_excess_u2"],
                   group["mean_log_attribution_l1_ratio"], label=dataset)
    ax.axhline(0, linewidth=0.6); ax.axvline(0, linewidth=0.6)
    ax.set_xlabel("Normalized signed composition excess (U-squared)")
    ax.set_ylabel("Mean log attribution L1-norm ratio")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "Figure_S1_composition_amplitude.pdf", bbox_inches="tight")
    fig.savefig(FIG / "Figure_S1_composition_amplitude.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    assoc = association(strict)
    (OUT / "prediction_attribution_association.json").write_text(
        json.dumps(assoc, indent=2), encoding="utf-8")
    make_figures(strict_splits, strict, datasets)

    headline = {
        "version": "1.0.0",
        "n_datasets": int(datasets.shape[0]),
        "n_strict_contrasts": int(strict.shape[0]),
        "n_prespecified_task_definitions": int(task_summary.shape[0]),
        "n_release_defined_task_definitions": int(task_summary.shape[0]),
        "positive_strict_contrasts":
            int((strict["mean_calibrated_excess_u2"] > 0).sum()),
        "mean_strict_excess_u2":
            float(strict["mean_calibrated_excess_u2"].mean()),
        "median_strict_excess_u2":
            float(strict["mean_calibrated_excess_u2"].median()),
        "positive_all_five_splits":
            int((strict["positive_split_fraction"] == 1).sum()),
        "negative_all_five_splits":
            int((strict["positive_split_fraction"] == 0).sum()),
        "split_ranges_crossing_zero": int(
            ((strict["min_split_mean_u2"] <= 0)
             & (strict["max_split_mean_u2"] >= 0)).sum()),
        "mean_target_prevalence_weighted_class_excess_u2":
            float(strict[
                "mean_target_prevalence_weighted_class_excess_u2"].mean()),
        "dataset_balanced_mean_excess_u2":
            float(datasets["mean_excess_u2"].mean()),
        "leave_one_dataset_out_min":
            float(loo["dataset_balanced_mean_excess_u2"].min()),
        "leave_one_dataset_out_max":
            float(loo["dataset_balanced_mean_excess_u2"].max()),
        "mean_symmetrized_excess_u2":
            float(strict["mean_symmetrized_calibrated_excess_u2"].mean()),
        "max_additivity_error": float(strict["max_additivity_error"].max()),
        "association": assoc,
        "inference_note":
            "Formal tests use five split-level summaries and remain descriptive.",
    }
    (OUT / "headline_results.json").write_text(
        json.dumps(headline, indent=2), encoding="utf-8")
    print(json.dumps(headline, indent=2))

if __name__ == "__main__":
    main()
