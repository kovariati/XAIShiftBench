from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "outputs" / "cross_explainer_exploratory"
CONF = ROOT / "outputs" / "cross_explainer_confirmation"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
ANALYSIS_SEED = 2026073066
N_BOOT = 20_000
NONCLEAN = ["MCAR", "MAR", "MNAR", "BLOCK"]


def load_run(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenarios = pd.concat(
        [pd.read_csv(x) for x in sorted(path.glob("cross_explainer_scenarios_pairs_*.csv"))],
        ignore_index=True,
    )
    observations = pd.concat(
        [pd.read_csv(x) for x in sorted(path.glob("cross_explainer_observations_pairs_*.csv.gz"))],
        ignore_index=True,
    )
    features = pd.concat(
        [pd.read_csv(x) for x in sorted(path.glob("cross_explainer_features_pairs_*.csv.gz"))],
        ignore_index=True,
    )
    return scenarios, observations, features


def bootstrap_mean(values: np.ndarray, seed: int, n_boot: int = N_BOOT) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def holm_adjust(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p[idx]
        running = max(running, candidate)
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def exact_one_sided_wilcoxon(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(wilcoxon(values, alternative="greater", method="exact").pvalue)


def pair_estimands(s: pd.DataFrame, o: pd.DataFrame) -> pd.DataFrame:
    nonclean = s[s["target_mechanism"].isin(NONCLEAN)].copy()

    h1 = (
        nonclean.assign(excess=lambda d: d["mean_shift_stv"] - d["mean_kernel_algorithm_noise_stv"])
        .groupby("pair", as_index=True)["excess"]
        .mean()
        .rename("h1_shift_excess_over_kernel_noise")
    )

    mechanism = (
        nonclean.groupby(["pair", "model", "explainer", "target_mechanism"], as_index=False)[
            "mean_shift_stv"
        ]
        .mean()
        .pivot(index=["pair", "model", "explainer"], columns="target_mechanism", values="mean_shift_stv")
    )
    h2 = (mechanism["MNAR"] - mechanism["MCAR"]).groupby("pair").mean().rename("h2_mnar_minus_mcar")

    scenario_corr = (
        o[o["target_mechanism"].isin(NONCLEAN)]
        .groupby(["pair", "model", "target_mechanism"], as_index=False)[
            "scenario_shift_stv_method_spearman"
        ]
        .first()
    )
    h3 = scenario_corr.groupby("pair")["scenario_shift_stv_method_spearman"].median().rename(
        "h3_median_case_rank_agreement"
    )

    logistic = (
        nonclean[nonclean["model"] == "logistic"]
        .pivot_table(
            index=["pair", "target_mechanism"], columns="explainer", values="mean_shift_stv", aggfunc="mean"
        )
    )
    h4 = (
        logistic["kernel_shap"] - logistic["model_specific_shap"]
    ).groupby("pair").mean().rename("h4_logistic_kernel_minus_exact")

    lightgbm = (
        nonclean[nonclean["model"] == "lightgbm"]
        .pivot_table(
            index=["pair", "target_mechanism"], columns="explainer", values="mean_shift_stv", aggfunc="mean"
        )
    )
    secondary_lgbm = (
        lightgbm["kernel_shap"] - lightgbm["model_specific_shap"]
    ).groupby("pair").mean().rename("secondary_lightgbm_kernel_minus_tree")

    return pd.concat([h1, h2, h3, h4, secondary_lgbm], axis=1).reset_index()


def summarize_primary(pair_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specs = [
        ("H1", "h1_shift_excess_over_kernel_noise", "greater_than_zero"),
        ("H2", "h2_mnar_minus_mcar", "greater_than_zero"),
        ("H3", "h3_median_case_rank_agreement", "lower_ci_greater_than_0.75"),
        ("H4", "h4_logistic_kernel_minus_exact", "ci_inside_-0.01_0.01"),
    ]
    pvals: list[float] = []
    pkeys: list[str] = []
    for i, (hypothesis, col, criterion) in enumerate(specs):
        values = pair_df[col].to_numpy(float)
        mean, lo, hi = bootstrap_mean(values, ANALYSIS_SEED + i)
        row: dict[str, object] = {
            "hypothesis": hypothesis,
            "estimand": col,
            "mean": mean,
            "median": float(np.median(values)),
            "ci_low": lo,
            "ci_high": hi,
            "positive_pairs": int(np.sum(values > 0)),
            "n_pairs": int(len(values)),
            "criterion": criterion,
        }
        if hypothesis in {"H1", "H2"}:
            p = exact_one_sided_wilcoxon(values)
            row["p_raw"] = p
            pvals.append(p)
            pkeys.append(hypothesis)
        elif hypothesis == "H3":
            row["confirmed"] = bool(lo > 0.75)
        elif hypothesis == "H4":
            row["confirmed"] = bool(lo >= -0.01 and hi <= 0.01)
        rows.append(row)
    adjusted = holm_adjust(pvals)
    amap = dict(zip(pkeys, adjusted, strict=True))
    for row in rows:
        if row["hypothesis"] in amap:
            row["p_holm"] = amap[row["hypothesis"]]
            row["confirmed"] = bool(row["ci_low"] > 0 and amap[row["hypothesis"]] < 0.05)
    return pd.DataFrame(rows)


def aggregate_tables(s: pd.DataFrame, o: pd.DataFrame) -> dict[str, pd.DataFrame]:
    nonclean = s[s["target_mechanism"].isin(NONCLEAN)].copy()
    by_condition = (
        nonclean.groupby(["model", "explainer", "target_mechanism"], as_index=False)
        .agg(
            mean_shift_stv=("mean_shift_stv", "mean"),
            sd_shift_stv=("mean_shift_stv", "std"),
            mean_kernel_noise=("mean_kernel_algorithm_noise_stv", "mean"),
            mean_global_rank_tau=("global_abs_rank_tau", "mean"),
            mean_global_top5=("global_abs_top5_jaccard", "mean"),
            mean_signed_consistency=("global_signed_consistency", "mean"),
            max_source_additivity_error=("source_additivity_error", "max"),
            max_target_additivity_error=("target_additivity_error", "max"),
        )
    )
    disagreement = (
        o[o["target_mechanism"].isin(NONCLEAN)]
        .groupby(["model", "target_mechanism"], as_index=False)
        .agg(
            mean_source_cross_explainer_stv=("source_cross_explainer_stv", "mean"),
            mean_target_cross_explainer_stv=("target_cross_explainer_stv", "mean"),
            mean_target_minus_source=("target_minus_source_cross_explainer_stv", "mean"),
            mean_kernel_algorithm_noise=("kernel_algorithm_noise_stv", "mean"),
            mean_shift_rank_spearman=("scenario_shift_stv_method_spearman", "mean"),
        )
    )
    faithfulness = (
        s.groupby(["model", "explainer"], as_index=False)
        .agg(
            signed_replacement_spearman=("source_signed_replacement_faithfulness", "mean"),
            absolute_replacement_spearman=("source_absolute_replacement_faithfulness", "mean"),
        )
    )
    return {"by_condition": by_condition, "disagreement": disagreement, "faithfulness": faithfulness}


def comparison(exp_pair: pd.DataFrame, conf_pair: pd.DataFrame) -> pd.DataFrame:
    columns = [c for c in exp_pair.columns if c != "pair"]
    rows = []
    for col in columns:
        # Compare pair-level patterns by matching the first five pair indices. This is a reproducibility
        # summary, not a paired inferential test because the seed families are independent.
        exp = exp_pair[col].to_numpy(float)
        conf = conf_pair[col].iloc[: len(exp)].to_numpy(float)
        rows.append(
            {
                "estimand": col,
                "exploratory_mean": float(exp.mean()),
                "confirmatory_mean": float(conf_pair[col].mean()),
                "absolute_mean_difference": float(abs(exp.mean() - conf_pair[col].mean())),
                "first_five_pattern_correlation": float(np.corrcoef(exp, conf)[0, 1])
                if np.std(exp) > 0 and np.std(conf) > 0
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def create_figures(s: pd.DataFrame, o: pd.DataFrame, pair_df: pd.DataFrame) -> None:
    nonclean = s[s["target_mechanism"].isin(NONCLEAN)]
    order = ["BLOCK", "MAR", "MCAR", "MNAR"]

    table = nonclean.groupby(["target_mechanism", "explainer"])["mean_shift_stv"].mean().unstack("explainer").loc[order]
    ax = table.plot(kind="bar", figsize=(8.2, 5.2))
    ax.set_xlabel("Deployment missingness mechanism")
    ax.set_ylabel("Mean paired explanation STV")
    ax.set_title("Explanation shift reproduced across two explainer families")
    ax.legend(title="Explainer")
    plt.tight_layout()
    plt.savefig(FIG / "cross_explainer_shift_by_mechanism_confirmation.png", dpi=220)
    plt.close()

    noise = (
        nonclean.groupby(["pair", "model", "explainer"], as_index=False)
        .agg(shift=("mean_shift_stv", "mean"), noise=("mean_kernel_algorithm_noise_stv", "mean"))
    )
    plt.figure(figsize=(7.2, 5.4))
    for (model, explainer), group in noise.groupby(["model", "explainer"]):
        plt.scatter(group["noise"], group["shift"], label=f"{model}, {explainer}", alpha=0.8)
    lo = min(noise["noise"].min(), noise["shift"].min())
    hi = max(noise["noise"].max(), noise["shift"].max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("KernelSHAP algorithmic-noise STV")
    plt.ylabel("Mean non-clean deployment-shift STV")
    plt.title("Deployment shift versus explainer algorithmic variability")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "cross_explainer_shift_vs_algorithm_noise_confirmation.png", dpi=220)
    plt.close()

    scenario = (
        o[o["target_mechanism"].isin(NONCLEAN)]
        .groupby(["pair", "model", "target_mechanism"], as_index=False)
        .agg(specific=("specific_shift_stv", "mean"), kernel=("kernel_shift_stv", "mean"))
    )
    plt.figure(figsize=(6.8, 5.5))
    for model, group in scenario.groupby("model"):
        plt.scatter(group["specific"], group["kernel"], label=model, alpha=0.75)
    lo = min(scenario["specific"].min(), scenario["kernel"].min())
    hi = max(scenario["specific"].max(), scenario["kernel"].max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Model-specific SHAP shift")
    plt.ylabel("KernelSHAP shift")
    plt.title("Scenario-level cross-explainer agreement")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "cross_explainer_scenario_agreement_confirmation.png", dpi=220)
    plt.close()

    cols = [
        "h1_shift_excess_over_kernel_noise",
        "h2_mnar_minus_mcar",
        "h4_logistic_kernel_minus_exact",
        "secondary_lightgbm_kernel_minus_tree",
    ]
    long = pair_df.melt(id_vars="pair", value_vars=cols, var_name="estimand", value_name="value")
    means = long.groupby("estimand")["value"].mean().reindex(cols)
    errors = []
    for i, col in enumerate(cols):
        _, lo, hi = bootstrap_mean(pair_df[col].to_numpy(float), ANALYSIS_SEED + 50 + i)
        errors.append([means[col] - lo, hi - means[col]])
    err = np.asarray(errors).T
    plt.figure(figsize=(9.0, 5.2))
    x = np.arange(len(cols))
    plt.errorbar(x, means.values, yerr=err, fmt="o", capsize=5)
    plt.axhline(0, linewidth=1)
    plt.xticks(x, ["Shift minus\nalgorithm noise", "MNAR minus\nMCAR", "Logistic Kernel\nminus exact", "LightGBM Kernel\nminus TreeSHAP"])
    plt.ylabel("Pair-level mean effect")
    plt.title("Confirmatory and secondary cross-explainer estimands")
    plt.tight_layout()
    plt.savefig(FIG / "cross_explainer_pair_estimands_confirmation.png", dpi=220)
    plt.close()


def main() -> None:
    s, o, f = load_run(CONF)
    exp_s, exp_o, _ = load_run(EXP)
    expected = {"scenarios": 400, "observations": 8000, "features": 8000, "pairs": 20}
    observed = {"scenarios": len(s), "observations": len(o), "features": len(f), "pairs": int(s["pair"].nunique())}
    if observed != expected:
        raise RuntimeError(f"Incomplete confirmatory run: observed={observed}, expected={expected}")
    if not set(s["pair"].unique()) == set(range(20)):
        raise RuntimeError("Pair indices are incomplete")
    numeric_primary = ["mean_shift_stv", "mean_kernel_algorithm_noise_stv", "global_abs_rank_tau"]
    if not np.isfinite(s[numeric_primary].to_numpy(float)).all():
        raise RuntimeError("Non-finite scenario values")
    max_additivity = float(
        max(s["source_additivity_error"].max(), s["target_additivity_error"].max())
    )
    if max_additivity > 1e-5:
        raise RuntimeError(f"Additivity failure: {max_additivity}")

    pairs = pair_estimands(s, o)
    primary = summarize_primary(pairs)
    aggregates = aggregate_tables(s, o)
    exp_pairs = pair_estimands(exp_s, exp_o)
    reproducibility = comparison(exp_pairs, pairs)

    s.to_csv(CONF / "cross_explainer_scenarios_combined.csv", index=False)
    o.to_csv(CONF / "cross_explainer_observations_combined.csv.gz", index=False, compression="gzip")
    f.to_csv(CONF / "cross_explainer_features_combined.csv.gz", index=False, compression="gzip")
    pairs.to_csv(CONF / "cross_explainer_pair_estimands.csv", index=False)
    primary.to_csv(CONF / "cross_explainer_primary_confirmation.csv", index=False)
    aggregates["by_condition"].to_csv(CONF / "cross_explainer_condition_summary.csv", index=False)
    aggregates["disagreement"].to_csv(CONF / "cross_explainer_disagreement_summary.csv", index=False)
    aggregates["faithfulness"].to_csv(CONF / "cross_explainer_faithfulness_summary.csv", index=False)
    reproducibility.to_csv(CONF / "cross_explainer_exploratory_confirmation_comparison.csv", index=False)

    create_figures(s, o, pairs)

    metadata_files = sorted(CONF.glob("cross_explainer_metadata_pairs_*.json"))
    total_runtime = sum(json.loads(x.read_text())["runtime_seconds"] for x in metadata_files)
    secondary_lgbm = bootstrap_mean(pairs["secondary_lightgbm_kernel_minus_tree"].to_numpy(float), ANALYSIS_SEED + 90)
    source_dis = float(o["source_cross_explainer_stv"].mean())
    target_dis = float(o[o["target_mechanism"].isin(NONCLEAN)]["target_cross_explainer_stv"].mean())
    kernel_noise_by_model = (
        o.groupby("model")["kernel_algorithm_noise_stv"].mean().to_dict()
    )
    summary = {
        "seed_base": 2026073016,
        "observed_counts": observed,
        "n_model_fits": 40,
        "n_explanation_runs": 440,
        "total_recorded_runtime_seconds": total_runtime,
        "max_additivity_error": max_additivity,
        "all_primary_confirmed": bool(primary["confirmed"].all()),
        "primary": primary.replace({np.nan: None}).to_dict(orient="records"),
        "secondary_lightgbm_kernel_minus_tree": {
            "mean": secondary_lgbm[0], "ci_low": secondary_lgbm[1], "ci_high": secondary_lgbm[2]
        },
        "mean_source_cross_explainer_stv": source_dis,
        "mean_target_cross_explainer_stv_nonclean": target_dis,
        "kernel_algorithm_noise_by_model": {k: float(v) for k, v in kernel_noise_by_model.items()},
        "mean_nonclean_shift_by_explainer": {
            k: float(v) for k, v in s[s["target_mechanism"].isin(NONCLEAN)].groupby("explainer")["mean_shift_stv"].mean().to_dict().items()
        },
    }
    (CONF / "cross_explainer_confirmation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print("\nPRIMARY\n", primary.to_string(index=False))
    print("\nBY CONDITION\n", aggregates["by_condition"].round(6).to_string(index=False))
    print("\nFAITHFULNESS\n", aggregates["faithfulness"].round(6).to_string(index=False))


if __name__ == "__main__":
    main()
