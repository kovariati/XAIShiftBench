from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "credit_missingness_confirmation"
PILOT = ROOT / "outputs" / "credit_missingness_pilot"
FIG = ROOT / "figures"


def holm(values: list[float]) -> list[float]:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def one_sided(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if np.allclose(values, 0):
        return 0.0, 1.0
    result = wilcoxon(values, alternative="greater", method="auto")
    return float(result.statistic), float(result.pvalue)


def primary_estimands(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows = []
    for pair, pair_df in df.groupby("pair"):
        slopes = []
        for _, group in pair_df[pair_df["mechanism"] != "CLEAN"].groupby(
            ["model", "indicator_mode", "mechanism"]
        ):
            group = group.sort_values("rate")
            slopes.append(float(np.polyfit(group["rate"], group["mean_shift_stv"], 1)[0]))

        nonclean = pair_df[pair_df["mechanism"] != "CLEAN"]
        mcar = nonclean[nonclean["mechanism"] == "MCAR"][
            ["model", "indicator_mode", "rate", "mean_shift_stv"]
        ].rename(columns={"mean_shift_stv": "mcar_stv"})
        mnar = nonclean[nonclean["mechanism"] == "MNAR"].merge(
            mcar, on=["model", "indicator_mode", "rate"]
        )
        mnar_effect = float((mnar["mean_shift_stv"] - mnar["mcar_stv"]).mean())

        wide = pair_df.pivot(
            index=["model", "scenario"],
            columns="indicator_mode",
            values="mean_shift_stv",
        )
        indicator_effect = float((wide["all"] - wide["none"]).mean())

        high = pair_df[(pair_df["rate"] == 0.30) & (pair_df["mechanism"] != "CLEAN")]
        high_excess = float(
            (high["mean_shift_stv"] - high["mean_mask_control_stv"]).mean()
        )
        pair_rows.append(
            {
                "pair": int(pair),
                "dose_response_mean_slope": float(np.mean(slopes)),
                "mnar_minus_equal_rate_mcar": mnar_effect,
                "indicator_all_minus_none": indicator_effect,
                "high_missingness_excess_over_mask_control": high_excess,
            }
        )
    pairs = pd.DataFrame(pair_rows)
    definitions = [
        ("H1 dose response", "dose_response_mean_slope"),
        ("H2 MNAR exceeds equal-rate MCAR", "mnar_minus_equal_rate_mcar"),
        ("H3 indicators increase explanation distance", "indicator_all_minus_none"),
        ("H4 30% missingness exceeds mask-control variation", "high_missingness_excess_over_mask_control"),
    ]
    tests = []
    for hypothesis, column in definitions:
        values = pairs[column].to_numpy()
        stat, p = one_sided(values)
        tests.append(
            {
                "hypothesis": hypothesis,
                "estimand": column,
                "n_pairs": len(values),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "positive_pairs": int(np.sum(values > 0)),
                "wilcoxon_stat": stat,
                "p_one_sided": p,
            }
        )
    tests_df = pd.DataFrame(tests)
    tests_df["p_holm_four_primary"] = holm(tests_df["p_one_sided"].tolist())
    return pairs, tests_df


def secondary_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in df.groupby(["model", "indicator_mode", "scenario", "mechanism", "rate"]):
        excess_control = group["mean_shift_stv"] - group["mean_mask_control_stv"]
        excess_refit = group["mean_shift_stv"] - group["mean_null_stv"]
        rows.append(
            {
                "model": key[0],
                "indicator_mode": key[1],
                "scenario": key[2],
                "mechanism": key[3],
                "rate": key[4],
                "mean_shift_stv": float(group["mean_shift_stv"].mean()),
                "mean_mask_control_stv": float(group["mean_mask_control_stv"].mean()),
                "mean_refit_null_stv": float(group["mean_null_stv"].mean()),
                "mean_excess_over_mask_control": float(excess_control.mean()),
                "pairs_above_mask_control": int((excess_control > 0).sum()),
                "mean_excess_over_refit": float(excess_refit.mean()),
                "pairs_above_refit": int((excess_refit > 0).sum()),
                "mean_delta_auroc": float(group["delta_auroc"].mean()),
                "median_delta_auroc": float(group["delta_auroc"].median()),
                "mean_delta_log_loss": float(group["delta_log_loss"].mean()),
                "mean_probability_shift": float(group["mean_abs_probability_shift"].mean()),
                "mean_shift_js": float(group["shift_js"].mean()),
                "mean_shift_rank_tau": float(group["shift_rank_tau"].mean()),
                "mean_target_indicator_share": float(group["target_indicator_abs_share"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "indicator_mode", "mechanism", "rate"])


def mechanism_summary(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[df["mechanism"] != "CLEAN"]
    mcar = subset[subset["mechanism"] == "MCAR"][
        ["pair", "model", "indicator_mode", "rate", "mean_shift_stv", "delta_auroc"]
    ].rename(columns={"mean_shift_stv": "mcar_stv", "delta_auroc": "mcar_delta_auc"})
    merged = subset.merge(mcar, on=["pair", "model", "indicator_mode", "rate"])
    merged = merged[merged["mechanism"] != "MCAR"].copy()
    merged["delta_stv_vs_mcar"] = merged["mean_shift_stv"] - merged["mcar_stv"]
    merged["delta_auc_vs_mcar"] = merged["delta_auroc"] - merged["mcar_delta_auc"]
    rows = []
    for key, group in merged.groupby(["model", "indicator_mode", "mechanism", "rate"]):
        rows.append(
            {
                "model": key[0],
                "indicator_mode": key[1],
                "mechanism": key[2],
                "rate": key[3],
                "mean_delta_stv_vs_mcar": float(group["delta_stv_vs_mcar"].mean()),
                "median_delta_stv_vs_mcar": float(group["delta_stv_vs_mcar"].median()),
                "pairs_above_mcar": int((group["delta_stv_vs_mcar"] > 0).sum()),
                "mean_delta_auc_vs_mcar": float(group["delta_auc_vs_mcar"].mean()),
            }
        )
    merged.to_csv(OUT / "confirmation_mechanism_pair_contrasts.csv", index=False)
    return pd.DataFrame(rows).sort_values(["model", "indicator_mode", "mechanism", "rate"])


def pilot_confirmation_comparison(confirmation: pd.DataFrame) -> pd.DataFrame:
    pilot = pd.read_csv(PILOT / "credit_missingness_refits.csv")
    keys = ["model", "indicator_mode", "scenario", "mechanism", "rate"]
    p = pilot.groupby(keys).agg(
        pilot_shift=("mean_shift_stv", "mean"),
        pilot_delta_auc=("delta_auroc", "mean"),
    ).reset_index()
    c = confirmation.groupby(keys).agg(
        confirmation_shift=("mean_shift_stv", "mean"),
        confirmation_delta_auc=("delta_auroc", "mean"),
    ).reset_index()
    out = p.merge(c, on=keys)
    out["shift_difference_confirmation_minus_pilot"] = (
        out["confirmation_shift"] - out["pilot_shift"]
    )
    out["delta_auc_difference_confirmation_minus_pilot"] = (
        out["confirmation_delta_auc"] - out["pilot_delta_auc"]
    )
    return out


def figures(pairs: pd.DataFrame, secondary: pd.DataFrame, mechanisms: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    long = pairs.melt(id_vars="pair", var_name="estimand", value_name="value")
    categories = list(long["estimand"].unique())
    for index, category in enumerate(categories):
        values = long.loc[long["estimand"] == category, "value"].to_numpy()
        jitter = np.linspace(-0.12, 0.12, len(values))
        ax.scatter(np.full(len(values), index) + jitter, values, s=28)
        ax.hlines(np.median(values), index - 0.25, index + 0.25, linewidth=2)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([
        "Dose-response\nslope",
        "MNAR minus\nMCAR",
        "Indicators\nminus none",
        "30% shift minus\nmask control",
    ])
    ax.set_ylabel("Pair-level primary estimand")
    ax.set_title("Independent 20-pair confirmation")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "credit_confirmation_primary_estimands.png", dpi=220)
    plt.close(fig)

    high = secondary[(secondary["rate"] == 0.30) & (secondary["mechanism"] != "CLEAN")].copy()
    high["label"] = high["model"] + ", ind=" + high["indicator_mode"] + ", " + high["mechanism"]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(high["mean_delta_auroc"], high["mean_excess_over_mask_control"], s=60)
    for row in high.itertuples():
        ax.annotate(row.label, (row.mean_delta_auroc, row.mean_excess_over_mask_control), fontsize=7)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Mean AUROC change")
    ax.set_ylabel("Explanation shift minus MCAR15 mask-control")
    ax.set_title("High missingness: prediction and explanation changes")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "credit_confirmation_high_missingness_decoupling.png", dpi=220)
    plt.close(fig)

    mn = mechanisms[mechanisms["mechanism"] == "MNAR"].copy()
    mn["label"] = (
        mn["model"] + ", ind=" + mn["indicator_mode"] + ", " + (mn["rate"] * 100).astype(int).astype(str) + "%"
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(mn))
    ax.scatter(mn["mean_delta_stv_vs_mcar"], positions)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(positions)
    ax.set_yticklabels(mn["label"], fontsize=8)
    ax.set_xlabel("MNAR explanation distance minus equal-rate MCAR")
    ax.set_title("MNAR mechanism effect replicated across models and pipelines")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "credit_confirmation_mnar_vs_mcar.png", dpi=220)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(OUT / "credit_missingness_confirmation_refits.csv")
    if df.shape != (1040, 56) or df["pair"].nunique() != 20:
        raise AssertionError("Confirmation result matrix is incomplete.")
    pairs, tests = primary_estimands(df)
    secondary = secondary_summary(df)
    mechanisms = mechanism_summary(df)
    comparison = pilot_confirmation_comparison(df)
    pairs.to_csv(OUT / "confirmation_primary_pair_estimands.csv", index=False)
    tests.to_csv(OUT / "confirmation_primary_tests.csv", index=False)
    secondary.to_csv(OUT / "confirmation_secondary_cell_summary.csv", index=False)
    mechanisms.to_csv(OUT / "confirmation_mechanism_summary.csv", index=False)
    comparison.to_csv(OUT / "pilot_confirmation_comparison.csv", index=False)
    figures(pairs, secondary, mechanisms)

    findings = {
        "rows": int(len(df)),
        "pairs": int(df["pair"].nunique()),
        "max_additivity_error": float(df["max_additivity_error"].max()),
        "all_four_primary_holm_below_0_05": bool((tests["p_holm_four_primary"] < 0.05).all()),
        "primary_tests": tests.to_dict(orient="records"),
        "mnar_cells_positive_vs_mcar": int(
            (mechanisms[mechanisms["mechanism"] == "MNAR"]["mean_delta_stv_vs_mcar"] > 0).sum()
        ),
        "mnar_cells_total": int((mechanisms["mechanism"] == "MNAR").sum()),
        "pilot_confirmation_shift_correlation": float(
            comparison[["pilot_shift", "confirmation_shift"]].corr().iloc[0, 1]
        ),
        "mean_absolute_pilot_confirmation_shift_difference": float(
            comparison["shift_difference_confirmation_minus_pilot"].abs().mean()
        ),
        "mean_refit_null": float(df["mean_null_stv"].mean()),
        "mean_mask_control": float(df["mean_mask_control_stv"].mean()),
    }
    (OUT / "confirmation_key_findings.json").write_text(
        json.dumps(findings, indent=2), encoding="utf-8"
    )
    print(tests.to_string(index=False))
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
