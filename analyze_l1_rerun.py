from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _load_domain(root: Path, domain: str, rel: str) -> pd.DataFrame:
    path = root / rel
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if domain == "acs":
        frame = frame.loc[frame["target_definition"] == "nominal_50k"].copy()
        frame["task_id"] = "ACSIncome_2018_to_2024"
    elif domain == "heart":
        frame["task_id"] = frame["target_site"].astype(str)
    elif domain == "student":
        frame["task_id"] = frame["direction"].astype(str) + "|" + frame["representation"].astype(str)
    elif domain == "oulad":
        frame["task_id"] = (
            frame["code_module"].astype(str) + "-" + frame["period"].astype(str)
            + "|day" + frame["horizon_day"].astype(int).astype(str)
        )
    frame.insert(0, "domain", domain)
    return frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path)
    ns = ap.parse_args()
    root = ns.root.resolve()
    out = (ns.output_dir or root / "outputs" / "l1_revision").resolve()
    out.mkdir(parents=True, exist_ok=True)

    specs = {
        "heart": "outputs/heart/heart_cross_site_refits.csv",
        "student": "outputs/student/student_institution_refits.csv",
        "oulad": "outputs/oulad/oulad_temporal_refits.csv",
        "acs": "outputs/acs/acs_temporal_refits.csv",
    }
    frames = [_load_domain(root, domain, rel) for domain, rel in specs.items()]
    combined = pd.concat(frames, ignore_index=True, sort=False)

    id_candidates = [
        "domain", "task_id", "model", "pair", "refit_pair", "split_index", "refit_index",
        "target_site", "direction", "representation", "code_module", "period", "horizon_day",
        "target_definition", "n_explanation", "calibrated_excess_u2",
    ]
    audit_cols = [
        c for c in combined.columns
        if c.startswith(("source_l1_", "null_l1_", "target_l1_", "l1thr_", "l1stratum_"))
    ]
    if not audit_cols:
        raise AssertionError("No the release L1 audit columns were found. Run the patched domain pipelines in a fresh folder.")
    run_level = combined[[c for c in id_candidates if c in combined.columns] + audit_cols].copy()
    run_level.to_csv(out / "l1_run_level_diagnostics.csv", index=False)

    numeric = [c for c in ["calibrated_excess_u2"] + audit_cols if c in run_level and pd.api.types.is_numeric_dtype(run_level[c])]
    task = (
        run_level.groupby(["domain", "task_id", "model"], dropna=False)[numeric]
        .median(numeric_only=True)
        .reset_index()
    )
    counts = run_level.groupby(["domain", "task_id", "model"], dropna=False).size().rename("n_run_rows").reset_index()
    task = counts.merge(task, on=["domain", "task_id", "model"], how="left")
    task.to_csv(out / "l1_task_median_summary.csv", index=False)

    thr_pat = re.compile(r"^(l1thr_.+)_excess_u2$")
    threshold_rows = []
    for col in audit_cols:
        m = thr_pat.match(col)
        if not m:
            continue
        stem = m.group(1)
        for _, row in task.iterrows():
            primary = row.get("calibrated_excess_u2", np.nan)
            value = row.get(col, np.nan)
            threshold_rows.append({
                "domain": row["domain"], "task_id": row["task_id"], "model": row["model"],
                "threshold_key": stem.removeprefix("l1thr_"),
                "primary_excess_u2_median": primary,
                "threshold_excess_u2_median": value,
                "sign_agreement": bool(np.isfinite(primary) and np.isfinite(value) and np.sign(primary) == np.sign(value)),
                "source_retained_fraction_median": row.get(f"{stem}_source_fraction", np.nan),
                "null_retained_fraction_median": row.get(f"{stem}_null_fraction", np.nan),
                "target_retained_fraction_median": row.get(f"{stem}_target_fraction", np.nan),
            })
    threshold = pd.DataFrame(threshold_rows)
    threshold.to_csv(out / "l1_threshold_sensitivity_task_summary.csv", index=False)

    strata_rows = []
    for q in range(1, 5):
        stem = f"l1stratum_q{q}"
        col = f"{stem}_excess_u2"
        if col not in task:
            continue
        for _, row in task.iterrows():
            strata_rows.append({
                "domain": row["domain"], "task_id": row["task_id"], "model": row["model"],
                "amplitude_stratum": q,
                "lower_median": row.get(f"{stem}_lower", np.nan),
                "upper_median": row.get(f"{stem}_upper", np.nan),
                "excess_u2_median": row.get(col, np.nan),
                "source_n_median": row.get(f"{stem}_source_n", np.nan),
                "null_n_median": row.get(f"{stem}_null_n", np.nan),
                "target_n_median": row.get(f"{stem}_target_n", np.nan),
            })
    strata = pd.DataFrame(strata_rows)
    strata.to_csv(out / "l1_amplitude_strata_task_summary.csv", index=False)

    near_cols = [c for c in task.columns if "fraction_l1_le_" in c or c.endswith("zero_row_fraction")]
    overview = {
        "version": "1.0.0",
        "domains": sorted(task["domain"].unique().tolist()),
        "n_task_model_cells": int(len(task)),
        "n_run_rows": int(len(run_level)),
        "audit_column_count": int(len(audit_cols)),
        "max_near_zero_or_zero_fraction_by_column": {
            c: float(pd.to_numeric(task[c], errors="coerce").max()) for c in near_cols
        },
    }
    if not threshold.empty:
        overview["threshold_sign_agreement_by_threshold"] = {
            str(k): float(g["sign_agreement"].mean())
            for k, g in threshold.groupby("threshold_key", sort=True)
        }
    (out / "l1_rerun_overview.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")

    lines = [
        "# XAIShiftBench the release local L1 rerun summary",
        "",
        f"- Run-level rows: {len(run_level)}",
        f"- Task × model cells: {len(task)}",
        f"- L1 diagnostic/sensitivity columns: {len(audit_cols)}",
        "",
        "The tables retain the release-defined primary endpoint and add threshold-conditioned and amplitude-stratified sensitivity views. Threshold-conditioned samples can have different retained sizes and are therefore reported with explicit retained counts and fractions.",
    ]
    if not threshold.empty:
        lines += ["", "## Threshold sign agreement"]
        for k, g in threshold.groupby("threshold_key", sort=True):
            lines.append(f"- {k}: {g['sign_agreement'].mean():.3f}")
    (out / "L1_RERUN_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(overview, indent=2))


if __name__ == "__main__":
    main()
