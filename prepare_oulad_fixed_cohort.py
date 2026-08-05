from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

KEY = ["code_module", "code_presentation", "id_student"]
HORIZONS = (14, 56)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-table", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.model_table)
    df = df[df["horizon_day"].isin(HORIZONS)].copy()

    keep_parts = []
    audit_rows = []
    for (module, presentation), g in df.groupby(["code_module", "code_presentation"], observed=True):
        ids14 = set(g.loc[g.horizon_day.eq(14), "id_student"].astype(int))
        ids56 = set(g.loc[g.horizon_day.eq(56), "id_student"].astype(int))
        fixed = ids14 & ids56
        sub = g[g["id_student"].isin(fixed)].copy()
        if len(fixed):
            c14 = set(sub.loc[sub.horizon_day.eq(14), "id_student"].astype(int))
            c56 = set(sub.loc[sub.horizon_day.eq(56), "id_student"].astype(int))
            if c14 != c56:
                raise AssertionError(f"Fixed-cohort mismatch for {module}-{presentation}")
        keep_parts.append(sub)
        for h in HORIZONS:
            gh = sub[sub.horizon_day.eq(h)]
            audit_rows.append({
                "code_module": module,
                "code_presentation": presentation,
                "horizon_day": h,
                "dynamic_n": int(len(g[g.horizon_day.eq(h)])),
                "fixed_n": int(len(gh)),
                "fixed_fraction_of_dynamic": float(len(gh) / max(len(g[g.horizon_day.eq(h)]), 1)),
                "fixed_prevalence_unsuccessful": float(gh["target_unsuccessful"].mean()) if len(gh) else float("nan"),
                "fixed_cohort_definition": "intersection of day-14 and day-56 risk sets within module-presentation",
            })

    fixed_df = pd.concat(keep_parts, ignore_index=True)
    fixed_df.to_csv(out / "oulad_fixed_cohort_h14_h56.csv.gz", index=False, compression="gzip")
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(out / "oulad_fixed_cohort_audit.csv", index=False)

    summary = {
        "input_rows_h14_h56": int(len(df)),
        "fixed_rows_h14_h56": int(len(fixed_df)),
        "n_module_presentations": int(audit[["code_module", "code_presentation"]].drop_duplicates().shape[0]),
        "horizons": list(HORIZONS),
        "definition": "same student IDs retained at both day 14 and day 56 within each module-presentation; this isolates feature accumulation from dynamic risk-set membership among retained students",
    }
    (out / "oulad_fixed_cohort_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
