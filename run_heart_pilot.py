from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from xaishiftbench.heart_pilot import run_heart_cross_site_pilot, summarize_heart_pilot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--explain-cap", type=int, default=160)
    args = parser.parse_args()
    pair_dir = args.output_dir / "pair_runs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    for pair in range(args.pairs):
        rows_path = pair_dir / f"pair_{pair:02d}_rows.csv"
        features_path = pair_dir / f"pair_{pair:02d}_features.csv"
        if rows_path.exists() and features_path.exists():
            print(f"pair {pair}: existing", flush=True)
            continue
        rows, _, features = run_heart_cross_site_pilot(
            args.data_dir,
            seed_base=args.seed,
            n_pairs=1,
            pair_start=pair,
            total_pairs=args.pairs,
            n_splits=args.splits,
            explain_cap=args.explain_cap,
        )
        rows.to_csv(rows_path, index=False)
        features.to_csv(features_path, index=False)
        print(f"pair {pair}: complete", flush=True)
        del rows, features
        gc.collect()
    row_files = sorted(pair_dir.glob("pair_*_rows.csv"))
    feature_files = sorted(pair_dir.glob("pair_*_features.csv"))
    if len(row_files) != args.pairs or len(feature_files) != args.pairs:
        raise AssertionError("Incomplete Heart split/refit grid.")
    refits = pd.concat([pd.read_csv(path) for path in row_files], ignore_index=True)
    importance = pd.concat([pd.read_csv(path) for path in feature_files], ignore_index=True)
    summary = summarize_heart_pilot(refits)
    refits.to_csv(args.output_dir / "heart_cross_site_refits.csv", index=False)
    summary.to_csv(args.output_dir / "heart_cross_site_summary.csv", index=False)
    importance.to_csv(args.output_dir / "heart_feature_importance.csv", index=False)
    metadata = {
        "version": "1.0.0", "runtime_seconds_current_invocation": perf_counter()-start,
        "seed_base": args.seed, "repeated_source_splits": args.splits,
        "refit_pairs_per_split": args.pairs // args.splits, "total_split_refit_pairs": args.pairs,
        "explanation_cap": args.explain_cap,
        "primary_statistic": "off-diagonal energy U-statistic squared",
        "models": ["logistic", "lightgbm"], "number_of_scenario_rows": int(len(refits)),
        "number_of_total_model_fits": int(2 * len(refits)), "resumable_pair_runs": True,
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
