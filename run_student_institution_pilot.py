from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import pandas as pd

from xaishiftbench.student_institution_pilot import run_student_institution_pilot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026073004)
    args = parser.parse_args()
    data = args.root / "data" / "student_performance" / "raw" / "data.csv"
    out = args.root / "outputs" / "student"
    pair_dir = out / "pair_runs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    for pair in range(args.pairs):
        rows_path = pair_dir / f"pair_{pair:02d}_rows.csv"
        features_path = pair_dir / f"pair_{pair:02d}_features.csv"
        if rows_path.exists() and features_path.exists():
            print(f"pair {pair}: existing", flush=True)
            continue
        rows, features, metadata = run_student_institution_pilot(
            data, seed_base=args.seed, n_pairs=1, pair_start=pair, total_pairs=args.pairs, n_splits=5
        )
        if rows.empty or features.empty:
            raise AssertionError("Student pair output must not be empty.")
        rows.to_csv(rows_path, index=False)
        features.to_csv(features_path, index=False)
        (pair_dir / f"pair_{pair:02d}_metadata.json").write_text(
            json.dumps(asdict(metadata), indent=2), encoding="utf-8"
        )
        print(f"pair {pair}: complete", flush=True)
        del rows, features
        gc.collect()

    row_files = sorted(pair_dir.glob("pair_*_rows.csv"))
    feature_files = sorted(pair_dir.glob("pair_*_features.csv"))
    if len(row_files) != args.pairs or len(feature_files) != args.pairs:
        raise AssertionError("Incomplete student split/refit grid.")
    rows = pd.concat([pd.read_csv(path) for path in row_files], ignore_index=True)
    features = pd.concat([pd.read_csv(path) for path in feature_files], ignore_index=True)
    rows = rows.sort_values(["pair", "direction", "representation", "model"])
    features = features.sort_values(["pair", "direction", "representation", "model", "feature"])
    rows.to_csv(out / "student_institution_refits.csv", index=False)
    features.to_csv(out / "student_institution_feature_profiles.csv", index=False)
    payload = {
        "version": "1.0.0",
        "runtime_seconds_current_invocation": perf_counter() - start,
        "n_pairs": args.pairs,
        "n_splits": 5,
        "refit_pairs_per_split": args.pairs // 5,
        "n_scenario_evaluations": int(len(rows)),
        "resumable_pair_runs": True,
        "primary_statistic": "off-diagonal energy U-statistic squared",
    }
    (out / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
