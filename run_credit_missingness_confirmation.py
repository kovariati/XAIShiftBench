from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from xaishiftbench.credit_missingness_pilot import run_credit_missingness_pilot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026073001)
    args = parser.parse_args()
    out = args.root / "outputs" / "credit_missingness_confirmation"
    pair_dir = out / "pair_runs"
    data = args.root / "data" / "south_german_credit" / "raw" / "SouthGermanCredit.asc"
    if not data.exists():
        raise FileNotFoundError(data)
    out.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    for pair in range(args.pairs):
        refit_path = pair_dir / f"pair_{pair:02d}_refits.csv"
        feature_path = pair_dir / f"pair_{pair:02d}_features.csv"
        if refit_path.exists() and feature_path.exists():
            print(f"pair {pair}: existing", flush=True)
            continue
        rows, features, _ = run_credit_missingness_pilot(
            data,
            seed_base=args.seed,
            n_pairs=1,
            pair_start=pair,
            models=("logistic", "lightgbm"),
            indicator_modes=("none", "all"),
        )
        rows.to_csv(refit_path, index=False)
        features.to_csv(feature_path, index=False)
        del rows, features
        gc.collect()
        print(f"pair {pair}: complete", flush=True)

    refit_files = sorted(pair_dir.glob("pair_*_refits.csv"))
    feature_files = sorted(pair_dir.glob("pair_*_features.csv"))
    if len(refit_files) != args.pairs or len(feature_files) != args.pairs:
        raise AssertionError("Incomplete controlled-missingness pair grid.")
    rows = pd.concat([pd.read_csv(path) for path in refit_files], ignore_index=True)
    features = pd.concat([pd.read_csv(path) for path in feature_files], ignore_index=True)
    rows = rows.sort_values(["pair", "model", "indicator_mode", "scenario"])
    features = features.sort_values(["pair", "model", "indicator_mode", "scenario", "feature"])
    rows.to_csv(out / "credit_missingness_confirmation_refits.csv", index=False)
    features.to_csv(out / "credit_missingness_confirmation_feature_profiles.csv", index=False)
    metadata = {
        "version": "1.0.0",
        "runtime_seconds_current_invocation": perf_counter() - start,
        "n_pairs": args.pairs,
        "n_scenario_evaluations": int(len(rows)),
        "resumable_pair_runs": True,
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
