from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from xaishiftbench.cross_explainer_audit import run_cross_explainer_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/south_german_credit/raw/SouthGermanCredit.asc")
    parser.add_argument("--output-dir", default="outputs/cross_explainer")
    parser.add_argument("--seed-base", type=int, default=2026073006)
    parser.add_argument("--pair-start", type=int, default=0)
    parser.add_argument("--n-pairs", type=int, default=5)
    parser.add_argument("--background-size", type=int, default=24)
    parser.add_argument("--explanation-sample-size", type=int, default=40)
    parser.add_argument("--kernel-nsamples", type=int, default=160)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    observations, scenarios, features, metadata = run_cross_explainer_audit(
        args.data,
        seed_base=args.seed_base,
        pair_start=args.pair_start,
        n_pairs=args.n_pairs,
        background_size=args.background_size,
        explanation_sample_size=args.explanation_sample_size,
        kernel_nsamples=args.kernel_nsamples,
    )
    suffix = f"pairs_{args.pair_start:02d}_{args.pair_start + args.n_pairs - 1:02d}"
    observations.to_csv(output / f"cross_explainer_observations_{suffix}.csv.gz", index=False, compression="gzip")
    scenarios.to_csv(output / f"cross_explainer_scenarios_{suffix}.csv", index=False)
    features.to_csv(output / f"cross_explainer_features_{suffix}.csv.gz", index=False, compression="gzip")
    (output / f"cross_explainer_metadata_{suffix}.json").write_text(
        json.dumps(asdict(metadata), indent=2), encoding="utf-8"
    )
    print(json.dumps(asdict(metadata), indent=2))


if __name__ == "__main__":
    main()
