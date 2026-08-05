from __future__ import annotations

import argparse
import json
from pathlib import Path

from xaishiftbench.oulad_temporal_pilot import run_pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=int, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    out = args.root / "outputs" / "oulad" / "pair_runs"
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / f"pair_{args.pair:02d}_rows.csv"
    features_path = out / f"pair_{args.pair:02d}_features.csv"
    metadata_path = out / f"pair_{args.pair:02d}_metadata.json"
    if rows_path.exists() and features_path.exists():
        print("existing")
        return
    rows, features, metadata = run_pair(
        args.root / "outputs" / "oulad_prepared" / "oulad_score_free_horizons.csv.gz",
        args.root / "outputs" / "oulad_prepared" / "oulad_temporal_pairs.csv",
        args.pair,
    )
    rows.to_csv(rows_path, index=False)
    features.to_csv(features_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
