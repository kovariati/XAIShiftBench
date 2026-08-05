from __future__ import annotations

import argparse
import json
from pathlib import Path

from xaishiftbench.acs_temporal_pilot import run_pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=int, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    out = args.root / "outputs" / "acs" / "pair_runs"
    out.mkdir(parents=True, exist_ok=True)
    rows, features, subgroups, metadata = run_pair(args.root, args.pair)
    rows.to_csv(out / f"pair_{args.pair:02d}_rows.csv", index=False)
    features.to_csv(out / f"pair_{args.pair:02d}_features.csv", index=False)
    subgroups.to_csv(out / f"pair_{args.pair:02d}_subgroups.csv", index=False)
    (out / f"pair_{args.pair:02d}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
