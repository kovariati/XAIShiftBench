"""Combine resumable OULAD or ACS pair outputs into canonical the release files.

The the release release retains the release filenames so that the audited aggregation
scripts remain byte-compatible with the corrected numerical release.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import time

import pandas as pd


def _read_complete(files: list[Path], label: str) -> pd.DataFrame:
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {label} pair outputs:\n" + "\n".join(missing))
    frames = [pd.read_csv(path) for path in files]
    if any(frame.empty for frame in frames):
        raise AssertionError(f"At least one {label} pair output is empty.")
    return pd.concat(frames, ignore_index=True)


def combine(root: Path, domain: str, pairs: int) -> dict[str, object]:
    if pairs < 1:
        raise ValueError("pairs must be positive")
    if domain == "oulad":
        out = root / "outputs" / "oulad"
        row_name = "oulad_temporal_refits.csv"
        feature_name = "oulad_temporal_feature_profiles.csv"
        row_sort = ["refit_pair", "task_index", "model"]
        feature_sort = ["refit_pair", "task_index", "model", "feature"]
        extras: list[tuple[str, str, list[str]]] = []
    elif domain == "acs":
        out = root / "outputs" / "acs"
        row_name = "acs_temporal_refits.csv"
        feature_name = "acs_temporal_feature_profiles.csv"
        row_sort = ["refit_pair", "model", "target_definition"]
        feature_sort = ["refit_pair", "model", "feature"]
        extras = [("subgroups", "acs_temporal_subgroups.csv", ["refit_pair", "model", "group", "level"])]
    else:
        raise ValueError(f"Unsupported domain: {domain}")

    pair_dir = out / "pair_runs"
    row_files = [pair_dir / f"pair_{i:02d}_rows.csv" for i in range(pairs)]
    feature_files = [pair_dir / f"pair_{i:02d}_features.csv" for i in range(pairs)]
    metadata_files = [pair_dir / f"pair_{i:02d}_metadata.json" for i in range(pairs)]
    rows = _read_complete(row_files, f"{domain} row")
    features = _read_complete(feature_files, f"{domain} feature")
    rows = rows.sort_values([c for c in row_sort if c in rows.columns], kind="mergesort").reset_index(drop=True)
    features = features.sort_values([c for c in feature_sort if c in features.columns], kind="mergesort").reset_index(drop=True)
    out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / row_name, index=False)
    features.to_csv(out / feature_name, index=False)

    extra_counts: dict[str, int] = {}
    for stem, name, sort_keys in extras:
        files = [pair_dir / f"pair_{i:02d}_{stem}.csv" for i in range(pairs)]
        extra = _read_complete(files, f"{domain} {stem}")
        extra = extra.sort_values([c for c in sort_keys if c in extra.columns], kind="mergesort").reset_index(drop=True)
        extra.to_csv(out / name, index=False)
        extra_counts[name] = int(len(extra))

    metadata_missing = [str(p) for p in metadata_files if not p.exists()]
    if metadata_missing:
        raise FileNotFoundError("Missing pair metadata:\n" + "\n".join(metadata_missing))
    pair_metadata = [json.loads(path.read_text(encoding="utf-8")) for path in metadata_files]
    payload = {
        "version": "1.0.0",
        "canonical_output_schema": "the release backward-compatible",
        "domain": domain,
        "n_pairs": pairs,
        "n_row_records": int(len(rows)),
        "n_feature_records": int(len(features)),
        "extra_record_counts": extra_counts,
        "pair_metadata_files": [p.name for p in metadata_files],
        "combined_unix_time": time(),
    }
    (out / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--domain", choices=["oulad", "acs"], required=True)
    ap.add_argument("--pairs", type=int, default=20)
    ns = ap.parse_args()
    print(json.dumps(combine(ns.root, ns.domain, ns.pairs), indent=2))


if __name__ == "__main__":
    main()
