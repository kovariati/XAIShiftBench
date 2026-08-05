"""Prepare corrected ACS benchmark samples directly from national PUMS ZIPs.

This portable preparer applies the official within-file ADJINC correction before
income filtering and target construction.  It retains deterministic reservoir
samples for the analysis instead of redistributing the full national extracts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

CPI_2018 = 251.107
CPI_2024 = 313.689
REAL_2024_THRESHOLD = 50_000 * CPI_2024 / CPI_2018
FEATURES = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP_HARM", "WKHP", "SEX", "RAC1P"]
REL_2018 = {
    0: "reference_person", 1: "spouse_or_partner", 2: "child", 3: "child", 4: "child",
    5: "sibling", 6: "parent", 7: "grandchild", 8: "in_law", 9: "in_law",
    10: "other_relative", 11: "roommate_housemate_boarder", 12: "roommate_housemate_boarder",
    13: "spouse_or_partner", 14: "foster_child", 15: "other_nonrelative",
    16: "institutionalized_group_quarters", 17: "noninstitutionalized_group_quarters",
}
REL_2024 = {
    20: "reference_person", 21: "spouse_or_partner", 22: "spouse_or_partner",
    23: "spouse_or_partner", 24: "spouse_or_partner", 25: "child", 26: "child", 27: "child",
    28: "sibling", 29: "parent", 30: "grandchild", 31: "in_law", 32: "in_law",
    33: "other_relative", 34: "roommate_housemate_boarder", 35: "foster_child",
    36: "other_nonrelative", 37: "institutionalized_group_quarters",
    38: "noninstitutionalized_group_quarters",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _keep_lowest_priority(current: pd.DataFrame | None, candidate: pd.DataFrame, n: int) -> pd.DataFrame:
    merged = candidate if current is None else pd.concat([current, candidate], ignore_index=True)
    if len(merged) > n:
        merged = merged.nsmallest(n, "_priority", keep="first")
    return merged.reset_index(drop=True)


def process_part(
    zip_path: Path,
    *,
    year: int,
    part: str,
    sample_n: int,
    seed: int,
    chunksize: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    state_col, rel_col, rel_map = ("ST", "RELP", REL_2018) if year == 2018 else ("STATE", "RELSHIPP", REL_2024)
    usecols = [
        "SERIALNO", "SPORDER", state_col, "PWGTP", "ADJINC", "AGEP", "COW", "SCHL", "MAR",
        "OCCP", "POBP", rel_col, "WKHP", "SEX", "RAC1P", "PINCP",
    ]
    rng = np.random.default_rng(seed)
    sample: pd.DataFrame | None = None
    n_raw = n_filtered = pos_nom = pos_real = 0
    weight_sum = weight_pos_nom = weight_pos_real = 0.0
    adj_min = float("inf")
    adj_max = float("-inf")
    factor_values: set[float] = set()
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"CRC error in {zip_path}: {bad}")
        names = archive.namelist()
        if len(names) != 1:
            raise ValueError(f"Expected one member in {zip_path}: {names}")
        with archive.open(names[0]) as raw:
            for chunk_index, chunk in enumerate(
                pd.read_csv(raw, usecols=usecols, chunksize=chunksize, low_memory=False)
            ):
                n_raw += len(chunk)
                for column in usecols:
                    if column != "SERIALNO":
                        chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
                chunk["PINCP_RAW"] = chunk["PINCP"]
                chunk["ADJINC_FACTOR"] = chunk["ADJINC"] / 1_000_000.0
                chunk["PINCP_ADJ"] = chunk["PINCP_RAW"] * chunk["ADJINC_FACTOR"]
                factor_values.update(np.round(chunk["ADJINC_FACTOR"].dropna().unique(), 9).tolist())
                chunk = chunk.loc[
                    (chunk["AGEP"] > 16)
                    & (chunk["PINCP_ADJ"] > 100)
                    & (chunk["WKHP"] > 0)
                    & (chunk["PWGTP"] >= 1)
                ].copy()
                if chunk.empty:
                    continue
                chunk["YEAR"] = year
                chunk["SOURCE_PART"] = part
                chunk["STATE"] = chunk[state_col].astype("Int16")
                chunk["RELP_RAW"] = chunk[rel_col].astype("Int16")
                chunk["RELP_HARM"] = chunk[rel_col].map(rel_map).fillna("unknown_relationship")
                chunk["PINCP"] = chunk["PINCP_ADJ"]
                chunk["TARGET_NOMINAL_50K"] = (chunk["PINCP_ADJ"] > 50_000).astype("int8")
                real_threshold = 50_000 if year == 2018 else REAL_2024_THRESHOLD
                chunk["TARGET_REAL_2018_50K"] = (chunk["PINCP_ADJ"] > real_threshold).astype("int8")
                n_filtered += len(chunk)
                pos_nom += int(chunk["TARGET_NOMINAL_50K"].sum())
                pos_real += int(chunk["TARGET_REAL_2018_50K"].sum())
                weights = chunk["PWGTP"].to_numpy(float)
                weight_sum += float(weights.sum())
                weight_pos_nom += float(chunk.loc[chunk["TARGET_NOMINAL_50K"] == 1, "PWGTP"].sum())
                weight_pos_real += float(chunk.loc[chunk["TARGET_REAL_2018_50K"] == 1, "PWGTP"].sum())
                adj_min = min(adj_min, float(chunk["PINCP_ADJ"].min()))
                adj_max = max(adj_max, float(chunk["PINCP_ADJ"].max()))
                chunk["_priority"] = rng.random(len(chunk))
                outcols = [
                    "YEAR", "SOURCE_PART", "SERIALNO", "SPORDER", "STATE", "PWGTP", "ADJINC",
                    "ADJINC_FACTOR", *FEATURES, "RELP_RAW", "PINCP_RAW", "PINCP_ADJ", "PINCP",
                    "TARGET_NOMINAL_50K", "TARGET_REAL_2018_50K", "_priority",
                ]
                sample = _keep_lowest_priority(sample, chunk[outcols], sample_n)
                print(
                    f"{zip_path.name}: chunk={chunk_index + 1} raw={n_raw:,} "
                    f"filtered={n_filtered:,} retained={len(sample):,}",
                    flush=True,
                )
    assert sample is not None and len(sample) == sample_n
    sample = sample.sort_values("_priority").drop(columns="_priority").reset_index(drop=True)
    audit: dict[str, object] = {
        "zip": str(zip_path),
        "zip_sha256": sha256(zip_path),
        "year": year,
        "part": part,
        "seed": seed,
        "sample_n": sample_n,
        "n_raw": n_raw,
        "n_filtered": n_filtered,
        "positive_nominal": pos_nom,
        "prevalence_nominal": pos_nom / n_filtered,
        "positive_real": pos_real,
        "prevalence_real": pos_real / n_filtered,
        "weighted_prevalence_nominal": weight_pos_nom / weight_sum,
        "weighted_prevalence_real": weight_pos_real / weight_sum,
        "weight_sum": weight_sum,
        "adjinc_factor_values": sorted(factor_values),
        "pincp_adjusted_min_after_filter": adj_min,
        "pincp_adjusted_max_after_filter": adj_max,
    }
    return sample, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ["zip_2018_a", "zip_2018_b", "zip_2024_a", "zip_2024_b"]:
        parser.add_argument("--" + name.replace("_", "-"), dest=name, type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=2026073008)
    args = parser.parse_args()
    start = time.time()
    out = args.root / "outputs" / "acs_temporal_pilot"
    out.mkdir(parents=True, exist_ok=True)
    specs = [
        (args.zip_2018_a, 2018, "A", 60_000, args.seed + 1),
        (args.zip_2018_b, 2018, "B", 60_000, args.seed + 2),
        (args.zip_2024_a, 2024, "A", 30_000, args.seed + 3),
        (args.zip_2024_b, 2024, "B", 30_000, args.seed + 4),
    ]
    samples: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for path, year, part, n, seed in specs:
        sample, audit = process_part(
            path, year=year, part=part, sample_n=n, seed=seed, chunksize=args.chunksize
        )
        samples.append(sample)
        audits.append(audit)
    source = pd.concat(samples[:2], ignore_index=True)
    target = pd.concat(samples[2:], ignore_index=True)
    source.to_csv(out / "acs_2018_source_pool.csv.gz", index=False, compression={"method": "gzip", "compresslevel": 1})
    target.to_csv(out / "acs_2024_target_eval.csv.gz", index=False, compression={"method": "gzip", "compresslevel": 1})

    def summarize(frame: pd.DataFrame) -> dict[str, object]:
        result: dict[str, object] = {"n": len(frame), "states": int(frame["STATE"].nunique())}
        for target_name in ["TARGET_NOMINAL_50K", "TARGET_REAL_2018_50K"]:
            y = frame[target_name].to_numpy(float)
            w = frame["PWGTP"].to_numpy(float)
            result[target_name] = {
                "prevalence": float(np.mean(y)),
                "weighted_prevalence": float(np.average(y, weights=w)),
            }
        return result

    payload = {
        "version": "1.0.0",
        "method": "direct deterministic reservoir samples after official within-file ADJINC adjustment",
        "income_adjustment": "PINCP_ADJ = PINCP_RAW * ADJINC / 1,000,000 before filtering and target construction",
        "adult_filter": "AGEP > 16; PINCP_ADJ > 100; WKHP > 0; PWGTP >= 1",
        "nominal_target": "PINCP_ADJ > 50,000 survey-year dollars",
        "real_2018_target": {
            "2018_threshold": 50_000,
            "2024_threshold": REAL_2024_THRESHOLD,
            "cpi_2018": CPI_2018,
            "cpi_2024": CPI_2024,
        },
        "source_pool": summarize(source),
        "target_eval": summarize(target),
        "parts": audits,
        "runtime_seconds": time.time() - start,
    }
    (out / "acs_temporal_preparation_audit.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    pd.DataFrame(audits).to_csv(out / "acs_part_audit.csv", index=False)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
