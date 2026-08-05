from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REFERENCE_MANIFEST = Path("reference_outputs/MANIFEST_SHA256_REFERENCE.txt")
DOMAIN_FILES = {
    "heart": ("reference_outputs/heart_cross_site_refits.csv", "outputs/heart/heart_cross_site_refits.csv"),
    "student": ("reference_outputs/student_institution_refits.csv", "outputs/student/student_institution_refits.csv"),
    "oulad": ("reference_outputs/oulad_temporal_refits.csv", "outputs/oulad/oulad_temporal_refits.csv"),
    "acs": ("reference_outputs/acs_temporal_refits.csv", "outputs/acs/acs_temporal_refits.csv"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def parse_sha256_manifest(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    entries: dict[str, str] = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"Malformed SHA-256 manifest line {line_no}: {raw!r}")
        digest, rel = parts
        rel = rel.lstrip("* ")
        if rel in entries:
            raise ValueError(f"Duplicate manifest path: {rel}")
        entries[rel] = digest.lower()
    return entries


def validate_reference_manifest(root: Path) -> dict[str, object]:
    manifest_path = root / REFERENCE_MANIFEST
    entries = parse_sha256_manifest(manifest_path)
    required = {rel_ref for rel_ref, _ in DOMAIN_FILES.values()}
    missing = sorted(required.difference(entries))
    if missing:
        raise AssertionError(f"Reference manifest does not cover required tables: {missing}")
    details = []
    for rel in sorted(required):
        p = root / rel
        if not p.exists():
            raise FileNotFoundError(p)
        actual = sha256(p)
        expected = entries[rel]
        if actual != expected:
            raise AssertionError(f"Reference hash mismatch for {rel}: {actual} != {expected}")
        details.append({"path": rel, "sha256": actual, "verified": True})
    return {
        "manifest": str(REFERENCE_MANIFEST),
        "manifest_sha256": sha256(manifest_path),
        "required_reference_files": len(required),
        "all_required_hashes_verified": True,
        "files": details,
    }


def compare_csv(reference: Path, current: Path, atol: float) -> dict:
    if not reference.exists():
        raise FileNotFoundError(reference)
    if not current.exists():
        raise FileNotFoundError(current)
    old = pd.read_csv(reference)
    new = pd.read_csv(current)
    key_candidates = [
        "pair", "refit_pair", "split_index", "refit_index", "task_index",
        "target_site", "source_school", "target_school", "direction",
        "representation", "code_module", "period", "horizon_day",
        "target_definition", "model",
    ]
    keys = [k for k in key_candidates if k in old.columns and k in new.columns]
    if keys:
        old = old.sort_values(keys, kind="stable").reset_index(drop=True)
        new = new.sort_values(keys, kind="stable").reset_index(drop=True)
    if len(old) != len(new):
        raise AssertionError(f"row count mismatch {reference.name}: {len(old)} != {len(new)}")
    missing = [c for c in old.columns if c not in new.columns]
    if missing:
        raise AssertionError(f"missing reference columns in {current}: {missing}")

    max_numeric = 0.0
    mismatched_columns: list[str] = []
    for c in old.columns:
        a = old[c]
        b = new[c]
        if pd.api.types.is_numeric_dtype(a):
            av = pd.to_numeric(a, errors="coerce").to_numpy(float)
            bv = pd.to_numeric(b, errors="coerce").to_numpy(float)
            finite = np.isfinite(av) & np.isfinite(bv)
            if finite.any():
                max_numeric = max(max_numeric, float(np.max(np.abs(av[finite] - bv[finite]))))
            if not np.allclose(av, bv, rtol=0.0, atol=atol, equal_nan=True):
                mismatched_columns.append(c)
        else:
            aa = a.fillna("<NA>").astype(str).to_numpy()
            bb = b.fillna("<NA>").astype(str).to_numpy()
            if not np.array_equal(aa, bb):
                mismatched_columns.append(c)
    if mismatched_columns:
        raise AssertionError(f"reference parity failure for {reference.name}: {mismatched_columns}")
    return {
        "reference": str(reference.relative_to(ROOT if reference.is_relative_to(ROOT) else reference.parent)),
        "current": str(current.relative_to(ROOT if current.is_relative_to(ROOT) else current.parent)),
        "reference_sha256": sha256(reference),
        "current_sha256": sha256(current),
        "rows": int(len(old)),
        "reference_columns": int(len(old.columns)),
        "current_columns": int(len(new.columns)),
        "new_columns": [c for c in new.columns if c not in old.columns],
        "max_abs_numeric_difference_reference_columns": max_numeric,
        "parity": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Produce the the release raw-rerun parity artefact after runtime validation of the bundled the release reference manifest.")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--atol", type=float, default=1e-12)
    ns = ap.parse_args()
    root = ns.root.resolve()
    out = ns.output_dir or (root / "outputs" / "raw_rerun_audit")
    out.mkdir(parents=True, exist_ok=True)

    manifest_audit = validate_reference_manifest(root)
    details = {}
    for domain, (rel_ref, rel_cur) in DOMAIN_FILES.items():
        details[domain] = compare_csv(root / rel_ref, root / rel_cur, ns.atol)

    payload = {
        "version": "1.0.0",
        "purpose": "Fresh raw-domain outputs versus immutable the release reference tables after runtime SHA-256 validation of the reference manifest.",
        "reference_manifest_validation": manifest_audit,
        "tolerance": {"rtol": 0.0, "atol": ns.atol},
        "all_domains_pass": all(x["parity"] for x in details.values()),
        "domains": details,
    }
    (out / "raw_rerun_parity.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Raw-rerun parity audit the release", "",
        "The parity producer first validates every required the release reference table against `reference_outputs/MANIFEST_SHA256_REFERENCE.txt` at runtime, then compares fresh domain outputs with those verified tables.", "",
        f"Reference manifest validation: **{'PASS' if manifest_audit['all_required_hashes_verified'] else 'FAIL'}**", "",
        f"Overall reference-column parity: **{'PASS' if payload['all_domains_pass'] else 'FAIL'}**", "",
        "| Domain | Rows | Historical cols | Current cols | Max abs numeric difference | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for d, x in details.items():
        lines.append(f"| {d} | {x['rows']} | {x['reference_columns']} | {x['current_columns']} | {x['max_abs_numeric_difference_reference_columns']:.3g} | PASS |")
    lines += ["", "This closes the the release documentation/implementation gap: the reference manifest is now consumed and its hash values are verified by the executable parity step itself.", ""]
    (out / "RAW_RERUN_PARITY.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
