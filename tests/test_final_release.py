from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np

from compare_raw_rerun_parity import validate_reference_manifest
from run_estimand_sanity import run_simulation


def test_reference_manifest_is_runtime_verified():
    root = Path(__file__).resolve().parents[1]
    out = validate_reference_manifest(root)
    assert out["all_required_hashes_verified"] is True
    assert out["required_reference_files"] == 4


def test_reference_manifest_detects_tampered_table(tmp_path):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "reference_outputs").mkdir(parents=True)
    manifest = root / "reference_outputs/MANIFEST_SHA256_REFERENCE.txt"
    shutil.copy2(manifest, tmp_path / "reference_outputs/MANIFEST_SHA256_REFERENCE.txt")
    for name in [
        "acs_temporal_refits.csv",
        "heart_cross_site_refits.csv",
        "oulad_temporal_refits.csv",
        "student_institution_refits.csv",
    ]:
        shutil.copy2(root / "reference_outputs" / name, tmp_path / "reference_outputs" / name)
    with (tmp_path / "reference_outputs/acs_temporal_refits.csv").open("ab") as f:
        f.write(b"\n#tamper")
    try:
        validate_reference_manifest(tmp_path)
    except AssertionError as exc:
        assert "Reference hash mismatch" in str(exc)
    else:
        raise AssertionError("tampered reference table was not rejected")


def test_estimand_sanity_has_expected_qualitative_behavior():
    _, summary = run_simulation(repeats=80, seed=1234, n_source=1200, n_target=900, cap=240, d=6)
    s = summary.set_index("case")
    # Pure prior shift moves the raw marginal explanation distribution but the
    # target-prevalence matched source null removes that component on average.
    assert s.loc["prior_shift", "mean_shift_u2"] > 0.03
    assert abs(s.loc["prior_shift", "mean_adjusted_excess_u2"]) < 0.01
    # A within-class attribution/covariate movement survives the adjustment.
    assert s.loc["covariate_shift", "mean_adjusted_excess_u2"] > 0.03
    assert s.loc["covariate_shift", "positive_adjusted_fraction"] > 0.9
    # Pure relabeling with unchanged fixed-model attribution marginal is not a
    # generic target of the unconditional endpoint.
    assert abs(s.loc["concept_shift", "mean_adjusted_excess_u2"]) < 0.01


def test_acs_rows_expose_target_definition_specific_matched_endpoint():
    from xaishiftbench import acs_temporal_pilot
    src = Path(acs_temporal_pilot.__file__).read_text(encoding="utf-8")
    for field in [
        "target_matched_shift_energy_u2",
        "target_matched_refit_null_energy_u2",
        "target_matched_excess_u2",
    ]:
        assert field in src
