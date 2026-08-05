"""Closed scientific raw-to-results orchestration for XAIShiftBench the release.

The ``full`` stage requires six small local raw files and five external public
archives. It generates the raw-rerun parity artefact itself from bundled,
hash-registered the release reference result tables before final scientific
finalization. The optional Scopus title-term audit is deliberately separate
from the scientific reproduction chain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REGISTRY = ROOT / "data_checksums" / "expected_raw_inputs.json"


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()


def verify(path: Path, expected: str, label: str, failures: list[str]) -> None:
    if not path.exists(): failures.append(f"MISSING {label}: {path}"); return
    actual=sha256(path)
    if actual != expected: failures.append(f"HASH {label}: {actual} != {expected} ({path})")


def check_inputs(ns: argparse.Namespace, require_external: bool) -> None:
    d=json.loads(REGISTRY.read_text(encoding="utf-8")); failures=[]
    for rel,expected in d["local_files"].items(): verify(ROOT/rel,expected,rel,failures)
    for key,item in d["external_archives"].items():
        p=getattr(ns,key)
        if p is None:
            if require_external: failures.append(f"MISSING ARGUMENT --{key.replace('_','-')}")
            continue
        verify(p,item["sha256"],key,failures)
    if failures: raise SystemExit("\n".join(failures))
    print("Input hashes: PASS")


def command(*parts: object) -> list[str]: return [str(p) for p in parts]

def execute(cmd: list[str], dry_run: bool=False) -> None:
    print("+", " ".join(cmd), flush=True)
    if dry_run: return
    env={**os.environ,"PYTHONPATH":str(ROOT/"src")}
    subprocess.run(cmd,cwd=ROOT,check=True,env=env)

def ph(v: Path|None,label:str)->str: return str(v) if v else f"<{label}>"


def scientific_full_commands(ns: argparse.Namespace) -> list[list[str]]:
    py=sys.executable
    cmds=[
        command(py,"prepare_oulad.py","--zip",ph(ns.oulad_zip,"oulad_zip"),"--output-dir",ROOT/"outputs/oulad_prepared"),
        command(py,"prepare_acs_temporal.py","--zip-2018-a",ph(ns.acs_2018_a,"acs_2018_a"),"--zip-2018-b",ph(ns.acs_2018_b,"acs_2018_b"),"--zip-2024-a",ph(ns.acs_2024_a,"acs_2024_a"),"--zip-2024-b",ph(ns.acs_2024_b,"acs_2024_b"),"--root",ROOT,"--chunksize",100000),
        command(py,"run_heart_pilot.py","--data-dir",ROOT/"data/heart_disease/raw","--output-dir",ROOT/"outputs/heart","--pairs",20,"--splits",5,"--explain-cap",160),
        command(py,"run_student_institution_pilot.py","--root",ROOT,"--pairs",20),
    ]
    cmds.extend(command(py,"run_oulad_temporal_pair.py","--pair",i,"--root",ROOT) for i in range(20))
    cmds.append(command(py,"combine_domain_pair_outputs.py","--root",ROOT,"--domain","oulad","--pairs",20))
    cmds.extend(command(py,"run_acs_temporal_pair.py","--pair",i,"--root",ROOT) for i in range(20))
    cmds.append(command(py,"combine_domain_pair_outputs.py","--root",ROOT,"--domain","acs","--pairs",20))
    cmds.extend([
        command(py,"analyze_l1_rerun.py","--root",ROOT),
        command(py,"audit_explanation_sample_counts.py","--root",ROOT),
        command(py,"run_explanation_sampling_sensitivity.py","--root",ROOT,"--repeats",50),
        command(py,"run_credit_missingness_pilot.py","--root",ROOT,"--pairs",10),
        command(py,"run_credit_missingness_confirmation.py","--root",ROOT,"--pairs",20),
        command(py,"analyze_credit_confirmation.py"),
    ])
    cmds.append(command(py,"run_cross_explainer_audit.py","--data",ROOT/"data/south_german_credit/raw/SouthGermanCredit.asc","--output-dir",ROOT/"outputs/cross_explainer_exploratory","--seed-base",2026073006,"--pair-start",0,"--n-pairs",5,"--background-size",24,"--explanation-sample-size",40,"--kernel-nsamples",160))
    for start in range(0,20,5):
        cmds.append(command(py,"run_cross_explainer_audit.py","--data",ROOT/"data/south_german_credit/raw/SouthGermanCredit.asc","--output-dir",ROOT/"outputs/cross_explainer_confirmation","--seed-base",2026073016,"--pair-start",start,"--n-pairs",5,"--background-size",24,"--explanation-sample-size",40,"--kernel-nsamples",160))
    cmds.extend([
        command(py,"analyze_cross_explainer_confirmation.py"),
        command(py,"run_oulad_hyperparameter_sensitivity.py"),
        command(py,"analyze_multidomain.py"),
        command(py,"analyze_signed_global.py"),
        command(py,"prepare_oulad_fixed_cohort.py","--model-table",ROOT/"outputs/oulad_prepared/oulad_score_free_horizons.csv.gz","--output-dir",ROOT/"outputs/oulad_fixed_cohort"),
    ])
    cmds.extend(command(py,"run_oulad_fixed_cohort_pair.py","--pair",i,"--root",ROOT) for i in range(20))
    cmds.extend(command(py,"run_acs_weighted_sensitivity_pair.py","--pair",i,"--root",ROOT) for i in [0,4,8,12,16])
    cmds.extend([
        command(py,"compare_raw_rerun_parity.py","--root",ROOT),
        command(py,"run_estimand_sanity.py","--output-dir",ROOT/"outputs/estimand_sanity","--repeats",400),
        command(py,"analyze_extended_results.py","--results-root",ROOT,"--output-dir",ROOT/"outputs/extended_analysis","--figure-dir",ROOT/"figures/extended_analysis"),
        command(py,"analyze_final_results.py","--root",ROOT,"--output-dir",ROOT/"outputs/final_analysis"),
        command(py,"finalize_results.py"),
        command(py,"claim_check_core_results.py"),
        command(py,"claim_check.py","--results-root",ROOT,"--output",ROOT/"outputs/final_analysis/CLAIM_CHECK.json"),
        command(py,"-m","pytest","-q","-rs"),
    ])
    return cmds


def add_external(ap: argparse.ArgumentParser) -> None:
    for n in ("oulad_zip","acs_2018_a","acs_2018_b","acs_2024_a","acs_2024_b"):
        ap.add_argument("--"+n.replace("_","-"),dest=n,type=Path)


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("stage",choices=["check-inputs","test","aggregate","all-lightweight","full-plan","full","scopus-audit"])
    add_external(ap)
    ap.add_argument("--scopus",type=Path,help="Optional title-term audit used only by scopus-audit stage.")
    ap.add_argument("--dry-run",action="store_true")
    ns=ap.parse_args()
    if ns.stage=="check-inputs": check_inputs(ns,False)
    elif ns.stage in {"full-plan","full"}:
        if ns.stage=="full" and not ns.dry_run: check_inputs(ns,True)
        for c in scientific_full_commands(ns): execute(c, ns.dry_run or ns.stage=="full-plan")
    elif ns.stage in {"test","all-lightweight"}: execute(command(sys.executable,"-m","pytest","-q","-rs"),ns.dry_run)
    if ns.stage in {"aggregate","all-lightweight"}:
        execute(command(sys.executable,"analyze_multidomain.py"),ns.dry_run)
        execute(command(sys.executable,"analyze_signed_global.py"),ns.dry_run)
    if ns.stage=="scopus-audit":
        if ns.scopus is None: raise SystemExit("--scopus is required only for the separate scopus-audit stage")
        execute(command(sys.executable,"analyze_extended_results.py","--results-root",ROOT,"--output-dir",ROOT/"outputs/scopus_audit","--figure-dir",ROOT/"figures/scopus_audit","--scopus",ns.scopus),ns.dry_run)


if __name__=="__main__": main()
