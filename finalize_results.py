from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "final_analysis"
EXTENDED = ROOT / "outputs" / "extended_analysis"


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    copies = {
        ROOT / "outputs/raw_rerun_audit/raw_rerun_parity.json": OUT / "raw_rerun_parity.json",
        ROOT / "outputs/raw_rerun_audit/RAW_RERUN_PARITY.md": OUT / "RAW_RERUN_PARITY.md",
        ROOT / "outputs/estimand_sanity/estimand_sanity_summary.csv": OUT / "estimand_sanity_summary.csv",
        ROOT / "outputs/estimand_sanity/ESTIMAND_SANITY.json": OUT / "ESTIMAND_SANITY.json",
    }
    for src,dst in copies.items(): shutil.copy2(require(src), dst)
    # Preserve the the release extended-analysis outputs that remain part of the release evidence.
    for name in [
        "dataset_specific_primary_summary.csv",
        "elementary_task_summary_27.csv",
        "forward_vs_symmetrized_headline.json",
        "amplitude_quartile_heterogeneity_summary.csv",
        "oulad_fixed_vs_dynamic_family_sensitivity.csv",
        "acs_pwgpt_weighted_sensitivity_split_representatives.csv",
        "predictive_performance_dataset_model.csv",
    ]:
        shutil.copy2(require(EXTENDED / name), OUT / name)

    parity=json.loads(require(OUT/'raw_rerun_parity.json').read_text(encoding='utf-8'))
    minor=json.loads(require(OUT/'final_results_headline.json').read_text(encoding='utf-8'))
    sanity=pd.read_csv(require(OUT/'estimand_sanity_summary.csv')).set_index('case')
    acs=pd.read_csv(require(OUT/'acs_target_definition_split_sensitivity.csv'))
    wide=acs.pivot_table(index=['model','split_index'],columns='target_definition',values='target_matched_excess_u2').reset_index()
    sign_agreement=float((wide['nominal_50k'].map(lambda x: 1 if x>0 else (-1 if x<0 else 0)) == wide['real_2018_50k'].map(lambda x: 1 if x>0 else (-1 if x<0 else 0))).mean())
    status={
        "version": "1.0.0",
        'title_retained_exactly':True,
        'mandatory_estimand_clarification_supported_by_sanity_simulation': bool(abs(float(sanity.loc['prior_shift','mean_adjusted_excess_u2'])) < 0.005 and float(sanity.loc['covariate_shift','mean_adjusted_excess_u2']) > 0.03),
        'reference_manifest_runtime_hash_validation': bool(parity['reference_manifest_validation']['all_required_hashes_verified']),
        'historical_raw_parity_all_domains': bool(parity['all_domains_pass']),
        'acs_target_definition_sign_agreement_fraction': sign_agreement,
        'acs_nominal_sign_not_robust_to_cpi_target': sign_agreement < 0.5,
        'elementary_19_of_27_is_descriptive_not_independent_inference': True,
        'amplitude_analysis_label':'amplitude-stratified endpoint heterogeneity',
        'scopus_is_not_scientific_full_dependency': True,
    }
    (OUT/'FINALIZATION_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps(status,indent=2))

if __name__=='__main__': main()
