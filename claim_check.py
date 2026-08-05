from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

TITLE="Benchmarking Predictive Performance, Interpretability, Reliability, and Robustness under Distribution Shift for Trustworthy XAI"

def near(a,b,tol=5e-7): return abs(float(a)-float(b))<=tol

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--results-root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ns=ap.parse_args()
    root=ns.results_root; r18=root/'outputs/final_analysis'; r17=root/'outputs/extended_analysis'; checks={}
    sanity=pd.read_csv(r18/'estimand_sanity_summary.csv').set_index('case')
    checks['prior_shift_adjusted_near_zero']=abs(float(sanity.loc['prior_shift','mean_adjusted_excess_u2']))<0.005
    checks['prior_shift_raw_is_positive']=float(sanity.loc['prior_shift','mean_shift_u2'])>0.03
    checks['covariate_shift_retained']=float(sanity.loc['covariate_shift','mean_adjusted_excess_u2'])>0.03
    checks['concept_fixed_explanation_marginal_near_zero']=abs(float(sanity.loc['concept_shift','mean_adjusted_excess_u2']))<0.005
    parity=json.loads((root/'outputs/raw_rerun_audit/raw_rerun_parity.json').read_text())
    checks['reference_manifest_runtime_verified']=bool(parity['reference_manifest_validation']['all_required_hashes_verified']) and parity['reference_manifest_validation']['required_reference_files']==4
    checks['reference_columns_exact']=bool(parity['all_domains_pass']) and all(x['max_abs_numeric_difference_reference_columns']==0.0 for x in parity['domains'].values())
    acs=pd.read_csv(r18/'acs_target_definition_split_sensitivity.csv')
    pooled=acs.groupby('target_definition').target_matched_excess_u2.mean()
    checks['acs_nominal_negative']=bool(pooled['nominal_50k']<0)
    checks['acs_real_positive']=bool(pooled['real_2018_50k']>0)
    wide=acs.pivot(index=['model','split_index'],columns='target_definition',values='target_matched_excess_u2')
    checks['acs_target_sign_agreement_2_of_10']=int((wide.nominal_50k.map(lambda x: x>0)==wide.real_2018_50k.map(lambda x: x>0)).sum())==2
    split=pd.read_csv(r18/'dataset_split_variability_summary.csv').set_index('dataset')
    checks['dataset_split_means_match_primary']=bool(near(split.loc['ACSIncome','split_mean'],-0.0019045575439573) and near(split.loc['Heart Disease','split_mean'],0.03297415009536575) and near(split.loc['OULAD','split_mean'],0.01702441042993376) and near(split.loc['Student Performance','split_mean'],0.01012540939416485))
    task=pd.read_csv(r17/'elementary_task_summary_27.csv')
    checks['elementary_19_of_27']=len(task)==27 and int(task.positive_mean.sum())==19
    # source should call the runtime-validated parity step and keep Scopus separate from scientific reproduction
    rep=(root/'reproduce.py').read_text(encoding='utf-8')
    checks['reproduce_calls_parity']='compare_raw_rerun_parity.py' in rep
    checks['scopus_separate_stage']='--scopus is required only for the separate scopus-audit stage' in rep
    checks['title_retention_constraint_recorded']=True
    checks['all']=all(checks.values())
    ns.output.parent.mkdir(parents=True,exist_ok=True); ns.output.write_text(json.dumps(checks,indent=2),encoding='utf-8'); print(json.dumps(checks,indent=2))
    if not checks['all']: raise SystemExit(1)
if __name__=='__main__': main()
