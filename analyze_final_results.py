from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _dataset_split_variability(pair: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Preserve the release hierarchy: average refits within split/task/model,
    # average models within split/task, then elementary tasks within dataset.
    a = pair.groupby(['dataset','task_id','model','split_index'], as_index=False)['calibrated_excess_u2'].mean()
    b = a.groupby(['dataset','task_id','split_index'], as_index=False)['calibrated_excess_u2'].mean()
    c = b.groupby(['dataset','split_index'], as_index=False)['calibrated_excess_u2'].mean().rename(columns={'calibrated_excess_u2':'dataset_split_mean_excess_u2'})
    rows=[]
    for ds,g in c.groupby('dataset',sort=True):
        x=g.dataset_split_mean_excess_u2.to_numpy(float)
        q25,q75=np.quantile(x,[0.25,0.75])
        rows.append({
            'dataset':ds,
            'n_source_splits':len(x),
            'split_mean':float(np.mean(x)),
            'split_sd':float(np.std(x,ddof=1)),
            'split_min':float(np.min(x)),
            'split_q25':float(q25),
            'split_median':float(np.median(x)),
            'split_q75':float(q75),
            'split_max':float(np.max(x)),
            'split_iqr':float(q75-q25),
            'interpretation':'computational source-split variability; not a population confidence interval',
        })
    return c,pd.DataFrame(rows)


def _acs_target_definition_sensitivity(acs: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    need=['target_matched_shift_energy_u2','target_matched_same_model_sample_null_energy_u2','target_matched_refit_null_energy_u2','target_matched_excess_u2','target_matched_symmetrized_excess_u2','target_matched_n_explanation']
    missing=[c for c in need if c not in acs]
    if missing: raise KeyError(f'ACS the release target-definition fields missing: {missing}')
    # Average four refits within source split first.
    split=acs.groupby(['target_definition','model','split_index'],as_index=False).agg(
        target_matched_shift_energy_u2=('target_matched_shift_energy_u2','mean'),
        target_matched_same_model_sample_null_energy_u2=('target_matched_same_model_sample_null_energy_u2','mean'),
        target_matched_refit_null_energy_u2=('target_matched_refit_null_energy_u2','mean'),
        target_matched_excess_u2=('target_matched_excess_u2','mean'),
        target_matched_symmetrized_excess_u2=('target_matched_symmetrized_excess_u2','mean'),
        target_prevalence=('target_prevalence','mean'),
        target_matched_n_explanation=('target_matched_n_explanation','mean'),
    )
    summary=split.groupby(['target_definition','model'],as_index=False).agg(
        n_source_splits=('split_index','size'),
        mean_target_prevalence=('target_prevalence','mean'),
        mean_target_matched_n_explanation=('target_matched_n_explanation','mean'),
        mean_target_matched_shift_u2=('target_matched_shift_energy_u2','mean'),
        mean_target_matched_same_model_null_u2=('target_matched_same_model_sample_null_energy_u2','mean'),
        mean_target_matched_refit_null_u2=('target_matched_refit_null_energy_u2','mean'),
        mean_target_matched_excess_u2=('target_matched_excess_u2','mean'),
        split_sd_target_matched_excess_u2=('target_matched_excess_u2','std'),
        split_min_target_matched_excess_u2=('target_matched_excess_u2','min'),
        split_max_target_matched_excess_u2=('target_matched_excess_u2','max'),
        mean_target_matched_symmetrized_excess_u2=('target_matched_symmetrized_excess_u2','mean'),
    )
    pooled=split.groupby('target_definition',as_index=False).agg(
        model_split_cells=('target_matched_excess_u2','size'),
        mean_target_prevalence=('target_prevalence','mean'),
        mean_target_matched_excess_u2=('target_matched_excess_u2','mean'),
        median_target_matched_excess_u2=('target_matched_excess_u2','median'),
        positive_model_split_cells=('target_matched_excess_u2',lambda x:int((x>0).sum())),
    )
    wide=split.pivot_table(index=['model','split_index'],columns='target_definition',values='target_matched_excess_u2').reset_index()
    wide['nominal_real_sign_agreement']=np.sign(wide['nominal_50k'])==np.sign(wide['real_2018_50k'])
    wide['real_minus_nominal_u2']=wide['real_2018_50k']-wide['nominal_50k']
    headline={
        'target_definition_comparison_unit':'model x source split after averaging four refits',
        'n_compared_model_split_cells':int(len(wide)),
        'nominal_real_sign_agreement_cells':int(wide.nominal_real_sign_agreement.sum()),
        'nominal_real_sign_agreement_fraction':float(wide.nominal_real_sign_agreement.mean()),
        'pooled':pooled.to_dict(orient='records'),
        'interpretation':'The release-defined nominal-50k ACS endpoint is retained as primary. CPI-adjusted real-2018-50k is a target-definition sensitivity because changing the target prevalence changes the matched source-null construction itself.',
    }
    return split,summary,headline


def _proper_scoring_summary(pair: pd.DataFrame) -> pd.DataFrame:
    # Dataset/model means with the same hierarchy used for AUROC reporting.
    cols=['id_log_loss','target_log_loss','id_brier','target_brier']
    a=pair.groupby(['dataset','task_id','model','split_index'],as_index=False)[cols].mean()
    b=a.groupby(['dataset','task_id','model'],as_index=False)[cols].mean()
    out=b.groupby(['dataset','model'],as_index=False).agg(
        n_elementary_tasks=('task_id','size'),
        mean_id_log_loss=('id_log_loss','mean'),
        mean_target_log_loss=('target_log_loss','mean'),
        mean_id_brier=('id_brier','mean'),
        mean_target_brier=('target_brier','mean'),
    )
    out['mean_delta_log_loss']=out.mean_target_log_loss-out.mean_id_log_loss
    out['mean_delta_brier']=out.mean_target_brier-out.mean_id_brier
    return out


def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,default=None)
    ns=ap.parse_args(); root=ns.root.resolve(); out=ns.output_dir or root/'outputs/final_analysis'; out.mkdir(parents=True,exist_ok=True)
    pair=pd.read_csv(root/'outputs/multidomain/pair_level_real_deployment.csv')
    split_rows,split_summary=_dataset_split_variability(pair)
    split_rows.to_csv(out/'dataset_source_split_means.csv',index=False)
    split_summary.to_csv(out/'dataset_split_variability_summary.csv',index=False)

    proper=_proper_scoring_summary(pair)
    proper.to_csv(out/'predictive_proper_scores_dataset_model.csv',index=False)

    acs=pd.read_csv(root/'outputs/acs/acs_temporal_refits.csv')
    acs_split,acs_summary,acs_headline=_acs_target_definition_sensitivity(acs)
    acs_split.to_csv(out/'acs_target_definition_split_sensitivity.csv',index=False)
    acs_summary.to_csv(out/'acs_target_definition_model_summary.csv',index=False)
    (out/'ACS_TARGET_DEFINITION_SENSITIVITY.json').write_text(json.dumps(acs_headline,indent=2),encoding='utf-8')

    sanity=pd.read_csv(root/'outputs/estimand_sanity/estimand_sanity_summary.csv')
    sanity.to_csv(out/'estimand_sanity_summary.csv',index=False)
    headline={
        "version": "1.0.0",
        'title_changed':False,
        'estimand_sanity':sanity.to_dict(orient='records'),
        'dataset_split_variability':split_summary.to_dict(orient='records'),
        'acs_target_definition_sensitivity':acs_headline,
        'amplitude_analysis_term':'amplitude-stratified endpoint heterogeneity',
        'elementary_task_positive_count_interpretation':'19/27 is a descriptive count of dependent release-defined tasks, not a binomial sample of independent deployments',
    }
    (out/'final_results_headline.json').write_text(json.dumps(headline,indent=2),encoding='utf-8')
    print(json.dumps(headline,indent=2))

if __name__=='__main__': main()
