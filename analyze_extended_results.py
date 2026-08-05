from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_helpers import metric_scale_simulation, small_denominator_simulation, scopus_audit


def task_key_for_amp(dataset: str, task_id: str) -> str:
    if dataset == 'ACSIncome':
        return 'ACSIncome_2018_to_2024'
    if dataset == 'OULAD':
        m, p, h = str(task_id).split('|')
        return f'{m}-{p}|day{h}'
    return str(task_id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-root', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--figure-dir', type=Path, required=True)
    ap.add_argument('--scopus', type=Path, default=None, help='Optional bibliometric title-term audit; not part of scientific full reproduction.')
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    root = args.results_root
    multi = root / 'outputs' / 'multidomain'

    task = pd.read_csv(multi / 'task_definition_summary_27.csv')
    strict = pd.read_csv(multi / 'strict_contrast_summary_16.csv')
    model = pd.read_csv(multi / 'model_family_split_summary.csv')
    pair = pd.read_csv(multi / 'pair_level_real_deployment.csv')

    # 27 elementary tasks remain the finest release-defined cross-domain result unit.
    task_out = task.copy()
    task_out['positive_mean'] = task_out['mean_calibrated_excess_u2'] > 0
    task_out['endpoint_name'] = 'matched_null_adjusted_signed_composition_energy_u2'
    task_out.to_csv(args.output_dir / 'elementary_task_summary_27.csv', index=False)

    # Dataset-specific results are the highest-level primary scientific summaries.
    dataset = task_out.groupby('dataset', as_index=False).agg(
        n_elementary_tasks=('task_id', 'size'),
        dataset_mean_raw_excess_u2=('mean_calibrated_excess_u2', 'mean'),
        dataset_median_raw_excess_u2=('mean_calibrated_excess_u2', 'median'),
        positive_elementary_tasks=('positive_mean', 'sum'),
        mean_shift_energy_u2=('mean_shift_energy_u2', 'mean'),
        mean_same_model_sample_null_u2=('mean_same_model_sample_null_energy_u2', 'mean'),
        mean_refit_sample_null_u2=('mean_refit_sample_null_energy_u2', 'mean'),
        mean_refit_increment_u2=('mean_refit_increment_u2', 'mean'),
    )
    dataset['raw_scale_index_only'] = True
    dataset['cross_dataset_commensurability'] = 'not_established'
    dataset.to_csv(args.output_dir / 'dataset_specific_primary_summary.csv', index=False)
    equal_dataset_index = float(dataset['dataset_mean_raw_excess_u2'].mean())
    (args.output_dir / 'equal_dataset_weighted_raw_scale_index.json').write_text(json.dumps({
        'value': equal_dataset_index,
        'name': 'equal-dataset-weighted raw-scale descriptive index',
        'interpretation': 'arithmetic mean of four dataset-specific raw energy-excess means; not a common-scale effect size',
        'primary_status': 'supporting_descriptive_index_not_primary_effect',
    }, indent=2), encoding='utf-8')

    # Baseline decomposition requested by the reviewer.
    decomp = dataset[['dataset','n_elementary_tasks','mean_shift_energy_u2','mean_same_model_sample_null_u2','mean_refit_sample_null_u2','mean_refit_increment_u2','dataset_mean_raw_excess_u2']].copy()
    decomp = decomp.rename(columns={'dataset_mean_raw_excess_u2':'mean_matched_null_adjusted_excess_u2'})
    decomp.to_csv(args.output_dir / 'dataset_baseline_decomposition.csv', index=False)

    # Pooled 54 task-model raw-scale decomposition is reported only as a diagnostic;
    # cross-dataset commensurability is not assumed.
    tm_components = pair.groupby(['dataset','task_id','model'], as_index=False).agg(
        shift_energy_u2=('shift_energy_u2','mean'),
        same_model_sample_null_energy_u2=('same_model_sample_null_energy_u2','mean'),
        refit_sample_null_energy_u2=('refit_sample_null_energy_u2','mean'),
        matched_null_adjusted_excess_u2=('calibrated_excess_u2','mean'),
    )
    pooled_diag = {
        'n_task_model_cells': int(len(tm_components)),
        'mean_shift_energy_u2': float(tm_components.shift_energy_u2.mean()),
        'mean_same_model_sample_null_energy_u2': float(tm_components.same_model_sample_null_energy_u2.mean()),
        'mean_refit_sample_null_energy_u2': float(tm_components.refit_sample_null_energy_u2.mean()),
        'mean_matched_null_adjusted_excess_u2': float(tm_components.matched_null_adjusted_excess_u2.mean()),
        'interpretation': 'raw-scale pooled diagnostic only; not a common cross-dataset effect size',
    }
    (args.output_dir / 'pooled_task_model_baseline_diagnostic.json').write_text(json.dumps(pooled_diag, indent=2), encoding='utf-8')

    # OULAD dynamic-risk-set horizon results.
    oulad = task_out[task_out.dataset.eq('OULAD')].copy()
    oulad[['module','period','horizon_day']] = oulad['task_id'].str.split('|', expand=True)
    oulad['horizon_day'] = oulad['horizon_day'].astype(int)
    oulad = oulad.sort_values(['module','period','horizon_day'])
    oulad.to_csv(args.output_dir / 'oulad_dynamic_riskset_horizon_summary.csv', index=False)
    wide = oulad.pivot(index='family_id', columns='horizon_day', values='mean_calibrated_excess_u2').reset_index()
    wide['sign_change_14_to_56'] = np.sign(wide[14]) != np.sign(wide[56])
    wide['day56_minus_day14'] = wide[56] - wide[14]
    wide.to_csv(args.output_dir / 'oulad_dynamic_riskset_horizon_contrast.csv', index=False)

    # Fixed-cohort sensitivity: same students retained at both horizons within each presentation.
    fixed_dir = root / 'outputs' / 'oulad_fixed_cohort' / 'pair_runs'
    fixed_files = sorted(fixed_dir.glob('pair_*_rows.csv'))
    fixed_summary = None
    fixed_family = None
    if len(fixed_files) == 20:
        fixed = pd.concat([pd.read_csv(p) for p in fixed_files], ignore_index=True)
        split = fixed.groupby(['split_index','code_module','period','horizon_day'], as_index=False)['calibrated_excess_u2'].mean()
        fixed_summary = split.groupby(['code_module','period','horizon_day'], as_index=False).agg(
            fixed_cohort_mean_excess_u2=('calibrated_excess_u2','mean'),
            fixed_cohort_split_sd=('calibrated_excess_u2','std'),
            fixed_cohort_min_split=('calibrated_excess_u2','min'),
            fixed_cohort_max_split=('calibrated_excess_u2','max'),
        )
        dyn_small = oulad[['module','period','horizon_day','mean_calibrated_excess_u2']].rename(columns={'module':'code_module','mean_calibrated_excess_u2':'dynamic_riskset_mean_excess_u2'})
        fixed_summary = dyn_small.merge(fixed_summary, on=['code_module','period','horizon_day'], how='inner')
        fixed_summary['fixed_minus_dynamic_u2'] = fixed_summary['fixed_cohort_mean_excess_u2'] - fixed_summary['dynamic_riskset_mean_excess_u2']
        fixed_summary.to_csv(args.output_dir / 'oulad_fixed_cohort_horizon_sensitivity.csv', index=False)
        fwide = fixed_summary.pivot(index=['code_module','period'], columns='horizon_day', values='fixed_cohort_mean_excess_u2').reset_index()
        fwide['fixed_sign_change_14_to_56'] = np.sign(fwide[14]) != np.sign(fwide[56])
        fwide['fixed_day56_minus_day14'] = fwide[56] - fwide[14]
        dwide = wide.copy()
        dwide[['code_module','period']] = dwide['family_id'].str.split('|', n=1, expand=True)
        dwide = dwide[['code_module','period',14,56,'sign_change_14_to_56','day56_minus_day14']].rename(columns={14:'dynamic_day14',56:'dynamic_day56','sign_change_14_to_56':'dynamic_sign_change_14_to_56','day56_minus_day14':'dynamic_day56_minus_day14'})
        fixed_family = dwide.merge(fwide,on=['code_module','period'])
        fixed_family.to_csv(args.output_dir / 'oulad_fixed_vs_dynamic_family_sensitivity.csv', index=False)

    # Model-specific results and forward/symmetric sensitivity.
    model_dataset = model.groupby(['model','dataset'], as_index=False).agg(
        n_strict_contrasts=('family_id','size'),
        dataset_raw_mean=('mean_calibrated_excess_u2','mean'),
        positive_strict_contrasts=('mean_calibrated_excess_u2', lambda x: int((x>0).sum())),
    )
    model_summary = model_dataset.groupby('model', as_index=False).agg(
        equal_dataset_weighted_raw_scale_index=('dataset_raw_mean','mean'),
        datasets=('dataset','size'),
    )
    model_dataset.to_csv(args.output_dir / 'model_dataset_specific_summary.csv', index=False)
    model_summary.to_csv(args.output_dir / 'model_equal_dataset_raw_scale_index.csv', index=False)

    # Forward vs symmetrized at task-model level, respecting split hierarchy.
    split_tm = pair.groupby(['dataset','task_id','model','split_index'], as_index=False).agg(
        forward=('calibrated_excess_u2','mean'),
        symmetric=('symmetrized_calibrated_excess_u2','mean'),
    )
    tm = split_tm.groupby(['dataset','task_id','model'], as_index=False).agg(
        forward_mean=('forward','mean'), symmetric_mean=('symmetric','mean'))
    tm['sign_agreement'] = np.sign(tm.forward_mean) == np.sign(tm.symmetric_mean)
    tm['symmetric_minus_forward'] = tm.symmetric_mean - tm.forward_mean
    tm.to_csv(args.output_dir / 'forward_vs_symmetrized_task_model.csv', index=False)
    sym_headline = {
        'task_model_cells': int(len(tm)),
        'sign_agreement_cells': int(tm.sign_agreement.sum()),
        'sign_agreement_fraction': float(tm.sign_agreement.mean()),
        'discordant_cells': tm.loc[~tm.sign_agreement, ['dataset','task_id','model','forward_mean','symmetric_mean']].to_dict(orient='records'),
    }
    (args.output_dir / 'forward_vs_symmetrized_headline.json').write_text(json.dumps(sym_headline, indent=2), encoding='utf-8')

    # Prediction performance receives a regular table because it remains in the title.
    pred_cols = ['id_auroc','target_auroc','delta_auroc','id_log_loss','target_log_loss','delta_log_loss','id_brier','target_brier','delta_brier']
    split_pred = pair.groupby(['dataset','task_id','model','split_index'], as_index=False)[pred_cols].mean()
    task_pred = split_pred.groupby(['dataset','task_id','model'], as_index=False)[pred_cols].mean()
    pred = task_pred.groupby(['dataset','model'], as_index=False).agg(**{f'mean_{c}':(c,'mean') for c in pred_cols}, n_elementary_tasks=('task_id','size'))
    pred.to_csv(args.output_dir / 'predictive_performance_dataset_model.csv', index=False)

    # Amplitude-stratum heterogeneity relative to the full task-model means.
    amp_path = root / 'outputs' / 'l1_revision' / 'l1_amplitude_strata_task_summary.csv'
    amp_summary = None
    if amp_path.exists():
        amp = pd.read_csv(amp_path)
        # Reviewer-defined heterogeneity check compares stratum medians with the full
        # task-model median over the same 20 computational repetitions. This is a
        # sensitivity diagnostic, not the mean-based primary endpoint.
        full_tm = pair.groupby(['dataset','task_id','model'], as_index=False).agg(
            full_task_model_median=('calibrated_excess_u2','median')
        )
        full_tm['amp_task_key'] = [task_key_for_amp(ds, task_id) for ds, task_id in zip(full_tm.dataset, full_tm.task_id)]
        domain_map = {'Heart Disease':'heart','Student Performance':'student','OULAD':'oulad','ACSIncome':'acs'}
        full_tm['domain'] = full_tm.dataset.map(domain_map)
        joined = amp.merge(full_tm[['domain','amp_task_key','model','full_task_model_median']], left_on=['domain','task_id','model'], right_on=['domain','amp_task_key','model'], how='left', validate='many_to_one')
        joined['sign_agreement_with_full'] = np.sign(joined.excess_u2_median) == np.sign(joined.full_task_model_median)
        amp_summary = joined.groupby('amplitude_stratum', as_index=False).agg(
            cells=('sign_agreement_with_full','size'),
            sign_agreement_fraction=('sign_agreement_with_full','mean'),
            median_stratum_excess_u2=('excess_u2_median','median'),
            mean_stratum_excess_u2=('excess_u2_median','mean'),
        )
        amp_summary.to_csv(args.output_dir / 'amplitude_quartile_heterogeneity_summary.csv', index=False)
        joined.to_csv(args.output_dir / 'amplitude_quartile_task_model.csv', index=False)

    # ACS PWGTP sensitivity on one representative refit from each source split.
    acs_dir = root / 'outputs' / 'acs_weighted_sensitivity' / 'pair_runs'
    rep_pairs = [0,4,8,12,16]
    available = [acs_dir / f'pair_{i:02d}_rows.csv' for i in rep_pairs]
    acs_weight = None
    if all(p.exists() for p in available):
        aw = pd.concat([pd.read_csv(p) for p in available], ignore_index=True)
        aw = aw[aw.target_definition.eq('nominal_50k')].copy()
        aw = aw[['refit_pair','split_index','model','calibrated_excess_u2','pwgpt_weighted_shift_energy_u2','pwgpt_weighted_refit_null_energy_u2','pwgpt_weighted_excess_u2','pwgpt_weighting_note']]
        aw['sign_agreement'] = np.sign(aw.calibrated_excess_u2) == np.sign(aw.pwgpt_weighted_excess_u2)
        aw.to_csv(args.output_dir / 'acs_pwgpt_weighted_sensitivity_split_representatives.csv', index=False)
        acs_weight = aw.groupby('model', as_index=False).agg(
            splits=('split_index','size'),
            unweighted_mean_excess_u2=('calibrated_excess_u2','mean'),
            pwgpt_weighted_mean_excess_u2=('pwgpt_weighted_excess_u2','mean'),
            sign_agreement_fraction=('sign_agreement','mean'),
        )
        acs_weight.to_csv(args.output_dir / 'acs_pwgpt_weighted_sensitivity_summary.csv', index=False)

    # Controlled simulations retained, now explicitly supporting scale/small-denominator limitations.
    metric_sim = metric_scale_simulation()
    metric_sim.to_csv(args.output_dir / 'metric_dimension_sparsity_simulation.csv', index=False)
    denom_sim = small_denominator_simulation()
    denom_sim.to_csv(args.output_dir / 'small_l1_denominator_simulation.csv', index=False)

    # Bibliometric audit is optional and separate from scientific finalization.
    scopus_summary = None
    if args.scopus is not None:
        _, scopus_summary, term_df = scopus_audit(args.scopus)
        (args.output_dir / 'scopus_audit_summary.json').write_text(json.dumps(scopus_summary, indent=2), encoding='utf-8')
        term_df.to_csv(args.output_dir / 'scopus_title_term_audit.csv', index=False)

    headline = {
        "version": "1.0.0",
        'title_retained_by_author_instruction': True,
        'dataset_specific_primary_results': dataset[['dataset','dataset_mean_raw_excess_u2','positive_elementary_tasks','n_elementary_tasks']].to_dict(orient='records'),
        'equal_dataset_weighted_raw_scale_descriptive_index': equal_dataset_index,
        'equal_dataset_index_is_common_scale_effect': False,
        'elementary_task_positive_count': int(task_out.positive_mean.sum()),
        'elementary_task_total': int(len(task_out)),
        'dynamic_oulad_horizon_sign_changes': int(wide.sign_change_14_to_56.sum()),
        'fixed_cohort_oulad_horizon_sign_changes': int(fixed_family.fixed_sign_change_14_to_56.sum()) if fixed_family is not None else None,
        'forward_symmetric_sign_agreement': sym_headline,
        'scopus_audit_included': scopus_summary is not None,
    }
    (args.output_dir / 'review_revision_headline.json').write_text(json.dumps(headline, indent=2), encoding='utf-8')

    # Figures: baseline decomposition, dynamic vs fixed OULAD, model forward vs symmetric.
    fig, ax = plt.subplots(figsize=(7.2,4.8))
    x = np.arange(len(decomp))
    ax.plot(x, decomp['mean_shift_energy_u2'], marker='o', label='Source-target shift')
    ax.plot(x, decomp['mean_same_model_sample_null_u2'], marker='s', label='Same-model sample null')
    ax.plot(x, decomp['mean_refit_sample_null_u2'], marker='^', label='Refit null')
    ax.set_xticks(x, decomp['dataset'], rotation=20, ha='right')
    ax.set_ylabel('Raw squared-energy U-statistic')
    ax.set_title('Dataset-specific shift and baseline components')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.figure_dir/'Figure_7_baseline_decomposition.pdf')
    fig.savefig(args.figure_dir/'Figure_7_baseline_decomposition.png', dpi=220)
    plt.close(fig)

    if fixed_summary is not None:
        fig, ax = plt.subplots(figsize=(7.4,4.8))
        labels = fixed_summary['code_module'].astype(str)+'-'+fixed_summary['period'].astype(str)+' d'+fixed_summary['horizon_day'].astype(str)
        ax.scatter(fixed_summary['dynamic_riskset_mean_excess_u2'], fixed_summary['fixed_cohort_mean_excess_u2'])
        lo = float(min(fixed_summary['dynamic_riskset_mean_excess_u2'].min(), fixed_summary['fixed_cohort_mean_excess_u2'].min()))
        hi = float(max(fixed_summary['dynamic_riskset_mean_excess_u2'].max(), fixed_summary['fixed_cohort_mean_excess_u2'].max()))
        ax.plot([lo,hi],[lo,hi],linewidth=0.8)
        ax.axhline(0,linewidth=0.7); ax.axvline(0,linewidth=0.7)
        ax.set_xlabel('Dynamic risk-set mean excess')
        ax.set_ylabel('Fixed-cohort mean excess')
        ax.set_title('OULAD fixed-cohort horizon sensitivity')
        fig.tight_layout()
        fig.savefig(args.figure_dir/'Figure_8_oulad_fixed_cohort_sensitivity.pdf')
        fig.savefig(args.figure_dir/'Figure_8_oulad_fixed_cohort_sensitivity.png',dpi=220)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.3,5.2))
    ax.scatter(tm.forward_mean, tm.symmetric_mean)
    lo=float(min(tm.forward_mean.min(),tm.symmetric_mean.min())); hi=float(max(tm.forward_mean.max(),tm.symmetric_mean.max()))
    ax.plot([lo,hi],[lo,hi],linewidth=0.8)
    ax.axhline(0,linewidth=0.7); ax.axvline(0,linewidth=0.7)
    ax.set_xlabel('Forward matched-null excess')
    ax.set_ylabel('Symmetrized matched-null excess')
    ax.set_title('Forward versus symmetrized task-model results')
    fig.tight_layout()
    fig.savefig(args.figure_dir/'Figure_9_forward_symmetry.pdf')
    fig.savefig(args.figure_dir/'Figure_9_forward_symmetry.png',dpi=220)
    plt.close(fig)

    print(json.dumps(headline, indent=2))


if __name__ == '__main__':
    main()
