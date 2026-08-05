from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from xaishiftbench.heart_pilot import run_heart_cross_site_pilot
from xaishiftbench.student_institution_pilot import run_student_institution_pilot
from xaishiftbench.oulad_temporal_pilot import run_pair as run_oulad_pair
from xaishiftbench.acs_temporal_pilot import run_pair as run_acs_pair


def _task_id(domain: str, frame: pd.DataFrame) -> pd.Series:
    if domain == 'heart':
        return frame['target_site'].astype(str)
    if domain == 'student':
        return frame['direction'].astype(str) + '|' + frame['representation'].astype(str)
    if domain == 'oulad':
        return frame['code_module'].astype(str) + '-' + frame['period'].astype(str) + '|day' + frame['horizon_day'].astype(int).astype(str)
    if domain == 'acs':
        return pd.Series(['ACSIncome_2018_to_2024'] * len(frame), index=frame.index)
    raise ValueError(domain)


def _compact(domain: str, rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    if domain == 'acs':
        rows = rows.loc[rows['target_definition'] == 'nominal_50k'].copy()
    rows.insert(0, 'domain', domain)
    rows.insert(1, 'task_id', _task_id(domain, rows))
    exp = [c for c in rows.columns if c.startswith('expsamp_')]
    ids = [c for c in [
        'domain','task_id','model','pair','refit_pair','split_index','refit_index',
        'target_site','direction','representation','code_module','period','horizon_day',
        'target_definition','n_explanation','calibrated_excess_u2'
    ] if c in rows.columns]
    out = rows[ids + exp].copy()
    if 'expsamp_full_excess_u2' not in out:
        raise AssertionError(f'{domain}: explanation-sampling columns missing')
    diff = np.abs(pd.to_numeric(out['calibrated_excess_u2']) - pd.to_numeric(out['expsamp_full_excess_u2']))
    if not np.all(diff.fillna(0).to_numpy() <= 1e-12):
        raise AssertionError(f'{domain}: full-sample endpoint does not reproduce primary endpoint')
    return out


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--repeats', type=int, default=50)
    ap.add_argument('--pairs', default='0,4,8,12,16')
    ns=ap.parse_args()
    root=ns.root.resolve(); pairs=[int(x) for x in ns.pairs.split(',') if x.strip()]
    outdir=root/'outputs'/'explanation_sampling_sensitivity'; outdir.mkdir(parents=True, exist_ok=True)
    t0=perf_counter(); frames=[]
    for pair in pairs:
        print(f'pair {pair}: heart', flush=True)
        h,_,_=run_heart_cross_site_pilot(
            root/'data'/'heart_disease'/'raw', n_pairs=1, pair_start=pair,
            total_pairs=20, n_splits=5, explain_cap=160, sampling_repeats=ns.repeats,
        )
        frames.append(_compact('heart',h))

        print(f'pair {pair}: student', flush=True)
        s,_,_=run_student_institution_pilot(
            root/'data'/'student_performance'/'raw'/'data.csv', n_pairs=1, pair_start=pair,
            total_pairs=20, n_splits=5, explain_cap=160, sampling_repeats=ns.repeats,
        )
        frames.append(_compact('student',s))

        print(f'pair {pair}: oulad', flush=True)
        o,_,_=run_oulad_pair(
            root/'outputs'/'oulad_prepared'/'oulad_score_free_horizons.csv.gz',
            root/'outputs'/'oulad_prepared'/'oulad_temporal_pairs.csv', pair,
            explain_cap=160, n_splits=5, refits_per_split=4, sampling_repeats=ns.repeats,
        )
        frames.append(_compact('oulad',o))

        print(f'pair {pair}: acs', flush=True)
        a,_,_,_=run_acs_pair(root,pair,explain_cap=256,n_splits=5,refits_per_split=4,sampling_repeats=ns.repeats)
        frames.append(_compact('acs',a))

    runs=pd.concat(frames,ignore_index=True,sort=False)
    runs.to_csv(outdir/'explanation_sampling_run_level.csv',index=False)

    group=['domain','task_id','model']
    records=[]
    for key,g in runs.groupby(group,dropna=False,sort=True):
        r=dict(zip(group,key)); r['n_split_runs']=int(len(g)); r['n_explanation_median']=float(g['n_explanation'].median())
        r['full_excess_median']=float(g['expsamp_full_excess_u2'].median())
        for f in ('f050','f075'):
            med=f'expsamp_{f}_median_excess_u2'; pos=f'expsamp_{f}_positive_fraction'; ncol=f'expsamp_{f}_n'
            r[f'{f}_subsample_n_median']=float(g[ncol].median())
            r[f'{f}_median_of_draw_medians']=float(g[med].median())
            r[f'{f}_median_abs_deviation_from_full']=float(np.median(np.abs(g[med]-g['expsamp_full_excess_u2'])))
            signs=(np.sign(g[med].to_numpy())==np.sign(g['expsamp_full_excess_u2'].to_numpy()))
            r[f'{f}_split_sign_agreement']=float(np.mean(signs))
            r[f'{f}_median_positive_draw_fraction']=float(g[pos].median())
        records.append(r)
    cells=pd.DataFrame(records)
    cells.to_csv(outdir/'explanation_sampling_task_model_summary.csv',index=False)

    overview={'version':'1.0.0','method':'repeated stratified subsampling without replacement within the release-defined matched explanation samples; models and matched full samples fixed within each run',
              'pairs':pairs,'repeats_per_fraction_per_run':ns.repeats,'n_run_rows':int(len(runs)),'n_task_model_cells':int(len(cells)),'fractions':[0.5,0.75]}
    for f in ('f050','f075'):
        x=cells['full_excess_median'].to_numpy(float); y=cells[f'{f}_median_of_draw_medians'].to_numpy(float)
        rho=spearmanr(x,y).statistic
        overview[f'{f}_task_model_sign_agreement']=float(np.mean(np.sign(x)==np.sign(y)))
        overview[f'{f}_task_model_spearman']=float(rho)
        overview[f'{f}_median_absolute_cell_deviation']=float(np.median(np.abs(x-y)))
        small=cells['n_explanation_median']<64
        if small.any():
            overview[f'{f}_small_n_lt64_sign_agreement']=float(np.mean(np.sign(x[small])==np.sign(y[small])))
            overview[f'{f}_small_n_lt64_cells']=int(small.sum())
    (outdir/'explanation_sampling_overview.json').write_text(json.dumps(overview,indent=2),encoding='utf-8')
    lines=['# Explanation-sampling sensitivity the release','',
           'The analysis holds the fitted model pair and the release-defined matched explanation sample fixed, then repeatedly draws stratified 50% and 75% subsamples without replacement. It therefore isolates finite explanation-sample sensitivity from model-refit variability.','']
    for k,v in overview.items(): lines.append(f'- {k}: {v}')
    (outdir/'EXPLANATION_SAMPLING_SENSITIVITY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({**overview,'runtime_seconds':perf_counter()-t0},indent=2),flush=True)

if __name__=='__main__': main()
