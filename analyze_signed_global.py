"""Signed global attribution-profile companion analysis for the release.

This descriptive companion preserves attribution signs and complements, rather
than replaces, the local distributional estimand.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs' / 'global_signed_profiles'
OUT.mkdir(parents=True, exist_ok=True)


def vector_rows(frame, *, dataset, task_cols, family_cols, pair_col,
                id_col='id_mean_signed', target_col='target_mean_signed'):
    d = frame.copy()
    if id_col not in d:
        id_col = 'mean_signed_explanation_id'
        target_col = 'mean_signed_explanation_target'
    d['dataset'] = dataset
    d['task_id'] = d[task_cols].astype(str).agg('|'.join, axis=1) if task_cols else '2018_to_2024'
    d['family_id'] = d[family_cols].astype(str).agg('|'.join, axis=1) if family_cols else '2018_to_2024'
    d['pair_id'] = d[pair_col].astype(int)
    group = ['dataset','task_id','family_id','model','split_index','refit_index','pair_id']
    rows=[]
    for keys,g in d.groupby(group, sort=True):
        delta = pd.to_numeric(g[target_col], errors='coerce').to_numpy(float) - pd.to_numeric(g[id_col], errors='coerce').to_numpy(float)
        rows.append(dict(zip(group,keys)) | {
            'signed_global_l2': float(np.linalg.norm(delta)),
            'signed_global_l1': float(np.abs(delta).sum()),
            'n_semantic_features': int(len(delta)),
        })
    return pd.DataFrame(rows)


def main():
    heart = vector_rows(pd.read_csv(ROOT/'outputs/heart/heart_feature_importance.csv'),
                        dataset='Heart Disease', task_cols=['target_site'], family_cols=['target_site'], pair_col='pair')
    student = vector_rows(pd.read_csv(ROOT/'outputs/student/student_institution_feature_profiles.csv'),
                          dataset='Student Performance', task_cols=['direction','representation'], family_cols=['direction'], pair_col='pair')
    oulad = vector_rows(pd.read_csv(ROOT/'outputs/oulad/oulad_temporal_feature_profiles.csv'),
                        dataset='OULAD', task_cols=['code_module','period','horizon_day'], family_cols=['code_module','period'], pair_col='refit_pair')
    acs = vector_rows(pd.read_csv(ROOT/'outputs/acs/acs_temporal_feature_profiles.csv'),
                      dataset='ACSIncome', task_cols=[], family_cols=[], pair_col='refit_pair')
    pair = pd.concat([heart,student,oulad,acs], ignore_index=True)
    pair.to_csv(OUT/'signed_global_profile_pair_values.csv', index=False)
    split = pair.groupby(['dataset','task_id','family_id','split_index'], as_index=False).agg(
        mean_signed_global_l2=('signed_global_l2','mean'),
        mean_signed_global_l1=('signed_global_l1','mean'),
        n_pair_rows=('signed_global_l2','size'))
    split.to_csv(OUT/'signed_global_profile_split_values.csv', index=False)
    strict = split.groupby(['dataset','family_id'], as_index=False).agg(
        mean_signed_global_l2=('mean_signed_global_l2','mean'),
        min_split_signed_global_l2=('mean_signed_global_l2','min'),
        max_split_signed_global_l2=('mean_signed_global_l2','max'),
        mean_signed_global_l1=('mean_signed_global_l1','mean'),
        n_splits=('split_index','nunique'))
    strict.to_csv(OUT/'signed_global_profile_strict_summary.csv', index=False)
    print(strict.to_string(index=False))

if __name__ == '__main__':
    main()
