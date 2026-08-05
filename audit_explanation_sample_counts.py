from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

from xaishiftbench.calibrated_shift import matched_composition_indices
from xaishiftbench.datasets.heart_disease import load_heart_sites
from xaishiftbench.heart_pilot import _source_split as heart_split
from xaishiftbench.datasets.student_performance import load_student_performance
from xaishiftbench.student_institution_pilot import _source_split as student_split
from xaishiftbench.oulad_temporal_pilot import _source_split as oulad_split
from xaishiftbench.acs_temporal_pilot import load_acs_temporal_frames, _source_split as acs_split, TARGETS


def rec(domain, task, split_index, idx):
    return {
        'domain':domain,'task_id':task,'split_index':split_index,'n_explanation':idx.sample_size,
        'source_shift_class0_n':idx.sample_size-idx.source_positive_count,
        'source_shift_class1_n':idx.source_positive_count,
        'source_null_class0_n':idx.sample_size-idx.target_positive_count,
        'source_null_class1_n':idx.target_positive_count,
        'target_class0_n':idx.sample_size-idx.target_positive_count,
        'target_class1_n':idx.target_positive_count,
    }

def main(root: Path):
    rows=[]
    # Heart
    heart=load_heart_sites(root/'data/heart_disease/raw').frame
    seed_base=20260729
    for target_index,target_site in enumerate(sorted(heart.site.unique())):
        target=heart[heart.site==target_site].reset_index(drop=True); yt=target.target.to_numpy(int)
        source=heart[heart.site!=target_site].reset_index(drop=True)
        for split_index in range(5):
            split_seed=seed_base+target_index*1_000_000+split_index*10_000
            _,ide=heart_split(source,split_seed); ys=ide.target.to_numpy(int)
            idx=matched_composition_indices(ys,yt,cap=160,seed=0,min_per_class=3)
            rows.append(rec('heart',str(target_site),split_index,idx))
    # Student
    stu=load_student_performance(root/'data/student_performance/raw/data.csv').frame; seed_base=2026073004
    for direction_index,(src,tgt) in enumerate((('GP','MS'),('MS','GP'))):
        source=stu[stu.school==src].reset_index(drop=True); target=stu[stu.school==tgt].reset_index(drop=True); yt=target.target.to_numpy(int)
        for split_index in range(5):
            split_seed=seed_base+direction_index*1_000_000+split_index*10_000
            _,ide=student_split(source,split_seed); ys=ide.target.to_numpy(int)
            idx=matched_composition_indices(ys,yt,cap=160,seed=0,min_per_class=3)
            for representation in ('early','late'):
                rows.append(rec('student',f'{src}_to_{tgt}|{representation}',split_index,idx))
    # OULAD
    model=pd.read_csv(root/'outputs/oulad_prepared/oulad_score_free_horizons.csv.gz')
    pairs=pd.read_csv(root/'outputs/oulad_prepared/oulad_temporal_pairs.csv'); seed_base=2026073007
    for task_i,pair in pairs.iterrows():
        for horizon in (14,56):
            source=model[(model.code_module==pair.code_module)&(model.code_presentation==pair.source_presentation)&(model.horizon_day==horizon)].copy()
            target=model[(model.code_module==pair.code_module)&(model.code_presentation==pair.target_presentation)&(model.horizon_day==horizon)].copy(); yt=target.target_unsuccessful.to_numpy(int)
            for split_index in range(5):
                split_seed=seed_base+task_i*1_000_000+horizon*1_000+split_index*10_000
                _,ide=oulad_split(source,split_seed); ys=ide.target_unsuccessful.to_numpy(int)
                idx=matched_composition_indices(ys,yt,cap=160,seed=0,min_per_class=3)
                rows.append(rec('oulad',f'{pair.code_module}-{pair.period}|day{horizon}',split_index,idx))
    # ACS
    source,target=load_acs_temporal_frames(root); seed_base=2026073008
    for split_index in range(5):
        train,ide=acs_split(source,seed_base+split_index*10_000)
        for name,col in TARGETS.items():
            idx=matched_composition_indices(ide[col].to_numpy(int),target[col].to_numpy(int),cap=256,seed=0,min_per_class=3)
            task='ACSIncome_2018_to_2024' if name=='nominal_50k' else 'ACSIncome_2018_to_2024_real2018'
            r=rec('acs',task,split_index,idx); r['target_definition']=name; rows.append(r)
    df=pd.DataFrame(rows)
    out=root/'outputs/explanation_sample_counts'; out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/'explanation_sample_class_counts_by_split.csv',index=False)
    summary=df.groupby(['domain','task_id'],dropna=False).agg(
        n_explanation_min=('n_explanation','min'),n_explanation_max=('n_explanation','max'),
        source_shift_class0_min=('source_shift_class0_n','min'),source_shift_class0_max=('source_shift_class0_n','max'),
        source_shift_class1_min=('source_shift_class1_n','min'),source_shift_class1_max=('source_shift_class1_n','max'),
        target_class0_min=('target_class0_n','min'),target_class0_max=('target_class0_n','max'),
        target_class1_min=('target_class1_n','min'),target_class1_max=('target_class1_n','max'),
    ).reset_index()
    summary.to_csv(out/'explanation_sample_class_counts_task_summary.csv',index=False)
    primary_df=df.loc[~df['task_id'].astype(str).str.endswith('_real2018')].copy()
    primary_summary=summary.loc[~summary['task_id'].astype(str).str.endswith('_real2018')].copy()
    primary_summary.to_csv(out/'explanation_sample_class_counts_primary27.csv',index=False)
    payload={'version':'1.0.0','n_split_task_rows_all':len(df),'n_task_definitions_all':len(summary),'n_primary_task_definitions':len(primary_summary),'minimum_explanation_n_primary':int(primary_df.n_explanation.min()),'minimum_any_matched_class_count_primary':int(primary_df[['source_shift_class0_n','source_shift_class1_n','target_class0_n','target_class1_n']].min().min())}
    (out/'explanation_sample_counts_overview.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps(payload,indent=2))

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ns=ap.parse_args(); main(ns.root.resolve())
