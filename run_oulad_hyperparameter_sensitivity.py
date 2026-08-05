"""Representative OULAD model-complexity sensitivity for the release."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xaishiftbench.calibrated_shift import calibrated_attribution_shift
from xaishiftbench.oulad_models import fit_oulad_model
from xaishiftbench.oulad_temporal_pilot import CAT, BASE_NUM, _bootstrap, _bundle, _source_only_feature_space

ROOT=Path(__file__).resolve().parent
INPUT=ROOT/'outputs/oulad_prepared/oulad_score_free_horizons.csv.gz'
OUT=ROOT/'outputs/model_sensitivity'; OUT.mkdir(parents=True,exist_ok=True)
BASE_SEED=2026073131
CONFIGS={
 'logistic_C025':('logistic',{'C':0.25}),
 'logistic_C1':('logistic',{'C':1.0}),
 'logistic_C4':('logistic',{'C':4.0}),
 'lightgbm_low':('lightgbm',{'n_estimators':100,'num_leaves':7,'min_child_samples':24}),
 'lightgbm_base':('lightgbm',{'n_estimators':180,'num_leaves':15,'min_child_samples':12}),
 'lightgbm_high':('lightgbm',{'n_estimators':260,'num_leaves':31,'min_child_samples':8}),
}

def main():
 d=pd.read_csv(INPUT)
 source=d[(d.code_module=='BBB')&(d.code_presentation=='2013J')&(d.horizon_day==56)].copy()
 target=d[(d.code_module=='BBB')&(d.code_presentation=='2014J')&(d.horizon_day==56)].copy()
 click_all=sorted(c for c in d if c.startswith('clicks_') and c not in BASE_NUM)
 rows=[]
 for split in range(5):
  tr_idx,id_idx=train_test_split(np.arange(len(source)),test_size=.30,random_state=BASE_SEED+split*1000,stratify=source.target_unsuccessful)
  pool=source.iloc[tr_idx].reset_index(drop=True); id_eval=source.iloc[id_idx].reset_index(drop=True)
  click_cols,target_only=_source_only_feature_space(pool,target,click_all)
  numeric=BASE_NUM+click_cols; features=numeric+CAT
  yi=id_eval.target_unsuccessful.to_numpy(int); yt=target.target_unsuccessful.to_numpy(int)
  for ci,(label,(model_name,overrides)) in enumerate(CONFIGS.items()):
   seed=BASE_SEED+split*10000+ci*100
   ta=_bootstrap(pool,seed+1); tb=_bootstrap(pool,seed+2)
   ma=fit_oulad_model(model_name,ta,ta.target_unsuccessful.to_numpy(int),features,numeric,CAT,seed+3,overrides)
   mb=fit_oulad_model(model_name,tb,tb.target_unsuccessful.to_numpy(int),features,numeric,CAT,seed+4,overrides)
   ia=_bundle(ma,id_eval); ta_b=_bundle(ma,target); ib=_bundle(mb,id_eval); tb_b=_bundle(mb,target)
   cal=calibrated_attribution_shift(ia['normalized'],ta_b['normalized'],ib['normalized'],yi,yt,normalized_target_b=tb_b['normalized'],seed=seed+41,cap=160,class_cap=80)
   config={'model':model_name,**overrides}; conf=json.dumps(config,sort_keys=True)
   rows.append({'split_index':split,'configuration':label,'model':model_name,'config_json':conf,'config_sha256':hashlib.sha256(conf.encode()).hexdigest(),'calibrated_excess_u2':cal.excess_over_refit_sample_null_u2,'symmetrized_excess_u2':cal.symmetrized_excess_u2,'target_prevalence_weighted_class_excess_u2':cal.class_weighted_excess_u2,'max_additivity_error':max(ia['additivity_error'],ta_b['additivity_error'],ib['additivity_error'],tb_b['additivity_error']),'n_features':len(features),'n_target_only_features':len(target_only)})
 raw=pd.DataFrame(rows); raw.to_csv(OUT/'oulad_bbbj_day56_hyperparameter_sensitivity_splits.csv',index=False)
 summary=raw.groupby(['configuration','model'],as_index=False).agg(mean_excess_u2=('calibrated_excess_u2','mean'),min_split_excess_u2=('calibrated_excess_u2','min'),max_split_excess_u2=('calibrated_excess_u2','max'),positive_splits=('calibrated_excess_u2',lambda x:int((x>0).sum())),mean_symmetrized_excess_u2=('symmetrized_excess_u2','mean'),max_additivity_error=('max_additivity_error','max'))
 summary.to_csv(OUT/'oulad_bbbj_day56_hyperparameter_sensitivity_summary.csv',index=False)
 print(summary.to_string(index=False))
if __name__=='__main__': main()
