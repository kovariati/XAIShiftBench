from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from xaishiftbench.oulad_models import oulad_model_config
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'/'multidomain'
checks=[]
def add(name,ok,detail): checks.append({'check':name,'pass':bool(ok),'detail':detail})
strict=pd.read_csv(OUT/'strict_contrast_summary_16.csv')
splits=pd.read_csv(OUT/'strict_contrast_split_level.csv')
pair=pd.read_csv(OUT/'pair_level_real_deployment.csv')
add('strict_count',len(strict)==16,len(strict))
add('positive_strict',int((strict.mean_calibrated_excess_u2>0).sum())==13,int((strict.mean_calibrated_excess_u2>0).sum()))
add('mean',np.isclose(strict.mean_calibrated_excess_u2.mean(),0.018966409718452475),strict.mean_calibrated_excess_u2.mean())
add('median',np.isclose(strict.mean_calibrated_excess_u2.median(),0.00971307538089863),strict.mean_calibrated_excess_u2.median())
add('split_signs',((strict.positive_split_fraction==1).sum()==12 and (strict.positive_split_fraction==0).sum()==3),{'all_positive':int((strict.positive_split_fraction==1).sum()),'all_negative':int((strict.positive_split_fraction==0).sum())})
add('one_crossing',int(((strict.min_split_mean_u2<=0)&(strict.max_split_mean_u2>=0)).sum())==1,int(((strict.min_split_mean_u2<=0)&(strict.max_split_mean_u2>=0)).sum()))
# Recompute every strict range from final split means.
ok=True
for _,r in strict.iterrows():
 g=splits[(splits.dataset==r.dataset)&(splits.family_id==r.family_id)]
 ok &= len(g)==5 and np.isclose(g.mean_calibrated_excess_u2.min(),r.min_split_mean_u2) and np.isclose(g.mean_calibrated_excess_u2.max(),r.max_split_mean_u2) and np.isclose((g.mean_calibrated_excess_u2>0).mean(),r.positive_split_fraction)
add('strict_ranges_from_final_split_means',ok,'all 16 recomputed')
# Recompute target prevalence weighting for all pair rows.
expected=(1-pair.target_prevalence)*(pair.class0_shift_u2-pair.class0_refit_null_u2)+pair.target_prevalence*(pair.class1_shift_u2-pair.class1_refit_null_u2)
add('target_prevalence_weighting',np.allclose(expected,pair.target_prevalence_weighted_class_excess_u2,equal_nan=True),'all pair rows')
add('symmetry_mean',np.isclose(strict.mean_symmetrized_calibrated_excess_u2.mean(),0.019724167806206894),strict.mean_symmetrized_calibrated_excess_u2.mean())
add('symmetry_sign_agreement',np.all(np.sign(strict.mean_symmetrized_calibrated_excess_u2)==np.sign(strict.mean_calibrated_excess_u2)),'16/16')
cfg=oulad_model_config('lightgbm')
add('oulad_config',cfg['n_estimators']==180 and cfg['learning_rate']==.035 and cfg['num_leaves']==15 and cfg['min_child_samples']==12,cfg)
add('max_additivity',strict.max_additivity_error.max()<1e-7,strict.max_additivity_error.max())
disallowed=[c for c in strict.columns if c.startswith('mean_') and (c.endswith('_p') or 'p_value' in c)]
add('no_lower_level_p_averaging',not disallowed,disallowed)
payload={"version": "1.0.0",'all_pass':all(c['pass'] for c in checks),'checks':checks}
(ROOT/'CLAIM_CHECK.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
print(json.dumps(payload,indent=2))
if not payload['all_pass']: raise SystemExit(1)
