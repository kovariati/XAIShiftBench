from __future__ import annotations
import numpy as np
import pandas as pd
from xaishiftbench.aggregation import contrast_summary_from_splits, prepare_pair_rows, split_level_summary
from xaishiftbench.oulad_models import oulad_model_config

def _base_pair_rows():
    rows=[]
    for split, vals in enumerate([[1.,3.],[2.,4.],[3.,5.],[4.,6.],[5.,7.]]):
        for refit, value in enumerate(vals):
            rows.append({
                'dataset':'D','family_id':'F','task_id':'T','model':'M',
                'split_index':split,'refit_index':refit,
                'calibrated_excess_u2':value,'shift_energy_u2':value+.5,
                'same_model_sample_null_energy_u2':.1,'refit_sample_null_energy_u2':.5,
                'class0_shift_u2':.4,'class0_refit_null_u2':.1,
                'class1_shift_u2':1.1,'class1_refit_null_u2':.1,
                'target_prevalence':.8,'delta_auroc':.01,'paired_excess_js':-.1,
                'shift_rank_tau':.9,'source_mean_attribution_l1':2.,
                'target_mean_attribution_l1':4.,'max_additivity_error':1e-12,
                'symmetrized_calibrated_excess_u2':value+.25,'symmetry_gap_u2':-.5,
                'class_macro_excess_u2':.65,
            })
    return pd.DataFrame(rows)

def test_target_prevalence_not_pooled_prevalence():
    d=_base_pair_rows(); d['source_prevalence']=.2
    out=prepare_pair_rows(d)
    expected=(1-.8)*(.4-.1)+.8*(1.1-.1)
    pooled=(1-.5)*(.4-.1)+.5*(1.1-.1)
    assert np.allclose(out.target_prevalence_weighted_class_excess_u2, expected)
    assert not np.allclose(out.target_prevalence_weighted_class_excess_u2, pooled)
    assert (out.class_weighting_scheme=='target_prevalence').all()

def test_strict_range_is_range_of_final_split_means():
    d=prepare_pair_rows(_base_pair_rows())
    splits=split_level_summary(d,['dataset','family_id','split_index'])
    summary=contrast_summary_from_splits(splits,['dataset','family_id'])
    assert np.isclose(summary.loc[0,'min_split_mean_u2'],2.)
    assert np.isclose(summary.loc[0,'max_split_mean_u2'],6.)
    assert np.isclose(summary.loc[0,'positive_split_fraction'],1.)

def test_p_values_are_recomputed_from_splits_not_averaged():
    d=prepare_pair_rows(_base_pair_rows())
    splits=split_level_summary(d,['dataset','family_id','split_index'])
    splits['p_value_from_lower_level']=np.linspace(.01,.99,5)
    summary=contrast_summary_from_splits(splits,['dataset','family_id'])
    assert 'p_value_from_lower_level' not in summary.columns
    assert np.isclose(summary.loc[0,'sign_test_greater_p'],.03125)

def test_oulad_config_matches_executed_constants():
    l=oulad_model_config('lightgbm')
    assert l['n_estimators']==180 and l['num_leaves']==15 and l['min_child_samples']==12
    g=oulad_model_config('logistic')
    assert g['C']==1 and g['solver']=='lbfgs' and g['max_iter']==5000

def test_symmetrized_arithmetic_definition():
    forward=.03; reverse=.01
    assert np.isclose(.5*(forward+reverse),.02)
    assert np.isclose(forward-reverse,.02)

def test_normalized_composition_can_ignore_amplitude():
    a=np.array([1.,2.,-1.]); b=10*a
    assert np.allclose(a/np.abs(a).sum(), b/np.abs(b).sum())
    assert not np.isclose(np.abs(a).sum(),np.abs(b).sum())

def test_matched_array_calibration_target_weighting_and_symmetry():
    from xaishiftbench.calibrated_shift import calibrated_shift_from_matched_arrays
    rng=np.random.default_rng(13)
    n=10; p=3
    ys=np.array([0]*5+[1]*5)
    yn=np.array([0]*2+[1]*8)
    yt=yn.copy()
    sa=rng.normal(size=(n,p)); sn=rng.normal(size=(n,p)); tt=rng.normal(.15,size=(n,p))
    sb=sa+rng.normal(scale=.05,size=(n,p)); snb=sn+rng.normal(scale=.05,size=(n,p)); ttb=tt+rng.normal(scale=.05,size=(n,p))
    out=calibrated_shift_from_matched_arrays(sa,sn,tt,sb,snb,ys,yn,yt,target_b=ttb)
    assert out['class_weighting_scheme']=='target_prevalence'
    assert np.isclose(out['class_weighting_prevalence'],.8)
    expected=.2*(out['class0_shift_u2']-out['class0_refit_null_u2'])+.8*(out['class1_shift_u2']-out['class1_refit_null_u2'])
    assert np.isclose(out['class_weighted_excess_u2'],expected)
    assert np.isclose(out['symmetrized_excess_u2'],.5*(out['excess_u2']+out['reverse_excess_u2']))
