from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from xaishiftbench.datasets.south_german_credit import FEATURES, NUMERIC_FEATURES
import xaishiftbench.cross_explainer_audit as cea


def _synthetic_credit(n=180, seed=1601):
    rng=np.random.default_rng(seed)
    d={}
    for f in FEATURES:
        if f in NUMERIC_FEATURES:
            if f=='age': d[f]=rng.integers(18,75,size=n)
            elif f=='duration': d[f]=rng.integers(4,60,size=n)
            else: d[f]=rng.integers(250,15000,size=n)
        else:
            d[f]=rng.integers(1,5,size=n)
    frame=pd.DataFrame(d)
    lin=(0.02*(frame['duration']-25)+0.00008*(frame['amount']-4000)-0.015*(frame['age']-35)
         +0.25*(frame['status']==1)-0.15*(frame['savings']==4))
    prob=1/(1+np.exp(-lin.to_numpy(float)))
    frame['target']=(rng.random(n)<prob).astype(int)
    # ensure both classes have enough observations for split and stratified sampling
    assert frame.target.value_counts().min() > 20
    frame['record_id']=[f'syn_{i:04d}' for i in range(n)]
    return frame


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_cross_explainer_small_end_to_end(monkeypatch):
    frame=_synthetic_credit()
    monkeypatch.setattr(cea,'load_south_german_credit',lambda _path: SimpleNamespace(frame=frame.copy()))
    obs, scen, feat, meta = cea.run_cross_explainer_audit(
        'synthetic-placeholder.asc', seed_base=1602, n_pairs=1, pair_start=0,
        models=('logistic',), background_size=6, explanation_sample_size=10,
        kernel_nsamples=40,
    )
    assert meta.n_pairs==1
    assert meta.n_model_fits==1
    assert len(obs)==50  # 5 target mechanisms x 10 observations
    assert len(scen)==10  # 5 mechanisms x 2 explainers
    assert len(feat)==200  # 5 mechanisms x 2 explainers x 20 semantic features
    assert scen['source_additivity_error'].max() < 1e-5
    assert scen['target_additivity_error'].max() < 1e-5
    clean=scen[scen.target_mechanism=='CLEAN']
    assert np.allclose(clean.mean_shift_stv,0.0,atol=1e-12)
    assert set(scen.explainer)=={'model_specific_shap','kernel_shap'}
