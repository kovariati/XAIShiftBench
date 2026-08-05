import numpy as np
import pytest

from xaishiftbench.metrics import repeated_explanation_subsample_sensitivity


def _fixture(seed=1):
    rng=np.random.default_rng(seed)
    n=40; p=6
    ys=np.array([0]*24+[1]*16)
    yn=np.array([0]*20+[1]*20)
    yt=np.array([0]*20+[1]*20)
    a=rng.normal(size=(n,p)); a/=np.abs(a).sum(axis=1,keepdims=True)
    b=a+rng.normal(scale=.03,size=(n,p)); b/=np.abs(b).sum(axis=1,keepdims=True)
    t=a+rng.normal(scale=.08,size=(n,p)); t/=np.abs(t).sum(axis=1,keepdims=True)
    return a,b,t,ys,yn,yt


def test_repeated_explanation_subsample_sensitivity_is_deterministic_and_finite():
    args=_fixture()
    x=repeated_explanation_subsample_sensitivity(*args,seed=123,repeats=12)
    y=repeated_explanation_subsample_sensitivity(*args,seed=123,repeats=12)
    assert x==y
    assert x['expsamp_full_n']==40
    assert x['expsamp_f050_n']==20
    assert x['expsamp_f075_n']==30
    for k,v in x.items():
        if isinstance(v,float): assert np.isfinite(v)


def test_repeated_explanation_subsample_sensitivity_validates_fraction():
    args=_fixture()
    with pytest.raises(ValueError):
        repeated_explanation_subsample_sensitivity(*args,seed=1,repeats=4,fractions=(1.0,))
