from pathlib import Path
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xaishiftbench.datasets.heart_disease import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    load_heart_sites,
)
from xaishiftbench.null_calibration import empirical_exceedance_fraction, paired_excess
from xaishiftbench.semantic import semantic_map_from_preprocessor


def test_heart_integrity_from_environment():
    path = Path(os.environ.get('XAISHIFTBENCH_HEART_RAW', Path(__file__).resolve().parents[1] / 'data' / 'heart_disease' / 'raw'))
    required = [path / name for name in ('processed.cleveland.data', 'processed.hungarian.data', 'processed.switzerland.data', 'processed.va.data')]
    if not path.exists() or not all(p.exists() for p in required):
        pytest.skip('Raw Heart Disease files are not present in this environment.')
    dataset = load_heart_sites(path)
    assert len(dataset.frame) == 920
    assert int(dataset.frame['target'].sum()) == 509


def test_semantic_aggregation_preserves_sum():
    frame = pd.DataFrame(
        {
            'age': [30.0, 40.0, 50.0], 'trestbps': [110, 120, 130],
            'chol': [180, 200, 220], 'thalach': [170, 160, 150],
            'oldpeak': [0.0, 1.0, 2.0], 'sex': [0, 1, 1], 'cp': [1, 2, 3],
            'fbs': [0, 0, 1], 'restecg': [0, 1, 2], 'exang': [0, 1, 0],
            'slope': [1, 2, 3], 'ca': [0, 1, np.nan], 'thal': [3, 6, 7],
        }
    )
    num = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    cat = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    prep = ColumnTransformer([('num', num, NUMERIC_FEATURES), ('cat', cat, CATEGORICAL_FEATURES)],
                             sparse_threshold=0.0)
    encoded = prep.fit_transform(frame[FEATURES])
    mapping = semantic_map_from_preprocessor(prep, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    values = np.arange(encoded.size, dtype=float).reshape(encoded.shape)
    semantic = mapping.aggregate(values)
    assert semantic.shape == (3, 13)
    assert np.allclose(semantic.sum(axis=1), values.sum(axis=1))


def test_paired_null_helpers():
    shift = np.array([0.4, 0.5, 0.6, 0.7])
    null = np.array([0.1, 0.2, 0.2, 0.3])
    assert np.all(paired_excess(shift, null) > 0)
    assert empirical_exceedance_fraction(1.0, null) == pytest.approx(0.2)
