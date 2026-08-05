import numpy as np
import pytest

from xaishiftbench.metrics import (
    jensen_shannon_distance,
    kendall_tau_b,
    multivariate_energy_distance,
    normalize_signed_rows,
    signed_total_variation,
    top_k_jaccard,
    weighted_sign_consistency,
)
from xaishiftbench.null_calibration import empirical_upper_tail_p, standardized_excess


def test_identical_profiles_are_stable():
    a = np.array([0.5, 0.3, 0.2])
    assert kendall_tau_b(a, a) == pytest.approx(1.0)
    assert top_k_jaccard(a, a, 2) == pytest.approx(1.0)
    assert jensen_shannon_distance(a, a) == pytest.approx(0.0)
    assert weighted_sign_consistency(a, a) == pytest.approx(1.0)


def test_signed_normalization_and_tv():
    values = np.array([[1.0, -1.0, 2.0], [0.0, 0.0, 0.0]])
    normalized = normalize_signed_rows(values)
    assert np.sum(np.abs(normalized[0])) == pytest.approx(1.0)
    assert np.all(normalized[1] == 0.0)
    assert signed_total_variation(normalized[0], normalized[0]) == pytest.approx(0.0)


def test_energy_distance_identity_and_separation():
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    y = x.copy()
    z = x + 5.0
    assert multivariate_energy_distance(x, y) == pytest.approx(0.0)
    assert multivariate_energy_distance(x, z) > 0.0


def test_null_calibration():
    null = np.arange(1.0, 11.0)
    assert standardized_excess(5.5, null) == pytest.approx(0.0)
    assert empirical_upper_tail_p(20.0, null) == pytest.approx(1.0 / 11.0)
