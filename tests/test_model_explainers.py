from __future__ import annotations

import numpy as np
import pandas as pd

from xaishiftbench.acs_models import FEATURES, fit_acs_model


def _acs_fixture(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    frame = pd.DataFrame(
        {
            "AGEP": rng.integers(18, 80, size=n),
            "COW": rng.integers(1, 9, size=n),
            "SCHL": rng.integers(10, 25, size=n),
            "MAR": rng.integers(1, 6, size=n),
            "OCCP": rng.integers(10, 80, size=n),
            "POBP": rng.integers(1, 60, size=n),
            "RELP_HARM": rng.choice(["reference_person", "child", "spouse_or_partner"], size=n),
            "WKHP": rng.integers(1, 70, size=n),
            "SEX": rng.integers(1, 3, size=n),
            "RAC1P": rng.integers(1, 5, size=n),
        }
    )
    frame["TARGET_NOMINAL_50K"] = (
        frame["AGEP"] + frame["WKHP"] + frame["SCHL"] > 85
    ).astype(int)
    return frame


def test_lightgbm_wrapper_booster_parity_and_additivity() -> None:
    frame = _acs_fixture()
    model = fit_acs_model(
        "lightgbm", frame, frame.TARGET_NOMINAL_50K.to_numpy(int), seed=11
    )
    z = model.transform(frame)
    wrapper = model.predict_proba(frame)
    booster = np.asarray(model.estimator.booster_.predict(z), float)
    assert np.allclose(wrapper, booster, atol=1e-12, rtol=1e-12)
    encoded, _, base = model.explain(frame.iloc[:40])
    reconstructed = base + encoded.sum(axis=1)
    assert np.max(np.abs(model.raw_score(frame.iloc[:40]) - reconstructed)) < 1e-8


def test_logistic_additivity() -> None:
    frame = _acs_fixture()
    model = fit_acs_model(
        "logistic", frame, frame.TARGET_NOMINAL_50K.to_numpy(int), seed=12
    )
    encoded, _, base = model.explain(frame.iloc[:40])
    reconstructed = base + encoded.sum(axis=1)
    assert np.max(np.abs(model.raw_score(frame.iloc[:40]) - reconstructed)) < 1e-10
