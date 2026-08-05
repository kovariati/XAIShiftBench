from __future__ import annotations

import pandas as pd

from prepare_acs_income import adjust_pincp


def test_adjinc_manual_example() -> None:
    raw = pd.Series([50_000.0, 100_000.0])
    adj = pd.Series([1_015_250, 1_000_000])
    adjusted = adjust_pincp(raw, adj)
    assert adjusted.iloc[0] == 50_762.5
    assert adjusted.iloc[1] == 100_000.0
