"""XAIShiftBench metrics, dataset adapters, and pilot utilities."""

from .metrics import (
    jensen_shannon_distance,
    kendall_tau_b,
    energy_u_statistic_squared,
    energy_v_statistic_squared,
    attribution_l1_diagnostics,
    multivariate_energy_distance,
    normalize_signed_rows,
    normalize_signed_rows_with_mask,
    signed_total_variation,
    top_k_jaccard,
    weighted_sign_consistency,
)
from .null_calibration import (
    empirical_exceedance_fraction,
    empirical_upper_tail_p,
    paired_excess,
    standardized_excess,
)

__all__ = [
    "jensen_shannon_distance",
    "kendall_tau_b",
    "energy_u_statistic_squared",
    "energy_v_statistic_squared",
    "attribution_l1_diagnostics",
    "multivariate_energy_distance",
    "normalize_signed_rows",
    "normalize_signed_rows_with_mask",
    "signed_total_variation",
    "top_k_jaccard",
    "weighted_sign_consistency",
    "empirical_exceedance_fraction",
    "empirical_upper_tail_p",
    "paired_excess",
    "standardized_excess",
]

__version__ = "1.0.0"
