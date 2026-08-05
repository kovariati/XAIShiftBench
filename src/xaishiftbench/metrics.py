"""Explanation-stability metrics with explicit input validation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist, jensenshannon
from scipy.stats import kendalltau

_EPS = np.finfo(float).eps


def _as_1d(values: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _paired_1d(a: ArrayLike, b: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    aa = _as_1d(a, "a")
    bb = _as_1d(b, "b")
    if aa.shape != bb.shape:
        raise ValueError(f"a and b must have identical shapes; got {aa.shape} and {bb.shape}.")
    return aa, bb


def normalize_signed_rows(values: ArrayLike, eps: float = 1e-12) -> NDArray[np.float64]:
    """Normalize each signed attribution vector by its L1 magnitude.

    Zero rows remain zero. This representation preserves direction and relative
    contribution while removing explainer-specific scale.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"values must be two-dimensional; got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("values contains non-finite values.")
    denom = np.sum(np.abs(arr), axis=1, keepdims=True)
    return np.divide(arr, denom + eps, out=np.zeros_like(arr), where=denom > eps)



def attribution_l1_diagnostics(
    values: ArrayLike,
    thresholds: tuple[float, ...] = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4),
) -> dict[str, float | int]:
    """Summarize row-wise attribution amplitudes before normalization.

    These diagnostics expose the small-denominator regime in which row-wise
    L1 normalization can amplify numerical or explainer noise. They do not
    alter the primary representation and can be written directly to run-level
    audit tables.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"values must be two-dimensional; got shape {arr.shape}.")
    if arr.shape[0] == 0:
        raise ValueError("values must contain at least one row.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("values contains non-finite values.")
    l1 = np.sum(np.abs(arr), axis=1)
    q = np.quantile(l1, [0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0])
    out: dict[str, float | int] = {
        "n_rows": int(l1.size),
        "l1_min": float(q[0]),
        "l1_q01": float(q[1]),
        "l1_q05": float(q[2]),
        "l1_q25": float(q[3]),
        "l1_median": float(q[4]),
        "l1_q75": float(q[5]),
        "l1_q95": float(q[6]),
        "l1_q99": float(q[7]),
        "l1_max": float(q[8]),
        "l1_mean": float(np.mean(l1)),
        "zero_row_fraction": float(np.mean(l1 == 0.0)),
    }
    for threshold in thresholds:
        if threshold < 0 or not np.isfinite(threshold):
            raise ValueError("thresholds must be finite and non-negative.")
        key = f"fraction_l1_le_{threshold:.0e}"
        out[key] = float(np.mean(l1 <= threshold))
    return out


def normalize_signed_rows_with_mask(
    values: ArrayLike,
    min_l1: float = 1e-12,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Normalize signed rows and return the rows retained above ``min_l1``.

    This helper supports threshold-sensitivity analyses without silently
    changing the release-defined primary endpoint.
    """
    if min_l1 < 0 or not np.isfinite(min_l1):
        raise ValueError("min_l1 must be finite and non-negative.")
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"values must be two-dimensional; got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("values contains non-finite values.")
    l1 = np.sum(np.abs(arr), axis=1)
    keep = l1 > min_l1
    normalized = np.zeros_like(arr)
    if np.any(keep):
        normalized[keep] = arr[keep] / l1[keep, None]
    return normalized, keep



def prefixed_attribution_l1_diagnostics(
    values: ArrayLike,
    prefix: str,
    thresholds: tuple[float, ...] = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4),
) -> dict[str, float | int]:
    """Return L1 diagnostics with stable, table-safe prefixed keys."""
    if not prefix or not prefix.replace("_", "").isalnum():
        raise ValueError("prefix must be a non-empty alphanumeric/underscore label.")
    base = attribution_l1_diagnostics(values, thresholds=thresholds)
    return {f"{prefix}_{key}": value for key, value in base.items()}


def _threshold_key(value: float) -> str:
    return f"{value:.0e}".replace("-", "m").replace("+", "p")


def matched_l1_sensitivity(
    source_shift: ArrayLike,
    source_null: ArrayLike,
    target_shift: ArrayLike,
    *,
    thresholds: tuple[float, ...] = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4),
    amplitude_strata: int = 4,
) -> dict[str, float | int]:
    """Threshold and amplitude-stratified sensitivity on fixed matched samples.

    Rows are conditioned independently within each of the three already selected
    samples.  The returned counts and retained fractions make the resulting
    threshold-conditioned estimand explicit.  The original release endpoint is
    not overwritten.
    """
    arrays = [np.asarray(v, dtype=float) for v in (source_shift, source_null, target_shift)]
    if any(a.ndim != 2 for a in arrays):
        raise ValueError("All attribution arrays must be two-dimensional.")
    if len({a.shape[1] for a in arrays}) != 1:
        raise ValueError("Attribution arrays must share a feature dimension.")
    if any(a.shape[0] == 0 for a in arrays):
        raise ValueError("Attribution arrays must contain at least one row.")
    if any(not np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("Attribution arrays contain non-finite values.")
    l1s = [np.sum(np.abs(a), axis=1) for a in arrays]
    out: dict[str, float | int] = {}

    def add_result(stem: str, masks: list[NDArray[np.bool_]]) -> None:
        selected = [a[m] for a, m in zip(arrays, masks)]
        counts = [int(len(a)) for a in selected]
        for label, n, total in zip(("source", "null", "target"), counts, map(len, arrays)):
            out[f"{stem}_{label}_n"] = n
            out[f"{stem}_{label}_fraction"] = float(n / total)
        if min(counts) < 2:
            out[f"{stem}_shift_u2"] = float("nan")
            out[f"{stem}_refit_null_u2"] = float("nan")
            out[f"{stem}_excess_u2"] = float("nan")
            return
        # Use the same release-defined normalization as the primary endpoint.
        # This keeps the sensitivity transform mathematically identical, including
        # the epsilon convention, after rows have been thresholded.
        normalized = [normalize_signed_rows(a) for a in selected]
        shift = energy_u_statistic_squared(normalized[0], normalized[2])
        null = energy_u_statistic_squared(normalized[0], normalized[1])
        out[f"{stem}_shift_u2"] = float(shift)
        out[f"{stem}_refit_null_u2"] = float(null)
        out[f"{stem}_excess_u2"] = float(shift - null)

    for threshold in thresholds:
        if threshold < 0 or not np.isfinite(threshold):
            raise ValueError("thresholds must be finite and non-negative.")
        add_result(
            f"l1thr_{_threshold_key(threshold)}",
            [l1 > threshold for l1 in l1s],
        )

    if amplitude_strata < 2:
        raise ValueError("amplitude_strata must be at least 2.")
    pooled = np.concatenate(l1s)
    edges = np.quantile(pooled, np.linspace(0.0, 1.0, amplitude_strata + 1))
    for i in range(amplitude_strata):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == amplitude_strata - 1:
            masks = [(l1 >= lo) & (l1 <= hi) for l1 in l1s]
        else:
            masks = [(l1 >= lo) & (l1 < hi) for l1 in l1s]
        stem = f"l1stratum_q{i+1}"
        out[f"{stem}_lower"] = lo
        out[f"{stem}_upper"] = hi
        add_result(stem, masks)
    return out


def repeated_explanation_subsample_sensitivity(
    source_shift: ArrayLike,
    source_null: ArrayLike,
    target_shift: ArrayLike,
    y_source_shift: ArrayLike,
    y_source_null: ArrayLike,
    y_target_shift: ArrayLike,
    *,
    seed: int,
    repeats: int = 50,
    fractions: tuple[float, ...] = (0.50, 0.75),
) -> dict[str, float | int]:
    """Repeated stratified subsampling of an already matched explanation sample.

    This is a sample-size sensitivity analysis conditional on a fitted model pair and
    on the release-defined matched explanation sample.  It does not redraw the
    original full explanation sample and therefore does not conflate explanation
    sampling with model-refit variability.  Each of the source-shift, source-null,
    and target arrays is independently subsampled without replacement while
    preserving its observed binary class composition as closely as integer counts
    allow.
    """
    arrays = [np.asarray(v, dtype=float) for v in (source_shift, source_null, target_shift)]
    labels = [np.asarray(v, dtype=int).reshape(-1) for v in (y_source_shift, y_source_null, y_target_shift)]
    if any(a.ndim != 2 for a in arrays):
        raise ValueError("Attribution arrays must be two-dimensional.")
    if len({a.shape[1] for a in arrays}) != 1:
        raise ValueError("Attribution arrays must share a feature dimension.")
    if any(len(a) != len(y) for a, y in zip(arrays, labels)):
        raise ValueError("Each attribution array must match its label vector.")
    if any(len(a) < 4 for a in arrays):
        raise ValueError("Each matched explanation sample must contain at least four rows.")
    if any(not np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("Attribution arrays contain non-finite values.")
    if any(not set(np.unique(y).tolist()).issubset({0, 1}) for y in labels):
        raise ValueError("Labels must be binary 0/1.")
    if repeats < 2:
        raise ValueError("repeats must be at least two.")
    for fraction in fractions:
        if not (0.0 < fraction < 1.0):
            raise ValueError("fractions must lie strictly between zero and one.")

    def stratified_draw(y: NDArray[np.int64], n: int, rng: np.random.Generator) -> NDArray[np.int64]:
        classes = np.unique(y)
        if len(classes) == 1:
            return np.asarray(rng.choice(len(y), size=n, replace=False), dtype=np.int64)
        positive = np.flatnonzero(y == 1)
        negative = np.flatnonzero(y == 0)
        n_pos = int(np.rint(n * float(np.mean(y))))
        n_pos = min(max(n_pos, 1), n - 1)
        n_pos = min(n_pos, len(positive))
        n_neg = n - n_pos
        if n_neg > len(negative):
            n_neg = len(negative)
            n_pos = n - n_neg
        if n_pos > len(positive) or n_neg > len(negative) or min(n_pos, n_neg) < 1:
            raise ValueError("Unable to preserve both classes at the requested subsample size.")
        idx = np.concatenate([
            rng.choice(positive, size=n_pos, replace=False),
            rng.choice(negative, size=n_neg, replace=False),
        ])
        return np.asarray(idx[rng.permutation(len(idx))], dtype=np.int64)

    full_shift = energy_u_statistic_squared(arrays[0], arrays[2])
    full_null = energy_u_statistic_squared(arrays[0], arrays[1])
    full_excess = float(full_shift - full_null)
    out: dict[str, float | int] = {
        "expsamp_full_n": int(min(map(len, arrays))),
        "expsamp_full_excess_u2": full_excess,
    }
    base_n = min(map(len, arrays))
    for fi, fraction in enumerate(fractions):
        n = max(4, int(np.floor(base_n * fraction)))
        n = min(n, base_n - 1)
        values: list[float] = []
        for r in range(repeats):
            rng = np.random.default_rng(seed + fi * 1_000_003 + r * 10_007)
            idx = [stratified_draw(y, n, rng) for y in labels]
            shift = energy_u_statistic_squared(arrays[0][idx[0]], arrays[2][idx[2]])
            null = energy_u_statistic_squared(arrays[0][idx[0]], arrays[1][idx[1]])
            values.append(float(shift - null))
        v = np.asarray(values, dtype=float)
        key = f"expsamp_f{int(round(100*fraction)):03d}"
        out[f"{key}_n"] = int(n)
        out[f"{key}_repeats"] = int(repeats)
        out[f"{key}_mean_excess_u2"] = float(np.mean(v))
        out[f"{key}_median_excess_u2"] = float(np.median(v))
        out[f"{key}_sd_excess_u2"] = float(np.std(v, ddof=1))
        out[f"{key}_q025_excess_u2"] = float(np.quantile(v, 0.025))
        out[f"{key}_q975_excess_u2"] = float(np.quantile(v, 0.975))
        out[f"{key}_positive_fraction"] = float(np.mean(v > 0.0))
        out[f"{key}_median_minus_full_u2"] = float(np.median(v) - full_excess)
    return out

def kendall_tau_b(a: ArrayLike, b: ArrayLike) -> float:
    """Kendall tau-b between two feature-importance vectors."""
    aa, bb = _paired_1d(a, b)
    result = kendalltau(aa, bb, variant="b", nan_policy="raise")
    return float(result.statistic) if np.isfinite(result.statistic) else float("nan")


def top_k_jaccard(a: ArrayLike, b: ArrayLike, k: int) -> float:
    """Jaccard overlap of the k largest absolute feature attributions."""
    aa, bb = _paired_1d(a, b)
    if not 1 <= k <= aa.size:
        raise ValueError(f"k must be in [1, {aa.size}]; got {k}.")
    # Stable tie policy: rank by decreasing absolute magnitude, then by
    # original feature index.  This avoids the implementation-dependent
    # cutoff behaviour of ``argpartition`` when magnitudes are tied.
    idx = np.arange(aa.size)
    ia = set(np.lexsort((idx, -np.abs(aa)))[:k].tolist())
    ib = set(np.lexsort((idx, -np.abs(bb)))[:k].tolist())
    return len(ia & ib) / len(ia | ib)


def jensen_shannon_distance(a: ArrayLike, b: ArrayLike) -> float:
    """Jensen-Shannon distance between non-negative importance profiles."""
    aa, bb = _paired_1d(a, b)
    if np.any(aa < 0) or np.any(bb < 0):
        raise ValueError("Jensen-Shannon inputs must be non-negative.")
    if aa.sum() <= 0 or bb.sum() <= 0:
        raise ValueError("Each Jensen-Shannon input must have positive mass.")
    return float(jensenshannon(aa / aa.sum(), bb / bb.sum(), base=2.0))


def weighted_sign_consistency(a: ArrayLike, b: ArrayLike, threshold: float = 1e-12) -> float:
    """Importance-weighted agreement of attribution directions.

    Features with negligible combined magnitude receive zero weight.
    """
    aa, bb = _paired_1d(a, b)
    weights = (np.abs(aa) + np.abs(bb)) / 2.0
    valid = weights > threshold
    if not np.any(valid):
        return 1.0
    agreement = np.sign(aa[valid]) == np.sign(bb[valid])
    return float(np.average(agreement.astype(float), weights=weights[valid]))


def signed_total_variation(a: ArrayLike, b: ArrayLike) -> float:
    """Half-L1 distance between two normalized signed attribution vectors."""
    aa, bb = _paired_1d(a, b)
    return float(0.5 * np.sum(np.abs(aa - bb)))



def energy_u_statistic_squared(x: ArrayLike, y: ArrayLike) -> float:
    """Unbiased off-diagonal estimator of squared energy distance.

    The estimate can be negative in finite samples, as expected for an
    unbiased U-statistic under the null.  It must not be truncated or square
    rooted before calibration.
    """
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    if xx.ndim != 2 or yy.ndim != 2:
        raise ValueError("x and y must both be two-dimensional.")
    if xx.shape[1] != yy.shape[1]:
        raise ValueError("x and y must have the same number of features.")
    n, m = xx.shape[0], yy.shape[0]
    if n < 2 or m < 2:
        raise ValueError("x and y must each contain at least two rows.")
    if not np.all(np.isfinite(xx)) or not np.all(np.isfinite(yy)):
        raise ValueError("x or y contains non-finite values.")
    cross = float(cdist(xx, yy, metric="euclidean").sum())
    dx = cdist(xx, xx, metric="euclidean")
    dy = cdist(yy, yy, metric="euclidean")
    within_x = float((dx.sum() - np.trace(dx)) / (n * (n - 1)))
    within_y = float((dy.sum() - np.trace(dy)) / (m * (m - 1)))
    return float(2.0 * cross / (n * m) - within_x - within_y)



def weighted_energy_u_statistic_squared(
    x: ArrayLike,
    y: ArrayLike,
    x_weights: ArrayLike,
    y_weights: ArrayLike,
) -> float:
    """Weighted off-diagonal analogue of squared energy distance.

    Positive finite weights are normalized separately within ``x`` and ``y``.
    Cross-sample distances use the product empirical weights. Within-sample
    terms exclude self-pairs and renormalize the remaining product weights.
    With equal weights this is exactly ``energy_u_statistic_squared``.

    This estimator is used only for survey-weight sensitivity analyses; it does
    not replace the release-defined unweighted primary endpoint.
    """
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    wx = _as_1d(x_weights, "x_weights")
    wy = _as_1d(y_weights, "y_weights")
    if xx.ndim != 2 or yy.ndim != 2:
        raise ValueError("x and y must both be two-dimensional.")
    if xx.shape[1] != yy.shape[1]:
        raise ValueError("x and y must have the same number of features.")
    if len(wx) != len(xx) or len(wy) != len(yy):
        raise ValueError("Weight vectors must match the corresponding row counts.")
    if len(xx) < 2 or len(yy) < 2:
        raise ValueError("x and y must each contain at least two rows.")
    if not np.all(np.isfinite(xx)) or not np.all(np.isfinite(yy)):
        raise ValueError("x or y contains non-finite values.")
    if np.any(wx <= 0) or np.any(wy <= 0):
        raise ValueError("Weights must be strictly positive.")

    px = wx / wx.sum()
    py = wy / wy.sum()
    dxy = cdist(xx, yy, metric="euclidean")
    dxx = cdist(xx, xx, metric="euclidean")
    dyy = cdist(yy, yy, metric="euclidean")
    cross = float(np.sum(dxy * np.outer(px, py)))

    mass_x = 1.0 - float(np.sum(px * px))
    mass_y = 1.0 - float(np.sum(py * py))
    if mass_x <= 0.0 or mass_y <= 0.0:
        raise ValueError("Effective weighted sample size is insufficient for off-diagonal terms.")
    within_x = float(np.sum(dxx * np.outer(px, px)) / mass_x)
    within_y = float(np.sum(dyy * np.outer(py, py)) / mass_y)
    return float(2.0 * cross - within_x - within_y)

def energy_v_statistic_squared(x: ArrayLike, y: ArrayLike) -> float:
    """Diagonal-including squared energy V-statistic retained for compatibility checks."""
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    if xx.ndim != 2 or yy.ndim != 2 or xx.shape[1] != yy.shape[1]:
        raise ValueError("x and y must be compatible two-dimensional arrays.")
    if xx.shape[0] == 0 or yy.shape[0] == 0:
        raise ValueError("x and y must each contain at least one row.")
    cross = cdist(xx, yy, metric="euclidean").mean()
    within_x = cdist(xx, xx, metric="euclidean").mean()
    within_y = cdist(yy, yy, metric="euclidean").mean()
    return float(2.0 * cross - within_x - within_y)

def multivariate_energy_distance(x: ArrayLike, y: ArrayLike) -> float:
    """Biased non-negative multivariate energy distance estimate.

    This population-level metric is appropriate when source and target records
    are not paired. Complexity is quadratic, so benchmark callers should use a
    release-defined stratified explanation sample.
    """
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    if xx.ndim != 2 or yy.ndim != 2:
        raise ValueError("x and y must both be two-dimensional.")
    if xx.shape[1] != yy.shape[1]:
        raise ValueError("x and y must have the same number of features.")
    if xx.shape[0] == 0 or yy.shape[0] == 0:
        raise ValueError("x and y must each contain at least one row.")
    if not np.all(np.isfinite(xx)) or not np.all(np.isfinite(yy)):
        raise ValueError("x or y contains non-finite values.")
    squared = max(0.0, energy_v_statistic_squared(xx, yy))
    return float(np.sqrt(squared))
