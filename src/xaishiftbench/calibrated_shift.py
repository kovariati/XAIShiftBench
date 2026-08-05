"""Matched-sample calibration for local attribution-distribution shift.

The primary statistic is the unbiased off-diagonal energy U-statistic on
row-wise L1-normalized signed semantic attributions.  Deployment contrasts are
calibrated against source-domain samples with the same total sample sizes and
class compositions.  A second source sample is evaluated under an independent
model refit so that sampling and refit variability remain visible as separate
components.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .metrics import energy_u_statistic_squared


@dataclass(frozen=True)
class ShiftSampleIndices:
    source_shift: NDArray[np.int64]
    source_null: NDArray[np.int64]
    target_shift: NDArray[np.int64]
    sample_size: int
    source_positive_count: int
    target_positive_count: int


@dataclass(frozen=True)
class CalibratedShiftResult:
    indices: ShiftSampleIndices
    shift_u2: float
    same_model_sample_null_u2: float
    refit_sample_null_u2: float
    excess_over_refit_sample_null_u2: float
    paired_refit_l2: float
    class_macro_shift_u2: float
    class_macro_sample_null_u2: float
    class_macro_refit_null_u2: float
    class_macro_excess_u2: float
    class_weighted_shift_u2: float
    class_weighted_refit_null_u2: float
    class_weighted_excess_u2: float
    class0_shift_u2: float
    class1_shift_u2: float
    class0_refit_null_u2: float
    class1_refit_null_u2: float
    class0_n: int
    class1_n: int
    class_weighting_prevalence: float
    reverse_shift_u2: float
    reverse_refit_sample_null_u2: float
    reverse_excess_u2: float
    symmetrized_excess_u2: float
    symmetry_gap_u2: float


def _binary_labels(values: NDArray[np.int_], name: str) -> NDArray[np.int64]:
    arr = np.asarray(values, dtype=int).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    unique = set(np.unique(arr).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"{name} must contain only 0/1 labels; got {sorted(unique)}.")
    return arr.astype(np.int64, copy=False)


def _positive_count(n: int, prevalence: float, min_per_class: int) -> int:
    if n < 2 * min_per_class:
        raise ValueError("Sample size is too small for both classes.")
    count = int(np.rint(n * prevalence))
    return min(max(count, min_per_class), n - min_per_class)


def _draw_from_class(
    indices: NDArray[np.int64], n: int, rng: np.random.Generator
) -> NDArray[np.int64]:
    if n > len(indices):
        raise ValueError(f"Requested {n} rows from a class containing only {len(indices)} rows.")
    return np.asarray(rng.choice(indices, size=n, replace=False), dtype=np.int64)


def _draw_stratified(
    labels: NDArray[np.int64], n: int, positive_count: int, rng: np.random.Generator
) -> NDArray[np.int64]:
    pos = np.flatnonzero(labels == 1).astype(np.int64)
    neg = np.flatnonzero(labels == 0).astype(np.int64)
    selected = np.concatenate(
        [
            _draw_from_class(pos, positive_count, rng),
            _draw_from_class(neg, n - positive_count, rng),
        ]
    )
    return selected[rng.permutation(len(selected))]


def matched_composition_indices(
    y_source: NDArray[np.int_],
    y_target: NDArray[np.int_],
    *,
    cap: int,
    seed: int,
    min_per_class: int = 3,
) -> ShiftSampleIndices:
    """Draw matched, non-overlapping source samples and one target sample.

    The first source sample follows the source class prevalence.  The second
    source sample follows the target class prevalence.  This makes the
    same-domain null match the deployment comparison in sample size and class
    composition.  The two source samples are disjoint.
    """
    ys = _binary_labels(y_source, "y_source")
    yt = _binary_labels(y_target, "y_target")
    if cap < 2 * min_per_class:
        raise ValueError("cap is too small for the requested minimum class counts.")
    source_counts = np.bincount(ys, minlength=2)
    target_counts = np.bincount(yt, minlength=2)
    source_prev = float(np.mean(ys))
    target_prev = float(np.mean(yt))
    max_n = min(int(cap), len(yt), len(ys) // 2)

    feasible: tuple[int, int, int] | None = None
    for n in range(max_n, 2 * min_per_class - 1, -1):
        pos_source = _positive_count(n, source_prev, min_per_class)
        pos_target = _positive_count(n, target_prev, min_per_class)
        required_source_pos = pos_source + pos_target
        required_source_neg = (n - pos_source) + (n - pos_target)
        required_target_pos = pos_target
        required_target_neg = n - pos_target
        if (
            source_counts[1] >= required_source_pos
            and source_counts[0] >= required_source_neg
            and target_counts[1] >= required_target_pos
            and target_counts[0] >= required_target_neg
        ):
            feasible = (n, pos_source, pos_target)
            break
    if feasible is None:
        raise ValueError(
            "Unable to construct disjoint matched samples with the requested "
            "minimum per-class counts."
        )

    n, pos_source, pos_target = feasible
    rng = np.random.default_rng(seed)
    source_pos = np.flatnonzero(ys == 1).astype(np.int64)
    source_neg = np.flatnonzero(ys == 0).astype(np.int64)
    source_pos = source_pos[rng.permutation(len(source_pos))]
    source_neg = source_neg[rng.permutation(len(source_neg))]

    source_shift = np.concatenate(
        [source_pos[:pos_source], source_neg[: n - pos_source]]
    )
    source_null = np.concatenate(
        [
            source_pos[pos_source : pos_source + pos_target],
            source_neg[n - pos_source : (n - pos_source) + (n - pos_target)],
        ]
    )
    source_shift = source_shift[rng.permutation(len(source_shift))]
    source_null = source_null[rng.permutation(len(source_null))]
    target_shift = _draw_stratified(yt, n, pos_target, rng)

    if np.intersect1d(source_shift, source_null).size:
        raise AssertionError("Matched source samples must be disjoint.")
    return ShiftSampleIndices(
        source_shift=source_shift,
        source_null=source_null,
        target_shift=target_shift,
        sample_size=n,
        source_positive_count=pos_source,
        target_positive_count=pos_target,
    )


def _class_component(
    a_source: NDArray[np.float64],
    b_source: NDArray[np.float64],
    target: NDArray[np.float64],
    y_source: NDArray[np.int64],
    y_target: NDArray[np.int64],
    cls: int,
    *,
    cap: int,
    seed: int,
    min_n: int,
) -> tuple[float, float, float, int]:
    src_idx = np.flatnonzero(y_source == cls).astype(np.int64)
    tgt_idx = np.flatnonzero(y_target == cls).astype(np.int64)
    n = min(cap, len(src_idx) // 2, len(tgt_idx))
    if n < min_n:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    src_idx = src_idx[rng.permutation(len(src_idx))]
    tgt_idx = tgt_idx[rng.permutation(len(tgt_idx))]
    s1 = src_idx[:n]
    s2 = src_idx[n : 2 * n]
    tt = tgt_idx[:n]
    shift = energy_u_statistic_squared(a_source[s1], target[tt])
    sample_null = energy_u_statistic_squared(a_source[s1], a_source[s2])
    refit_null = energy_u_statistic_squared(a_source[s1], b_source[s2])
    return shift, sample_null, refit_null, int(n)


def calibrated_attribution_shift(
    normalized_source_a: NDArray[np.float64],
    normalized_target_a: NDArray[np.float64],
    normalized_source_b: NDArray[np.float64],
    y_source: NDArray[np.int_],
    y_target: NDArray[np.int_],
    *,
    normalized_target_b: NDArray[np.float64] | None = None,
    seed: int,
    cap: int = 160,
    class_cap: int = 80,
    min_per_class: int = 3,
) -> CalibratedShiftResult:
    """Compute matched U-energy shift and separated null components."""
    a = np.asarray(normalized_source_a, dtype=float)
    t = np.asarray(normalized_target_a, dtype=float)
    b = np.asarray(normalized_source_b, dtype=float)
    tb = t if normalized_target_b is None else np.asarray(normalized_target_b, dtype=float)
    ys = _binary_labels(y_source, "y_source")
    yt = _binary_labels(y_target, "y_target")
    if a.ndim != 2 or t.ndim != 2 or b.ndim != 2 or tb.ndim != 2:
        raise ValueError("All attribution arrays must be two-dimensional.")
    if a.shape != b.shape:
        raise ValueError("The two source-refit attribution arrays must have identical shapes.")
    if a.shape[0] != len(ys) or t.shape[0] != len(yt):
        raise ValueError("Attribution arrays and labels have incompatible row counts.")
    if a.shape[1] != t.shape[1] or tb.shape != t.shape:
        raise ValueError("Source and target attributions must have the same feature dimension.")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or not np.all(np.isfinite(t)) or not np.all(np.isfinite(tb)):
        raise ValueError("Attribution arrays contain non-finite values.")

    idx = matched_composition_indices(
        ys, yt, cap=cap, seed=seed, min_per_class=min_per_class
    )
    sa = a[idx.source_shift]
    sn_a = a[idx.source_null]
    sn_b = b[idx.source_null]
    tt = t[idx.target_shift]
    shift = energy_u_statistic_squared(sa, tt)
    sample_null = energy_u_statistic_squared(sa, sn_a)
    refit_null = energy_u_statistic_squared(sa, sn_b)
    reverse_shift = energy_u_statistic_squared(b[idx.source_shift], tb[idx.target_shift])
    reverse_refit_null = energy_u_statistic_squared(b[idx.source_shift], a[idx.source_null])
    reverse_excess = float(reverse_shift - reverse_refit_null)
    forward_excess = float(shift - refit_null)
    symmetrized_excess = float(0.5 * (forward_excess + reverse_excess))
    paired_refit = float(np.mean(np.linalg.norm(a[idx.source_shift] - b[idx.source_shift], axis=1)))

    class_values = []
    for cls in (0, 1):
        class_values.append(
            _class_component(
                a,
                b,
                t,
                ys,
                yt,
                cls,
                cap=class_cap,
                seed=seed + 1000 + cls,
                min_n=min_per_class,
            )
        )
    c0s, c0n, c0r, n0 = class_values[0]
    c1s, c1n, c1r, n1 = class_values[1]
    valid = np.array([n0 > 0, n1 > 0], dtype=bool)
    shifts = np.array([c0s, c1s], dtype=float)
    samples = np.array([c0n, c1n], dtype=float)
    refits = np.array([c0r, c1r], dtype=float)
    if not np.any(valid):
        macro_shift = macro_sample = macro_refit = float("nan")
        weighted_shift = weighted_refit = float("nan")
    else:
        macro_shift = float(np.nanmean(shifts[valid]))
        macro_sample = float(np.nanmean(samples[valid]))
        macro_refit = float(np.nanmean(refits[valid]))
        target_prevalence = float(np.mean(yt))
        weights = np.array([1.0 - target_prevalence, target_prevalence], dtype=float)
        weights = np.where(valid, weights, 0.0)
        weights = weights / weights.sum()
        weighted_shift = float(np.nansum(shifts * weights))
        weighted_refit = float(np.nansum(refits * weights))

    return CalibratedShiftResult(
        indices=idx,
        shift_u2=float(shift),
        same_model_sample_null_u2=float(sample_null),
        refit_sample_null_u2=float(refit_null),
        excess_over_refit_sample_null_u2=forward_excess,
        paired_refit_l2=paired_refit,
        class_macro_shift_u2=macro_shift,
        class_macro_sample_null_u2=macro_sample,
        class_macro_refit_null_u2=macro_refit,
        class_macro_excess_u2=float(macro_shift - macro_refit),
        class_weighted_shift_u2=weighted_shift,
        class_weighted_refit_null_u2=weighted_refit,
        class_weighted_excess_u2=float(weighted_shift - weighted_refit),
        class0_shift_u2=float(c0s),
        class1_shift_u2=float(c1s),
        class0_refit_null_u2=float(c0r),
        class1_refit_null_u2=float(c1r),
        class0_n=n0,
        class1_n=n1,
        class_weighting_prevalence=float(np.mean(yt)),
        reverse_shift_u2=float(reverse_shift),
        reverse_refit_sample_null_u2=float(reverse_refit_null),
        reverse_excess_u2=reverse_excess,
        symmetrized_excess_u2=symmetrized_excess,
        symmetry_gap_u2=float(forward_excess - reverse_excess),
    )


def calibrated_shift_from_matched_arrays(
    source_shift_a: NDArray[np.float64],
    source_null_a: NDArray[np.float64],
    target_a: NDArray[np.float64],
    source_shift_b: NDArray[np.float64],
    source_null_b: NDArray[np.float64],
    y_source_shift: NDArray[np.int_],
    y_source_null: NDArray[np.int_],
    y_target: NDArray[np.int_],
    *,
    target_b: NDArray[np.float64] | None = None,
) -> dict[str, float | int | str]:
    """Calculate calibrated components from already matched samples.

    This is used by large-data adapters that select rows before generating
    explanations.  All three samples must have equal total size; the source
    null and target samples must have equal class counts by construction.
    """
    sa = np.asarray(source_shift_a, float)
    sn = np.asarray(source_null_a, float)
    tt = np.asarray(target_a, float)
    sb = np.asarray(source_shift_b, float)
    snb = np.asarray(source_null_b, float)
    ttb = tt if target_b is None else np.asarray(target_b, float)
    ys = _binary_labels(y_source_shift, "y_source_shift")
    yn = _binary_labels(y_source_null, "y_source_null")
    yt = _binary_labels(y_target, "y_target")
    arrays = [sa, sn, tt, sb, snb, ttb]
    if any(a.ndim != 2 for a in arrays):
        raise ValueError("All attribution arrays must be two-dimensional.")
    if not (sa.shape == sb.shape and sn.shape == snb.shape and tt.shape == ttb.shape):
        raise ValueError("Source A/B samples must have matching shapes.")
    if not (sa.shape[0] == sn.shape[0] == tt.shape[0]):
        raise ValueError("Matched samples must have equal total sizes.")
    if not all(a.shape[1] == sa.shape[1] for a in arrays):
        raise ValueError("All attribution samples must share a feature dimension.")
    if len(ys) != len(sa) or len(yn) != len(sn) or len(yt) != len(tt):
        raise ValueError("Sample labels have incompatible lengths.")
    if not np.array_equal(np.bincount(yn, minlength=2), np.bincount(yt, minlength=2)):
        raise ValueError("Source-null and target class counts must match.")

    shift = energy_u_statistic_squared(sa, tt)
    sample_null = energy_u_statistic_squared(sa, sn)
    refit_null = energy_u_statistic_squared(sa, snb)
    reverse_shift = energy_u_statistic_squared(sb, ttb)
    reverse_refit_null = energy_u_statistic_squared(sb, sn)
    forward_excess = float(shift - refit_null)
    reverse_excess = float(reverse_shift - reverse_refit_null)
    symmetrized_excess = float(0.5 * (forward_excess + reverse_excess))
    paired_refit = float(np.mean(np.linalg.norm(sa - sb, axis=1)))
    class_shift: list[float] = []
    class_sample: list[float] = []
    class_refit: list[float] = []
    class_n: list[int] = []
    for cls in (0, 1):
        xa = sa[ys == cls]
        xn = sn[yn == cls]
        xb = snb[yn == cls]
        xt = tt[yt == cls]
        n_min = min(len(xa), len(xn), len(xt))
        class_n.append(int(n_min))
        if min(len(xa), len(xn), len(xt)) < 2:
            class_shift.append(float("nan"))
            class_sample.append(float("nan"))
            class_refit.append(float("nan"))
            continue
        class_shift.append(energy_u_statistic_squared(xa, xt))
        class_sample.append(energy_u_statistic_squared(xa, xn))
        class_refit.append(energy_u_statistic_squared(xa, xb))
    cs = np.asarray(class_shift, float)
    cn = np.asarray(class_sample, float)
    cr = np.asarray(class_refit, float)
    valid = np.isfinite(cs) & np.isfinite(cr)
    macro_shift = float(np.mean(cs[valid])) if np.any(valid) else float("nan")
    macro_sample = float(np.mean(cn[valid])) if np.any(valid) else float("nan")
    macro_refit = float(np.mean(cr[valid])) if np.any(valid) else float("nan")
    target_prev = float(np.mean(yt))
    weights = np.asarray([1.0 - target_prev, target_prev], float)
    weights = np.where(valid, weights, 0.0)
    weights = weights / weights.sum() if weights.sum() else weights
    weighted_shift = float(np.nansum(cs * weights))
    weighted_refit = float(np.nansum(cr * weights))
    return {
        "shift_u2": float(shift),
        "same_model_sample_null_u2": float(sample_null),
        "refit_sample_null_u2": float(refit_null),
        "excess_u2": forward_excess,
        "reverse_shift_u2": float(reverse_shift),
        "reverse_refit_sample_null_u2": float(reverse_refit_null),
        "reverse_excess_u2": reverse_excess,
        "symmetrized_excess_u2": symmetrized_excess,
        "symmetry_gap_u2": float(forward_excess - reverse_excess),
        "paired_refit_l2": paired_refit,
        "class_macro_shift_u2": macro_shift,
        "class_macro_sample_null_u2": macro_sample,
        "class_macro_refit_null_u2": macro_refit,
        "class_macro_excess_u2": float(macro_shift - macro_refit),
        "class_weighted_shift_u2": weighted_shift,
        "class_weighted_refit_null_u2": weighted_refit,
        "class_weighted_excess_u2": float(weighted_shift - weighted_refit),
        "class0_shift_u2": float(cs[0]),
        "class1_shift_u2": float(cs[1]),
        "class0_refit_null_u2": float(cr[0]),
        "class1_refit_null_u2": float(cr[1]),
        "class0_n": class_n[0],
        "class1_n": class_n[1],
        "class_weighting_prevalence": target_prev,
        "class_weighting_scheme": "target_prevalence",
    }
