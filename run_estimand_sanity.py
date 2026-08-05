from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from xaishiftbench.calibrated_shift import matched_composition_indices
from xaishiftbench.metrics import energy_u_statistic_squared, normalize_signed_rows


def _signed_composition(x: np.ndarray) -> np.ndarray:
    # Synthetic explanation vectors are already signed contributions; apply the
    # release-defined row-wise L1 normalization before the energy statistic.
    return normalize_signed_rows(x)


def _draw_class_conditional(rng: np.random.Generator, y: np.ndarray, d: int) -> np.ndarray:
    # Two distinct but overlapping attribution-composition clouds. The class
    # dependence makes pure prior shift visible in the marginal explanation
    # distribution, which is exactly what the target-prevalence matched null is
    # designed to remove as a nuisance component.
    mu0 = np.zeros(d); mu1 = np.zeros(d)
    mu0[: min(3, d)] = np.array([0.8, -0.35, 0.20])[: min(3, d)]
    mu1[: min(3, d)] = np.array([-0.55, 0.75, -0.25])[: min(3, d)]
    mu = np.where(y[:, None] == 1, mu1, mu0)
    return mu + rng.normal(scale=0.55, size=(len(y), d))


def _single_rep(case: str, rng: np.random.Generator, n_source: int, n_target: int, cap: int, d: int) -> dict[str, float]:
    if case == "prior_shift":
        ys = rng.binomial(1, 0.30, size=n_source)
        yt = rng.binomial(1, 0.70, size=n_target)
        xs = _draw_class_conditional(rng, ys, d)
        xt = _draw_class_conditional(rng, yt, d)  # P(phi|Y) unchanged
    elif case == "covariate_shift":
        ys = rng.binomial(1, 0.40, size=n_source)
        yt = rng.binomial(1, 0.40, size=n_target)
        xs = _draw_class_conditional(rng, ys, d)
        xt = _draw_class_conditional(rng, yt, d)
        # A representation/covariate movement that changes the source-model
        # attribution geometry within both classes.
        xt[:, 0] += 0.85
        if d > 1:
            xt[:, 1] -= 0.45
    elif case == "concept_shift":
        # Same X/attribution marginal in both domains, but a different labeling
        # rule with approximately the same class prior. This idealized case
        # illustrates that the unconditional endpoint is not a generic detector
        # of P(Y|X) change when the fixed-model explanation distribution itself
        # does not move.
        latent_s = rng.normal(size=(n_source, d))
        latent_t = rng.normal(size=(n_target, d))
        ys = (latent_s[:, 0] + 0.15 * rng.normal(size=n_source) > 0).astype(int)
        if d > 1:
            yt = (latent_t[:, 1] + 0.15 * rng.normal(size=n_target) > 0).astype(int)
        else:
            yt = (-latent_t[:, 0] + 0.15 * rng.normal(size=n_target) > 0).astype(int)
        # Explanation operator is held fixed: it depends on X/model, not on the
        # newly generated target label rule.
        xs = latent_s.copy(); xt = latent_t.copy()
    else:
        raise ValueError(case)

    qs = _signed_composition(xs); qt = _signed_composition(xt)
    idx = matched_composition_indices(ys, yt, cap=cap, seed=int(rng.integers(0, 2**31 - 1)), min_per_class=3)
    s_a = qs[idx.source_shift]
    s_b = qs[idx.source_null]
    t_a = qt[idx.target_shift]
    shift = energy_u_statistic_squared(s_a, t_a)
    matched_null = energy_u_statistic_squared(s_a, s_b)
    return {
        "shift_u2": float(shift),
        "matched_source_null_u2": float(matched_null),
        "adjusted_excess_u2": float(shift - matched_null),
        "source_prevalence": float(ys[idx.source_shift].mean()),
        "matched_null_prevalence": float(ys[idx.source_null].mean()),
        "target_prevalence": float(yt[idx.target_shift].mean()),
        "n": int(idx.sample_size),
    }


def run_simulation(repeats: int = 400, seed: int = 2026080101, n_source: int = 1800, n_target: int = 1200, cap: int = 320, d: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    for case in ("prior_shift", "covariate_shift", "concept_shift"):
        for rep in range(repeats):
            row = _single_rep(case, rng, n_source, n_target, cap, d)
            row.update({"case": case, "replicate": rep})
            rows.append(row)
    raw = pd.DataFrame(rows)
    summary = raw.groupby("case", as_index=False).agg(
        repeats=("replicate", "size"),
        mean_shift_u2=("shift_u2", "mean"),
        sd_shift_u2=("shift_u2", "std"),
        mean_matched_source_null_u2=("matched_source_null_u2", "mean"),
        sd_matched_source_null_u2=("matched_source_null_u2", "std"),
        mean_adjusted_excess_u2=("adjusted_excess_u2", "mean"),
        sd_adjusted_excess_u2=("adjusted_excess_u2", "std"),
        positive_adjusted_fraction=("adjusted_excess_u2", lambda x: float((x > 0).mean())),
        mean_source_prevalence=("source_prevalence", "mean"),
        mean_target_prevalence=("target_prevalence", "mean"),
        mean_matched_null_prevalence=("matched_null_prevalence", "mean"),
        mean_n=("n", "mean"),
    )
    return raw, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal the release sanity check for the matched-null estimand under prior, covariate, and concept shift.")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=400)
    ap.add_argument("--seed", type=int, default=2026080101)
    ns = ap.parse_args()
    ns.output_dir.mkdir(parents=True, exist_ok=True)
    raw, summary = run_simulation(repeats=ns.repeats, seed=ns.seed)
    raw.to_csv(ns.output_dir / "estimand_sanity_replicates.csv", index=False)
    summary.to_csv(ns.output_dir / "estimand_sanity_summary.csv", index=False)
    payload = {
        "version": "1.0.0",
        "seed": ns.seed,
        "repeats_per_case": ns.repeats,
        "cases": {
            "prior_shift": "P(phi|Y) fixed while P(Y) changes; target-prevalence source matching is intended to absorb this marginal composition component.",
            "covariate_shift": "P(Y) held approximately fixed while target attribution geometry changes within class; adjusted excess should remain positive.",
            "concept_shift": "X/attribution marginal held fixed while the target labeling rule changes; the unconditional fixed-model explanation endpoint is not expected to be a generic detector of P(Y|X) change.",
        },
        "interpretation_boundary": "This is an estimand sanity check, not a power study or proof for all label/covariate/concept shifts.",
        "summary": summary.to_dict(orient="records"),
    }
    (ns.output_dir / "ESTIMAND_SANITY.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
