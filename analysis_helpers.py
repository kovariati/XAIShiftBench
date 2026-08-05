from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def energy_u2(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n, m = len(x), len(y)
    cross = cdist(x, y).sum()
    dx = cdist(x, x)
    dy = cdist(y, y)
    return float(2 * cross / (n * m) - (dx.sum() - np.trace(dx)) / (n * (n - 1)) - (dy.sum() - np.trace(dy)) / (m * (m - 1)))


def signed_compositions(rng: np.random.Generator, n: int, p: int, sparsity: float) -> np.ndarray:
    z = rng.laplace(size=(n, p))
    if sparsity > 0:
        mask = rng.random((n, p)) < sparsity
        z[mask] = 0.0
        empty = np.sum(np.abs(z), axis=1) == 0
        if np.any(empty):
            z[empty, rng.integers(0, p, size=int(empty.sum()))] = rng.choice([-1.0, 1.0], size=int(empty.sum()))
    return z / np.sum(np.abs(z), axis=1, keepdims=True)


def metric_scale_simulation(seed: int = 2026080101, reps: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for p in (5, 10, 20, 40, 80):
        for sparsity in (0.0, 0.5, 0.8):
            for n in (30, 60, 160):
                nulls, shifted = [], []
                for _ in range(reps):
                    a = signed_compositions(rng, n, p, sparsity)
                    b = signed_compositions(rng, n, p, sparsity)
                    c = signed_compositions(rng, n, p, sparsity)
                    # A fixed relative composition intervention: magnify the first
                    # two coordinates and renormalize, preserving signed L1 mass.
                    c[:, : min(2, p)] *= 1.8
                    c = c / np.maximum(np.sum(np.abs(c), axis=1, keepdims=True), 1e-15)
                    nulls.append(energy_u2(a, b))
                    shifted.append(energy_u2(a, c))
                nulls = np.asarray(nulls)
                shifted = np.asarray(shifted)
                excess = shifted - nulls
                sd_null = float(np.std(nulls, ddof=1))
                rows.append({
                    'semantic_dimension': p,
                    'sparsity_fraction': sparsity,
                    'sample_size': n,
                    'replicates': reps,
                    'null_mean_u2': float(np.mean(nulls)),
                    'null_sd_u2': sd_null,
                    'shift_mean_u2': float(np.mean(shifted)),
                    'mean_matched_null_excess_u2': float(np.mean(excess)),
                    'standardized_excess_null_sd': float(np.mean(excess) / sd_null) if sd_null > 0 else math.nan,
                    'positive_excess_fraction': float(np.mean(excess > 0)),
                })
    return pd.DataFrame(rows)


def small_denominator_simulation(seed: int = 2026080102, reps: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    p, n = 20, 160
    for noise_sd in (1e-8, 1e-6, 1e-4):
        for amplitude_median in (1e-6, 1e-4, 1e-2, 1.0):
            for threshold in (0.0, 1e-8, 1e-6, 1e-4):
                distances, retained = [], []
                for _ in range(reps):
                    q = signed_compositions(rng, n, p, 0.3)
                    amp = rng.lognormal(mean=np.log(amplitude_median), sigma=1.0, size=(n, 1))
                    phi1 = amp * q + rng.normal(scale=noise_sd, size=(n, p))
                    phi2 = amp * q + rng.normal(scale=noise_sd, size=(n, p))
                    l1a = np.sum(np.abs(phi1), axis=1)
                    l1b = np.sum(np.abs(phi2), axis=1)
                    keep = (l1a > threshold) & (l1b > threshold)
                    retained.append(float(np.mean(keep)))
                    if keep.sum() < 3:
                        distances.append(np.nan)
                        continue
                    qa = phi1[keep] / np.sum(np.abs(phi1[keep]), axis=1, keepdims=True)
                    qb = phi2[keep] / np.sum(np.abs(phi2[keep]), axis=1, keepdims=True)
                    distances.append(float(np.mean(0.5 * np.sum(np.abs(qa - qb), axis=1))))
                valid_distances = np.asarray([d for d in distances if np.isfinite(d)], dtype=float)
                rows.append({
                    'noise_sd': noise_sd,
                    'amplitude_median': amplitude_median,
                    'l1_exclusion_threshold': threshold,
                    'replicates': reps,
                    'valid_replicates': int(valid_distances.size),
                    'mean_retained_fraction': float(np.mean(retained)),
                    'mean_paired_signed_total_variation': float(np.mean(valid_distances)) if valid_distances.size else math.nan,
                    'sd_paired_signed_total_variation': float(np.std(valid_distances, ddof=1)) if valid_distances.size > 1 else math.nan,
                })
    return pd.DataFrame(rows)


def scopus_audit(scopus_path: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    df = pd.read_csv(scopus_path, encoding='utf-8-sig')
    required = {'Title', 'Year', 'Cited by', 'Abstract', 'Author Keywords', 'Index Keywords'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing Scopus columns: {sorted(missing)}')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Cited by'] = pd.to_numeric(df['Cited by'], errors='coerce')
    text = (df['Title'].fillna('') + ' ' + df['Abstract'].fillna('') + ' ' +
            df['Author Keywords'].fillna('') + ' ' + df['Index Keywords'].fillna('')).str.lower()
    terms = [
        'benchmark', 'reliability', 'robustness', 'interpretability',
        'explainable artificial intelligence', 'distribution shift',
        'trustworthy', 'framework', 'survey', 'review', 'comprehensive',
        'dataset', 'guideline', 'machine learning', 'artificial intelligence'
    ]
    term_rows = []
    for term in terms:
        m = text.str.contains(re.escape(term), regex=True, na=False)
        tm = df['Title'].str.lower().str.contains(re.escape(term), regex=True, na=False)
        term_rows.append({
            'term': term,
            'records_with_term': int(m.sum()),
            'titles_with_term': int(tm.sum()),
            'mean_citations': float(df.loc[m, 'Cited by'].mean()) if m.any() else math.nan,
            'median_citations': float(df.loc[m, 'Cited by'].median()) if m.any() else math.nan,
            'maximum_citations': float(df.loc[m, 'Cited by'].max()) if m.any() else math.nan,
        })
    term_df = pd.DataFrame(term_rows).sort_values(['mean_citations', 'records_with_term'], ascending=[False, False])
    q99 = float(df['Cited by'].quantile(0.99))
    audit = {
        'records': int(len(df)),
        'year_counts': {str(int(k)): int(v) for k, v in df['Year'].value_counts().sort_index().items()},
        'citations_mean': float(df['Cited by'].mean()),
        'citations_median': float(df['Cited by'].median()),
        'citations_maximum': float(df['Cited by'].max()),
        'citations_q99': q99,
        'citations_winsorized_mean_at_q99': float(df['Cited by'].clip(upper=q99).mean()),
        'missing_doi': int(df['DOI'].isna().sum()) if 'DOI' in df else None,
        'note': ('The export has no document-type, original-publication-date, or edition field. '
                 'Consequently, probable reissues or bibliographic carry-over records cannot be '
                 'definitively removed from this file alone; title-term statistics are descriptive.'),
    }
    return df, audit, term_df
