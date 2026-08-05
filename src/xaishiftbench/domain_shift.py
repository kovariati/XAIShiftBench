"""Diagnostics that separate covariate separability from explanation separability."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def domain_classifier_auc(source: ArrayLike, target: ArrayLike, seed: int) -> float:
    """Cross-validated AUROC for discriminating source from target representations."""
    x0 = np.asarray(source, dtype=float)
    x1 = np.asarray(target, dtype=float)
    if x0.ndim != 2 or x1.ndim != 2 or x0.shape[1] != x1.shape[1]:
        raise ValueError("source and target must be two-dimensional with matching columns.")
    x = np.vstack([x0, x1])
    y = np.concatenate([np.zeros(len(x0), dtype=int), np.ones(len(x1), dtype=int)])
    minimum_class = int(np.bincount(y).min())
    n_splits = min(5, minimum_class)
    if n_splits < 2:
        raise ValueError("Insufficient records for domain-classifier cross-validation.")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=3000, random_state=seed),
    )
    scores = cross_val_score(estimator, x, y, cv=cv, scoring="roc_auc", n_jobs=1)
    return float(scores.mean())
