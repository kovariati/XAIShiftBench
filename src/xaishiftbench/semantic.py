"""Semantic feature alignment across encoded model inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.compose import ColumnTransformer


@dataclass(frozen=True)
class SemanticMap:
    names: tuple[str, ...]
    groups: tuple[tuple[int, ...], ...]

    def aggregate(self, values: ArrayLike) -> NDArray[np.float64]:
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"values must be two-dimensional; got {arr.shape}.")
        if any(max(group, default=-1) >= arr.shape[1] for group in self.groups):
            raise ValueError("Semantic group index exceeds encoded feature count.")
        result = np.column_stack([arr[:, group].sum(axis=1) for group in self.groups])
        if not np.allclose(result.sum(axis=1), arr.sum(axis=1), atol=1e-10, rtol=1e-10):
            raise AssertionError("Semantic aggregation did not preserve row-wise attribution sums.")
        return result


def semantic_map_from_preprocessor(
    preprocessor: ColumnTransformer,
    numeric_features: list[str],
    categorical_features: list[str],
    indicator_features: list[str] | None = None,
) -> SemanticMap:
    """Construct a deterministic map from encoded columns to original features."""
    if not hasattr(preprocessor, "transformers_"):
        raise ValueError("preprocessor must be fitted before building a semantic map.")
    groups: dict[str, list[int]] = {name: [] for name in numeric_features + categorical_features}
    cursor = 0

    numeric_transformer = preprocessor.named_transformers_["num"]
    if numeric_transformer == "drop":
        raise ValueError("Numeric transformer unexpectedly dropped all numeric features.")
    for name in numeric_features:
        groups[name].append(cursor)
        cursor += 1

    cat_pipeline = preprocessor.named_transformers_["cat"]
    encoder = cat_pipeline.named_steps["onehot"]
    for name, categories in zip(categorical_features, encoder.categories_, strict=True):
        width = len(categories)
        groups[name].extend(range(cursor, cursor + width))
        cursor += width

    if indicator_features is not None:
        indicator = preprocessor.named_transformers_.get("ind")
        if indicator is None or indicator == "drop":
            raise ValueError("indicator_features were supplied but no fitted 'ind' transformer exists.")
        features_out = np.asarray(indicator.features_, dtype=int)
        for original_index in features_out:
            name = indicator_features[int(original_index)]
            if name not in groups:
                raise ValueError(f"Indicator feature {name!r} is not a semantic feature.")
            groups[name].append(cursor)
            cursor += 1

    encoded_count = len(preprocessor.get_feature_names_out())
    if cursor != encoded_count:
        raise AssertionError(f"Semantic mapping covers {cursor} columns, encoder emitted {encoded_count}.")
    names = tuple(numeric_features + categorical_features)
    return SemanticMap(names=names, groups=tuple(tuple(groups[name]) for name in names))
