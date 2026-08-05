"""Model pipelines for the South German Credit missingness pilot."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .datasets.south_german_credit import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from .explainers import explain_lightgbm_native_contrib, explain_logistic_centered
from .semantic import SemanticMap, semantic_map_from_preprocessor


@dataclass
class CreditFittedModel:
    model_name: str
    indicator_mode: str
    preprocessor: ColumnTransformer
    estimator: object
    semantic_map: SemanticMap
    encoded_background: np.ndarray

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        transformed = self.preprocessor.transform(x[FEATURES])
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        return np.asarray(transformed, dtype=float)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        encoded = self.transform(x)
        return np.asarray(self.estimator.predict_proba(encoded)[:, 1], dtype=float)

    def explain(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
        encoded = self.transform(x)
        if self.model_name == "logistic":
            result = explain_logistic_centered(
                self.estimator, encoded, self.encoded_background, self.semantic_map
            )
        elif self.model_name == "lightgbm":
            result = explain_lightgbm_native_contrib(self.estimator, encoded, self.semantic_map)
        else:
            raise ValueError(self.model_name)
        return result.encoded, result.semantic, result.base_value


def build_credit_preprocessor(indicator_mode: str) -> ColumnTransformer:
    if indicator_mode not in {"none", "all"}:
        raise ValueError("indicator_mode must be 'none' or 'all'.")
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transformers: list[tuple[str, object, list[str]]] = [
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ]
    if indicator_mode == "all":
        transformers.append(
            (
                "ind",
                MissingIndicator(features="all", sparse=False, error_on_new=False),
                FEATURES,
            )
        )
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def fit_credit_model(
    model_name: str,
    indicator_mode: str,
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    seed: int,
) -> CreditFittedModel:
    preprocessor = build_credit_preprocessor(indicator_mode)
    encoded = np.asarray(preprocessor.fit_transform(train_x[FEATURES]), dtype=float)
    semantic_map = semantic_map_from_preprocessor(
        preprocessor,
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
        indicator_features=FEATURES if indicator_mode == "all" else None,
    )
    if model_name == "logistic":
        estimator = LogisticRegression(
            C=1.0,
            max_iter=3000,
            solver="lbfgs",
            random_state=seed,
        )
    elif model_name == "lightgbm":
        estimator = LGBMClassifier(
            n_estimators=180,
            learning_rate=0.035,
            num_leaves=15,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
    else:
        raise ValueError(model_name)
    estimator.fit(encoded, train_y)
    return CreditFittedModel(
        model_name=model_name,
        indicator_mode=indicator_mode,
        preprocessor=preprocessor,
        estimator=estimator,
        semantic_map=semantic_map,
        encoded_background=encoded,
    )
