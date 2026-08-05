"""Leakage-controlled model pipelines for OULAD temporal transfer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .explainers import explain_lightgbm_native_contrib, explain_logistic_centered
from .semantic import SemanticMap, semantic_map_from_preprocessor

OULAD_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "logistic": {
        "C": 1.0,
        "max_iter": 5000,
        "solver": "lbfgs",
        "class_weight": "balanced",
    },
    "lightgbm": {
        "objective": "binary",
        "n_estimators": 180,
        "learning_rate": 0.035,
        "num_leaves": 15,
        "min_child_samples": 12,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 1.0,
        "class_weight": "balanced",
        "n_jobs": 1,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    },
}


def oulad_model_config(
    model_name: str,
    seed: int | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if model_name not in OULAD_MODEL_CONFIGS:
        raise ValueError(model_name)
    config = dict(OULAD_MODEL_CONFIGS[model_name])
    if overrides:
        config.update(overrides)
    if seed is not None:
        config["random_state"] = int(seed)
    return config


@dataclass
class OULADFittedModel:
    model_name: str
    features: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    preprocessor: ColumnTransformer
    estimator: object
    semantic_map: SemanticMap
    encoded_background: np.ndarray

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        transformed = self.preprocessor.transform(x[self.features])
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        return np.asarray(transformed, dtype=float)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        encoded = self.transform(x)
        return np.asarray(self.estimator.predict_proba(encoded)[:, 1], dtype=float)

    def raw_score(self, x: pd.DataFrame) -> np.ndarray:
        encoded = self.transform(x)
        if self.model_name == "lightgbm":
            return np.asarray(
                self.estimator.booster_.predict(encoded, raw_score=True), dtype=float
            )
        return np.asarray(self.estimator.decision_function(encoded), dtype=float)

    def explain(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
        encoded = self.transform(x)
        if self.model_name == "logistic":
            result = explain_logistic_centered(
                self.estimator, encoded, self.encoded_background, self.semantic_map
            )
        elif self.model_name == "lightgbm":
            result = explain_lightgbm_native_contrib(
                self.estimator, encoded, self.semantic_map
            )
        else:
            raise ValueError(self.model_name)
        return result.encoded, result.semantic, result.base_value


def build_preprocessor(
    numeric: list[str], categorical: list[str]
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, dtype=float
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric),
            ("cat", categorical_pipeline, categorical),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )


def fit_oulad_model(
    model_name: str,
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    features: list[str],
    numeric: list[str],
    categorical: list[str],
    seed: int,
    overrides: dict[str, Any] | None = None,
) -> OULADFittedModel:
    preprocessor = build_preprocessor(numeric, categorical)
    encoded = np.asarray(preprocessor.fit_transform(train_x[features]), dtype=float)
    config = oulad_model_config(model_name, seed, overrides)
    if model_name == "logistic":
        estimator: object = LogisticRegression(**config)
    elif model_name == "lightgbm":
        estimator = LGBMClassifier(**config)
    else:
        raise ValueError(model_name)
    estimator.fit(encoded, np.asarray(train_y, dtype=int))
    semantic_map = semantic_map_from_preprocessor(
        preprocessor, numeric, categorical
    )
    return OULADFittedModel(
        model_name=model_name,
        features=features,
        numeric_features=numeric,
        categorical_features=categorical,
        preprocessor=preprocessor,
        estimator=estimator,
        semantic_map=semantic_map,
        encoded_background=encoded,
    )
