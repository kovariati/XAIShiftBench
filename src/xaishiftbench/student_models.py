"""Model pipelines for the UCI Student Performance institutional pilot."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .explainers import explain_lightgbm_native_contrib, explain_logistic_centered
from .semantic import SemanticMap, semantic_map_from_preprocessor


@dataclass
class StudentFittedModel:
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

    def raw_score(self, x: pd.DataFrame) -> np.ndarray:
        encoded = self.transform(x)
        if self.model_name == "logistic":
            return np.asarray(self.estimator.decision_function(encoded), dtype=float)
        return np.asarray(self.estimator.booster_.predict(encoded, raw_score=True), dtype=float)


def build_student_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    numeric = Pipeline([("scale", StandardScaler())])
    categorical = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float))]
    )
    return ColumnTransformer(
        [("num", numeric, numeric_features), ("cat", categorical, categorical_features)],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )


def fit_student_model(
    model_name: str,
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    features: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    seed: int,
) -> StudentFittedModel:
    preprocessor = build_student_preprocessor(numeric_features, categorical_features)
    encoded = np.asarray(preprocessor.fit_transform(train_x[features]), dtype=float)
    if model_name == "logistic":
        estimator: object = LogisticRegression(
            C=1.0,
            max_iter=5000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=seed,
        )
    elif model_name == "lightgbm":
        estimator = LGBMClassifier(
            objective="binary",
            n_estimators=180,
            learning_rate=0.035,
            num_leaves=15,
            min_child_samples=12,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=1.0,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
    else:
        raise ValueError(model_name)
    estimator.fit(encoded, np.asarray(train_y, dtype=int))
    semantic_map = semantic_map_from_preprocessor(
        preprocessor, numeric_features, categorical_features
    )
    return StudentFittedModel(
        model_name=model_name,
        features=features,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        preprocessor=preprocessor,
        estimator=estimator,
        semantic_map=semantic_map,
        encoded_background=encoded,
    )
