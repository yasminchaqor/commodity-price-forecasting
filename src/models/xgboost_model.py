"""XGBoost wrappers around the engineered feature set."""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from src.models import BaseModel


class XGBoostRegressor(BaseModel):
    """Thin wrapper around xgboost.XGBRegressor matching the project's model interface.

    Args:
        params: Hyperparameters forwarded to XGBRegressor (see
            config.yaml `models.xgboost`).
    """

    def __init__(self, params: dict | None = None):
        self.params = params or {}
        self.model = XGBRegressor(**self.params)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> XGBoostRegressor:
        """Fit the underlying XGBRegressor."""
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict next-day close price."""
        return self.model.predict(X_test)


class XGBoostClassifier(BaseModel):
    """Thin wrapper around xgboost.XGBClassifier matching the project's model interface.

    Args:
        params: Hyperparameters forwarded to XGBClassifier (see
            config.yaml `models.xgboost`).
    """

    def __init__(self, params: dict | None = None):
        self.params = params or {}
        self.model = XGBClassifier(**self.params, eval_metric="logloss")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> XGBoostClassifier:
        """Fit the underlying XGBClassifier."""
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict next-day direction (1 = up, 0 = down)."""
        return self.model.predict(X_test)
