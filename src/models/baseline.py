"""Naive baselines: the minimum bar every other model must clear.

For regression, "tomorrow's price = today's price" (a random walk
assumption). For classification, always predict the majority class seen in
training. Neither baseline learns anything from the features — they exist
so that a model with a lower MAE or higher accuracy than these can
actually claim to have found signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models import BaseModel


class NaiveRegressor(BaseModel):
    """Predicts tomorrow's close as today's close.

    Args:
        price_col: Column in X holding today's price.
    """

    def __init__(self, price_col: str = "close"):
        self.price_col = price_col

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> NaiveRegressor:
        """No-op: this model has no parameters to learn."""
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Return today's price as the forecast for tomorrow."""
        return X_test[self.price_col].to_numpy()


class NaiveClassifier(BaseModel):
    """Always predicts the majority class observed at fit time."""

    def __init__(self):
        self.majority_class_: int | None = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> NaiveClassifier:
        """Store the most frequent label in y_train."""
        self.majority_class_ = int(y_train.mode().iloc[0])
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Return an array filled with the stored majority class."""
        if self.majority_class_ is None:
            raise RuntimeError("NaiveClassifier must be fit before predict.")
        return np.full(len(X_test), self.majority_class_)
