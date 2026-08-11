"""Common model interface shared by baseline, XGBoost, and LSTM models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Minimal fit/predict contract every model in this project implements."""

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> BaseModel:
        """Fit the model on training data.

        Args:
            X_train: Feature matrix.
            y_train: Target vector (regression) or labels (classification).

        Returns:
            self, to allow chaining.
        """

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict on new data.

        Args:
            X_test: Feature matrix, same columns as X_train.

        Returns:
            1D array of predictions, one per row of X_test.
        """
