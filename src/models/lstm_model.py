"""Small PyTorch LSTM for sequence-based forecasting.

Unlike the baseline and XGBoost models, the LSTM consumes rolling windows
of `sequence_length` consecutive days rather than a single row. As a
result predict() returns one prediction per window, not one per input row:
for `X_test` of length n, the output has length `n - sequence_length + 1`,
aligned to `X_test.index[sequence_length - 1:]`. Callers must slice their
target/backtest series with that same offset before comparing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.models import BaseModel

# Sharing a process with XGBoost's OpenMP threads can deadlock/segfault
# PyTorch's own thread pool; keep this model single-threaded defensively.
torch.set_num_threads(1)


def create_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    """Slide a fixed-length window over a feature matrix.

    Args:
        X: 2D array of shape (n_samples, n_features), already sorted
            chronologically.
        seq_len: Number of consecutive rows per sequence.

    Returns:
        3D array of shape (n_samples - seq_len + 1, seq_len, n_features).
    """
    n = X.shape[0]
    if n < seq_len:
        raise ValueError(f"Need at least {seq_len} rows to build one sequence, got {n}.")
    return np.stack([X[i : i + seq_len] for i in range(n - seq_len + 1)])


class _LSTMNet(nn.Module):
    """2-layer LSTM followed by a linear head producing a single scalar."""

    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        return self.head(last_hidden).squeeze(-1)


class _LSTMBase(BaseModel):
    """Shared training/inference logic for the regression and classification LSTMs."""

    def __init__(self, params: dict | None = None):
        p = params or {}
        self.seq_len = p.get("sequence_length", 20)
        self.hidden_size = p.get("hidden_size", 50)
        self.num_layers = p.get("num_layers", 2)
        self.dropout = p.get("dropout", 0.2)
        self.epochs = p.get("epochs", 50)
        self.batch_size = p.get("batch_size", 32)
        self.lr = p.get("learning_rate", 1e-3)
        self.random_state = p.get("random_state", 42)

        torch.manual_seed(self.random_state)
        self.scaler = StandardScaler()
        self.net: _LSTMNet | None = None
        self.loss_fn: nn.Module | None = None

    def _build_net(self, n_features: int) -> None:
        self.net = _LSTMNet(n_features, self.hidden_size, self.num_layers, self.dropout)

    def _train(self, X_train: pd.DataFrame, y_train: np.ndarray) -> None:
        X_scaled = self.scaler.fit_transform(X_train.to_numpy())
        X_seq = create_sequences(X_scaled, self.seq_len)
        y_seq = y_train[self.seq_len - 1 :]

        self._build_net(n_features=X_scaled.shape[1])
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        y_tensor = torch.tensor(y_seq, dtype=torch.float32)

        self.net.train()
        n = X_tensor.shape[0]
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                optimizer.zero_grad()
                out = self.net(X_tensor[idx])
                loss = self.loss_fn(out, y_tensor[idx])
                loss.backward()
                optimizer.step()

    def _raw_predict(self, X_test: pd.DataFrame) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("Model must be fit before predict.")
        X_scaled = self.scaler.transform(X_test.to_numpy())
        X_seq = create_sequences(X_scaled, self.seq_len)
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)

        self.net.eval()
        with torch.no_grad():
            out = self.net(X_tensor)
        return out.numpy()


class LSTMRegressor(_LSTMBase):
    """LSTM trained with MSE loss to predict next-day close price."""

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.loss_fn = nn.MSELoss()

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> LSTMRegressor:
        """Fit the LSTM on rolling sequences of X_train."""
        self._train(X_train, y_train.to_numpy(dtype=np.float32))
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict next-day close price for each sequence in X_test."""
        return self._raw_predict(X_test)


class LSTMClassifier(_LSTMBase):
    """LSTM trained with BCE loss to predict next-day direction."""

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> LSTMClassifier:
        """Fit the LSTM on rolling sequences of X_train."""
        self._train(X_train, y_train.to_numpy(dtype=np.float32))
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict next-day direction (1 = up, 0 = down) for each sequence in X_test."""
        logits = self._raw_predict(X_test)
        probs = 1 / (1 + np.exp(-logits))
        return (probs > 0.5).astype(int)
