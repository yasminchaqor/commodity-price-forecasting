"""Metrics and final comparison table across baseline / XGBoost / LSTM."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, and MAPE.

    Args:
        y_true: Ground-truth next-day prices.
        y_pred: Predicted next-day prices.

    Returns:
        Dict with mae, rmse, mape.
    """
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute accuracy, F1, and the confusion matrix.

    Args:
        y_true: Ground-truth direction labels (0/1).
        y_pred: Predicted direction labels (0/1).

    Returns:
        Dict with accuracy, f1, and confusion_matrix (as a nested list,
        rows = true class, columns = predicted class).
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def build_comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """Flatten per-model ML + backtest metrics into one comparison table.

    Args:
        results: Mapping of model name -> {"ml_metrics": dict,
            "backtest": {"strategy": dict, "buy_and_hold": dict}}. Each
            backtest sub-dict may contain an "equity_curve" key, which is
            dropped before flattening.

    Returns:
        DataFrame indexed by model name, one column per metric.
    """
    rows = {}
    for model_name, model_results in results.items():
        row = dict(model_results.get("ml_metrics", {}))
        row.pop("confusion_matrix", None)

        backtest = model_results.get("backtest", {})
        strategy = {k: v for k, v in backtest.get("strategy", {}).items() if k != "equity_curve"}
        for k, v in strategy.items():
            row[f"strategy_{k}"] = v

        buy_and_hold = {
            k: v for k, v in backtest.get("buy_and_hold", {}).items() if k != "equity_curve"
        }
        for k, v in buy_and_hold.items():
            row[f"buy_and_hold_{k}"] = v

        rows[model_name] = row

    return pd.DataFrame.from_dict(rows, orient="index")


def save_results(table: pd.DataFrame, results: dict[str, dict], config: dict) -> tuple[Path, Path]:
    """Persist the comparison table and full results to data/processed/results/.

    Args:
        table: Output of build_comparison_table.
        results: Full nested results dict (including confusion matrices),
            used for the JSON export. Equity curves are dropped since
            they're not JSON-serializable as-is.
        config: Parsed project config.

    Returns:
        Tuple of (csv_path, json_path).
    """
    results_dir = PROJECT_ROOT / config["data"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "comparison_table.csv"
    table.to_csv(csv_path)

    json_safe = {}
    for model_name, model_results in results.items():
        entry = {"ml_metrics": model_results.get("ml_metrics", {})}
        backtest = model_results.get("backtest", {})
        entry["backtest"] = {
            side: {k: v for k, v in metrics.items() if k != "equity_curve"}
            for side, metrics in backtest.items()
        }
        json_safe[model_name] = entry

    json_path = results_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(json_safe, f, indent=2)

    return csv_path, json_path
