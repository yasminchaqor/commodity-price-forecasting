"""End-to-end pipeline: load data -> features -> split -> train -> backtest -> evaluate.

Ties together data_loader, features, models, backtest, and evaluate into a
single reproducible run. This is what the notebooks call into and what CI
could run as a smoke test; it contains no modeling logic of its own.
"""

from __future__ import annotations

import os

# XGBoost and PyTorch each bundle their own OpenMP runtime; initializing both
# in one process can deadlock or segfault unless threading is constrained.
# Must be set before xgboost/torch are imported anywhere below.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import pandas as pd

from src.backtest import positions_from_classification, positions_from_regression, run_backtest
from src.data_loader import load_and_merge_data, load_config, save_raw_data
from src.evaluate import (
    build_comparison_table,
    classification_metrics,
    regression_metrics,
    save_results,
)
from src.features import build_feature_dataset, save_processed_data
from src.models.baseline import NaiveClassifier, NaiveRegressor
from src.models.lstm_model import LSTMClassifier, LSTMRegressor
from src.models.xgboost_model import XGBoostClassifier, XGBoostRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NON_FEATURE_COLS = {"target_reg", "target_clf"}


def split_data(df: pd.DataFrame, train_frac: float, val_frac: float) -> tuple[pd.DataFrame, ...]:
    """Chronologically split a DataFrame into train/val/test with no shuffling.

    Args:
        df: DataFrame sorted by date ascending.
        train_frac: Fraction of rows for training.
        val_frac: Fraction of rows for validation.

    Returns:
        (train_df, val_df, test_df).
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def get_or_build_features(config: dict) -> pd.DataFrame:
    """Load the processed feature dataset, building it from scratch if absent.

    Args:
        config: Parsed project config.

    Returns:
        Feature-complete DataFrame (see features.build_feature_dataset).
    """
    processed_path = PROJECT_ROOT / config["data"]["processed_dir"] / "features.csv"
    if processed_path.exists():
        return pd.read_csv(processed_path, index_col="date", parse_dates=True)

    raw_path = PROJECT_ROOT / config["data"]["raw_dir"] / "merged_raw.csv"
    if raw_path.exists():
        raw_df = pd.read_csv(raw_path, index_col="date", parse_dates=True)
    else:
        raw_df = load_and_merge_data(
            config["data"]["start_date"], config["data"]["end_date"], config
        )
        save_raw_data(raw_df, config)

    features_df = build_feature_dataset(raw_df, config)
    save_processed_data(features_df, config)
    return features_df


def run_pipeline(config: dict | None = None) -> dict:
    """Run the full pipeline and return everything the notebooks need.

    Args:
        config: Parsed project config. Loaded from config.yaml if not given.

    Returns:
        Dict with keys "results" (per-model metrics, see evaluate.py),
        "comparison_table" (pd.DataFrame), "predictions" (pd.DataFrame of
        test-period predictions per model), and "equity_curves"
        (pd.DataFrame of backtest equity curves per model).
    """
    if config is None:
        config = load_config()

    df = get_or_build_features(config)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    train, val, test = split_data(df, config["split"]["train"], config["split"]["val"])
    train_full = pd.concat([train, val])

    X_train, X_test = train_full[feature_cols], test[feature_cols]
    y_train_reg, y_test_reg = train_full["target_reg"], test["target_reg"]
    y_train_clf, y_test_clf = train_full["target_clf"].astype(int), test["target_clf"].astype(int)

    seq_len = config["models"]["lstm"]["sequence_length"]

    model_specs = {
        "baseline": {
            "reg": NaiveRegressor(),
            "clf": NaiveClassifier(),
            "offset": 0,
        },
        "xgboost": {
            "reg": XGBoostRegressor(config["models"]["xgboost"]),
            "clf": XGBoostClassifier(config["models"]["xgboost"]),
            "offset": 0,
        },
        "lstm": {
            "reg": LSTMRegressor(config["models"]["lstm"]),
            "clf": LSTMClassifier(config["models"]["lstm"]),
            "offset": seq_len - 1,
        },
    }

    results: dict[str, dict] = {}
    predictions: dict[str, pd.Series] = {}
    equity_curves: dict[str, pd.Series] = {}

    for model_name, spec in model_specs.items():
        offset = spec["offset"]
        test_index = test.index[offset:]
        close_test = test["close"].iloc[offset:]
        returns_test = (y_test_reg.iloc[offset:] - close_test) / close_test

        # Regression
        reg_model = spec["reg"].fit(X_train, y_train_reg)
        pred_reg = reg_model.predict(X_test)
        ml_metrics_reg = regression_metrics(y_test_reg.iloc[offset:].to_numpy(), pred_reg)
        positions_reg = pd.Series(
            positions_from_regression(pred_reg, close_test.to_numpy()), index=test_index
        )
        backtest_reg = run_backtest(
            returns_test,
            positions_reg,
            config["backtest"]["transaction_cost_bps"],
            config["backtest"]["initial_capital"],
        )
        results[f"{model_name}_reg"] = {"ml_metrics": ml_metrics_reg, "backtest": backtest_reg}
        predictions[f"{model_name}_reg"] = pd.Series(pred_reg, index=test_index)
        equity_curves[f"{model_name}_reg"] = backtest_reg["strategy"]["equity_curve"]

        # Classification
        clf_model = spec["clf"].fit(X_train, y_train_clf)
        pred_clf = clf_model.predict(X_test)
        ml_metrics_clf = classification_metrics(y_test_clf.iloc[offset:].to_numpy(), pred_clf)
        positions_clf = pd.Series(positions_from_classification(pred_clf), index=test_index)
        backtest_clf = run_backtest(
            returns_test,
            positions_clf,
            config["backtest"]["transaction_cost_bps"],
            config["backtest"]["initial_capital"],
        )
        results[f"{model_name}_clf"] = {"ml_metrics": ml_metrics_clf, "backtest": backtest_clf}
        predictions[f"{model_name}_clf"] = pd.Series(pred_clf, index=test_index)
        equity_curves[f"{model_name}_clf"] = backtest_clf["strategy"]["equity_curve"]

    equity_curves["buy_and_hold"] = backtest_reg["buy_and_hold"]["equity_curve"]

    comparison_table = build_comparison_table(results)
    csv_path, json_path = save_results(comparison_table, results, config)

    results_dir = PROJECT_ROOT / config["data"]["results_dir"]
    pd.DataFrame(predictions).to_csv(results_dir / "predictions.csv")
    pd.DataFrame(equity_curves).to_csv(results_dir / "equity_curves.csv")

    print(f"Comparison table saved to {csv_path}")
    print(f"Full results saved to {json_path}")
    print(comparison_table)

    return {
        "results": results,
        "comparison_table": comparison_table,
        "predictions": pd.DataFrame(predictions),
        "equity_curves": pd.DataFrame(equity_curves),
    }


if __name__ == "__main__":
    run_pipeline()
