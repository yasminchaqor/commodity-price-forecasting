"""Feature engineering for the WTI forecasting dataset.

Every feature at row t is computed only from data available up to and
including t (lags, trailing rolling windows). Targets are the only columns
that look forward, via an explicit shift(-1), and are kept separate so it's
obvious at a glance which columns are legitimate model inputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def add_price_lags(df: pd.DataFrame, lags: list[int], price_col: str = "close") -> pd.DataFrame:
    """Add lagged price columns.

    Args:
        df: DataFrame indexed by date with a price column.
        lags: List of lag sizes in trading days, e.g. [1, 5, 20].
        price_col: Name of the price column to lag.

    Returns:
        Copy of df with one `lag_{n}` column added per entry in lags.
    """
    out = df.copy()
    for lag in lags:
        out[f"lag_{lag}"] = out[price_col].shift(lag)
    return out


def add_moving_averages(
    df: pd.DataFrame, windows: list[int], price_col: str = "close"
) -> pd.DataFrame:
    """Add trailing simple moving averages.

    Args:
        df: DataFrame indexed by date with a price column.
        windows: List of window sizes in trading days, e.g. [5, 20, 50].
        price_col: Name of the price column to average.

    Returns:
        Copy of df with one `ma_{n}` column added per entry in windows.
    """
    out = df.copy()
    for window in windows:
        out[f"ma_{window}"] = out[price_col].rolling(window=window).mean()
    return out


def add_volatility(df: pd.DataFrame, window: int, price_col: str = "close") -> pd.DataFrame:
    """Add rolling volatility of daily returns.

    Args:
        df: DataFrame indexed by date with a price column.
        window: Rolling window size in trading days.
        price_col: Name of the price column to compute returns from.

    Returns:
        Copy of df with a `volatility_{window}` column (std of daily pct
        returns over the trailing window).
    """
    out = df.copy()
    returns = out[price_col].pct_change()
    out[f"volatility_{window}"] = returns.rolling(window=window).std()
    return out


def add_rsi(df: pd.DataFrame, window: int = 14, price_col: str = "close") -> pd.DataFrame:
    """Add the Relative Strength Index.

    Args:
        df: DataFrame indexed by date with a price column.
        window: RSI lookback window in trading days, typically 14.
        price_col: Name of the price column.

    Returns:
        Copy of df with an `rsi_{window}` column in [0, 100].
    """
    out = df.copy()
    delta = out[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)
    out[f"rsi_{window}"] = rsi
    return out


def add_macro_deltas(df: pd.DataFrame, macro_cols: list[str], window: int = 5) -> pd.DataFrame:
    """Add level + delta features for macro columns already present in df.

    Args:
        df: DataFrame indexed by date containing raw macro columns.
        macro_cols: Names of macro columns to derive deltas from. The level
            columns are assumed to already exist in df.
        window: Number of trading days over which to compute the delta.

    Returns:
        Copy of df with one `{col}_delta_{window}` column added per entry
        in macro_cols.
    """
    out = df.copy()
    for col in macro_cols:
        out[f"{col}_delta_{window}"] = out[col].diff(window)
    return out


def create_targets(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """Add regression and classification targets for next-day price.

    Args:
        df: DataFrame indexed by date with a price column.
        price_col: Name of the price column to predict.

    Returns:
        Copy of df with `target_reg` (next day's close) and `target_clf`
        (1 if next day's close is higher than today's, else 0) added.
    """
    out = df.copy()
    out["target_reg"] = out[price_col].shift(-1)
    out["target_clf"] = (out["target_reg"] > out[price_col]).astype("Int64")
    out.loc[out["target_reg"].isna(), "target_clf"] = pd.NA
    return out


def build_feature_dataset(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Run the full feature pipeline and drop rows with any remaining NaNs.

    Args:
        df: Raw merged price + macro DataFrame, as returned by
            `data_loader.load_and_merge_data`.
        config: Parsed project config (see config.yaml `features` section).

    Returns:
        Feature-complete DataFrame with no NaNs, ready for modeling.
    """
    feat_cfg = config["features"]
    macro_cols = [name.lower() for name in config["data"]["fred_series"].values()]

    out = df.copy()
    out = add_price_lags(out, feat_cfg["price_lags"])
    out = add_moving_averages(out, feat_cfg["moving_averages"])
    out = add_volatility(out, feat_cfg["volatility_window"])
    out = add_rsi(out, feat_cfg["rsi_window"])
    out = add_macro_deltas(out, macro_cols, feat_cfg["macro_delta_window"])
    out = create_targets(out)

    out = out.dropna()
    return out


def save_processed_data(df: pd.DataFrame, config: dict) -> Path:
    """Save the final feature dataset to data/processed/.

    Args:
        df: Feature-complete DataFrame from build_feature_dataset.
        config: Parsed project config.

    Returns:
        Path the data was written to.
    """
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "features.csv"
    df.to_csv(out_path)
    return out_path


if __name__ == "__main__":
    from src.data_loader import load_config

    cfg = load_config()
    raw_path = PROJECT_ROOT / cfg["data"]["raw_dir"] / "merged_raw.csv"
    raw_df = pd.read_csv(raw_path, index_col="date", parse_dates=True)

    features_df = build_feature_dataset(raw_df, cfg)
    saved_path = save_processed_data(features_df, cfg)
    print(f"Saved {len(features_df)} rows x {len(features_df.columns)} cols to {saved_path}")
