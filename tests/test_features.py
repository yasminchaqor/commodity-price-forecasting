"""Tests for src/features.py: no NaN leakage, correct lags, no look-ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    add_moving_averages,
    add_price_lags,
    add_rsi,
    add_volatility,
    build_feature_dataset,
    create_targets,
)


@pytest.fixture
def price_df() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    rng = np.random.default_rng(42)
    close = 70 + np.cumsum(rng.normal(0, 1, size=100))
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": rng.integers(1000, 5000, size=100),
            "dtwexbgs": 100 + np.cumsum(rng.normal(0, 0.1, size=100)),
            "dgs10": 4 + np.cumsum(rng.normal(0, 0.01, size=100)),
        },
        index=dates,
    )
    df.index.name = "date"
    return df


@pytest.fixture
def config() -> dict:
    return {
        "data": {"fred_series": {"dollar_index": "DTWEXBGS", "treasury_10y": "DGS10"}},
        "features": {
            "price_lags": [1, 5, 20],
            "moving_averages": [5, 20, 50],
            "volatility_window": 20,
            "rsi_window": 14,
            "macro_delta_window": 5,
        },
    }


def test_price_lags_are_correct(price_df):
    out = add_price_lags(price_df, [1, 5])
    assert (out["lag_1"].iloc[10] == price_df["close"].iloc[9]).all()
    assert (out["lag_5"].iloc[10] == price_df["close"].iloc[5]).all()
    assert out["lag_1"].iloc[:1].isna().all()


def test_moving_average_uses_only_past_and_present(price_df):
    out = add_moving_averages(price_df, [5])
    expected = price_df["close"].iloc[6:11].mean()
    assert np.isclose(out["ma_5"].iloc[10], expected)


def test_moving_average_unaffected_by_future_values(price_df):
    """Changing a future close price must not change a past feature value (no look-ahead)."""
    out_before = add_moving_averages(price_df, [20])
    mutated = price_df.copy()
    mutated.iloc[-1, mutated.columns.get_loc("close")] = 99999.0
    out_after = add_moving_averages(mutated, [20])

    assert np.isclose(out_before["ma_20"].iloc[50], out_after["ma_20"].iloc[50])


def test_rsi_bounded_between_0_and_100(price_df):
    out = add_rsi(price_df, window=14)
    valid = out["rsi_14"].dropna()
    assert (valid >= 0).all()
    assert (valid <= 100).all()


def test_volatility_is_nonnegative(price_df):
    out = add_volatility(price_df, window=20)
    valid = out["volatility_20"].dropna()
    assert (valid >= 0).all()


def test_target_reg_is_next_day_close(price_df):
    out = create_targets(price_df)
    assert out["target_reg"].iloc[0] == price_df["close"].iloc[1]
    assert out["target_reg"].iloc[-1] != out["target_reg"].iloc[-1]  # NaN at last row


def test_target_clf_matches_direction_of_target_reg(price_df):
    out = create_targets(price_df)
    valid = out.dropna(subset=["target_reg"])
    expected_up = (valid["target_reg"] > valid["close"]).astype(int)
    assert (valid["target_clf"].astype(int) == expected_up).all()


def test_build_feature_dataset_has_no_nans(price_df, config):
    out = build_feature_dataset(price_df, config)
    assert out.isna().sum().sum() == 0
    assert len(out) > 0


def test_build_feature_dataset_drops_only_incomplete_rows(price_df, config):
    out = build_feature_dataset(price_df, config)
    # The longest lookback here is the 50-day moving average plus the 1-day
    # forward-looking target, so we lose at most 50 rows at the start and 1 at the end.
    assert len(out) >= len(price_df) - 51
