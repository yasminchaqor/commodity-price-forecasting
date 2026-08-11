"""Tests for src/data_loader.py.

Network calls (yfinance, FRED) are never exercised here — CI has no API
key and no guaranteed network access. Instead we monkeypatch the fetch
functions and focus on what this module is actually responsible for:
config loading and date alignment across series with different calendars.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import data_loader


def test_load_config_has_expected_sections():
    config = data_loader.load_config()
    assert "data" in config
    assert "features" in config
    assert "split" in config
    assert config["data"]["ticker"] == "CL=F"


def test_load_and_merge_data_aligns_mismatched_calendars(monkeypatch):
    price_dates = pd.date_range("2024-01-01", periods=6, freq="B")
    prices = pd.DataFrame(
        {
            "open": range(6),
            "high": range(6),
            "low": range(6),
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "volume": range(6),
        },
        index=price_dates,
    )
    prices.index.name = "date"

    # Macro series published on a sparser, non-identical calendar (simulates
    # a Fed holiday that isn't an NYSE holiday, plus a reporting gap).
    macro_dates = [price_dates[0], price_dates[2], price_dates[4]]
    macro = pd.Series([10.0, 11.0, 12.0], index=pd.Index(macro_dates, name="date"), name="dgs10")

    monkeypatch.setattr(data_loader, "fetch_price_data", lambda *a, **k: prices)
    monkeypatch.setattr(data_loader, "fetch_fred_series", lambda *a, **k: macro)

    config = {
        "data": {
            "ticker": "CL=F",
            "fred_series": {"treasury_10y": "DGS10"},
        }
    }

    merged = data_loader.load_and_merge_data("2024-01-01", "2024-01-10", config)

    assert len(merged) == len(prices)
    assert list(merged.index) == list(prices.index)
    # Forward fill: day after a macro reading carries that same value forward.
    assert merged.loc[price_dates[1], "dgs10"] == 10.0
    assert merged.loc[price_dates[3], "dgs10"] == 11.0
    assert merged.loc[price_dates[4], "dgs10"] == 12.0
    # No look-ahead: the filled value never comes from a future reading.
    assert merged.loc[price_dates[3], "dgs10"] != 12.0


def test_fetch_fred_series_requires_api_key(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(ValueError):
        data_loader.fetch_fred_series("DGS10", "2024-01-01", "2024-01-10", api_key=None)
