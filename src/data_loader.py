"""Fetch and merge WTI price data with macro series from FRED.

Two data sources with different publishing calendars are combined here:
yfinance follows the NYSE trading calendar, while FRED series follow the
Federal Reserve's own business-day calendar (different holidays, occasional
reporting gaps). All series are aligned onto the price index and forward
filled, so a given row only ever contains information that was already
public as of that date.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from fredapi import Fred

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path = "config.yaml") -> dict:
    """Load the project YAML config.

    Args:
        config_path: Path to the config file, relative to the project root
            or absolute.

    Returns:
        Parsed config as a nested dict.
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path) as f:
        return yaml.safe_load(f)


def fetch_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download historical OHLCV data for a ticker via yfinance.

    Args:
        ticker: Yahoo Finance ticker symbol, e.g. "CL=F" for WTI crude.
        start_date: ISO date string, inclusive.
        end_date: ISO date string, inclusive.

    Returns:
        DataFrame indexed by date with columns open, high, low, close, volume.

    Raises:
        ValueError: If yfinance returns no data for the given range.
    """
    import yfinance as yf

    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No price data returned for {ticker} in [{start_date}, {end_date}]")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.lower)
    df.index.name = "date"
    return df[["open", "high", "low", "close", "volume"]]


def fetch_fred_series(
    series_id: str,
    start_date: str,
    end_date: str,
    api_key: str | None = None,
) -> pd.Series:
    """Download a single macro series from FRED.

    Args:
        series_id: FRED series identifier, e.g. "DGS10".
        start_date: ISO date string, inclusive.
        end_date: ISO date string, inclusive.
        api_key: FRED API key. Falls back to the FRED_API_KEY env var.

    Returns:
        Series indexed by date, named after series_id (lowercased).

    Raises:
        ValueError: If no API key is available.
    """
    key = api_key or os.getenv("FRED_API_KEY")
    if not key:
        raise ValueError(
            "No FRED API key found. Set FRED_API_KEY in .env "
            "(see .env.example) or pass api_key explicitly."
        )

    fred = Fred(api_key=key)
    series = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
    series.index.name = "date"
    series.name = series_id.lower()
    return series


def load_and_merge_data(
    start_date: str,
    end_date: str,
    config: dict | None = None,
) -> pd.DataFrame:
    """Fetch price and macro data and align them onto a single calendar.

    Macro series are reindexed onto the price index and forward filled,
    since e.g. a Monday's 10Y yield reading is still the most recent known
    value on a Tuesday holiday. This avoids introducing NaNs purely from
    calendar mismatches while never pulling in a future value.

    Args:
        start_date: ISO date string, inclusive.
        end_date: ISO date string, inclusive.
        config: Parsed config dict. Loaded from config.yaml if not given.

    Returns:
        DataFrame indexed by date with price columns (open, high, low,
        close, volume) plus one column per configured FRED series.
    """
    if config is None:
        config = load_config()

    ticker = config["data"]["ticker"]
    prices = fetch_price_data(ticker, start_date, end_date)

    merged = prices.copy()
    for _, series_id in config["data"]["fred_series"].items():
        try:
            macro = fetch_fred_series(series_id, start_date, end_date)
        except ValueError:
            raise
        except Exception as exc:  # pragma: no cover - network/API errors
            print(f"Warning: could not fetch FRED series {series_id} ({exc}), skipping.")
            continue
        macro = macro.reindex(merged.index.union(macro.index)).sort_index()
        macro = macro.ffill()
        merged[series_id.lower()] = macro.reindex(merged.index)

    merged = merged.sort_index()
    return merged


def save_raw_data(df: pd.DataFrame, config: dict | None = None) -> Path:
    """Save the merged raw dataset to data/raw/.

    Args:
        df: Merged price + macro DataFrame.
        config: Parsed config dict. Loaded from config.yaml if not given.

    Returns:
        Path the data was written to.
    """
    if config is None:
        config = load_config()

    raw_dir = PROJECT_ROOT / config["data"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / "merged_raw.csv"
    df.to_csv(out_path)
    return out_path


if __name__ == "__main__":
    cfg = load_config()
    data = load_and_merge_data(cfg["data"]["start_date"], cfg["data"]["end_date"], cfg)
    saved_path = save_raw_data(data, cfg)
    print(f"Saved {len(data)} rows to {saved_path}")
