"""Simple long/flat backtest driven by model predictions.

Positions are decided at the close of day t using only information known
by then (the model's prediction for day t+1), then held to earn the actual
t -> t+1 return. This mirrors how the position would really be entered:
you can't trade on tomorrow's close today. Transaction costs are charged
in basis points of notional every time the position changes, so a model
that flips its call every day gets penalized for it, and a raw Sharpe > 3
should be read as a bug, not a win (see README limitations).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def positions_from_regression(pred_prices: np.ndarray, current_prices: np.ndarray) -> np.ndarray:
    """Convert next-day price predictions into long(1)/flat(0) positions.

    Args:
        pred_prices: Predicted next-day close prices.
        current_prices: Today's close prices, same length and alignment.

    Returns:
        Array of 1 (long) where the model predicts a rise, 0 (flat) otherwise.
    """
    return (pred_prices > current_prices).astype(int)


def positions_from_classification(pred_labels: np.ndarray) -> np.ndarray:
    """Convert direction predictions (1=up, 0=down) into long/flat positions.

    Args:
        pred_labels: Predicted direction labels.

    Returns:
        Array of positions, identical to pred_labels (1 = long, 0 = flat).
    """
    return np.asarray(pred_labels).astype(int)


def compute_strategy_returns(
    actual_returns: pd.Series,
    positions: pd.Series,
    transaction_cost_bps: float,
) -> pd.Series:
    """Turn a position series into net-of-cost daily strategy returns.

    Args:
        actual_returns: Realized close-to-close pct returns of the underlying,
            indexed by the date the position was held into.
        positions: Position (0 or 1) decided at the close of the prior day,
            same index as actual_returns.
        transaction_cost_bps: Round-trip-agnostic cost, in basis points of
            notional, charged whenever the position changes.

    Returns:
        Series of net daily strategy returns, same index as actual_returns.
    """
    gross = positions * actual_returns
    position_changes = positions.diff().abs().fillna(positions.iloc[0])
    costs = position_changes * (transaction_cost_bps / 10_000)
    return gross - costs


def compute_metrics(strategy_returns: pd.Series, initial_capital: float) -> dict:
    """Compute standard backtest metrics from a daily return series.

    Args:
        strategy_returns: Daily net returns.
        initial_capital: Starting capital used to build the equity curve.

    Returns:
        Dict with cumulative_return, sharpe_ratio, max_drawdown, win_rate,
        and the equity_curve (pd.Series).
    """
    equity_curve = initial_capital * (1 + strategy_returns).cumprod()
    cumulative_return = equity_curve.iloc[-1] / initial_capital - 1

    std = strategy_returns.std()
    sharpe_ratio = (
        float(strategy_returns.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR)) if std > 0 else 0.0
    )

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    max_drawdown = float(drawdown.min())

    traded = strategy_returns[strategy_returns != 0]
    win_rate = float((traded > 0).mean()) if len(traded) > 0 else 0.0

    return {
        "cumulative_return": float(cumulative_return),
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "equity_curve": equity_curve,
    }


def run_backtest(
    actual_returns: pd.Series,
    positions: pd.Series,
    transaction_cost_bps: float = 5,
    initial_capital: float = 10_000,
) -> dict:
    """Run the model-driven strategy and a buy-and-hold benchmark side by side.

    Args:
        actual_returns: Realized close-to-close pct returns of the underlying.
        positions: Model-driven positions (0/1), same index as actual_returns.
        transaction_cost_bps: Cost per position change, in basis points.
        initial_capital: Starting capital for both strategies.

    Returns:
        Dict with "strategy" and "buy_and_hold" sub-dicts, each containing
        the metrics from compute_metrics.
    """
    strategy_returns = compute_strategy_returns(actual_returns, positions, transaction_cost_bps)
    strategy_metrics = compute_metrics(strategy_returns, initial_capital)

    bh_positions = pd.Series(1, index=actual_returns.index)
    bh_returns = compute_strategy_returns(actual_returns, bh_positions, transaction_cost_bps)
    bh_metrics = compute_metrics(bh_returns, initial_capital)

    return {"strategy": strategy_metrics, "buy_and_hold": bh_metrics}
