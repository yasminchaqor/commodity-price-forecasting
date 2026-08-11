# Commodity Price Forecasting — WTI Crude Oil

Predicting the next-day price (and direction) of WTI crude oil (`CL=F`) from price history and macro indicators, then checking whether that prediction is worth anything in a simple trading backtest.

## 1. Problem

Oil is one of the hardest commodities to forecast: it's driven by OPEC supply decisions, geopolitical shocks, USD strength, and inventory data that arrive on their own noisy schedules — none of which are fully priced into a rolling window of past closes. That's exactly what makes it interesting. If a simple feature set and a handful of models can extract *any* reliable signal beyond "tomorrow looks like today," that's a meaningful result. If they can't, that's a meaningful result too — and matches what efficient-market theory predicts.

This project treats both outcomes as valid. The goal isn't to prove a trading edge; it's to build a clean, reproducible pipeline and report what it actually finds.

## 2. Data

| Source | Series | Role |
|---|---|---|
| `yfinance` | `CL=F` (WTI front-month future) | Target price, OHLCV |
| FRED (`fredapi`) | `DTWEXBGS` (trade-weighted USD index, broad) | Dollar strength proxy for DXY |
| FRED (`fredapi`) | `DGS10` (10-year Treasury yield) | Macro/rate environment |

- **Period:** configurable in [config.yaml](config.yaml), defaults to 2019–2024 (~5 years of trading days).
- **Alignment:** yfinance and FRED follow different holiday calendars. Macro series are reindexed onto the price calendar and forward-filled — so a given row only ever uses the most recently *known* macro reading, never an interpolated or future one.
- **Features** ([src/features.py](src/features.py)): price lags (1/5/20d), moving averages (5/20/50d), rolling volatility (20d), RSI(14), and macro levels + 5-day deltas.
- **Targets:** two versions, built from the same feature set —
  - `target_reg`: next-day closing price (regression)
  - `target_clf`: next-day direction, up/down (classification)

## 3. Approach

Three models, deliberately in order of increasing complexity, each implementing the same `fit(X_train, y_train)` / `predict(X_test)` interface ([src/models/](src/models/)):

1. **Naive baseline** ([baseline.py](src/models/baseline.py)) — regression: tomorrow's price = today's price. Classification: always predict the majority class. This is the bar every other model has to clear; a model that can't beat it has learned nothing.
2. **XGBoost** ([xgboost_model.py](src/models/xgboost_model.py)) — gradient-boosted trees over the full engineered feature set. Fast, interpretable via feature importance, a reasonable ceiling for tabular signal.
3. **LSTM** ([lstm_model.py](src/models/lstm_model.py)) — small 2-layer, ~50-unit PyTorch LSTM over rolling 20-day sequences, to check whether sequential structure adds anything beyond what the engineered lag/MA features already capture for XGBoost.

Data is split **chronologically** (no shuffling) into train/val/test ([config.yaml](config.yaml), default 70/15/15). Models are fit on train+val and evaluated strictly on the held-out test period.

Every model is trained on **both** targets, so the comparison table has 6 entries (`baseline_reg`, `baseline_clf`, `xgboost_reg`, `xgboost_clf`, `lstm_reg`, `lstm_clf`).

## 4. Results

Run the pipeline to generate results:

```bash
python -m src.pipeline
```

This writes `data/processed/results/comparison_table.csv`, `results.json`, `predictions.csv`, and `equity_curves.csv`. Open [notebooks/02_model_comparison.ipynb](notebooks/02_model_comparison.ipynb) for the plotted predictions-vs-actual and equity curve comparison.

Latest run (2019–2024 WTI, 70/15/15 chronological split, 5bps transaction cost):

**Regression — predict next-day price**

| Model | MAE | RMSE | MAPE | Strategy Sharpe | Strategy cum. return | Strategy max DD | Buy & hold cum. return |
|---|---|---|---|---|---|---|---|
| Naive baseline | 1.07 | 1.36 | 1.42% | 0.00 | 0.0% | 0.0% | -9.1% |
| XGBoost | 1.30 | 1.63 | 1.74% | -0.43 | -10.4% | -22.2% | -9.1% |
| LSTM | 7.13 | 8.93 | 8.95% | 2.24 | +17.8% | -4.3% | -12.7% |

**Classification — predict next-day direction**

| Model | Accuracy | F1 | Strategy Sharpe | Strategy cum. return | Strategy max DD | Buy & hold cum. return |
|---|---|---|---|---|---|---|
| Naive baseline (majority class) | 47.5% | 0.644 | -0.24 | -9.1% | -24.3% | -9.1% |
| XGBoost | 50.2% | 0.592 | -0.43 | -11.1% | -26.5% | -9.1% |
| LSTM | 55.5% | 0.608 | 0.39 | +5.1% | -17.7% | -12.7% |

*(Buy & hold differs slightly between the two tables because the LSTM's rolling-sequence input consumes its first 19 test-period rows; each model's buy & hold benchmark is computed over the exact same dates it was evaluated on. Full numbers in `data/processed/results/comparison_table.csv`.)*

**Reading these honestly:** on this run, XGBoost's MAE is *worse* than just predicting "tomorrow = today," and none of the classifiers clear 56% accuracy — consistent with the efficient-market framing in the limitations section below, not a failure of the pipeline. The LSTM regression's Sharpe of 2.24 is the only backtest result approaching the "too good, check for a bug" zone flagged below; it's driven by a small number of correctly-timed large moves in a short test window, not a robust edge — see [notebooks/02_model_comparison.ipynb](notebooks/02_model_comparison.ipynb) for the equity curve.

## 5. Honest limitations

- **Efficient markets:** WTI futures are heavily traded and macro-driven; a handful of technical + macro features are not expected to reveal a large, stable edge. If the ML metrics (MAE/RMSE/accuracy) only marginally beat the naive baseline, that's the expected, credible outcome — not a bug.
- **Non-stationarity:** the relationship between price, USD, and rates shifts across regimes (COVID crash, 2022 rate-hike cycle, etc.). A model trained on one regime and tested on another will degrade, and this backtest does not attempt walk-forward re-training.
- **The backtest may not beat buy-and-hold**, and that's fine to report as-is. A **Sharpe ratio above ~3 on this kind of simple daily backtest is a red flag for a bug** (look-ahead leakage, mis-aligned dates, or costs not applied) — not a result to celebrate.
- **Transaction costs are simplified**: a flat basis-point charge per position change, not a realistic futures-market cost/slippage model.
- **Small model, small search:** the LSTM uses fixed hyperparameters (no tuning sweep) and no early stopping on the validation set — it's included to check whether sequence modeling helps at all, not to be a tuned production model.

## 6. Reproduce

```bash
git clone <this-repo>
cd commodity-price-forecasting
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Get a free FRED API key at [fred.stlouisfed.org](https://fred.stlouisfed.org) (Account → API Keys), then create `.env`:

```
FRED_API_KEY=your_key_here
```

Run the pipeline end to end (fetches data if `data/raw/`/`data/processed/` are empty, trains all 6 models, runs the backtest, writes results):

```bash
python -m src.pipeline
```

Run the test suite:

```bash
pytest -v
```

Explore interactively:

```bash
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_model_comparison.ipynb
```

### Project structure

```
commodity-price-forecasting/
├── src/
│   ├── data_loader.py    # yfinance + FRED fetch & calendar alignment
│   ├── features.py       # lags, MAs, volatility, RSI, macro deltas, targets
│   ├── models/            # baseline / xgboost / lstm, common fit/predict interface
│   ├── backtest.py        # long/flat strategy, Sharpe/drawdown/win rate, costs
│   ├── evaluate.py        # ML + financial metrics, comparison table
│   └── pipeline.py        # ties it all together
├── notebooks/              # EDA + model comparison (call into src/, no duplicated logic)
├── tests/                  # feature correctness, no look-ahead, date alignment
└── .github/workflows/      # pytest on every push
```
