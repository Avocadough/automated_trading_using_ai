# Crypto Trading with Reinforcement Learning

A student research scaffold for exploring PPO-based trading policies on historical BTCUSDT futures data.

> **Research only.** The repository contains historical data, models and evaluation outputs from earlier experiments. They are not evidence of a profitable strategy, live execution, or readiness to trade.

## Workflow

The code covers historical kline ingestion, feature construction, Gymnasium trading environments, PPO training, backtesting and a Streamlit data viewer. Several experimental variants coexist; their inputs, environment settings and splits are not interchangeable.

- [`src/data_ingest/`](src/data_ingest/) — Binance Futures kline download scripts.
- [`src/features/`](src/features/) and [`src/spa/`](src/spa/) — feature engineering, including SPA-based features.
- [`src/rl_env/`](src/rl_env/) and [`src/train/`](src/train/) — trading environments and Stable-Baselines3 PPO training variants.
- [`src/backtest/`](src/backtest/) and [`src/eval/`](src/eval/) — baseline backtests and PPO evaluation.
- [`src/dashboard/app.py`](src/dashboard/app.py) — Streamlit viewer for a feature Parquet file.
- `data/` — tracked historical data, model checkpoints and previous evaluation files; `ppo_logs_spa/` contains prior TensorBoard logs.

## Getting started

From the repository root, create a virtual environment and install the pinned packages in [`requirements.txt`](requirements.txt). PyTorch availability and compatibility depend on the local platform; the requirements file advises installing it separately. No full environment or pipeline run has been verified for this README.

```bash
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -r requirements.txt
```

To inspect the existing feature viewer, the source accepts a Parquet path (the default is `data/features/btc_1m_features.parquet`):

```bash
streamlit run src/dashboard/app.py -- --features data/features/btc_1m_features.parquet
```

For other commands, inspect each script's arguments and output paths before use. The downloaders make network requests and write data files; training and evaluation scripts write models or results and may replace existing outputs. `src/train/train_ppo.py` currently passes `idle_cost_bps` and `idle_cost_ramp` to `CryptoTradingEnv`, whose constructor does not accept them, so that training route requires code repair before it can run. The previous README also referred to feature and baseline script paths that are not present in this revision.

## Evaluation limits

Some scripts use chronological splits and model fees; assumptions differ across variants, including slippage. Selection history, data provenance, leakage controls, and the independence of tracked evaluation outputs have not been established here. Review those details before interpreting any historical result. `src/paper_trade/` contains only an initializer in this revision; a working paper-trading system is not demonstrated.
