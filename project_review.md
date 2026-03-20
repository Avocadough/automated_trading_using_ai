# Project Review: `automated_trading_using_ai`

> **Last Updated:** 2026-03-20 | **Status:** 🏰 Institutional / Thesis-Ready / thanos

## 1. System Architecture

```mermaid
graph TD;
    A[download_klines.py] -->|1H BTCUSDT 2022-2024| B[Raw Parquet]
    B --> C[optimize_spa_ga.py]
    C -->|Walk-Forward GA: Sharpe+Sortino+DD| D[make_features_spa.py]
    B --> D
    D -->|23 I(0) Stationary Features| E[CryptoTradingEnv]
    E -->|Diff Sharpe Reward + Liquidation| F[train_ppo_spa.py]
    F -->|CNN+LSTM PPO| G[eval_ppo_spa.py]
    G --> H[Academic Report]
    G --> I[Institutional Tearsheet]
    F --> J[paper_trader.py]
    J -->|CCXT Real-Time| K[Live Inference]
```

---

## 2. Component Review

### ✅ Data Pipeline (`data_ingest/download_klines.py`)
- Adaptive chunking Binance API downloader. 3 years of 1H BTCUSDT covers bear/sideways/bull regimes.

### ✅ GA Optimizer (`optimize/optimize_spa_ga.py`)
- **Walk-Forward K-fold** temporal validation — strongest anti-overfitting for time series.
- **Composite fitness:** `0.4*Sharpe + 0.3*Sortino + 0.3*(1-DD_penalty)` — prevents selecting high-variance strategies.
- **Elitism:** Top 10% survive unchanged per generation.

### ✅ Feature Engineering (`features/make_features_spa.py`)
- **23 features**, all I(0) stationary, scale-invariant, and strictly bounded to prevent CNN poisoning (e.g., `rvol` clipped to [0, 0.5]).
- RSI/Stochastic rescaled [-1,1], MACD normalized by close and by ATR, `vol_regime` for regime detection.
- **Trend-awareness:** ADX (trend strength), normalized EMA distances (50/200), MACD hist/ATR.
- Runtime ADF stationarity audit.

### ✅ RL Environment (`rl_env/crypto_env.py`)
- **Differential Sharpe Ratio** reward (Moody & Saffell 1998) with reduced 1.0x scale.
- **Incremental Drawdown Penalty:** Triggers only on new peak drawdowns > 5% to prevent the "Rational Cowardice" death spiral.
- **Dynamic Trend Masking:** Causal action interception (at `t`) strictly forbids Shorts when Price > 3% vs EMA200, and Longs when < -3%.
- **Trend-Alignment Reward Boost:** 2.0x reward multiplier for profitable trend-aligned positions. Cures "Volatility Fear" regime bias.
- **Execution:** Precise Binance VIP 0 Taker fees (0.05%) + realistic volatility-scaled slippage (0.015%). Position limit at 50% of equity for improved capital efficiency.
- **Liquidation** at 50% equity loss (margin call simulation).
- **5-dim account state:** pos_frac, unrealized_pct, free_margin, drawdown, steps_since_trade.

### ✅ Neural Architecture (`models/custom_policy.py`)
- **Conv1D × 2** (k=3, 32ch): local candlestick pattern extraction.
- **LSTM × 1** (32 hidden): temporal regime tracking. Shrunk from 262k → 16k params to cure massive OOS degradation.
- **LayerNorm + Dropout(0.3):** strong regularization for noisy financial data.
- **Orthogonal Init + forget-gate bias=1.0:** PPO convergence best practice.

### ✅ Training (`train/train_ppo_spa.py`)
- **LSTM-Safe Optimizations:** `ent_coef=0.01` (prevents 100% flat entropy collapse), `LR=1e-4`, `max_grad_norm=0.5`.
- **Mini-batch Variance:** `batch_size=512` vs `n_steps=4096` and `n_epochs=5` to prevent per-batch curve fitting.
- **Pure PnL Learning:** Zero action-masking limits (deadband, inactivity penalty) during training.
- **Cross-Platform:** OS-aware multiprocessing (`fork` on Linux cluster, `spawn` on Windows).
- **Local Training Ready:** Scaled parallel environments (`--n_envs`) to maximize local CPU/GPU throughput.

### ✅ Evaluation (`eval/`)
- **Academic Report** (`academic_report.py`): Agent vs BTC B&H vs S&P 500 (^GSPC), Underwater, Monthly Heatmap, Train vs Test Sharpe. Includes auto S&P 500 data fetch and alignment via `yfinance`. **Benchmark Sharpe/Sortino comparison** for Agent, BTC B&H, and S&P 500.
- **Institutional Tearsheet** (`institutional_tearsheet.py`): VaR/CVaR, Alpha-Beta decomposition, capacity estimation, and mathematically rigorous **Bootstrap Resampling (with replacement)** for Sharpe p-value significance tests. **S&P 500 equity line** on equity chart and **Benchmark Comparison** section on Page 3.

### ✅ Live Trading (`live/paper_trader.py`)
- **CCXT** for Binance Futures data.
- **Train-serve parity:** imports feature functions directly from `make_features_spa.py`, loads `_meta.json` and `norm_mu/std`.
- **Portfolio tracking:** fees, slippage, realized PnL, equity HWM.
- **Production logging:** console + file, graceful shutdown.

---

## 3. Remaining Steps

### 🟢 Priority 1: Train (Local)
```bash
python src/train/train_ppo_spa.py    # 5M steps local
```

### 🟢 Priority 2: Evaluate & Generate Reports
```bash
python src/eval/eval_ppo_spa.py --model data/models/ppo_spa_btc_1h.zip --features data/features/btc_1h_spa.parquet --out_dir data/eval
```
- Check: Monte Carlo p-value < 0.05, Train→Test Sharpe degradation < 50%.

### 🟢 Priority 3: Live Demo
```bash
python src/live/paper_trader.py --model data/models/ppo_spa_btc_1h.zip --meta data/features/btc_1h_spa_meta.json
```

### 🟡 Optional: Dashboard
- `src/dashboard/app.py` (Streamlit) — WIP, needs integration with live trade log.

---

## 4. Files Archived
The following legacy files have been moved to `archive/`:
- `paper_trade/`: Superseded by `src/live/paper_trader.py`
- `backtest/`: Old baseline backtesting scripts
- `make_meta.py`, `make_meta_from_parquet.py`: One-off utility scripts
- `plan.md`, `research.md`: Early planning documents
- `scripts/check_gpu.py`: GPU diagnostic utility
