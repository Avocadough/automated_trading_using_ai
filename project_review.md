# Project Review: `automated_trading_using_ai`

> **Last Updated:** 2026-03-20 | **Status:** 🏰 Institutional / Thesis-Ready / thanos

## 1. System Architecture

```mermaid
graph TD;
    A[download_klines.py] -->|1H BTCUSDT 2022-2024| B[Raw Parquet]
    B --> C[optimize_spa_ga.py]
    C -->|Walk-Forward GA: Sharpe+Sortino+DD| D[make_features_spa.py]
    B --> D
    D -->|19 I(0) Stationary Features| E[CryptoTradingEnv]
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
- **19 features**, all I(0) stationary, scale-invariant, and bounded.
- RSI/Stochastic rescaled [-1,1], MACD normalized by close, `vol_regime` for regime detection.
- Runtime ADF stationarity audit.

### ✅ RL Environment (`rl_env/crypto_env.py`)
- **Differential Sharpe Ratio** reward (Moody & Saffell 1998).
- **Quadratic drawdown penalty** beyond 2%.
- **Liquidation** at 50% equity loss (margin call simulation).
- **5-dim account state:** pos_frac, unrealized_pct, free_margin, drawdown, steps_since_trade.

### ✅ Neural Architecture (`models/custom_policy.py`)
- **Conv1D × 2** (k=3, 64ch): local candlestick pattern extraction.
- **LSTM × 2** (128 hidden): temporal regime tracking.
- **LayerNorm + Dropout(0.1):** regularization for noisy financial data.
- **Orthogonal Init + forget-gate bias=1:** PPO convergence best practice.

### ✅ Training (`train/train_ppo_spa.py`)
- PPO with `gamma=0.995`, `ent_coef=0.02`, linear LR decay `3e-4→0`.
- Train env: DD penalty ON, shaping ON.
- Eval env: all shaping OFF (clean measurement).
- No data leak: norm stats computed from training split only.

### ✅ Evaluation (`eval/`)
- **Academic Report** (`academic_report.py`): Agent vs B&H, Underwater, Monthly Heatmap, Train vs Test Sharpe.
- **Institutional Tearsheet** (`institutional_tearsheet.py`): VaR/CVaR, Monte Carlo permutation test, Alpha-Beta decomposition, capacity estimation.

### ✅ Live Trading (`live/paper_trader.py`)
- **CCXT** for Binance Futures data.
- **Train-serve parity:** imports feature functions directly from `make_features_spa.py`, loads `_meta.json` and `norm_mu/std`.
- **Portfolio tracking:** fees, slippage, realized PnL, equity HWM.
- **Production logging:** console + file, graceful shutdown.

---

## 3. Remaining Steps

### 🟢 Priority 1: Train on H100
```bash
sbatch train_ppo_spa.sh    # 5M steps ≈ 6h on H100
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
