# Research Report: `automated_trading_using_ai`

> **Analyst:** Senior Solutions Architect & Lead Full-Stack Developer
> **Date:** 2026-03-13
> **Branch:** `main`
> **Scope:** Exhaustive static analysis of all source files, configurations, and data pipeline logic

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Module-by-Module Deep Dive](#3-module-by-module-deep-dive)
4. [Data Flow Analysis](#4-data-flow-analysis)
5. [Dependency Inventory](#5-dependency-inventory)
6. [Critical Bugs & Issues](#6-critical-bugs--issues)
7. [Code Quality Assessment](#7-code-quality-assessment)
8. [Strengths & Best Practices](#8-strengths--best-practices)
9. [Risk Assessment](#9-risk-assessment)
10. [Prioritized Improvement Roadmap](#10-prioritized-improvement-roadmap)
11. [Appendix: File Inventory](#11-appendix-file-inventory)

---

## 1. Executive Summary

This project is a **student-grade PPO-based automated cryptocurrency trading system** targeting Binance Futures (BTCUSDT, 15-minute timeframe). The system implements a complete ML pipeline: data ingestion → feature engineering (with a custom SPA boundary algorithm) → Gymnasium RL environment → PPO training via Stable-Baselines3 → evaluation and backtest comparison. A Genetic Algorithm optimizer is included for hyperparameter search of the SPA algorithm.

Despite being framed as a student scaffold, the codebase exhibits significant engineering depth in its RL environment design and modular pipeline structure. However, **several critical correctness bugs** undermine the reliability of evaluation results, and multiple modules remain in stub or incomplete state.

**Key findings:**
- **2 P0 (critical) bugs** that silently invalidate OOS evaluation metrics
- **2 P1 (high-priority) logic defects** affecting training fidelity
- **3 empty/stub modules** that prevent a complete end-to-end run
- A well-designed RL env with sophisticated execution smoothing
- A custom SPA boundary algorithm that is cleanly modularized
- Duplicate helper functions spread across multiple files instead of a shared `utils/` module

---

## 2. System Architecture Overview

### 2.1 High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                     AUTOMATED TRADING PIPELINE                      │
│                                                                     │
│  [Binance Futures API]                                              │
│        │                                                            │
│        ▼                                                            │
│  download_klines.py          ──→  data/raw/*.parquet                │
│        │                                                            │
│        ▼                                                            │
│  make_features_spa.py        ──→  data/features/*_spa.parquet       │
│  (runs spa_core.py)                      + *_meta.json             │
│        │                                                            │
│        ├──────────────────────────────────────────────────┐        │
│        ▼                                                  ▼        │
│  train_ppo_spa.py            ──→  data/models/*.zip       │        │
│  (uses CryptoTradingEnv)          ppo_logs_spa/           │        │
│  (uses EvalCallback)              data/models/_eval_spa/  │        │
│        │                                                  │        │
│        ▼                                                  │        │
│  eval_ppo_spa.py             ──→  data/eval/*.csv         │        │
│                                                           │        │
│  baseline_backtest_H1.py ◄────────────────────────────────┘        │
│  (vectorbt MA crossover)                                           │
│                                                                     │
│  optimize_spa_ga.py          ──→  data/params/best_spa_ga.json     │
│  (Genetic Algorithm)                                                │
│                                                                     │
│  dashboard/app.py            ──→  Streamlit UI (stub)              │
│  paper_trade/                ──→  (empty)                          │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Dual Feature Paths

The project implements **two distinct feature pipelines**:

| Pipeline | Script | Features Produced | Used By |
|---|---|---|---|
| **Standard** | `make_features.py` | `ret_1`, `rolling_std_20`, RSI, MACD, BB, ATR (full mode) | `train_ppo.py`, `baseline_backtest_*` |
| **SPA** | `make_features_spa.py` | `log_ret_1`, `rolling_std_20`, `spa_sig_num` | `train_ppo_spa.py`, `eval_ppo_spa.py` |

### 2.3 Dual RL Environment Variants

| Environment | File | Action Space | Reward | Status |
|---|---|---|---|---|
| `CryptoTradingEnv` | `crypto_env.py` | Discrete(3) or Box | Step return (bps-scaled) + optional shaping | **Primary / In use** |
| `AdvancedCryptoTradingEnv` | `AdvancedEnv.py` | Continuous Box(2) | Rolling Sharpe + step return | **Unused / Experimental** |
| `SimpleEnv` | `rl_env_simple.py` | — | — | **Unused** |

---

## 3. Module-by-Module Deep Dive

### 3.1 `src/data_ingest/download_klines.py`

**Purpose:** Downloads OHLCV kline data from Binance Futures REST API, resamples to the desired timeframe, and saves as Parquet.

**Key Design Decisions:**
- Uses an **anonymous Binance client** (`Client()`) — no API key required for public market data
- Iterates **day-by-day** with 0.5s sleep to respect rate limits
- Supports configurable interval mapping (`1m` → `1min`, `15m` → `15min`, etc.)
- Deduplicates by `open_time` and sorts index after downloading

**Parameters (CLI):**

| Arg | Default | Description |
|---|---|---|
| `--pair` | `BTCUSDT` | Trading pair |
| `--start` | required | Start date `YYYY-MM-DD` |
| `--end` | required | End date `YYYY-MM-DD` |
| `--interval` | `15m` | Binance kline interval |
| `--output` | `data/raw/btc_15m.parquet` | Output Parquet path |

**Known Issue (P1):** After OHLCV resample, `open`, `high`, and `low` are forward-filled (`ffill()`) even for zero-volume periods. This pads flat bars into empty sessions, potentially **leaking look-ahead information** for any sparse interval. Only `volume` is correctly zeroed.

---

### 3.2 `src/features/make_features.py`

**Purpose:** Computes technical indicators from raw OHLCV data into a feature-engineered Parquet file, plus a `_meta.json` sidecar for pipeline contract enforcement.

**Features Produced:**

| Feature | Mode | Formula |
|---|---|---|
| `ret_1` | slim + full | `close.pct_change()` |
| `log_ret_1` | slim + full | `log(close / close.shift(1))` |
| `momentum_5/10` | full | `close / close.shift(5/10) - 1` |
| `rolling_std_20` | slim + full | `std(log_ret_1, 20)` |
| `rolling_std_50` | full | `std(log_ret_1, 50)` |
| `ATR_14` | full | True Range (14 period EMA) |
| `BB_width` | full | (Upper − Lower) / Middle |
| `BB_position` | full | (close − Lower) / (Upper − Lower) |
| `RSI_14` | full | 14-period Wilder RSI |
| `MACD`, `MACD_hist` | full | EMA(12,26,9) |
| `close_z_60` | full | Z-score over 60-bar window |

**Known Issue (P1):** `freq_hint` is hardcoded to `"1H"` regardless of actual data timeframe (line 152). Since `eval_ppo_spa.py` uses `periods_per_year=70000` (≈15-minute bars), this metadata mismatch will silently produce wrong annualized Sharpe/return figures if a different frequency is used.

**Meta contract (`_meta.json` schema):**
```json
{
  "features": ["list", "of", "column_names"],
  "window_size": 64,
  "freq_hint": "1H",
  "created_from": "data/raw/btc_1h.parquet",
  "rows": 12345
}
```

---

### 3.3 `src/features/make_features_spa.py`

**Purpose:** Builds RL-ready features with SPA signals embedded as a numeric encoding.

**Output features:** `log_ret_1`, `rolling_std_20`, `spa_sig_num` (int8: −1=short, 0=none, 1=long)

**SPA Parameters (CLI):**

| Param | Default | Description |
|---|---|---|
| `--spa_d` | 89 | Market depth window for swing stats |
| `--spa_alpha` | 3.0 | Inner boundary sensitivity |
| `--spa_gamma` | 1.0 | Outer boundary ATR extension multiplier |
| `--spa_m_ma` | 5 | Confirmation MA window |
| `--spa_source` | `close` | Price source (`close`, `hl2`, `hlc3`) |

The SPA meta extends the standard meta with `spa_params` dict — this is a good defensive design for reproducibility.

---

### 3.4 `src/spa/spa_core.py`

**Purpose:** Encapsulates the full SPA (Support-Pressure-Area) boundary algorithm in a pure, stateless pipeline.

**Algorithm Flow:**
```
Input OHLC DataFrame
      │
      ▼
1. Swing Stats:  mu = mean(high-low, d),  sigma = std(high-low, d)
      │
      ▼
2. Boundaries:
   h_inner = rolling_high(d) − α·mu
   H_outer = max(rolling_high(d), close) + (mu + γ·sigma)
   l_inner = rolling_low(d) + α·mu
   L_outer = min(rolling_low(d), close) − (mu + γ·sigma)
      │
      ▼
3. Raw Signals (bar-close logic):
   upper_reject: prev_close > h AND close < h  → short
   upper_break:  prev_close ≤ H AND close > H  → long
   lower_reject: prev_close < l AND close > l  → long
   lower_break:  prev_close ≥ L AND close < L  → short
      │
      ▼
4. MA Confirmation:  long requires src > MA(m),  short requires src < MA(m)
      │
      ▼
5. Dynamic TP/SL:
   delta = mean(high-low, 200)
   LONG:  SL = close − δ,  TP = close + 2δ  (1:2 R/R)
   SHORT: SL = close + δ,  TP = close − 2δ
      │
      ▼
Output: {bands, signals_raw, signals, orders}
```

**Assessment:** This is the most cleanly implemented module in the codebase. It is:
- Fully stateless (no side effects)
- Parameterized via `SPAParams` dataclass
- Returns a well-typed dict with full signal decomposition
- Validated with `min_d` guard

---

### 3.5 `src/rl_env/crypto_env.py` — `CryptoTradingEnv`

**Purpose:** Main Gymnasium environment used in all SPA training/evaluation runs.

**Observation Space:** `(window_size, n_features + 3)` — rolling market feature window concatenated with account state block (broadcast).

**Account State Vector:** `[signed_pos_frac, unrealized_pct, free_margin_pct]`

**Action Modes:**
- `discrete`: `{0=short, 1=flat, 2=long}` → maps to target notional fraction `{-0.30, 0.0, +0.30}`
- `continuous`: Box(1) clamped to `±position_limit`

**Execution Smoothing (3-layer gate):**

| Gate | Condition | Effect |
|---|---|---|
| **Cooldown** | `steps_since_trade < cooldown_steps` | Blocks all new trades |
| **Min Hold** | `steps_since_trade < min_hold_steps` | Blocks position flip |
| **Deadband** | `|Δnotional| / equity < deadband_frac` | Ignores tiny adjustments |

**Reward Function:**
```
base_reward = reward_scale × (equity_after / equity_before − 1)

Optional shaping (all default to 0.0):
  + flat_penalty   if position == 0
  + turnover_reward if |turnover| > threshold
  − inactivity_penalty if steps_since_trade > inactivity_steps
```

**Normalization:** Z-score normalization is computed once at `__init__` using full-episode stats (not rolling) — adequate for RL but note this computes on *all* data in the df passed, not just train set.

**Known Issue (noted):** `_fit_norm()` is called in `__init__` and uses the entire df. In training context this is fine (df is `train_df`), but it precomputes normalization statistics using future data if the caller passes the full dataset.

---

### 3.6 `src/rl_env/AdvancedEnv.py` — `AdvancedCryptoTradingEnv`

**Purpose:** An experimental environment with richer features (including volume and rolling Sharpe reward), **currently unused** in any training script.

**Notable Differences from `crypto_env.py`:**
- Continuous 2D action space: `[direction, size_fraction]`
- Rolling Sharpe as reward component
- Dynamic slippage model: `base_slippage + |order_size| × volatility × 0.1`
- No execution smoothing gates (deadband / cooldown)
- Simpler position tracking (no VWAP average entry logic)

**Assessment:** Hardcodes feature list `['ret_1', 'MA7', 'MA30', 'rolling_std_20', 'close_z', 'volume']` which won't match any parquet produced by the current feature scripts (e.g., no `MA7` or `MA30` in any `make_features*.py`). This env would crash on first use without new feature data.

---

### 3.7 `src/train/train_ppo_spa.py`

**Purpose:** Main PPO training script. Loads SPA features, splits train/eval, builds `CryptoTradingEnv`, wraps in `DummyVecEnv + VecMonitor`, trains with `EvalCallback`, saves best and final model.

**PPO Hyperparameters:**

| Param | Value | Rationale |
|---|---|---|
| `n_steps` | 1024 | Rollout buffer size |
| `batch_size` | 512 | Minibatch for gradient update |
| `learning_rate` | 2e-4 | Conservative for stability |
| `gamma` | 0.99 | Long-horizon discounting |
| `gae_lambda` | 0.95 | Standard GAE |
| `ent_coef` | 0.005 | Encourages exploration |
| `clip_range` | 0.20 | Standard PPO clipping |

**Train/Eval Split:** `train_split=0.8` (default, from CLI) with automatic boundary guard to ensure eval set has `≥ window_size + 1` rows.

**Known Issue (P1 — Bug):** `common_env_kwargs` dict is constructed on lines 181–195 but is **never used**. Both `build_env()` calls are written out manually. The manual call for `train_env` (lines 197–216) is **missing `inactivity_steps`**, causing the env to fall back to its `__init__` default of `256` instead of the CLI-provided value of `128`. The eval env correctly zeroes all shaping parameters.

---

### 3.8 `src/eval/eval_ppo_spa.py`

**Purpose:** Loads a saved PPO model, runs one full episode on the OOS eval set, computes comprehensive metrics, and exports equity curve + trade log as CSV.

**Metrics Computed:**
- Total return, annualized return, annualized volatility
- Sharpe Ratio (using `periods_per_year=70000`)
- Maximum Drawdown (peak-to-trough equity)
- N Trades, Win Rate, Avg Win, Avg Loss, Avg R/R
- Avg trade duration (steps → hours)

**Trade State Machine:** Tracks open/close/flip transitions from `position_frac` sign changes — a reasonable proxy, though sign flips within the same step are treated as simultaneous close+open.

**Known Issue (P0 — Critical):** `train_split` defaults to `0.7` in this script, but `train_ppo_spa.py` defaults to `0.8`. If these differ, the evaluator accesses the last 30% of data while the trainer trained on the first 80%, meaning **17.5% of the OOS eval set overlaps with training data**, rendering OOS results invalid.

**Known Issue (P0 — Critical):** `build_env()` in this file has non-zero shaping parameters:
```python
flat_penalty_bps=0.01,
inactivity_penalty_bps=0.01,
turnover_reward_coeff=0.02,
```
Although small, these introduce reward shaping during evaluation, subtly distorting the equity curve and invalidating clean OOS performance measurement.

---

### 3.9 `src/backtest/baseline_backtest_H1.py`

**Purpose:** Implements a vectorized MA crossover strategy using `vectorbt` as a benchmark for comparison with the PPO agent.

**Strategy Logic:**
- Grid search over fast MA and slow MA windows
- 3-bar confirmation: `prev_below & curr_below & next_above` for long entries
- Optional volume filter (activated if `volume` column present)
- Realistic transaction costs: 0.06% fees + 0.03% slippage

**Data Quality Checks (`check_data_quality`):**
- Duplicate timestamp detection and removal
- Time gap detection (>2× expected frequency)
- Invalid price detection (zero / negative)
- Extreme outlier detection (>100× or <1/100× median)
- OHLC logical consistency (`high ≥ low`, `high ≥ close`, etc.)

**Output:** Best strategy stats vs. Buy & Hold, Top 10 by Sharpe Ratio.

**Known Issue (P1):** `ensure_datetime_index` is a copy-paste duplicate of the same function in `make_features.py`. Should be extracted to `src/utils/`.

---

### 3.10 `src/optimize/optimize_spa_ga.py`

**Purpose:** Genetic Algorithm (GA) to find optimal SPA parameters by maximizing final portfolio capital on historical data.

**Search Space:**

| Gene | Options |
|---|---|
| `n` (depth window) | [13, 21, 34, 55, 89, 144, 233] |
| `alpha` (fast MA) | [3, 5, 8, 13] |
| `d` (slow MA) | [5, 8, 13, 21, 34] |
| `beta` (volatility multiplier) | [0.38, 0.50, 0.61, 1.0, 1.44] |
| `src` (price source) | ['close', 'hl2', 'hlc3'] |

**GA Parameters (defaults):** Population=80, Generations=30, Mutation Rate=0.2, Tournament k=5

**Fitness Function:** Simplified bar-by-bar simulation of long/short entries on computed band breaks. Uses `open_price` vs `close_price` crossing the band threshold within the same bar — this is an **intra-bar lookahead** heuristic (not a true open-of-next-bar fill). The fitness metric is raw final capital (not Sharpe), which doesn't penalize drawdown.

**Parallelism:** Uses `multiprocessing.Pool` with `cpu_count() − 1` workers. Memoization via dict prevents redundant evaluations. The `memo` dict is passed as a kwarg but spawned in separate processes — **memo sharing across workers does NOT work** as intended because each process has its own memory space. This doesn't break correctness but wastes CPU cycles re-evaluating identical individuals.

**Output:** `data/params/best_spa_ga.json` with winning `{n, alpha, d, beta, src, fitness_final_capital}`.

---

### 3.11 `src/optimize/` — Vestigial Scripts

The `src/optimize/` directory contains **3 large vestigial scripts** from earlier development iterations:

| File | Size | Status |
|---|---|---|
| `Feature+Genetic+GPU_Version_2.py` | 21.9 KB | Iterative predecessor — **superseded** |
| `Fix_Feature.py` | 19.9 KB | Intermediate iteration — **superseded** |
| `Train_Test_Split_Feature.py` | 18.8 KB | Intermediate iteration — **superseded** |
| `optimize_spa_ga.py` | 8.0 KB | **Canonical current version** |

These should be removed or archived to prevent confusion.

---

### 3.12 `src/dashboard/app.py`

**Purpose:** Streamlit dashboard for data visualization.

**Current State:** Minimal but functional stub (43 lines). Loads a Parquet from `--features` path and renders:
- Title and success/error message
- Close price line chart
- Tail(100) data table
- Feature selectbox with line chart

**Not yet implemented:**
- Equity curve visualization
- Live trade tracking
- SPA boundary plots
- Model performance comparison
- Paper trading integration

---

### 3.13 `src/paper_trade/` and `src/utils/`

Both directories contain **only `__init__.py`** and are completely empty. Paper trading against the Binance Testnet is mentioned in the README but not implemented.

---

## 4. Data Flow Analysis

### 4.1 End-to-End Data Flow

```
STEP 1: DATA INGESTION
  CLI: python src/data_ingest/download_klines.py --start 2024-01-01 --end 2024-06-01
  Input:  Binance Futures REST API (unauthenticated)
  Output: data/raw/btc_15m.parquet
          Columns: open, high, low, close, volume (DatetimeIndex UTC)

STEP 2: FEATURE ENGINEERING
  CLI: python src/features/make_features_spa.py --input data/raw/btc_15m.parquet
  Input:  data/raw/btc_15m.parquet
  Process:
    ├── log_ret_1 = log(close / close.shift(1))
    ├── rolling_std_20 = std(log_ret_1, 20)
    └── SPA Pipeline:
        ├── swing stats (mu, sigma)
        ├── boundaries (h_inner, H_outer, l_inner, L_outer)
        ├── raw signals → MA confirmation
        └── spa_sig_num = encode(confirmed_signal)  # {-1, 0, 1}
  Output: data/features/btc_15m_spa.parquet  [close, log_ret_1, rolling_std_20, spa_sig_num]
          data/features/btc_15m_spa_meta.json  [features, window_size, spa_params]

STEP 3: TRAINING
  CLI: python src/train/train_ppo_spa.py --features data/features/btc_15m_spa.parquet --timesteps 300000
  Input:  features parquet + meta json
  Process:
    ├── Load meta → features list, window_size
    ├── Split: train_df (80%) / eval_df (20%)
    ├── CryptoTradingEnv(train_df) → DummyVecEnv → VecMonitor
    ├── CryptoTradingEnv(eval_df) → DummyVecEnv → VecMonitor (shaping=OFF)
    ├── PPO.learn(timesteps) with EvalCallback
    └── Save best model via EvalCallback + final model
  Output: data/models/ppo_spa_btc_15m.zip
          data/models/_eval_spa/best_model.zip
          ppo_logs_spa/PPO_N/  (TensorBoard logs)

STEP 4: EVALUATION
  CLI: python src/eval/eval_ppo_spa.py --model data/models/_eval_spa/best_model.zip
  Input:  best_model.zip + features parquet (loads meta)
  Process:
    ├── Slice eval_df using train_split (⚠ must match trainer)
    ├── Run one deterministic episode
    ├── Track equity, actions, position_frac
    ├── State machine → trade list
    └── Compute: total_return, Sharpe, MaxDD, n_trades, winrate, avg_RR, avg_duration
  Output: data/eval/ppo_spa_btc_15m_eval_best.csv       (equity / action curve)
          data/eval/ppo_spa_btc_15m_eval_best_trades.csv (trade log)

OPTIONAL STEP: GA OPTIMIZATION
  CLI: python src/optimize/optimize_spa_ga.py
  Input:  data/raw/btc_1h.parquet (hardcoded)
  Output: data/params/best_spa_ga.json

OPTIONAL STEP: BASELINE BACKTEST
  CLI: python src/backtest/baseline_backtest_H1.py --data data/raw/btc_1h.parquet
  Output: Console report (no file output)
```

### 4.2 Meta Contract Flow

```
make_features_spa.py
    └── writes *_meta.json
            ├── features = ["log_ret_1", "rolling_std_20", "spa_sig_num"]
            ├── window_size = 64
            └── spa_params = {d, alpha, gamma, m_ma, source}

train_ppo_spa.py
    └── reads *_meta.json → (features, window_size)
        → builds CryptoTradingEnv observation_space = (window_size, n_features+3)
        → PPO policy input shape locked at init time

eval_ppo_spa.py
    └── reads *_meta.json → (features, window_size)
        → must match trained model's policy input shape exactly (enforced at load time)
```

---

## 5. Dependency Inventory

### 5.1 Runtime Dependencies (`requirements.txt`)

| Package | Version | Role |
|---|---|---|
| `pandas` | 2.2.2 | Data manipulation, time series |
| `numpy` | **1.23.5** | Numerical computing ⚠ old |
| `pyarrow` | 16.1.0 | Parquet I/O backend |
| `stable-baselines3` | 2.3.2 | PPO implementation |
| `gymnasium` | 0.29.1 | RL environment interface |
| `python-binance` | 1.0.19 | Binance API client |
| `vectorbt` | 0.26.2 | Vectorized backtesting |
| `plotly` | 5.15.0 | Interactive charting |
| `streamlit` | 1.35.0 | Dashboard framework |
| `tqdm` | 4.66.4 | Progress bars |
| `tensorboard` | 2.16.2 | Training metrics visualization |
| `torch` | — | Not pinned (comment-only) ⚠ |

**Note:** `torch` / `pytorch` is commented out with a note that it must be installed separately due to CUDA version complexity. This is reasonable but should be documented more formally.

### 5.2 Implicit Dependencies

- **Python:** 3.10+ (uses `list[str]` type hint syntax in function signatures)
- **CUDA toolkit:** Optional, for GPU training
- **Internet access:** Required for `download_klines.py` (Binance API)

### 5.3 Dependency Risks

| Risk | Severity | Details |
|---|---|---|
| `numpy==1.23.5` is old | Medium | SB3 2.3.2 needs ≥1.20; modern usage works best with 1.26+. Some NumPy 2.x API changes may cause compat issues with other deps. |
| `vectorbt==0.26.2` + pandas 2.x | Medium | vectorbt has known compatibility issues with pandas 2.x DatetimeIndex behaviors |
| No `torch` pin | High | Different CUDA/torch combinations produce different numerical results, breaking reproducibility |
| `python-binance` 1.0.19 | Low | Library is maintained; testnet endpoints use different URLs which the code does not configure |

---

## 6. Critical Bugs & Issues

### 🔴 P0 — Critical (Breaks Evaluation Correctness)

#### Bug 1: `train_split` Default Mismatch

**File:** `src/eval/eval_ppo_spa.py`, line 238 and 249
```python
# TRAINER (train_ppo_spa.py, line 131)
train_split: float = 0.8   # 80% train, 20% eval

# EVALUATOR (eval_ppo_spa.py, line 238)
train_split: float = 0.7   # ← WRONG: 70% train, 30% eval
```
**Impact:** With 70/30 split in eval vs 80/20 in training, 10% of the data used as "OOS eval" in the evaluator was actually part of the training set. All reported OOS metrics (Sharpe, return, winrate) are **overstated and invalid**.

**Fix:**
```python
# Option A: Match defaults
train_split: float = 0.8

# Option B (better): Store in _meta.json during training and read during eval
meta["train_split"] = train_split  # in train_ppo_spa.py
# then in eval_ppo_spa.py:
train_split = meta.get("train_split", 0.8)
```

---

#### Bug 2: Reward Shaping Active in Eval Environment

**File:** `src/eval/eval_ppo_spa.py`, lines 51–54
```python
def build_env(df, features, window_size):
    return CryptoTradingEnv(
        ...
        flat_penalty_bps=0.01,          # ← should be 0.0
        inactivity_penalty_bps=0.01,    # ← should be 0.0
        turnover_reward_coeff=0.02,     # ← should be 0.0
    )
```
**Impact:** The eval equity curve is subtly distorted by shaping rewards. The equity is not a pure PnL curve — it includes phantom rewards/penalties that inflate or deflate reported metrics.

**Fix:** Set all shaping coefficients to `0.0` in the eval `build_env()`.

---

### 🟡 P1 — High Priority (Training Fidelity Defect)

#### Bug 3: `inactivity_steps` Not Passed to Train Env

**File:** `src/train/train_ppo_spa.py`, lines 181–216
```python
common_env_kwargs = dict(
    ...
    inactivity_steps=inactivity_steps,  # built but NEVER USED
)

train_env = build_env(
    train_df, features, window_size,
    flat_penalty_bps=flat_penalty_bps,
    # inactivity_steps=inactivity_steps  ← MISSING HERE
    ...
)
```
**Impact:** The train environment always uses `inactivity_steps=256` (env default) regardless of CLI value `--inactivity_steps=128`. The `common_env_kwargs` dict is built but never passed.

**Fix:**
```python
train_env = build_env(train_df, features, window_size,
    **common_env_kwargs,
    min_hold_steps=min_hold_steps,
    cooldown_steps=cooldown_steps,
)
```

---

#### Bug 4: `ffill()` on OHLC After Resample

**File:** `src/data_ingest/download_klines.py`, lines 87–91
```python
out = pd.DataFrame({
    'open':  o.ffill(),  # ← forward-fills into zero-volume periods
    'high':  h.ffill(),  # ← same issue
    'low':   l.ffill(),  # ← same issue
    'close': c.ffill(),  # ← close ffill is acceptable
    'volume': v.fillna(0)  # ← correct
})
```
**Impact:** For sparse data (e.g., long holiday weekends, exchange maintenance), this creates phantom candles with real-looking OHLC values but zero volume, which can introduce artifacts in rolling indicators (RSI, ATR, BB).

**Fix:** Use `dropna()` on the resampled OHLC (drop empty sessions) or explicitly flag zero-volume bars for downstream filtering.

---

### 🟢 P2 / P3 — Medium / Low Priority

| ID | Location | Issue | Priority |
|---|---|---|---|
| P2-1 | `src/utils/` | **Empty** — `ensure_datetime_index` is copy-pasted in `make_features.py` and `baseline_backtest_H1.py` | P2 |
| P2-2 | `src/paper_trade/` | **Empty** — paper trading mentioned in README, not implemented | P2 |
| P2-3 | `src/dashboard/app.py` | Minimal stub — no equity curves, no SPA overlays, no trade history | P2 |
| P2-4 | `eval_ppo_spa.py` | `freq_hint` not stored/read from meta → hardcoded `periods_per_year=70000` | P2 |
| P2-5 | `optimize_spa_ga.py` | `memo` dict not shared between processes (multiprocessing semantics) → wasted evaluations | P2 |
| P3-1 | `scaffold.py` | References `src/feature_engineering/` — actual module is `src/features/` | P3 |
| P3-2 | `src/optimize/` | 3 vestigial scripts totaling ~60 KB should be archived/removed | P3 |
| P3-3 | `check_ga_readiness.py` | Dead equity loop (lines 46–50) never used — should be removed | P3 |
| P3-4 | `requirements.txt` | `numpy==1.23.5` should be bumped to `>=1.26.0` | P3 |
| P3-5 | `AdvancedEnv.py` | Hardcoded feature list `['ret_1', 'MA7', 'MA30', ...]` incompatible with any current feature script | P3 |

---

## 7. Code Quality Assessment

### 7.1 Metrics Summary

| Dimension | Score | Notes |
|---|---|---|
| **Modularity** | 7/10 | Good separation of concerns; SPA, env, train are distinct. Shared helpers not extracted. |
| **Readability** | 8/10 | Well-commented (bilingual Thai/English). Descriptive variable names. |
| **Type Safety** | 6/10 | Good use of type hints and dataclasses in `spa_core.py`. Other files use informal typing. |
| **Error Handling** | 6/10 | Good input validation in env `__init__`. Missing in GA memo sharing. Try/except overused in backtest. |
| **Reproducibility** | 8/10 | Excellent seed management (random, numpy, torch, cuDNN). Meta contract enforced. |
| **Testability** | 3/10 | No test files anywhere. No CI/CD configuration. Hard to unit-test without data files. |
| **Documentation** | 7/10 | README covers pipeline steps. Docstrings present but inconsistent. No inline API docs. |

### 7.2 Duplicate Code Locations

```
_ensure_datetime_index() / ensure_datetime_index() / ensure_dtindex()
  ├── src/features/make_features.py       (line 10)
  ├── src/features/make_features_spa.py   (line 17)
  ├── src/backtest/baseline_backtest_H1.py (line 10)
  └── src/optimize/optimize_spa_ga.py     (line 22)

→ Should be one canonical function in src/utils/__init__.py
```

---

## 8. Strengths & Best Practices

### 8.1 Architecture Excellence

1. **Meta contract (`_meta.json`):** Tying `features` and `window_size` between feature engineering, training, and evaluation prevents silent shape mismatches — a production-grade defensive pattern.

2. **Evaluation env shaping separation:** The trainer correctly enables reward shaping to guide exploration, while the evaluator (intends to) disable it for clean PnL measurement. The approach is sound; only the implementation has bugs.

3. **SPA modularization:** `spa_core.py` is a textbook example of clean separation — pure functions, dataclass params, no global state, stateless pipeline.

4. **Execution smoothing gates:** The 3-layer gate (cooldown + min_hold + deadband) in `CryptoTradingEnv` realistically models trading friction that most student RL environments ignore. This prevents unrealistic high-frequency trading artifacts.

5. **Seed determinism:** Setting seeds across `random`, `numpy`, `torch`, and `cuDNN` ensures complete training reproducibility — a critical property for research.

### 8.2 Data Engineering

6. **Day-by-day download loop:** Respects API rate limits while downloading large date ranges. The 0.5s sleep is conservative but correct.

7. **Data quality gating in backtest:** `check_data_quality()` performs 5 distinct checks (duplicates, gaps, invalid prices, outliers, OHLC consistency) — more rigorous than most academic backtests.

8. **Dual action mode support:** The env gracefully handles both discrete and continuous action spaces via a single `action_mode` flag, enabling easy experimentation.

---

## 9. Risk Assessment

### 9.1 Financial / Operational Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| OOS metrics invalid due to data leakage (Bug 1) | **Certain** (bug confirmed) | **High** | Fix `train_split` mismatch immediately |
| Overfitting to training set | High | High | Stricter walk-forward validation needed |
| No live trading safeguards | N/A (paper only) | Med | `paper_trade/` module implementation needed before any live deployment |
| Binance API key exposure | Low | High | Current code uses no API keys (public endpoints only) — safe for now |

### 9.2 Technical Risks

| Risk | Likelihood | Impact |
|---|---|---|
| `vectorbt` + pandas 2.x breaking changes | Medium | Would break baseline backtest |
| PyTorch CUDA version conflict on new machines | High | Training would silently fall back to CPU (slow) or crash |
| `multiprocessing.Pool` + memo not shared (Bug in optimize_spa_ga.py) | Certain | ~2–3× slower GA optimization than intended |
| Large optimize/ scripts misleading future contributors | Medium | Confusion, duplicate work |

---

## 10. Prioritized Improvement Roadmap

### Phase 1 — Immediate Fixes (1–2 days)

| Priority | Change | Files Affected |
|---|---|---|
| **P0** | Match `train_split` default to `0.8` in evaluator OR store/read from meta | `eval_ppo_spa.py`, `train_ppo_spa.py`, `make_features_spa.py` |
| **P0** | Zero all shaping params in eval `build_env()` | `eval_ppo_spa.py` |
| **P1** | Pass `inactivity_steps` to `train_env` via `common_env_kwargs` | `train_ppo_spa.py` |
| **P1** | Store `train_split` + `freq_hint` in `_meta.json`; read in eval | `make_features_spa.py`, `eval_ppo_spa.py` |

### Phase 2 — Code Quality (3–5 days)

| Priority | Change | Files Affected |
|---|---|---|
| **P2** | Extract `ensure_datetime_index` to `src/utils/__init__.py` | All 4 files with duplicates |
| **P2** | Fix `ffill()` on OHLC resample — drop or flag empty bars | `download_klines.py` |
| **P2** | Fix GA memoization — use `multiprocessing.Manager().dict()` for shared state | `optimize_spa_ga.py` |
| **P3** | Archive/delete 3 vestigial optimize scripts | `src/optimize/` |
| **P3** | Fix scaffold path `feature_engineering` → `features` | `scaffold.py` |

### Phase 3 — Feature Completion (1–2 weeks)

| Priority | Feature | Notes |
|---|---|---|
| **P2** | Implement `paper_trade/` module | Minimal live loop against Binance Testnet with position tracking |
| **P2** | Build Streamlit dashboard | Equity curves, SPA overlay, trade log, model comparison panel |
| **P2** | Integrate `AdvancedEnv.py` or deprecate | Update feature list or remove |
| **P3** | Add unit tests (`tests/`) | At minimum: env reset/step, SPA signal generation, meta loading |
| **P3** | Bump `numpy` to `>=1.26.0` | Aligns with SB3 2.3.2 best practices |
| **P3** | Pin `torch` in requirements (with GPU/CPU variants documented) | Critical for reproducibility |

---

## 11. Appendix: File Inventory

### Source Files

| File | Size | Status | Role |
|---|---|---|---|
| `src/data_ingest/download_klines.py` | 4.7 KB | ✅ Complete | Binance data downloader |
| `src/features/make_features.py` | 7.6 KB | ✅ Complete | Standard feature engineering |
| `src/features/make_features_simple.py` | 2.4 KB | ⚠️ Superseded | Simple feature stub |
| `src/features/make_features_spa.py` | 5.5 KB | ✅ Complete | SPA feature engineering |
| `src/spa/spa_core.py` | 6.6 KB | ✅ Complete | SPA algorithm (best module) |
| `src/rl_env/crypto_env.py` | 12.0 KB | ✅ Complete (bugs) | Primary RL environment |
| `src/rl_env/AdvancedEnv.py` | 7.7 KB | ⚠️ Unused | Experimental env |
| `src/rl_env/rl_env_simple.py` | 6.0 KB | ⚠️ Unused | Simple env |
| `src/train/train_ppo_spa.py` | 12.7 KB | ✅ Complete (bugs) | PPO SPA trainer |
| `src/train/train_ppo.py` | 4.0 KB | ⚠️ Superseded | Basic PPO trainer |
| `src/train/train_ppo_simple.py` | 2.8 KB | ⚠️ Superseded | Simple trainer |
| `src/eval/eval_ppo_spa.py` | 9.9 KB | ✅ Complete (bugs) | Model evaluator |
| `src/backtest/baseline_backtest_H1.py` | 14.7 KB | ✅ Complete | MA crossover baseline |
| `src/backtest/baseline_backtest_M1.py` | 1.8 KB | ⚠️ Minimal | 1-min baseline stub |
| `src/backtest/backtest_ppo.py` | 3.4 KB | ⚠️ Superseded | PPO backtest utility |
| `src/optimize/optimize_spa_ga.py` | 8.0 KB | ✅ Complete | GA optimizer |
| `src/optimize/Feature+Genetic+GPU_Version_2.py` | 21.9 KB | 🗑️ Vestigial | Archive/delete |
| `src/optimize/Fix_Feature.py` | 19.9 KB | 🗑️ Vestigial | Archive/delete |
| `src/optimize/Train_Test_Split_Feature.py` | 18.8 KB | 🗑️ Vestigial | Archive/delete |
| `src/dashboard/app.py` | 1.4 KB | ⚠️ Stub | Minimal Streamlit dashboard |
| `src/paper_trade/` | — | ❌ Empty | Not implemented |
| `src/utils/` | — | ❌ Empty | Shared helpers not extracted |
| `scaffold.py` | 5.4 KB | ⚠️ Stale | Path mismatch (`feature_engineering`) |
| `check_ga_readiness.py` | 3.4 KB | ⚠️ Debug script | Dead code (equity loop) |
| `check_gpu.py` | 0.4 KB | ✅ Utility | GPU availability check |
| `make_meta.py` | 0.9 KB | ✅ Utility | Meta JSON generator |
| `make_meta_from_parquet.py` | 0.1 KB | ✅ Utility | Meta from parquet |

### Data Directories

| Directory | Contents | Notes |
|---|---|---|
| `data/raw/` | `.parquet` files of raw OHLCV | Downloaded by `download_klines.py` |
| `data/features/` | Feature parquets + `_meta.json` | Generated by `make_features_*.py` |
| `data/models/` | `.zip` PPO model checkpoints | Saved by `train_ppo_spa.py` |
| `data/eval/` | `*_eval.csv`, `*_trades.csv` | Generated by `eval_ppo_spa.py` |
| `ppo_logs_spa/` | TensorBoard logs (PPO_1..PPO_8) | 8 training runs recorded |

---

*Report generated: 2026-03-13 | Codebase: `d:\Workshop\TradeProject\automated_trading_using_ai` | Branch: `main`*
