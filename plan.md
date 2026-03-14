# Code Update Plan: `automated_trading_using_ai`

> **Based on:** [research.md](./research.md)
> **Planned:** 2026-03-13
> **Status:** 📋 Plan Only — Not Yet Implemented

---

## Overview

This plan translates every finding from `research.md` into concrete, ordered code changes. Updates are grouped into **3 phases** by impact and dependency order. Phase 1 must be completed before re-running any training/evaluation, as the bugs it fixes directly corrupt all recorded metrics.

---

## Phase 1 — Critical Bug Fixes
> **Timeline:** ~1–2 days | **Do this before any new training runs**

These fixes correct silent correctness failures that invalidate all OOS evaluation output.

---

### 1.1 Fix `train_split` Data Leakage
**Priority:** 🔴 P0 — renders all OOS metrics invalid

**Root cause:** `eval_ppo_spa.py` defaults `train_split=0.7`, while `train_ppo_spa.py` defaults `train_split=0.8`. The evaluator slices 30% of data as "OOS", but 10% of that overlaps with the training set.

**Strategy — Option B (preferred): Store in meta, read in eval**

#### [MODIFY] `src/features/make_features_spa.py`
- The `make_features_spa()` function currently writes `meta` without `train_split`
- Add `train_split` as a new parameter (default `0.8`) to `make_features_spa()`
- Write it into `_meta.json` under the key `"train_split"`
- Add `--train_split` CLI argument

**Exact change:**
```python
# In make_features_spa() signature, add:
train_split: float = 0.8

# In meta dict, add:
"train_split": train_split

# In argparse block, add:
ap.add_argument("--train_split", type=float, default=0.8)

# Pass to function:
make_features_spa(..., train_split=args.train_split)
```

#### [MODIFY] `src/train/train_ppo_spa.py`
- After loading meta, also write `train_split` into the meta file
- This ensures the meta created by the trainer reflects the actual split used, regardless of what the feature script wrote

**Exact change:**
```python
# After load_meta_or_infer(), update meta file:
meta_path = Path(str(features_path).replace(".parquet", "_meta.json"))
if meta_path.exists():
    meta_data = json.loads(meta_path.read_text())
    meta_data["train_split"] = train_split
    meta_path.write_text(json.dumps(meta_data, indent=2))
```

#### [MODIFY] `src/eval/eval_ppo_spa.py`
- In `load_meta()`, also read `train_split` from the meta dict and return it
- In `run_eval()` and `main()`, remove the `train_split` parameter; derive it from meta
- Update the `--train_split` CLI arg to serve only as a fallback override

**Exact change in `load_meta()`:**
```python
def load_meta(features_path: Path):
    ...
    train_split = float(meta.get("train_split", 0.8))   # read from meta
    return features, window_size, train_split           # add to return tuple
```

**Exact change in `main()` default:**
```python
# Remove hardcoded default=0.7; derive from meta instead
# CLI --train_split acts as an explicit override only
```

---

### 1.2 Zero All Reward Shaping in Eval Environment
**Priority:** 🔴 P0 — distorts equity curve and all derived metrics

**Root cause:** `eval_ppo_spa.py :: build_env()` passes non-zero shaping coefficients (`flat_penalty_bps=0.01`, `inactivity_penalty_bps=0.01`, `turnover_reward_coeff=0.02`), contradicting the comment `"keep shaping OFF in eval"`.

#### [MODIFY] `src/eval/eval_ppo_spa.py`
**Exact change in `build_env()`:**
```python
def build_env(df, features, window_size):
    return CryptoTradingEnv(
        df=df,
        features=features,
        window_size=window_size,
        initial_balance=10_000.0,
        taker_fee=0.0005,
        position_limit=0.5,
        slippage_bps=0.0,
        reward_scale=100.0,
        normalize=True,
        action_mode="discrete",
        deadband_frac=0.02,
        min_hold_steps=2,
        # --- ALL shaping OFF for clean OOS eval ---
        flat_penalty_bps=0.0,           # was 0.01
        inactivity_steps=256,
        inactivity_penalty_bps=0.0,     # was 0.01
        turnover_reward_coeff=0.0,      # was 0.02
        trade_threshold=0.1,
    )
```

---

### 1.3 Pass `inactivity_steps` to Train Environment
**Priority:** 🟡 P1 — CLI parameter silently ignored; train env uses wrong default

**Root cause:** `train_ppo_spa.py` builds `common_env_kwargs` with `inactivity_steps` but never uses it; the manual `build_env()` call for `train_env` omits `inactivity_steps`, so the env defaults to `256` instead of the CLI-provided `128`.

#### [MODIFY] `src/train/train_ppo_spa.py`
Replace the manual duplicate train/eval `build_env()` calls with a single pattern using `common_env_kwargs`:

```python
# Step 1: common_env_kwargs already has all shared params — use it!
train_env = build_env(
    train_df, features, window_size,
    **common_env_kwargs,            # includes inactivity_steps
    min_hold_steps=min_hold_steps,
    cooldown_steps=cooldown_steps,
)

eval_env = build_env(
    eval_df, features, window_size,
    # override shaping to zero for eval during training
    initial_balance=10_000.0,
    taker_fee=0.0005,
    position_limit=0.30,
    slippage_bps=0.0,
    reward_scale=100.0,
    normalize=True,
    action_mode="discrete",
    flat_penalty_bps=0.0,
    inactivity_penalty_bps=0.0,
    turnover_reward_coeff=0.0,
    deadband_frac=deadband_frac,
    min_hold_steps=min_hold_steps,
    cooldown_steps=cooldown_steps,
)
```

Also **delete `common_env_kwargs`** dict from the function body (it's built but never used, a source of confusion).

---

### 1.4 Store `freq_hint` in Meta and Read in Eval
**Priority:** 🟡 P1 — hardcoded `"1H"` causes wrong annualized Sharpe if data is 15m

**Root cause:** `make_features.py` hardcodes `"freq_hint": "1H"` (line 152) regardless of input data frequency. `eval_ppo_spa.py` uses `periods_per_year=70000` (correct for 15m), but a future user running on hourly data would get wrong metrics silently.

#### [MODIFY] `src/features/make_features_spa.py`
- Infer `freq_hint` from the DataFrame index using `pd.infer_freq()`
- Store the inferred hint in meta
- Map known hints to `periods_per_year` values in the meta

```python
# Infer frequency from index
inferred_freq = pd.infer_freq(out.index) or "unknown"
FREQ_TO_PERIODS = {"1min": 525600, "5min": 105120, "15min": 35040, "1H": 8760, "4H": 2190}
periods_per_year = FREQ_TO_PERIODS.get(inferred_freq, 35040)  # default 15m

meta = {
    ...
    "freq_hint": inferred_freq,
    "periods_per_year": periods_per_year,
    "train_split": train_split,
}
```

#### [MODIFY] `src/eval/eval_ppo_spa.py`
- Read `periods_per_year` from meta in `load_meta()` instead of receiving it as a CLI param
- Keep `--periods_per_year` as an explicit override fallback

```python
# In load_meta():
periods_per_year = int(meta.get("periods_per_year", 35040))
return features, window_size, train_split, periods_per_year
```

#### [MODIFY] `src/features/make_features.py`
- Same `freq_hint` inference: replace line 152's hardcoded `"1H"` with `pd.infer_freq(df.index) or "1H"`

---

## Phase 2 — Code Quality & Technical Debt
> **Timeline:** ~3–5 days

---

### 2.1 Create `src/utils/` Shared Helpers
**Priority:** 🟡 P2 — eliminates 4 copies of the same function

#### [MODIFY] `src/utils/__init__.py`
Create a single canonical `ensure_datetime_index()` function here. All 4 files currently have their own variant:

```python
# src/utils/__init__.py
import pandas as pd

def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Canonical helper: ensure UTC DatetimeIndex, sorted, no duplicates.
    Tries common timestamp column names as fallback.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df[~df.index.duplicated(keep='last')].sort_index()
    for col in ('timestamp', 'time', 'open_time', 'date', 'Date', 'datetime'):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
            df = df.set_index(col)
            break
    else:
        df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
    return df[~df.index.duplicated(keep='last')].sort_index()
```

#### [MODIFY] (4 files — remove local copies, import from utils)

| File | Local function to remove | Replace with |
|---|---|---|
| `src/features/make_features.py` | `_ensure_datetime_index()` (line 10) | `from src.utils import ensure_datetime_index` |
| `src/features/make_features_spa.py` | `ensure_dtindex()` (line 17) | `from src.utils import ensure_datetime_index` |
| `src/backtest/baseline_backtest_H1.py` | `ensure_datetime_index()` (line 10) | `from src.utils import ensure_datetime_index` |
| `src/optimize/optimize_spa_ga.py` | `_ensure_dtindex()` (line 22) | `from src.utils import ensure_datetime_index` |

Each file's sys.path setup already handles project root — the import will work.

---

### 2.2 Fix OHLC `ffill()` After Resample
**Priority:** 🟡 P2 — creates phantom bars from zero-volume periods

#### [MODIFY] `src/data_ingest/download_klines.py`
Replace blind `ffill()` on open/high/low with conditional drop or flag:

```python
# Build resampled OHLCV with NaN where no trades occurred
out = pd.DataFrame({
    'open':   o,              # NaN for empty periods — do NOT ffill
    'high':   h,              # NaN for empty periods
    'low':    l,              # NaN for empty periods
    'close':  c.ffill(),      # close ffill is acceptable (last known price)
    'volume': v.fillna(0.0)   # zero volume is correct
})

# Drop empty bars (no open trade in the period)
out = out.dropna(subset=['open', 'high', 'low'])
print(f"[info] Dropped {initial_len - len(out)} zero-volume bars after resample")
```

---

### 2.3 Fix GA Memo Sharing Across Processes
**Priority:** 🟡 P2 — ~2–3× wasted compute on duplicate evaluations

**Root cause:** `memo` dict is a regular Python dict passed by value to each subprocess; changes in worker processes don't propagate back to the parent. Each generation re-evaluates individuals that were already computed.

#### [MODIFY] `src/optimize/optimize_spa_ga.py`
Replace the standard dict with a `multiprocessing.Manager().dict()` for true shared state:

```python
from multiprocessing import Pool, cpu_count, Manager

def run_ga(...):
    with Manager() as manager:
        memo = manager.dict()   # shared across all workers
        best_params, best_score = None, -np.inf

        for gen in range(1, generations + 1):
            with Pool(processes=n_processes) as pool:
                eval_fn = partial(_evaluate_individual, df=df, memo=memo)
                results = pool.map(eval_fn, population)
            ...
```

Also improve fitness metric: replace raw final capital with **Sharpe Ratio** to penalize high-drawdown strategies:

```python
def calc_fitness_sharpe(df_original, params):
    """Fitness = Sharpe ratio instead of raw capital."""
    ...
    equity_curve = []
    # track equity at each step
    # return: mean_return / std_return * sqrt(len)
```

---

### 2.4 Clean Up Vestigial Files
**Priority:** 🟢 P3 — reduces project confusion

#### Files to archive (move to `archive/` at project root):

| File | Action |
|---|---|
| `src/optimize/Feature+Genetic+GPU_Version_2.py` | Move to `archive/` |
| `src/optimize/Fix_Feature.py` | Move to `archive/` |
| `src/optimize/Train_Test_Split_Feature.py` | Move to `archive/` |
| `src/train/train_ppo.py` | Move to `archive/` (superseded by `train_ppo_spa.py`) |
| `src/train/train_ppo_simple.py` | Move to `archive/` |
| `src/rl_env/rl_env_simple.py` | Move to `archive/` |
| `src/backtest/backtest_ppo.py` | Move to `archive/` |

**Do not delete** — preserve history; create `archive/README.md` explaining they are superseded prototypes.

---

### 2.5 Fix `scaffold.py` Path Mismatch
**Priority:** 🟢 P3

#### [MODIFY] `scaffold.py`
Change `"src/feature_engineering"` → `"src/features"` in the `dirs` list (line 42) and all matching file path entries (lines 89–96).

---

### 2.6 Remove Dead Code in `check_ga_readiness.py`
**Priority:** 🟢 P3

#### [MODIFY] `check_ga_readiness.py`
Remove the dead equity loop on lines 46–50 that builds an `eq` array which is never used. Keep the rest of the script intact.

---

### 2.7 Bump `requirements.txt` Dependencies
**Priority:** 🟢 P3

#### [MODIFY] `requirements.txt`

| Package | From | To | Reason |
|---|---|---|---|
| `numpy` | `==1.23.5` | `>=1.26.0,<2.0` | SB3 2.3.2 compatibility; avoid NumPy 2.x API breaks |
| `torch` | (comment only) | Add as comment with install instructions | Reproducibility documentation |

Add a section header comment explaining PyTorch must be installed separately with the correct CUDA variant, referencing the PyTorch installation selector at `pytorch.org/get-started`.

---

## Phase 3 — Feature Completion
> **Timeline:** ~1–2 weeks

---

### 3.1 Implement `src/paper_trade/paper_trade.py`
**Priority:** 🟡 P2

#### [NEW] `src/paper_trade/paper_trade.py`
A minimal live paper-trading loop against Binance Futures Testnet:

**Design:**
```
paper_trade.py
  ├── load_model(model_path)        → PPO model
  ├── live_feature_loop()           → real-time OHLCV + SPA features (rolling window)
  ├── PaperPortfolio                → tracks cash, qty, avg_entry, equity
  │     ├── execute_order(signal, price)
  │     └── get_state() → obs vector for model
  ├── run_loop()
  │     ├── fetch latest bar (REST poll, 15s interval)
  │     ├── update rolling feature window
  │     ├── model.predict(obs)
  │     ├── execute on paper portfolio
  │     └── log equity / trade to CSV
  └── CLI: --model, --pair, --interval, --log_csv, --duration_hours
```

**Safeguards:**
- Hard position size cap: max 30% of equity (mirror training `position_limit`)
- Configurable `--dry_run` flag (no order submission, log only)
- Graceful Ctrl+C shutdown with position summary

---

### 3.2 Build Full Streamlit Dashboard
**Priority:** 🟡 P2

#### [MODIFY] `src/dashboard/app.py`
Expand from 43-line stub to a multi-page dashboard. Proposed page structure:

**Page 1 — Data Explorer**
- Raw OHLCV candlestick chart (Plotly)
- SPA boundary overlay (h_inner, H_outer, l_inner, L_outer) with signal markers
- Feature correlation heatmap

**Page 2 — Training Monitor**
- TensorBoard log reader: plot reward, entropy, value_loss over timesteps
- Reads from `ppo_logs_spa/` directory

**Page 3 — Evaluation Results**
- Equity curve with drawdown area chart
- Trade log table (filterable by side, profitable/losing)
- Metrics summary card: Total Return, Sharpe, MaxDD, winrate, avg R/R
- Comparison panel: PPO Agent vs Baseline MA Crossover

**Page 4 — Paper Trading**
- Live position display (equity, qty, avg entry, unrealized P&L)
- Real-time equity chart (auto-refresh every 15s with `st.rerun()`)
- Trade history table

---

### 3.3 Deprecate or Integrate `AdvancedEnv.py`
**Priority:** 🟡 P2 — currently crashes on first use (wrong feature names)

**Decision tree:**

| Option | Criteria | Action |
|---|---|---|
| **Integrate** | If rolling-Sharpe reward and continuous action space are worth testing | Update `make_features.py` to output `MA7`, `MA30`, `close_z`, `volume`; create `train_ppo_advanced.py` |
| **Deprecate** | If team wants to stay with discrete PPO | Move to `archive/` with comment: "experimental — requires updated feature set" |

**Recommendation:** Deprecate for now (move to `archive/`). The primary RL loop with discrete actions is maturing and should stabilize first.

---

### 3.4 Add Unit Tests
**Priority:** 🟢 P3

#### [NEW] `tests/` directory structure:
```
tests/
  __init__.py
  test_spa_core.py          # test boundary math, signal generation, confirmation
  test_crypto_env.py        # test reset/step, action modes, shaping gates
  test_feature_pipeline.py  # test make_features_spa output shape/dtypes/meta
  test_utils.py             # test ensure_datetime_index edge cases
  conftest.py               # shared fixtures (synthetic OHLCV DataFrames)
```

**Key test cases:**
- `spa_core`: given known price series, verify signal direction is correct
- `crypto_env`: verify deadband gate blocks small trades; verify episode ends correctly; verify negative equity is handled
- `feature_pipeline`: verify meta.json is written with correct keys; verify no NaN in output
- `utils`: verify timezone handling, duplicate removal, missing column fallback

**Run command:**
```bash
python -m pytest tests/ -v
```

---

## Verification Plan

Once all changes are implemented, verify correctness in this order:

### V1 — Unit Tests (Phase 3 only, but runnable immediately for utils/spa)
```bash
cd d:\Workshop\TradeProject\automated_trading_using_ai
venv\Scripts\activate
python -m pytest tests/ -v
```

### V2 — Phase 1 Bug Fix Verification (manual, step-by-step)
Run the full pipeline end-to-end and check that:

1. **Meta contains `train_split`:**
   ```bash
   python src/features/make_features_spa.py --input data/raw/btc_15m.parquet
   cat data/features/btc_15m_spa_meta.json
   # ✅ Expect: "train_split" key present with value 0.8
   ```

2. **Eval uses meta's `train_split`, not its own hardcoded default:**
   ```bash
   python src/eval/eval_ppo_spa.py --model data/models/_eval_spa/best_model.zip \
       --features data/features/btc_15m_spa.parquet
   # ✅ Expect: [info] line shows train_split loaded from meta (0.8, not 0.7)
   ```

3. **Eval equity curve matches expected pure PnL (no phantom shaping):**
   - Run eval before and after the shaping fix
   - Compare `data/eval/ppo_spa_btc_15m_eval_best.csv` — equity values should differ
   - Post-fix equity is the ground truth

4. **`inactivity_steps` correctly propagated to train env:**
   - Add a temporary `print(f"[debug] inactivity_steps={self.inactivity_steps}")` in `crypto_env.py __init__`
   - Run `python src/train/train_ppo_spa.py --inactivity_steps 128 --timesteps 1000`
   - ✅ Expect: console shows `inactivity_steps=128`, not `256`
   - Remove the debug print after verification

### V3 — GA Memo Sharing Verification (Phase 2)
```bash
python src/optimize/optimize_spa_ga.py
# ✅ Expect: "Using N parallel workers" message
# Compare total evaluations printed vs population_size * generations
# Should be substantially fewer with shared memo
```

### V4 — Dashboard Smoke Test (Phase 3)
```bash
streamlit run src/dashboard/app.py -- --features data/features/btc_15m_spa.parquet
# ✅ Expect: Browser opens, all pages load without errors
# Check equity curve renders from data/eval/
# Check SPA overlay visible on candlestick chart
```

---

## Summary Table — All Changes

| # | File | Change Type | Priority | Phase |
|---|---|---|---|---|
| 1 | `src/features/make_features_spa.py` | MODIFY — add `train_split` + `freq_hint` + `periods_per_year` to meta | P0/P1 | 1 |
| 2 | `src/features/make_features.py` | MODIFY — fix `freq_hint` hardcode | P1 | 1 |
| 3 | `src/eval/eval_ppo_spa.py` | MODIFY — read split from meta; zero shaping | P0/P1 | 1 |
| 4 | `src/train/train_ppo_spa.py` | MODIFY — use `common_env_kwargs`; write split to meta | P0/P1 | 1 |
| 5 | `src/utils/__init__.py` | MODIFY — add canonical `ensure_datetime_index` | P2 | 2 |
| 6 | `src/features/make_features.py` | MODIFY — import from utils, remove duplicate | P2 | 2 |
| 7 | `src/features/make_features_spa.py` | MODIFY — import from utils, remove duplicate | P2 | 2 |
| 8 | `src/backtest/baseline_backtest_H1.py` | MODIFY — import from utils, remove duplicate | P2 | 2 |
| 9 | `src/optimize/optimize_spa_ga.py` | MODIFY — import from utils; fix memo sharing | P2 | 2 |
| 10 | `src/data_ingest/download_klines.py` | MODIFY — remove `ffill()` on O/H/L | P2 | 2 |
| 11 | `requirements.txt` | MODIFY — bump numpy; document torch | P3 | 2 |
| 12 | `scaffold.py` | MODIFY — fix path `feature_engineering` → `features` | P3 | 2 |
| 13 | `check_ga_readiness.py` | MODIFY — remove dead equity loop | P3 | 2 |
| 14 | `archive/` directory | NEW — move 7 vestigial scripts here | P3 | 2 |
| 15 | `src/paper_trade/paper_trade.py` | NEW — implement paper trading loop | P2 | 3 |
| 16 | `src/dashboard/app.py` | MODIFY — full multi-page dashboard | P2 | 3 |
| 17 | `src/rl_env/AdvancedEnv.py` | DEPRECATE — move to archive | P2 | 3 |
| 18 | `tests/` | NEW — unit test suite | P3 | 3 |

---

*Plan drafted: 2026-03-13 | Implementation not started*
