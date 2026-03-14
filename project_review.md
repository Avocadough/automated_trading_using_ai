# Project Review: `automated_trading_using_ai`

> Reviewed: 2026-03-13 | Branch: main

## 1. Architecture Overview

```
download_klines → make_features → CryptoTradingEnv → train_ppo_spa → eval_ppo_spa
                      ↕                                      ↕
                 spa_core (SPA algo)              baseline_backtest_H1 (MA crossover)
                 optimize/GA                      backtest_ppo
```

The project is a well-structured **PPO-based automated crypto trading** pipeline with:
- Binance Futures data ingestion (1m → any TF via resample)
- Feature engineering (RSI, MACD, ATR, BB, OBV, SPA boundaries)
- A custom Gymnasium RL environment (`CryptoTradingEnv`)
- Two training paths: standard PPO and GA-optimized SPA-feature PPO
- A baseline MA crossover backtest (vectorbt) for comparison
- Evaluation with trade-level metrics + equity curve export

---

## 2. Strengths ✅

- **Clean pipeline with metadata**: `_meta.json` ties features ↔ window_size across train/eval, preventing shape mismatches — good defensive design.
- **Execution smoothing in env**: deadband, min_hold_steps, cooldown_steps all reduce unrealistic over-trading. Very practical for RL trading envs.
- **Train/eval env shaping separation**: shaping rewards are correctly zeroed-out in the eval env to avoid biasing evaluation metrics.
- **Data quality gating**: `baseline_backtest_H1.py` has a thorough `check_data_quality` function (duplicates, gaps, OHLC validity, outlier detection).
- **Dual action modes**: `CryptoTradingEnv` supports both discrete and continuous action spaces — good extensibility.
- **Seed determinism**: `set_global_seeds()` covers random, numpy, torch, and cuDNN — reproducibility is taken seriously.
- **SPA algorithm modularised**: `spa_core.py` is a clean, well-decomposed implementation of the boundary-based signal logic.

---

## 3. Issues & Recommendations

### 🔴 Critical

#### `eval_ppo_spa.py` — `train_split` mismatch risk
The evaluator has `train_split=0.7` hardcoded as default, but the trainer uses `0.8`. If these differ, the eval set overlaps with training data, making OOS results invalid.

```python
# eval_ppo_spa.py line 250
ap.add_argument("--train_split", type=float, default=0.7)   # <-- should match trainer default (0.8)
```
**Fix**: Use the same default, or better, store `train_split` in `_meta.json` and read it in eval.

#### `eval_ppo_spa.py` — Residual reward shaping in eval env
`build_env()` in eval still has shaping on:
```python
flat_penalty_bps=0.01, inactivity_penalty_bps=0.01, turnover_reward_coeff=0.02
```
These are non-zero, which slightly distorts the eval equity curve. They should all be `0.0` for a clean OOS evaluation.

---

### 🟡 Medium

#### `train_ppo_spa.py` — `inactivity_steps` not forwarded to `build_env`
```python
main(inactivity_steps=128, ...)
# but train_env build call on line 197:
build_env(train_df, features, window_size,
    ...
    # inactivity_steps is MISSING from the call!
)
```
The `common_env_kwargs` dict is built (line 181) but then **not used** — both `build_env` calls are written out manually, and `inactivity_steps` is missing from the train env call. The default of `256` is used instead of the CLI value.

#### `check_ga_readiness.py` — Redundant equity computation
The equity loop (lines 46–50) builds `eq` incorrectly (off-by-one position logic) and the result is never used. The script falls back to `strat_returns` directly. The dead loop should be removed.

#### `download_klines.py` — `ffill()` on OHLC after resample
Lines 87–91 apply `ffill()` on `high` and `low` after resampling. This forward-fills price data into zero-volume periods, padding flat bars that could leak look-ahead. Consider dropping or flagging empty bars instead.

#### `make_features.py` — `freq_hint` hardcoded to `"1H"`
```python
"freq_hint": "1H",   # line 152
```
This is hardcoded regardless of the actual data frequency. The evaluator uses `periods_per_year` (hardcoded to `70000` ≈ 15m bars), but a mismatch on this will silently produce wrong Sharpe/annualization.

**Fix**: Infer or pass the actual freq, store it in meta, and use it in eval.

---

### 🟢 Low / Polish

| Location | Issue |
|---|---|
| `src/utils/` | Folder is empty — duplicate helpers exist across files (`ensure_datetime_index` is copy-pasted between `make_features.py` and `baseline_backtest_H1.py`) |
| `src/paper_trade/` | Module is empty (`__init__.py` only). Paper-trading against Binance Testnet is mentioned in the README but not implemented. |
| `src/dashboard/app.py` | 22-line stub that only prints — no actual Streamlit UI implemented. |
| `requirements.txt` | `numpy==1.23.5` and `pandas==2.2.2` is an old NumPy. `stable-baselines3==2.3.2` requires numpy ≥ 1.20 but modern SB3 works best with 1.26+. Consider bumping. |
| `optimize/` folder | Contains 4 files with overlapping names (`Fix_Feature.py`, `Feature+Genetic+GPU_Version_2.py`, `Train_Test_Split_Feature.py`) — suggests iterative dev without cleanup. |
| `scaffold.py` | References `src/feature_engineering/` but the actual module is at `src/features/`. Path mismatch — scaffold would create a dead folder. |

---

## 4. Prioritized Roadmap

| Priority | Task |
|---|---|
| P0 | Fix `train_split` default mismatch between trainer and evaluator |
| P0 | Zero out all shaping in eval env (`flat_penalty_bps`, `inactivity_penalty_bps`, `turnover_reward_coeff`) |
| P1 | Fix missing `inactivity_steps` in `train_ppo_spa.py` `build_env` call |
| P1 | Store & read `train_split` and `freq_hint` from `_meta.json` |
| P1 | Move `ensure_datetime_index` to `src/utils/` and import from there |
| P2 | Implement `paper_trade` module (even a minimal testnet loop) |
| P2 | Build the Streamlit dashboard (`app.py`) |
| P3 | Clean up `optimize/` — keep only the canonical version |
| P3 | Fix `scaffold.py` path (`feature_engineering` → `features`) |
