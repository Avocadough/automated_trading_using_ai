<project_metadata>
<goal>Build a CNN+LSTM PPO Reinforcement Learning agent beating the S&P 500 index on 1H cryptocurrency data (Final year CS undergrad thesis).</goal>
<stack>Stable-Baselines3, PyTorch, Pandas, yfinance, CCXT, DEAP (Genetic Algorithm), VectorBT.</stack>
<status>Institutional-grade, train-serve parity achieved.</status>
</project_metadata>

<pipeline_architecture>
1. Data Ingestion: Fetch 1H BTCUSDT Klines -> Parquet
2. Parameter Optimization: Walk-Forward Genetic Algorithm (DEAP) over 11-gene space for optimal indicator lookbacks.
3. Feature Engineering: Make 23 strictly normalized, I(0) stationary features (MACD, ADX, SPA).
4. RL Environment: Custom Gymnasium tracking equity peak, differential Sharpe, and drawdown.
5. PPO Training: CNN+LSTM feature extractor on sequences of length 64 (batch_size=256, n_steps=4096).
6. Evaluation: OOS Testing vs BTC Buy&Hold and S&P 500 benchmark (`yfinance`). Includes Sharpe/Sortino comparison for Agent, BTC B&H, and S&P 500.
7. Live Trading: Real-time inference paper trader using CCXT.
</pipeline_architecture>

<core_components_state>
<features>
Pipeline utilizes 23 stationary, symmetrically clipped features.
Fixed "Oscillator Ceiling Blindness" (Agent missing mega-trends because RSI/Stoch maxed out) by injecting normalized trend-following features:
- `ADX_14` (scaled [-1, 1])
- `ema_dist_50` and `ema_dist_200` (normalized by EMA value itself)
- `MACD_hist_atr` (volatility-invariant momentum pulse)
Rule: NEVER pass raw unscaled prices to the LSTM.
</features>

<genetic_algorithm>
Location: `optimize_spa_ga.py`.
Optimizes 11 genes (5 SPA base parameters + 6 trend indicator lookbacks).
Anti-Overfit: 5-fold temporal Walk-Forward validation.
Selection: OOS-aware. Top 20% of in-sample population is evaluated on pure OOS fold; the highest scoring genome that maintains a *positive* OOS score wins.
Fix: Removed a catastrophic lookahead bias in MACD momentum thresholding caused by `np.std()` calculating over the entire future validation fold. Replaced with causal pd.Series.rolling().std().
</genetic_algorithm>

<environment>
Location: `crypto_env.py`
Reward Function: Differential Sharpe Ratio (Moody & Saffell 1998) scaled 1.0x to heavily reward risk-adjusted consistency over lottery-ticket variance.
Penalty: Incremental Peak Drawdown Penalty (`_max_session_dd`).
Context: Early randomness caused the agent to incur a 15% drawdown. A previous "continuous" drawdown penalty bled the agent *every single step* afterward, leading into the "Rational Cowardice" death spiral where the agent permanently outputted Action 1 (Flat) to end the episodic continuous bleeding. The environment now ONLY penalizes the agent incrementally when generating a *new* worst drawdown below a 5% tolerance zone.
Action Masking: Dynamic Trend Masking strictly forbids counter-trend actions (e.g., Shorting during a Bull regime) by intercepting the logits if `(close - EMA200)/EMA200 > 0.03`. Fix applied using `current_step - 1` to preserve absolute physical causality.
Trend-Alignment Reward Boost: After computing primary reward, if the agent takes a trend-aligned action (Long in Bull / Short in Bear) AND the step is profitable, reward is multiplied by `2.0x`. This cures "Volatility Fear" regime bias where the agent refuses to go Long during bull runs.
Position Limit: Increased from 30% to 50% of equity to improve capital efficiency towards the S&P 500 annualized return target (~12%).
</environment>

<agent>
Location: `train_ppo_spa.py`
Algorithm: Stable-Baselines3 PPO with a Custom PyTorch Feature Extractor (CNN x2 -> LSTM x1).
Configurations: `ent_coef=0.01` (prevents LSTM highest-entropy collapse), minimum learning rate pinned at `1e-5` (prevents late-stage neural plasticity death), `n_steps=4096`, `batch_size=512`, `n_epochs=5`, `max_grad_norm=0.5`. Total network capacity drastically shrunk from 262k to 16k parameters to cure catastrophic Train Sharpe overfitting.
</agent>
</core_components_state>

<critical_rules>
<rule id="1">DO NOT revert `drawdown_penalty` tightly bound to the `_max_session_dd` incremental delta. Changing this to a continuous flat tax will instantly resurrect the "Rational Cowardice" 100% Flat policy collapse.</rule>
<rule id="2">DO NOT introduce lookahead bias into the GA fitness function. Use causal rolling windows instead of `np.mean` or `np.std` over future slices.</rule>
<rule id="3">DO NOT increase `ent_coef` above 0.01. A standard 0.03 coefficient rewards chaos too aggressively, destroying the LSTM's capability to learn latent temporal dependencies.</rule>
<rule id="4">DO NOT drop or `reset_index(drop=True)` the `DatetimeIndex` when saving parquet files in the feature engineering pipeline. The `eval_ppo_spa.py` module explicitly relies on hourly timestamps to align the S&P 500 `yfinance` benchmark download.</rule>
</critical_rules>
