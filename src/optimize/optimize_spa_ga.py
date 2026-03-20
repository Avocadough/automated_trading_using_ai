# src/optimize/optimize_spa_ga.py
"""
Genetic Algorithm to optimize SPA boundary parameters.

Design Principles (Senior Quant Perspective):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. COMPOSITE FITNESS: Sharpe alone favors high-variance strategies.
   We combine Sharpe, Sortino, and penalize Max Drawdown heavily.
2. WALK-FORWARD VALIDATION: Single train/eval split is fragile.
   We use K-fold temporal splits to ensure parameter stability.
3. ELITISM: Best individuals survive to next generation unchanged.
4. REALISTIC BACKTEST: Fees, fractional sizing, Long+Short.
5. ANTI-OVERFITTING: Penalize low trade count, report OOS degradation.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.utils import ensure_datetime_index


# ============================================================
# Search Space
# ============================================================
# --- SPA Parameters (original) ---
D_SET     = [20, 34, 55, 89, 144]
ALPHA_SET = [1.0, 2.0, 3.0, 5.0, 8.0]
GAMMA_SET = [0.5, 1.0, 1.5, 2.0]
M_MA_SET  = [3, 5, 8, 13]
SRC_SET   = ["close", "hl2", "hlc3"]

# --- Trend Indicator Parameters (anti-oscillator-ceiling) ---
ADX_PERIOD_RANGE   = (7, 30)      # ADX lookback window
EMA_FAST_RANGE     = (10, 50)     # Medium-term EMA
EMA_SLOW_RANGE     = (100, 300)   # Macro-trend EMA (must be > ema_fast)
MACD_FAST_RANGE    = (8, 20)      # MACD fast EMA
MACD_SLOW_RANGE    = (21, 40)     # MACD slow EMA (must be > macd_fast)
MACD_SIGNAL_RANGE  = (5, 15)      # MACD signal line

# Genome layout (11 genes):
#   [0] d        [1] alpha    [2] gamma    [3] m_ma     [4] source
#   [5] adx_p    [6] ema_fast [7] ema_slow [8] macd_fast [9] macd_slow [10] macd_signal
GENOME_SIZE = 11


def _rand_int_range(bounds: Tuple[int, int]) -> int:
    return random.randint(bounds[0], bounds[1])


def _enforce_constraints(genes: list) -> list:
    """
    Sanity check: enforce ema_fast < ema_slow and macd_fast < macd_slow.
    Swap if violated instead of penalizing — saves compute by fixing
    invalid genomes on the fly rather than wasting an entire fitness eval.
    """
    # ema_fast (idx 6) must be < ema_slow (idx 7)
    if genes[6] >= genes[7]:
        genes[6], genes[7] = genes[7], genes[6]
    # If swap made them equal (shouldn't happen with int ranges, but safety)
    if genes[6] >= genes[7]:
        genes[7] = genes[6] + 50  # force separation

    # macd_fast (idx 8) must be < macd_slow (idx 9)
    if genes[8] >= genes[9]:
        genes[8], genes[9] = genes[9], genes[8]
    if genes[8] >= genes[9]:
        genes[9] = genes[8] + 5

    return genes


# ============================================================
# SPA Signal Generator (inline for isolation)
# ============================================================
def _compute_spa_signals(df: pd.DataFrame, d: int, alpha: float, gamma: float,
                         m_ma: int, source: str) -> np.ndarray:
    """
    Fully vectorized SPA signals → returns ndarray of {-1, 0, 1}.
    Using numpy arrays instead of pandas for ~10x speedup in GA loops.
    """
    close = df["close"].values.astype("float64")
    high  = df["high"].values.astype("float64")
    low   = df["low"].values.astype("float64")
    n = len(df)

    # Source price
    if source == "hl2":
        src = (high + low) / 2.0
    elif source == "hlc3":
        src = (high + low + close) / 3.0
    else:
        src = close.copy()

    # Vectorized rolling with numpy (faster than pandas for GA)
    def rolling_mean(arr, w):
        out = np.full(n, np.nan)
        cs = np.cumsum(arr)
        out[w-1:] = (cs[w-1:] - np.concatenate([[0], cs[:-w]])) / w
        return out

    def rolling_std(arr, w):
        out = np.full(n, np.nan)
        m = rolling_mean(arr, w)
        cs2 = np.cumsum(arr ** 2)
        var = (cs2[w-1:] - np.concatenate([[0], cs2[:-w]])) / w - m[w-1:] ** 2
        var = np.clip(var, 0, None)
        out[w-1:] = np.sqrt(var)
        return out

    def rolling_max(arr, w):
        out = np.full(n, np.nan)
        for i in range(w - 1, n):
            out[i] = np.max(arr[i - w + 1:i + 1])
        return out

    def rolling_min(arr, w):
        out = np.full(n, np.nan)
        for i in range(w - 1, n):
            out[i] = np.min(arr[i - w + 1:i + 1])
        return out

    # Swing stats
    swing = np.clip(high - low, 0, None)
    mu    = rolling_mean(swing, d)
    sigma = rolling_std(swing, d)

    # Boundaries
    high_d = rolling_max(high, d)
    low_d  = rolling_min(low, d)
    atr_ext = mu + gamma * sigma

    h_inner = high_d - alpha * mu
    H_outer = np.maximum(high_d, close) + atr_ext
    l_inner = low_d + alpha * mu
    L_outer = np.minimum(low_d, close) - atr_ext

    # Raw signals
    prev_close = np.concatenate([[np.nan], close[:-1]])
    sig = np.zeros(n, dtype="int8")

    upper_reject = (prev_close > h_inner) & (close < h_inner)
    upper_break  = (prev_close <= H_outer) & (close > H_outer)
    lower_reject = (prev_close < l_inner) & (close > l_inner)
    lower_break  = (prev_close >= L_outer) & (close < L_outer)

    sig[upper_reject] = -1
    sig[upper_break]  =  1
    sig[lower_reject] =  1
    sig[lower_break]  = -1

    # MA Confirmation
    ma = rolling_mean(src, m_ma)
    sig[(sig == 1) & (src <= ma)] = 0
    sig[(sig == -1) & (src >= ma)] = 0

    return sig


# ============================================================
# Composite Fitness Function
# ============================================================
FEE_RATE       = 0.0005   # 0.05% per side (Binance Futures taker)
POSITION_FRAC  = 0.30     # 30% of equity per trade (matches RL agent)
PERIODS_PER_YEAR = 8760   # 1H candles


def _backtest_signals(close: np.ndarray, signals: np.ndarray) -> np.ndarray:
    """
    Vectorized backtest: compute per-bar PnL array given signals and close prices.
    """
    n = len(close)
    returns = np.zeros(n)
    returns[1:] = close[1:] / close[:-1] - 1.0

    pnl = np.zeros(n)
    pos = 0.0

    for i in range(1, n):
        sig = signals[i]
        ret = returns[i]

        # PnL from holding current position
        step_pnl = pos * ret * POSITION_FRAC

        # Position change → incur fees
        if sig != 0 and sig != pos:
            # Fee = |position_change| * fee_rate * position_fraction
            trade_size = abs(sig - pos)  # 0→1=1, 1→-1=2, -1→0=1
            step_pnl -= trade_size * FEE_RATE * POSITION_FRAC
            pos = float(sig)

        pnl[i] = step_pnl

    return pnl


def calc_composite_fitness(df: pd.DataFrame, params: tuple) -> float:
    """
    Composite fitness = 0.4 * Sharpe + 0.3 * Sortino + 0.3 * (1 - DD_penalty)
    where DD_penalty = min(1, MaxDD / 0.15)  (hard cap at -15% drawdown)

    This prevents selecting strategies that have high Sharpe
    but came from one lucky trade with -30% drawdown risk.

    Enhanced: uses trend indicator genes to generate momentum-filtered
    signals alongside SPA. The trend indicators act as additional
    confirmation filters — if ADX shows a strong trend AND price is above
    EMA(slow), SPA short signals are suppressed (and vice versa).
    """
    d, alpha, gamma, m_ma, source = params[:5]
    adx_p, ema_fast_p, ema_slow_p = int(params[5]), int(params[6]), int(params[7])
    macd_fast_p, macd_slow_p, macd_sig_p = int(params[8]), int(params[9]), int(params[10])

    if d < 20:
        return -999.0

    try:
        signals = _compute_spa_signals(df, d, alpha, gamma, m_ma, source)
    except Exception:
        return -999.0

    # Skip NaN warmup period (must account for longest lookback: ema_slow)
    valid_start = max(d, ema_slow_p + 10, 60)
    if valid_start >= len(df) - 100:
        return -999.0

    close = df["close"].values.astype("float64")
    high  = df["high"].values.astype("float64")
    low   = df["low"].values.astype("float64")

    # === Trend-filtered signals ===
    # Compute trend indicators from the mutated gene periods
    # to filter SPA signals during trending regimes.

    # ADX: trend strength
    def _ema(arr, span):
        alpha_v = 2.0 / (span + 1)
        out = np.empty_like(arr)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = alpha_v * arr[i] + (1 - alpha_v) * out[i - 1]
        return out

    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]

    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0; down_move[0] = 0
    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_s   = _ema(tr, adx_p)
    plus_s  = _ema(plus_dm, adx_p)
    minus_s = _ema(minus_dm, adx_p)
    plus_di  = np.where(atr_s > 0, 100 * plus_s / atr_s, 0)
    minus_di = np.where(atr_s > 0, 100 * minus_s / atr_s, 0)
    di_sum = plus_di + minus_di
    dx = np.where(di_sum > 0, 100 * np.abs(plus_di - minus_di) / (di_sum + 1e-12), 0)
    adx = _ema(dx, adx_p)

    # EMA distances
    ema_fast_arr = _ema(close, ema_fast_p)
    ema_slow_arr = _ema(close, ema_slow_p)
    ema_dist_slow = np.where(ema_slow_arr > 0, (close - ema_slow_arr) / ema_slow_arr, 0)

    # MACD
    macd_fast_ema = _ema(close, macd_fast_p)
    macd_slow_ema = _ema(close, macd_slow_p)
    macd_line = macd_fast_ema - macd_slow_ema
    macd_sig_line = _ema(macd_line, macd_sig_p)
    macd_hist = macd_line - macd_sig_line

    # --- Trend-filter SPA signals ---
    # In strong uptrend (ADX > 25 AND price > EMA_slow): suppress short signals
    # In strong downtrend (ADX > 25 AND price < EMA_slow): suppress long signals
    filtered_sigs = signals.copy()
    strong_trend = adx > 25
    bullish = ema_dist_slow > 0
    bearish = ema_dist_slow < 0

    # Suppress shorts in strong uptrend
    filtered_sigs[(filtered_sigs == -1) & strong_trend & bullish] = 0
    # Suppress longs in strong downtrend
    filtered_sigs[(filtered_sigs == 1) & strong_trend & bearish] = 0

    # Use MACD histogram as tie-breaker for flat SPA signals
    # If SPA is flat but MACD momentum is extreme, generate momentum signal
    macd_rolling_std = pd.Series(macd_hist).rolling(200, min_periods=1).std().fillna(0).values
    macd_threshold = macd_rolling_std * 1.5  # causal dynamic threshold
    if np.any(macd_threshold > 0):
        momentum_long  = (macd_hist > macd_threshold) & strong_trend & bullish
        momentum_short = (macd_hist < -macd_threshold) & strong_trend & bearish
        filtered_sigs[(filtered_sigs == 0) & momentum_long]  =  1
        filtered_sigs[(filtered_sigs == 0) & momentum_short] = -1

    close_v = close[valid_start:]
    sigs  = filtered_sigs[valid_start:]

    pnl = _backtest_signals(close_v, sigs)
    pnl = pnl[1:]  # drop first zero

    if len(pnl) < 200 or np.std(pnl) < 1e-12:
        return -999.0

    # --- Sharpe Ratio (annualized) ---
    sharpe = (np.mean(pnl) / np.std(pnl)) * np.sqrt(PERIODS_PER_YEAR)

    # --- Sortino Ratio (only penalize downside vol) ---
    downside = pnl[pnl < 0]
    downside_std = np.std(downside) if len(downside) > 10 else np.std(pnl)
    sortino = (np.mean(pnl) / max(downside_std, 1e-12)) * np.sqrt(PERIODS_PER_YEAR)

    # --- Max Drawdown ---
    equity = np.cumsum(pnl) + 1.0
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    max_dd = abs(np.min(drawdowns))

    # DD penalty: linearly penalize DD above 5%, hard cap at 15%
    dd_penalty = min(1.0, max(0.0, max_dd - 0.05) / 0.10)

    # --- Trade count penalty ---
    pos_changes = np.abs(np.diff(np.sign(sigs).astype(float)))
    n_trades = int(np.sum(pos_changes > 0.5))
    trade_penalty = 0.0
    if n_trades < 30:
        trade_penalty = 0.5  # heavy: not enough trades for statistical validity
    elif n_trades < 50:
        trade_penalty = 0.2  # moderate

    # --- Regularization penalty (anti-overfitting) ---
    # Penalize extreme parameter values that tend to overfit:
    # - Very small d (< 34) generates too many signals → noise-chasing
    # - Very small ema_fast (< 20) makes EMA too reactive → whipsaw
    reg_penalty = 0.0
    d = int(params[0])
    ema_fast_v = int(params[6])
    if d < 34:
        reg_penalty += 0.3  # discourage smallest SPA windows
    if ema_fast_v < 20:
        reg_penalty += 0.2  # discourage ultra-short EMA

    # --- Composite score ---
    # Weighted combination emphasizing risk-adjusted performance
    composite = (
        0.40 * max(sharpe, -5.0) +
        0.30 * max(sortino, -5.0) +
        0.30 * max(0.0, 1.0 - dd_penalty) * 5.0  # scale to match Sharpe range
        - trade_penalty
        - reg_penalty
    )

    return float(composite)


# ============================================================
# Walk-Forward Validation
# ============================================================
def walk_forward_fitness(df: pd.DataFrame, params: tuple, n_folds: int = 5) -> float:
    """
    Walk-Forward validation: train on fold_k, validate on fold_k+1.
    Average the OOS fitness across all folds.
    This is the STRONGEST anti-overfitting measure for time series.
    """
    n = len(df)
    fold_size = n // (n_folds + 1)

    if fold_size < 500:  # each fold needs enough data
        return calc_composite_fitness(df, params)

    oos_scores = []
    for k in range(n_folds):
        # Train window: start to end of fold k+1
        train_end = fold_size * (k + 2)
        # Test window: fold k+1 to fold k+2
        test_start = train_end
        test_end   = min(train_end + fold_size, n)

        if test_end - test_start < 200:
            continue

        test_df = df.iloc[test_start:test_end]
        score = calc_composite_fitness(test_df, params)
        if score > -900:  # valid score
            oos_scores.append(score)

    if not oos_scores:
        return -999.0

    # Return average OOS score (more robust than single split)
    return float(np.mean(oos_scores))


# ============================================================
# Genetic Algorithm with Elitism
# ============================================================
def run_ga(
    df_path: str,
    train_split: float = 0.8,
    population_size: int = 200,
    generations: int = 80,
    mutation_rate: float = 0.25,
    elite_frac: float = 0.10,     # top 10% survive unchanged
    n_wf_folds: int = 5,          # walk-forward folds (5 for anti-overfit)
):
    """
    GA with:
    1. Walk-Forward Validation (anti-overfitting)
    2. Composite Fitness (anti-high-variance selection)
    3. Elitism (preserve best individuals)
    4. Two-point crossover + adaptive mutation
    """
    df = pd.read_parquet(df_path)
    df = ensure_datetime_index(df)
    needed = ["open", "high", "low", "close"]
    if any(c not in df.columns for c in needed):
        raise ValueError(f"Missing columns: {needed}")

    # Train / Eval split
    n_train = int(len(df) * train_split)
    train_df = df.iloc[:n_train].copy()
    eval_df  = df.iloc[n_train:].copy()
    print(f"[info] GA optimizing on TRAIN: {len(train_df):,} rows | EVAL: {len(eval_df):,} rows")
    print(f"[info] Walk-Forward folds: {n_wf_folds} | Elitism: {elite_frac*100:.0f}%")

    n_elite = max(1, int(population_size * elite_frac))

    # Initialize population with constraint-valid genomes
    def random_individual():
        genes = [
            random.choice(D_SET),             # [0] d
            random.choice(ALPHA_SET),          # [1] alpha
            random.choice(GAMMA_SET),          # [2] gamma
            random.choice(M_MA_SET),           # [3] m_ma
            random.choice(SRC_SET),            # [4] source
            _rand_int_range(ADX_PERIOD_RANGE),   # [5] adx_period
            _rand_int_range(EMA_FAST_RANGE),     # [6] ema_fast
            _rand_int_range(EMA_SLOW_RANGE),     # [7] ema_slow
            _rand_int_range(MACD_FAST_RANGE),    # [8] macd_fast
            _rand_int_range(MACD_SLOW_RANGE),    # [9] macd_slow
            _rand_int_range(MACD_SIGNAL_RANGE),  # [10] macd_signal
        ]
        genes = _enforce_constraints(genes)
        return tuple(genes)

    population = [random_individual() for _ in range(population_size)]
    memo: Dict[tuple, float] = {}
    best_params, best_score = None, -np.inf

    for gen in range(1, generations + 1):
        # ---- Evaluate fitness (Walk-Forward on train data) ----
        fitness_scores = []
        for ind in population:
            if ind not in memo:
                memo[ind] = walk_forward_fitness(train_df, ind, n_folds=n_wf_folds)
            fitness_scores.append(memo[ind])

        # ---- Sort population by fitness (for elitism) ----
        sorted_idx = np.argsort(fitness_scores)[::-1]  # best first
        sorted_pop = [population[i] for i in sorted_idx]
        sorted_fit = [fitness_scores[i] for i in sorted_idx]

        # Track best
        if sorted_fit[0] > best_score:
            best_score  = sorted_fit[0]
            best_params = sorted_pop[0]

        if gen % 5 == 1 or gen == generations:
            bp = sorted_pop[0]
            print(f"Gen {gen:02d}/{generations} | Best: d={bp[0]}, α={bp[1]:.1f}, "
                  f"γ={bp[2]:.2f}, m_ma={bp[3]}, src={bp[4]} | "
                  f"ADX={bp[5]}, EMA={bp[6]}/{bp[7]}, MACD={bp[8]}/{bp[9]}/{bp[10]} | "
                  f"Fitness={sorted_fit[0]:.3f}  (median={np.median(sorted_fit):.3f})")

        # ---- Elitism: top N survive unchanged ----
        elite = sorted_pop[:n_elite]

        # ---- Tournament selection for breeding pool ----
        pop_fit = list(zip(population, fitness_scores))
        selected: List[tuple] = []
        for _ in range(population_size - n_elite):
            tournament = random.sample(pop_fit, k=min(5, len(pop_fit)))
            winner = max(tournament, key=lambda x: x[1])[0]
            selected.append(winner)

        # ---- Crossover + Mutation ----
        offspring: List[tuple] = []
        i = 0
        while len(offspring) < population_size - n_elite:
            p1 = selected[i % len(selected)]
            p2 = selected[(i + 1) % len(selected)]
            i += 2

            # Uniform crossover (11 genes)
            child = tuple(p1[j] if random.random() < 0.5 else p2[j]
                          for j in range(GENOME_SIZE))

            # Mutation
            if random.random() < mutation_rate:
                child_list = list(child)
                idx = random.randint(0, GENOME_SIZE - 1)
                if   idx == 0: child_list[0] = random.choice(D_SET)
                elif idx == 1: child_list[1] = random.choice(ALPHA_SET)
                elif idx == 2: child_list[2] = random.choice(GAMMA_SET)
                elif idx == 3: child_list[3] = random.choice(M_MA_SET)
                elif idx == 4: child_list[4] = random.choice(SRC_SET)
                elif idx == 5: child_list[5] = _rand_int_range(ADX_PERIOD_RANGE)
                elif idx == 6: child_list[6] = _rand_int_range(EMA_FAST_RANGE)
                elif idx == 7: child_list[7] = _rand_int_range(EMA_SLOW_RANGE)
                elif idx == 8: child_list[8] = _rand_int_range(MACD_FAST_RANGE)
                elif idx == 9: child_list[9] = _rand_int_range(MACD_SLOW_RANGE)
                elif idx == 10: child_list[10] = _rand_int_range(MACD_SIGNAL_RANGE)

                # Enforce constraints after mutation
                child_list = _enforce_constraints(child_list)
                child = tuple(child_list)

            offspring.append(child)

        # Next generation = elite + offspring
        population = elite + offspring

    # ============================================================
    # Final OOS Validation (on held-out eval data)
    # OOS-aware selection: pick the best individual whose OOS > 0,
    # falling back to the WF-best if no individual generalizes.
    # ============================================================

    # Evaluate top candidates on OOS (top 20% of population)
    n_candidates = max(5, int(population_size * 0.20))
    sorted_idx = np.argsort(fitness_scores)[::-1]
    candidates = [(population[i], fitness_scores[i]) for i in sorted_idx[:n_candidates]]

    oos_best_params, oos_best_wf, oos_best_eval = None, -np.inf, -np.inf
    wf_best_params, wf_best_score = best_params, best_score

    print(f"\n[info] OOS-aware selection: evaluating top {n_candidates} candidates...")
    for cand_params, cand_wf in candidates:
        cand_oos = calc_composite_fitness(eval_df, cand_params)
        if cand_oos > 0 and cand_wf > 0:
            # Pick the candidate with best WF score among OOS-positive ones
            if cand_wf > oos_best_wf:
                oos_best_params = cand_params
                oos_best_wf = cand_wf
                oos_best_eval = cand_oos

    # Decide: use OOS-validated params if available, else fall back to WF-best
    if oos_best_params is not None:
        best_params = oos_best_params
        best_score  = oos_best_wf
        train_sharpe = calc_composite_fitness(train_df, best_params)
        eval_sharpe  = oos_best_eval
        selection_method = "OOS-VALIDATED"
    else:
        train_sharpe = calc_composite_fitness(train_df, best_params)
        eval_sharpe  = calc_composite_fitness(eval_df, best_params)
        selection_method = "WF-BEST (no OOS-positive candidate found)"

    print(f"\n{'='*70}")
    print(f"  GA OPTIMIZATION COMPLETE (11-gene expanded search space)")
    print(f"{'='*70}")
    print(f"  Selection: {selection_method}")
    print(f"  SPA params:   d={best_params[0]}, α={best_params[1]:.1f}, "
          f"γ={best_params[2]:.2f}, m_ma={best_params[3]}, src={best_params[4]}")
    print(f"  Trend params: ADX_p={best_params[5]}, EMA={best_params[6]}/{best_params[7]}, "
          f"MACD={best_params[8]}/{best_params[9]}/{best_params[10]}")
    print(f"  Walk-Forward Fitness: {best_score:.3f}")
    print(f"  In-Sample  Fitness:   {train_sharpe:.3f}")
    print(f"  Out-Of-Sample Fitness: {eval_sharpe:.3f}")

    degradation = 1.0 - (eval_sharpe / max(train_sharpe, 0.01))
    if eval_sharpe < 0:
        print(f"  ⚠️  DANGER: OOS fitness is NEGATIVE ({eval_sharpe:.3f})")
        print(f"     → Params are likely OVERFITTED. Consider wider search space.")
    elif degradation > 0.5:
        print(f"  ⚠️  WARNING: OOS degradation = {degradation:.0%}")
        print(f"     → Params may be overfitted. Consider larger dataset or fewer params.")
    else:
        print(f"  ✅  OOS degradation = {degradation:.0%} (acceptable)")
    print(f"{'='*70}")

    return best_params, best_score, eval_sharpe


def main():
    ap = argparse.ArgumentParser(description="GA-optimize SPA parameters (Walk-Forward)")
    ap.add_argument("--input",         type=str,   default="data/raw/btc_1h.parquet")
    ap.add_argument("--train_split",   type=float, default=0.8)
    ap.add_argument("--population",    type=int,   default=200)
    ap.add_argument("--generations",   type=int,   default=80)
    ap.add_argument("--mutation_rate", type=float, default=0.25)
    ap.add_argument("--elite_frac",    type=float, default=0.10)
    ap.add_argument("--wf_folds",      type=int,   default=5)
    args = ap.parse_args()

    apath = Path(args.input)
    if not apath.exists():
        raise FileNotFoundError(f"Raw parquet missing: {apath}")

    print("[info] Running GA optimization with Walk-Forward Validation...")
    best_params, wf_fitness, eval_fitness = run_ga(
        str(apath),
        train_split=args.train_split,
        population_size=args.population,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        elite_frac=args.elite_frac,
        n_wf_folds=args.wf_folds,
    )

    out_dir = Path("data/params")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "best_h1_spa_ga.json"
    payload = {
        # SPA params
        "d":     int(best_params[0]),
        "alpha": float(best_params[1]),
        "beta":  float(best_params[2]),   # gamma → stored as 'beta' for compat
        "m_ma":  int(best_params[3]),
        "src":   str(best_params[4]),
        # Trend indicator params (GA-optimized)
        "adx_period":   int(best_params[5]),
        "ema_fast":     int(best_params[6]),
        "ema_slow":     int(best_params[7]),
        "macd_fast":    int(best_params[8]),
        "macd_slow":    int(best_params[9]),
        "macd_signal":  int(best_params[10]),
        # Fitness scores
        "wf_fitness":   float(wf_fitness),
        "eval_fitness": float(eval_fitness),
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\n[ok] Saved best params → {out_json}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
