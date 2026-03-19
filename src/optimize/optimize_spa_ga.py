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
D_SET     = [20, 34, 55, 89, 144]
ALPHA_SET = [1.0, 2.0, 3.0, 5.0, 8.0]
GAMMA_SET = [0.5, 1.0, 1.5, 2.0]
M_MA_SET  = [3, 5, 8, 13]
SRC_SET   = ["close", "hl2", "hlc3"]


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
    """
    d, alpha, gamma, m_ma, source = params

    if d < 20:
        return -999.0

    try:
        signals = _compute_spa_signals(df, d, alpha, gamma, m_ma, source)
    except Exception:
        return -999.0

    # Skip NaN warmup period
    valid_start = max(d, 60)
    if valid_start >= len(df) - 100:
        return -999.0

    close = df["close"].values.astype("float64")[valid_start:]
    sigs  = signals[valid_start:]

    pnl = _backtest_signals(close, sigs)
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

    # --- Composite score ---
    # Weighted combination emphasizing risk-adjusted performance
    composite = (
        0.40 * max(sharpe, -5.0) +
        0.30 * max(sortino, -5.0) +
        0.30 * max(0.0, 1.0 - dd_penalty) * 5.0  # scale to match Sharpe range
        - trade_penalty
    )

    return float(composite)


# ============================================================
# Walk-Forward Validation
# ============================================================
def walk_forward_fitness(df: pd.DataFrame, params: tuple, n_folds: int = 3) -> float:
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
    population_size: int = 120,
    generations: int = 50,
    mutation_rate: float = 0.25,
    elite_frac: float = 0.10,     # top 10% survive unchanged
    n_wf_folds: int = 3,          # walk-forward folds
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

    # Initialize population
    def random_individual():
        return (
            random.choice(D_SET),
            random.choice(ALPHA_SET),
            random.choice(GAMMA_SET),
            random.choice(M_MA_SET),
            random.choice(SRC_SET),
        )

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

            # Uniform crossover
            child = tuple(p1[j] if random.random() < 0.5 else p2[j] for j in range(5))

            # Mutation (FIXED: child reassignment was lost in previous version)
            if random.random() < mutation_rate:
                child_list = list(child)
                idx = random.randint(0, 4)
                if   idx == 0: child_list[0] = random.choice(D_SET)
                elif idx == 1: child_list[1] = random.choice(ALPHA_SET)
                elif idx == 2: child_list[2] = random.choice(GAMMA_SET)
                elif idx == 3: child_list[3] = random.choice(M_MA_SET)
                else:          child_list[4] = random.choice(SRC_SET)
                child = tuple(child_list)

            offspring.append(child)

        # Next generation = elite + offspring
        population = elite + offspring

    # ============================================================
    # Final OOS Validation (on held-out eval data)
    # ============================================================
    train_sharpe = calc_composite_fitness(train_df, best_params)
    eval_sharpe  = calc_composite_fitness(eval_df, best_params)

    print(f"\n{'='*60}")
    print(f"  GA OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Best params: d={best_params[0]}, α={best_params[1]:.1f}, "
          f"γ={best_params[2]:.2f}, m_ma={best_params[3]}, src={best_params[4]}")
    print(f"  Walk-Forward Fitness: {best_score:.3f}")
    print(f"  In-Sample  Fitness:   {train_sharpe:.3f}")
    print(f"  Out-Of-Sample Fitness: {eval_sharpe:.3f}")

    degradation = 1.0 - (eval_sharpe / max(train_sharpe, 0.01))
    if eval_sharpe < 0:
        print(f"  ⚠️  DANGER: OOS fitness is NEGATIVE ({eval_sharpe:.3f})")
        print(f"     → SPA params are likely OVERFITTED. Consider wider search space.")
    elif degradation > 0.5:
        print(f"  ⚠️  WARNING: OOS degradation = {degradation:.0%}")
        print(f"     → Params may be overfitted. Consider larger dataset or fewer params.")
    else:
        print(f"  ✅  OOS degradation = {degradation:.0%} (acceptable)")
    print(f"{'='*60}")

    return best_params, best_score, eval_sharpe


def main():
    ap = argparse.ArgumentParser(description="GA-optimize SPA parameters (Walk-Forward)")
    ap.add_argument("--input",         type=str,   default="data/raw/btc_1h.parquet")
    ap.add_argument("--train_split",   type=float, default=0.8)
    ap.add_argument("--population",    type=int,   default=120)
    ap.add_argument("--generations",   type=int,   default=50)
    ap.add_argument("--mutation_rate", type=float, default=0.25)
    ap.add_argument("--elite_frac",    type=float, default=0.10)
    ap.add_argument("--wf_folds",      type=int,   default=3)
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
        "d":     int(best_params[0]),
        "alpha": float(best_params[1]),
        "beta":  float(best_params[2]),   # gamma → stored as 'beta' for compat
        "m_ma":  int(best_params[3]),
        "src":   str(best_params[4]),
        "wf_fitness":   float(wf_fitness),
        "eval_fitness": float(eval_fitness),
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\n[ok] Saved best params → {out_json}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
