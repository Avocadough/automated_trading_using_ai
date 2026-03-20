# src/eval/backtest_unseen.py
"""
Pure Out-of-Sample Walk-Forward Backtest on Completely Unseen Data.

PURPOSE (CS Thesis Final Chapter):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Evaluate the trained RL agent on data it has NEVER seen (Jan 2025 → Mar 2026).
This is the ultimate test of generalisation — no hyperparameter tuning,
no architecture changes, just a pure forward test.

CRITICAL ML LAW — NO DATA LEAKAGE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The CryptoTradingEnv normalises features using mean/std statistics.
If we compute these from the UNSEEN data, we introduce lookahead bias
(the agent implicitly "knows" the future distribution).

Solution: We re-derive norm_mu / norm_std from the ORIGINAL training
parquet's first 80% (the training split) — mathematically identical to
what the agent saw during training. These are injected via norm_mu_ext
and norm_std_ext into the CryptoTradingEnv.

USAGE:
━━━━━━
# Step 1: Download & featurise unseen data
python src/data_ingest/download_klines.py \\
    --pair BTCUSDT --start 2025-01-01 --end 2026-03-15 \\
    --interval 1h --output data/raw/btc_1h_unseen.parquet

python src/features/make_features_spa.py \\
    --input data/raw/btc_1h_unseen.parquet \\
    --output data/features/btc_1h_unseen.parquet \\
    --train_split 1.0

# Step 2: Run this backtest
python src/eval/backtest_unseen.py \\
    --model data/models/ppo_spa_btc_1h.zip \\
    --features-train data/features/btc_1h_spa.parquet \\
    --features-unseen data/features/btc_1h_unseen.parquet \\
    --out_dir data/eval/unseen
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.rl_env.crypto_env import CryptoTradingEnv
from src.eval.eval_ppo_spa import (
    build_env,
    _run_episode,
    fetch_sp500_equity,
    load_meta,
)
from src.eval.academic_report import (
    AcademicReport,
    compute_metrics,
    compute_train_sharpe,
)
from src.eval.institutional_tearsheet import (
    run_institutional_eval,
    RiskAnalytics,
)


# ====================================================================
# Norm Stats Extraction (from original training parquet)
# ====================================================================

def extract_training_norm_stats(
    train_features_path: Path,
) -> tuple[pd.Series, pd.Series, list[str], int, int]:
    """
    Re-derive norm_mu and norm_std from the ORIGINAL training parquet.

    This is mathematically identical to what train_ppo_spa.py computes
    at lines 210-212. We load the same parquet, apply the same 80%
    split, and compute mean()/std() on the training portion.

    Returns: (norm_mu, norm_std, features, window_size, periods_per_year)
    """
    features, window_size, train_split, periods_per_year = load_meta(
        train_features_path
    )

    print(f"[norm] Loading original training parquet: {train_features_path}")
    df_all = pd.read_parquet(train_features_path)
    need_cols = ["close"] + features
    missing = [c for c in need_cols if c not in df_all.columns]
    if missing:
        raise ValueError(f"Missing columns in training parquet: {missing}")

    df_all = df_all[need_cols].dropna().reset_index(drop=True)
    n_train = int(len(df_all) * train_split)

    print(f"[norm] Total rows: {len(df_all):,} | Training split: "
          f"{train_split:.0%} → {n_train:,} rows")

    train_df = df_all.iloc[:n_train]
    feat_df = train_df[features].astype("float64")
    norm_mu = feat_df.mean()
    norm_std = feat_df.std().replace(0, 1.0)

    print(f"[norm] Extracted norm stats from {n_train:,} training rows")
    print(f"[norm] norm_mu range: [{norm_mu.min():.6f}, {norm_mu.max():.6f}]")
    print(f"[norm] norm_std range: [{norm_std.min():.6f}, {norm_std.max():.6f}]")

    return norm_mu, norm_std, features, window_size, periods_per_year


# ====================================================================
# Main Backtest
# ====================================================================

def run_unseen_backtest(
    model_path: Path,
    train_features_path: Path,
    unseen_features_path: Path,
    out_dir: Path,
):
    """
    Execute the pure OOS walk-forward backtest.

    Pipeline:
        1. Extract norm_mu/norm_std from original training data
        2. Load unseen feature parquet (Jan 2025 → Mar 2026)
        3. Build CryptoTradingEnv with training norm stats (no leak)
        4. Run episode → equity, actions, trades
        5. Fetch S&P 500 for exact same window
        6. Generate Academic Report + Institutional Tearsheet
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Extract training norm stats ----
    print(f"\n{'='*70}")
    print(f" PURE OOS WALK-FORWARD BACKTEST")
    print(f" Unseen Window: Jan 2025 → Mar 2026")
    print(f"{'='*70}\n")

    norm_mu, norm_std, features, window_size, periods_per_year = \
        extract_training_norm_stats(train_features_path)

    # ---- 2. Load unseen data ----
    print(f"\n[data] Loading unseen features: {unseen_features_path}")
    unseen_df = pd.read_parquet(unseen_features_path)

    # Preserve DatetimeIndex for S&P 500 alignment
    raw_datetime_index = (
        unseen_df.index
        if isinstance(unseen_df.index, pd.DatetimeIndex)
        else None
    )

    need_cols = ["close"] + features
    missing = [c for c in need_cols if c not in unseen_df.columns]
    if missing:
        raise ValueError(
            f"Unseen parquet missing columns: {missing}\n"
            f"Ensure you ran make_features_spa.py with the same params."
        )

    unseen_df = unseen_df[need_cols].dropna()
    print(f"[data] Unseen rows: {len(unseen_df):,}")
    if raw_datetime_index is not None:
        # Align datetime index after dropna
        valid_mask = ~unseen_df.index.duplicated(keep="first")
        unseen_df = unseen_df[valid_mask]
        # Re-check alignment
        if len(raw_datetime_index) != len(unseen_df):
            # Re-derive from the parquet index
            tmp = pd.read_parquet(unseen_features_path)
            raw_datetime_index = tmp.index[tmp.index.isin(unseen_df.index)]
            if isinstance(raw_datetime_index, pd.DatetimeIndex):
                unseen_df.index = raw_datetime_index[:len(unseen_df)]

    if raw_datetime_index is not None and isinstance(unseen_df.index, pd.DatetimeIndex):
        print(f"[data] Date range: {unseen_df.index.min()} → {unseen_df.index.max()}")

    # Reset index for the env (it expects integer index)
    unseen_df_reset = unseen_df.reset_index(drop=True)

    # ---- 3. Load model ----
    print(f"\n[model] Loading: {model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(model_path, device=device)
    print(f"[model] Device: {device}")

    # ---- 4. Build env with TRAINING norm stats ----
    print(f"\n[env] Building CryptoTradingEnv with TRAINING norm stats (no leak)")
    env = CryptoTradingEnv(
        df=unseen_df_reset,
        features=features,
        window_size=window_size,
        norm_mu=norm_mu,
        norm_std=norm_std,
        initial_balance=10_000.0,
        taker_fee=0.0005,
        position_limit=0.30,
        slippage_bps=0.5,
        reward_scale=10.0,
        reward_clip=5.0,
        normalize=True,
        action_mode="discrete",
        deadband_frac=0.05,
        min_hold_steps=0,
        cooldown_steps=0,
        flat_penalty_bps=0.0,
        inactivity_steps=256,
        inactivity_penalty_bps=0.0,
        turnover_reward_coeff=0.0,
        trade_threshold=0.01,
        drawdown_penalty_coeff=0.0,
        liquidation_threshold=0.5,
        sharpe_window=48,
        sharpe_eta=0.01,
    )

    # ---- 5. Run episode ----
    print(f"[run] Running walk-forward episode on unseen data...")
    equity, actions, pos_frac, trades, n_trades = _run_episode(model, env)
    print(f"[run] Episode complete: {len(equity):,} steps, {n_trades} trades")
    print(f"[run] Final equity: ${equity[-1]:,.2f} "
          f"(return: {(equity[-1]/equity[0]-1)*100:+.2f}%)")

    # ---- 6. Benchmarks ----
    # BTC Buy & Hold
    bm_prices = unseen_df_reset["close"].values.astype("float64")
    bm_offset = window_size
    bm_prices_aligned = bm_prices[bm_offset:bm_offset + len(equity)]
    if len(bm_prices_aligned) < len(equity):
        bm_prices_aligned = np.pad(
            bm_prices_aligned,
            (0, len(equity) - len(bm_prices_aligned)),
            mode="edge",
        )
    bm_equity = 10_000.0 * (bm_prices_aligned / bm_prices_aligned[0])

    # S&P 500 — use the DatetimeIndex of the unseen data
    sp500_equity = None
    if isinstance(unseen_df.index, pd.DatetimeIndex):
        oos_timestamps = unseen_df.index[bm_offset:bm_offset + len(equity)]
        sp500_equity = fetch_sp500_equity(oos_timestamps, 10_000.0)
    else:
        print("[warn] No DatetimeIndex available — skipping S&P 500 fetch")

    # ---- 7. Compute returns & metrics ----
    returns = np.diff(equity) / equity[:-1]
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    metrics = compute_metrics(
        equity=equity,
        benchmark_equity=bm_equity,
        trades=trades,
        periods_per_year=periods_per_year,
        initial_balance=10_000.0,
        sp500_equity=sp500_equity,
    )

    # ---- 8. Benchmark Comparison Table ----
    _bm_rets = np.diff(bm_equity) / bm_equity[:-1]
    _bm_rets = np.nan_to_num(_bm_rets, nan=0.0, posinf=0.0, neginf=0.0)
    _bm_ra = RiskAnalytics(_bm_rets, periods_per_year)

    _sp_ra = None
    if sp500_equity is not None and len(sp500_equity) > 1:
        _sp_rets = np.diff(sp500_equity) / sp500_equity[:-1]
        _sp_rets = np.nan_to_num(_sp_rets, nan=0.0, posinf=0.0, neginf=0.0)
        _sp_ra = RiskAnalytics(_sp_rets, periods_per_year)

    _ag_ra = RiskAnalytics(returns, periods_per_year)

    print(f"\n{'='*70}")
    print(f" BENCHMARK COMPARISON — Pure OOS (Jan 2025 → Mar 2026)")
    print(f"{'='*70}")
    print(f"  {'Metric':<25s} {'Agent':>12s} {'BTC B&H':>12s}", end="")
    if _sp_ra: print(f" {'S&P 500':>12s}", end="")
    print()
    print(f"  {'-'*25} {'-'*12} {'-'*12}", end="")
    if _sp_ra: print(f" {'-'*12}", end="")
    print()
    print(f"  {'Ann. Return':<25s} "
          f"{_ag_ra.annualized_return()*100:>11.2f}% "
          f"{_bm_ra.annualized_return()*100:>11.2f}%", end="")
    if _sp_ra: print(f" {_sp_ra.annualized_return()*100:>11.2f}%", end="")
    print()
    print(f"  {'Sharpe Ratio':<25s} "
          f"{_ag_ra.sharpe_ratio():>12.3f} "
          f"{_bm_ra.sharpe_ratio():>12.3f}", end="")
    if _sp_ra: print(f" {_sp_ra.sharpe_ratio():>12.3f}", end="")
    print()
    print(f"  {'Sortino Ratio':<25s} "
          f"{_ag_ra.sortino_ratio():>12.3f} "
          f"{_bm_ra.sortino_ratio():>12.3f}", end="")
    if _sp_ra: print(f" {_sp_ra.sortino_ratio():>12.3f}", end="")
    print()
    print(f"  {'Max Drawdown':<25s} "
          f"{_ag_ra.max_drawdown()*100:>11.2f}% "
          f"{_bm_ra.max_drawdown()*100:>11.2f}%", end="")
    if _sp_ra: print(f" {_sp_ra.max_drawdown()*100:>11.2f}%", end="")
    print()
    print(f"{'='*70}\n")

    # ---- 9. Academic Report ----
    print("[report] Generating Academic Report (thesis charts)...")
    report = AcademicReport("CNN+LSTM PPO × SPA (Pure OOS)")
    reports_dir = out_dir / "reports"
    report.generate(
        equity=equity,
        benchmark_equity=bm_equity,
        returns=returns,
        trades=trades,
        metrics=metrics,
        train_sharpe=None,   # No train Sharpe — this IS the final test
        periods_per_year=periods_per_year,
        output_dir=reports_dir,
        sp500_equity=sp500_equity,
    )

    # ---- 10. Institutional Tearsheet ----
    print("[report] Generating Institutional Tearsheet...")
    inst_results = run_institutional_eval(
        equity=equity,
        actions=actions,
        trades=trades,
        benchmark_prices=bm_prices_aligned,
        initial_balance=10_000.0,
        periods_per_year=periods_per_year,
        n_trades_count=n_trades,
        output_dir=out_dir,
        strategy_name="CNN+LSTM PPO × SPA (Pure OOS Walk-Forward)",
        sp500_equity=sp500_equity,
    )

    # ---- 11. Save raw results ----
    results = {
        "window": "2025-01-01 to 2026-03-15",
        "total_steps": len(equity),
        "n_trades": n_trades,
        "final_equity": float(equity[-1]),
        "metrics": {k: float(v) if v is not None else None
                    for k, v in metrics.items()},
    }
    results_path = out_dir / "unseen_backtest_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"[ok] Saved results: {results_path}")

    # ---- Final Summary ----
    print(f"\n{'='*70}")
    print(f" PURE OOS WALK-FORWARD BACKTEST COMPLETE")
    print(f"{'='*70}")
    print(f"  Window             : Jan 2025 → Mar 2026")
    print(f"  Agent Return       : {metrics['total_return']*100:+.2f}%")
    print(f"  Agent Ann. Return  : {metrics['annualized_return']*100:.2f}%")
    print(f"  Agent Sharpe       : {metrics['sharpe_ratio']:.3f}")
    print(f"  Agent Max Drawdown : {metrics['max_drawdown']*100:.2f}%")
    print(f"  BTC B&H Return     : {metrics['bm_total_return']*100:+.2f}%")
    if metrics.get("sp500_total_return") is not None:
        print(f"  S&P 500 Return     : {metrics['sp500_total_return']*100:+.2f}%")
    print(f"  —")
    winner = "AGENT ✅" if metrics["annualized_return"] > 0.12 else "BELOW TARGET ⚠️"
    print(f"  vs S&P 500 Target  : {winner} "
          f"(target >12%, got {metrics['annualized_return']*100:.2f}%)")
    print(f"{'='*70}")
    print(f"  Reports: {reports_dir}")
    print(f"  Tearsheet: {out_dir / 'institutional_tearsheet.pdf'}")
    print(f"{'='*70}\n")


# ====================================================================
# CLI
# ====================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Pure OOS Walk-Forward Backtest on Unseen Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--model", type=str, required=True,
        help="Path to trained model .zip (e.g. data/models/ppo_spa_btc_1h.zip)",
    )
    ap.add_argument(
        "--features-train", type=str, required=True,
        help="Path to ORIGINAL training features parquet (for norm stats)",
    )
    ap.add_argument(
        "--features-unseen", type=str, required=True,
        help="Path to UNSEEN features parquet (Jan 2025 → Mar 2026)",
    )
    ap.add_argument(
        "--out_dir", type=str, default="data/eval/unseen",
        help="Output directory for reports and results",
    )
    args = ap.parse_args()

    run_unseen_backtest(
        model_path=Path(args.model),
        train_features_path=Path(args.features_train),
        unseen_features_path=Path(args.features_unseen),
        out_dir=Path(args.out_dir),
    )
