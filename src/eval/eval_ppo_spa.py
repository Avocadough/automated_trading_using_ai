# src/eval/eval_ppo_spa.py
"""
PPO Evaluation Pipeline — Academic Report + Institutional Tearsheet.

Runs the trained model on both Training and OOS data, computes
Train vs Test Sharpe for the overfitting sanity check, and generates:
  1. Academic Report   → data/eval/reports/ (PNG charts + PDF)
  2. Institutional PDF → data/eval/institutional_tearsheet.pdf
  3. Raw data          → data/eval/equity_curve.csv, trades.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False
    print("[warn] yfinance not installed — S&P 500 benchmark will be skipped.")
    print("       Install with:  pip install yfinance")

from src.rl_env.crypto_env import CryptoTradingEnv
from src.eval.institutional_tearsheet import run_institutional_eval
from src.eval.academic_report import (
    compute_metrics, compute_train_sharpe, AcademicReport
)


def fetch_sp500_equity(
    timestamps: pd.DatetimeIndex | pd.Index,
    initial_balance: float = 10_000.0,
) -> np.ndarray | None:
    """
    Download ^GSPC daily closes via yfinance and forward-fill missing values
    (weekends / US holidays) so that every crypto 24/7 hourly bar has a price.

    Returns an equity array starting at initial_balance, or None on failure.
    """
    if not _YF_AVAILABLE:
        return None

    # --- Determine the date range of the OOS window ---
    if isinstance(timestamps, pd.DatetimeIndex) and len(timestamps) > 0:
        start_dt = timestamps[0].strftime("%Y-%m-%d")
        end_dt   = (timestamps[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        print("[warn] eval_df has no DatetimeIndex — S&P 500 fetch skipped.")
        return None

    try:
        print(f"[info] Fetching S&P 500 (^GSPC) from {start_dt} to {end_dt} ...")
        ticker = yf.Ticker("^GSPC")
        sp_df  = ticker.history(start=start_dt, end=end_dt, interval="1d")
        if sp_df.empty:
            print("[warn] yfinance returned empty dataframe for ^GSPC.")
            return None

        # Keep only the Close column; use UTC-aware index
        sp_close = sp_df["Close"].copy()
        sp_close.index = sp_close.index.tz_convert("UTC") if sp_close.index.tz else sp_close.index.tz_localize("UTC")

        # Build a target index that matches the OOS bars (UTC)
        if timestamps.tz is None:
            target_idx = timestamps.tz_localize("UTC")
        else:
            target_idx = timestamps.tz_convert("UTC")

        # Reindex to hourly, forward-fill weekends/holidays
        sp_hourly = sp_close.reindex(target_idx, method="ffill")

        # If the very first bar is NaN (S&P trading hasn't started yet), backfill
        sp_hourly = sp_hourly.bfill()

        if sp_hourly.isna().all():
            print("[warn] S&P 500 data could not be aligned to OOS timestamps.")
            return None

        # Normalise to an equity curve starting at initial_balance
        sp_vals = sp_hourly.values.astype("float64")
        first_valid = sp_vals[~np.isnan(sp_vals)][0]
        sp_equity = initial_balance * (sp_vals / first_valid)
        print(f"[info] S&P 500 equity curve: {len(sp_equity):,} bars, "
              f"return={(sp_equity[-1]/sp_equity[0]-1)*100:+.2f}%")
        return sp_equity

    except Exception as exc:
        print(f"[warn] S&P 500 fetch failed: {exc}")
        return None


def load_meta(features_path: Path):
    meta_path = Path(str(features_path).replace(".parquet", "_meta.json"))
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    features = meta.get("features")
    window_size = int(meta.get("window_size", 64))
    train_split = float(meta.get("train_split", 0.8))
    periods_per_year = int(meta.get("periods_per_year", 8760))
    if not features or not isinstance(features, list):
        raise ValueError("Invalid meta: 'features' must be a non-empty list")
    return features, window_size, train_split, periods_per_year


def build_env(df: pd.DataFrame, features: list[str], window_size: int,
              norm_mu=None, norm_std=None):
    """Build eval env — all shaping disabled for clean measurement."""
    return CryptoTradingEnv(
        df=df,
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
        inactivity_steps=72,
        inactivity_penalty_bps=0.0,
        turnover_reward_coeff=0.0,
        trade_threshold=0.01,
        drawdown_penalty_coeff=0.0,
        liquidation_threshold=0.5,
        sharpe_window=48,
        sharpe_eta=0.01,
    )


def _run_episode(model, env):
    """Run one complete episode. Returns equity, actions, trades."""
    obs, _ = env.reset()
    equities, actions_taken, pos_frac_hist = [], [], []
    trades = []
    in_trade = False
    entry_equity = entry_step = entry_side = prev_side = 0
    done = False
    step_idx = 0
    n_trades_total = 0

    while not done:
        obs_b = np.expand_dims(obs, axis=0)
        action, _ = model.predict(obs_b, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        eq = float(info["equity"])
        equities.append(eq)
        act = int(action) if np.isscalar(action) else int(action[0])
        actions_taken.append(act)
        side = int(np.sign(info.get("position_frac", 0.0)))
        pos_frac_hist.append(float(info.get("position_frac", 0.0)))
        n_trades_total = info.get("n_trades", n_trades_total)

        # Trade state machine
        if prev_side == 0 and side != 0:
            in_trade, entry_equity, entry_step, entry_side = True, eq, step_idx, side
        if prev_side != 0 and side != 0 and side != prev_side:
            if in_trade:
                trades.append(dict(
                    entry_step=entry_step, exit_step=step_idx, side=entry_side,
                    entry_equity=entry_equity, exit_equity=eq,
                    **{"return": (eq / entry_equity) - 1.0},
                    duration_steps=step_idx - entry_step))
            in_trade, entry_equity, entry_step, entry_side = True, eq, step_idx, side
        if prev_side != 0 and side == 0 and in_trade:
            trades.append(dict(
                entry_step=entry_step, exit_step=step_idx, side=entry_side,
                entry_equity=entry_equity, exit_equity=eq,
                **{"return": (eq / entry_equity) - 1.0},
                duration_steps=step_idx - entry_step))
            in_trade = False
        prev_side = side
        step_idx += 1

    if in_trade and equities:
        trades.append(dict(
            entry_step=entry_step, exit_step=step_idx - 1, side=entry_side,
            entry_equity=entry_equity, exit_equity=equities[-1],
            **{"return": (equities[-1] / entry_equity) - 1.0},
            duration_steps=step_idx - 1 - entry_step))

    return (np.array(equities), np.array(actions_taken),
            np.array(pos_frac_hist), trades, n_trades_total)


def run_eval(model_path: Path, features_path: Path, out_dir: Path):
    print(f"[info] Loading model: {model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(model_path, device=device)

    print(f"[info] Loading features: {features_path}")
    features, window_size, train_split, periods_per_year = load_meta(features_path)
    df_all = pd.read_parquet(features_path)
    need_cols = ["close"] + features
    missing = [c for c in need_cols if c not in df_all.columns]
    if missing:
        raise ValueError(f"Columns missing: {missing}")
    # Preserve the DatetimeIndex BEFORE resetting so S&P 500 fetch can use
    # real timestamps when forward-filling to 24/7 crypto hourly bars.
    raw_datetime_index = df_all.index if isinstance(
        df_all.index, pd.DatetimeIndex) else None
    df_all = df_all[need_cols].dropna().reset_index(drop=True)

    n_train = int(len(df_all) * train_split)
    train_df = df_all.iloc[:n_train].copy()
    eval_df = df_all.iloc[n_train:].copy()

    # Re-assign the datetime index to eval_df so fetch_sp500_equity() works
    if raw_datetime_index is not None and len(raw_datetime_index) == len(df_all):
        eval_df.index = raw_datetime_index[n_train:n_train + len(eval_df)]
    print(f"[info] Train: {len(train_df):,} | Test (OOS): {len(eval_df):,}")

    # ==================================================================
    # 1. RUN ON TRAINING DATA (for overfitting check only)
    # ==================================================================
    print("\n[step 1/4] Running model on TRAINING data (overfitting check)...")
    train_env = build_env(train_df, features, window_size)
    train_equity, _, _, _, _ = _run_episode(model, train_env)
    train_sharpe = compute_train_sharpe(train_equity, periods_per_year)
    print(f"  Train Sharpe: {train_sharpe:.3f}")

    # ==================================================================
    # 2. RUN ON TEST (OOS) DATA
    # ==================================================================
    print("[step 2/4] Running model on TEST (OOS) data...")
    eval_env = build_env(eval_df, features, window_size)
    equity, actions, pos_frac, trades, n_trades = _run_episode(model, eval_env)

    # Benchmark (Buy & Hold BTC)
    bm_prices = eval_df["close"].values.astype("float64")
    bm_offset = window_size
    bm_prices_aligned = bm_prices[bm_offset:bm_offset + len(equity)]
    if len(bm_prices_aligned) < len(equity):
        bm_prices_aligned = np.pad(bm_prices_aligned,
                                    (0, len(equity) - len(bm_prices_aligned)),
                                    mode="edge")
    bm_equity = 10_000.0 * (bm_prices_aligned / bm_prices_aligned[0])

    # ---------------------------------------------------------------
    # S&P 500 Benchmark — fetched automatically via yfinance
    # Aligns ^GSPC daily closes to the crypto 24/7 hourly OOS index
    # by forward-filling weekends and US market holidays.
    # ---------------------------------------------------------------
    sp500_equity: np.ndarray | None = None
    if isinstance(eval_df.index, pd.DatetimeIndex):
        ts_index = eval_df.index[bm_offset:bm_offset + len(equity)]
        sp500_equity = fetch_sp500_equity(ts_index, initial_balance=10_000.0)
        if sp500_equity is not None and len(sp500_equity) != len(equity):
            # Trim/pad to match exactly
            sp500_equity = sp500_equity[:len(equity)]
            if len(sp500_equity) < len(equity):
                sp500_equity = np.pad(sp500_equity,
                                      (0, len(equity) - len(sp500_equity)),
                                      mode="edge")
    else:
        print("[info] eval_df has no DatetimeIndex — S&P 500 benchmark skipped.")

    # Save raw data
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"equity": equity, "action": actions,
                   "position_frac": pos_frac}).to_csv(
        out_dir / "equity_curve.csv", index=False)
    if trades:
        pd.DataFrame(trades).to_csv(out_dir / "trades.csv", index=False)

    # ==================================================================
    # 3. ACADEMIC REPORT
    # ==================================================================
    print("[step 3/4] Generating Academic Report (thesis charts)...")
    returns = np.diff(equity) / equity[:-1]
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    metrics = compute_metrics(
        equity=equity, benchmark_equity=bm_equity, trades=trades,
        periods_per_year=periods_per_year, initial_balance=10_000.0,
        sp500_equity=sp500_equity)

    # ---- Print Benchmark Comparison Table ----
    from src.eval.institutional_tearsheet import RiskAnalytics
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
    print(f" BENCHMARK COMPARISON (Out-of-Sample)")
    print(f"{'='*70}")
    print(f"  {'Metric':<25s} {'Agent':>12s} {'BTC B&H':>12s}", end="")
    if _sp_ra: print(f" {'S&P 500':>12s}", end="")
    print()
    print(f"  {'-'*25} {'-'*12} {'-'*12}", end="")
    if _sp_ra: print(f" {'-'*12}", end="")
    print()
    print(f"  {'Ann. Return':<25s} {_ag_ra.annualized_return()*100:>11.2f}% {_bm_ra.annualized_return()*100:>11.2f}%", end="")
    if _sp_ra: print(f" {_sp_ra.annualized_return()*100:>11.2f}%", end="")
    print()
    print(f"  {'Sharpe Ratio':<25s} {_ag_ra.sharpe_ratio():>12.3f} {_bm_ra.sharpe_ratio():>12.3f}", end="")
    if _sp_ra: print(f" {_sp_ra.sharpe_ratio():>12.3f}", end="")
    print()
    print(f"  {'Sortino Ratio':<25s} {_ag_ra.sortino_ratio():>12.3f} {_bm_ra.sortino_ratio():>12.3f}", end="")
    if _sp_ra: print(f" {_sp_ra.sortino_ratio():>12.3f}", end="")
    print()
    print(f"  {'Max Drawdown':<25s} {_ag_ra.max_drawdown()*100:>11.2f}% {_bm_ra.max_drawdown()*100:>11.2f}%", end="")
    if _sp_ra: print(f" {_sp_ra.max_drawdown()*100:>11.2f}%", end="")
    print()
    print(f"{'='*70}\n")

    report = AcademicReport("CNN+LSTM PPO × SPA Day Trader")
    reports_dir = out_dir / "reports"
    report.generate(
        equity=equity, benchmark_equity=bm_equity, returns=returns,
        trades=trades, metrics=metrics, train_sharpe=train_sharpe,
        periods_per_year=periods_per_year, output_dir=reports_dir,
        sp500_equity=sp500_equity)

    # ==================================================================
    # 4. INSTITUTIONAL TEARSHEET
    # ==================================================================
    print("[step 4/4] Generating Institutional Tearsheet (PDF)...")
    inst_results = run_institutional_eval(
        equity=equity, actions=actions, trades=trades,
        benchmark_prices=bm_prices_aligned, initial_balance=10_000.0,
        periods_per_year=periods_per_year, n_trades_count=n_trades,
        output_dir=out_dir,
        strategy_name="CNN+LSTM PPO × SPA Day Trading Strategy",
        sp500_equity=sp500_equity)

    # Save combined results
    all_results = {
        "academic_metrics": metrics,
        "train_sharpe": train_sharpe,
        "institutional": {
            k: v for k, v in inst_results.items() if k != "pdf_path"
        },
    }
    (out_dir / "eval_results.json").write_text(
        json.dumps(all_results, indent=2, default=str))

    print(f"\n[ok] All outputs saved to: {out_dir}/")
    print(f"     reports/academic_report.pdf  (thesis charts)")
    print(f"     institutional_tearsheet.pdf  (hedge fund metrics)")
    print(f"     eval_results.json            (all metrics)")


def main(
    model_path: Path = Path("data/models/ppo_spa_btc_1h.zip"),
    features_path: Path = Path("data/features/btc_1h_spa.parquet"),
    out_dir: Path = Path("data/eval"),
):
    run_eval(model_path, features_path, out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Evaluate PPO — Academic Report + Institutional Tearsheet")
    ap.add_argument("--model", type=str,
                    default="data/models/ppo_spa_btc_1h.zip")
    ap.add_argument("--features", type=str,
                    default="data/features/btc_1h_spa.parquet")
    ap.add_argument("--out_dir", type=str, default="data/eval")
    args = ap.parse_args()

    main(
        model_path=Path(args.model),
        features_path=Path(args.features),
        out_dir=Path(args.out_dir),
    )
