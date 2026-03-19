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

from src.rl_env.crypto_env import CryptoTradingEnv
from src.eval.institutional_tearsheet import run_institutional_eval
from src.eval.academic_report import (
    compute_metrics, compute_train_sharpe, AcademicReport
)


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
        inactivity_steps=256,
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
    df_all = df_all[need_cols].dropna().reset_index(drop=True)

    n_train = int(len(df_all) * train_split)
    train_df = df_all.iloc[:n_train].copy()
    eval_df = df_all.iloc[n_train:].copy()
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

    # Benchmark (Buy & Hold)
    bm_prices = eval_df["close"].values.astype("float64")
    bm_offset = window_size
    bm_prices_aligned = bm_prices[bm_offset:bm_offset + len(equity)]
    if len(bm_prices_aligned) < len(equity):
        bm_prices_aligned = np.pad(bm_prices_aligned,
                                    (0, len(equity) - len(bm_prices_aligned)),
                                    mode="edge")
    bm_equity = 10_000.0 * (bm_prices_aligned / bm_prices_aligned[0])

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
        periods_per_year=periods_per_year, initial_balance=10_000.0)

    report = AcademicReport("CNN+LSTM PPO × SPA Day Trader")
    reports_dir = out_dir / "reports"
    report.generate(
        equity=equity, benchmark_equity=bm_equity, returns=returns,
        trades=trades, metrics=metrics, train_sharpe=train_sharpe,
        periods_per_year=periods_per_year, output_dir=reports_dir)

    # ==================================================================
    # 4. INSTITUTIONAL TEARSHEET
    # ==================================================================
    print("[step 4/4] Generating Institutional Tearsheet (PDF)...")
    inst_results = run_institutional_eval(
        equity=equity, actions=actions, trades=trades,
        benchmark_prices=bm_prices_aligned, initial_balance=10_000.0,
        periods_per_year=periods_per_year, n_trades_count=n_trades,
        output_dir=out_dir,
        strategy_name="CNN+LSTM PPO × SPA Day Trading Strategy")

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
