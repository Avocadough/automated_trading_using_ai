# src/eval/eval_ppo_spa.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

# ---- import env ----
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.rl_env.crypto_env import CryptoTradingEnv


def load_meta(features_path: Path):
    meta_path = Path(str(features_path).replace(".parquet", "_meta.json"))
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Meta not found: {meta_path}\n"
            f"ต้องมีไฟล์ meta อยู่คู่กับ parquet (ชื่อเดียวกัน เติม _meta.json)"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    features = meta.get("features")
    window_size = int(meta.get("window_size", 64))
    train_split = float(meta.get("train_split", 0.8))
    periods_per_year = int(meta.get("periods_per_year", 35040))
    if not features or not isinstance(features, list):
        raise ValueError("Invalid meta: 'features' ต้องเป็น list และต้องไม่ว่าง")
    return features, window_size, train_split, periods_per_year


def build_env(df: pd.DataFrame, features: list[str], window_size: int):
    return CryptoTradingEnv(
        df=df,
        features=features,
        window_size=window_size,
        initial_balance=10_000.0,
        taker_fee=0.0005,        # ตรงกับ train env (Binance Futures)
        position_limit=0.5,
        slippage_bps=0.5,        # ตรงกับ train env
        reward_scale=100.0,
        normalize=True,
        action_mode="discrete",
        deadband_frac=0.05,      # ตรงกับ train env
        min_hold_steps=0,        # ปิด: ดู policy จริง
        cooldown_steps=0,
        # --- keep shaping OFF in eval ---
        flat_penalty_bps=0.0,
        inactivity_steps=256,
        inactivity_penalty_bps=0.0,
        turnover_reward_coeff=0.0,
        trade_threshold=0.01,
    )



def compute_max_drawdown(equity: np.ndarray) -> float:
    """return max drawdown as negative fraction, e.g. -0.1234 = -12.34%"""
    curve = pd.Series(equity, dtype=float)
    peak = curve.cummax()
    dd = (curve - peak) / peak
    return float(dd.min()) if len(dd) else 0.0


def run_eval(model_path: Path, features_path: Path, out_csv: Path):
    print(f"[info] Loading model: {model_path}")
    model = PPO.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")

    print(f"[info] Loading features: {features_path}")
    # ✅ enforce meta to match training setup
    features, window_size, train_split, periods_per_year = load_meta(features_path)
    df_all = pd.read_parquet(features_path)
    need_cols = ["close"] + features
    missing = [c for c in need_cols if c not in df_all.columns]
    if missing:
        raise ValueError(f"Columns missing in parquet: {missing}")

    df_all = df_all[need_cols].dropna().reset_index(drop=True)

    n_train = int(len(df_all) * train_split)
    eval_df = df_all.iloc[n_train:].copy()
    print(f"[info] Eval rows: {len(eval_df):,} | window_size={window_size} | n_features={len(features)}")

    env = build_env(eval_df, features, window_size)
    obs, _ = env.reset()

    # --- run episode ---
    equities: list[float] = []
    actions_taken: list[int] = []
    pos_frac_hist: list[float] = []

    # trade tracking
    trades = []  # list of dicts
    in_trade = False
    entry_equity = 0.0
    entry_step = 0
    entry_side = 0  # -1 short, +1 long
    prev_side = 0

    done = False
    step_idx = 0
    while not done:
        # SB3 expects batch dimension
        obs_b = np.expand_dims(obs, axis=0)
        action, _ = model.predict(obs_b, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        eq = float(info["equity"])
        equities.append(eq)
        actions_taken.append(int(action) if np.isscalar(action) else int(action[0]))
        # side from position_frac sign
        side = int(np.sign(info.get("position_frac", 0.0)))
        pos_frac_hist.append(float(info.get("position_frac", 0.0)))

        # --- trade state machine ---
        # open: 0 -> nonzero
        if prev_side == 0 and side != 0:
            in_trade = True
            entry_equity = eq
            entry_step = step_idx
            entry_side = side

        # flip: nonzero -> opposite sign === close old + open new (at same step)
        if prev_side != 0 and side != 0 and side != prev_side:
            # close old
            if in_trade:
                trades.append({
                    "entry_step": entry_step,
                    "exit_step": step_idx,
                    "side": entry_side,
                    "entry_equity": entry_equity,
                    "exit_equity": eq,
                    "return": (eq / entry_equity) - 1.0,
                    "duration_steps": step_idx - entry_step
                })
            # open new
            in_trade = True
            entry_equity = eq
            entry_step = step_idx
            entry_side = side

        # close: nonzero -> 0
        if prev_side != 0 and side == 0 and in_trade:
            trades.append({
                "entry_step": entry_step,
                "exit_step": step_idx,
                "side": entry_side,
                "entry_equity": entry_equity,
                "exit_equity": eq,
                "return": (eq / entry_equity) - 1.0,
                "duration_steps": step_idx - entry_step
            })
            in_trade = False
            entry_equity = 0.0
            entry_step = 0
            entry_side = 0

        prev_side = side
        step_idx += 1

    # if episode ends with open trade, close it at last equity
    if in_trade and len(equities) > 0:
        last_idx = step_idx - 1
        last_eq = equities[-1]
        trades.append({
            "entry_step": entry_step,
            "exit_step": last_idx,
            "side": entry_side,
            "entry_equity": entry_equity,
            "exit_equity": last_eq,
            "return": (last_eq / entry_equity) - 1.0,
            "duration_steps": last_idx - entry_step
        })

    # --- metrics ---
    equity = np.asarray(equities, dtype=float)
    rets = pd.Series(equity).pct_change().fillna(0.0).values
    total_return = (equity[-1] / equity[0]) - 1.0 if len(equity) > 1 else 0.0
    ann_ret = (1.0 + rets.mean()) ** periods_per_year - 1.0 if len(rets) > 0 else 0.0
    ann_vol = rets.std() * np.sqrt(periods_per_year) if len(rets) > 0 else 0.0
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else np.nan
    max_dd = compute_max_drawdown(equity)

    # trade stats
    n_trades = len(trades)
    wins = [t for t in trades if t["return"] > 1e-12]
    losses = [t for t in trades if t["return"] < -1e-12]
    breakevens = [t for t in trades if -1e-12 <= t["return"] <= 1e-12]
    
    # Winrate only out of trades that actually won or lost (exclude exact 0)
    resolved_trades = len(wins) + len(losses)
    winrate = (len(wins) / resolved_trades) if resolved_trades > 0 else 0.0
    avg_win = np.mean([t["return"] for t in wins]) if wins else 0.0
    avg_loss = np.mean([t["return"] for t in losses]) if losses else 0.0
    avg_rr = (avg_win / abs(avg_loss)) if losses and abs(avg_loss) > 1e-12 else np.nan
    avg_dur_steps = np.mean([t["duration_steps"] for t in trades]) if trades else 0.0

    # convert average duration to hours (approx) using periods_per_year
    # minutes per step = 525_600 minutes / periods_per_year
    minutes_per_step = 525_600 / float(periods_per_year)
    avg_dur_hours = (avg_dur_steps * minutes_per_step) / 60.0

    # save equity/action curve
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "equity": equity,
        "action": actions_taken,
        "position_frac": pos_frac_hist
    }).to_csv(out_csv, index=False)

    # also save trades log
    trades_csv = out_csv.with_name(out_csv.stem + "_trades.csv")
    if trades:
        pd.DataFrame(trades).to_csv(trades_csv, index=False)
    else:
        pd.DataFrame(columns=[
            "entry_step","exit_step","side","entry_equity","exit_equity","return","duration_steps"
        ]).to_csv(trades_csv, index=False)

    # --- print summary ---
    n_short = actions_taken.count(0)
    n_flat  = actions_taken.count(1)
    n_long  = actions_taken.count(2)
    total_actions = len(actions_taken)

    print("\n=== EVAL SUMMARY ===")
    print(f"Total return: {total_return*100:.2f}%")
    print(f"Sharpe      : {sharpe:.2f}")
    print(f"Max DD      : {max_dd*100:.2f}%")

    print(f"\n--- Action Distribution (ดูว่า model เรียนรู้อะไร) ---")
    print(f"Short (0)   : {n_short:,}  ({n_short/max(1,total_actions)*100:.1f}%)")
    print(f"Flat  (1)   : {n_flat:,}  ({n_flat/max(1,total_actions)*100:.1f}%)")
    print(f"Long  (2)   : {n_long:,}  ({n_long/max(1,total_actions)*100:.1f}%)")

    print(f"\n--- Trade Stats ---")
    print(f"Trades      : {n_trades}")
    print(f"  Wins      : {len(wins)}  ({winrate*100:.2f}% of resolved)")
    print(f"  Losses    : {len(losses)}")
    print(f"  Breakeven : {len(breakevens)}")
    print(f"Avg Win     : {avg_win*100:.3f}%" if wins else "Avg Win     : n/a")
    print(f"Avg Loss    : {avg_loss*100:.3f}%" if losses else "Avg Loss    : n/a")
    print(f"Avg R/R     : {avg_rr:.2f}" if not np.isnan(avg_rr) else "Avg R/R     : n/a")
    print(f"Avg duration: {avg_dur_steps:.1f} steps  (~{avg_dur_hours:.2f} hours)")

    print(f"\n[ok] Saved equity curve to {out_csv}")
    print(f"[ok] Saved trade log    to {trades_csv}")


def main(
    model_path: Path = Path("data/models/ppo_spa_btc_1h.zip"),
    features_path: Path = Path("data/features/btc_1h_spa.parquet"),
    out_csv: Path = Path("data/eval/ppo_spa_btc_1h_eval.csv"),
):
    run_eval(model_path, features_path, out_csv)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate PPO SPA model on held-out data (uses *_meta.json)")
    ap.add_argument("--model", type=str, default="data/models/ppo_spa_btc_1h.zip")
    ap.add_argument("--train_split", type=float, default=0.8)
    ap.add_argument("--features", type=str, default="data/features/btc_1h_spa.parquet")
    ap.add_argument("--out_csv", type=str, default="data/eval/ppo_spa_btc_1h_eval.csv")
    args = ap.parse_args()

    main(
        model_path=Path(args.model),
        features_path=Path(args.features),
        out_csv=Path(args.out_csv),
    )
