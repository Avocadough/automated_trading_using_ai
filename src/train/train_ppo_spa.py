# src/train/train_ppo_spa.py
from __future__ import annotations

import os
import sys
import json
from pathlib import Path
import argparse
import random
import inspect

import numpy as np
import pandas as pd
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
from typing import Callable

def linear_schedule(initial_value: float, min_value: float = 0.0) -> Callable[[float], float]:
    """
    Linear learning rate schedule with a minimum floor.
    Decays from `initial_value` → `min_value` as training progresses.
    When min_value > 0, the network never fully freezes — critical for
    preventing late-stage "brain death" where the agent can't adapt.

    :param initial_value: Starting learning rate.
    :param min_value: Minimum learning rate floor (default 0.0 for backward compat).
    :return: callable schedule for SB3.
    """
    def func(progress_remaining: float) -> float:
        # progress_remaining goes from 1.0 → 0.0 during training
        return max(min_value, progress_remaining * initial_value)

    return func

# project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.rl_env.crypto_env import CryptoTradingEnv
from src.models.custom_policy import get_policy_kwargs


def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def load_meta_or_infer(features_path: Path):
    meta_path = Path(str(features_path).replace(".parquet", "_meta.json"))
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        features = meta.get("features")
        window_size = int(meta.get("window_size", 64))
        if not features or not isinstance(features, list):
            raise ValueError("Invalid meta: 'features' must be a non-empty list.")
        return features, window_size, True
    df_tmp = pd.read_parquet(features_path, columns=None)
    all_cols = list(df_tmp.columns)
    if "close" not in all_cols:
        raise ValueError("Parquet must contain 'close' column.")
    inferred = [c for c in all_cols if c != "close"]
    if not inferred:
        raise ValueError("No feature columns found (only 'close' present).")
    return inferred, 64, False


def build_env(
    df: pd.DataFrame,
    features: list[str],
    window_size: int,
    *,
    norm_mu=None,
    norm_std=None,
    initial_balance: float = 10_000.0,
    taker_fee: float = 0.0005,
    position_limit: float = 0.30,
    slippage_bps: float = 0.5,
    reward_scale: float = 10.0,
    normalize: bool = True,
    action_mode: str = "discrete",
    flat_penalty_bps: float = 0.0,
    inactivity_steps: int = 256,
    inactivity_penalty_bps: float = 0.0,
    turnover_reward_coeff: float = 0.0,
    trade_threshold: float = 0.01,
    deadband_frac: float = 0.02,
    min_hold_steps: int = 0,
    cooldown_steps: int = 0,
    reward_clip: float = 5.0,
    drawdown_penalty_coeff: float = 0.5,
    liquidation_threshold: float = 0.5,
    sharpe_window: int = 48,
    sharpe_eta: float = 0.01,
    n_envs: int = 1,
):
    def _make():
        common_kwargs = dict(
            df=df,
            features=features,
            window_size=window_size,
            norm_mu=norm_mu,
            norm_std=norm_std,
            initial_balance=initial_balance,
            taker_fee=taker_fee,
            position_limit=position_limit,
            slippage_bps=slippage_bps,
            reward_scale=reward_scale,
            normalize=normalize,
            action_mode=action_mode,
            flat_penalty_bps=flat_penalty_bps,
            inactivity_steps=inactivity_steps,
            inactivity_penalty_bps=inactivity_penalty_bps,
            turnover_reward_coeff=turnover_reward_coeff,
            trade_threshold=trade_threshold,
            deadband_frac=deadband_frac,
            min_hold_steps=min_hold_steps,
            cooldown_steps=cooldown_steps,
            reward_clip=reward_clip,
        )
        env_sig = inspect.signature(CryptoTradingEnv.__init__).parameters
        allowed = {k: v for k, v in common_kwargs.items() if k in env_sig}
        return CryptoTradingEnv(**allowed)

    if n_envs > 1:
        # ใช้ 'fork' บน Linux เพื่อแก้ปัญหา cv2 ใน Singularity, แต่ Windows ไม่รองรับ fork ต้องใช้ 'spawn'
        import os
        sm = "spawn" if os.name == "nt" else "fork"
        env = SubprocVecEnv([_make for _ in range(n_envs)], start_method=sm)
    else:
        env = DummyVecEnv([_make])
    env = VecMonitor(env)
    return env


def main(
    features_path: Path,
    output_prefix: Path,
    timesteps: int = 2_600_000,
    seed: int = 42,
    eval_every_steps: int = 50_000,
    train_split: float = 0.8,
    device_arg: str | None = None,
    n_envs: int = 8,
    # ---- shaping knobs (all zeroed: agent learns raw PnL, no action masking) ----
    flat_penalty_bps: float = 0.0,
    inactivity_steps: int = 256,
    inactivity_penalty_bps: float = 0.0,   # FIX: was 1.0 — masked exploration
    turnover_reward_coeff: float = 0.0,
    trade_threshold: float = 0.0,           # FIX: was 0.01 — deadband on actions
    deadband_frac: float = 0.0,             # FIX: was 0.05 — deadband on actions
    min_hold_steps: int = 0,
    cooldown_steps: int = 0,                # FIX: was 3 — forced inaction
):
    set_global_seeds(seed)

    print(f"[info] Loading features from: {features_path}")
    features, window_size, from_meta = load_meta_or_infer(features_path)
    print(f"[info] Features: {len(features)} | window_size={window_size} | from_meta={from_meta}")

    # Update train_split in meta
    meta_path = Path(str(features_path).replace(".parquet", "_meta.json"))
    if meta_path.exists():
        meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
        meta_data["train_split"] = train_split
        meta_path.write_text(json.dumps(meta_data, indent=2), encoding="utf-8")

    df_all = pd.read_parquet(features_path)
    needed = ["close"] + features
    missing = [c for c in needed if c not in df_all.columns]
    if missing:
        raise ValueError(f"Missing columns in features parquet: {missing}")

    df_all = df_all[needed].dropna().reset_index(drop=True)

    if len(df_all) < window_size + 100:
        raise ValueError(
            f"Dataset too small ({len(df_all)}) for window_size={window_size}. "
            f"Need at least {window_size+100} rows."
        )

    # Split
    n_train = max(window_size + 1, int(len(df_all) * train_split))
    n_train = min(n_train, len(df_all) - max(window_size + 1, 100))
    train_df = df_all.iloc[:n_train].copy()
    eval_df  = df_all.iloc[n_train:].copy()

    if len(eval_df) < window_size + 1:
        shift = (window_size + 1) - len(eval_df)
        n_train = max(window_size + 1, n_train - shift)
        train_df = df_all.iloc[:n_train].copy()
        eval_df  = df_all.iloc[n_train:].copy()

    print(f"[info] Dataset: total={len(df_all):,} | train={len(train_df):,} | eval={len(eval_df):,}")

    # ====================================================================
    # Compute normalization stats from TRAINING DATA ONLY
    # Pass to eval env to prevent data leak
    # ====================================================================
    feat_df_train = train_df[features].astype("float64")
    norm_mu  = feat_df_train.mean()
    norm_std = feat_df_train.std().replace(0, 1.0)
    print(f"[info] Computed norm stats from training data ({len(train_df):,} rows)")

    # Common env kwargs (shared between train and eval, except shaping)
    common_env_kwargs = dict(
        initial_balance=10_000.0,
        taker_fee=0.0005,
        position_limit=0.30,
        slippage_bps=0.5,
        reward_scale=10.0,
        reward_clip=5.0,
        normalize=True,
        action_mode="discrete",
        trade_threshold=trade_threshold,
        deadband_frac=deadband_frac,
        liquidation_threshold=0.5,
        sharpe_window=48,
        sharpe_eta=0.01,
    )

    # Train env: uses shaping, computes its own norm stats (self-contained)
    train_env = build_env(
        train_df, features, window_size,
        n_envs=n_envs,
        norm_mu=None, norm_std=None,
        **common_env_kwargs,
        flat_penalty_bps=flat_penalty_bps,
        inactivity_steps=inactivity_steps,
        inactivity_penalty_bps=inactivity_penalty_bps,
        turnover_reward_coeff=turnover_reward_coeff,
        min_hold_steps=min_hold_steps,
        cooldown_steps=cooldown_steps,
        drawdown_penalty_coeff=0.5,   # DD penalty ON in training
    )

    # Eval env: receives training norm stats → no future data leak
    eval_env = build_env(
        eval_df, features, window_size,
        norm_mu=norm_mu, norm_std=norm_std,
        **common_env_kwargs,
        flat_penalty_bps=0.0,
        inactivity_steps=inactivity_steps,
        inactivity_penalty_bps=0.0,
        turnover_reward_coeff=0.0,
        min_hold_steps=0,
        cooldown_steps=0,
        drawdown_penalty_coeff=0.0,   # DD penalty OFF in eval (pure equity)
    )

    # Device
    if device_arg is None or device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg
    print(f"[info] Device: {device}")

    # ====================================================================
    # PPO Model — Sequence-Aware Architecture (CNN+LSTM)
    # ====================================================================
    # Custom feature extractor: Conv1D → LayerNorm → LSTM → Linear
    # Replaces flat MLP which destroys temporal structure
    custom_policy_kwargs = get_policy_kwargs(
        features_dim=128,    # LSTM output → flat 128-dim feature vector
        pi_layers=[128],     # Policy head: 128 → action logits
        vf_layers=[128],     # Value head:  128 → scalar value
    )

    model = PPO(
        "MlpPolicy",          # SB3 base — custom extractor overrides feature processing
        train_env,
        verbose=1,
        device=device,
        seed=seed,
        tensorboard_log="./ppo_logs_spa/",
        # --- On-policy buffer ---
        # FIX: n_steps 2048→4096 — LSTM needs longer horizon for BPTT
        n_steps=4096,
        # FIX: batch_size 256→512 — reduce gradient variance for recurrent updates
        batch_size=512,        # 4096/512 = 8 mini-batches per update
        n_epochs=5,
        # --- Learning rate (linear decay 1e-4 → 1e-5, never hits zero) ---
        # FIX: LR decaying to 0.0 caused "brain freeze" — weights locked and
        # the agent couldn't adapt in late training. Floor of 1e-5 keeps the
        # network plastic enough for late-stage fine-tuning.
        learning_rate=linear_schedule(1e-4, min_value=1e-5),
        # --- Discount & GAE ---
        gamma=0.995,           # ~200 steps lookahead = ~8 days @ 1H
        gae_lambda=0.95,
        # --- Policy gradient ---
        # FIX: ent_coef 0.001 → 0.01 — "Rational Cowardice" fix.
        # At 0.001 the agent immediately collapsed to 100% Flat (action=1)
        # to avoid trading fees → 0% return. At 0.01 there's enough entropy
        # pressure to maintain exploration, but not so much that the policy
        # stays random (which happened at 0.03). This is the sweet spot.
        ent_coef=0.01,
        clip_range=0.20,
        vf_coef=0.5,
        # FIX: max_grad_norm 0.3 → 0.5 — standard LSTM gradient clipping value;
        # 0.3 was too tight and starved recurrent weight updates.
        max_grad_norm=0.5,
        # --- CNN+LSTM policy ---
        policy_kwargs=custom_policy_kwargs,
    )

    eval_dir = Path("data/models/_eval_spa")
    eval_dir.mkdir(parents=True, exist_ok=True)

    eval_every_steps = max(1, min(eval_every_steps, max(1, timesteps // 2)))
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(eval_dir),
        log_path=str(eval_dir),
        eval_freq=eval_every_steps,
        n_eval_episodes=1,
        deterministic=True,
        render=False,
    )

    print(f"[info] Training for {timesteps:,} timesteps... (eval every {eval_every_steps:,} steps)")
    model.learn(total_timesteps=timesteps, callback=eval_cb)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_prefix)
    print(f"[ok] Saved final model → {output_prefix}.zip")

    best_model_files = sorted(eval_dir.glob("best_model.zip"))
    if best_model_files:
        print(f"[ok] Best model (EvalCallback): {best_model_files[-1]}")
    else:
        print("[info] No separate best model produced by EvalCallback.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train PPO agent using SPA-based features")
    ap.add_argument("--features",         type=str, default="data/features/btc_1h_spa.parquet")
    ap.add_argument("--output",           type=str, default="data/models/ppo_spa_btc_1h")
    ap.add_argument("--timesteps",        type=int, default=5_000_000)
    ap.add_argument("--seed",             type=int, default=42)
    ap.add_argument("--eval_every_steps", type=int, default=20_000)
    ap.add_argument("--train_split",      type=float, default=0.8)
    ap.add_argument("--device",           type=str, default="cuda", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--n_envs",           type=int, default=12, help="Number of parallel environments")

    # shaping knobs — all zeroed by default so agent learns raw PnL dynamics
    ap.add_argument("--flat_penalty_bps",       type=float, default=0.0)
    ap.add_argument("--inactivity_steps",       type=int,   default=256)
    ap.add_argument("--inactivity_penalty_bps", type=float, default=0.0)  # FIX: was 1.0
    ap.add_argument("--turnover_reward_coeff",  type=float, default=0.0)
    ap.add_argument("--trade_threshold",        type=float, default=0.0)  # FIX: was 0.01
    ap.add_argument("--deadband_frac",          type=float, default=0.0)  # FIX: was 0.05
    ap.add_argument("--min_hold_steps",         type=int,   default=0)
    ap.add_argument("--cooldown_steps",         type=int,   default=0)    # FIX: was 3

    args = ap.parse_args()

    main(
        features_path=Path(args.features),
        output_prefix=Path(args.output),
        timesteps=args.timesteps,
        seed=args.seed,
        eval_every_steps=args.eval_every_steps,
        train_split=args.train_split,
        device_arg=args.device,
        n_envs=args.n_envs,
        flat_penalty_bps=args.flat_penalty_bps,
        inactivity_steps=args.inactivity_steps,
        inactivity_penalty_bps=args.inactivity_penalty_bps,
        turnover_reward_coeff=args.turnover_reward_coeff,
        trade_threshold=args.trade_threshold,
        deadband_frac=args.deadband_frac,
        min_hold_steps=args.min_hold_steps,
        cooldown_steps=args.cooldown_steps,
    )
