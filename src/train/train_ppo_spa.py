# src/train/train_ppo_spa.py
from __future__ import annotations

import os
import sys
import json
from pathlib import Path
import argparse

import pandas as pd
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback

# project root so we can import env
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.rl_env.crypto_env import CryptoTradingEnv


def load_meta(features_path: Path):
    meta_path = Path(str(features_path).replace(".parquet", "_meta.json"))
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta not found: {meta_path}")
    meta = json.loads(meta_path.read_text())
    features = meta.get("features")
    window_size = int(meta.get("window_size", 64))
    if not features or not isinstance(features, list):
        raise ValueError("Invalid meta: 'features' must be a list.")
    return features, window_size


def build_env(df: pd.DataFrame, features: list[str], window_size: int,
              initial_balance: float = 10_000.0,
              taker_fee: float = 0.0005,
              position_limit: float = 0.30,
              slippage_bps: float = 0.0,
              reward_scale: float = 100.0,
              normalize: bool = True,
              action_mode: str = "discrete"):

    def _make():
        return CryptoTradingEnv(
            df=df,
            features=features,
            window_size=window_size,
            initial_balance=initial_balance,
            taker_fee=taker_fee,
            position_limit=position_limit,
            slippage_bps=slippage_bps,
            reward_scale=reward_scale,
            normalize=normalize,
            action_mode=action_mode
        )
    env = DummyVecEnv([_make])
    env = VecMonitor(env)  # VERY important for SB3 logging (prevents KeyError: 'r')
    return env


def main(features_path: Path,
        output_prefix: Path,
        timesteps: int = 300_000,
        seed: int = 42,
        eval_every_steps: int = 10_000,
        train_split: float = 0.8):

    print(f"[info] Loading features from: {features_path}")
    features, window_size = load_meta(features_path)
    print(f"[info] Using features={features} | window_size={window_size}")

    df_all = pd.read_parquet(features_path)
    needed = ["close"] + features
    missing = [c for c in needed if c not in df_all.columns]
    if missing:
        raise ValueError(f"Missing columns in features parquet: {missing}")

    df_all = df_all[needed].dropna().reset_index(drop=True)

    n_train = int(len(df_all) * train_split)
    train_df = df_all.iloc[:n_train].copy()
    eval_df  = df_all.iloc[n_train:].copy()

    print(f"[info] Dataset rows: total={len(df_all):,} | train={len(train_df):,} | eval={len(eval_df):,}")

    # Build envs
    train_env = build_env(
        train_df, features, window_size,
        initial_balance=10_000.0,
        taker_fee=0.0005,
        position_limit=0.30,
        slippage_bps=0.0,
        reward_scale=100.0,
        normalize=True,
        action_mode="discrete"
    )

    eval_env = build_env(
        eval_df, features, window_size,
        initial_balance=10_000.0,
        taker_fee=0.0005,
        position_limit=0.30,
        slippage_bps=0.0,
        reward_scale=100.0,
        normalize=True,
        action_mode="discrete"
    )

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] Device: {device}")

    # Model + callbacks
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        device=device,
        seed=seed,
        tensorboard_log="./ppo_logs_spa/",
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.20
    )

    eval_dir = Path("data/models/_eval_spa")
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(eval_dir),
        log_path=str(eval_dir),
        eval_freq=eval_every_steps,
        n_eval_episodes=1,   # eval เป็นแบบ walk-forward ทีละ episode
        deterministic=True,
        render=False
    )

    print(f"[info] Training for {timesteps:,} timesteps...")
    model.learn(total_timesteps=timesteps, callback=eval_cb)

    # Save final model
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    model_path = f"{output_prefix}.zip"
    model.save(output_prefix)
    print(f"[ok] Saved final model to {model_path}")

    # If a best model was saved during eval, print its path
    best_model_files = sorted(eval_dir.glob("best_model.zip"))
    if best_model_files:
        print(f"[ok] Best model saved by EvalCallback: {best_model_files[-1]}")
    else:
        print("[info] No separate best model produced (EvalCallback conditions not met).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train PPO agent using SPA-based features")
    ap.add_argument("--features", type=str, default="data/features/btc_1h_spa.parquet",
                    help="Path to features parquet generated by make_features_spa.py")
    ap.add_argument("--output", type=str, default="data/models/ppo_spa_btc_1h",
                    help="Prefix path to save final model (without .zip)")
    ap.add_argument("--timesteps", type=int, default=300000)
    
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_every_steps", type=int, default=10000)
    ap.add_argument("--train_split", type=float, default=0.8)
    args = ap.parse_args()

    main(
        features_path=Path(args.features),
        output_prefix=Path(args.output),
        timesteps=args.timesteps,
        seed=args.seed,
        eval_every_steps=args.eval_every_steps,
        train_split=args.train_split
    )
