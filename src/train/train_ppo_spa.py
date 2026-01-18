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
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback

# project root so we can import env
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.rl_env.crypto_env import CryptoTradingEnv


def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # คุม deterministic ให้มากขึ้น
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def load_meta_or_infer(features_path: Path):
    """
    พยายามโหลด <features> และ <window_size> จากไฟล์ *_meta.json
    ถ้าไม่มี meta: เดา features = คอลัมน์ทั้งหมดที่ไม่ใช่ 'close'
    และตั้ง window_size = 64
    """
    meta_path = Path(str(features_path).replace(".parquet", "_meta.json"))
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        features = meta.get("features")
        window_size = int(meta.get("window_size", 64))
        if not features or not isinstance(features, list):
            raise ValueError("Invalid meta: 'features' must be a non-empty list.")
        return features, window_size, True  # loaded_from_meta
    # fallback: infer
    df_tmp = pd.read_parquet(features_path, columns=None)
    all_cols = list(df_tmp.columns)
    if "close" not in all_cols:
        raise ValueError("Parquet must contain 'close' column.")
    inferred = [c for c in all_cols if c != "close"]
    if not inferred:
        raise ValueError("No feature columns were found to infer (only 'close' present).")
    return inferred, 64, False  # not from meta


# ==== แทนที่ฟังก์ชัน build_env ทั้งก้อนใน src/train/train_ppo_spa.py ====
def build_env(
    df: pd.DataFrame,
    features: list[str],
    window_size: int,
    *,
    initial_balance: float = 10_000.0,
    taker_fee: float = 0.0005,
    position_limit: float = 0.30,
    slippage_bps: float = 0.0,
    reward_scale: float = 100.0,
    normalize: bool = True,
    action_mode: str = "discrete",
    # ---- shaping knobs (ตั้ง 0 เพื่อปิดได้) ----
    flat_penalty_bps: float = 0.0,
    inactivity_steps: int = 256,
    inactivity_penalty_bps: float = 0.0,
    turnover_reward_coeff: float = 0.0,
    trade_threshold: float = 0.02,
    # ---- execution smoothing ----
    deadband_frac: float = 0.25,
    min_hold_steps: int = 64,
    cooldown_steps: int = 16,
):
    def _make():
        common_kwargs = dict(
            df=df,
            features=features,
            window_size=window_size,
            initial_balance=initial_balance,
            taker_fee=taker_fee,
            position_limit=position_limit,
            slippage_bps=slippage_bps,
            reward_scale=reward_scale,
            normalize=normalize,
            action_mode=action_mode,
            # shaping
            flat_penalty_bps=flat_penalty_bps,
            inactivity_steps=inactivity_steps,
            inactivity_penalty_bps=inactivity_penalty_bps,
            turnover_reward_coeff=turnover_reward_coeff,
            trade_threshold=trade_threshold,
            # smoothing
            deadband_frac=deadband_frac,
            min_hold_steps=min_hold_steps,
            cooldown_steps=cooldown_steps,
        )
        # ส่งเฉพาะ args ที่ __init__ ของ Env รองรับ
        env_sig = inspect.signature(CryptoTradingEnv.__init__).parameters
        allowed = {k: v for k, v in common_kwargs.items() if k in env_sig}
        return CryptoTradingEnv(**allowed)

    env = DummyVecEnv([_make])
    env = VecMonitor(env)
    return env




def main(
    features_path: Path,
    output_prefix: Path,
    timesteps: int = 300_000,
    seed: int = 42,
    eval_every_steps: int = 10_000,
    train_split: float = 0.8,
    device_arg: str | None = None,
    # ---- CLI-exposed shaping knobs ----
    flat_penalty_bps: float = 0.25,
    inactivity_steps: int = 128,
    inactivity_penalty_bps: float = 3.0,
    turnover_reward_coeff: float = 0.05,
    trade_threshold: float = 0.01,
    deadband_frac: float = 0.10,
):
    set_global_seeds(seed)

    print(f"[info] Loading features from: {features_path}")
    features, window_size, from_meta = load_meta_or_infer(features_path)
    print(f"[info] Using features={features} | window_size={window_size} | from_meta={from_meta}")

    df_all = pd.read_parquet(features_path)
    needed = ["close"] + features
    missing = [c for c in needed if c not in df_all.columns]
    if missing:
        raise ValueError(f"Missing columns in features parquet: {missing}")

    # clean
    df_all = df_all[needed].dropna().reset_index(drop=True)

    # ต้องให้ยาวกว่า window_size + อย่างน้อย 100 step เพื่อเกิดสัญญาณ/เรียนรู้
    if len(df_all) < window_size + 100:
        raise ValueError(
            f"Dataset too small ({len(df_all)}) for window_size={window_size}. "
            f"Need at least {window_size+100} rows."
        )

    # split
    n_train = max(window_size + 1, int(len(df_all) * train_split))
    n_train = min(n_train, len(df_all) - max(window_size + 1, 100))  # กัน eval ว่าง
    train_df = df_all.iloc[:n_train].copy()
    eval_df = df_all.iloc[n_train:].copy()

    if len(eval_df) < window_size + 1:
        # ถ้า eval สั้นเกินไป ให้ย้าย boundary ลงมาอีกเล็กน้อย
        shift = (window_size + 1) - len(eval_df)
        n_train = max(window_size + 1, n_train - shift)
        train_df = df_all.iloc[:n_train].copy()
        eval_df = df_all.iloc[n_train:].copy()

    print(f"[info] Dataset rows: total={len(df_all):,} | train={len(train_df):,} | eval={len(eval_df):,}")

    # Build envs
    common_env_kwargs = dict(
        initial_balance=10_000.0,
        taker_fee=0.0005,
        position_limit=0.30,
        slippage_bps=0.0,
        reward_scale=100.0,
        normalize=True,
        action_mode="discrete",
        flat_penalty_bps=flat_penalty_bps,
        inactivity_steps=inactivity_steps,
        inactivity_penalty_bps=inactivity_penalty_bps,
        turnover_reward_coeff=turnover_reward_coeff,
        trade_threshold=trade_threshold,
        deadband_frac=deadband_frac,
    )

    train_env = build_env(
        train_df, features, window_size,
        initial_balance=10_000.0, taker_fee=0.0005, position_limit=0.30,
        slippage_bps=0.0, reward_scale=100.0, normalize=True, action_mode="discrete",
        # shaping – ปิดไว้ก่อน (กันบังคับเทรด)
        flat_penalty_bps=0.0, inactivity_steps=256, inactivity_penalty_bps=0.0,
        turnover_reward_coeff=0.0, trade_threshold=0.02,
        # execution smoothing – “ยาที่แรงพอจะทำให้ ~200 ไม้”
        deadband_frac=0.25,   # 25% ของ equity ต่อการเปลี่ยนพอร์ตถึงจะยอมขยับ
        min_hold_steps=64,    # ต้องถือขั้นต่ำ 64 แท่ง
        cooldown_steps=16,    # เทรดแล้วพัก 16 แท่ง
    )

    eval_env = build_env(
        eval_df, features, window_size,
        initial_balance=10_000.0, taker_fee=0.0005, position_limit=0.30,
        slippage_bps=0.0, reward_scale=100.0, normalize=True, action_mode="discrete",
        flat_penalty_bps=0.0, inactivity_steps=256, inactivity_penalty_bps=0.0,
        turnover_reward_coeff=0.0, trade_threshold=0.02,
        deadband_frac=0.25, min_hold_steps=64, cooldown_steps=16,
    )



    # Device
    if device_arg is None or device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg
    print(f"[info] Device: {device}")

    # Model + callbacks
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        device=device,
        seed=seed,
        tensorboard_log="./ppo_logs_spa/",
        n_steps=1024,           # อัปเดตถี่ขึ้น
        batch_size=512,         # ใหญ่ขึ้นเพื่อความนิ่ง
        learning_rate=2e-4,     # ช้าลงเล็กน้อย
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,          # สำคัญ: สำรวจมากขึ้น
        clip_range=0.20,
    )

    eval_dir = Path("data/models/_eval_spa")
    eval_dir.mkdir(parents=True, exist_ok=True)

    # eval_every_steps ต้องน้อยกว่าจำนวน timesteps และอย่างน้อย 1
    eval_every_steps = max(1, min(eval_every_steps, max(1, timesteps // 2)))
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(eval_dir),
        log_path=str(eval_dir),
        eval_freq=eval_every_steps,
        n_eval_episodes=1,   # walk-forward one episode
        deterministic=True,
        render=False,
    )

    print(f"[info] Training for {timesteps:,} timesteps... (eval every {eval_every_steps} steps)")
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
    ap.add_argument("--features", type=str,
                    default="data/features/btc_15m_rl_features_split_validated.parquet",
                    help="Path to features parquet generated by your GA/feature script")
    ap.add_argument("--output", type=str, default="data/models/ppo_spa_btc_15m",
                    help="Prefix path to save final model (without .zip)")
    ap.add_argument("--timesteps", type=int, default=150000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_every_steps", type=int, default=10000)
    ap.add_argument("--train_split", type=float, default=0.8)
    ap.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"],
                    help='Device to use for training. Default: "auto" -> "cuda" if available else "cpu"')

    # ---- expose shaping knobs ----
    ap.add_argument("--flat_penalty_bps", type=float, default=0.25)
    ap.add_argument("--inactivity_steps", type=int, default=128)
    ap.add_argument("--inactivity_penalty_bps", type=float, default=3.0)
    ap.add_argument("--turnover_reward_coeff", type=float, default=0.05)
    ap.add_argument("--trade_threshold", type=float, default=0.01)
    ap.add_argument("--deadband_frac", type=float, default=0.10)
    ap.add_argument("--min_hold_steps", type=int, default=4)

    args = ap.parse_args()
    device = args.device  # "cpu" | "cuda" | "auto"

    main(
        features_path=Path(args.features),
        output_prefix=Path(args.output),
        timesteps=args.timesteps,
        seed=args.seed,
        eval_every_steps=args.eval_every_steps,
        train_split=args.train_split,
        device_arg=device,
        flat_penalty_bps=args.flat_penalty_bps,
        inactivity_steps=args.inactivity_steps,
        inactivity_penalty_bps=args.inactivity_penalty_bps,
        turnover_reward_coeff=args.turnover_reward_coeff,
        trade_threshold=args.trade_threshold,
        deadband_frac=args.deadband_frac,
    )
