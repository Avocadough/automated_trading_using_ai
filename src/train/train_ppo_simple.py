# src/train/train_ppo_simple.py
import os, sys, json
from pathlib import Path
import argparse
import pandas as pd
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import os, sys, json
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.rl_env.rl_env_simple import SimpleTradingEnv

def load_meta(features_path: Path):
    meta_path = Path(str(features_path).replace(".parquet", "_meta.json"))
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return meta.get("features", ["log_ret_1","rolling_std_20"]), meta.get("window_size", 64)
    return ["log_ret_1","rolling_std_20"], 64

def main(features_path: Path, output_path: Path, timesteps: int = 200_000, seed: int = 42):
    print(f"Loading features: {features_path}")
    df = pd.read_parquet(features_path)

    features, ws = load_meta(features_path)
    needed = ["close"] + features
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")
    df = df[needed].dropna().reset_index(drop=True)

    # split train (80%)
    train_df = df.iloc[: int(len(df) * 0.8)]

    def make_env():
        return SimpleTradingEnv(
            df=train_df,
            window_size=ws,
            initial_balance=10_000.0,
            taker_fee=0.0005,
            position_limit=0.30,
            reward_scale=100.0,
            features=features
        )

    env = DummyVecEnv([make_env])
    env = VecMonitor(env)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        seed=seed,
        tensorboard_log="./ppo_logs_simple/",
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2
    )

    print(f"Training for {timesteps:,} timesteps...")
    model.learn(total_timesteps=timesteps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"Saved model to {output_path}.zip")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train simple PPO agent for crypto trading")
    ap.add_argument("--features", type=str, default="data/features/btc_1h_simple.parquet")
    ap.add_argument("--output", type=str, default="data/models/ppo_simple_btc_1h")
    ap.add_argument("--timesteps", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    main(Path(args.features), Path(args.output), args.timesteps, args.seed)
