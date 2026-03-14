# src/train/train_ppo.py (หรือไฟล์เดิมของคุณ)
import sys, os, json
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import argparse
from pathlib import Path
import torch

# หา PROJECT_ROOT
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.rl_env.crypto_env import CryptoTradingEnv  # ให้ชี้ไฟล์ env ที่เราใช้จริง

def load_meta(features_path: Path):
    meta_path = Path(str(features_path).replace(".parquet", "_meta.json"))
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return meta.get("features", None), meta.get("window_size", 64)
    # fallback ถ้าไม่มี meta
    return ['ret_1', 'rolling_std_20'], 64

def train_agent(features_path: Path, model_output_path: Path, total_timesteps=300_000, seed: int = 42):
    print(f"Loading features from {features_path}...")
    df = pd.read_parquet(features_path)

    # --- อ่าน meta เพื่อ “ล็อก” feature list + window_size ให้ตรงกันทุกไฟล์ ---
    features, ws = load_meta(features_path)
    print(f"Using features from meta: {features} | window_size={ws}")

    # ตรวจให้มีคอลัมน์ที่ต้องใช้จริง
    needed_cols = ['close'] + features
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    df = df[needed_cols].dropna().reset_index(drop=True)

    # แบ่งเทรน/วาลิเดชัน
    train_df = df.iloc[: int(len(df) * 0.8)]

    print("Creating trading environment...")
    # ส่งพารามิเตอร์สำคัญให้ Env ให้ตรงกับที่ใช้ตอน backtest
    def make_env():
        return CryptoTradingEnv(
            df=train_df,
            window_size=ws,
            # ให้ agent ไม่แช่ FLAT นาน (เหมือนที่เราเพิ่มไว้)
            idle_cost_bps=5e-5,
            idle_cost_ramp=0.02
        )
    env = DummyVecEnv([make_env])

    # seed ให้ผลซ้ำได้
    env.seed(seed)
    torch.manual_seed(seed)

    # เลือกอุปกรณ์
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    print("Training PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_crypto_tensorboard/",
        device=device,
        seed=seed,
        # ค่าพื้นฐานโอเคสำหรับเริ่มต้น; ปรับทีหลังได้
        n_steps=2048,            # ควรหารด้วย batch_size ลงตัว
        batch_size=256,
        learning_rate=3e-4,
        ent_coef=0.01,
        gae_lambda=0.95,
        gamma=0.99,
        clip_range=0.2
    )
    model.learn(total_timesteps=total_timesteps)

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_output_path)
    print(f"Model saved successfully to {model_output_path}.zip")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a PPO agent for crypto trading.")
    parser.add_argument("--features", type=str, default="data/features/btc_1h_features.parquet")
    parser.add_argument("--output", type=str, default="data/models/ppo_btc_1h_v3")
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_agent(Path(args.features), Path(args.output), args.timesteps, args.seed)
