import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import argparse
from pathlib import Path
from src.rl_env.crypto_env import CryptoTradingEnv
import torch # Import torch เพื่อเช็ค GPU

def train_agent(features_path, model_output_path, total_timesteps=300000):
    """Trains a PPO agent and saves the model."""
    print(f"Loading features from {features_path}...")
    df = pd.read_parquet(features_path)
    
    train_df = df.iloc[:int(len(df) * 0.8)]

    print("Creating trading environment...")
    env = DummyVecEnv([lambda: CryptoTradingEnv(df=train_df)])

    # --- ส่วนที่แก้ไข ---
    # ตรวจสอบว่ามี GPU หรือไม่ และกำหนด device
    if torch.cuda.is_available():
        device = "cuda"
        print("GPU is available! Training on GPU...")
    else:
        device = "cpu"
        print("GPU not available, training on CPU...")

    print("Training PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_crypto_tensorboard/",
        device=device  # ระบุให้ใช้ device ที่เรากำหนด (cuda หรือ cpu)
    )
    # --- จบส่วนที่แก้ไข ---
    
    model.learn(total_timesteps=total_timesteps)

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_output_path)
    print(f"Model saved successfully to {model_output_path}.zip")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a PPO agent for crypto trading.")
    parser.add_argument("--features", type=str, default="data/features/btc_1h_features.parquet", help="Path to the features file.")
    parser.add_argument("--output", type=str, default="data/models/ppo_btc_1h", help="Path to save the trained model (without extension).")
    parser.add_argument("--timesteps", type=int, default=500000, help="Total timesteps to train the agent.")
    args = parser.parse_args()

    train_agent(Path(args.features), Path(args.output), args.timesteps)