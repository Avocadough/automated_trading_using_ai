import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import argparse
from pathlib import Path
import sys
import os
# หา Path ของ Root Directory (สองระดับขึ้นไปจาก __file__)
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..' 
))
sys.path.append(PROJECT_ROOT)
# --- จบส่วนที่แก้ไข ---

from src.rl_env.crypto_env import CryptoTradingEnv

def backtest_agent(features_path, model_path):
    """
    Loads a trained PPO agent and runs a backtest on the out-of-sample data.
    """
    print("--- Starting PPO Agent Backtest ---")
    
    # 1. โหลดข้อมูลฟีเจอร์ทั้งหมด
    print(f"Loading feature data from {features_path}...")
    df = pd.read_parquet(features_path)

    # 2. แบ่งข้อมูล: ใช้ 20% สุดท้ายเป็นข้อมูลทดสอบ (Test Set)
    split_index = int(len(df) * 0.8)
    test_df = df.iloc[split_index:].reset_index(drop=True)
    print(f"Using last {len(test_df)} rows for out-of-sample testing...")

    # 3. โหลดโมเดล PPO ที่เทรนเสร็จแล้ว
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return
    print(f"Loading trained model from {model_path}...")
    model = PPO.load(model_path)

    # 4. สร้าง Environment ด้วยข้อมูลทดสอบ
    eval_env = CryptoTradingEnv(df=test_df)
    
    # 5. เริ่ม Backtest Loop
    obs, info = eval_env.reset()
    done = False
    
    initial_balance = eval_env.balance
    print(f"Initial backtest balance: ${initial_balance:,.2f}")

    while not done:
        # ใช้โมเดลในการตัดสินใจ (predict) จาก observation ปัจจุบัน
        # deterministic=True หมายถึงให้โมเดลเลือก Action ที่ดีที่สุดเสมอ (ไม่สุ่ม)
        action, _states = model.predict(obs, deterministic=True)
        
        # ให้ Environment ทำงานตาม Action ที่โมเดลตัดสินใจ
        obs, reward, terminated, truncated, info = eval_env.step(action)
        
        done = terminated or truncated

    # 6. แสดงผลลัพธ์
    final_balance = eval_env.balance
    total_return_pct = ((final_balance - initial_balance) / initial_balance) * 100
    
    print("\n--- PPO Backtest Results ---")
    print(f"Final Balance: ${final_balance:,.2f}")
    print(f"Initial Balance: ${initial_balance:,.2f}")
    print(f"Total Return: {total_return_pct:.2f}%")
    print("----------------------------\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Backtest a trained PPO agent.")
    parser.add_argument("--features", type=str, default="data/features/btc_15m_rl_features_split_validated.parquet", help="Path to the features file.")
    parser.add_argument("--model", type=str, default="data/models/_eval_spa/best_model.zip", help="Path to the trained PPO model file.")
    args = parser.parse_args()

    backtest_agent(Path(args.features), Path(args.model))