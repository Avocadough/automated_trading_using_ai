import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import argparse
from pathlib import Path
import sys
import os
import json
import matplotlib.pyplot as plt

# --- Setup Project Root ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.rl_env.crypto_env import CryptoTradingEnv

def load_meta_data(features_path: Path):
    """โหลด Metadata เพื่อให้ Config ตรงกับตอนเทรน"""
    meta_path = Path(str(features_path).replace(".parquet", "_meta.json"))
    if not meta_path.exists():
        # Fallback: ถ้าไม่มี meta ให้เดา window_size=288 (ค่าที่เราใช้ล่าสุด)
        print(f"⚠️ Meta file not found at {meta_path}. Using default window_size=288.")
        return None, 288
    
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta.get("features"), int(meta.get("window_size", 64))

def backtest_agent(features_path, model_path):
    print("--- 🚀 Starting PPO Agent Backtest (Final Fixed Version) ---")
    
    # 1. Load FULL Data
    print(f"[Info] Loading FULL feature data from {features_path}...")
    df_full = pd.read_parquet(features_path)
    features_from_meta, window_size = load_meta_data(features_path)
    
    # ถ้า meta ไม่มี feature list ให้ใช้ทุกคอลัมน์ยกเว้น close
    if features_from_meta:
        features = features_from_meta
    else:
        features = [c for c in df_full.columns if c != 'close']

    print(f"[Info] Window Size: {window_size} | Features: {len(features)}")

    # 2. Determine Split Index (Start Testing at last 20%)
    split_index = int(len(df_full) * 0.8)
    
    # 3. Create Environment (REALITY MODE + DAY TRADE ENFORCEMENT)
    print(f"[Info] Initializing Env...")
    
    eval_env = CryptoTradingEnv(
        df=df_full,
        features=features,
        window_size=window_size,
        initial_balance=10_000.0,
        taker_fee=0.0005,
        slippage_bps=0.0, 
        reward_scale=1.0,
        
        # 🚨 IMPORTANT: ต้องตรงกับตอนเทรน (แนะนำให้ใช้ False ตามสูตรล่าสุด)
        normalize=False,   
        
        # --- Reality Mode (No Fake Penalties) ---
        flat_penalty_bps=0.0,
        inactivity_penalty_bps=0.0,  # ไม่ปรับเงินจริง
        turnover_reward_coeff=0.0,
        
        # --- 🚨 Critical Fix: Force Close Logic ---
        # ต้องเปิด inactivity_steps ไว้เพื่อให้ logic บังคับขายทำงานเมื่อครบ 24 ชม.
        inactivity_steps=288,        
        
        # Constraints
        deadband_frac=0.10,
        min_hold_steps=12,
        cooldown_steps=16
    )

    # 4. Warp to Test Set (ข้ามข้อมูล Train ไปเลย)
    obs, info = eval_env.reset()
    eval_env.current_step = split_index
    obs = eval_env._build_obs() 
    
    # Reset Portfolio to Initial State
    eval_env.balance = 10000.0
    eval_env.equity = 10000.0
    eval_env.qty = 0
    eval_env.avg_entry = 0
    
    print(f"[Info] Simulation Warped to Step: {split_index}")

    # 5. Load Model
    print(f"[Info] Loading model from {model_path}...")
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 6. Run Simulation Loop
    done = False
    
    # Data collecting for plots
    equity_curve = [eval_env.equity] 
    price_curve = []
    buy_markers = []  # (step_index, price)
    sell_markers = [] # (step_index, price)
    
    trade_count = 0
    win_count = 0
    last_qty = 0

    # สร้าง Index สำหรับพล็อต (เริ่มที่ 0 เพื่อความง่าย)
    plot_steps = []
    current_plot_step = 0

    print("running...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        
        # Collect Data
        equity_curve.append(info['equity'])
        current_price = df_full.iloc[eval_env.current_step]['close']
        price_curve.append(current_price)
        plot_steps.append(current_plot_step)
        
        # Detect Trades for Plotting
        # ถ้า Qty เพิ่มขึ้น -> Buy
        if info['qty'] > last_qty:
            buy_markers.append((current_plot_step, current_price))
        # ถ้า Qty ลดลง -> Sell
        elif info['qty'] < last_qty:
            sell_markers.append((current_plot_step, current_price))
            
        # Count Closed Trades & Win Rate (Check Realized PnL)
        if 'realized_pnl' in info and info['realized_pnl'] != 0:
            trade_count += 1
            if info['realized_pnl'] > 0:
                win_count += 1
        
        last_qty = info['qty']
        current_plot_step += 1
        done = terminated or truncated

    # 7. Calculate Statistics
    final_equity = info['equity']
    net_profit = final_equity - 10000
    total_return_pct = (net_profit / 10000) * 100
    win_rate = (win_count / trade_count * 100) if trade_count > 0 else 0.0
    
    # Max Drawdown
    equity_arr = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_arr)
    drawdowns = (equity_arr - running_max) / running_max
    max_dd_pct = drawdowns.min() * 100

    print("\n" + "="*40)
    print(f"📊 FINAL BACKTEST RESULTS")
    print("="*40)
    print(f"Final Equity:     ${final_equity:,.2f}")
    print(f"Net Profit:       ${net_profit:,.2f}")
    print(f"Total Return:     {total_return_pct:.2f}%")
    print(f"Max Drawdown:     {max_dd_pct:.2f}%")
    print("-" * 20)
    print(f"Total Trades:     {trade_count}")
    print(f"Win Rate:         {win_rate:.2f}%")
    print("="*40)

    # 8. Advanced Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1.5]})

    # --- Plot 1: Equity Curve ---
    ax1.plot(equity_curve, label='Portfolio Equity', color='blue', linewidth=1.5)
    ax1.axhline(y=10000, color='red', linestyle='--', alpha=0.5, label='Initial Capital')
    ax1.set_ylabel('Value ($)')
    ax1.set_title(f'Equity Curve (Net Profit: ${net_profit:,.0f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Price Action & Trades ---
    ax2.plot(price_curve, label='BTC Price', color='gray', alpha=0.5)
    
    # Scatter markers for trades
    if buy_markers:
        bx, by = zip(*buy_markers)
        ax2.scatter(bx, by, color='green', marker='^', s=60, label='Buy / Open Long', zorder=5)
    
    if sell_markers:
        sx, sy = zip(*sell_markers)
        ax2.scatter(sx, sy, color='red', marker='v', s=60, label='Sell / Close / Short', zorder=5)

    ax2.set_ylabel('Price ($)')
    ax2.set_xlabel('Steps (5m Candles)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ปรับ default ให้ตรงกับไฟล์ของคุณ
    parser.add_argument("--features", type=str, default="data/features/btc_5m_spa_v2.parquet")
    parser.add_argument("--model", type=str, default="data/models/ppo_spa_btc_5m_v3.zip")
    
    args = parser.parse_args()
    backtest_agent(Path(args.features), Path(args.model))