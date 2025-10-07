import pandas as pd
import vectorbt as vbt
import argparse
from pathlib import Path

def run_baseline_backtest(raw_data_path):
    """Runs a simple Moving Average Crossover baseline backtest using vectorbt."""
    if not raw_data_path.exists():
        print(f"Error: Raw data file not found at {raw_data_path}")
        return

    print(f"Loading data from {raw_data_path}...")
    df = pd.read_parquet(raw_data_path)
    close_price = df['close']

    # Baseline Strategy: MA Crossover
    fast_ma = vbt.MA.run(close_price, 7, short_name='fast')
    slow_ma = vbt.MA.run(close_price, 30, short_name='slow')

    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    
    # --- ส่วนที่แก้ไข ---
    # ตรวจสอบชื่อไฟล์เพื่อกำหนด Frequency โดยอัตโนมัติ
    if '1h' in raw_data_path.name:
        freq = '1h'
        print("Detected 1-hour timeframe. Setting frequency to '1H'.")
    else:
        freq = '1min' # ใช้ '1min' แทน '1T' เพื่อแก้ FutureWarning
        print("Detected 1-minute timeframe. Setting frequency to '1min'.")
    # --- จบส่วนที่แก้ไข ---

    portfolio = vbt.Portfolio.from_signals(
        close_price,
        entries,
        exits,
        fees=0.0005,
        slippage=0.0002,
        freq=freq
    )

    print("\n--- MA Crossover Baseline Backtest: Full Stats ---")
    stats = portfolio.stats()
    print(stats)
    print("--------------------------------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a baseline backtest.")
    # เปลี่ยนค่า default ให้เป็นไฟล์ 1 ชั่วโมงเพื่อความสะดวก
    parser.add_argument("--data", type=str, default="data/raw/btc_1h.parquet", help="Path to the raw kline data.")
    args = parser.parse_args()
    
    run_baseline_backtest(Path(args.data))