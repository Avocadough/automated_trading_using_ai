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
    
    # Run backtest with costs
    portfolio = vbt.Portfolio.from_signals(
        close_price,
        entries,
        exits,
        fees=0.0005,    # 0.05%
        slippage=0.0002, # 0.02%
        freq='1T'       # Set frequency to 1 minute for accurate time-based stats
    )

    print("\n--- MA Crossover Baseline Backtest: Full Stats ---")
    # --- ส่วนที่แก้ไข ---
    # เราจะพิมพ์ stats ทั้งหมดที่ vectorbt คำนวณให้ แทนการเลือกพิมพ์ทีละค่า
    stats = portfolio.stats()
    print(stats)
    # --- จบส่วนที่แก้ไข ---
    print("--------------------------------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a baseline backtest.")
    parser.add_argument("--data", type=str, default="data/raw/btc_1m.parquet", help="Path to the raw kline data.")
    args = parser.parse_args()
    
    run_baseline_backtest(Path(args.data))