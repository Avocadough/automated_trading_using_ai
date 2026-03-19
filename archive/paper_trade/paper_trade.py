import time
import argparse
from pathlib import Path
import json

import pandas as pd
import numpy as np

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

class PaperPortfolio:
    def __init__(self, initial_balance=10000.0):
        self.cash = initial_balance
        self.qty = 0.0
        self.avg_entry = 0.0
        self.equity = initial_balance

    def update(self, price: float):
        if self.qty > 0:
            unrealized = (price - self.avg_entry) * self.qty
            self.equity = self.cash + (self.qty * self.avg_entry) + unrealized
        else:
            self.equity = self.cash

    def execute_order(self, target_pos_frac: float, price: float):
        current_alloc = (self.qty * price) / self.equity if self.equity > 0 else 0.0
        diff = target_pos_frac - current_alloc
        
        if diff > 0.01: # buy
            buy_val = diff * self.equity
            buy_qty = buy_val / price
            self.cash -= buy_val
            
            total_val = (self.qty * self.avg_entry) + buy_val
            self.qty += buy_qty
            self.avg_entry = total_val / self.qty if self.qty > 0 else 0.0
            
        elif diff < -0.01: # sell
            sell_val = abs(diff) * self.equity
            sell_qty = sell_val / price
            sell_qty = min(sell_qty, self.qty)
            
            self.cash += (sell_qty * price)
            self.qty -= sell_qty
            if self.qty < 1e-8:
                self.qty = 0.0
                self.avg_entry = 0.0
                
        self.update(price)
        
    def to_dict(self):
        return {
            "cash": self.cash,
            "qty": self.qty,
            "avg_entry": self.avg_entry,
            "equity": self.equity,
            "timestamp": time.time()
        }

def run_loop(model_path: str, pair: str, interval: str, duration_hours: float):
    print(f"[info] Bootstrapping Paper Trading Loop for {pair}")
    print(f"[info] Model: {model_path} | Interval: {interval}")
    
    port = PaperPortfolio()
    state_file = Path("data/paper_trade_state.json")
    state_file.parent.mkdir(exist_ok=True, parents=True)
    
    # Simple placeholder loop for demonstration
    # In full production, we poll Binance REST API for Klines every 15s
    try:
        steps = int(duration_hours * 3600 / 15)
        for i in range(steps):
            current_price = 60000.0 + (np.sin(i) * 1000.0) # mock price
            
            # mock PPO action (0: hold, 1: long)
            action = int(np.random.choice([0, 1], p=[0.9, 0.1])) 
            target_frac = 0.3 if action == 1 else 0.0
            
            port.execute_order(target_frac, current_price)
            state_file.write_text(json.dumps(port.to_dict(), indent=2))
            
            print(f"[{time.strftime('%H:%M:%S')}] Price: {current_price:.2f} | Eq: ${port.equity:.2f} | Qty: {port.qty:.4f}")
            time.sleep(1) # mock 1 sec for speed
            
    except KeyboardInterrupt:
        print("\n[info] Stopping Paper Trading.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Dummy Testnet Paper Trading Loop")
    ap.add_argument("--model", type=str, default="data/models/ppo_spa_btc_1h.zip")
    ap.add_argument("--pair", type=str, default="BTCUSDT")
    ap.add_argument("--interval", type=str, default="1h")
    ap.add_argument("--duration_hours", type=float, default=1.0)
    args = ap.parse_args()
    
    run_loop(args.model, args.pair, args.interval, args.duration_hours)
