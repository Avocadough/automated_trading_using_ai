import pandas as pd
import json
from src.rl_env.crypto_env import CryptoTradingEnv

df = pd.read_parquet('data/features/btc_1h_spa.parquet')
with open('data/features/btc_1h_spa_meta.json') as f:
    meta = json.load(f)
features = meta['features']

env = CryptoTradingEnv(
    df=df, features=features, window_size=64, 
    normalize=True, drawdown_penalty_coeff=0.5, # the train time setting!
    # match train settings:
    flat_penalty_bps=0.0,
    inactivity_steps=256,
    inactivity_penalty_bps=0.0,
    trade_threshold=0.0,
    deadband_frac=0.0,
    cooldown_steps=0,
)
obs, info = env.reset()

print('Initial balance:', env.balance, 'equity peak:', env.equity_peak)

# Force Action 1 (Flat) for 10 steps
total_r = 0.0
for i in range(10):
   # Action map: 0=short, 1=flat, 2=long 
   obs, r, done, _, info = env.step(1)
   total_r += float(r)
   print(f"step {i}: action=1, r={r:.6f}, pos={info['qty']:.6f}, dd={info['drawdown']:.6f}")
   if done: break

print('\nTotal reward for first 10 strictly Flat steps:', total_r)

print('\nNow testing with a simulated 10% drawdown...')
env.equity_peak = 10000.0
env.balance = 9000.0
env.qty = 0.0
for i in range(10):
   obs, r, done, _, info = env.step(1)
   print(f"step {i}: action=1, r={r:.6f}, pos={info['qty']:.6f}, dd={info['drawdown']:.6f}")
