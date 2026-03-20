import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import pandas as pd
import json
from stable_baselines3 import PPO
from src.rl_env.crypto_env import CryptoTradingEnv

df = pd.read_parquet('data/features/btc_1h_spa.parquet')
with open('data/features/btc_1h_spa_meta.json') as f:
    meta = json.load(f)
features = meta['features']

model = PPO.load('data/models/ppo_spa_btc_1h.zip', device='cpu')
env = CryptoTradingEnv(
    df=df, features=features, window_size=64, 
    normalize=True, drawdown_penalty_coeff=0.0,
    # match train settings:
    flat_penalty_bps=0.0,
    inactivity_steps=256,
    inactivity_penalty_bps=0.0,
)
obs, info = env.reset()

print("Initial qty:", info)
total_r = 0.0
for i in range(10):
   action, _ = model.predict(obs, deterministic=True)
   obs, r, done, _, info = env.step(action)
   total_r += float(r)
   print(f'step {i}: action={action}, r={r}, pos={info["qty"]}')
   if done: break
print('Total R first 10 steps:', total_r)

print("\n--- Testing pure Flat behavior directly ---")
obs, info = env.reset()
total_r_flat = 0.0
for i in range(10):
   # Force Action 1 (Flat)
   obs, r, done, _, info = env.step(1)
   total_r_flat += float(r)
   print(f'step {i}: forced=1, r={r}, pos={info["qty"]}')
   if done: break
print('Total R first 10 steps (Forced Flat):', total_r_flat)
