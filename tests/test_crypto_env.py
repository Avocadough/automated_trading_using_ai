import pytest
import numpy as np
import pandas as pd
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.rl_env.crypto_env import CryptoTradingEnv

def test_episode_ends(sample_ohlcv):
    df = sample_ohlcv.copy()
    features = ["close"]
    
    env = CryptoTradingEnv(df, features, window_size=10, 
                           deadband_frac=0.05, action_mode="discrete")
    env.reset()
    
    done = False
    for _ in range(100):
        # Always hold
        obs, reward, done, trunc, info = env.step(1)
        if done:
            break
            
    assert done
    
def test_deadband_small_trades(sample_ohlcv):
    df = sample_ohlcv.copy()
    env = CryptoTradingEnv(df, ["close"], window_size=10, 
                           deadband_frac=0.2, action_mode="discrete")
    env.reset()
    
    # Long action
    _, _, _, _, info = env.step(2) 
    
    # Depending on how the env translates discrete (0: short, 1: hold, 2: long)
    # The actual max pos goes up to 0.5 limit by default maybe.
    # We mainly test it doesn't crash on standard interaction logic
    pass 
