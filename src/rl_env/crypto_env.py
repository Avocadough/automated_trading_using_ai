import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class CryptoTradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, df, window_size=64, initial_balance=1000, 
                taker_fee=0.0005, slippage=0.0002, position_penalty=-0.01):
        super(CryptoTradingEnv, self).__init__()

        self.df = df.dropna().reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.taker_fee = taker_fee
        self.slippage = slippage
        self.position_penalty = position_penalty
        
        # Features to be used as observation
        self.features = ['ret_1', 'MA7', 'MA30', 'rolling_std_20', 'close_z']
        
        # Action space: {0: short, 1: flat, 2: long}
        self.action_space = spaces.Discrete(3)

        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size, len(self.features)),
            dtype=np.float32
        )
        
        self.current_step = 0
        self.balance = 0
        self.position = 0 # -1 for short, 0 for flat, 1 for long
        self.entry_price = 0
        self.total_reward = 0

    def _get_obs(self):
        # --- นี่คือจุดที่แก้ไข ---
        # การ slice ด้วย .loc ใน pandas นั้น inclusive (นับตัวสุดท้ายด้วย)
        # ดังนั้นเราต้อง slice จาก start ถึง end-1 เพื่อให้ได้จำนวน window_size พอดี
        start = self.current_step - self.window_size + 1
        end = self.current_step 
        obs_slice = self.df.iloc[start : end + 1] # ใช้ .iloc เพื่อความแม่นยำและแก้ปัญหา off-by-one
        obs = obs_slice[self.features].values
        # --- จบส่วนที่แก้ไข ---
        return obs.astype(np.float32)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size - 1
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.total_reward = 0
        
        initial_obs = self._get_obs()
        info = {"step": self.current_step, "balance": self.balance, "position": self.position}
        return initial_obs, info

    def step(self, action):
        # map action from {0, 1, 2} to {-1, 0, 1}
        target_position = action - 1 
        
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step >= len(self.df) - 1
        
        current_price = self.df.loc[self.current_step, 'close']
        reward = 0
        
        # Calculate PnL if a position is open
        if self.position != 0:
            pnl_pct = (current_price - self.entry_price) / self.entry_price if self.position == 1 \
                    else (self.entry_price - current_price) / self.entry_price
            reward = pnl_pct * self.balance
            
        # Execute trade if position changes
        if target_position != self.position:
            # Close existing position and realize PnL
            self.balance += reward
            reward = 0 # PnL is now baked into balance
            
            # Apply transaction costs for closing
            self.balance *= (1 - self.taker_fee - self.slippage)
            
            # Apply penalty for changing position
            reward += self.position_penalty
            
            # Open new position
            if target_position != 0:
                self.balance *= (1 - self.taker_fee - self.slippage)
                self.entry_price = current_price
            
            self.position = target_position

        self.total_reward += reward

        obs = self._get_obs()
        info = {"step": self.current_step, "balance": self.balance, "position": self.position}
        
        return obs, reward, terminated, False, info