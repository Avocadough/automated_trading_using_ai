import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class AdvancedCryptoTradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, df, window_size=64, initial_balance=1000, 
                  taker_fee=0.0005, position_limit=0.9, 
                  sharpe_ratio_window=100): # เพิ่ม parameter ใหม่ๆ
        super(AdvancedCryptoTradingEnv, self).__init__()

        self.df = df.dropna().reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.taker_fee = taker_fee
        self.position_limit = position_limit # giới hạn vị thế (ví dụ: 90% số dư)
        self.sharpe_ratio_window = sharpe_ratio_window
        
        # === 1. Richer Observation Space ===
        self.features = ['ret_1', 'MA7', 'MA30', 'rolling_std_20', 'close_z', 'volume'] # เพิ่ม volume
        
        # เพิ่มข้อมูลเกี่ยวกับสถานะของบัญชีเข้าไปใน Observation
        # [position_size, unrealized_pnl, available_margin]
        self.account_info_shape = (3,)
        
        self.action_space = spaces.Box(
            low=np.array([-1, 0]), high=np.array([1, 1]), dtype=np.float32
        ) # [position_direction, position_size]

        # Observation space = Market Data + Account Info
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size, len(self.features) + self.account_info_shape[0]),
            dtype=np.float32
        )
        
        # --- Internal State ---
        self.current_step = 0
        self.balance = 0
        self.position = 0.0  # -1 (max short) to 1 (max long)
        self.entry_price = 0.0
        self.total_reward = 0.0
        self.equity_curve = []
        self.daily_returns = []

    def _get_obs(self):
        # --- Market Data ---
        start = self.current_step - self.window_size + 1
        end = self.current_step
        market_obs_slice = self.df.iloc[start : end + 1]
        market_obs = market_obs_slice[self.features].values.astype(np.float32)

        # --- Account Info ---
        unrealized_pnl = 0
        current_price = self.df.loc[self.current_step, 'close']
        if self.position != 0:
            pnl_pct = (current_price - self.entry_price) / self.entry_price if self.position > 0 \
                      else (self.entry_price - current_price) / self.entry_price
            unrealized_pnl = self.position * self.balance * pnl_pct
        
        available_margin = self.balance * (1 - abs(self.position))
        
        account_info = np.array([self.position, unrealized_pnl / self.balance, available_margin / self.balance], dtype=np.float32)
        
        # สร้าง Account Info ให้มีขนาดเท่ากับ market_obs (window_size, num_features)
        # โดยการ broadcast ค่าเดียวกันไปทุกๆ step ใน window
        account_obs = np.tile(account_info, (self.window_size, 1))

        # --- Combine market and account observations ---
        obs = np.concatenate((market_obs, account_obs), axis=1)
        return obs

    def _dynamic_slippage(self, order_size_pct):
        # Slippage จะมากขึ้นตามขนาดของ Order และความผันผวนของตลาด
        volatility = self.df.loc[self.current_step, 'rolling_std_20']
        base_slippage = 0.0001
        return base_slippage + (abs(order_size_pct) * volatility * 0.1) # ตัวคูณสามารถปรับได้

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size - 1
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.total_reward = 0.0
        self.equity_curve = [self.initial_balance]
        self.daily_returns = [0.0] * self.sharpe_ratio_window # khởi tạo lợi nhuận hàng ngày
        
        initial_obs = self._get_obs()
        info = {"equity": self.balance, "position": self.position}
        return initial_obs, info

    def step(self, action):
        target_direction, target_size_pct = action
        target_size_pct = np.clip(target_size_pct, 0, self.position_limit)
        target_position = target_direction * target_size_pct

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        
        current_price = self.df.loc[self.current_step, 'close']
        
        # === 2. PnL and Transaction Costs ===
        # คำนวณ PnL ของ Position ที่มีอยู่
        unrealized_pnl = 0
        if self.position != 0:
            pnl_pct = (current_price - self.entry_price) / self.entry_price if self.position > 0 \
                      else (self.entry_price - current_price) / self.entry_price
            unrealized_pnl = self.position * self.balance * pnl_pct

        # คำนวณขนาด Trade ที่เกิดขึ้นจริง
        trade_size = target_position - self.position
        
        # คำนวณต้นทุนการเทรด
        slippage = self._dynamic_slippage(trade_size)
        transaction_cost = abs(trade_size) * self.balance * (self.taker_fee + slippage)
        
        # อัปเดต Balance
        self.balance += unrealized_pnl - transaction_cost
        
        # อัปเดต Entry Price (ถัวเฉลี่ย)
        if trade_size != 0:
            if target_position == 0:
                self.entry_price = 0
            else:
                # Weighted average of old and new position price
                self.entry_price = ((self.position * self.entry_price) + (trade_size * current_price)) / target_position
        
        self.position = target_position
        
        # === 3. Smarter Reward Function ===
        # บันทึก Equity และผลตอบแทนเพื่อคำนวณ Sharpe Ratio
        self.equity_curve.append(self.balance)
        today_return = (self.balance / self.equity_curve[-2]) - 1 if len(self.equity_curve) > 1 else 0
        self.daily_returns.pop(0)
        self.daily_returns.append(today_return)

        # คำนวณ Rolling Sharpe Ratio
        if np.std(self.daily_returns) > 1e-6: # ป้องกันการหารด้วยศูนย์
            sharpe_ratio = np.mean(self.daily_returns) / np.std(self.daily_returns)
        else:
            sharpe_ratio = 0
            
        # Reward = ผลตอบแทนของ step นั้นๆ + โบนัส/บทลงโทษจาก Sharpe Ratio
        reward = today_return + (sharpe_ratio * 0.01) # ปรับ weight ของ sharpe ได้

        # บทลงโทษหากเปิดสถานะใหญ่เกินไปในสภาวะตลาดผันผวนสูง
        volatility = self.df.loc[self.current_step, 'rolling_std_20']
        if abs(self.position) * volatility > 0.1: # ตัวเลขนี้ต้องปรับจูน
            reward -= 0.01

        self.total_reward += reward

        obs = self._get_obs()
        info = {"equity": self.balance, "position": self.position, "sharpe_ratio": sharpe_ratio}
        
        return obs, reward, terminated, False, info