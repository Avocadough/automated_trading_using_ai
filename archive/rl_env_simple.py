# src/env/rl_env_simple.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List

class SimpleTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 64,
        initial_balance: float = 10_000.0,
        taker_fee: float = 0.0005,
        position_limit: float = 0.30,   # max |pos| as fraction of equity
        reward_scale: float = 100.0,
        features: List[str] = None
    ):
        super().__init__()
        self.df = df.dropna().reset_index(drop=True)
        if "close" not in self.df.columns:
            raise ValueError("df must contain 'close'")
        self.window_size = int(window_size)
        self.initial_balance = float(initial_balance)
        self.taker_fee = float(taker_fee)
        self.position_limit = float(position_limit)
        self.reward_scale = float(reward_scale)

        default_features = ["log_ret_1", "rolling_std_20"]
        self.features = features if features is not None else [c for c in default_features if c in self.df.columns]
        for c in ["log_ret_1", "rolling_std_20"]:
            if c not in self.features:
                raise ValueError(f"Missing required feature '{c}' in df")

        # Actions: 0=short, 1=flat, 2=long
        self.action_space = spaces.Discrete(3)

        # Obs: window x (len(features) + 3 acct info)
        self.n_acct = 3  # [signed_pos_frac, unrealized_pct, free_margin_pct]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, len(self.features) + self.n_acct),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.current_step = self.window_size - 1
        self.balance = self.initial_balance  # cash
        self.qty = 0.0                       # asset quantity (+ long, - short)
        self.avg_entry = 0.0
        self.total_reward = 0.0
        self.episode_length = 0

        # record equity
        self.equity = self.initial_balance

        obs = self._get_obs()
        info = {"equity": self.equity}
        return obs, info

    def _get_obs(self) -> np.ndarray:
        s, e = self.current_step - self.window_size + 1, self.current_step
        market = self.df.iloc[s:e+1][self.features].values.astype(np.float32)

        price = float(self.df.loc[self.current_step, "close"])
        unrealized = self.qty * (price - self.avg_entry)
        equity = float(self.balance + unrealized)
        pos_frac_abs = float((abs(self.qty) * price) / max(1e-12, equity))
        signed_pos_frac = float(np.sign(self.qty) * pos_frac_abs)
        unrealized_pct = float(unrealized / max(1e-12, equity))
        free_margin_pct = float(max(0.0, 1.0 - pos_frac_abs))

        acct = np.array([signed_pos_frac, unrealized_pct, free_margin_pct], dtype=np.float32)
        acct = np.tile(acct, (self.window_size, 1))

        return np.concatenate([market, acct], axis=1)

    def step(self, action: int):
        self.episode_length += 1

        # map action to target position fraction
        target_frac = {-1: -self.position_limit, 0: 0.0, 1: self.position_limit}[int(action) - 1]

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1

        price = float(self.df.loc[self.current_step, "close"])
        # equity before trade
        unrealized = self.qty * (price - self.avg_entry)
        equity_before = float(self.balance + unrealized)
        equity_before = max(equity_before, 1e-12)

        # desired qty
        desired_notional = equity_before * target_frac
        desired_qty = desired_notional / price

        # trade to target
        trade_qty = desired_qty - self.qty
        trade_notional = trade_qty * price

        # simple fee model (no slippage for simplicity)
        trading_cost = abs(trade_notional) * self.taker_fee
        # update VWAP & qty
        if abs(trade_qty) > 0:
            if (self.qty == 0) or (np.sign(trade_qty) == np.sign(self.qty)):
                new_notional = abs(self.qty) * self.avg_entry + abs(trade_qty) * price
                new_qty = self.qty + trade_qty
                self.avg_entry = float(new_notional / max(1e-12, abs(new_qty)))
                self.qty = float(new_qty)
            else:
                new_qty = self.qty + trade_qty
                if np.sign(self.qty) != np.sign(new_qty):
                    self.qty = float(new_qty)
                    self.avg_entry = float(price if self.qty != 0 else 0.0)
                else:
                    self.qty = float(new_qty)

        # pay fee from cash
        self.balance -= trading_cost

        # equity after
        unrealized_after = self.qty * (price - self.avg_entry)
        equity_after = float(self.balance + unrealized_after)

        step_ret = (equity_after / equity_before) - 1.0
        reward = self.reward_scale * step_ret
        self.total_reward += reward
        self.equity = equity_after

        obs = self._get_obs()
        info = {"equity": self.equity, "position_frac": target_frac, "qty": self.qty, "avg_entry": self.avg_entry}

        if terminated:
            info["episode"] = {"r": float(self.total_reward), "l": int(self.episode_length)}

        return obs.astype(np.float32), float(reward), bool(terminated), False, info

    def render(self, mode="human"):
        price = float(self.df.loc[self.current_step, "close"])
        unrealized = self.qty * (price - self.avg_entry)
        equity = float(self.balance + unrealized)
        print(f"[{self.current_step}] price={price:.2f} cash={self.balance:.2f} qty={self.qty:.6f} avg={self.avg_entry:.2f} eq={equity:.2f}")
