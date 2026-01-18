# src/rl_env/crypto_env.py
from __future__ import annotations

from typing import Tuple, Dict, List, Optional, Literal
import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

ActionMode = Literal["discrete", "continuous"]


class CryptoTradingEnv(gym.Env):
    """
    SB3-ready crypto env with execution smoothing:
      - deadband on notional change
      - minimum hold steps
      - cooldown after each trade
    Observation = window of features + [signed_pos_frac, unrealized_pct, free_margin_pct]
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        features: List[str],
        window_size: int = 64,
        initial_balance: float = 10_000.0,
        taker_fee: float = 0.0005,
        position_limit: float = 0.30,
        slippage_bps: float = 0.0,
        reward_scale: float = 100.0,
        normalize: bool = True,
        action_mode: ActionMode = "discrete",
        seed: Optional[int] = None,
        # ---- shaping knobs (สามารถตั้ง 0 เพื่อปิด) ----
        flat_penalty_bps: float = 0.0,
        inactivity_steps: int = 256,
        inactivity_penalty_bps: float = 0.0,
        turnover_reward_coeff: float = 0.0,
        trade_threshold: float = 0.02,
        # ---- execution smoothing ----
        deadband_frac: float = 0.25,  # ไม่ขยับถ้า |delta_notional|/equity < 25%
        min_hold_steps: int = 2,     # ต้องถืออย่างน้อย 64 แท่งก่อนยอมเปลี่ยน
        cooldown_steps: int = 1,     # หลังเทรด คูลดาวน์อีก 16 แท่ง
    ):
        super().__init__()

        # ---------- Data checks ----------
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if "close" not in df.columns:
            raise ValueError("df must contain 'close' column")
        if features is None or len(features) == 0:
            raise ValueError("features must be a non-empty list")
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise ValueError(f"Features not found in df: {missing}")

        self.df = df.dropna().reset_index(drop=True).copy()
        self.features = list(features)
        self.window_size = int(window_size)
        if len(self.df) < self.window_size + 2:
            raise ValueError("Not enough rows in df for the specified window_size")

        # ---------- Trading params ----------
        self.initial_balance = float(initial_balance)
        self.taker_fee = float(taker_fee)
        self.position_limit = float(position_limit)
        self.slippage_bps = float(slippage_bps)
        self.reward_scale = float(reward_scale)
        self.normalize_features = bool(normalize)
        self.action_mode = action_mode

        # ---------- Shaping knobs ----------
        self.flat_penalty_bps = float(flat_penalty_bps)
        self.inactivity_steps = int(inactivity_steps)
        self.inactivity_penalty_bps = float(inactivity_penalty_bps)
        self.turnover_reward_coeff = float(turnover_reward_coeff)
        self.trade_threshold = float(trade_threshold)

        # ---------- Execution smoothing ----------
        self.deadband_frac = float(deadband_frac)
        self.min_hold_steps = int(min_hold_steps)
        self.cooldown_steps = int(cooldown_steps)

        # ---------- RNG ----------
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # ---------- Feature normalization ----------
        if self.normalize_features:
            self._fit_norm()

        # ---------- Spaces ----------
        if self.action_mode == "discrete":
            # 0=short, 1=flat, 2=long
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Box(
                low=-self.position_limit, high=self.position_limit, shape=(1,), dtype=np.float32
            )

        self.n_acct = 3  # signed_pos_frac, unrealized_pct, free_margin_pct
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, len(self.features) + self.n_acct),
            dtype=np.float32
        )

        # ---------- Internal state ----------
        self.current_step = 0
        self.balance = 0.0
        self.qty = 0.0
        self.avg_entry = 0.0
        self.total_reward = 0.0
        self.episode_length = 0
        self.equity = 0.0

        # tracking
        self.last_trade_step = 0
        self.last_trade_price = 0.0

        self.reset()

    # ============================ Helpers ============================

    def _fit_norm(self):
        feats = self.df[self.features].astype("float64")
        self._mu = feats.mean()
        self._std = feats.std().replace(0, 1.0)
        self._feat_norm = ((feats - self._mu) / self._std).astype("float32").values

    def _obs_features_block(self, s: int, e: int) -> np.ndarray:
        if self.normalize_features:
            return self._feat_norm[s:e+1]
        return self.df.iloc[s:e+1][self.features].astype("float32").values

    def _account_block(self) -> np.ndarray:
        price = float(self.df.loc[self.current_step, "close"])
        unrealized = self.qty * (price - self.avg_entry)
        equity = float(max(self.balance + unrealized, 1e-12))

        pos_frac_abs = float((abs(self.qty) * price) / equity)
        signed_pos_frac = float(np.sign(self.qty) * pos_frac_abs)
        unrealized_pct = float(unrealized / equity)
        free_margin_pct = float(max(0.0, 1.0 - pos_frac_abs))

        acct_row = np.array([signed_pos_frac, unrealized_pct, free_margin_pct], dtype=np.float32)
        return np.tile(acct_row, (self.window_size, 1))

    def _build_obs(self) -> np.ndarray:
        s = self.current_step - self.window_size + 1
        e = self.current_step
        market = self._obs_features_block(s, e)
        acct = self._account_block()
        return np.concatenate([market, acct], axis=1).astype(np.float32)

    def _map_action_to_target_frac(self, action) -> float:
        if self.action_mode == "discrete":
            # 0=short, 1=flat, 2=long
            a = int(action) - 1
            table = {-1: -self.position_limit, 0: 0.0, 1: self.position_limit}
            return float(table[a])
        return float(np.clip(action[0], -self.position_limit, self.position_limit))

    # ============================ Gym API ============================

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.current_step = self.window_size - 1
        self.balance = self.initial_balance
        self.qty = 0.0
        self.avg_entry = 0.0
        self.total_reward = 0.0
        self.episode_length = 0
        self.equity = self.initial_balance
        self.last_trade_step = self.current_step
        self.last_trade_price = float(self.df.loc[self.current_step, "close"])
        obs = self._build_obs()
        info = {"equity": self.equity, "position_frac": 0.0}
        return obs, info

    def step(self, action):
        self.episode_length += 1
        # time
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        price = float(self.df.loc[self.current_step, "close"])

        # equity before
        unrealized = self.qty * (price - self.avg_entry)
        equity_before = float(max(self.balance + unrealized, 1e-12))

        # desired target from action
        target_frac = self._map_action_to_target_frac(action)
        desired_notional = equity_before * target_frac
        desired_qty = desired_notional / price
        raw_trade_qty = desired_qty - self.qty
        raw_trade_notional = raw_trade_qty * price

        # ---- Execution smoothing gates ----
        steps_since_trade = self.current_step - self.last_trade_step

        # (a) cooldown
        blocked_by_cooldown = (steps_since_trade < self.cooldown_steps)

        # (b) min-hold (บังคับถือให้ถึงก่อน)
        blocked_by_min_hold = (steps_since_trade < self.min_hold_steps)

        # (c) deadband (ignore trade if too tiny)
        change_frac = abs(raw_trade_notional) / max(1e-12, equity_before)
        blocked_by_deadband = (change_frac < self.deadband_frac)

        block_trade = blocked_by_cooldown or blocked_by_min_hold or blocked_by_deadband
        trade_qty = 0.0 if block_trade else raw_trade_qty
        trade_notional = trade_qty * price

        # fees + slippage
        slippage_rate = self.slippage_bps * 1e-4
        cost_rate = self.taker_fee + slippage_rate
        trading_cost = abs(trade_notional) * cost_rate

        # update VWAP & qty (only if we actually trade)
        if abs(trade_qty) > 0:
            if (self.qty == 0) or (np.sign(trade_qty) == np.sign(self.qty)):
                new_notional = abs(self.qty) * self.avg_entry + abs(trade_qty) * price
                new_qty = self.qty + trade_qty
                self.avg_entry = float(new_notional / max(1e-12, abs(new_qty)))
                self.qty = float(new_qty)
            else:
                new_qty = self.qty + trade_qty
                if np.sign(self.qty) != np.sign(new_qty):
                    # crossed zero: closed then opened
                    self.qty = float(new_qty)
                    self.avg_entry = float(price if self.qty != 0 else 0.0)
                else:
                    self.qty = float(new_qty)

            self.last_trade_step = self.current_step
            self.last_trade_price = price

        # pay fees
        self.balance -= trading_cost

        # equity after
        unrealized_after = self.qty * (price - self.avg_entry)
        equity_after = float(self.balance + unrealized_after)

        # base reward
        step_ret = (equity_after / equity_before) - 1.0
        reward = self.reward_scale * step_ret

        # ---- Optional shaping (kept but default=0) ----
        if self.flat_penalty_bps > 0 and abs(self.qty) < 1e-12:
            reward -= self.reward_scale * (self.flat_penalty_bps * 1e-4)

        turnover = abs(trade_notional) / max(1e-12, equity_before)
        if turnover > self.trade_threshold and self.turnover_reward_coeff > 0.0:
            reward += self.reward_scale * (self.turnover_reward_coeff * turnover)

        if (self.current_step - self.last_trade_step) > self.inactivity_steps and self.inactivity_penalty_bps > 0:
            reward -= self.reward_scale * (self.inactivity_penalty_bps * 1e-4)
            self.last_trade_step = self.current_step  # avoid repeated hits

        self.total_reward += reward
        self.equity = equity_after

        obs = self._build_obs()
        info = {
            "equity": self.equity,
            "position_frac": float((abs(self.qty) * price) / max(equity_after, 1e-12)) * float(np.sign(self.qty)),
            "qty": self.qty,
            "avg_entry": self.avg_entry,
        }
        if terminated:
            info["episode"] = {
                "r": float(self.total_reward),
                "l": int(self.episode_length),
                "total_return_pct": float((self.equity / self.initial_balance - 1.0) * 100.0),
            }
        return obs, float(reward), bool(terminated), False, info

    def render(self, mode: str = "human"):
        price = float(self.df.loc[self.current_step, "close"])
        unrealized = self.qty * (price - self.avg_entry)
        equity = float(self.balance + unrealized)
        print(f"[{self.current_step}] price={price:.2f} cash={self.balance:.2f} "
              f"qty={self.qty:.6f} avg={self.avg_entry:.2f} equity={equity:.2f}")
