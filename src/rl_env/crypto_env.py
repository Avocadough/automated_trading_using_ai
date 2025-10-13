# src/env/crypto_env.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional, Literal

import gymnasium as gym
from gymnasium import spaces


ActionMode = Literal["discrete", "continuous"]


class CryptoTradingEnv(gym.Env):
    """
    Simple, SB3-ready crypto trading environment.

    Key features:
    - Action space:
        * discrete (default): 0=short, 1=flat, 2=long
        * continuous: single scalar desired position fraction in [-position_limit, +position_limit]
    - Position represented as (qty, avg_entry); PnL = qty * (price - avg_entry)
    - Observation: rolling window of features + [signed_pos_frac, unrealized_pct, free_margin_pct]
    - SB3-compatible episode info: info['episode']={'r','l'}

    Expected df columns:
        'close'  (required)
        (optional) any feature columns you specify in `features`
    Recommended minimal features (from your simple feature pipeline):
        ['log_ret_1', 'rolling_std_20']

    Parameters:
        df: DataFrame sorted by time (index can be DatetimeIndex or RangeIndex)
        features: list of feature column names (must exist in df)
        window_size: obs lookback
        initial_balance: starting cash (quote currency)
        taker_fee: proportional fee on notional of each trade side
        position_limit: max |position notional| as fraction of equity (e.g. 0.3 = 30%)
        slippage_bps: simple linear slippage in bps per side (0.0 to disable)
        reward_scale: scale step return to reward (e.g., 100 = 1% step return -> reward 1.0)
        normalize: if True, z-score features across whole df at init
        action_mode: "discrete" or "continuous"
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        window_size: int = 64,
        initial_balance: float = 10_000.0,
        taker_fee: float = 0.0005,
        position_limit: float = 0.30,
        slippage_bps: float = 0.0,
        reward_scale: float = 100.0,
        normalize: bool = True,
        action_mode: ActionMode = "discrete",
        seed: Optional[int] = None,
    ):
        super().__init__()

        # ---------- Data checks ----------
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if "close" not in df.columns:
            raise ValueError("df must contain 'close' column")
        self.df = df.dropna().reset_index(drop=True).copy()

        # Feature columns
        if features is None:
            # minimal default (must exist in df)
            default = ["log_ret_1", "rolling_std_20"]
            for c in default:
                if c not in self.df.columns:
                    raise ValueError(f"Missing feature '{c}'. Provide features or create it in df.")
            self.features = default
        else:
            missing = [c for c in features if c not in self.df.columns]
            if missing:
                raise ValueError(f"Features not found in df: {missing}")
            self.features = features

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

        # ---------- RNG ----------
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # ---------- Feature normalization (optional) ----------
        if self.normalize_features:
            self._fit_norm()

        # ---------- Spaces ----------
        # Action
        if self.action_mode == "discrete":
            # 0=short, 1=flat, 2=long
            self.action_space = spaces.Discrete(3)
        else:
            # one scalar in [-position_limit, +position_limit]
            self.action_space = spaces.Box(
                low=-self.position_limit, high=self.position_limit, shape=(1,), dtype=np.float32
            )

        # Observation = window x (n_features + 3 account info)
        self.n_acct = 3  # signed_pos_frac, unrealized_pct, free_margin_pct
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, len(self.features) + self.n_acct),
            dtype=np.float32
        )

        # Internal state placeholders
        self.current_step = 0
        self.balance = 0.0        # cash
        self.qty = 0.0            # asset quantity (+ long, - short)
        self.avg_entry = 0.0      # VWAP of open position
        self.total_reward = 0.0
        self.episode_length = 0
        self.equity = 0.0

        # Ready to go
        self.reset()

    # ============================ Helpers ============================

    def _fit_norm(self):
        """Fit mean/std for features on full df (stable z-score)."""
        feats = self.df[self.features].astype("float64")
        self._mu = feats.mean()
        self._std = feats.std().replace(0, 1.0)
        # Store normalized copy to speed up _get_obs
        self._feat_norm = ((feats - self._mu) / self._std).astype("float32").values

    def _obs_features_block(self, s: int, e: int) -> np.ndarray:
        """Return (e-s+1, n_features) normalized or raw feature window."""
        if self.normalize_features:
            # self._feat_norm is aligned to df index
            return self._feat_norm[s:e+1]
        return self.df.iloc[s:e+1][self.features].astype("float32").values

    def _account_block(self) -> np.ndarray:
        """Broadcast [signed_pos_frac, unrealized_pct, free_margin_pct] to (window_size, 3)."""
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
        obs = np.concatenate([market, acct], axis=1)
        return obs.astype(np.float32)

    def _map_action_to_target_frac(self, action) -> float:
        if self.action_mode == "discrete":
            # 0=short, 1=flat, 2=long
            discrete_to_frac = {-1: -self.position_limit, 0: 0.0, 1: self.position_limit}
            a = int(action) - 1
            return float(discrete_to_frac[a])
        # continuous
        return float(np.clip(action[0], -self.position_limit, self.position_limit))

    # ============================ Gym API ============================

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        # Start after enough history for the first window
        self.current_step = self.window_size - 1

        self.balance = self.initial_balance
        self.qty = 0.0
        self.avg_entry = 0.0
        self.total_reward = 0.0
        self.episode_length = 0

        # initial equity
        self.equity = self.initial_balance

        obs = self._build_obs()
        info = {"equity": self.equity, "position_frac": 0.0}
        return obs, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.episode_length += 1

        # 1) map action -> target position fraction
        target_frac = self._map_action_to_target_frac(action)

        # 2) move time
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1

        price = float(self.df.loc[self.current_step, "close"])

        # 3) equity before trade
        unrealized = self.qty * (price - self.avg_entry)
        equity_before = float(max(self.balance + unrealized, 1e-12))

        # 4) target qty and trade size
        desired_notional = equity_before * target_frac
        desired_qty = desired_notional / price
        trade_qty = desired_qty - self.qty
        trade_notional = trade_qty * price

        # 5) fees + simple slippage
        slippage_rate = self.slippage_bps * 1e-4  # bps to rate
        cost_rate = self.taker_fee + slippage_rate
        trading_cost = abs(trade_notional) * cost_rate

        # 6) update VWAP & qty
        if abs(trade_qty) > 0:
            if (self.qty == 0) or (np.sign(trade_qty) == np.sign(self.qty)):
                # same direction -> average in
                new_notional = abs(self.qty) * self.avg_entry + abs(trade_qty) * price
                new_qty = self.qty + trade_qty
                self.avg_entry = float(new_notional / max(1e-12, abs(new_qty)))
                self.qty = float(new_qty)
            else:
                # reducing or flipping
                new_qty = self.qty + trade_qty
                if np.sign(self.qty) != np.sign(new_qty):
                    # crossed zero -> closed then opened new side
                    self.qty = float(new_qty)
                    self.avg_entry = float(price if self.qty != 0 else 0.0)
                else:
                    # partial reduce
                    self.qty = float(new_qty)

        # 7) pay fees from cash
        self.balance -= trading_cost

        # 8) equity after
        unrealized_after = self.qty * (price - self.avg_entry)
        equity_after = float(self.balance + unrealized_after)

        # 9) reward (scaled step return)
        step_ret = (equity_after / equity_before) - 1.0
        reward = self.reward_scale * step_ret
        self.total_reward += reward
        self.equity = equity_after

        # 10) obs & info
        obs = self._build_obs()
        info = {
            "equity": self.equity,
            "position_frac": float((abs(self.qty) * price) / max(equity_after, 1e-12)) * float(np.sign(self.qty)),
            "qty": self.qty,
            "avg_entry": self.avg_entry
        }

        # SB3 episode summary
        if terminated:
            info["episode"] = {
                "r": float(self.total_reward),
                "l": int(self.episode_length),
                "total_return_pct": float((self.equity / self.initial_balance - 1.0) * 100.0),
            }

        return obs, float(reward), bool(terminated), False, info

    # ============================ Render ============================

    def render(self, mode: str = "human"):
        price = float(self.df.loc[self.current_step, "close"])
        unrealized = self.qty * (price - self.avg_entry)
        equity = float(self.balance + unrealized)
        print(f"[{self.current_step}] price={price:.2f} cash={self.balance:.2f} "
            f"qty={self.qty:.6f} avg={self.avg_entry:.2f} equity={equity:.2f}")
