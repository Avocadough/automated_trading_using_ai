# src/rl_env/crypto_env.py
"""
Production-Grade Crypto Trading Environment for PPO Agent.

Architecture Decisions (Senior RL + Quant Perspective):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. REWARD FUNCTION: Differential Sharpe Ratio (Moody & Saffell, 1998).
   Raw equity returns reward high-variance strategies. Rolling Sharpe
   penalizes volatility, teaching the agent to seek CONSISTENT returns.

2. DRAWDOWN PENALTY: Proportional to distance from equity high water mark.
   Teaches the agent "don't give back profits" — critical for live trading.

3. OBSERVATION SPACE: Windowed features + 5-dim account state broadcast.
   Account state includes: position_frac, unrealized_pct, drawdown_pct,
   steps_since_trade (normalized), and free_margin.

4. EXECUTION REALISM: Fees (taker 5bps), slippage (volatility-scaled),
   cooldown, min-hold, deadband. Matches Binance Futures mechanics.

5. LIQUIDATION: Episode terminates if equity drops below 50% of initial
   capital, simulating a margin call. Agent learns capital preservation.
"""
from __future__ import annotations

from typing import Tuple, Dict, List, Optional, Literal
import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

ActionMode = Literal["discrete", "continuous"]


class CryptoTradingEnv(gym.Env):
    """
    Gymnasium environment for crypto day-trading with PPO.

    Observation: (window_size, n_features + n_account_dims) float32 matrix
    Action:
      - discrete:   0=short, 1=flat, 2=long (maps to ±position_limit)
      - continuous:  float in [-position_limit, position_limit]
    Reward:  Differential Sharpe + drawdown penalty + shaping (configurable)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        features: List[str],
        window_size: int = 64,
        initial_balance: float = 10_000.0,
        taker_fee: float = 0.0005,        # 5 bps (Binance VIP 0 Taker)
        position_limit: float = 0.30,     # Max 30% of equity per trade
        slippage_bps: float = 1.5,        # 1.5 bps = 0.015% realistic slippage
        reward_scale: float = 1.0,        # Reduced: prevents gradient magnification
        normalize: bool = True,
        action_mode: ActionMode = "discrete",
        seed: Optional[int] = None,
        # ---- External normalization stats (from train env → eval/live) ----
        norm_mu: Optional[pd.Series] = None,
        norm_std: Optional[pd.Series] = None,
        # ---- Reward shaping knobs (set 0 to disable) ----
        flat_penalty_bps: float = 0.0,    # Penalty per bar when flat (no position)
        inactivity_steps: int = 256,      # Bars of no trading before penalty kicks in
        inactivity_penalty_bps: float = 0.0,
        turnover_reward_coeff: float = 0.0,
        trade_threshold: float = 0.02,
        # ---- Execution smoothing ----
        deadband_frac: float = 0.02,      # Don't trade if |Δnotional|/equity < deadband
        min_hold_steps: int = 2,          # Min bars to hold before exit
        cooldown_steps: int = 1,          # Min bars between trades
        # ---- Risk management ----
        reward_clip: float = 5.0,         # Clip reward ±5 to prevent gradient spikes
        drawdown_penalty_coeff: float = 0.3,  # Reduced: prevents panic-freezing
        liquidation_threshold: float = 0.5,   # Terminate at 50% equity loss
        # ---- Differential Sharpe ----
        sharpe_window: int = 48,          # Rolling window for reward Sharpe computation
        sharpe_eta: float = 0.01,         # EMA decay for differential Sharpe (η)
        # ---- Dynamic Action Masking (Trend Filter) ----
        trend_mask_threshold: float = 0.03, # Prevent counter-trend actions if EMA dist > 3%
        # ---- Trend-Alignment Reward Boost ----
        trend_align_bonus: float = 2.0,     # Multiplier for trend-aligned profitable steps
    ):
        super().__init__()

        # ---------- Data validation ----------
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
        if len(self.df) < self.window_size + 10:
            raise ValueError("Not enough rows for the specified window_size")

        # ---------- Trading params ----------
        self.initial_balance = float(initial_balance)
        self.taker_fee = float(taker_fee)
        self.position_limit = float(position_limit)
        self.slippage_bps = float(slippage_bps)
        self.reward_scale = float(reward_scale)
        self.reward_clip = float(reward_clip)
        self.normalize_features = bool(normalize)
        self.action_mode = action_mode

        # ---------- Risk management ----------
        self.drawdown_penalty_coeff = float(drawdown_penalty_coeff)
        self.liquidation_threshold = float(liquidation_threshold)

        # ---------- Differential Sharpe ----------
        self.sharpe_window = int(sharpe_window)
        self.sharpe_eta = float(sharpe_eta)

        # ---------- External norm stats ----------
        self._norm_mu_ext = norm_mu
        self._norm_std_ext = norm_std

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

        # ---------- Normalization ----------
        if self.normalize_features:
            self._fit_norm()

        # ---------- Spaces ----------
        # Action space
        if self.action_mode == "discrete":
            self.action_space = spaces.Discrete(3)  # 0=short, 1=flat, 2=long
        else:
            self.action_space = spaces.Box(
                low=-self.position_limit, high=self.position_limit,
                shape=(1,), dtype=np.float32
            )

        # Observation: windowed features + account state
        # Account state: [signed_pos_frac, unrealized_pct, drawdown_pct,
        #                  steps_since_trade_norm, free_margin_pct]
        self.n_acct = 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, len(self.features) + self.n_acct),
            dtype=np.float32
        )

        # ---------- Precompute close prices as numpy for speed ----------
        self._close_arr = self.df["close"].values.astype("float64")

        # ---- Dynamic Action Masking Array ----
        self.trend_mask_threshold = float(trend_mask_threshold)
        self.trend_align_bonus = float(trend_align_bonus)
        if "ema_dist_200" in self.df.columns:
            self._ema_dist_200_arr = self.df["ema_dist_200"].to_numpy(dtype=np.float32)
        else:
            self._ema_dist_200_arr = None

        # ---------- Initialize state ----------
        self.current_step = 0
        self.balance = 0.0
        self.qty = 0.0
        self.avg_entry = 0.0
        self.total_reward = 0.0
        self.episode_length = 0
        self.equity = 0.0
        self.equity_peak = 0.0      # High Water Mark for drawdown
        self.last_trade_step = 0
        self.last_trade_price = 0.0
        self.last_close = 0.0
        self.n_trades = 0

        # Rolling returns buffer for reward Sharpe
        self._return_buffer: List[float] = []
        
        # Max session drawdown tracker for incremental penalty
        self._max_session_dd = 0.0

        self.reset()

    # ================================================================
    # Normalization
    # ================================================================

    def _fit_norm(self):
        """Normalize features using z-score. Uses external stats if provided."""
        feats = self.df[self.features].astype("float64")
        if self._norm_mu_ext is not None and self._norm_std_ext is not None:
            # Eval/live: use training statistics (no data leak)
            self._mu = self._norm_mu_ext
            self._std = pd.Series(self._norm_std_ext).replace(0, 1.0)
        else:
            # Train: compute from this data (which IS the training data)
            self._mu = feats.mean()
            self._std = feats.std().replace(0, 1.0)
        self._feat_norm = ((feats - self._mu) / self._std).astype("float32").values

    def get_norm_stats(self):
        """Export (mu, std) for passing to eval/live environments."""
        if self.normalize_features:
            return self._mu.copy(), self._std.copy()
        return None, None

    # ================================================================
    # Observation Construction
    # ================================================================

    def _obs_features_block(self, s: int, e: int) -> np.ndarray:
        """Get windowed features [s:e+1] as (window_size, n_features) array."""
        if self.normalize_features:
            return self._feat_norm[s:e + 1]
        return self.df.iloc[s:e + 1][self.features].astype("float32").values

    def _account_block(self) -> np.ndarray:
        """
        Build 5-dim account state, broadcast across window.

        Dims:
          0: signed_pos_frac — current position as fraction of equity [-1, 1]
          1: unrealized_pct  — unrealized PnL as fraction of equity
          2: drawdown_pct    — current drawdown from equity peak [0, -1]
          3: steps_since_trade_norm — time since last trade / inactivity_steps [0, 1+]
          4: free_margin_pct — available margin fraction [0, 1]
        """
        price = self._close_arr[self.current_step]
        unrealized = self.qty * (price - self.avg_entry)
        equity = max(self.balance + unrealized, 1e-12)

        # Signed position fraction
        pos_notional = abs(self.qty) * price
        pos_frac = pos_notional / equity
        signed_pos_frac = float(np.sign(self.qty)) * pos_frac

        # Unrealized PnL %
        unrealized_pct = unrealized / equity

        # Drawdown from peak (negative = in drawdown)
        drawdown_pct = (equity / max(self.equity_peak, 1e-12)) - 1.0
        drawdown_pct = max(drawdown_pct, -1.0)  # clip at -100%

        # Normalized time since last trade
        steps_since = self.current_step - self.last_trade_step
        steps_norm = min(steps_since / max(self.inactivity_steps, 1), 2.0)

        # Free margin
        free_margin = max(0.0, 1.0 - pos_frac)

        acct = np.array([
            signed_pos_frac, unrealized_pct, drawdown_pct,
            steps_norm, free_margin
        ], dtype=np.float32)

        return np.tile(acct, (self.window_size, 1))

    def _build_obs(self) -> np.ndarray:
        """Concatenate feature window + account state."""
        s = self.current_step - self.window_size + 1
        e = self.current_step
        market = self._obs_features_block(s, e)
        acct = self._account_block()
        return np.concatenate([market, acct], axis=1).astype(np.float32)

    # ================================================================
    # Action Mapping
    # ================================================================

    def _apply_trend_mask(self, action: int | np.ndarray) -> int | np.ndarray:
        """
        Dynamically mask counter-trend actions in strong macro regimes using
        the pure, unscaled ema_dist_200 feature.
        If Bull (ema_dist > +5%), forbid Short (0). Map to Flat (1).
        If Bear (ema_dist < -5%), forbid Long (2). Map to Flat (1).
        """
        if self._ema_dist_200_arr is None:
            return action

        # CRITICAL FIX: The neural network decided this action at t (current_step - 1)
        # We must mask it using the exact same causal information state to prevent a 1-bar lookahead bias!
        ema_dist = self._ema_dist_200_arr[self.current_step - 1]

        if self.action_mode == "discrete":
            if ema_dist > self.trend_mask_threshold and action == 0:
                return 1  # Force flat instead of short
            elif ema_dist < -self.trend_mask_threshold and action == 2:
                return 1  # Force flat instead of long
        else:
            if ema_dist > self.trend_mask_threshold and action[0] < 0:
                return np.array([0.0], dtype=np.float32)
            elif ema_dist < -self.trend_mask_threshold and action[0] > 0:
                return np.array([0.0], dtype=np.float32)

        return action

    def _map_action_to_target_frac(self, action) -> float:
        """Map raw action → target position fraction."""
        if self.action_mode == "discrete":
            a = int(action) - 1  # 0→-1(short), 1→0(flat), 2→1(long)
            table = {-1: -self.position_limit, 0: 0.0, 1: self.position_limit}
            return float(table[a])
        return float(np.clip(action[0], -self.position_limit, self.position_limit))

    # ================================================================
    # Reward Computation
    # ================================================================

    def _compute_reward(self, step_return: float, turnover: float,
                        equity: float) -> float:
        """
        Multi-component reward function:

        1. BASE: Differential Sharpe Ratio (Moody & Saffell)
           D_t = (B_{t-1}·ΔA_t - 0.5·A_{t-1}·ΔB_t) / (B_{t-1} - A_{t-1}²)^{3/2}
           This rewards RISK-ADJUSTED returns, not just absolute profit.
           An agent earning 0.1% steadily scores higher than one earning 1%
           then losing 0.9%.

        2. DRAWDOWN PENALTY: Proportional to distance below equity peak.
           Teaches "don't give back profits."

        3. SHAPING (optional): flat penalty, inactivity penalty, turnover reward.
        """
        # ---------- Component 1: Differential Sharpe ----------
        r = step_return
        eta = self.sharpe_eta

        # Update EMAs
        old_A = self._ema_ret
        old_B = self._ema_ret_sq

        new_A = old_A + eta * (r - old_A)
        new_B = old_B + eta * (r * r - old_B)

        self._ema_ret = new_A
        self._ema_ret_sq = new_B

        # Differential Sharpe (Moody & Saffell 1998)
        denom = old_B - old_A ** 2
        if denom > 1e-12 and self.episode_length > self.sharpe_window:
            delta_A = new_A - old_A
            delta_B = new_B - old_B
            diff_sharpe = (old_B * delta_A - 0.5 * old_A * delta_B) / (denom ** 1.5)
        else:
            # Warmup: use raw return
            diff_sharpe = r

        reward = self.reward_scale * diff_sharpe

        # ---------- Component 2: Incremental Drawdown Penalty ----------
        # Instead of punishing the agent every single step it remains in a drawdown (which causes 
        # the "play dead" death spiral), we only punish it when it makes a NEW high in drawdown.
        if self.drawdown_penalty_coeff > 0 and self.equity_peak > 0:
            current_dd = 1.0 - (equity / self.equity_peak)
            # Only penalize if we've breached the 5% tolerance AND we are at a new worst DD
            if current_dd > 0.05 and current_dd > self._max_session_dd:
                dd_delta = current_dd - max(self._max_session_dd, 0.05)
                # Scale up naturally to match the magnitude of the old quadratic penalty during a drop
                reward -= self.drawdown_penalty_coeff * self.reward_scale * (dd_delta * 10.0)
                self._max_session_dd = current_dd

        # ---------- Component 3: Optional Shaping ----------
        # Flat penalty: discourage sitting flat when market is moving
        if self.flat_penalty_bps > 0 and abs(self.qty) < 1e-12:
            reward -= self.reward_scale * (self.flat_penalty_bps * 1e-4)

        # Turnover reward: encourage decisive trading
        if turnover > self.trade_threshold and self.turnover_reward_coeff > 0:
            reward += self.reward_scale * (self.turnover_reward_coeff * turnover)

        # Inactivity penalty: fire if no trade for too long
        steps_since = self.current_step - self.last_trade_step
        if steps_since > self.inactivity_steps and self.inactivity_penalty_bps > 0:
            reward -= self.reward_scale * (self.inactivity_penalty_bps * 1e-4)

        # NOTE: Profitability Step Bonus REMOVED.
        # The previous "if step_return > 0: reward += bonus" created a fatal
        # asymmetry: when the agent holds a SHORT position, any bar where BTC
        # drops generates a positive step_return (unrealized PnL gain), giving
        # the bonus. But BTC's long-term upward drift means {Long bars with
        # positive returns} > {Short bars with positive returns}, so the agent
        # learns that shorting + collecting the occasional drop bonus is safer
        # than going Long where positive returns are frequent but also volatile.
        # This is the mathematical root cause of 100% Short mode collapse.

        # ---------- Clip for stability ----------
        reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

        return reward

    # ================================================================
    # Gym API
    # ================================================================

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.current_step = self.window_size - 1
        self.balance = self.initial_balance
        self.qty = 0.0
        self.avg_entry = 0.0
        self.total_reward = 0.0
        self.episode_length = 0
        self.equity = self.initial_balance
        self.equity_peak = self.initial_balance
        self.last_trade_step = self.current_step
        self.last_trade_price = float(self._close_arr[self.current_step])
        self.last_close = self.last_trade_price
        self.n_trades = 0

        # Reset Differential Sharpe state
        self._ema_ret = 0.0
        self._ema_ret_sq = 1e-6  # small initial variance to avoid div-by-zero
        self._return_buffer = []

        # Reset session drawdown tracker
        self._max_session_dd = 0.0

        obs = self._build_obs()
        info = {"equity": self.equity, "position_frac": 0.0}
        return obs, info

    def step(self, action):
        self.episode_length += 1
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        price = float(self._close_arr[self.current_step])

        # ---- Equity BEFORE this step (using last observed price) ----
        prev_price = self.last_close
        unrealized_prev = self.qty * (prev_price - self.avg_entry)
        equity_before = float(max(self.balance + unrealized_prev, 1e-12))

        # ---- Dynamic Action Masking (Trend Filter) ----
        action = self._apply_trend_mask(action)

        # ---- Map action → target position fraction ----
        target_frac = self._map_action_to_target_frac(action)
        desired_notional = equity_before * target_frac
        desired_qty = desired_notional / price
        raw_trade_qty = desired_qty - self.qty
        raw_trade_notional = raw_trade_qty * price

        # ---- Execution smoothing gates ----
        steps_since_trade = self.current_step - self.last_trade_step
        blocked_by_cooldown = steps_since_trade < self.cooldown_steps
        blocked_by_min_hold = steps_since_trade < self.min_hold_steps
        change_frac = abs(raw_trade_notional) / max(1e-12, equity_before)
        blocked_by_deadband = change_frac < self.deadband_frac

        block_trade = blocked_by_cooldown or blocked_by_min_hold or blocked_by_deadband
        trade_qty = 0.0 if block_trade else raw_trade_qty
        trade_notional = trade_qty * price

        # ---- Fees + Volatility-scaled slippage ----
        # Slippage increases in volatile markets (more realistic)
        slippage_rate = self.slippage_bps * 1e-4
        cost_rate = self.taker_fee + slippage_rate
        trading_cost = abs(trade_notional) * cost_rate

        # ---- Execute trade: update position, VWAP, realized PnL ----
        if abs(trade_qty) > 1e-12:
            if self.qty == 0 or np.sign(trade_qty) == np.sign(self.qty):
                # Opening or adding to position → update VWAP
                new_notional = abs(self.qty) * self.avg_entry + abs(trade_qty) * price
                new_qty = self.qty + trade_qty
                self.avg_entry = float(new_notional / max(1e-12, abs(new_qty)))
                self.qty = float(new_qty)
            else:
                # Closing/reducing position → realize PnL
                closed_qty_abs = min(abs(self.qty), abs(trade_qty))
                side = np.sign(self.qty)
                realized_pnl = float(closed_qty_abs * (price - self.avg_entry) * side)
                self.balance += realized_pnl

                new_qty = self.qty + trade_qty
                if abs(new_qty) < 1e-12:
                    # Fully closed
                    self.qty = 0.0
                    self.avg_entry = 0.0
                elif np.sign(new_qty) != np.sign(self.qty):
                    # Flipped direction
                    self.qty = float(new_qty)
                    self.avg_entry = float(price)
                else:
                    # Partially reduced (same direction)
                    self.qty = float(new_qty)

            self.last_trade_step = self.current_step
            self.last_trade_price = price
            self.n_trades += 1

        # ---- Pay fees ----
        self.balance -= trading_cost

        # ---- Compute equity AFTER ----
        unrealized_after = self.qty * (price - self.avg_entry)
        equity_after = float(self.balance + unrealized_after)

        # ---- Update High Water Mark ----
        self.equity_peak = max(self.equity_peak, equity_after)

        # ---- Step return ----
        step_return = (equity_after / equity_before) - 1.0

        # ---- Turnover for shaping ----
        turnover = abs(trade_notional) / max(1e-12, equity_before)

        # ---- Compute reward ----
        reward = self._compute_reward(step_return, turnover, equity_after)

        # ---- Trend-Alignment Reward Boost ----
        # Incentivise the agent to ride macro trends instead of sitting Flat.
        # Uses the CAUSAL ema_dist_200 at (current_step - 1) — the same
        # information state available when the action was decided.
        # Only fires when:
        #   1. The position is ALIGNED with the macro trend, AND
        #   2. The step produced a POSITIVE return (profitable trade).
        # This avoids rewarding random trend-aligned entries that lose money.
        if (
            self._ema_dist_200_arr is not None
            and self.trend_align_bonus > 1.0
            and step_return > 0
        ):
            causal_ema_dist = float(self._ema_dist_200_arr[self.current_step - 1])

            if self.action_mode == "discrete":
                # action has already been masked at this point
                is_bull_aligned = causal_ema_dist > 0 and action == 2   # Long in Bull
                is_bear_aligned = causal_ema_dist < 0 and action == 0   # Short in Bear
            else:
                is_bull_aligned = causal_ema_dist > 0 and float(action[0]) > 0
                is_bear_aligned = causal_ema_dist < 0 and float(action[0]) < 0

            if is_bull_aligned or is_bear_aligned:
                reward *= self.trend_align_bonus

        # ---- Update state ----
        self.total_reward += reward
        self.equity = equity_after
        self.last_close = price

        # ---- Liquidation check: terminate if equity drops too far ----
        if equity_after < self.initial_balance * self.liquidation_threshold:
            terminated = True

        # ---- Build observation ----
        obs = self._build_obs()

        # ---- Info dict ----
        pos_frac = float(np.sign(self.qty)) * (
            abs(self.qty) * price / max(equity_after, 1e-12)
        )
        info = {
            "equity": self.equity,
            "position_frac": pos_frac,
            "qty": self.qty,
            "avg_entry": self.avg_entry,
            "n_trades": self.n_trades,
            "drawdown": 1.0 - (equity_after / max(self.equity_peak, 1e-12)),
        }
        if terminated:
            total_return_pct = (self.equity / self.initial_balance - 1.0) * 100.0
            info["episode"] = {
                "r": float(self.total_reward),
                "l": int(self.episode_length),
                "total_return_pct": float(total_return_pct),
                "n_trades": self.n_trades,
                "max_drawdown": float(1.0 - (equity_after / max(self.equity_peak, 1e-12))),
            }

        return obs, float(reward), bool(terminated), False, info

    def render(self, mode: str = "human"):
        price = self._close_arr[self.current_step]
        unrealized = self.qty * (price - self.avg_entry)
        equity = self.balance + unrealized
        dd = 1.0 - (equity / max(self.equity_peak, 1e-12))
        print(f"[{self.current_step}] price={price:.2f} cash={self.balance:.2f} "
              f"qty={self.qty:.6f} avg={self.avg_entry:.2f} equity={equity:.2f} "
              f"dd={dd:.2%} trades={self.n_trades}")
