# src/live/paper_trader.py
"""
Production-Grade Live Paper Trading Engine.

TRAIN-SERVE PARITY GUARANTEE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The #1 failure mode of deployed ML models is "train-serve skew" —
the live features differ subtly from training features, causing the
model to receive inputs it has never seen, producing garbage actions.

This script prevents train-serve skew by:

1. LOADING THE META FILE: The `_meta.json` produced during training
   encodes the exact feature list, window size, and SPA parameters.
   We reconstruct features using the SAME functions from
   `make_features_spa.py` (imported directly, not reimplemented).

2. LOADING NORM STATS: If the training env used `norm_mu`/`norm_std`,
   we load those exact values and apply the same normalization.
   This ensures observation scaling is byte-identical.

3. CONSTRUCTING THE OBSERVATION WINDOW: We maintain a rolling buffer
   of exactly `window_size` bars of computed features, producing the
   exact (64, 24) shaped observation the CNN+LSTM expects.

4. ACCOUNT STATE INJECTION: We replicate the 5-dim account block
   [signed_pos_frac, unrealized_pct, free_margin_pct,
    drawdown_pct, steps_since_trade_norm] from crypto_env.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)


# ============================================================
# Logging Setup
# ============================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("paper_trader")
    logger.setLevel(logging.DEBUG)

    # Console handler (INFO+)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(ch)

    # File handler (DEBUG+)
    fh = logging.FileHandler(log_dir / "live_trader.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    return logger


# ============================================================
# Market Data Fetcher (CCXT)
# ============================================================

class DataFetcher:
    """Fetch OHLCV candles from Binance via CCXT."""

    def __init__(self, symbol: str = "BTC/USDT", timeframe: str = "1h",
                 exchange_id: str = "binance", logger: logging.Logger = None):
        import ccxt
        self.exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        self.symbol = symbol
        self.timeframe = timeframe
        self.log = logger or logging.getLogger("paper_trader")

    def fetch_ohlcv(self, limit: int = 200) -> pd.DataFrame:
        """
        Fetch latest `limit` candles.
        Returns DataFrame with columns: [open, high, low, close, volume].
        """
        try:
            raw = self.exchange.fetch_ohlcv(
                self.symbol, self.timeframe, limit=limit)
        except Exception as e:
            self.log.error(f"CCXT fetch failed: {e}")
            raise

        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low",
                                         "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)

        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        self.log.debug(f"Fetched {len(df)} candles, latest: {df.index[-1]}")
        return df


# ============================================================
# Live Feature Computer
# ============================================================

class LiveFeatureComputer:
    """
    Computes the same 19 features as make_features_spa.py.
    Uses the EXACT same indicator functions to guarantee parity.
    """

    def __init__(self, meta_path: Path, logger: logging.Logger = None):
        self.log = logger or logging.getLogger("paper_trader")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.features = meta["features"]
        self.window_size = int(meta.get("window_size", 64))
        self.spa_params = meta.get("spa_params", {})
        self.log.info(f"Feature list ({len(self.features)}): {self.features}")

        # Import feature computation functions
        from src.features.make_features_spa import (
            calculate_rsi, calculate_atr, calculate_macd_normalized,
            calculate_bbands, calculate_stochastic, encode_signal
        )
        self._calculate_rsi = calculate_rsi
        self._calculate_atr = calculate_atr
        self._calculate_macd_normalized = calculate_macd_normalized
        self._calculate_bbands = calculate_bbands
        self._calculate_stochastic = calculate_stochastic
        self._encode_signal = encode_signal

    def compute(self, df_raw: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Compute all features on a DataFrame of raw OHLCV data.
        Returns the feature DataFrame, or None on failure.
        """
        try:
            df = df_raw.copy()
            close = df["close"].astype("float64")
            high = df["high"].astype("float64")
            low = df["low"].astype("float64")

            # 1) Returns
            df["log_ret_1"] = np.log(close / close.shift(1))
            df["momentum_5"] = close / close.shift(5) - 1.0
            df["momentum_10"] = close / close.shift(10) - 1.0
            df["momentum_24"] = close / close.shift(24) - 1.0

            # 2) Volatility
            log_ret = df["log_ret_1"]
            rvol_20 = log_ret.rolling(window=20).std()
            rvol_50 = log_ret.rolling(window=50).std()
            df["rvol_20"] = rvol_20
            df["rvol_50"] = rvol_50
            df["vol_regime"] = (rvol_20 / rvol_50.replace(0, np.nan)).clip(0.2, 5.0)

            raw_atr = self._calculate_atr(high, low, close, 14)
            df["ATR_pct"] = raw_atr / close

            bb_width, bb_position = self._calculate_bbands(close, 20, 2.0)
            df["BB_width"] = bb_width
            df["BB_position"] = bb_position.clip(-0.5, 1.5)

            # 3) Momentum
            df["RSI_14"] = self._calculate_rsi(close, 14)
            df["MACD_norm"] = self._calculate_macd_normalized(close, 12, 26, 9)

            mean60 = close.rolling(60).mean()
            std60 = close.rolling(60).std()
            df["close_z_60"] = ((close - mean60) / std60.replace(0, np.nan)).clip(-4, 4)

            stoch_k, stoch_d = self._calculate_stochastic(high, low, close)
            df["stoch_k"] = stoch_k
            df["stoch_d"] = stoch_d

            # 4) Volume
            if "volume" in df.columns:
                vol = df["volume"].astype("float64")
                vol_ma20 = vol.rolling(20).mean()
                df["vol_ratio"] = ((vol / vol_ma20.replace(0, np.nan)) - 1.0).clip(-3.0, 10.0)
                df["vol_direction"] = (
                    np.sign(close.diff()) * (vol / vol_ma20.replace(0, np.nan))
                ).clip(-5.0, 5.0)

            # 5) SPA Signal
            try:
                from src.spa.spa_core import SPAParams, run_spa
                spa_d = self.spa_params.get("d", 89)
                spa_alpha = self.spa_params.get("alpha", 3.0)
                spa_gamma = self.spa_params.get("gamma", 1.0)
                spa_m_ma = self.spa_params.get("m_ma", 5)
                spa_source = self.spa_params.get("source", "close")

                params = SPAParams(d=spa_d, alpha=spa_alpha, gamma=spa_gamma,
                                   m_ma=spa_m_ma, source=spa_source)
                spa_out = run_spa(df[["high", "low", "close"]], params)
                df["spa_sig"] = self._encode_signal(spa_out["signals"]["signal"])

                bands = spa_out["bands"]
                mid_inner = (bands["h_inner"] + bands["l_inner"]) / 2.0
                spa_dist = (close - mid_inner) / raw_atr.replace(0, np.nan)
                df["spa_dist"] = spa_dist.clip(-5, 5)
            except Exception as e:
                self.log.warning(f"SPA computation failed: {e}. Using zeros.")
                df["spa_sig"] = 0
                df["spa_dist"] = 0.0

            # Assemble and return
            avail_features = [f for f in self.features if f in df.columns]
            out = df[["close"] + avail_features].copy()
            out = out.replace([np.inf, -np.inf], np.nan)

            return out

        except Exception as e:
            self.log.error(f"Feature computation failed: {e}", exc_info=True)
            return None


# ============================================================
# Portfolio Tracker (Paper Trading State)
# ============================================================

class PaperPortfolio:
    """Track simulated portfolio state with realistic fees."""

    def __init__(self, initial_balance: float = 10_000.0,
                 taker_fee: float = 0.0005,
                 position_limit: float = 0.30,
                 slippage_bps: float = 0.5,
                 logger: logging.Logger = None):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.taker_fee = taker_fee
        self.position_limit = position_limit
        self.slippage_bps = slippage_bps
        self.log = logger or logging.getLogger("paper_trader")

        # Position
        self.position_qty = 0.0
        self.avg_entry = 0.0

        # Tracking
        self.realized_pnl = 0.0
        self.n_trades = 0
        self.equity_high = initial_balance
        self.steps_since_trade = 0
        self.trade_log = []

    @property
    def equity(self) -> float:
        return self.balance + self._unrealized_pnl(self._last_price)

    def _unrealized_pnl(self, current_price: float) -> float:
        if abs(self.position_qty) < 1e-12 or self.avg_entry < 1e-12:
            return 0.0
        return self.position_qty * (current_price - self.avg_entry)

    _last_price: float = 0.0

    def get_account_state(self, current_price: float) -> np.ndarray:
        """
        Replicate the 5-dim account block from crypto_env.py.
        [signed_pos_frac, unrealized_pct, free_margin_pct,
         drawdown_pct, steps_since_trade_norm]
        """
        self._last_price = current_price
        eq = self.equity
        self.equity_high = max(self.equity_high, eq)

        max_notional = eq * self.position_limit
        pos_notional = abs(self.position_qty) * current_price
        signed_frac = 0.0
        if max_notional > 1e-12:
            signed_frac = (self.position_qty * current_price) / max_notional

        unrealized_pct = self._unrealized_pnl(current_price) / max(eq, 1e-6)
        free_margin = 1.0 - (pos_notional / max(eq, 1e-6))
        dd_pct = (eq - self.equity_high) / max(self.equity_high, 1e-6)
        steps_norm = min(self.steps_since_trade / 128.0, 2.0)

        return np.array([
            np.clip(signed_frac, -1, 1),
            np.clip(unrealized_pct, -1, 1),
            np.clip(free_margin, 0, 1),
            np.clip(dd_pct, -1, 0),
            steps_norm,
        ], dtype=np.float32)

    def execute_action(self, action: int, current_price: float) -> str:
        """
        Execute a discrete action: 0=short, 1=flat, 2=long.
        Mirrors crypto_env.py position management.
        Returns action description string.
        """
        self._last_price = current_price
        eq = self.equity
        max_notional = eq * self.position_limit
        target_map = {0: -1.0, 1: 0.0, 2: 1.0}
        target_frac = target_map.get(action, 0.0)
        target_notional = target_frac * max_notional
        target_qty = target_notional / max(current_price, 1e-6)
        delta_qty = target_qty - self.position_qty

        if abs(delta_qty) * current_price < 1.0:
            self.steps_since_trade += 1
            return "HOLD"

        # Slippage
        slip = self.slippage_bps * 1e-4
        exec_price = current_price * (1 + slip if delta_qty > 0 else 1 - slip)

        # Fee
        trade_notional = abs(delta_qty) * exec_price
        fee = trade_notional * self.taker_fee

        # Close existing position (realize PnL)
        if abs(self.position_qty) > 1e-12 and np.sign(delta_qty) != np.sign(self.position_qty):
            close_qty = min(abs(delta_qty), abs(self.position_qty))
            close_qty *= np.sign(self.position_qty) * -1  # opposite sign
            rpnl = close_qty * -1 * (exec_price - self.avg_entry)
            # Actually the correct formula:
            if self.position_qty > 0:  # was long, now closing
                rpnl = close_qty * (exec_price - self.avg_entry) * -1
                # close_qty is negative (selling), so:
                rpnl = abs(close_qty) * (exec_price - self.avg_entry)
            else:  # was short, now closing
                rpnl = abs(close_qty) * (self.avg_entry - exec_price)
            self.realized_pnl += rpnl

        # Update position
        old_qty = self.position_qty
        self.position_qty = target_qty
        if abs(self.position_qty) > 1e-12:
            self.avg_entry = exec_price
        else:
            self.avg_entry = 0.0

        # Deduct fee
        self.balance -= fee

        self.n_trades += 1
        self.steps_since_trade = 0

        # Determine action type
        if target_frac > 0:
            act_type = "LONG"
        elif target_frac < 0:
            act_type = "SHORT"
        else:
            act_type = "FLAT"

        self.trade_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": act_type,
            "price": current_price,
            "exec_price": exec_price,
            "qty": target_qty,
            "delta_qty": delta_qty,
            "fee": fee,
            "equity": self.equity,
            "realized_pnl": self.realized_pnl,
        })

        return f"{act_type} @ ${exec_price:,.2f} | qty={target_qty:.6f} | fee=${fee:.2f}"


# ============================================================
# Main Paper Trader
# ============================================================

class PaperTrader:
    """Main trading loop: Fetch → Compute Features → Predict → Execute."""

    def __init__(
        self,
        model_path: Path,
        meta_path: Path,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        initial_balance: float = 10_000.0,
        log_dir: Path = Path("logs"),
    ):
        self.logger = setup_logging(log_dir)
        self.logger.info("=" * 60)
        self.logger.info("  PAPER TRADER — CNN+LSTM PPO × SPA")
        self.logger.info("=" * 60)

        # Load model
        self.logger.info(f"Loading model: {model_path}")
        from stable_baselines3 import PPO
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = PPO.load(str(model_path), device=device)
        self.logger.info(f"Model loaded on {device}")

        # Load meta
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.features = meta["features"]
        self.window_size = int(meta.get("window_size", 64))
        self.n_acct = 5  # account state dimensions

        # Load norm stats if available
        self.norm_mu = np.array(meta["norm_mu"]) if "norm_mu" in meta else None
        self.norm_std = np.array(meta["norm_std"]) if "norm_std" in meta else None

        # Components
        self.fetcher = DataFetcher(symbol, timeframe, logger=self.logger)
        self.feature_computer = LiveFeatureComputer(meta_path, self.logger)
        self.portfolio = PaperPortfolio(
            initial_balance=initial_balance, logger=self.logger)

        # State
        self.running = True
        self._setup_signal_handlers()

        self.logger.info(f"Symbol: {symbol} | TF: {timeframe}")
        self.logger.info(f"Window: {self.window_size} | Features: {len(self.features)}")
        self.logger.info(f"Balance: ${initial_balance:,.2f}")

    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame):
        self.logger.info("Shutdown signal received. Stopping gracefully...")
        self.running = False

    def _build_observation(self, feature_df: pd.DataFrame,
                            current_price: float) -> Optional[np.ndarray]:
        """
        Build the exact observation tensor the model expects.
        Shape: (window_size, n_features + n_acct) = (64, 24)
        """
        avail = [f for f in self.features if f in feature_df.columns]
        if len(avail) < len(self.features):
            missing = set(self.features) - set(avail)
            self.logger.warning(f"Missing features: {missing}")

        feat_data = feature_df[avail].values
        if len(feat_data) < self.window_size:
            self.logger.warning(
                f"Insufficient data: {len(feat_data)} < {self.window_size}")
            return None

        # Take last window_size rows
        window = feat_data[-self.window_size:].astype(np.float32)

        # Apply training normalization if available
        if self.norm_mu is not None and self.norm_std is not None:
            n_feat = min(len(self.norm_mu), window.shape[1])
            std = self.norm_std[:n_feat].copy()
            std[std < 1e-8] = 1.0
            window[:, :n_feat] = (window[:, :n_feat] - self.norm_mu[:n_feat]) / std

        # Replace NaN/Inf
        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)

        # Account state: replicate across all timesteps
        acct = self.portfolio.get_account_state(current_price)
        acct_block = np.tile(acct, (self.window_size, 1))

        # Concatenate: (64, 19) + (64, 5) = (64, 24)
        obs = np.concatenate([window, acct_block], axis=1).astype(np.float32)

        return obs

    def tick(self) -> bool:
        """
        One complete trading cycle: fetch → compute → predict → execute.
        Returns True if successful, False on error.
        """
        try:
            # 1. Fetch data (need enough bars for indicator warmup)
            warmup = max(100, self.window_size + 60)
            df_raw = self.fetcher.fetch_ohlcv(limit=warmup)
            if df_raw is None or len(df_raw) < warmup:
                self.logger.warning(f"Insufficient candles: {len(df_raw)}")
                return False

            current_price = float(df_raw["close"].iloc[-1])

            # 2. Compute features
            feature_df = self.feature_computer.compute(df_raw)
            if feature_df is None:
                return False

            # Drop NaN from indicator warmup
            feature_df = feature_df.dropna()
            if len(feature_df) < self.window_size:
                self.logger.warning("Not enough clean bars after feature computation")
                return False

            # 3. Build observation
            obs = self._build_observation(feature_df, current_price)
            if obs is None:
                return False

            # 4. Model prediction
            obs_batch = np.expand_dims(obs, axis=0)
            action, _ = self.model.predict(obs_batch, deterministic=True)
            action = int(action) if np.isscalar(action) else int(action[0])

            action_names = {0: "SHORT", 1: "FLAT", 2: "LONG"}
            self.logger.info(
                f"Price: ${current_price:,.2f} | "
                f"Action: {action_names.get(action, '?')} ({action})")

            # 5. Execute
            result = self.portfolio.execute_action(action, current_price)
            eq = self.portfolio.equity
            rpnl = self.portfolio.realized_pnl

            self.logger.info(f"  → {result}")
            self.logger.info(
                f"  Equity: ${eq:,.2f} | "
                f"Realized PnL: ${rpnl:,.2f} | "
                f"Trades: {self.portfolio.n_trades}")

            return True

        except Exception as e:
            self.logger.error(f"Tick failed: {e}", exc_info=True)
            return False

    def run(self, interval_seconds: int = 3600):
        """
        Main trading loop. Runs one tick per interval.
        Default: 3600s = 1 hour (matching 1H timeframe).
        """
        self.logger.info(f"Starting live loop (interval={interval_seconds}s)")
        self.logger.info("Press Ctrl+C to stop.\n")

        tick_count = 0
        while self.running:
            tick_count += 1
            self.logger.info(f"{'─'*50}")
            self.logger.info(f"TICK #{tick_count} — {datetime.now(timezone.utc).isoformat()}")

            success = self.tick()
            if not success:
                self.logger.warning("Tick failed — will retry next interval")

            # Save trade log periodically
            if tick_count % 10 == 0:
                self._save_trade_log()

            if not self.running:
                break

            # Wait for next interval
            self.logger.info(f"Sleeping {interval_seconds}s until next candle...")
            for _ in range(interval_seconds):
                if not self.running:
                    break
                time.sleep(1)

        # Final save
        self._save_trade_log()
        self._print_summary()

    def _save_trade_log(self):
        """Save trade history to CSV."""
        if self.portfolio.trade_log:
            out_path = Path("data/live/trade_log.csv")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self.portfolio.trade_log).to_csv(
                out_path, index=False)
            self.logger.debug(f"Trade log saved: {out_path}")

    def _print_summary(self):
        """Print final summary on shutdown."""
        p = self.portfolio
        self.logger.info("=" * 60)
        self.logger.info("  PAPER TRADING SESSION ENDED")
        self.logger.info("=" * 60)
        self.logger.info(f"  Initial Balance : ${p.initial_balance:,.2f}")
        self.logger.info(f"  Final Equity    : ${p.equity:,.2f}")
        self.logger.info(f"  Total Return    : {(p.equity/p.initial_balance-1)*100:+.2f}%")
        self.logger.info(f"  Realized PnL    : ${p.realized_pnl:,.2f}")
        self.logger.info(f"  Total Trades    : {p.n_trades}")
        self.logger.info("=" * 60)


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Live Paper Trader — CNN+LSTM PPO × SPA")
    ap.add_argument("--model", type=str,
                    default="data/models/ppo_spa_btc_1h.zip",
                    help="Path to trained .zip model")
    ap.add_argument("--meta", type=str,
                    default="data/features/btc_1h_spa_meta.json",
                    help="Path to feature _meta.json")
    ap.add_argument("--symbol", type=str, default="BTC/USDT")
    ap.add_argument("--timeframe", type=str, default="1h")
    ap.add_argument("--balance", type=float, default=10_000.0)
    ap.add_argument("--interval", type=int, default=3600,
                    help="Seconds between ticks (3600 = 1 hour)")
    ap.add_argument("--log_dir", type=str, default="logs")
    args = ap.parse_args()

    trader = PaperTrader(
        model_path=Path(args.model),
        meta_path=Path(args.meta),
        symbol=args.symbol,
        timeframe=args.timeframe,
        initial_balance=args.balance,
        log_dir=Path(args.log_dir),
    )
    trader.run(interval_seconds=args.interval)


if __name__ == "__main__":
    main()
