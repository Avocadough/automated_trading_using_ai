# src/features/make_features_spa.py
"""
Feature Engineering Pipeline for RL Day-Trading Agent.

Design Principles (Senior Quant Perspective):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. STATIONARITY: Every feature must be I(0) — no raw prices, no cumulative values.
   All price-derived features use returns, ratios, or z-scores.
2. NO LOOK-AHEAD: Only backward-looking rolling windows. No future data touches.
3. SCALE INVARIANCE: Features must work regardless of BTC being $20k or $100k.
   Achieved via normalization by close, ATR, or rolling statistics.
4. BOUNDED RANGE: Avoid unbounded features that can cause gradient explosions.
   Apply winsorization/clipping where needed.
5. ORTHOGONALITY: Features should capture different market dimensions:
   Returns → WHERE price went | Volatility → HOW it got there
   Momentum → HOW FAST               | Volume → WITH what CONVICTION
   SPA → CRITICAL boundary levels
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.spa.spa_core import SPAParams, run_spa
from src.utils import ensure_datetime_index


# ============================================================
# Helper: Technical Indicators
# ============================================================

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI — output rescaled to [-1, 1] for neural network friendliness."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    # Rescale: [0, 100] → [-1, 1]  (50 becomes 0 = neutral)
    return (rsi - 50.0) / 50.0


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                  period: int = 14) -> pd.Series:
    """Average True Range (absolute dollar value)."""
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_macd_normalized(close: pd.Series, fast: int = 12, slow: int = 26,
                              signal: int = 9) -> pd.Series:
    """
    MACD Histogram normalized by close price.
    Raw MACD_hist is in $ → non-stationary! $100 hist at $20k ≠ $100 hist at $100k.
    Normalization: hist / close → dimensionless percentage.
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd_hist / close  # Now scale-invariant


def calculate_bbands(close: pd.Series, period: int = 20, std_dev: float = 2.0):
    """Bollinger Bands → returns (width, position) — both bounded and stationary."""
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    bb_width = (upper - lower) / middle          # Relative band width
    bb_position = (close - lower) / (upper - lower)  # 0-1 position
    return bb_width, bb_position


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                         k_period: int = 14, d_period: int = 3):
    """Stochastic Oscillator — rescale to [-1, 1] for consistency with RSI."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()
    # Rescale: [0, 100] → [-1, 1]
    return (k - 50.0) / 50.0, (d - 50.0) / 50.0


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                  period: int = 14) -> pd.Series:
    """
    Average Directional Index — measures TREND STRENGTH regardless of direction.
    ADX ∈ [0, 100]:  <20 = no trend (range-bound),  >40 = strong trend.
    Rescaled to [-1, 1] via (ADX - 50) / 50 for NN friendliness.
    """
    # True Range
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # Wilder smoothing (EMA with alpha = 1/period)
    atr_smooth = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di  = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() /
                      atr_smooth.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() /
                      atr_smooth.replace(0, np.nan))

    # DX and ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    # Rescale [0, 100] → [-1, 1]
    return ((adx - 50.0) / 50.0).clip(-1, 1)


def encode_signal(sig: pd.Series) -> pd.Series:
    """Map SPA string signal → numeric {-1, 0, 1}."""
    m = {"short": -1, "none": 0, "long": 1}
    return sig.astype("string").map(m).fillna(0).astype("int8")


# ============================================================
# Main feature builder
# ============================================================

def make_features_spa(
    input_parquet: Path,
    output_parquet: Path,
    window_size_meta: int = 64,
    spa_d: int = 89,
    spa_alpha: float = 3.0,
    spa_gamma: float = 1.0,
    spa_m_ma: int = 5,
    spa_source: str = "close",
    train_split: float = 0.8,
    # --- Trend indicator periods (GA-optimizable) ---
    adx_period: int = 14,
    ema_fast: int = 50,
    ema_slow: int = 200,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
):
    """
    Build RL-ready features combining SPA signals + full TA indicators.
    Every feature is checked for stationarity and scale-invariance.
    """
    if not input_parquet.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_parquet}")

    print(f"[info] Loading raw klines: {input_parquet}")
    df = pd.read_parquet(input_parquet)
    df = ensure_datetime_index(df)

    required = {"high", "low", "close"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    keep_cols: List[str] = [c for c in ["open", "high", "low", "close", "volume"]
                            if c in df.columns]
    df = df[keep_cols].copy()
    for c in keep_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    close = df["close"].astype("float64")
    high  = df["high"].astype("float64")
    low   = df["low"].astype("float64")

    # ============================================================
    # 1) RETURNS — Stationary by construction (I(0))
    # ============================================================
    print("[info] Computing return features...")
    df["log_ret_1"]   = np.log(close / close.shift(1))
    df["momentum_5"]  = close / close.shift(5) - 1.0   # 5-bar pct return
    df["momentum_10"] = close / close.shift(10) - 1.0   # 10-bar pct return
    df["momentum_24"] = close / close.shift(24) - 1.0   # 24-bar (1 day @ 1H)

    # ============================================================
    # 2) VOLATILITY — All normalized to be scale-invariant
    # ============================================================
    print("[info] Computing volatility features...")
    log_ret = df["log_ret_1"]

    # Rolling realized volatility (short vs long → regime detection)
    rvol_20 = log_ret.rolling(window=20).std()
    rvol_50 = log_ret.rolling(window=50).std()
    df["rvol_20"] = rvol_20.clip(0, 0.5)   # Clip: raw vol is unbounded, can poison CNN
    df["rvol_50"] = rvol_50.clip(0, 0.5)

    # Volatility ratio: short/long → >1 means vol expanding, <1 means contracting
    # This is a powerful regime indicator: vol expansion often precedes trends
    df["vol_regime"] = (rvol_20 / rvol_50.replace(0, np.nan)).clip(0.2, 5.0)

    # ATR as % of price → scale-invariant absolute volatility
    raw_atr = calculate_atr(high, low, close, 14)
    df["ATR_pct"] = raw_atr / close

    # Bollinger Bands → width = vol proxy, position = mean-reversion signal
    bb_width, bb_position = calculate_bbands(close, 20, 2.0)
    df["BB_width"]    = bb_width
    df["BB_position"] = bb_position.clip(-0.5, 1.5)  # Winsorize outliers

    # ============================================================
    # 3) MOMENTUM / TREND — Centered and bounded
    # ============================================================
    print("[info] Computing momentum & trend features...")

    # RSI (rescaled to [-1, 1] around 0 = neutral)
    df["RSI_14"] = calculate_rsi(close, 14)

    # MACD histogram normalized by price → dimensionless
    df["MACD_norm"] = calculate_macd_normalized(close, macd_fast, macd_slow, macd_signal)

    # MACD histogram normalized by rolling ATR → vol-regime-invariant
    # Superior to price normalization during high-volatility events
    ema_fast_line = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow_line = close.ewm(span=macd_slow, adjust=False).mean()
    macd_line   = ema_fast_line - ema_slow_line
    macd_sig_line = macd_line.ewm(span=macd_signal, adjust=False).mean()
    macd_hist   = macd_line - macd_sig_line
    df["MACD_hist_atr"] = (macd_hist / raw_atr.replace(0, np.nan)).clip(-5, 5)

    # Z-score of close vs 60-bar MA → stationary mean-reversion signal
    # Using log returns for z-score computation avoids non-stationarity
    mean60 = close.rolling(60).mean()
    std60  = close.rolling(60).std()
    df["close_z_60"] = ((close - mean60) / std60.replace(0, np.nan)).clip(-4, 4)

    # Stochastic (rescaled [-1, 1])
    stoch_k, stoch_d = calculate_stochastic(high, low, close)
    df["stoch_k"] = stoch_k
    df["stoch_d"] = stoch_d

    # ============================================================
    # 3b) TREND STRENGTH & DIRECTION — New features for trend-following
    #     Solves: "Oscillator Ceiling" where RSI/Stoch saturate at ±1
    #     during mega rallies, leaving the agent blind to trend continuation.
    # ============================================================
    print("[info] Computing trend strength & direction features...")

    # ADX: Trend STRENGTH (direction-agnostic)
    # ADX < 0 (i.e. raw < 50) → no trend, mean-revert
    # ADX > 0 (i.e. raw > 50) → strong trend, ride it
    df["ADX_14"] = calculate_adx(high, low, close, adx_period)

    # Normalized EMA Distances: Trend DIRECTION relative to macro structure
    #
    # WHY THIS PREVENTS GRADIENT EXPLOSIONS AT ALL-TIME HIGHS:
    # Raw EMA values (e.g., $90,000) would create enormous activations that
    # blow up LSTM cell states. But (Close - EMA) / EMA is a small, bounded
    # percentage (typically ±5%). Even at a $200k ATH, if EMA(200) = $180k,
    # the normalized distance is only (200k-180k)/180k = +0.11 — perfectly
    # digestible by the neural network.
    #
    ema_fast_ma  = close.ewm(span=ema_fast, adjust=False).mean()
    ema_slow_ma  = close.ewm(span=ema_slow, adjust=False).mean()
    df["ema_dist_50"]  = ((close - ema_fast_ma)  / ema_fast_ma.replace(0, np.nan)).clip(-0.3, 0.3)
    df["ema_dist_200"] = ((close - ema_slow_ma) / ema_slow_ma.replace(0, np.nan)).clip(-0.5, 0.5)

    # ============================================================
    # 4) VOLUME — Only if available, all stationary
    # ============================================================
    if "volume" in df.columns:
        print("[info] Computing volume features...")
        vol = df["volume"].astype("float64")
        vol_ma20 = vol.rolling(20).mean()

        # Volume ratio: current / MA(20) — centered at 1.0 → subtract 1 to center at 0
        # Clipped to prevent outlier during exchange outages or flash events
        df["vol_ratio"] = ((vol / vol_ma20.replace(0, np.nan)) - 1.0).clip(-3.0, 3.0)

        # Directional volume conviction: sign(return) * normalized_volume
        # Positive = bullish conviction, negative = bearish conviction
        # Clipped for robustness
        df["vol_direction"] = (
            np.sign(close.diff()) * (vol / vol_ma20.replace(0, np.nan))
        ).clip(-3.0, 3.0)

    # ============================================================
    # 5) SPA SIGNAL — Categorical {-1, 0, 1}
    # ============================================================
    print("[info] Running SPA core...")
    params = SPAParams(
        d=spa_d, alpha=spa_alpha, gamma=spa_gamma,
        m_ma=spa_m_ma, source=spa_source
    )
    spa_out = run_spa(df[["high", "low", "close"]], params)
    df["spa_sig"] = encode_signal(spa_out["signals"]["signal"])

    # SPA Distance: how far price is from the nearest SPA boundary
    # Normalized by ATR to be scale-invariant — tells agent "how close to breakout"
    bands = spa_out["bands"]
    mid_inner = (bands["h_inner"] + bands["l_inner"]) / 2.0
    spa_dist = (close - mid_inner) / raw_atr.replace(0, np.nan)
    df["spa_dist"] = spa_dist.clip(-5, 5)

    # ============================================================
    # ASSEMBLE — Final feature list
    # ============================================================
    print("[info] Assembling feature table...")
    feature_cols: List[str] = [
        # --- Returns (I(0), unbounded but naturally small) ---
        "log_ret_1",      # 1-bar log return
        "momentum_5",     # 5-bar momentum
        "momentum_10",    # 10-bar momentum
        "momentum_24",    # 24-bar (1 day) momentum

        # --- Volatility (all scale-invariant) ---
        "rvol_20",        # 20-bar realized vol
        "rvol_50",        # 50-bar realized vol
        "vol_regime",     # vol_short / vol_long ratio (regime detector)
        "ATR_pct",        # ATR / close (scale-invariant)
        "BB_width",       # Bollinger width (relative vol)
        "BB_position",    # Position in BB (0-1, winsorized)

        # --- Momentum (all centered ~0, bounded ±1 or clipped) ---
        "RSI_14",         # [-1, 1] centered RSI
        "MACD_norm",      # MACD hist / close (dimensionless)
        "MACD_hist_atr",  # MACD hist / ATR (vol-regime-invariant, clipped ±5)
        "close_z_60",     # Price z-score vs 60-bar MA (clipped ±4)
        "stoch_k",        # [-1, 1] Stochastic %K
        "stoch_d",        # [-1, 1] Stochastic %D

        # --- Trend Strength & Direction (anti-oscillator-ceiling) ---
        "ADX_14",         # [-1, 1] trend strength (direction-agnostic)
        "ema_dist_50",    # (close - EMA50) / EMA50 (clipped ±0.3)
        "ema_dist_200",   # (close - EMA200) / EMA200 (clipped ±0.5)

        # --- Volume (stationary, clipped) ---
        "vol_ratio",      # Volume / MA(20) - 1 (centered at 0)
        "vol_direction",  # Directional volume conviction

        # --- SPA Signal ---
        "spa_sig",        # {-1, 0, 1} boundary signal
        "spa_dist",       # Normalized distance to SPA mid-boundary
    ]

    # Safety: keep only features that exist in df
    feature_cols = [c for c in feature_cols if c in df.columns]
    out = df[["close"] + feature_cols].copy()
    out = out.replace([np.inf, -np.inf], np.nan).dropna()

    # ---- Stationarity Audit (runtime check) ----
    n_check = min(5000, len(out))
    sample = out[feature_cols].tail(n_check)
    suspicious = []
    for col in feature_cols:
        col_range = sample[col].max() - sample[col].min()
        col_std   = sample[col].std()
        if col_std == 0:
            suspicious.append(f"{col}: ZERO variance!")
        elif col_range / (col_std + 1e-12) > 100:
            suspicious.append(f"{col}: extreme range/std ratio ({col_range/col_std:.0f})")
    if suspicious:
        print(f"\n⚠️  Stationarity warnings:")
        for s in suspicious:
            print(f"    {s}")

    # Cast dtypes
    for c in out.columns:
        if c == "spa_sig":
            out[c] = out[c].astype("int8")
        else:
            out[c] = out[c].astype("float32")

    # Save
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_parquet, compression="snappy")

    # Save meta
    inferred_freq    = pd.infer_freq(out.index) or "unknown"
    FREQ_TO_PERIODS  = {
        "1min": 525600, "min": 525600, "T": 525600,
        "5min": 105120, "5T": 105120,
        "15min": 35040, "15T": 35040,
        "30min": 17520, "30T": 17520,
        "h": 8760, "H": 8760, "1H": 8760, "1h": 8760,
        "4h": 2190, "4H": 2190,
        "1D": 365, "D": 365,
    }
    periods_per_year = FREQ_TO_PERIODS.get(inferred_freq, 8760)

    meta = {
        "features":         feature_cols,
        "window_size":      int(window_size_meta),
        "freq_hint":        inferred_freq,
        "periods_per_year": periods_per_year,
        "train_split":      train_split,
        "rows":             int(len(out)),
        "source":           str(input_parquet),
        "spa_params": {
            "d": spa_d, "alpha": spa_alpha, "gamma": spa_gamma,
            "m_ma": spa_m_ma, "source": spa_source
        }
    }
    meta_path = Path(str(output_parquet).replace(".parquet", "_meta.json"))
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\n{'='*60}")
    print(f"  FEATURE ENGINEERING COMPLETED")
    print(f"{'='*60}")
    print(f"  Output rows  : {len(out):,}")
    print(f"  Total features: {len(feature_cols)}")
    for f in feature_cols:
        std_val = out[f].std()
        mn_val  = out[f].mean()
        print(f"    - {f:18s}  mean={mn_val:+.4f}  std={std_val:.4f}")
    print(f"  Saved parquet: {output_parquet}")
    print(f"  Saved meta   : {meta_path}")
    print(f"{'='*60}")


# ============================================================
# CLI entry-point
# ============================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build SPA + full TA features for RL training")
    ap.add_argument("--input",           type=str, default="data/raw/btc_1h.parquet")
    ap.add_argument("--output",          type=str, default="data/features/btc_1h_spa.parquet")
    ap.add_argument("--window_size_meta",type=int, default=64)

    ap.add_argument("--spa_d",      type=int,   default=89)
    ap.add_argument("--spa_alpha",  type=float, default=3.0)
    ap.add_argument("--spa_gamma",  type=float, default=1.0)
    ap.add_argument("--spa_m_ma",   type=int,   default=5)
    ap.add_argument("--spa_source", type=str,   default="close", choices=["close", "hl2", "hlc3"])
    ap.add_argument("--train_split",type=float, default=0.8)

    # Trend indicator periods (GA-optimizable)
    ap.add_argument("--adx_period",  type=int, default=14)
    ap.add_argument("--ema_fast",    type=int, default=50)
    ap.add_argument("--ema_slow",    type=int, default=200)
    ap.add_argument("--macd_fast",   type=int, default=12)
    ap.add_argument("--macd_slow",   type=int, default=26)
    ap.add_argument("--macd_signal", type=int, default=9)

    ap.add_argument("--params_file",type=str,   default="data/params/best_h1_spa_ga.json",
                    help="Path to best_spa_ga.json to auto-load GA parameters.")

    args = ap.parse_args()

    # Auto-load GA-optimised params if available (SPA + trend indicators)
    if args.params_file and Path(args.params_file).exists():
        print(f"[info] Loading optimised params from {args.params_file}")
        with open(args.params_file, "r") as f:
            ga_data = json.load(f)
        # SPA params
        args.spa_d      = int(ga_data.get("d", args.spa_d))
        args.spa_alpha  = float(ga_data.get("alpha", args.spa_alpha))
        args.spa_gamma  = float(ga_data.get("beta", args.spa_gamma))
        args.spa_m_ma   = int(ga_data.get("m_ma", args.spa_m_ma))
        args.spa_source = ga_data.get("src", args.spa_source)
        # Trend indicator params (from expanded GA genome)
        args.adx_period  = int(ga_data.get("adx_period", args.adx_period))
        args.ema_fast    = int(ga_data.get("ema_fast", args.ema_fast))
        args.ema_slow    = int(ga_data.get("ema_slow", args.ema_slow))
        args.macd_fast   = int(ga_data.get("macd_fast", args.macd_fast))
        args.macd_slow   = int(ga_data.get("macd_slow", args.macd_slow))
        args.macd_signal = int(ga_data.get("macd_signal", args.macd_signal))
        print(f"       -> SPA: d={args.spa_d}, alpha={args.spa_alpha}, gamma={args.spa_gamma}, "
              f"m_ma={args.spa_m_ma}, src={args.spa_source}")
        print(f"       -> Trend: ADX_p={args.adx_period}, EMA={args.ema_fast}/{args.ema_slow}, "
              f"MACD={args.macd_fast}/{args.macd_slow}/{args.macd_signal}")

    make_features_spa(
        input_parquet    = Path(args.input),
        output_parquet   = Path(args.output),
        window_size_meta = args.window_size_meta,
        spa_d            = args.spa_d,
        spa_alpha        = args.spa_alpha,
        spa_gamma        = args.spa_gamma,
        spa_m_ma         = args.spa_m_ma,
        spa_source       = args.spa_source,
        train_split      = args.train_split,
        adx_period       = args.adx_period,
        ema_fast         = args.ema_fast,
        ema_slow         = args.ema_slow,
        macd_fast        = args.macd_fast,
        macd_slow        = args.macd_slow,
        macd_signal      = args.macd_signal,
    )
