# src/features/make_features_spa.py
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
# Helper: Technical Indicators  (ported from make_features.py)
# ============================================================

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def calculate_bbands(close: pd.Series, period: int = 20, std_dev: float = 2.0):
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    bb_width = (upper - lower) / middle
    bb_position = (close - lower) / (upper - lower)
    return upper, middle, lower, bb_width, bb_position


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                          k_period: int = 14, d_period: int = 3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()
    return k, d


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (np.sign(close.diff()) * volume).fillna(0).cumsum()


def encode_signal(sig: pd.Series) -> pd.Series:
    """Map SPA string signal -> numeric {-1, 0, 1}."""
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
    train_split: float = 0.8
):
    """
    Build RL-ready features combining SPA signals + full TA indicators.

    Required columns in input parquet: open, high, low, close  (volume optional)
    Output:
      - parquet with columns: ['close'] + all features
      - meta json: {features, window_size, rows, source, params}
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

    # Keep only OHLCV columns
    keep_cols: List[str] = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep_cols].copy()
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ========================
    # 1) PRICE & RETURNS
    # ========================
    print("[info] Computing price & return features...")
    df["ret_1"]       = df["close"].pct_change()
    df["log_ret_1"]   = np.log(df["close"] / df["close"].shift(1))
    df["momentum_5"]  = df["close"] / df["close"].shift(5) - 1
    df["momentum_10"] = df["close"] / df["close"].shift(10) - 1

    # ========================
    # 2) VOLATILITY
    # ========================
    print("[info] Computing volatility features...")
    df["rolling_std_20"] = df["log_ret_1"].rolling(window=20).std()
    df["rolling_std_50"] = df["log_ret_1"].rolling(window=50).std()
    df["ATR_14"]         = calculate_atr(df["high"], df["low"], df["close"], 14)

    bb_u, bb_m, bb_l, bb_w, bb_pos = calculate_bbands(df["close"], 20, 2.0)
    df["BB_width"]    = bb_w
    df["BB_position"] = bb_pos

    # ========================
    # 3) MOMENTUM / TREND
    # ========================
    print("[info] Computing momentum & trend features...")
    df["RSI_14"] = calculate_rsi(df["close"], 14)

    macd, macd_signal, macd_hist = calculate_macd(df["close"], 12, 26, 9)
    df["MACD"]      = macd
    df["MACD_hist"] = macd_hist

    mean60 = df["close"].rolling(60).mean()
    std60  = df["close"].rolling(60).std()
    df["close_z_60"] = (df["close"] - mean60) / (std60.replace(0, np.nan))

    # ========================
    # 4) STOCHASTIC
    # ========================
    stoch_k, stoch_d = calculate_stochastic(df["high"], df["low"], df["close"])
    df["stoch_k"] = stoch_k
    df["stoch_d"] = stoch_d

    # ========================
    # 5) OBV  (if volume present) — log-differenced for stationarity
    # ========================
    if "volume" in df.columns:
        print("[info] Computing OBV feature...")
        raw_obv = calculate_obv(df["close"], df["volume"])
        # แปลง OBV เป็น log-diff เพื่อให้ stationary (ค่าในช่วง [-1, 1])
        df["OBV"] = np.log1p(raw_obv.abs()) * np.sign(raw_obv)
        df["OBV"] = df["OBV"].diff().fillna(0.0)

    # ========================
    # 6) SPA SIGNAL
    # ========================
    print("[info] Running SPA core...")
    params = SPAParams(
        d=spa_d,
        alpha=spa_alpha,
        gamma=spa_gamma,
        m_ma=spa_m_ma,
        source=spa_source
    )
    spa_out = run_spa(df[["high", "low", "close"]], params)
    spa_sig_num = encode_signal(spa_out["signals"]["signal"]).rename("spa_sig_num")
    df["spa_sig_num"] = spa_sig_num

    # ========================
    # CLEANUP & ASSEMBLE
    # ========================
    print("[info] Assembling feature table...")
    feature_cols: List[str] = [
        "log_ret_1",      # stationary price return
        "rolling_std_20", # volatility regime
        "RSI_14",         # momentum / overbought-oversold
        "BB_position",    # price location in band (naturally 0-1)
        "MACD_hist",      # trend momentum
        "spa_sig_num",    # SPA buy/sell signal
    ]

    # keep only existing cols (safety)
    feature_cols = [c for c in feature_cols if c in df.columns]
    out = df[["close"] + feature_cols].copy()
    out = out.replace([np.inf, -np.inf], np.nan).dropna()

    # Cast dtypes
    for c in out.columns:
        if c == "spa_sig_num":
            out[c] = out[c].astype("int8")
        else:
            out[c] = out[c].astype("float32")

    # Save parquet
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_parquet, compression="snappy")

    # Save meta
    inferred_freq    = pd.infer_freq(out.index) or "unknown"
    FREQ_TO_PERIODS  = {"1min": 525600, "5min": 105120, "15min": 35040, "1H": 8760, "4H": 2190, "1D": 365}
    periods_per_year = FREQ_TO_PERIODS.get(inferred_freq, 35040)

    meta = {
        "features":        feature_cols,
        "window_size":     int(window_size_meta),
        "freq_hint":       inferred_freq,
        "periods_per_year": periods_per_year,
        "train_split":     train_split,
        "rows":            int(len(out)),
        "source":          str(input_parquet),
        "spa_params": {
            "d":      spa_d,
            "alpha":  spa_alpha,
            "gamma":  spa_gamma,
            "m_ma":   spa_m_ma,
            "source": spa_source
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
        print(f"    - {f}")
    print(f"  Saved parquet: {output_parquet}")
    print(f"  Saved meta   : {meta_path}")
    print(f"{'='*60}")


# ============================================================
# CLI entry-point
# ============================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build SPA + full TA features for RL training")
    ap.add_argument("--input",           type=str, default="data/raw/btc_15m.parquet")
    ap.add_argument("--output",          type=str, default="data/features/btc_15m_spa.parquet")
    ap.add_argument("--window_size_meta",type=int, default=64)

    # SPA params
    ap.add_argument("--spa_d",      type=int,   default=89)
    ap.add_argument("--spa_alpha",  type=float, default=3.0)
    ap.add_argument("--spa_gamma",  type=float, default=1.0)
    ap.add_argument("--spa_m_ma",   type=int,   default=5)
    ap.add_argument("--spa_source", type=str,   default="close", choices=["close", "hl2", "hlc3"])
    ap.add_argument("--train_split",type=float, default=0.8)
    ap.add_argument("--params_file",type=str,   default="data/params/best_spa_ga.json",
                    help="Path to best_spa_ga.json to auto-load GA parameters.")

    args = ap.parse_args()

    # Auto-load GA-optimised SPA params if available
    if args.params_file and Path(args.params_file).exists():
        print(f"[info] Loading optimised SPA params from {args.params_file}")
        with open(args.params_file, "r") as f:
            ga_data = json.load(f)
        # best_spa_ga.json stores keys at top level: n, alpha, d, beta, src
        args.spa_d      = ga_data.get("d",     args.spa_d)
        args.spa_alpha  = ga_data.get("alpha", args.spa_alpha)
        args.spa_gamma  = ga_data.get("beta",  args.spa_gamma)   # GA 'beta' → SPA 'gamma'
        args.spa_m_ma   = ga_data.get("m_ma",  args.spa_m_ma)
        args.spa_source = ga_data.get("src",   args.spa_source)
        print(f"       -> d={args.spa_d}, alpha={args.spa_alpha}, gamma={args.spa_gamma}, src={args.spa_source}")

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
    )
