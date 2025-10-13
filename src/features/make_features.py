# src/features/create_features.py
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DatetimeIndex (UTC), sorted, no dup."""
    if not isinstance(df.index, pd.DatetimeIndex):
        for c in ['timestamp', 'time', 'open_time', 'date', 'datetime']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], utc=True, errors='coerce')
                df = df.set_index(c)
                break
        else:
            # fallback: try index
            df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
    df = df[~df.index.duplicated(keep='last')].sort_index()
    return df

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

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

def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()
    return k, d

def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (np.sign(close.diff()) * volume).fillna(0).cumsum()

def create_advanced_features(input_path: Path, output_path: Path, slim: bool, meta_out: Path, window_size_meta: int):
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return

    print(f"Loading raw data from {input_path}...")
    df = pd.read_parquet(input_path)
    df = _ensure_datetime_index(df)

    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        print(f"Warning: Missing required columns. Available: {df.columns.tolist()}")

    initial_rows = len(df)
    print(f"Initial data: {initial_rows} rows")

    # ========================
    # 1) PRICE & RETURNS
    # ========================
    print("Creating price-based features...")
    df['close'] = df['close'].astype('float64')  # keep precision first
    df['ret_1'] = df['close'].pct_change()
    df['log_ret_1'] = np.log(df['close'] / df['close'].shift(1))

    # momentum extras (keep for full mode)
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1

    # ========================
    # 2) VOLATILITY
    # ========================
    print("Creating volatility features...")
    # 🔁 CHANGE: rolling_std_20 = std of log returns (stationary), ตรงกับที่โมเดลใช้
    df['rolling_std_20'] = df['log_ret_1'].rolling(window=20).std()

    # (optional extra vols for full mode)
    df['rolling_std_50'] = df['log_ret_1'].rolling(window=50).std()

    # ATR, BB (if OHLC present)
    if all(col in df.columns for col in ['high', 'low']):
        df['ATR_14'] = calculate_atr(df['high'], df['low'], df['close'], 14)
    bb_u, bb_m, bb_l, bb_w, bb_pos = calculate_bbands(df['close'], 20, 2.0)
    df['BB_width'] = bb_w
    df['BB_position'] = bb_pos

    # ========================
    # 3) MOMENTUM / TREND (full mode only)
    # ========================
    if not slim:
        print("Creating extra momentum/trend features (full mode)...")
        df['RSI_14'] = calculate_rsi(df['close'], 14)
        macd, macd_signal, macd_hist = calculate_macd(df['close'], 12, 26, 9)
        df['MACD'] = macd
        df['MACD_hist'] = macd_hist

        # z-score long window
        mean60 = df['close'].rolling(60).mean()
        std60 = df['close'].rolling(60).std()
        df['close_z_60'] = (df['close'] - mean60) / (std60.replace(0, np.nan))

    # ========================
    # CLEANUP
    # ========================
    print("Cleaning and optimizing data...")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # ---- SLIM MODE: keep only what model/backtest uses ----
    if slim:
        # ✅ ฟีเจอร์ขั้นต่ำที่ backtest/agent ใช้
        feature_list = ['ret_1', 'rolling_std_20']
        keep_cols = ['close'] + feature_list
        df = df[keep_cols]
    else:
        feature_list = [c for c in df.columns if c != 'close']

    # cast to float32 for size
    float_cols = df.select_dtypes(include=['float64', 'float32']).columns
    df[float_cols] = df[float_cols].astype('float32')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, compression='snappy')

    # save metadata for strict matching in train/backtest
    meta = {
        "features": feature_list,
        "window_size": window_size_meta,
        "freq_hint": "1H",
        "created_from": str(input_path),
        "rows": int(len(df))
    }
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.write_text(json.dumps(meta, indent=2))
    print(f"Saved meta: {meta_out}")

    final_rows = len(df)
    print("\n" + "="*60)
    print("   FEATURE ENGINEERING COMPLETED")
    print("="*60)
    print(f"Input rows:        {initial_rows:,}")
    print(f"Output rows:       {final_rows:,}")
    print(f"Rows dropped:      {initial_rows - final_rows:,} ({(initial_rows-final_rows)/max(1,initial_rows)*100:.1f}%)")
    print(f"Total features:    {len(feature_list)}")
    print(f"Output file:       {output_path}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create features from raw kline data.")
    parser.add_argument("--input", type=str, default="data/raw/btc_1h.parquet")
    parser.add_argument("--output", type=str, default="data/features/btc_1h_features.parquet")
    parser.add_argument("--slim", action="store_true", help="Keep only features required by the model/backtest")
    parser.add_argument("--window_size_meta", type=int, default=64, help="Window size used by RL policy")
    args = parser.parse_args()

    meta_out = Path(str(args.output).replace(".parquet", "_meta.json"))
    create_advanced_features(Path(args.input), Path(args.output), args.slim, meta_out, args.window_size_meta)
