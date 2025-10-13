# src/features/make_features_simple.py
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json

def ensure_dtindex(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    for c in ["timestamp", "time", "open_time", "date", "datetime"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
            df = df.set_index(c)
            break
    df = df.sort_index()
    return df

def make_features(input_path: Path, output_path: Path, window_size_meta: int = 64):
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    print(f"Loading raw: {input_path}")
    df = pd.read_parquet(input_path)
    df = ensure_dtindex(df)

    if "close" not in df.columns:
        raise ValueError("Input parquet must contain 'close' column.")

    # --- Simple, stationary features ---
    df["close"] = df["close"].astype("float64")
    df["log_ret_1"] = np.log(df["close"] / df["close"].shift(1))
    df["rolling_std_20"] = df["log_ret_1"].rolling(20).std()

    # cleanup
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    # keep minimal set
    features = ["log_ret_1", "rolling_std_20"]
    out = df[["close"] + features].astype("float32")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, compression="snappy")

    meta = {
        "features": features,
        "window_size": int(window_size_meta),
        "freq_hint": "1H",
        "rows": int(len(out)),
        "source": str(input_path)
    }
    meta_path = Path(str(output_path).replace(".parquet", "_meta.json"))
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Saved features: {output_path}")
    print(f"Saved meta:     {meta_path}")
    print(f"Rows: {len(out):,}, Cols: {len(out.columns)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Make simple features for RL trading")
    ap.add_argument("--input", type=str, default="data/raw/btc_1h.parquet")
    ap.add_argument("--output", type=str, default="data/features/btc_1h_simple.parquet")
    ap.add_argument("--window_size_meta", type=int, default=64)
    args = ap.parse_args()
    make_features(Path(args.input), Path(args.output), args.window_size_meta)
