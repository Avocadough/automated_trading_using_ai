# src/features/make_features_spa.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import os, sys, json
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
from src.spa.spa_core import SPAParams, run_spa


def ensure_dtindex(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame is indexed by UTC DatetimeIndex, sorted ascending."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    for c in ["timestamp", "time", "open_time", "date", "datetime"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
            df = df.set_index(c)
            break
    df = df.sort_index()
    return df


def encode_signal(sig: pd.Series) -> pd.Series:
    """Map SPA string signal -> numeric {-1,0,1}."""
    m = {"short": -1, "none": 0, "long": 1}
    out = sig.astype("string").map(m).fillna(0).astype("int8")
    return out


def make_features_spa(
    input_parquet: Path,
    output_parquet: Path,
    window_size_meta: int = 64,
    spa_d: int = 89,
    spa_alpha: float = 3.0,
    spa_gamma: float = 1.0,
    spa_m_ma: int = 5,
    spa_source: str = "close"
):
    """
    Build RL-ready features that include SPA signals.

    Required columns in input parquet: high, low, close  (volume optional)
    Output:
      - parquet with columns: ['close'] + features
      - meta json: {features, window_size, rows, source, params}
    """
    if not input_parquet.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_parquet}")

    print(f"[info] Loading raw klines: {input_parquet}")
    df = pd.read_parquet(input_parquet)
    df = ensure_dtindex(df)

    required = {"high", "low", "close"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep only needed columns
    keep_cols: List[str] = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep_cols].copy()

    # numeric dtypes
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---------- Baseline stationary features ----------
    df["log_ret_1"] = np.log(df["close"] / df["close"].shift(1))
    df["rolling_std_20"] = df["log_ret_1"].rolling(20).std()

    # ---------- Run SPA core ----------
    params = SPAParams(
        d=spa_d,
        alpha=spa_alpha,
        gamma=spa_gamma,
        m_ma=spa_m_ma,
        source=spa_source
    )
    spa_out = run_spa(df[["high", "low", "close"]], params)

    # Signals after confirmation → numeric
    spa_sig = spa_out["signals"]["signal"]               # 'long'/'short'/'none' (index-aligned)
    spa_sig_num = encode_signal(spa_sig).rename("spa_sig_num")

    # === ใส่กลับเข้า df อย่างชัดเจน ===
    df["spa_sig_num"] = spa_sig_num

    # ---------- Assemble feature table ----------
    features = ["log_ret_1", "rolling_std_20", "spa_sig_num"]
    out = df[["close"] + features]

    # Cleanup (rolling windows introduce NaN at head)
    out = out.replace([np.inf, -np.inf], np.nan).dropna()

    # Cast dtypes
    out = out.astype({
        "close": "float32",
        "log_ret_1": "float32",
        "rolling_std_20": "float32",
        "spa_sig_num": "int8"
    })

    # Save parquet
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_parquet, compression="snappy")

    # Save meta
    meta = {
        "features": features,
        "window_size": int(window_size_meta),
        "rows": int(len(out)),
        "source": str(input_parquet),
        "spa_params": {
            "d": spa_d,
            "alpha": spa_alpha,
            "gamma": spa_gamma,
            "m_ma": spa_m_ma,
            "source": spa_source
        }
    }
    meta_path = Path(str(output_parquet).replace(".parquet", "_meta.json"))
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"[ok] Saved features: {output_parquet} (rows={len(out):,})")
    print(f"[ok] Saved meta:     {meta_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build SPA-based RL features")
    ap.add_argument("--input", type=str, default="data/raw/btc_1h.parquet")
    ap.add_argument("--output", type=str, default="data/features/btc_1h_spa.parquet")
    ap.add_argument("--window_size_meta", type=int, default=64)

    # SPA params
    ap.add_argument("--spa_d", type=int, default=89)
    ap.add_argument("--spa_alpha", type=float, default=3.0)
    ap.add_argument("--spa_gamma", type=float, default=1.0)
    ap.add_argument("--spa_m_ma", type=int, default=5)
    ap.add_argument("--spa_source", type=str, default="close", choices=["close", "hl2", "hlc3"])

    args = ap.parse_args()

    make_features_spa(
        input_parquet=Path(args.input),
        output_parquet=Path(args.output),
        window_size_meta=args.window_size_meta,
        spa_d=args.spa_d,
        spa_alpha=args.spa_alpha,
        spa_gamma=args.spa_gamma,
        spa_m_ma=args.spa_m_ma,
        spa_source=args.spa_source
    )
