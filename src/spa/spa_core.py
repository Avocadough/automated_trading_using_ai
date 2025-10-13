# src/spa/spa_core.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple, Dict
import numpy as np
import pandas as pd

SourceType = Literal["close", "hl2", "hlc3"]
SignalType = Literal["long", "short", "none"]

@dataclass
class SPAParams:
    """Parameters of the SPA algorithm."""
    d: int = 89                 # market depth (window for swing stats)
    alpha: float = 3.0          # α (used in inner boundaries)
    gamma: float = 1.0          # γ (ATR_ext = mu + γ*sigma)
    m_ma: int = 5               # confirmation MA length
    source: SourceType = "close"
    min_d: int = 20

def _source_series(df: pd.DataFrame, source: SourceType) -> pd.Series:
    close = df["close"].astype("float64")
    if source == "close":
        return close
    if source == "hl2":
        return (df["high"].astype("float64") + df["low"].astype("float64")) / 2.0
    if source == "hlc3":
        return (df["high"].astype("float64") + df["low"].astype("float64") + close) / 3.0
    raise ValueError(f"Unsupported source: {source}")

def compute_swing_stats(df: pd.DataFrame, d: int) -> Tuple[pd.Series, pd.Series]:
    """mu = mean(high-low, d), sigma = std(high-low, d)"""
    d = max(int(d), 1)
    swing = (df["high"].astype("float64") - df["low"].astype("float64")).clip(lower=0)
    mu = swing.rolling(d, min_periods=d).mean()
    sigma = swing.rolling(d, min_periods=d).std(ddof=0)
    return mu, sigma

def compute_boundaries(df: pd.DataFrame, mu: pd.Series, sigma: pd.Series,
                      alpha: float, gamma: float, d: int) -> pd.DataFrame:
    """
    Build dynamic inner/outer boundaries with explicit window d:
      h (upper inner) = rolling_high(d) - α * mu
      H (upper outer) = max(rolling_high(d), close_t) + ATR_ext
      l (lower inner) = rolling_low(d)  + α * mu
      L (lower outer) = min(rolling_low(d),  close_t) - ATR_ext
      where ATR_ext = mu + γ*sigma
    """
    d = max(int(d), 1)
    high_d = df["high"].rolling(d, min_periods=d).max()
    low_d  = df["low"].rolling(d, min_periods=d).min()
    close = df["close"].astype("float64")
    atr_ext = (mu + gamma * sigma)

    h = (high_d - alpha * mu)
    H = pd.concat([high_d, close], axis=1).max(axis=1) + atr_ext

    l = (low_d + alpha * mu)
    L = pd.concat([low_d, close], axis=1).min(axis=1) - atr_ext

    out = pd.DataFrame({"h_inner": h, "H_outer": H, "l_inner": l, "L_outer": L}, index=df.index)
    return out

def generate_raw_signals(df: pd.DataFrame, bands: pd.DataFrame) -> pd.Series:
    """
    Raw SPA logic (bar-close):
      Upper rejection: (prev_close > h) & (close < h)   -> short
      Upper breakout : (prev_close <= H) & (close > H)  -> long
      Lower rejection: (prev_close < l) & (close > l)   -> long
      Lower breakout : (prev_close >= L) & (close < L)  -> short
    """
    close = df["close"].astype("float64")
    prev_close = close.shift(1)

    h = bands["h_inner"]; H = bands["H_outer"]
    l = bands["l_inner"]; L = bands["L_outer"]

    sig = pd.Series("none", index=df.index, dtype="string")
    upper_reject = (prev_close > h) & (close < h)
    upper_break  = (prev_close <= H) & (close > H)
    lower_reject = (prev_close < l) & (close > l)
    lower_break  = (prev_close >= L) & (close < L)

    sig[upper_reject] = "short"
    sig[upper_break]  = "long"
    sig[lower_reject] = "long"
    sig[lower_break]  = "short"
    return sig

def apply_confirmation(df: pd.DataFrame, signals: pd.Series, m_ma: int, source: SourceType) -> pd.Series:
    """Confirm with short MA on chosen source: LONG needs src>MA, SHORT needs src<MA."""
    m_ma = max(int(m_ma), 1)
    src = _source_series(df, source)
    ma = src.rolling(m_ma, min_periods=m_ma).mean()

    confirmed = signals.copy()
    cond_long = (signals == "long") & (src > ma)
    cond_short = (signals == "short") & (src < ma)
    confirmed[~(cond_long | cond_short)] = "none"
    return confirmed.astype("string")

def dynamic_tp_sl(df: pd.DataFrame, risk_window: int = 200) -> pd.Series:
    """
    Δ proxy: average bar range over last risk_window bars.
    ใช้ mean(high-low, risk_window) (ไม่หารด้วย window เพื่อให้ scale สมเหตุสมผล)
    """
    risk_window = max(int(risk_window), 1)
    rng = (df["high"].astype("float64") - df["low"].astype("float64"))
    delta = rng.rolling(risk_window, min_periods=risk_window).mean()
    return delta

def build_orders(df: pd.DataFrame, signals: pd.Series, delta: pd.Series) -> pd.DataFrame:
    """
    For LONG:  SL = close - Δ,  TP = close + 2Δ
    For SHORT: SL = close + Δ,  TP = close - 2Δ
    """
    c = df["close"].astype("float64")
    sl = pd.Series(np.nan, index=df.index)
    tp = pd.Series(np.nan, index=df.index)

    long_mask = signals == "long"
    short_mask = signals == "short"

    sl[long_mask] = c[long_mask] - delta[long_mask]
    tp[long_mask] = c[long_mask] + 2.0 * delta[long_mask]

    sl[short_mask] = c[short_mask] + delta[short_mask]
    tp[short_mask] = c[short_mask] - 2.0 * delta[short_mask]

    return pd.DataFrame({"signal": signals, "sl": sl, "tp": tp}, index=df.index)

def run_spa(df: pd.DataFrame, params: SPAParams) -> Dict[str, pd.DataFrame]:
    """
    Pipeline:
      1) mu, sigma from swing stats on window d
      2) boundaries (h/H, l/L) with explicit d
      3) raw signals
      4) MA confirmation
      5) dynamic TP/SL
    Returns:
      - 'bands'
      - 'signals_raw'
      - 'signals'
      - 'orders'
    """
    if params.d < params.min_d:
        raise ValueError(f"d too small: {params.d} < {params.min_d}")

    required = {"high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"df must contain {required}")

    df = df.copy().sort_index()

    # 1) swing stats
    mu, sigma = compute_swing_stats(df, params.d)

    # 2) boundaries (explicit d)
    bands = compute_boundaries(df, mu, sigma, params.alpha, params.gamma, params.d)

    # 3) signals
    sig_raw = generate_raw_signals(df, bands)

    # 4) confirmation
    sig = apply_confirmation(df, sig_raw, params.m_ma, params.source)

    # 5) dynamic TP/SL
    delta = dynamic_tp_sl(df, risk_window=200)
    orders = build_orders(df, sig, delta)

    return {
        "bands": bands,
        "signals_raw": sig_raw.to_frame("signal_raw"),
        "signals": sig.to_frame("signal"),
        "orders": orders
    }
