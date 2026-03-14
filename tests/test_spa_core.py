import pytest
import pandas as pd
import numpy as np

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.spa.spa_core import run_spa, SPAParams

def test_spa_signal_generation(sample_ohlcv):
    params = SPAParams(d=21, alpha=5, gamma=1.0, m_ma=2)
    # create artificial breakout
    sample_ohlcv.loc[sample_ohlcv.index[-1], 'close'] = 70000 
    
    out = run_spa(sample_ohlcv, params)
    
    assert "signals" in out
    assert "bands" in out
    
    signals = out["signals"]
    assert not signals.empty
    assert "signal" in signals.columns
    # Ensure signal handles string categorical logic correctly
    assert set(signals["signal"].dropna().unique()).issubset({"none", "long", "short"})
