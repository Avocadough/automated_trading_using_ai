import pytest
import pandas as pd
import numpy as np

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.utils import ensure_datetime_index

def test_ensure_datetime_index():
    df = pd.DataFrame({"close": [1, 2, 3]}, index=["2024-01-01", "invalid", "2024-01-02"])
    
    # should coerce invalid to NaT and drop from start, or at least handle strings
    df_out = ensure_datetime_index(df)
    assert isinstance(df_out.index, pd.DatetimeIndex)
    assert df_out.index.tz is not None # UTC
    
def test_ensure_datetime_fallback():
    # test falling back to columns like 'timestamp'
    df = pd.DataFrame({
        "timestamp": [1704067200000, 1704153600000],
        "close": [60000, 61000]
    })
    df_out = ensure_datetime_index(df)
    assert isinstance(df_out.index, pd.DatetimeIndex)
