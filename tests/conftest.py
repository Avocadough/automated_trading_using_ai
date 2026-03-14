import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_ohlcv():
    dates = pd.date_range("2024-01-01", periods=100, tz="UTC", freq="15min")
    df = pd.DataFrame({
        "open": np.linspace(60000, 61000, 100),
        "high": np.linspace(60500, 61500, 100),
        "low": np.linspace(59500, 60500, 100),
        "close": np.linspace(60100, 61100, 100),
        "volume": np.random.rand(100) * 10
    }, index=dates)
    return df
