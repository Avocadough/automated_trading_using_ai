import pytest
import pandas as pd
import os
import json

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.features.make_features_spa import make_features_spa

def test_feature_pipeline(sample_ohlcv, tmp_path):
    in_pq = tmp_path / "raw.parquet"
    out_pq = tmp_path / "out.parquet"
    
    sample_ohlcv.to_parquet(in_pq)
    
    make_features_spa(in_pq, out_pq)
    
    assert out_pq.exists()
    
    meta_file = tmp_path / "out_meta.json"
    assert meta_file.exists()
    
    df_out = pd.read_parquet(out_pq)
    assert not df_out.empty
    
    meta = json.loads(meta_file.read_text())
    assert "features" in meta
    assert "freq_hint" in meta
    assert meta["train_split"] == 0.8
