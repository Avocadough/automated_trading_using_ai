import pandas as pd
import argparse
from pathlib import Path

def create_features(input_path, output_path):
    """Loads raw data, creates basic features, and saves to a new Parquet file."""
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return

    print(f"Loading raw data from {input_path}...")
    df = pd.read_parquet(input_path)

    # Feature Creation
    df['ret_1'] = df['close'].pct_change()
    df['MA7'] = df['close'].rolling(window=7).mean()
    df['MA30'] = df['close'].rolling(window=30).mean()
    df['rolling_std_20'] = df['close'].rolling(window=20).std()
    
    # Z-score of close price over a rolling window
    rolling_mean_60 = df['close'].rolling(window=60).mean()
    rolling_std_60 = df['close'].rolling(window=60).std()
    df['close_z'] = (df['close'] - rolling_mean_60) / rolling_std_60

    df = df.dropna()
    df = df.astype({col: 'float32' for col in df.columns if df[col].dtype == 'float64'})
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    print(f"Features created. Saved {len(df)} rows to {output_path}")
    print("Feature columns:", df.columns.tolist())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create features from raw kline data.")
    parser.add_argument("--input", type=str, default="data/raw/btc_1h.parquet", help="Input raw data file")
    parser.add_argument("--output", type=str, default="data/features/btc_1h_features.parquet", help="Output features file")
    args = parser.parse_args()

    create_features(Path(args.input), Path(args.output))