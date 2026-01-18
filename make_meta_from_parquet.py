import pandas as pd

df = pd.read_parquet("data/features/btc_15m_rl_features_split_validated.parquet")
print(df.head())