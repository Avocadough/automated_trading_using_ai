import streamlit as st
import pandas as pd
import argparse
from pathlib import Path

st.set_page_config(layout="wide")

@st.cache_data
def load_data(file_path):
    """Loads feature data from a Parquet file."""
    if file_path.exists():
        return pd.read_parquet(file_path)
    return None

def main(features_path):
    st.title("📈 AI Trading Project Dashboard")

    st.header("Data & Features Visualization")
    
    df = load_data(features_path)

    if df is not None:
        st.success(f"Loaded {len(df)} rows of data from {features_path.name}")
        
        st.subheader("BTCUSDT Close Price")
        st.line_chart(df['close'])
        
        st.subheader("Feature Samples")
        st.dataframe(df.tail(100))
        
        selected_feature = st.selectbox("Select a feature to plot:", df.columns)
        if selected_feature:
            st.line_chart(df[selected_feature])

    else:
        st.error(f"Could not find feature file at '{features_path}'. Please run the feature engineering script first.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streamlit Dashboard for Trading AI.")
    parser.add_argument("--features", type=str, default="data/features/btc_1m_features.parquet", help="Path to the features file.")
    args = parser.parse_args()
    
    main(Path(args.features))