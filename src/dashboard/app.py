import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import os
import json

st.set_page_config(page_title="AI Trading Dashboard", layout="wide")

def load_data(path: str) -> pd.DataFrame:
    if Path(path).exists():
        return pd.read_parquet(path)
    return pd.DataFrame()

def load_eval_data(path: str) -> pd.DataFrame:
    if Path(path).exists():
        return pd.read_csv(path)
    return pd.DataFrame()

def load_trades_data(path: str) -> pd.DataFrame:
    if Path(path).exists():
        return pd.read_csv(path)
    return pd.DataFrame()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Explorer", "Training Monitor", "Evaluation Results", "Paper Trading"])

if page == "Data Explorer":
    st.title("Data Explorer")
    st.write("Visualizing raw OHLCV and SPA boundaries.")
    
    data_path = st.text_input("Features Parquet Path", "data/features/btc_15m_spa.parquet")
    df = load_data(data_path)
    
    if not df.empty:
        df_slice = df.tail(500)  # show last 500
        fig = go.Figure(data=[go.Candlestick(x=df_slice.index,
                        open=df_slice['open'] if 'open' in df_slice.columns else df_slice['close'],
                        high=df_slice['high'] if 'high' in df_slice.columns else df_slice['close'],
                        low=df_slice['low'] if 'low' in df_slice.columns else df_slice['close'],
                        close=df_slice['close'])])
        
        # Add SPA boundaries if they exist
        if 'spa_sig_num' in df_slice.columns:
            st.write("Feature Correlation Heatmap")
            st.dataframe(df_slice[['close', 'log_ret_1', 'rolling_std_20', 'spa_sig_num']].corr())
            
        fig.update_layout(xaxis_rangeslider_visible=False, title="Candlestick Chart")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Data not found.")

elif page == "Training Monitor":
    st.title("Training Monitor")
    st.write("Reads TensorBoard logs directly or use the command `tensorboard --logdir ./ppo_logs_spa/` in your terminal.")
    st.info("Currently, run `tensorboard --logdir ./ppo_logs_spa/` to view rich metric graphs.")

elif page == "Evaluation Results":
    st.title("Evaluation Results")
    eval_csv = st.text_input("Evaluation CSV Path", "data/eval/ppo_spa_btc_15m_eval_best.csv")
    trades_csv = st.text_input("Trades CSV Path", "data/eval/ppo_spa_btc_15m_eval_best_trades.csv")
    
    df_eval = load_eval_data(eval_csv)
    df_trades = load_trades_data(trades_csv)
    
    if not df_eval.empty:
        st.subheader("Equity Curve")
        st.line_chart(df_eval['equity'])
        
        if not df_trades.empty:
            st.subheader("Trade Log")
            st.dataframe(df_trades)
            
            wins = df_trades[df_trades['return'] > 0]
            st.metric("Total Trades", len(df_trades))
            st.metric("Winrate", f"{(len(wins) / len(df_trades)) * 100:.2f}%" if len(df_trades) > 0 else "0.00%")
    else:
        st.warning("Evaluation records not found. Have you evaluated the model yet?")

elif page == "Paper Trading":
    st.title("Paper Trading Live Monitor")
    st.write("Connects to the paper trading module state...")
    st.info("Run `python src/paper_trade/paper_trade.py` in the terminal to start submitting trades.")
    
    # Simple auto-refresh mock
    state_file = Path("data/paper_trade_state.json")
    if state_file.exists():
        state = json.loads(state_file.read_text())
        col1, col2, col3 = st.columns(3)
        col1.metric("Equity", f"${state.get('equity', 0):.2f}")
        col2.metric("Position Qty", f"{state.get('qty', 0):.4f}")
        col3.metric("Avg Entry", f"${state.get('avg_entry', 0):.2f}")
    else:
        st.warning("Paper trading state file not found.")