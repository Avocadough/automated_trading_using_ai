# Automated Trading using AI (Student Project Scaffold)

This project provides a runnable scaffold for developing and testing an automated trading bot using Reinforcement Learning (PPO) on Binance Futures Testnet data.

## Project Structure

- `data/`: Stores raw kline data, generated features, and trained models.
- `src/`: Contains all source code, organized by functionality (data ingestion, feature engineering, RL environment, training, etc.).
- `reports/`: For saving figures and tables from analysis.
- `requirements.txt`: Python package dependencies.

## Setup

1.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `.\venv\Scripts\activate`
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Pipeline

Follow these steps in order.

1.  **Download Historical Data:**
    Download 1-minute kline data for a specific period.
    ```bash
    python src/data_ingest/download_klines.py --start "2024-05-01" --end "2024-05-10"
    ```

2.  **Create Features:**
    Process the raw data to generate features for the model.
    ```bash
    python src/feature_engineering/make_features.py
    ```

3.  **Run Baseline Backtest:**
    Test a simple MA crossover strategy to establish a baseline performance.
    ```bash
    python src/backtest/baseline_backtest.py
    ```

4.  **Train the RL Agent (PPO):**
    Train the PPO agent on the generated features. Use a small number of timesteps for a quick test.
    ```bash
    python src/train/train_ppo.py --timesteps 10000
    ```
    For a more serious training run, increase `--timesteps` to `300000` or more.

5.  **Launch the Dashboard:**
    Visualize the data and features using Streamlit.
    ```bash
    streamlit run src/dashboard/app.py
    ```