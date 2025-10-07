import pandas as pd
from binance.client import Client
from datetime import datetime
import time
import argparse
from pathlib import Path
from tqdm import tqdm

def download_futures_klines(pair, start_str, end_str, interval, output_path):
    """
    Downloads historical kline data from Binance Futures using the correct
    `futures_klines` method with daily iteration to handle API limits.
    """
    client = Client()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching data for {pair} from {start_str} to {end_str}...")
    
    start_dt = datetime.strptime(start_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_str, "%Y-%m-%d")

    all_klines_df = []
    
    # We must loop day-by-day because the API has a limit per request.
    date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')

    for day in tqdm(date_range, desc="Downloading daily data"):
        day_start_ms = int(day.timestamp() * 1000)
        # Get data for the full 24-hour period
        day_end_ms = int((day + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)).timestamp() * 1000)

        try:
            # --- THIS IS THE CORRECT FUNCTION CALL ---
            klines = client.futures_klines(
                symbol=pair,
                interval=interval,
                startTime=day_start_ms,
                endTime=day_end_ms,
                limit=1500  # Max limit per request
            )
            # --- END OF CORRECTED SECTION ---
            
            if klines:
                df = pd.DataFrame(klines, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                all_klines_df.append(df)
            
            # A brief pause to respect API rate limits
            time.sleep(0.5)

        except Exception as e:
            print(f"An error occurred on {day.strftime('%Y-%m-%d')}: {e}")

    if not all_klines_df:
        print("No data downloaded. Exiting.")
        return

    # Combine all daily dataframes
    final_df = pd.concat(all_klines_df, ignore_index=True)

    # Data cleaning and processing
    final_df['open_time'] = pd.to_datetime(final_df['open_time'], unit='ms', utc=True)
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'number_of_trades']
    for col in numeric_cols:
        final_df[col] = pd.to_numeric(final_df[col])
    
    final_df = final_df.drop_duplicates(subset=['open_time'])
    final_df = final_df.set_index('open_time').sort_index()
    
    # Resample to ensure a continuous 1-minute timeline and forward-fill any gaps
    final_df = final_df.resample('1min').ffill().dropna()

    final_df.to_parquet(output_path)
    print(f"Successfully saved {len(final_df)} rows to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Binance Futures Kline Data.")
    parser.add_argument("--pair", type=str, default="BTCUSDT", help="Trading pair symbol (e.g., BTCUSDT)")
    parser.add_argument("--start", type=str, required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end", type=str, required=True, help="End date in YYYY-MM-DD format")
    parser.add_argument("--interval", type=str, default="1m", help="Kline interval (e.g., 1m, 5m, 1h)")
    parser.add_argument("--output", type=str, default="data/raw/btc_1m.parquet", help="Output file path")
    args = parser.parse_args()

    download_futures_klines(args.pair, args.start, args.end, args.interval, Path(args.output))