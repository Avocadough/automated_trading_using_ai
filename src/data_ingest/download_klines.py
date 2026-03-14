import pandas as pd
from binance.client import Client
from datetime import datetime
import time
import argparse
from pathlib import Path
from tqdm import tqdm

# แปลง interval ของ Binance -> pandas freq สำหรับ resample ให้ตรง TF ที่ขอ
BINANCE_TO_PANDAS = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1H", "2h": "2H", "4h": "4H", "6h": "6H", "8h": "8H", "12h": "12H",
    "1d": "1D", "3d": "3D", "1w": "1W", "1M": "1MS"
}

def download_futures_klines(pair, start_str, end_str, interval, output_path):
    """
    ดึง kline จาก Binance Futures (futures_klines) แบบไล่รายวัน เพื่อติดเพดาน API
    และ resample ตาม timeframe ที่ขอ (ไม่ fix ที่ 1 นาทีแล้ว)
    """
    client = Client()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if interval not in BINANCE_TO_PANDAS:
        raise ValueError(f"Unsupported interval: {interval}")

    print(f"Fetching data for {pair} from {start_str} to {end_str} (interval={interval})...")

    start_dt = datetime.strptime(start_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_str, "%Y-%m-%d")

    all_klines_df = []
    date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')

    for day in tqdm(date_range, desc="Downloading daily data"):
        day_start_ms = int(day.timestamp() * 1000)
        day_end_ms = int((day + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)).timestamp() * 1000)

        try:
            klines = client.futures_klines(
                symbol=pair,
                interval=interval,
                startTime=day_start_ms,
                endTime=day_end_ms,
                limit=1500
            )
            if klines:
                df = pd.DataFrame(klines, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                all_klines_df.append(df)

            time.sleep(0.5)  # เคารพ rate limit

        except Exception as e:
            print(f"An error occurred on {day.strftime('%Y-%m-%d')}: {e}")

    if not all_klines_df:
        print("No data downloaded. Exiting.")
        return

    final_df = pd.concat(all_klines_df, ignore_index=True)

    # ทำความสะอาด/แปลงชนิด
    final_df['open_time'] = pd.to_datetime(final_df['open_time'], unit='ms', utc=True)
    numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                    'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
    for col in numeric_cols:
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce')

    final_df = final_df.drop_duplicates(subset=['open_time'])
    final_df = final_df.set_index('open_time').sort_index()

    # 🔁 Resample ให้ตรงกับ interval ที่ขอ (เช่น 15m -> 15min)
    freq = BINANCE_TO_PANDAS[interval]
    # ใช้ OHLCV ที่เหมาะสมตอน resample (ถ้า missing จะ forward-fill ราคา, volume ใช้ sum)
    o = final_df['open'].resample(freq).first()
    h = final_df['high'].resample(freq).max()
    l = final_df['low'].resample(freq).min()
    c = final_df['close'].resample(freq).last()
    v = final_df['volume'].resample(freq).sum()

    out = pd.DataFrame({
        'open': o.ffill(),
        'high': h.ffill(),
        'low': l.ffill(),
        'close': c.ffill(),
        'volume': v.fillna(0)
    })

    out.to_parquet(output_path)
    print(f"Successfully saved {len(out)} rows to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Binance Futures Kline Data.")
    parser.add_argument("--pair", type=str, default="BTCUSDT", help="Trading pair symbol (e.g., BTCUSDT)")
    parser.add_argument("--start", type=str, required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end", type=str, required=True, help="End date in YYYY-MM-DD format")
    parser.add_argument("--interval", type=str, default="m5", help="Kline interval (e.g., 1m, 5m, 15m, 1h)")
    parser.add_argument("--output", type=str, default="data/raw/btc_5m.parquet", help="Output file path")
    args = parser.parse_args()

    download_futures_klines(args.pair, args.start, args.end, args.interval, Path(args.output))
