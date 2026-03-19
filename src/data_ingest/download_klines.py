import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta
import time
import argparse
from pathlib import Path
from tqdm import tqdm

# แปลง Binance interval → pandas freq สำหรับ validate ภายหลัง
BINANCE_TO_PANDAS = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1H", "2h": "2H", "4h": "4H", "6h": "6H", "8h": "8H", "12h": "12H",
    "1d": "1D", "3d": "3D", "1w": "1W", "1M": "1MS"
}

# Adaptive chunk size (วันต่อ 1 API call) ตาม interval
# ปรับให้ได้ใกล้เคียง 500-1000 candles ต่อ call เพื่อลด overhead
INTERVAL_CHUNK_DAYS = {
    "1m": 1, "3m": 2, "5m": 3, "15m": 7, "30m": 14,
    "1h": 30, "2h": 60, "4h": 120, "6h": 180, "12h": 365,
    "1d": 365, "3d": 365, "1w": 365, "1M": 365,
}


def download_futures_klines(pair: str, start_str: str, end_str: str, interval: str, output_path: Path):
    """
    ดึง kline จาก Binance Futures แบบ Adaptive Chunking
    - chunk size ขึ้นอยู่กับ interval เพื่อลดจำนวน API calls
    - มี sanity-check ตรวจ resolution จริงหลัง download
    """
    if interval not in BINANCE_TO_PANDAS:
        raise ValueError(f"Unsupported interval: {interval}. Supported: {list(BINANCE_TO_PANDAS)}")

    client = Client()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.strptime(start_str, "%Y-%m-%d")
    end_dt   = datetime.strptime(end_str,   "%Y-%m-%d")
    chunk_days = INTERVAL_CHUNK_DAYS.get(interval, 7)

    print(f"[info] Downloading {pair} | interval={interval} | {start_str} → {end_str}")
    print(f"[info] Chunk size: {chunk_days} days per API call")

    all_frames = []
    cursor = start_dt
    total_chunks = max(1, (end_dt - start_dt).days // chunk_days + 1)

    with tqdm(total=total_chunks, desc="Downloading") as pbar:
        while cursor < end_dt:
            chunk_end = min(cursor + timedelta(days=chunk_days), end_dt)
            start_ms  = int(cursor.timestamp() * 1000)
            end_ms    = int((chunk_end - timedelta(milliseconds=1)).timestamp() * 1000)

            try:
                klines = client.futures_klines(
                    symbol=pair,
                    interval=interval,
                    startTime=start_ms,
                    endTime=end_ms,
                    limit=1500
                )
                if klines:
                    df = pd.DataFrame(klines, columns=[
                        'open_time', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    all_frames.append(df)
                time.sleep(0.3)  # rate limit
            except Exception as e:
                print(f"\n[warn] Error on chunk {cursor.date()} → {chunk_end.date()}: {e}")
                time.sleep(2.0)

            cursor = chunk_end
            pbar.update(1)

    if not all_frames:
        print("[error] No data downloaded. Exiting.")
        return

    # รวม + ทำความสะอาด
    final_df = pd.concat(all_frames, ignore_index=True)
    final_df['open_time'] = pd.to_datetime(final_df['open_time'], unit='ms', utc=True)

    numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                    'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
    for col in numeric_cols:
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce')

    final_df = (final_df
                .drop_duplicates(subset=['open_time'])
                .set_index('open_time')
                .sort_index())

    # เลือกแค่ OHLCV
    out = final_df[['open', 'high', 'low', 'close', 'volume']].copy()
    out['close'] = out['close'].ffill()
    out = out.dropna(subset=['open', 'high', 'low'])

    # ======= Sanity check: ตรวจ resolution จริงๆ =======
    diffs = out.index.to_series().diff().dropna()
    if not diffs.empty:
        actual_mode = diffs.mode()[0]
        expected_freq = BINANCE_TO_PANDAS[interval]
        expected_td   = pd.tseries.frequencies.to_offset(expected_freq).nanos / 1e9
        actual_secs   = actual_mode.total_seconds()

        if abs(actual_secs - expected_td) > expected_td * 0.05:  # ยอมให้เพี้ยน <5%
            print(f"\n⚠️  WARNING: ข้อมูลที่ download มีช่วงเวลาจริง = {actual_mode}")
            print(f"   แต่ขอ interval={interval} (คาดว่าช่วง = {expected_td / 60:.0f} นาที)")
            print(f"   กรุณาตรวจสอบว่า --interval และ --output path ถูกต้อง!")
        else:
            print(f"\n[ok] Sanity check passed: resolution = {actual_mode} ตรงกับ interval={interval}")

    out.to_parquet(output_path)
    print(f"[ok] Saved {len(out):,} rows → {output_path}")
    print(f"     Date range: {out.index.min()} → {out.index.max()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Binance Futures Kline Data (Adaptive Chunking)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--pair",     type=str, required=True, help="Trading pair, e.g. BTCUSDT")
    parser.add_argument("--start",    type=str, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",      type=str, required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--interval", type=str, default="1h",
                        choices=list(BINANCE_TO_PANDAS.keys()),
                        help="Kline interval")
    parser.add_argument("--output",   type=str, required=True,
                        help="Output .parquet path (e.g. data/raw/btc_1h.parquet)")
    args = parser.parse_args()

    download_futures_klines(args.pair, args.start, args.end, args.interval, Path(args.output))
