import pandas as pd
from binance.client import Client
from datetime import datetime
import time
import argparse
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timezone  # <--- เพิ่ม timezone เข้ามา

# แปลง interval ของ Binance -> pandas freq สำหรับ resample ให้ตรง TF ที่ขอ
BINANCE_TO_PANDAS = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1H", "2h": "2H", "4h": "4H", "6h": "6H", "8h": "8H", "12h": "12H",
    "1d": "1D", "3d": "3D", "1w": "1W", "1M": "1MS"
}

def download_futures_klines(pair, start_str, end_str, interval, output_path):
    """
    ดึง kline จาก Binance Futures (futures_klines) แบบไล่รายวัน
    และ resample เพื่อจัดการ Missing data ให้เป็นระเบียบ
    """
    client = Client() # ไม่ต้องใส่ API Key ก็ได้สำหรับการดึง klines สาธารณะ
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if interval not in BINANCE_TO_PANDAS:
        raise ValueError(f"Unsupported interval: {interval}")

    print(f"🚀 Starting download: {pair}")
    print(f"📅 Range: {start_str} to {end_str}")
    print(f"⏱️ Timeframe: {interval}")
    print(f"💾 Output: {output_path}")

    start_dt = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    all_klines_df = []
    # สร้าง List ของวันทั้งหมดที่จะดึง
    date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')

    # ใช้ tqdm แสดง progress bar
    for day in tqdm(date_range, desc=f"Downloading {interval} Data"):
        day_start_ms = int(day.timestamp() * 1000)
        # สิ้นสุดวัน (23:59:59.999)
        day_end_ms = int((day + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)).timestamp() * 1000)

        try:
            klines = client.futures_klines(
                symbol=pair,
                interval=interval,
                startTime=day_start_ms,
                endTime=day_end_ms,
                limit=1500  # M5 1 วันมี 288 แท่ง, limit 1500 เหลือเฟือ
            )
            
            if klines:
                df = pd.DataFrame(klines, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                # เก็บเฉพาะคอลัมน์ที่ใช้ ลดขนาดหน่วยความจำ
                df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
                all_klines_df.append(df)

            # ลดเวลา sleep ลงเล็กน้อยเพื่อให้เร็วขึ้น (0.2s ปลอดภัยสำหรับ 1 request/day)
            time.sleep(0.15)

        except Exception as e:
            print(f"⚠️ Error on {day.strftime('%Y-%m-%d')}: {e}")
            time.sleep(1) # ถ้า error ให้พักนานหน่อยแล้วค่อยไปต่อ

    if not all_klines_df:
        print("❌ No data downloaded. Exiting.")
        return

    print("🔨 Processing and merging data...")
    final_df = pd.concat(all_klines_df, ignore_index=True)

    # แปลงชนิดข้อมูล
    final_df['open_time'] = pd.to_datetime(final_df['open_time'], unit='ms', utc=True)
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce')

    final_df = final_df.drop_duplicates(subset=['open_time'])
    final_df = final_df.set_index('open_time').sort_index()

    # 🔁 Resample เพื่ออุดช่องว่าง (Missing Data)
    freq = BINANCE_TO_PANDAS[interval]
    
    # Logic: 
    # - ราคา (OHLC): ถ้าข้อมูลหาย ให้ใช้ค่าก่อนหน้า (Forward Fill) เพื่อไม่ให้กราฟขาด
    # - Volume: ถ้าข้อมูลหาย แปลว่าไม่มีเทรด ให้เป็น 0
    resampled_df = pd.DataFrame({
        'open': final_df['open'].resample(freq).first().ffill(),
        'high': final_df['high'].resample(freq).max().ffill(),
        'low': final_df['low'].resample(freq).min().ffill(),
        'close': final_df['close'].resample(freq).last().ffill(),
        'volume': final_df['volume'].resample(freq).sum().fillna(0)
    })

    # ตัดขอบข้อมูลให้ตรงกับวันที่ขอเป๊ะๆ
    resampled_df = resampled_df.loc[start_dt:end_dt]

    # Save
    resampled_df.to_parquet(output_path)
    print(f"✅ Completed! Saved {len(resampled_df):,} rows to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Binance Futures Kline Data (M5 6-Years).")
    
    # ปรับ Default ให้ตรงโจทย์ 6 ปี M5
    parser.add_argument("--pair", type=str, default="BTCUSDT", help="Trading pair symbol")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2026-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", type=str, default="5m", help="Kline interval")
    parser.add_argument("--output", type=str, default="data/raw/btc_5m_2020_2026.parquet", help="Output file path")
    
    args = parser.parse_args()

    download_futures_klines(args.pair, args.start, args.end, args.interval, Path(args.output))