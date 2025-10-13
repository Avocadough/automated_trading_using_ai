# save as tools/make_meta.py (ชั่วคราว)
import pandas as pd, json, sys
from pathlib import Path

parquet_path = Path("data/features/btc_1m_rl_features_validated.parquet")  # <-- แก้ชื่อไฟล์ถ้าไม่ตรง
df = pd.read_parquet(parquet_path)

# เลือกทุกคอลัมน์เป็นฟีเจอร์ ยกเว้น 'open','close' (ปรับได้ตามที่ต้องการ)
features = [c for c in df.columns if c not in ["open","close"]]

meta = {
    "features": features,
    "window_size": 64   # ตั้งค่าหน้าต่างอินพุตให้ env/ppo ใช้
}

meta_path = Path(str(parquet_path).replace(".parquet", "_meta.json"))
meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(f"[ok] wrote: {meta_path}")
print(f"features = {features}")
