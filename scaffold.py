# scaffold.py
from pathlib import Path
import sys

BASENAME = "automated-trading-ai"

def get_base_dir() -> Path:
    """รองรับทั้งการรันจากในโฟลเดอร์โปรเจกต์หรือโฟลเดอร์แม่"""
    cwd = Path.cwd()
    if cwd.name == BASENAME:
        return cwd
    else:
        return cwd / BASENAME

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_file(path: Path, content: str):
    if not path.exists():
        path.write_text(content, encoding="utf-8")
        print(f"[CREATE] {path}")
    else:
        print(f"[SKIP]   {path} (exists)")

def main():
    base = get_base_dir()
    if not base.exists():
        print(f"[INFO] สร้างโฟลเดอร์ฐาน: {base}")
        base.mkdir(parents=True, exist_ok=True)

    # --- สร้างโฟลเดอร์ย่อย ---
    dirs = [
        "data/features",
        "data/models",
        "data/raw",
        "reports/figs",
        "reports/tables",
        "src/backtest",
        "src/dashboard",
        "src/data_ingest",
        "src/feature_engineering",
        "src/paper_trade",
        "src/rl_env",
        "src/train",
        "src/utils",
    ]
    for d in dirs:
        ensure_dir(base / d)

    # --- เนื้อหาไฟล์ตั้งต้นแบบบางเบา (ปรับเพิ่มได้ภายหลัง) ---
    files = {
        "src/__init__.py": '"""automated-trading-ai package root."""\n',
        "src/backtest/__init__.py": "",
        "src/backtest/baseline_backtest.py": """\
\"\"\"Baseline backtest stub.

ใส่ logic การ backtest ที่นี่ เช่นอ่านราคาจาก data/raw,
คำนวณสัญญาณจาก data/features แล้วสรุป performance ลง reports/tables
\"\"\"

def run_backtest():
    print("Running baseline backtest... (TODO)")

if __name__ == "__main__":
    run_backtest()
""",
        "src/dashboard/__init__.py": "",
        "src/dashboard/app.py": """\
\"\"\"Dashboard app stub.

คุณอาจใช้ Streamlit/FastAPI/Gradio ตามถนัด
\"\"\"

def main():
    print("Dashboard placeholder — เพิ่ม UI ภายหลัง")

if __name__ == "__main__":
    main()
""",
        "src/data_ingest/__init__.py": "",
        "src/data_ingest/download_klines.py": """\
\"\"\"Downloader stub for market data (e.g., Binance klines).\"\"\"

def download(symbol: str, interval: str = "1h", limit: int = 1000):
    # TODO: ใส่โค้ดดึงราคาจริง (เช่น ccxt/requests)
    print(f"Downloading klines for {symbol} @ {interval}, limit={limit} (TODO)")
""",
        "src/feature_engineering/__init__.py": "",
        "src/feature_engineering/make_features.py": """\
\"\"\"Feature engineering stub.\"\"\"

def make_features():
    # TODO: คำนวณ indicator / features และบันทึกที่ data/features
    print("Making features... (TODO)")
""",
        "src/paper_trade/__init__.py": "",
        "src/rl_env/__init__.py": "",
        "src/rl_env/crypto_env.py": """\
\"\"\"Minimal RL environment stub (gym-like).\"\"\"

class CryptoEnv:
    def __init__(self):
        self.state = None

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        # TODO: ใส่ reward/transition จริง
        self.state += 1
        reward = 0.0
        done = self.state > 10
        info = {}
        return self.state, reward, done, info
""",
        "src/train/__init__.py": "",
        "src/train/train_ppo.py": """\
\"\"\"Training stub (e.g., PPO).\"\"\"

def train():
    # TODO: ต่อกับ RL env และ lib ที่ต้องใช้ (เช่น SB3)
    print("Training PPO... (TODO)")

if __name__ == "__main__":
    train()
""",
        "src/utils/__init__.py": "",
        "README.md": """\
# automated-trading-ai

โครงสร้างโปรเจกต์สำหรับระบบซื้อขายอัตโนมัติด้วย AI

## โฟลเดอร์สำคัญ
- `data/` ข้อมูลดิบ, คุณลักษณะ, รุ่น
- `reports/` ตาราง/รูปผลลัพธ์
- `src/` โค้ดหลัก (ingest, features, backtest, RL env, training, dashboard)

เริ่มต้นแก้ไขไฟล์ stub ใน `src/` ตาม workflow ของคุณ
""",
        "requirements.txt": """\
# ใส่ไลบรารีตามที่ใช้จริงภายหลัง (ตัวอย่างพื้นฐาน)
pandas
numpy

# ถ้าจะทำ Dashboard ด้วย Streamlit ให้ปลดคอมเมนต์
# streamlit

# ถ้าจะทำ RL ด้วย SB3/Gymnasium ให้ปลดคอมเมนต์ (อาจต้องใช้ Python/OS ที่รองรับ)
# gymnasium
# stable-baselines3
# torch
"""
    }

    for rel, content in files.items():
        write_file(base / rel, content)

    print("\nDone. ✅")

if __name__ == "__main__":
    sys.exit(main())
