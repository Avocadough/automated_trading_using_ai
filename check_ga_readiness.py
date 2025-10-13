# save as: tools/check_ga_readiness.py
import pandas as pd
import numpy as np
from pathlib import Path

def evaluate_ga_df(path: Path, split=0.8, fee=0.0005, slippage=0.0002):
    df = pd.read_parquet(path).reset_index(drop=True)
    # ใช้กฎจาก GA (strat_position: 0/1 ถือเงินสด/ถือ long 1 หน่วย)
    # คิด PnL แบบง่าย + ค่าธรรมเนียม/สลิปเพจ
    n = len(df)
    n_train = int(n*split)
    oos = df.iloc[n_train:].copy()

    # คำนวณธุรกรรมจาก position change
    pos = oos['strat_position'].astype(int).values
    close = oos['close'].astype(float).values

    trades = []
    cash = 100000.0
    position = 0
    entry_price = np.nan

    for i in range(len(oos)):
        # สลับสถานะ?
        if i>0 and pos[i] != pos[i-1]:
            # ปิดก่อน
            if position==1:
                cash += close[i]*(1 - fee - slippage)
                trades.append((entry_price, close[i]))
                entry_price = np.nan
                position = 0
            # เปิดใหม่
            if pos[i]==1:
                cash -= close[i]*(1 + fee + slippage)
                entry_price = close[i]
                position = 1

    # ปิดค้างท้าย
    if position==1:
        cash += close[-1]*(1 - fee - slippage)
        trades.append((entry_price, close[-1]))

    # คำนวณเมตริก
    oos_ret = (cash/100000.0) - 1
    eq = [100000.0]
    position = 0
    for i in range(len(oos)):
        position = pos[i] if i==0 else pos[i-1]
        eq.append(eq[-1] + (pos[i]- (pos[i-1] if i>0 else 0))* (-close[i]*(1+fee+slippage) if pos[i]> (pos[i-1] if i>0 else 0) else 0))
    # ใช้วิธีง่ายกว่า: สร้าง equity จากสัญญาณย่อยๆ ได้ซับซ้อนเกินไปสำหรับสรุปสั้น
    # เอา returns รายแท่งแบบ proxy:
    r = oos['strat_returns'].fillna(0).values  # มีจากไฟล์ที่คุณสร้าง
    if r.std() > 0:
        sharpe = (r.mean()/r.std()) * np.sqrt(24*365)  # annualize 1H
    else:
        sharpe = 0.0

    # MaxDD จาก strat_equity ถ้ามี
    if 'strat_equity' in oos.columns:
        eq_series = oos['strat_equity'].values
        peak = np.maximum.accumulate(eq_series)
        mdd = np.max((peak - eq_series)/peak) if len(eq_series) else np.nan
    else:
        mdd = np.nan

    n_trades = len(trades)
    wins = sum(1 for en,ex in trades if ex>en)
    winrate = (wins/n_trades) if n_trades>0 else 0.0
    exposure = oos['strat_position'].mean()  # สัดส่วนเวลาที่ถือสถานะ

    # Buy&Hold OOS
    bh = (oos['close'].iloc[-1] / oos['close'].iloc[0]) - 1

    print(f"OOS rows: {len(oos):,}")
    print(f"Trades: {n_trades} | Win rate: {winrate*100:.1f}% | Exposure: {exposure*100:.1f}%")
    print(f"OOS Return: {oos_ret*100:.2f}% | Sharpe: {sharpe:.2f} | MaxDD: {mdd*100:.2f}%")
    print(f"Buy&Hold OOS: {bh*100:.2f}%")

if __name__ == "__main__":
    evaluate_ga_df(Path('data/features/btc_1h_rl_features_advanced.parquet'))
