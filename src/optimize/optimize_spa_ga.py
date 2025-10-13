# src/optimize/optimize_spa_ga.py
from __future__ import annotations

import json
import random
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd

# --- Search Spaces (คุณปรับได้) ---
N_SET = [13, 21, 34, 55, 89, 144, 233]
ALPHA_SET = [3, 5, 8, 13]
BETA_SET = [0.38, 0.5, 0.61, 1.0, 1.44]
D_SET = [5, 8, 13, 21, 34]
SRC_SET = ['close', 'hl2', 'hlc3']


def _ensure_dtindex(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    for c in ["timestamp", "time", "open_time", "date", "datetime"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
            df = df.set_index(c)
            break
    return df.sort_index()


def create_features_with_params(df_original: pd.DataFrame, params: Tuple[int, int, int, float, str]) -> pd.DataFrame | None:
    """
    params = (n, alpha, d, beta, src)
    n     : window สำหรับคำนวณ high/low range & std
    alpha : fast MA window
    d     : slow MA window
    beta  : ตัวคูณความผันผวนสำหรับขยับ H/L
    src   : close / hl2 / hlc3
    """
    n, alpha, d, beta, src = params
    df = df_original.copy()

    if alpha >= d or n < 2:
        return None

    if src == 'hl2':
        df['src_price'] = (df['high'] + df['low']) / 2
    elif src == 'hlc3':
        df['src_price'] = (df['high'] + df['low'] + df['close']) / 3
    else:
        df['src_price'] = df['close']

    # ความผันผวนแบบง่าย: ส่วนเบี่ยงเบนมาตรฐานของ (high-low) ย้อนหลัง n
    df['range'] = (df['high'] - df['low']).astype('float64')
    delta_n = df['range'].rolling(window=n, min_periods=n).std()

    # เส้นบน/ล่างแบบ adaptive
    highest_src_price_n = df['src_price'].rolling(window=n, min_periods=n).max()
    h = highest_src_price_n - (beta * delta_n)
    H = np.maximum(df['high'].rolling(window=n, min_periods=n).max(), h + 1e-6)

    lowest_src_price_n = df['src_price'].rolling(window=n, min_periods=n).min()
    l = lowest_src_price_n + (beta * delta_n)
    L = np.minimum(df['low'].rolling(window=n, min_periods=n).min(), l - 1e-6)

    df['upper_band_H'] = H
    df['lower_band_L'] = L

    # MAs
    df['MA_fast'] = df['src_price'].rolling(window=alpha, min_periods=alpha).mean()
    df['MA_slow'] = df['src_price'].rolling(window=d, min_periods=d).mean()

    return df.dropna()


def calc_fitness_final_capital(df_original: pd.DataFrame, params: Tuple[int, int, int, float, str]) -> float:
    feats = create_features_with_params(df_original, params)
    if feats is None or feats.empty:
        return 0.0

    cash, position = 100_000.0, 0
    for _, row in feats.iterrows():
        open_price = float(row['open'])
        close_price = float(row['close'])
        upper_band = float(row['upper_band_H'])
        lower_band = float(row['lower_band_L'])

        buy_rejection = (close_price > lower_band) and (open_price < lower_band)
        buy_breakout  = (close_price > upper_band) and (open_price < upper_band)
        buy_signal = buy_rejection or buy_breakout

        sell_rejection = (close_price < upper_band) and (open_price > upper_band)
        sell_breakout  = (close_price < lower_band) and (open_price > lower_band)
        sell_signal = sell_rejection or sell_breakout

        if position == 0 and buy_signal:
            position, cash = 1, cash - close_price
        elif position == 1 and sell_signal:
            position, cash = 0, cash + close_price

    if position == 1 and not feats.empty:
        cash += float(feats['close'].iloc[-1])

    return cash


def _evaluate_individual(individual, df, memo):
    # map individual -> params tuple
    params = (individual[0], individual[1], individual[3], individual[2], individual[4])
    if params not in memo:
        memo[params] = calc_fitness_final_capital(df, params)
    return params, memo[params]


def run_ga(
    df: pd.DataFrame,
    population_size: int = 80,
    generations: int = 30,
    mutation_rate: float = 0.2,
    n_processes: int | None = None
):
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)
    print(f"[info] Using {n_processes} parallel workers")

    # init population
    population = [
        (random.choice(N_SET), random.choice(ALPHA_SET), random.choice(BETA_SET),
        random.choice(D_SET), random.choice(SRC_SET))
        for _ in range(population_size)
    ]

    memo: Dict[tuple, float] = {}
    best_params, best_score = None, -np.inf

    for gen in range(1, generations + 1):
        with Pool(processes=n_processes) as pool:
            eval_fn = partial(_evaluate_individual, df=df, memo=memo)
            results = pool.map(eval_fn, population)

        # extract fitness
        fitness_scores = [score for _, score in results]
        # tournament selection
        selected = [
            max(random.sample(list(zip(population, fitness_scores)), k=5), key=lambda x: x[1])[0]
            for _ in range(population_size)
        ]

        # crossover + mutation
        offspring = []
        for i in range(0, population_size, 2):
            p1, p2 = selected[i], selected[min(i + 1, population_size - 1)]
            child = (p1[0], p2[1], p1[2], p2[3], p1[4])  # simple 1-point mix
            if random.random() < mutation_rate:
                idx = random.randint(0, 4)
                child = list(child)
                if idx == 0: child[0] = random.choice(N_SET)
                elif idx == 1: child[1] = random.choice(ALPHA_SET)
                elif idx == 2: child[2] = random.choice(BETA_SET)
                elif idx == 3: child[3] = random.choice(D_SET)
                else: child[4] = random.choice(SRC_SET)
                child = tuple(child)
            offspring.extend([child, p2])
        population = offspring[:population_size]

        # track best
        gen_best_idx = int(np.argmax(fitness_scores))
        gen_best = population[gen_best_idx]
        gen_best_params = (gen_best[0], gen_best[1], gen_best[3], gen_best[2], gen_best[4])  # map to (n,alpha,d,beta,src)
        gen_best_score = fitness_scores[gen_best_idx]
        if gen_best_score > best_score:
            best_score, best_params = gen_best_score, gen_best_params

        print(f"Gen {gen:02d}/{generations} | Best: n={gen_best_params[0]}, α={gen_best_params[1]}, d={gen_best_params[2]}, β={gen_best_params[3]:.2f}, src={gen_best_params[4]} | Fitness: {gen_best_score:,.2f}")

    return best_params, best_score


def main():
    apath = Path("data/raw/btc_1h.parquet")
    if not apath.exists():
        raise FileNotFoundError(f"Raw parquet missing: {apath}")

    df = pd.read_parquet(apath)
    df = _ensure_dtindex(df)
    needed = ["open", "high", "low", "close"]
    if any(c not in df.columns for c in needed):
        raise ValueError(f"Missing columns: {needed}")

    print("[info] Running GA optimization...")
    best_params, best_score = run_ga(df, population_size=80, generations=30, mutation_rate=0.2)

    out_dir = Path("data/params"); out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "best_spa_ga.json"
    payload = {
        "n": int(best_params[0]),
        "alpha": int(best_params[1]),
        "d": int(best_params[2]),
        "beta": float(best_params[3]),
        "src": str(best_params[4]),
        "fitness_final_capital": float(best_score)
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"[ok] Saved best params → {out_json}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
