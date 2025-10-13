import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import time

warnings.filterwarnings('ignore', category=FutureWarning)

# ✅ ตรวจสอบ GPU
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cp_nd
    GPU_AVAILABLE = True
    print("✅ CuPy detected - Full GPU acceleration enabled!")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️  CuPy not found. Install with: pip install cupy-cuda12x")
    print("    Falling back to CPU mode...")

# --- Constants ---
N_SET = np.array([13, 21, 34, 55, 89, 144, 233], dtype=np.int32)
ALPHA_SET = np.array([1, 2, 3, 5, 8, 11], dtype=np.int32)
BETA_SET = np.array([0.38, 0.5, 0.61, 1, 1.44, 1.61, 2.61], dtype=np.float32)
D_SET = np.array([2, 3, 5, 8, 13], dtype=np.int32)
SRC_SET = np.array([0, 1, 2], dtype=np.int32)  # 0=close, 1=hl2, 2=hlc3

# --- Fast Rolling Functions ---
def _roll_max_gpu(x: "cp.ndarray", window: int) -> "cp.ndarray":
    # rolling max using sliding window; pad head with NaN so index alignsกับ input
    if window <= 1:
        return x.copy()
    sw = cp.lib.stride_tricks.sliding_window_view(x, window)  # shape: (n - w + 1, w)
    out = sw.max(axis=-1)
    out = cp.pad(out, (window - 1, 0), mode="constant", constant_values=cp.nan)
    return out

def _roll_min_gpu(x: "cp.ndarray", window: int) -> "cp.ndarray":
    if window <= 1:
        return x.copy()
    sw = cp.lib.stride_tricks.sliding_window_view(x, window)
    out = sw.min(axis=-1)
    out = cp.pad(out, (window - 1, 0), mode="constant", constant_values=cp.nan)
    return out

def _roll_std_gpu(x: "cp.ndarray", window: int) -> "cp.ndarray":
    if window <= 1:
        return cp.zeros_like(x)
    sw = cp.lib.stride_tricks.sliding_window_view(x, window)
    out = sw.std(axis=-1)
    out = cp.pad(out, (window - 1, 0), mode="constant", constant_values=cp.nan)
    return out

def calculate_fitness_batch_gpu(open_prices, high_prices, low_prices, close_prices, population,
                                fee=0.0005, slippage=0.0002,
                                min_trades=5, no_trade_penalty=50.0,
                                flat_penalty_per_step=0.0):
    """
    Vectorized fitness บน GPU (เร็ว) + intrabar cross + ค่าธรรมเนียม/สลิปเพจ + ลงโทษ no-trade
    - fee/slippage: ใช้แบบง่าย ๆ ต่อธุรกรรม
    - min_trades: หากเทรดน้อยกว่า คิดโทษ (เป็นหน่วย $)
    - flat_penalty_per_step: ค่าปรับถือเงินสดต่อบาร์ (เช่น 0.01 = หัก 1 cent ต่อบาร์) ใส่ 0 ถ้าไม่ใช้
    """
    xp = cp.get_array_module(population)
    assert xp is cp, "This function expects CuPy arrays on GPU"

    n_ind = int(population.shape[0])
    n_rows = int(open_prices.shape[0])

    # params
    n_vals     = population[:, 0].astype(cp.int32)
    alpha_vals = population[:, 1].astype(cp.int32)
    d_vals     = population[:, 2].astype(cp.int32)
    beta_vals  = population[:, 3].astype(cp.float32)
    src_vals   = population[:, 4].astype(cp.int32)  # 0=close,1=hl2,2=hlc3

    # sources
    src_close = close_prices
    src_hl2   = (high_prices + low_prices) / 2.0
    src_hlc3  = (high_prices + low_prices + close_prices) / 3.0
    price_range = high_prices - low_prices

    unique_n = cp.unique(n_vals)
    max_n = int(unique_n.max())
    start_idx = max_n - 1

    # rolling precompute ต่อ n ที่จำเป็น
    roll_high_max = {}
    roll_low_min  = {}
    roll_std_rng  = {}
    roll_src_max = {0: {}, 1: {}, 2: {}}
    roll_src_min = {0: {}, 1: {}, 2: {}}

    def _roll_max_gpu(x, w):
        if w <= 1: return x.copy()
        sw = cp.lib.stride_tricks.sliding_window_view(x, w)
        out = sw.max(axis=-1)
        return cp.pad(out, (w-1, 0), mode="constant", constant_values=cp.nan)

    def _roll_min_gpu(x, w):
        if w <= 1: return x.copy()
        sw = cp.lib.stride_tricks.sliding_window_view(x, w)
        out = sw.min(axis=-1)
        return cp.pad(out, (w-1, 0), mode="constant", constant_values=cp.nan)

    def _roll_std_gpu(x, w):
        if w <= 1: return cp.zeros_like(x)
        sw = cp.lib.stride_tricks.sliding_window_view(x, w)
        out = sw.std(axis=-1)
        return cp.pad(out, (w-1, 0), mode="constant", constant_values=cp.nan)

    for n in unique_n.tolist():
        n = int(n)
        roll_high_max[n] = _roll_max_gpu(high_prices, n)
        roll_low_min[n]  = _roll_min_gpu(low_prices, n)
        roll_std_rng[n]  = _roll_std_gpu(price_range, n)
        # src max/min
        roll_src_max[0][n] = _roll_max_gpu(src_close, n)
        roll_src_min[0][n] = _roll_min_gpu(src_close, n)
        roll_src_max[1][n] = _roll_max_gpu(src_hl2,   n)
        roll_src_min[1][n] = _roll_min_gpu(src_hl2,   n)
        roll_src_max[2][n] = _roll_max_gpu(src_hlc3,  n)
        roll_src_min[2][n] = _roll_min_gpu(src_hlc3,  n)

    # build bands per individual
    upper_mat = cp.empty((n_rows, n_ind), dtype=cp.float32)
    lower_mat = cp.empty((n_rows, n_ind), dtype=cp.float32)
    for j in range(n_ind):
        n    = int(n_vals[j])
        beta = float(beta_vals[j])
        s    = int(src_vals[j])

        highest_src = roll_src_max[s][n]
        lowest_src  = roll_src_min[s][n]
        delta_n     = roll_std_rng[n]
        highest_hi  = roll_high_max[n]
        lowest_lo   = roll_low_min[n]

        h = highest_src - (beta * delta_n)
        l = lowest_src  + (beta * delta_n)
        upper = cp.maximum(highest_hi, h + 0.001)
        lower = cp.minimum(lowest_lo, l - 0.001)

        upper_mat[:, j] = upper
        lower_mat[:, j] = lower

    # simulate trading (vectorized on individuals; loop only on time)
    cash = cp.ones((n_ind,), dtype=cp.float32) * 100000.0
    pos  = cp.zeros((n_ind,), dtype=cp.int8)
    trades_count = cp.zeros((n_ind,), dtype=cp.int32)

    open_mat  = cp.broadcast_to(open_prices[:, None],  (n_rows, n_ind))
    high_mat  = cp.broadcast_to(high_prices[:, None],  (n_rows, n_ind))
    low_mat   = cp.broadcast_to(low_prices[:, None],   (n_rows, n_ind))
    close_mat = cp.broadcast_to(close_prices[:, None], (n_rows, n_ind))

    buy_cost_mult  = (1.0 + fee + slippage)
    sell_pro_mult  = (1.0 - fee - slippage)

    for i in range(start_idx, n_rows):
        o  = open_mat[i]
        h_ = high_mat[i]
        l_ = low_mat[i]
        c  = close_mat[i]
        up = upper_mat[i]
        lo = lower_mat[i]

        # intrabar cross (เปิดโอกาสให้เกิดสัญญาณมากขึ้น)
        buy_sig  = ((l_ <= lo) & (c > lo))  | ((l_ <= up) & (c > up))
        sell_sig = ((h_ >= up) & (c < up))  | ((h_ >= lo) & (c < lo))

        # flat penalty (ถือเงินสด)
        if flat_penalty_per_step > 0.0:
            cash[pos == 0] -= flat_penalty_per_step

        # buy
        buy_idx = (pos == 0) & buy_sig
        if buy_idx.any():
            cash[buy_idx] -= c[buy_idx] * buy_cost_mult
            pos[buy_idx] = 1
            trades_count[buy_idx] += 1

        # sell
        sell_idx = (pos == 1) & sell_sig
        if sell_idx.any():
            cash[sell_idx] += c[sell_idx] * sell_pro_mult
            pos[sell_idx] = 0
            trades_count[sell_idx] += 1

    # close leftover
    cash += pos.astype(cp.float32) * close_prices[-1] * sell_pro_mult

    # no-trade penalty (ลงโทษถ้าเทรดน้อยกว่า min_trades)
    shortfall = cp.maximum(0, (min_trades - trades_count))
    cash -= shortfall.astype(cp.float32) * no_trade_penalty

    return cash


# --- GA Components ---
def tournament_selection_gpu(population, fitness_scores, tournament_size=5, n_parents=None):
    xp = cp.get_array_module(population)
    if n_parents is None:
        n_parents = population.shape[0]
    selected = xp.zeros((n_parents, population.shape[1]), dtype=population.dtype)
    for i in range(n_parents):
        idxs = xp.random.choice(population.shape[0], tournament_size, replace=False)
        best_idx = idxs[xp.argmax(fitness_scores[idxs])]
        selected[i] = population[best_idx]
    return selected

def crossover_mutation_gpu(parents, mutation_rate=0.2):
    xp = cp.get_array_module(parents)
    n_offspring = parents.shape[0]
    offspring = xp.zeros_like(parents)

    for i in range(0, n_offspring, 2):
        if i + 1 < n_offspring:
            p1 = parents[i]
            p2 = parents[i + 1]
            offspring[i, 0] = p1[0]
            offspring[i, 1] = p2[1]
            offspring[i, 2] = p1[2]
            offspring[i, 3] = p2[3]
            offspring[i, 4] = p1[4]
            offspring[i + 1] = p2

    for i in range(n_offspring):
        if float(xp.random.random()) < mutation_rate:
            gene = int(xp.random.randint(0, 5))
            if gene == 0:
                offspring[i, 0] = N_SET[int(xp.random.randint(0, len(N_SET)))]
            elif gene == 1:
                offspring[i, 1] = ALPHA_SET[int(xp.random.randint(0, len(ALPHA_SET)))]
            elif gene == 2:
                offspring[i, 2] = D_SET[int(xp.random.randint(0, len(D_SET)))]
            elif gene == 3:
                offspring[i, 3] = BETA_SET[int(xp.random.randint(0, len(BETA_SET)))]
            else:
                offspring[i, 4] = SRC_SET[int(xp.random.randint(0, len(SRC_SET)))]
    return offspring

# --- Main GA ---
def run_genetic_algorithm_full_gpu(df, population_size=128, generations=20, mutation_rate=0.2):
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU not available. Please install cupy-cuda and re-run.")

    print(f"🚀 FULL GPU MODE - All operations on GPU")
    dev_id = cp.cuda.Device().id
    gpu_name = cp.cuda.runtime.getDeviceProperties(dev_id)["name"].decode("utf-8")
    print(f"   GPU: {gpu_name}")
    print(f"   Memory: {cp.cuda.Device().mem_info[1] / 1e9:.2f} GB total")

    open_prices = cp.array(df['open'].values, dtype=cp.float32)
    high_prices = cp.array(df['high'].values, dtype=cp.float32)
    low_prices = cp.array(df['low'].values, dtype=cp.float32)
    close_prices = cp.array(df['close'].values, dtype=cp.float32)

    # Init population
    population = cp.zeros((population_size, 5), dtype=cp.float32)
    for i in range(population_size):
        population[i, 0] = N_SET[np.random.randint(len(N_SET))]
        population[i, 1] = ALPHA_SET[np.random.randint(len(ALPHA_SET))]
        population[i, 2] = D_SET[np.random.randint(len(D_SET))]
        population[i, 3] = BETA_SET[np.random.randint(len(BETA_SET))]
        population[i, 4] = SRC_SET[np.random.randint(len(SRC_SET))]

    best_ever_fitness = 0
    best_ever_params = None

    for gen in range(generations):
        t0 = time.time()
        fitness_scores = calculate_fitness_batch_gpu(
        open_prices, high_prices, low_prices, close_prices, population,
        fee=0.0005,               # ค่าธรรมเนียมฝั่งฟิวเจอร์ส ~0.05%
        slippage=0.0002,          # สลิปเพจนิดหน่อย
        min_trades=max(10, int(close_prices.shape[0]) // 1200),  # อย่างน้อย ~1 ดีลต่อ 300 แท่ง
        no_trade_penalty=100.0,   # โทษถ้าเทรดน้อยกว่ากำหนด
        flat_penalty_per_step=0.01  # ถ้าถือเงินสด หักต่อแท่ง (จะปิดก็ได้ตั้งเป็น 0.0)
        )
        cp.cuda.runtime.deviceSynchronize()  # ✅ sync GPU

        best_idx = int(cp.argmax(fitness_scores))
        best_fitness = float(fitness_scores[best_idx])

        if best_fitness > best_ever_fitness:
            best_ever_fitness = best_fitness
            best_ever_params = cp.asnumpy(population[best_idx])

        selected = tournament_selection_gpu(population, fitness_scores)
        population = crossover_mutation_gpu(selected, mutation_rate)

        best_ind = cp.asnumpy(population[best_idx])
        src_names = ['close', 'hl2', 'hlc3']
        elapsed = time.time() - t0
        print(f"Gen {gen+1:02d}/{generations} | Best: (n={int(best_ind[0])}, α={int(best_ind[1])}, d={int(best_ind[2])}, β={best_ind[3]:.2f}, src={src_names[int(best_ind[4])]}) | Fitness: {best_fitness:,.2f} | Time: {elapsed:.2f}s | GPU Free: {cp.cuda.Device().mem_info[0]/1e9:.2f} GB", flush=True)

    src_names = ['close', 'hl2', 'hlc3']
    final_params = (
        int(best_ever_params[0]),
        int(best_ever_params[1]),
        int(best_ever_params[2]),
        float(best_ever_params[3]),
        src_names[int(best_ever_params[4])]
    )
    return final_params, best_ever_fitness

# --- Strategy Feature Generation ---
def create_features_with_params(df_original, params):
    n, alpha, d, beta, src = params
    df = df_original.copy()

    if src == 'hl2':
        df['src_price'] = (df['high'] + df['low']) / 2
    elif src == 'hlc3':
        df['src_price'] = (df['high'] + df['low'] + df['close']) / 3
    else:
        df['src_price'] = df['close']

    df['range'] = df['high'] - df['low']
    delta_n = df['range'].rolling(window=n).std()
    highest_src_price_n = df['src_price'].rolling(window=n).max()
    h = highest_src_price_n - (beta * delta_n)
    df['H'] = np.maximum(df['high'].rolling(window=n).max(), h + 0.001)

    lowest_src_price_n = df['src_price'].rolling(window=n).min()
    l = lowest_src_price_n + (beta * delta_n)
    df['L'] = np.minimum(df['low'].rolling(window=n).min(), l - 0.001)

    df['MA_fast'] = df['src_price'].rolling(window=alpha).mean()
    df['MA_slow'] = df['src_price'].rolling(window=d).mean()
    df.rename(columns={'H': 'upper_band_H', 'L': 'lower_band_L'}, inplace=True)
    return df.dropna()

def generate_strategy_features(df_original, best_params):
    print("\nGenerating strategy performance features with best params...")
    features_df = create_features_with_params(df_original, best_params)

    positions, equity_curve = [], []
    cash, position = 100000.0, 0

    for i, row in features_df.iterrows():
        current_equity = cash + (position * row['close'])
        equity_curve.append(current_equity)
        positions.append(position)

        open_price = row['open']
        close_price = row['close']
        upper_band = row['upper_band_H']
        lower_band = row['lower_band_L']

        buy_signal = (close_price > lower_band and open_price < lower_band) or (close_price > upper_band and open_price < upper_band)
        sell_signal = (close_price < upper_band and open_price > upper_band) or (close_price < lower_band and open_price > lower_band)

        if position == 0 and buy_signal:
            position, cash = 1, cash - close_price
        elif position == 1 and sell_signal:
            position, cash = 0, cash + close_price

    features_df['strat_position'] = positions
    features_df['strat_equity'] = equity_curve
    features_df['strat_returns'] = pd.Series(equity_curve).pct_change().values
    peak = features_df['strat_equity'].expanding(min_periods=1).max()
    features_df['strat_drawdown'] = (features_df['strat_equity'] - peak) / peak
    return features_df.fillna(0)

# --- Main ---
if __name__ == "__main__":
    input_file = Path("data/raw/btc_1h.parquet")
    df_raw = pd.read_parquet(input_file)

    # ✅ Fast test mode
    MAX_ROWS = 10000
    if len(df_raw) > MAX_ROWS:
        df_raw = df_raw.tail(MAX_ROWS).reset_index(drop=True)
        print(f"[FAST TEST] Using last {MAX_ROWS:,} rows")

    print("🚀 PHASE 1: Running FULL GPU-ACCELERATED GA...")
    best_params, best_score = run_genetic_algorithm_full_gpu(
        df_raw,
        population_size=64,
        generations=15,
        mutation_rate=0.2
    )

    print("\n" + "="*80)
    print("✅ GA Finished!")
    print(f"🏆 Best Params: n={best_params[0]}, a={best_params[1]}, d={best_params[2]}, β={best_params[3]:.2f}, src='{best_params[4]}'")
    print(f"💰 Best Fitness: {best_score:,.2f}")
    print("="*80)

    print("\n🚀 PHASE 2: Generating final DataFrame with strategy features...")
    final_df_for_rl = generate_strategy_features(df_raw, best_params)

    print("\n✅ Done! Feature set preview:")
    print(final_df_for_rl[['open', 'close', 'upper_band_H', 'lower_band_L', 'strat_position']].tail())

    output_path = Path("data/features/btc_1h_rl_features_advanced.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df_for_rl.to_parquet(output_path)
    print(f"\n💾 Saved to: {output_path}")
