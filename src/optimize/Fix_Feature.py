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
N_SET = np.array([13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597], dtype=np.int32)
ALPHA_SET = np.array([1, 2, 3, 5, 8, 11], dtype=np.int32)
BETA_SET = np.array([0.38, 0.5, 0.61, 1, 1.44, 1.61, 2.61], dtype=np.float32)
D_SET = np.array([2, 3, 5, 8, 13], dtype=np.int32)
SRC_SET = np.array([0, 1, 2], dtype=np.int32)  # 0=close, 1=hl2, 2=hlc3

# --- Fast Rolling Functions ---
def _roll_max_gpu(x: "cp.ndarray", window: int) -> "cp.ndarray":
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

def _roll_mean_gpu(x: "cp.ndarray", window: int) -> "cp.ndarray":
    if window <= 1:
        return x.copy()
    sw = cp.lib.stride_tricks.sliding_window_view(x, window)
    out = sw.mean(axis=-1)
    out = cp.pad(out, (window - 1, 0), mode="constant", constant_values=cp.nan)
    return out

def _roll_std_gpu(x: "cp.ndarray", window: int) -> "cp.ndarray":
    if window <= 1:
        return cp.zeros_like(x)
    sw = cp.lib.stride_tricks.sliding_window_view(x, window)
    out = sw.std(axis=-1)
    out = cp.pad(out, (window - 1, 0), mode="constant", constant_values=cp.nan)
    return out

# ----------------------
#  FITNESS (with penalties)
# ----------------------
def calculate_fitness_batch_gpu(
    open_prices, high_prices, low_prices, close_prices, population,
    *, fee: float = 0.0, slippage: float = 0.0,
    min_trades: int | None = None, no_trade_penalty: float = 0.0,
    flat_penalty_per_step: float = 0.0
):
    xp = cp.get_array_module(population)
    assert xp is cp, "This function expects CuPy arrays on GPU"

    n_ind  = int(population.shape[0])
    n_rows = int(open_prices.shape[0])

    if min_trades is None:
        min_trades = max(10, n_rows // 1200)

    n_vals     = population[:, 0].astype(cp.int32)
    alpha_vals = population[:, 1].astype(cp.int32)
    d_vals     = population[:, 2].astype(cp.int32)
    beta_vals  = population[:, 3].astype(cp.float32)
    src_vals   = population[:, 4].astype(cp.int32)

    # เตรียมราคา
    src_close = close_prices
    src_hl2   = (high_prices + low_prices) / 2.0
    src_hlc3  = (high_prices + low_prices + close_prices) / 3.0
    price_range = high_prices - low_prices

    # ====== NEW: กันกรณี n > n_rows ======
    unique_n_all = cp.unique(n_vals)
    valid_n_mask = unique_n_all <= n_rows
    unique_n_valid = unique_n_all[valid_n_mask]
    if unique_n_valid.size == 0:
        # ไม่มี n ไหนพอคำนวณ rolling ได้เลย → คืนผล baseline - โทษ (ถ้ามี)
        cash = cp.ones((n_ind,), dtype=cp.float32) * 100000.0
        if flat_penalty_per_step > 0.0:
            cash -= float(flat_penalty_per_step) * n_rows
        if no_trade_penalty > 0 and min_trades > 0:
            cash -= float(no_trade_penalty) * float(min_trades)
        return cash

    max_n_valid = int(unique_n_valid.max())
    start_idx   = max_n_valid - 1
    # =====================================

    # Precompute เฉพาะ n ที่ valid
    roll_high_max = {}
    roll_low_min  = {}
    roll_std_rng  = {}
    roll_mean_rng = {}
    roll_src_max = {0: {}, 1: {}, 2: {}}
    roll_src_min = {0: {}, 1: {}, 2: {}}

    for n in unique_n_valid.tolist():
        n_int = int(n)
        roll_high_max[n_int] = _roll_max_gpu(high_prices, n_int)
        roll_low_min[n_int]  = _roll_min_gpu(low_prices, n_int)
        roll_std_rng[n_int]  = _roll_std_gpu(price_range, n_int)
        roll_mean_rng[n_int] = _roll_mean_gpu(price_range, n_int)
        roll_src_max[0][n_int] = _roll_max_gpu(src_close, n_int)
        roll_src_min[0][n_int] = _roll_min_gpu(src_close, n_int)
        roll_src_max[1][n_int] = _roll_max_gpu(src_hl2, n_int)
        roll_src_min[1][n_int] = _roll_min_gpu(src_hl2, n_int)
        roll_src_max[2][n_int] = _roll_max_gpu(src_hlc3, n_int)
        roll_src_min[2][n_int] = _roll_min_gpu(src_hlc3, n_int)

    # แผง bands
    upper_mat = cp.empty((n_rows, n_ind), dtype=cp.float32)
    lower_mat = cp.empty((n_rows, n_ind), dtype=cp.float32)

    for j in range(n_ind):
        n_j     = int(n_vals[j])
        alpha_j = float(alpha_vals[j])
        beta_j  = float(beta_vals[j])
        s_j     = int(src_vals[j])  # 0/1/2

        if n_j > n_rows:
            # n ใหญ่กว่าข้อมูล → บังคับ NaN จะไม่มีสัญญาณ/ไม่เทรด
            upper_mat[:, j] = cp.nan
            lower_mat[:, j] = cp.nan
            continue

        highest_src = roll_src_max[s_j][n_j]
        lowest_src  = roll_src_min[s_j][n_j]
        delta_n     = roll_std_rng[n_j]
        highest_hi  = roll_high_max[n_j]
        lowest_lo   = roll_low_min[n_j]

        mu_n        = roll_mean_rng[n_j]
        margin      = mu_n + (beta_j * delta_n)

        h = highest_src - (alpha_j * delta_n)
        l = lowest_src  + (alpha_j * delta_n)
        upper = cp.maximum(highest_hi, h + margin)
        lower = cp.minimum(lowest_lo, l - margin)

        upper_mat[:, j] = upper
        lower_mat[:, j] = lower

    # จำลองเทรด + penalties
    cash = cp.ones((n_ind,), dtype=cp.float32) * 100000.0
    pos  = cp.zeros((n_ind,), dtype=cp.int8)
    trades_count = cp.zeros((n_ind,), dtype=cp.int32)

    open_mat  = cp.broadcast_to(open_prices[:, None],  (n_rows, n_ind))
    close_mat = cp.broadcast_to(close_prices[:, None], (n_rows, n_ind))
    total_cost_rate = fee + slippage

    for i in range(start_idx, n_rows):
        pos_before = pos.copy()
        o  = open_mat[i]
        c  = close_mat[i]
        up = upper_mat[i]
        lo = lower_mat[i]

        # จะได้ False อัตโนมัติเมื่อ up/lo = NaN
        buy_sig  = ((c > lo) & (o < lo)) | ((c > up) & (o < up))
        sell_sig = ((c < up) & (o > up)) | ((c < lo) & (o > lo))

        if flat_penalty_per_step > 0.0:
            flat_idx = (pos_before == 0)
            if flat_idx.any():
                cash[flat_idx] -= flat_penalty_per_step

        buy_idx = (pos == 0) & buy_sig
        if buy_idx.any():
            cash[buy_idx] -= c[buy_idx] * (1.0 + total_cost_rate)
            pos[buy_idx] = 1

        sell_idx = (pos == 1) & sell_sig
        if sell_idx.any():
            cash[sell_idx] += c[sell_idx] * (1.0 - total_cost_rate)
            pos[sell_idx] = 0
            trades_count[sell_idx] += 1

    if pos.any():
        last_price = close_prices[-1]
        idx_open = (pos == 1)
        cash[idx_open] += last_price * (1.0 - total_cost_rate)
        pos[idx_open] = 0

    if no_trade_penalty > 0 and min_trades > 0:
        shortfall = cp.maximum(0, min_trades - trades_count)
        if shortfall.any():
            cash -= shortfall.astype(cp.float32) * float(no_trade_penalty)

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
    low_prices  = cp.array(df['low'].values, dtype=cp.float32)
    close_prices= cp.array(df['close'].values, dtype=cp.float32)

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
        # ✅ เรียกฟิตเนสพร้อม penalties (ไม่แตะส่วนฟีเจอร์)
        fitness_scores = calculate_fitness_batch_gpu(
            open_prices, high_prices, low_prices, close_prices, population,
            fee=0.0005,               # ค่าธรรมเนียม ~0.05%
            slippage=0.0002,          # สลิปเพจเล็กน้อย
            min_trades=max(10, int(close_prices.shape[0]) // 1200),
            no_trade_penalty=100.0,   # โทษหากเทรดต่ำกว่าเกณฑ์ (ต่อดีลที่ขาด)
            flat_penalty_per_step=0.01  # โทษถือเงินสด ต่อแท่ง
        )
        cp.cuda.runtime.deviceSynchronize()

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
        print(
            f"Gen {gen+1:02d}/{generations} | Best: (n={int(best_ind[0])}, α={int(best_ind[1])}, "
            f"d={int(best_ind[2])}, β={best_ind[3]:.2f}, src={src_names[int(best_ind[4])]}) | "
            f"Fitness: {best_fitness:,.2f} | Time: {elapsed:.2f}s | GPU Free: {cp.cuda.Device().mem_info[0]/1e9:.2f} GB",
            flush=True
        )

    src_names = ['close', 'hl2', 'hlc3']
    final_params = (
        int(best_ever_params[0]),
        int(best_ever_params[1]),
        int(best_ever_params[2]),
        float(best_ever_params[3]),
        src_names[int(best_ever_params[4])]
    )
    return final_params, best_ever_fitness

# --- Strategy Feature Generation (ไม่แก้ logic เดิม) ---
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
    mu_n    = df['range'].rolling(window=n).mean()

    highest_src_price_n = df['src_price'].rolling(window=n).max()
    lowest_src_price_n  = df['src_price'].rolling(window=n).min()
    highest_high_n = df['high'].rolling(window=n).max()
    lowest_low_n   = df['low'].rolling(window=n).min()

    margin = mu_n + (beta * delta_n)

    h = highest_src_price_n - (alpha * delta_n)
    l = lowest_src_price_n + (alpha * delta_n)

    df['upper_band_H'] = np.maximum(highest_high_n, h + margin)
    df['lower_band_L'] = np.minimum(lowest_low_n, l - margin)

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

def evaluate_performance_gpu(df, params_tuple):
    """
    Runs a backtest on a given dataframe using a single set of parameters.
    Returns the final equity.
    """
    src_names = ['close', 'hl2', 'hlc3']
    params_list = list(params_tuple)
    params_list[4] = src_names.index(params_tuple[4])  # str -> int

    population = cp.array([params_list], dtype=cp.float32)

    open_p  = cp.array(df['open'].values,  dtype=cp.float32)
    high_p  = cp.array(df['high'].values,  dtype=cp.float32)
    low_p   = cp.array(df['low'].values,  dtype=cp.float32)
    close_p = cp.array(df['close'].values, dtype=cp.float32)

    # ✅ ใช้ penalties/costs เดียวกับรอบ GA
    fitness = calculate_fitness_batch_gpu(
        open_p, high_p, low_p, close_p, population,
        fee=0.0005,
        slippage=0.0002,
        min_trades=max(10, int(close_p.shape[0]) // 1200),
        no_trade_penalty=100.0,
        flat_penalty_per_step=0.01
    )
    cp.cuda.runtime.deviceSynchronize()
    return float(fitness[0])


# --- Main ---
if __name__ == "__main__":
    input_file = Path("data/raw/btc_1m.parquet")
    df_raw = pd.read_parquet(input_file)
    
    n_rows_total = len(df_raw)
    train_end = int(n_rows_total * 0.6)
    val_end = int(n_rows_total * 0.8)
    
    df_train = df_raw.iloc[:train_end].reset_index(drop=True)
    df_val = df_raw.iloc[train_end:val_end].reset_index(drop=True)
    df_test = df_raw.iloc[val_end:].reset_index(drop=True)
    # ✅ Fast test mode
    MAX_ROWS = 20000
    if len(df_raw) > MAX_ROWS:
        df_raw = df_raw.tail(MAX_ROWS).reset_index(drop=True)
        print(f"[FAST TEST] Using last {MAX_ROWS:,} rows")

    print("🚀 PHASE 1: Running FULL GPU-ACCELERATED GA...")
    best_params_from_train, train_score = run_genetic_algorithm_full_gpu(
        df_train,
        population_size=64,
        generations=50,
        mutation_rate=0.2
    )

    print("\n" + "="*80)
    print("✅ GA Finished on Training Set!")
    print(f"Found Params: n={best_params_from_train[0]}, a={best_params_from_train[1]}, d={best_params_from_train[2]}, β={best_params_from_train[3]:.2f}, src='{best_params_from_train[4]}'")
    print(f"💰 Fitness on Train Set: {train_score:,.2f}")
    print("="*80)


    # 🧪 PHASE 2: VALIDATE ON UNSEEN DATA
    print("\n🧪 PHASE 2: Validating parameters on VALIDATION & TEST sets...")
    
    val_score = evaluate_performance_gpu(df_val, best_params_from_train)
    test_score = evaluate_performance_gpu(df_test, best_params_from_train)
    
    print(f"💰 Performance on Val Set:  {val_score:,.2f}")
    print(f"💰 Performance on Test Set: {test_score:,.2f}  <-- This is the most realistic performance estimate.")
    
    # ⚠️ คำเตือนสำคัญ
    if train_score > val_score * 1.5: # ถ้าผลงานใน train ดีกว่า val เกิน 50%
        print("\n⚠️ WARNING: Significant performance drop on validation data.")
        print("   This indicates high risk of overfitting. The found parameters may not be robust.")
    else:
        print("\n✅ Good News: Performance is consistent on validation data. Low risk of overfitting.")

    # 📝 PHASE 3: GENERATE FEATURES FOR RL (using the entire dataset)
    print("\n📝 PHASE 3: Generating final features for RL using the validated parameters...")
    final_df_for_rl = generate_strategy_features(df_raw, best_params_from_train)

    print("\n✅ Done! Feature set preview:")
    print(final_df_for_rl[['open', 'close', 'upper_band_H', 'lower_band_L', 'strat_position']].tail())

    output_path = Path("data/features/btc_1m_rl_features_validated.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df_for_rl.to_parquet(output_path)
    print(f"\n💾 Saved to: {output_path}")