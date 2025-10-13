# src/backtest/baseline_backtest_H1.py
import pandas as pd
import numpy as np
import vectorbt as vbt
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """พยายามทำให้ DataFrame มี DatetimeIndex ที่เรียงเวลา"""
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    for col in ('timestamp', 'time', 'open_time', 'date', 'Date', 'datetime'):
        if col in df.columns:
            idx = pd.to_datetime(df[col], errors='coerce', utc=True)
            if idx.notna().all():
                return df.set_index(idx).sort_index()
    df.index = pd.to_datetime(df.index, errors='coerce', utc=True)
    return df.sort_index()

def check_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """ตรวจสอบและทำความสะอาดข้อมูล"""
    print("\n" + "="*60)
    print("DATA QUALITY CHECK")
    print("="*60)
    
    initial_len = len(df)
    print(f"Initial rows: {initial_len:,}")
    
    # ตรวจสอบ missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"\nMissing values found:")
        print(missing[missing > 0])
    
    # ตรวจสอบ duplicates
    duplicates = df.index.duplicated().sum()
    if duplicates > 0:
        print(f"\n⚠️  Found {duplicates} duplicate timestamps - removing...")
        df = df[~df.index.duplicated(keep='first')]
    
    # ตรวจสอบ gaps ในข้อมูล
    if isinstance(df.index, pd.DatetimeIndex):
        time_diff = df.index.to_series().diff()
        expected_freq = time_diff.mode()[0] if len(time_diff.mode()) > 0 else pd.Timedelta('1h')
        gaps = time_diff[time_diff > expected_freq * 2]
        if len(gaps) > 0:
            print(f"\n⚠️  Found {len(gaps)} time gaps (>2x expected frequency)")
            print(f"Expected frequency: {expected_freq}")
            print(f"Largest gap: {time_diff.max()}")
    
    # ตรวจสอบ price anomalies
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            # Remove zeros and negative prices
            invalid = (df[col] <= 0).sum()
            if invalid > 0:
                print(f"\n⚠️  Found {invalid} invalid {col} prices (<= 0) - removing...")
                df = df[df[col] > 0]
            
            # Check for extreme outliers (>100x median)
            median = df[col].median()
            outliers = ((df[col] > median * 100) | (df[col] < median / 100)).sum()
            if outliers > 0:
                print(f"\n⚠️  Found {outliers} extreme outliers in {col}")
    
    # Ensure OHLC logic
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        invalid_ohlc = (
            (df['high'] < df['low']) | 
            (df['high'] < df['close']) | 
            (df['high'] < df['open']) |
            (df['low'] > df['close']) |
            (df['low'] > df['open'])
        ).sum()
        if invalid_ohlc > 0:
            print(f"\n⚠️  Found {invalid_ohlc} rows with invalid OHLC logic - removing...")
            df = df[
                (df['high'] >= df['low']) &
                (df['high'] >= df['close']) &
                (df['high'] >= df['open']) &
                (df['low'] <= df['close']) &
                (df['low'] <= df['open'])
            ]
    
    final_len = len(df)
    removed = initial_len - final_len
    
    print(f"\nFinal rows: {final_len:,}")
    print(f"Removed: {removed:,} ({removed/initial_len*100:.2f}%)")
    print("="*60 + "\n")
    
    return df

def vectorized_improved_ma_v2(df: pd.DataFrame, fast_windows: np.ndarray, slow_windows: np.ndarray,
                            min_holding_periods: int = 2, use_volume_filter: bool = False):
    """
    Improved MA strategy v2 with:
    - Minimum holding period (reduce overtrading)
    - Optional volume filter
    - Better signal generation
    """
    close = df['close']
    
    # Calculate all MAs
    fast_ma = vbt.MA.run(close, window=fast_windows, short_name='fast').ma
    slow_ma = vbt.MA.run(close, window=slow_windows, short_name='slow').ma
    
    fast_ma = fast_ma.reindex(close.index).sort_index()
    slow_ma = slow_ma.reindex(close.index).sort_index()
    
    print(f"Fast MA shape: {fast_ma.shape}, Slow MA shape: {slow_ma.shape}")
    
    # Vectorized operations
    f = fast_ma.values[..., None]
    s = slow_ma.values[:, None, :]
    
    # MA crosses with confirmation (wait 1 bar)
    prev_below = f[:-2] < s[:-2]
    curr_below = f[1:-1] < s[1:-1]
    next_above = f[2:] >= s[2:]
    cross_up_3d = prev_below & curr_below & next_above
    
    prev_above = f[:-2] > s[:-2]
    curr_above = f[1:-1] > s[1:-1]
    next_below = f[2:] <= s[2:]
    cross_down_3d = prev_above & curr_above & next_below
    
    # Volume filter (if available)
    if use_volume_filter and 'volume' in df.columns:
        volume = df['volume']
        vol_ma = volume.rolling(window=20).mean()
        high_volume = (volume > vol_ma * 1.2).values[1:-1, None, None]
        cross_up_3d = cross_up_3d & high_volume
        cross_down_3d = cross_down_3d & high_volume
    
    # Pad to match original length
    pad_front = np.zeros((1, cross_up_3d.shape[1], cross_up_3d.shape[2]), dtype=bool)
    pad_back = np.zeros((1, cross_up_3d.shape[1], cross_up_3d.shape[2]), dtype=bool)
    
    long_entries_3d = np.vstack([pad_front, cross_up_3d, pad_back])
    long_exits_3d = np.vstack([pad_front, cross_down_3d, pad_back])
    
    # Convert to DataFrame
    cols = pd.MultiIndex.from_product([fast_windows, slow_windows], names=['fast', 'slow'])
    T = len(close)
    
    entries = pd.DataFrame(long_entries_3d.reshape(T, -1), index=close.index, columns=cols)
    exits = pd.DataFrame(long_exits_3d.reshape(T, -1), index=close.index, columns=cols)
    
    # Filter valid pairs
    valid = [c for c in entries.columns if c[0] < c[1]]
    entries = entries[valid]
    exits = exits[valid]
    
    # Debug info
    total_entries = entries.sum().sum()
    total_exits = exits.sum().sum()
    print(f"\nSignals generated:")
    print(f"  Total entries: {total_entries:,}")
    print(f"  Total exits: {total_exits:,}")
    print(f"  Avg entries per strategy: {total_entries / len(valid):.1f}")
    
    if entries.shape[1] == 0:
        raise ValueError("No valid (fast, slow) pairs after filtering fast < slow.")
    
    return entries, exits

def run_advanced_baseline_backtest(raw_data_path: Path, fast_win_range, slow_win_range, 
                                position_size=0.05, min_holding_periods=2):
    """
    Advanced baseline backtest - FIXED VERSION
    """
    if not raw_data_path.exists():
        print(f"Error: Raw data file not found at {raw_data_path}")
        return

    print(f"Loading data from {raw_data_path}...")
    df = pd.read_parquet(raw_data_path)
    df = ensure_datetime_index(df)
    
    # Data quality check
    df = check_data_quality(df)

    if 'close' not in df.columns:
        raise KeyError("Input data must contain a 'close' column.")
    
    close_price = df['close'].astype(float)
    
    print(f"\nData Summary:")
    print(f"  Period: {df.index[0]} to {df.index[-1]}")
    print(f"  Duration: {(df.index[-1] - df.index[0]).days} days")
    print(f"  Total rows: {len(df):,}")
    print(f"  Price range: [{close_price.min():.2f}, {close_price.max():.2f}]")

    # Define param grids
    fast_windows = np.arange(fast_win_range[0], fast_win_range[1], fast_win_range[2]).astype(int)
    slow_windows = np.arange(slow_win_range[0], slow_win_range[1], slow_win_range[2]).astype(int)

    print(f"\n{'='*60}")
    print(f"MA CROSSOVER STRATEGY OPTIMIZATION")
    print(f"{'='*60}")
    print(f"Fast MA periods: {fast_windows.tolist()}")
    print(f"Slow MA periods: {slow_windows.tolist()}")
    print(f"Total combinations: {len(fast_windows) * len(slow_windows)}")
    print(f"Position size: {position_size*100:.0f}%")
    print(f"Min holding period: {min_holding_periods} bars")
    
    entries, exits = vectorized_improved_ma_v2(
        df, fast_windows, slow_windows,
        min_holding_periods=min_holding_periods,
        use_volume_filter=('volume' in df.columns)
    )

    # Check if any entries exist
    total_entries = entries.sum().sum()
    if total_entries == 0:
        print("\n❌ ERROR: NO ENTRIES GENERATED!")
        return

    # Detect frequency
    name_lower = raw_data_path.name.lower()
    freq = '1h' if '1h' in name_lower else ('1min' if '1m' in name_lower else pd.infer_freq(close_price.index))
    if freq is None or freq == '1H':
        freq = '1h'

    # Portfolio with realistic settings
    print(f"\nRunning portfolio simulation...")
    print(f"  Frequency: {freq}")
    print(f"  Fees: 0.06%")
    print(f"  Slippage: 0.03%")
    
    portfolio = vbt.Portfolio.from_signals(
        close_price,
        entries,
        exits,
        fees=0.0006,
        slippage=0.0003,
        freq=freq,
        size=position_size,
        size_type='percent',
        init_cash=10000,
        upon_opposite_entry='ignore'
    )

    # Statistics
    sharpe = portfolio.sharpe_ratio()
    total_return = portfolio.total_return()
    max_dd = portfolio.max_drawdown()
    
    # Filter valid strategies
    trades_count = portfolio.trades.count()
    strategies_with_trades = trades_count[trades_count > 0]
    
    print(f"\nStrategies with trades: {len(strategies_with_trades)} / {len(trades_count)}")
    
    if len(strategies_with_trades) == 0:
        print("❌ No strategies generated any trades!")
        return
    
    # Find best strategy
    valid_strategies = total_return[strategies_with_trades.index]
    profitable_strategies = valid_strategies[valid_strategies > 0]
    
    if len(profitable_strategies) == 0:
        print("\n⚠️  No profitable strategies found.")
        best_col = valid_strategies.idxmax()
    else:
        best_col = sharpe[profitable_strategies.index].idxmax()
    
    best_stats = portfolio[best_col].stats()
    buy_and_hold_return = (close_price.iloc[-1] / close_price.iloc[0] - 1) * 100

    # Print results
    print("\n" + "="*70)
    print(" " * 15 + "BACKTEST RESULTS")
    print("="*70)
    print(f"\n📊 Best Strategy: Fast MA={best_col[0]}, Slow MA={best_col[1]}")
    print("-" * 70)
    print(f"{'Metric':<30} {'Strategy':>15} {'Buy & Hold':>15}")
    print("-" * 70)
    print(f"{'Total Return':<30} {best_stats['Total Return [%]']:>14.2f}% {buy_and_hold_return:>14.2f}%")
    print(f"{'Alpha (vs B&H)':<30} {best_stats['Total Return [%]'] - buy_and_hold_return:>14.2f}%")
    print(f"{'Max Drawdown':<30} {best_stats['Max Drawdown [%]']:>14.2f}%")
    print(f"{'Sharpe Ratio':<30} {best_stats['Sharpe Ratio']:>14.2f}")
    print(f"{'Win Rate':<30} {best_stats['Win Rate [%]']:>14.2f}%")
    print(f"{'Total Trades':<30} {best_stats['Total Trades']:>15.0f}")
    print(f"{'Avg Trade Duration':<30} {str(best_stats.get('Avg Winning Trade Duration', 'N/A')):>15}")
    
    # Additional metrics
    if best_stats['Total Trades'] > 0:
        avg_win = best_stats.get('Avg Winning Trade [%]', 0)
        avg_loss = best_stats.get('Avg Losing Trade [%]', 0)
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        print(f"{'Avg Win':<30} {avg_win:>14.2f}%")
        print(f"{'Avg Loss':<30} {avg_loss:>14.2f}%")
        print(f"{'Profit Factor':<30} {profit_factor:>14.2f}")
    
    print("="*70 + "\n")
    
    # Show top 10 strategies - FIXED
    print("🏆 Top 10 Strategies by Sharpe Ratio:")
    print("-" * 90)
    print(f"{'#':<3} {'Fast':<6} {'Slow':<6} {'Return %':>10} {'Sharpe':>8} {'MaxDD %':>10} {'Trades':>8} {'Win %':>8}")
    print("-" * 90)
    
    valid_sharpe = sharpe[strategies_with_trades.index].replace([np.inf, -np.inf], np.nan).dropna()
    top_n = min(10, len(valid_sharpe))
    top_idx = valid_sharpe.nlargest(top_n).index
    
    for i, idx in enumerate(top_idx, 1):
        # Get win rate from stats - FIXED
        try:
            strat_stats = portfolio[idx].stats()
            win_rate = strat_stats.get('Win Rate [%]', 0)
            # Handle case where win_rate might be NaN or None
            if pd.isna(win_rate):
                win_rate = 0.0
        except:
            win_rate = 0.0
        
        print(f"{i:<3} {idx[0]:<6} {idx[1]:<6} {total_return[idx]:>10.2f} "
            f"{sharpe[idx]:>8.2f} {max_dd[idx]:>10.2f} {trades_count[idx]:>8.0f} "
            f"{win_rate:>7.1f}%")
    
    print("="*90 + "\n")
    
    # Strategy comparison
    if len(profitable_strategies) > 0:
        print(f"✅ {len(profitable_strategies)} profitable strategies found ({len(profitable_strategies)/len(strategies_with_trades)*100:.1f}%)")
    else:
        print(f"❌ No profitable strategies. Best loss: {valid_strategies.max():.2f}%")
    
    print(f"\n💡 Recommendation: {'✅ Strategy is viable' if best_stats['Total Return [%]'] > 0 else '❌ Strategy needs improvement'}")
    
    if best_stats['Total Trades'] > len(df) / 10:
        print(f"⚠️  Warning: High trade frequency ({best_stats['Total Trades']} trades). Consider increasing MA periods.")
    
    if best_stats['Max Drawdown [%]'] > 30:
        print(f"⚠️  Warning: High drawdown ({best_stats['Max Drawdown [%]']:.1f}%). Consider adding risk management.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run improved baseline backtest.")
    parser.add_argument("--data", type=str, default="data/raw/btc_1h.parquet", 
                    help="Path to the raw kline data.")
    parser.add_argument("--position-size", type=float, default=0.50,
                    help="Position size as fraction of capital (default: 0.50)")
    args = parser.parse_args()

    # ปรับพารามิเตอร์ให้ลด overtrading
    fast_window_range = (20, 60, 10)    # 20, 30, 40, 50
    slow_window_range = (60, 200, 20)   # 60, 80, 100, 120, 140, 160, 180

    run_advanced_baseline_backtest(
        Path(args.data),
        fast_win_range=fast_window_range,
        slow_win_range=slow_window_range,
        position_size=args.position_size,
        min_holding_periods=2
    )