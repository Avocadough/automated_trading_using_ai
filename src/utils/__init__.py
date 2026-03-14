import pandas as pd

def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Canonical helper: ensure UTC DatetimeIndex, sorted, no duplicates.
    Tries common timestamp column names as fallback.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df[~df.index.duplicated(keep='last')].sort_index()
    for col in ('timestamp', 'time', 'open_time', 'date', 'Date', 'datetime'):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
            df = df.set_index(col)
            break
    else:
        df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
    return df[~df.index.duplicated(keep='last')].sort_index()
