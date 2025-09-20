# src/data_cleaner.py
import os
import pandas as pd

DATA_FOLDER = 'data/'

def interval_to_seconds(interval_str):
    try:
        unit = interval_str[-1]
        value = int(interval_str[:-1])
        if unit == 'm':
            return value * 60
        elif unit == 'h':
            return value * 3600
        elif unit == 'd':
            return value * 86400
        elif unit == 'w':
            return value * 604800
        elif unit == 'M':
            return value * 2592000
    except (ValueError, IndexError):
        return 0
    return 0

def to_pandas_freq(binance_tf):
    d = {'m': 'T', 'h': 'H', 'd': 'D', 'w': 'W', 'M': 'M'}
    try:
        unit = binance_tf[-1]
        val = binance_tf[:-1]
        if unit in d:
            return f"{val}{d[unit]}"
    except (ValueError, IndexError):
        pass
    return binance_tf

def _normalize_klines_df(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce') \
                         if df['timestamp'].dtype == object or not pd.api.types.is_datetime64_any_dtype(df['timestamp']) \
                         else pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.set_index('timestamp')
    else:
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')
        else:
            try:
                df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], utc=True, errors='coerce')
                df = df.dropna(subset=[df.columns[0]])
                df = df.set_index(df.columns[0])
            except Exception:
                return None
    for c in ['open','high','low','close','volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.sort_index()
    return df

def load_data(symbol, interval, start_date, end_date):
    filename = f"{symbol}_klines_{interval}_{start_date}_{end_date}.csv"
    filepath = os.path.join(DATA_FOLDER, filename)
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    try:
        df = pd.read_csv(filepath, parse_dates=['timestamp'])
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
    return _normalize_klines_df(df)

def merge_timeframes(symbol, main_tf='5m', start_date=None, end_date=None, context_tfs=None, klines_map: dict = None):
    if context_tfs is None:
        context_tfs = ['1m']
    if not klines_map:
        klines_map = {}

    df_main = klines_map.get(main_tf)
    if isinstance(df_main, str):
        try:
            df_main = pd.read_csv(df_main, parse_dates=['timestamp'])
        except Exception:
            df_main = None
    df_main = _normalize_klines_df(df_main) if df_main is not None else load_data(symbol, main_tf, start_date, end_date)

    if df_main is None or df_main.empty:
        print(f"Main timeframe ({main_tf}) data could not be loaded. Cannot merge.")
        return None

    df_main = df_main.rename(columns={
        'open': f'open_{main_tf}', 'high': f'high_{main_tf}',
        'low': f'low_{main_tf}', 'close': f'close_{main_tf}',
        'volume': f'volume_{main_tf}'
    })
    merged_df = df_main.copy()

    main_tf_seconds = interval_to_seconds(main_tf)
    main_tf_freq = to_pandas_freq(main_tf)

    for tf in context_tfs:
        df_context = klines_map.get(tf)
        if isinstance(df_context, str):
            try:
                df_context = pd.read_csv(df_context, parse_dates=['timestamp'])
            except Exception:
                df_context = None
        df_context = _normalize_klines_df(df_context) if df_context is not None else load_data(symbol, tf, start_date, end_date)
        if df_context is None or df_context.empty:
            print(f"Skipping {tf} — no data")
            continue

        df_context = df_context.rename(columns={
            'open': f'open_{tf}', 'high': f'high_{tf}',
            'low': f'low_{tf}', 'close': f'close_{tf}',
            'volume': f'volume_{tf}'
        })
        tf_seconds = interval_to_seconds(tf)
        if tf_seconds < main_tf_seconds:
            agg_rules = {
                f'open_{tf}': 'first', f'high_{tf}': 'max',
                f'low_{tf}': 'min', f'close_{tf}': 'last',
                f'volume_{tf}': 'sum'
            }
            df_agg = df_context.resample(main_tf_freq).agg(agg_rules)
            merged_df = merged_df.merge(df_agg, left_index=True, right_index=True, how='left')
        elif tf_seconds > main_tf_seconds:
            df_resampled = df_context.reindex(merged_df.index, method='ffill')
            merged_df = merged_df.merge(df_resampled, left_index=True, right_index=True, how='left')
        else:
            merged_df = merged_df.merge(df_context, left_index=True, right_index=True, how='left')

    merged_df.fillna(method='ffill', inplace=True)
    main_cols = [f'open_{main_tf}', f'close_{main_tf}', f'high_{main_tf}', f'low_{main_tf}']
    if any(c in merged_df.columns for c in main_cols):
        merged_df = merged_df.dropna(subset=[c for c in main_cols if c in merged_df.columns], how='all')
    else:
        merged_df = merged_df.dropna(how='all')

    merged_df = merged_df.reset_index().rename(columns={'index': 'timestamp'})
    return merged_df
