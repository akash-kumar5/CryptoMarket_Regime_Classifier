import os
import pandas as pd

DATA_FOLDER = 'data/'

def interval_to_seconds(interval_str):
    """
    Converts a Binance interval string (e.g., '1m', '4h', '1d') to seconds for comparison.
    Approximates 1M as 30 days.
    """
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
            return value * 2592000  # Approximation for 30 days
    except (ValueError, IndexError):
        return 0
    return 0

def to_pandas_freq(binance_tf):
    """
    Converts Binance timeframe string to pandas frequency string for resampling.
    e.g., '1m' -> '1T', '1h' -> '1H', '1M' -> '1M'
    """
    d = {'m': 'T', 'h': 'H', 'd': 'D', 'w': 'W', 'M': 'M'}
    try:
        unit = binance_tf[-1]
        val = binance_tf[:-1]
        if unit in d:
            return f"{val}{d[unit]}"
    except (ValueError, IndexError):
        pass
    # Fallback for simple cases that pandas might understand
    return binance_tf


def save_data(df, symbol, interval):
    """Saves a DataFrame to a CSV file with a standardized name."""
    os.makedirs(DATA_FOLDER, exist_ok=True)
    filename = f"{symbol}_{interval}.csv"
    filepath = os.path.join(DATA_FOLDER, filename)
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")

def load_data(symbol, interval):
    """Loads a DataFrame from a CSV file, parsing the timestamp."""
    filename = f"{symbol}_{interval}.csv"
    filepath = os.path.join(DATA_FOLDER, filename)
    if os.path.exists(filepath):
        try:
            # Set timestamp as index for easier resampling/merging
            df = pd.read_csv(filepath, parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    else:
        print(f"File not found: {filepath}")
        return None

def merge_timeframes(symbol, main_tf='5m', context_tfs=None):
    """
    Merges multiple timeframe datasets into one, based on a main timeframe.

    Args:
        symbol (str): The symbol (e.g., 'BTCUSDT').
        main_tf (str): The primary timeframe that will serve as the index.
        context_tfs (list): A list of other timeframes to merge into the main one.
    """
    if context_tfs is None:
        context_tfs = ['1m', '15m']

    # 1. Load the main timeframe data, which will serve as our master index.
    df_main = load_data(symbol, main_tf)
    if df_main is None or df_main.empty:
        print(f"Main timeframe ({main_tf}) data not found or is empty. Cannot merge.")
        return None

    # Rename main df columns to be specific
    df_main.rename(columns={
        'open': f'open_{main_tf}', 'high': f'high_{main_tf}',
        'low': f'low_{main_tf}', 'close': f'close_{main_tf}',
        'volume': f'volume_{main_tf}'
    }, inplace=True)

    merged_df = df_main
    main_tf_seconds = interval_to_seconds(main_tf)
    main_tf_freq = to_pandas_freq(main_tf)

    # 2. Process and merge context timeframes
    for tf in context_tfs:
        df_context = load_data(symbol, tf)
        if df_context is None or df_context.empty:
            print(f"Skipping context timeframe {tf} as data was not found or is empty.")
            continue

        # Rename context columns before merging
        df_context.rename(columns={
            'open': f'open_{tf}', 'high': f'high_{tf}',
            'low': f'low_{tf}', 'close': f'close_{tf}',
            'volume': f'volume_{tf}'
        }, inplace=True)

        tf_seconds = interval_to_seconds(tf)

        if tf_seconds < main_tf_seconds:
            # --- Aggregate UP (e.g., from 1m -> 5m) ---
            print(f"Aggregating {tf} up to {main_tf}...")
            agg_rules = {
                f'open_{tf}': 'first', f'high_{tf}': 'max',
                f'low_{tf}': 'min', f'close_{tf}': 'last',
                f'volume_{tf}': 'sum'
            }
            df_agg = df_context.resample(main_tf_freq).agg(agg_rules)
            merged_df = merged_df.merge(df_agg, left_index=True, right_index=True, how='left')

        elif tf_seconds > main_tf_seconds:
            # --- Reindex DOWN (e.g., from 15m -> 5m) ---
            print(f"Reindexing {tf} down to {main_tf}...")
            df_resampled = df_context.reindex(merged_df.index, method='ffill')
            merged_df = merged_df.merge(df_resampled, left_index=True, right_index=True, how='left')
        
        else:
             print(f"Skipping {tf} as it is the same as main_tf or could not be compared.")

    # Forward-fill any gaps that might have been created during left merges
    merged_df.fillna(method='ffill', inplace=True)
    # The first few rows might have NaNs from larger timeframes before they have data
    merged_df.dropna(inplace=True)
    
    # Reset index to have timestamp as a column again
    merged_df.reset_index(inplace=True)

    return merged_df
