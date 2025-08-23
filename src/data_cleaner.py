# src/data_cleaner.py
import os
import pandas as pd

DATA_FOLDER = 'data/'

def save_data(df, symbol, interval, start_date, end_date):
    """Saves a DataFrame to a CSV file with a standardized name."""
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
    filename = f"{symbol}_{interval}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.csv"
    filepath = os.path.join(DATA_FOLDER, filename)
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")

def load_data(symbol, interval, start_date, end_date):
    """Loads a DataFrame from a CSV file, parsing the timestamp."""
    filename = f"{symbol}_{interval}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.csv"
    filepath = os.path.join(DATA_FOLDER, filename)
    if os.path.exists(filepath):
        # Set timestamp as index right away for easier resampling/merging
        return pd.read_csv(filepath, parse_dates=['timestamp'], index_col='timestamp')
    else:
        print(f"File not found: {filepath}")
        return None

def merge_timeframes(symbol, start_date, end_date, main_tf='15m', context_tfs=['5m', '1h']):
    """
    Merges multiple timeframe data into a single DataFrame aligned to the main timeframe,
    preventing any lookahead bias.

    - Smaller timeframes (e.g., 5m) are aggregated UP to the main timeframe.
    - Larger timeframes (e.g., 1h) are reindexed and forward-filled DOWN to the main timeframe.
    """
    # 1. Load the main timeframe data, which will serve as our master index.
    df_main = load_data(symbol, main_tf, start_date, end_date)
    if df_main is None:
        print(f"Main timeframe ({main_tf}) data not found. Cannot merge.")
        return None

    # Rename main df columns to be specific
    df_main = df_main.rename(columns={
        'open': f'open_{main_tf}', 'high': f'high_{main_tf}',
        'low': f'low_{main_tf}', 'close': f'close_{main_tf}',
        'volume': f'volume_{main_tf}'
    })

    merged_df = df_main

    # 2. Process and merge context timeframes
    for tf in context_tfs:
        df_context = load_data(symbol, tf, start_date, end_date)
        if df_context is None:
            print(f"Skipping context timeframe {tf} as data was not found.")
            continue

        # Rename context columns before merging
        df_context = df_context.rename(columns={
            'open': f'open_{tf}', 'high': f'high_{tf}',
            'low': f'low_{tf}', 'close': f'close_{tf}',
            'volume': f'volume_{tf}'
        })

        if pd.to_timedelta(tf) < pd.to_timedelta(main_tf):
            # --- Aggregate UP (e.g., 5m -> 15m) ---
            # Define aggregation rules
            agg_rules = {
                f'open_{tf}': 'first',
                f'high_{tf}': 'max',
                f'low_{tf}': 'min',
                f'close_{tf}': 'last',
                f'volume_{tf}': 'sum'
            }
            # Resample to the main timeframe's frequency
            df_agg = df_context.resample(main_tf).agg(agg_rules)
            merged_df = merged_df.merge(df_agg, left_index=True, right_index=True, how='left')

        elif pd.to_timedelta(tf) > pd.to_timedelta(main_tf):
            # --- Reindex DOWN (e.g., 1h -> 15m) ---
            # Reindex the larger timeframe data to the main index, then forward-fill
            df_resampled = df_context.reindex(merged_df.index, method='ffill')
            merged_df = merged_df.merge(df_resampled, left_index=True, right_index=True, how='left')

    # The first few rows might have NaNs from the larger timeframe, so we drop them
    merged_df.dropna(inplace=True)
    
    # Reset index to have timestamp as a column again, matching original format
    merged_df.reset_index(inplace=True)

    return merged_df

