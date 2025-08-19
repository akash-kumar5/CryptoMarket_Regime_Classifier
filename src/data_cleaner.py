# src/data_cleaner.py
import os
import pandas as pd

DATA_FOLDER = 'data/'

def save_data(df, symbol, interval, start_date, end_date):
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
    filename = f"{symbol}_{interval}_{start_date}_{end_date}.csv"
    filepath = os.path.join(DATA_FOLDER, filename)
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")

def load_data(symbol, interval, start_date, end_date):
    filepath = os.path.join(DATA_FOLDER, f"{symbol}_{interval}_{start_date}_{end_date}.csv")
    if os.path.exists(filepath):
        return pd.read_csv(filepath, parse_dates=['timestamp'])
    else:
        print("File not found!")
        return None

def merge_timeframes(symbol, intervals, start_date, end_date):
    """Merge multiple timeframe CSVs into one DataFrame, aligning on timestamp.
    Forward-fill bigger timeframe data to avoid NaNs in smaller timeframe rows."""

    merged_df = None

    for interval in intervals:
        df = load_data(symbol, interval, start_date, end_date)
        if df is not None:
            # Rename columns to avoid clashes
            renamed_df = df.rename(columns={
                'open': f'open_{interval}',
                'high': f'high_{interval}',
                'low': f'low_{interval}',
                'close': f'close_{interval}',
                'volume': f'volume_{interval}'
            })

            if merged_df is None:
                merged_df = renamed_df
            else:
                merged_df = pd.merge(
                    merged_df, 
                    renamed_df, 
                    on='timestamp', 
                    how='outer'
                )
    
    if merged_df is not None:
        merged_df = merged_df.sort_values(by='timestamp').reset_index(drop=True)
        # Forward-fill all columns except timestamp to fill NaNs from bigger timeframes
        merged_df.ffill(inplace=True)
    return merged_df
