import pandas as pd

def ema(df, span):
    return df['close'].ewm(span=span, adjust=False).mean()

def rsi(df, period=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    # Use ewm for correct smoothing
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10) # Add epsilon to prevent division by zero
    return 100 - (100 / (1 + rs))

def macd_histogram(df, fast=12, slow=26, signal=9):
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line

def atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def bollinger_band_width(df, period=20):
    sma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    upper_band = sma + (2 * std)
    lower_band = sma - (2 * std)
    return (upper_band - lower_band) / sma

def adx(df, period=14):
    # Make a copy to avoid side effects
    df_ = df.copy()
    
    # Calculate True Range
    high_low = df_['high'] - df_['low']
    high_close = abs(df_['high'] - df_['close'].shift())
    low_close = abs(df_['low'] - df_['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    up_move = df_['high'] - df_['high'].shift(1)
    down_move = df_['low'].shift(1) - df_['low']
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # Use ewm for all smoothing steps (Wilder's method)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    
    # Calculate DX and the final ADX
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
    adx_line = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return adx_line

def compute_indicators(df):
    df['EMA20'] = ema(df, 20)
    df['EMA50'] = ema(df, 50)
    df['EMA200'] = ema(df, 200)
    df['RSI14'] = rsi(df)
    df['MACD_Hist'] = macd_histogram(df)
    df['ATR14'] =atr(df)
    df['BB_Width'] = bollinger_band_width(df)
    df['ADX14'] = adx(df)
    
    df.drop(['up_move', 'down_move', 'plus_dm', 'minus_dm', 'plus_di', 'minus_di'], axis=1, inplace=True, errors='ignore')

    return df

def compute_multi_tf_indicators(merged_df):
    


    dfs = []
    timeframes = ['5m', '15m', '1h']
    indicator_cols = ['EMA20', 'EMA50', 'EMA200', 'RSI14', 'MACD_Hist', 'ATR14', 'BB_Width', 'ADX14']

    for tf in timeframes:
        # Select columns for this timeframe + timestamp
        cols = ['timestamp',
                f'open_{tf}', f'high_{tf}', f'low_{tf}', f'close_{tf}', f'volume_{tf}']
        print(merged_df[cols].tail(50).isna().sum())
        # Drop rows with all NaNs for this timeframe (optional but cleaner)
        subset = merged_df[cols].dropna(how='all', subset=cols[1:]).copy()

        # Rename to expected column names for compute_indicators
        subset.rename(columns={
            f'open_{tf}': 'open',
            f'high_{tf}': 'high',
            f'low_{tf}': 'low',
            f'close_{tf}': 'close',
            f'volume_{tf}': 'volume'
        }, inplace=True)
        

        # Compute indicators on this subset
        ind_df = compute_indicators(subset)

        # Rename indicator cols to include timeframe suffix
        ind_rename = {col: f"{col}_{tf}" for col in indicator_cols}
        ind_df.rename(columns=ind_rename, inplace=True)

        # Keep only timestamp + indicator cols
        ind_df = ind_df[['timestamp'] + list(ind_rename.values())]
        dfs.append(ind_df)

    # Merge all indicator dfs on timestamp
  # back-fill if needed
    
    final_df = merged_df.copy()
    for ind_df in dfs:
        final_df = final_df.merge(ind_df, on='timestamp', how='left')

    final_df.sort_values('timestamp', inplace=True)
    final_df.ffill(inplace=True)  # forward-fill all columns
    final_df.bfill( inplace=True)
    return final_df

