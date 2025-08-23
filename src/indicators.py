# src/feature_engineering.py
import pandas as pd
import numpy as np
import talib

def build_features(df, main_tf='15m', context_tfs=['5m', '1h']):
    """
    Builds a comprehensive feature set from the merged multi-timeframe data.
    Crucially, it shifts context features to prevent lookahead bias.
    
    Args:
        df (pd.DataFrame): The merged dataframe from data_cleaner.
        main_tf (str): The primary timeframe for the model (e.g., '15m').
        context_tfs (list): The other timeframes providing context.

    Returns:
        pd.DataFrame: DataFrame with all features and OHLCV data, ready for modeling.
    """
    
    # Make a copy to avoid modifying the original dataframe
    df_features = df.copy()
    
    all_timeframes = [main_tf] + context_tfs

    # --- 1. Calculate indicators for ALL timeframes ---
    for tf in all_timeframes:
        # Get the correct column names for the timeframe
        close = df_features[f'close_{tf}']
        high = df_features[f'high_{tf}']
        low = df_features[f'low_{tf}']
        volume = df_features[f'volume_{tf}']
        
        # Returns
        df_features[f'log_ret_1_{tf}'] = np.log(close / close.shift(1))
        
        # Trend
        ema21 = talib.EMA(close, timeperiod=21)
        ema55 = talib.EMA(close, timeperiod=55)
        df_features[f'ema_slope_21_{tf}'] = (ema21 - ema21.shift(1))
        df_features[f'price_vs_ema55_{tf}'] = close / ema55
        _, _, df_features[f'macd_hist_{tf}'] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df_features[f'adx_{tf}'] = talib.ADX(high, low, close, timeperiod=14)
        
        # Volatility
        atr14 = talib.ATR(high, low, close, timeperiod=14)
        df_features[f'atr_norm_{tf}'] = atr14 / close
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df_features[f'bb_width_{tf}'] = (upper - lower) / middle
        df_features[f'realized_vol_20_{tf}'] = df_features[f'log_ret_1_{tf}'].rolling(window=20).std() * np.sqrt(20)

        # Momentum
        df_features[f'rsi_{tf}'] = talib.RSI(close, timeperiod=14)
        df_features[f'roc_{tf}'] = talib.ROC(close, timeperiod=10)
        df_features[f'stoch_k_{tf}'], _ = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        
        # Microstructure
        volume_mean_50 = volume.rolling(window=50).mean()
        volume_std_50 = volume.rolling(window=50).std()
        df_features[f'volume_zscore_50_{tf}'] = (volume - volume_mean_50) / volume_std_50
        
        body = abs(df_features[f'open_{tf}'] - close)
        upper_wick = high - np.maximum(df_features[f'open_{tf}'], close)
        lower_wick = np.minimum(df_features[f'open_{tf}'], close) - low
        df_features[f'wick_ratio_{tf}'] = (upper_wick + lower_wick) / (body + 1e-6)

    # --- 2. CRITICAL STEP: Shift context features to prevent leakage ---
    # Identify all feature columns that are NOT from the main timeframe
    feature_cols = [col for col in df_features.columns if not col.startswith(('open_', 'high_', 'low_', 'close_', 'volume_', 'timestamp'))]
    
    context_feature_cols = []
    for tf in context_tfs:
        context_feature_cols.extend([col for col in feature_cols if col.endswith(f'_{tf}')])
    
    # Shift these context features by 1 bar.
    # This ensures that at bar `t`, we only use context information available at the close of `t-1`.
    df_features[context_feature_cols] = df_features[context_feature_cols].shift(1)
    
    # --- 3. Final Cleanup ---
    # Drop rows with NaN values, which are created by indicator calculations and shifting
    df_features.dropna(inplace=True)
    
    return df_features.reset_index(drop=True)

