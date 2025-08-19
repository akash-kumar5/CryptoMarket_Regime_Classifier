import pandas as pd

def label_market_regime(df: pd.DataFrame, prefix: str = "5m") -> pd.DataFrame:
    # Map columns dynamically based on prefix
    close = f"close_{prefix}"
    ema20 = f"EMA20_{prefix}"
    ema50 = f"EMA50_{prefix}"
    atr14 = f"ATR14_{prefix}"
    bb = f"BB_Width_{prefix}"
    adx = f"ADX14_{prefix}"

    # Precompute rolling ATR mean
    atr_mean_col = f"{atr14}_Mean20"
    df[atr_mean_col] = df[atr14].rolling(window=20).mean()

    regimes = []
    for _, row in df.iterrows():
        if pd.isna(row[ema20]) or pd.isna(row[ema50]) or pd.isna(row[adx]) or pd.isna(row[bb]) or pd.isna(row[atr_mean_col]):
            regimes.append("Undefined")
            continue

       # A more robust logical flow for regime classification

# 1. First, check if a clear trend exists (using a standard ADX threshold of 25)
        if row[adx] >= 25:
            if row[ema20] > row[ema50] and row[close] > row[ema50]:
                regimes.append("Uptrend")
            elif row[ema20] < row[ema50] and row[close] < row[ema50]:
                regimes.append("Downtrend")
            else:
                # Catches cases where ADX is high but EMAs might be conflicting
                regimes.append("Undefined")

        # 2. If no clear trend exists, determine the non-trending regime
        elif row[adx] < 25:
            # A "Squeeze" is a range with very low volatility. We use BB_Width to check.
            # Note: The 0.015 threshold is an example; you may need to adjust it.
            if row[bb] < 0.015:
                regimes.append("Squeeze")
            else:
                # If ADX is low but volatility isn't critically low, it's a normal range.
                regimes.append("Range")

        # 3. Fallback for any other case
        else:
            regimes.append("Undefined")

        # Add result as new column
    df[f"Regime_{prefix}"] = regimes
    return df
