# src/data_fetcher.py
import requests
import pandas as pd
import time

BASE_URL = "https://api.binance.com/api/v3/klines"

def fetch_historical_data(symbol="BTCUSDT", interval="1h", start_time=None, end_time=None):
    """
    Fetch ALL available historical OHLCV data from Binance for given symbol/interval.
    Handles pagination (1000 candle limit per request).
    Returns clean DataFrame: timestamp, open, high, low, close, volume
    """
    all_data = []
    limit = 1000
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time

    while True:
        response = requests.get(BASE_URL, params=params)
        data = response.json()

        if isinstance(data, dict) and data.get("code"):
            raise Exception(f"Error fetching data: {data}")

        if not data:
            break

        all_data.extend(data)

        # Binance returns [open_time, open, high, low, close, volume, ...]
        last_open_time = data[-1][0]
        next_start = last_open_time + 1
        if end_time and next_start >= end_time:
            break

        params["startTime"] = next_start
        time.sleep(0.25)  # avoid rate limit

        if len(data) < limit:
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = df.astype({"open": "float", "high": "float", "low": "float", "close": "float", "volume": "float"})

    return df
