import os
import sys
import time
import requests
import pandas as pd
import streamlit as st
import concurrent.futures
from datetime import datetime, timedelta

# Add src path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_cleaner import save_data, merge_timeframes

DATA_FOLDER = "data/"
BASE_URL = "https://api.binance.com/api/v3/klines"

# ------------------- FETCHER -------------------
def fetch_historical_data(symbol="BTCUSDT", interval="1h", start_time=None, end_time=None):
    """
    Fetch ALL available historical data from Binance for given symbol/interval.
    Handles pagination (1000 candle limit per request).
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
        time.sleep(0.2)  # avoid hitting rate limits

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

# ------------------- STREAMLIT APP -------------------
st.set_page_config(page_title="Multi-Timeframe Data Fetcher", layout="wide")
st.title("Binance Historical Data Fetcher - Multi Timeframe")

symbol = st.text_input("Symbol (e.g., BTCUSDT)", value="BTCUSDT")

# Date pickers
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=90))
with col2:
    end_date = st.date_input("End Date", value=datetime.now())

timeframes = ["5m", "15m", "1h"]

def fetch_and_save(symbol, interval, start_time, end_time):
    df = fetch_historical_data(symbol=symbol, interval=interval, start_time=start_time, end_time=end_time)
    if not df.empty:
        save_data(df, symbol, interval, start_date, end_date)
    return interval, df

if st.button("Fetch Multi-Timeframe Data"):
    start_time = int(datetime.combine(start_date, datetime.min.time()).timestamp() * 1000)
    end_time = int(datetime.combine(end_date, datetime.min.time()).timestamp() * 1000)

    with st.spinner("Fetching multiple timeframes..."):
        results = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(fetch_and_save, symbol, tf, start_time, end_time) for tf in timeframes]
            for future in concurrent.futures.as_completed(futures):
                tf, df = future.result()
                results[tf] = df

    st.success("All timeframes fetched and saved!")

    # Preview
    for tf in sorted(results.keys()):
        if not results[tf].empty:
            with st.expander(f"Preview {tf}"):
                st.dataframe(results[tf].tail(10))

    merged_df = merge_timeframes(symbol, timeframes, start_date, end_date)
    if merged_df is not None:
        merged_filename = f"{symbol}_merged_{start_date}_{end_date}.csv"
        os.makedirs(DATA_FOLDER, exist_ok=True)
        merged_df.to_csv(os.path.join(DATA_FOLDER, merged_filename), index=False)
        st.success(f"Merged data saved as {merged_filename}")
        with st.expander("Preview of merged data"):
            st.dataframe(merged_df.tail(20))
