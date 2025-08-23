# dashboard/page/1_Fetch_data.py
import os
import sys
import concurrent.futures
from datetime import datetime, timedelta
import streamlit as st

# Add src path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_fetcher import fetch_historical_data
from src.data_cleaner import save_data, merge_timeframes

DATA_FOLDER = "data/"

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

    # Preview per timeframe
    for tf in sorted(results.keys()):
        if not results[tf].empty:
            with st.expander(f"Preview {tf}"):
                st.dataframe(results[tf].tail(10))

    # Merge timeframes
    merged_df = merge_timeframes(symbol, timeframes, start_date, end_date)
    if merged_df is not None:
        merged_filename = f"{symbol}_merged_{start_date}_{end_date}.csv"
        os.makedirs(DATA_FOLDER, exist_ok=True)
        merged_df.to_csv(os.path.join(DATA_FOLDER, merged_filename), index=False)
        st.success(f"Merged data saved as {merged_filename}")
        with st.expander("Preview of merged data"):
            st.dataframe(merged_df.tail(20))
