import os
import sys
import concurrent.futures
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd

# Add src path
# Ensure the path is correct relative to where streamlit is run
try:
    # This assumes the script is run from the root directory (e.g., `streamlit run dashboard/page/1_Fetch_data.py`)
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from src.data_fetcher import fetch_historical_data
    from src.data_cleaner import save_data, merge_timeframes
except ImportError:
    st.error("Could not import custom modules. Please ensure the `src` directory is in the project's root and you are running streamlit from the root directory.")
    # Provide dummy functions to avoid crashing the app on import error
    def fetch_historical_data(**kwargs): return pd.DataFrame()
    def save_data(**kwargs): pass
    def merge_timeframes(**kwargs): return None


DATA_FOLDER = "data/"

# ------------------- STREAMLIT APP -------------------
st.set_page_config(page_title="Multi-Timeframe Data Fetcher", layout="wide")
st.title("Binance Historical Data Fetcher - Multi Timeframe")
st.markdown("""
This tool fetches historical cryptocurrency data from Binance for multiple timeframes concurrently.
After fetching, it can merge the data into a single file, aligning all timeframes to a selected 'main' timeframe.
""")

# --- User Inputs ---
symbol = st.text_input("Symbol (e.g., BTCUSDT)", value="BTCUSDT").upper()

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=90))
with col2:
    end_date = st.date_input("End Date", value=datetime.now())

available_timeframes = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
timeframes = st.multiselect(
    "Select Timeframes to Fetch",
    options=available_timeframes,
    default=["1m", "5m", "15m"],
    help="Select one or more timeframes to download."
)

main_tf = None
if len(timeframes) > 1:
    # Try to find '5m' or '15m' as a sensible default index
    default_index = 0
    if '5m' in timeframes:
        default_index = timeframes.index('5m')
    elif '15m' in timeframes and '5m' not in timeframes:
        default_index = timeframes.index('15m')

    main_tf = st.selectbox(
        "Select Main Timeframe for Merging",
        options=timeframes,
        index=default_index,
        help="This timeframe will be the base index for the final merged dataset."
    )

# --- Fetching and Merging Logic ---
def fetch_and_save(symbol, interval, start_time, end_time):
    """Fetches data and saves it, returning the interval and resulting dataframe."""
    df = fetch_historical_data(symbol=symbol, interval=interval, start_time=start_time, end_time=end_time)
    if not df.empty:
        save_data(df, symbol, interval)
    return interval, df

if st.button("Fetch and Process Data", type="primary"):
    if not timeframes:
        st.warning("Please select at least one timeframe.")
    else:
        # Convert dates to timestamps
        start_time = int(datetime.combine(start_date, datetime.min.time()).timestamp() * 1000)
        end_time = int(datetime.combine(end_date, datetime.min.time()).timestamp() * 1000)

        # Fetch data concurrently
        with st.spinner(f"Fetching {len(timeframes)} timeframe(s) for {symbol}..."):
            results = {}
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Create a future for each timeframe to be fetched
                futures = {executor.submit(fetch_and_save, symbol, tf, start_time, end_time): tf for tf in timeframes}
                for future in concurrent.futures.as_completed(futures):
                    tf, df = future.result()
                    results[tf] = df

        st.success("Data fetching complete!")

        # Display previews for each fetched timeframe
        for tf in sorted(results.keys()):
            df = results[tf]
            if not df.empty:
                with st.expander(f"Preview for {tf} ({len(df)} rows)"):
                    st.dataframe(df.tail(10))
            else:
                st.warning(f"No data was returned for the {tf} timeframe.")

        # Merge the timeframes if more than one was selected
        if len(timeframes) > 1 and main_tf:
            context_tfs = [tf for tf in timeframes if tf != main_tf]
            st.info(f"Merging data with '{main_tf}' as the main timeframe.")
            with st.spinner("Merging timeframes..."):
                merged_df = merge_timeframes(symbol, main_tf=main_tf, context_tfs=context_tfs)

            if merged_df is not None and not merged_df.empty:
                merged_filename = f"{symbol}_merged_{start_date.strftime('%Y')}-{end_date.strftime('%Y')}.csv"
                os.makedirs(DATA_FOLDER, exist_ok=True)
                merged_df.to_csv(os.path.join(DATA_FOLDER, merged_filename), index=False)
                st.success(f"Merged data saved as `{os.path.join(DATA_FOLDER, merged_filename)}`")

                with st.expander("Preview of merged data"):
                    st.dataframe(merged_df.tail(20))
            else:
                st.error("Failed to merge data. Ensure the main timeframe and at least one context timeframe were fetched successfully.")
        elif len(timeframes) == 1:
            st.info("Only one timeframe selected. Skipping merge operation.")
