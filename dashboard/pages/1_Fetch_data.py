# dashboard/pages/1_Fetch_data.py
import os
import sys
import concurrent.futures
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import json
import time

# Add src path
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from src.data_fetcher import (
        fetch_historical_klines,
        fetch_agg_trades,
        fetch_order_book_snapshot,
        fetch_futures_funding_rate_history,
        fetch_futures_open_interest,
        aggtrades_to_5s_bars
    )
    from src.data_cleaner import save_data, merge_timeframes
except ImportError as e:
    st.error(f"Could not import custom modules: {e}. Ensure `src` exists and you're running streamlit from project root.")
    # fallback stubs to avoid crashes during edit
    def fetch_historical_klines(**kwargs): return pd.DataFrame()
    def fetch_agg_trades(**kwargs): return pd.DataFrame()
    def fetch_order_book_snapshot(**kwargs): return {}
    def fetch_futures_funding_rate_history(**kwargs): return pd.DataFrame()
    def fetch_futures_open_interest(**kwargs): return {}
    def aggtrades_to_5s_bars(**kwargs): return pd.DataFrame()
    def save_data(df, symbol, interval, dtype="klines", suffix=""): pass
    def merge_timeframes(symbol, main_tf, context_tfs): return pd.DataFrame()

DATA_FOLDER = "data/"
os.makedirs(DATA_FOLDER, exist_ok=True)

st.set_page_config(page_title="Multi-Timeframe Data Fetcher", layout="wide")
st.title("Binance Historical Data Fetcher - Multi Timeframe")
st.markdown("""
This tool fetches historical cryptocurrency data from Binance for multiple timeframes concurrently.
You can fetch OHLCV klines and optionally trade-level aggTrades, a single order-book snapshot, and futures data (funding rate history / open interest).
After fetching, the page can merge multiple klines timeframes into one dataset aligned to a chosen main timeframe.
""")

# --- User Inputs ---
symbol = st.text_input("Symbol (e.g., BTCUSDT)", value="BTCUSDT").upper()

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=90))
with col2:
    end_date = st.date_input("End Date", value=datetime.now())

available_timeframes = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]
timeframes = st.multiselect(
    "Select Timeframes to Fetch (klines)",
    options=available_timeframes,
    default=["1m", "5m"],
    help="Select one or more timeframes to download KLINES (OHLCV)."
)

# Extra microstructure / derivatives options
st.markdown("**Optional extra data** (will be fetched alongside klines if selected)")
fetch_trades = st.checkbox("Fetch aggTrades (trade-level)", value=False, help="Backfill aggTrades (may be slow for long date ranges)")
fetch_orderbook_snapshot = st.checkbox("Fetch current order-book snapshot (depth)", value=False, help="Fetch a single snapshot at the time of fetch")
fetch_futures = st.checkbox("Fetch futures data (funding rate history & open interest)", value=False)

# main timeframe selection for merging (if multiple)
main_tf = None
if len(timeframes) > 1:
    default_index = 0
    if '5m' in timeframes:
        default_index = timeframes.index('5m')
    main_tf = st.selectbox(
        "Select Main Timeframe for Merging",
        options=timeframes,
        index=default_index,
        help="This timeframe will be the base index for the final merged dataset."
    )

# helper to make timestamps in ms
def dt_to_ms(dt: datetime):
    return int(datetime.combine(dt, datetime.min.time()).timestamp() * 1000)

# ---------- Fetching worker ----------
def fetch_and_save_klines(sym: str, interval: str, start_ms: int, end_ms: int):
    df = fetch_historical_klines(symbol=sym, interval=interval, start_time=start_ms, end_time=end_ms)
    if not df.empty:
        fname = f"{sym}_klines_{interval}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        path = os.path.join(DATA_FOLDER, fname)
        df.to_csv(path, index=False)
        # also call save_data if you have other processing
        try:
            save_data(df, sym, interval, dtype="klines", suffix=f"_{start_date.strftime('%Y%m%d')}")
        except Exception:
            pass
        return ("klines", interval, path, len(df))
    return ("klines", interval, None, 0)

def fetch_and_save_aggtrades(sym: str, start_ms: int, end_ms: int, bucket_minutes: int = None):
    # For large ranges, users should slice into smaller windows; here we attempt naive fetch
    try:
        df = fetch_agg_trades(symbol=sym, start_time=start_ms, end_time=end_ms)
    except Exception as e:
        st.warning(f"aggTrades fetch error for {sym}: {e}")
        df = pd.DataFrame()
    if not df.empty:
        fname = f"{sym}_aggtrades_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        path = os.path.join(DATA_FOLDER, fname)
        df.to_csv(path, index=False)
        # optional aggregation to 5s bars for quick preview
        preview_df = None
        if bucket_minutes is None:
            preview_df = aggtrades_to_5s_bars(df, bucket_ms=5000)
            preview_name = f"{sym}_aggtrades_5s_preview_{start_date.strftime('%Y%m%d')}.csv"
            preview_path = os.path.join(DATA_FOLDER, preview_name)
            preview_df.to_csv(preview_path, index=False)
        else:
            preview_df = aggtrades_to_5s_bars(df, bucket_ms=bucket_minutes*60*1000)
        return ("aggtrades", None, path, len(df), preview_df)
    return ("aggtrades", None, None, 0, None)

def fetch_and_save_orderbook(sym: str, limit: int = 100):
    try:
        snapshot = fetch_order_book_snapshot(symbol=sym, limit=limit)
    except Exception as e:
        st.warning(f"Orderbook fetch error: {e}")
        snapshot = {}
    if snapshot:
        fname = f"{sym}_depth_{limit}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
        path = os.path.join(DATA_FOLDER, fname)
        with open(path, "w") as f:
            json.dump(snapshot, f)
        return ("depth", None, path, snapshot)
    return ("depth", None, None, None)

def fetch_and_save_futures(sym: str, start_ms: int, end_ms: int):
    results = {}
    try:
        fr = fetch_futures_funding_rate_history(symbol=sym, start_time=start_ms, end_time=end_ms)
        if not fr.empty:
            fname = f"{sym}_funding_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
            path = os.path.join(DATA_FOLDER, fname)
            fr.to_csv(path, index=False)
            results['funding'] = path
    except Exception as e:
        st.warning(f"Funding history fetch error: {e}")

    try:
        oi = fetch_futures_open_interest(symbol=sym)
        results['open_interest_now'] = oi
    except Exception as e:
        st.warning(f"Open interest fetch error: {e}")

    return ("futures", None, results, None)

# ---------- Main action ----------
if st.button("Fetch and Process Data", type="primary"):
    if not timeframes and not (fetch_trades or fetch_orderbook_snapshot or fetch_futures):
        st.warning("Select at least one fetch option: timeframes or microstructure/futures.")
        st.stop()

    start_ms = dt_to_ms(start_date)
    end_ms = dt_to_ms(end_date)

    fetched = {}
    with st.spinner("Fetching data... this may take a while for long ranges"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = {}
            # klines
            for tf in timeframes:
                futures[executor.submit(fetch_and_save_klines, symbol, tf, start_ms, end_ms)] = ("klines", tf)
            # trades
            if fetch_trades:
                futures[executor.submit(fetch_and_save_aggtrades, symbol, start_ms, end_ms)] = ("aggtrades", None)
            # depth snapshot (single)
            if fetch_orderbook_snapshot:
                futures[executor.submit(fetch_and_save_orderbook, symbol, 100)] = ("depth", None)
            # futures
            if fetch_futures:
                futures[executor.submit(fetch_and_save_futures, symbol, start_ms, end_ms)] = ("futures", None)

            for fut in concurrent.futures.as_completed(futures):
                try:
                    res = fut.result()
                    key = futures[fut]
                    fetched[key] = res
                except Exception as e:
                    st.warning(f"Worker failed: {e}")

    st.success("Fetching finished. Results summary below.")

    # --- Display previews ---
    for k, v in fetched.items():
        kind, tf = k
        if v is None:
            st.write(f"{kind} : No data")
            continue
        if v[0] == "klines":
            _, interval, path, nrows = v
            if path:
                with st.expander(f"Klines {interval} — {nrows} rows — {path}"):
                    try:
                        df_preview = pd.read_csv(path, parse_dates=['timestamp'])
                    except Exception:
                        df_preview = pd.read_csv(path)
                    st.dataframe(df_preview.tail(10))
        elif v[0] == "aggtrades":
            _, _, path, nrows, preview_df = v
            if path:
                with st.expander(f"AggTrades — {nrows} rows — {path}"):
                    st.write(f"Saved aggTrades to `{path}`")
                    if preview_df is not None and not preview_df.empty:
                        st.markdown("Preview (5s aggregated):")
                        st.dataframe(preview_df.tail(10))
        elif v[0] == "depth":
            _, _, path, snapshot = v
            if path:
                with st.expander(f"Orderbook snapshot — {path}"):
                    st.write(f"Saved depth snapshot JSON to `{path}`")
                    st.write("Top 5 bids / asks (sample):")
                    bids = snapshot.get("bids", [])[:5]
                    asks = snapshot.get("asks", [])[:5]
                    st.write({"bids": bids, "asks": asks})
        elif v[0] == "futures":
            _, _, results, _ = v
            with st.expander("Futures results"):
                if 'funding' in results:
                    st.write(f"Funding saved to: {results['funding']}")
                    try:
                        df_f = pd.read_csv(results['funding'])
                        st.dataframe(df_f.tail(10))
                    except Exception:
                        st.write(results['funding'])
                if 'open_interest_now' in results:
                    st.write("Current open interest (raw):")
                    st.json(results['open_interest_now'])

    # --- Merge logic ---
    if len(timeframes) > 1 and main_tf:
        context_tfs = [tf for tf in timeframes if tf != main_tf]
        st.info(f"Merging data with '{main_tf}' as the main timeframe.")
        with st.spinner("Merging timeframes..."):
            merged_df = merge_timeframes(symbol, main_tf=main_tf, context_tfs=context_tfs)

        if merged_df is not None and not merged_df.empty:
            merged_filename = f"{symbol}_merged_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
            merged_path = os.path.join(DATA_FOLDER, merged_filename)
            merged_df.to_csv(merged_path, index=False)
            st.success(f"Merged data saved as `{merged_path}`")
            with st.expander("Preview of merged data"):
                st.dataframe(merged_df.tail(20))
        else:
            st.error("Failed to merge data. Ensure the main timeframe and at least one context timeframe were fetched successfully.")
    else:
        st.info("Skipping merge (either only one timeframe selected or main timeframe not set).")
