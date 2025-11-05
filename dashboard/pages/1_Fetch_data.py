# dashboard/pages/1_Fetch_data.py
import os, sys, json
from datetime import datetime, timedelta
import concurrent.futures
import streamlit as st
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data_fetcher import (
    fetch_klines,
    fetch_orderbook_snapshot,  # keep
)
from src.data_cleaner import merge_timeframes

DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

st.set_page_config(page_title="Binance Data Fetcher", layout="wide")
st.title("Binance Historical Data Fetcher")

# UI
symbol = st.text_input("Symbol (e.g., BTCUSDT)", value="BTCUSDT").upper()
c1, c2 = st.columns(2)
with c1:
    start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365*3))
with c2:
    end_date = st.date_input("End Date", value=datetime.now())

timeframes = st.multiselect("Timeframes (klines)", options=["1m","5m","15m"], default=["15m","5m"])

TF_TO_MINUTES = {'1m':1,'5m':5,'15m':15,'30m':30,'1h':60}
def dt_to_ms(dt): return int(datetime.combine(dt, datetime.min.time()).timestamp() * 1000)
def make_fname(sym, kind, start, end, tf=None, ext='csv'):
    parts = [sym, kind] + ([tf] if tf else []) + [f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"]
    return "_".join(parts) + f".{ext}"

# Utility: normalize klines DF to minimal OHLCV
def normalize_ohlcv(df):
    if df is None or df.empty:
        return None
    df = df.copy()
    if "timestamp" not in df.columns:
        if "open_time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
        elif "open_time_ms" in df.columns:
            df["timestamp"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True, errors="coerce")
        else:
            df.iloc[:,0] = pd.to_datetime(df.iloc[:,0], utc=True, errors="coerce")
            df = df.rename(columns={df.columns[0]: "timestamp"})
    keep = ["timestamp","open","high","low","close","volume"]
    df = df[[c for c in keep if c in df.columns]].dropna(subset=["timestamp"])
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("timestamp").reset_index(drop=True)

import gc
try:
    import psutil
except Exception:
    psutil = None

if st.button("Fetch Data"):
    s_ms, e_ms = dt_to_ms(start_date), dt_to_ms(end_date)
    # Only klines and depth
    jobs = [("klines", tf) for tf in timeframes] + [("depth", None)]

    max_workers = min(4, (os.cpu_count() or 1))
    st.info(f"Using max_workers={max_workers}")

    def enough_memory(threshold_ratio=0.15):
        if psutil is None:
            return True
        mem = psutil.virtual_memory()
        return mem.available / mem.total > threshold_ratio

    klines_saved_paths = {}
    with st.spinner("Fetching..."):
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {}
            for job, tf in jobs:
                if not enough_memory():
                    st.warning("Low memory detected — limiting concurrent submissions.")
                    import time; time.sleep(1)

                if job == "klines":
                    fut = ex.submit(fetch_klines, symbol, tf, s_ms, e_ms)
                elif job == "depth":
                    fut = ex.submit(fetch_orderbook_snapshot, symbol, 100)
                futs[fut] = (job, tf)

            for fut in concurrent.futures.as_completed(futs):
                job, tf = futs[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    st.warning(f"Error fetching {job} {tf}: {e}")
                    continue

                if job == "klines":
                    df = normalize_ohlcv(res)
                    if df is not None and not df.empty:
                        path = os.path.join(DATA_FOLDER, make_fname(symbol, "klines", start_date, end_date, tf))
                        df.to_csv(path, index=False)
                        klines_saved_paths[tf] = path
                        del df
                        gc.collect()

                elif job == "depth":
                    if isinstance(res, dict) and res.get("bids") is not None:
                        bids = res["bids"]; asks = res["asks"]
                        try:
                            bids.to_csv(os.path.join(DATA_FOLDER, f"{symbol}_depth_bids.csv"), index=False)
                            asks.to_csv(os.path.join(DATA_FOLDER, f"{symbol}_depth_asks.csv"), index=False)
                        except Exception:
                            snap = {
                                "lastUpdateId": res.get("lastUpdateId"),
                                "fetched_at": str(res.get("fetched_at")),
                                "bids": bids[["price", "qty"]].astype(str).values.tolist() if hasattr(bids, "values") else bids,
                                "asks": asks[["price", "qty"]].astype(str).values.tolist() if hasattr(asks, "values") else asks
                            }
                            with open(os.path.join(DATA_FOLDER, f"{symbol}_depth_snapshot.json"), "w") as f:
                                json.dump(snap, f)
                        del bids, asks, res
                        gc.collect()

    # Merge only klines that were saved to disk
    selectable = [tf for tf in timeframes if tf in TF_TO_MINUTES and tf in klines_saved_paths]
    if selectable:
        main_tf = min(selectable, key=lambda t: TF_TO_MINUTES[t])
        ctx = [t for t in selectable if t != main_tf]
        try:
            klines_map = {}
            for tf in selectable:
                path = klines_saved_paths[tf]
                tmp = pd.read_csv(path, parse_dates=["timestamp"])
                klines_map[tf] = normalize_ohlcv(tmp)
                del tmp
                gc.collect()

            merged = merge_timeframes(
                symbol,
                main_tf=main_tf,
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
                context_tfs=ctx,
                klines_map=klines_map
            )
            if merged is not None and not merged.empty:
                out_name = make_fname(symbol, "combined_klines", start_date, end_date)
                out_path = os.path.join(DATA_FOLDER, out_name)
                merged.to_csv(out_path, index=False)
                st.success(f"Merged klines saved: {out_path}")
                with st.expander("Preview merged (tail)"):
                    st.dataframe(merged.tail(20))
                del merged, klines_map
                gc.collect()
        except Exception as e:
            st.warning(f"Merge failed: {e}")
