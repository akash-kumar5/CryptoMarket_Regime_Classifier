# dashboard/pages/2_Feature_Engineering.py
import streamlit as st
import sys
import os
import glob
import json
import pandas as pd

# Add src path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.compute_features import build_features, merge_all_sources_to_5m, load_and_normalize_funding, parse_datetime_series

DATA_FOLDER = 'data'

st.set_page_config(page_title="Feature Engineering", layout="wide")
st.title("Build Feature Set")
st.markdown("""
This step loads your merged 5m+15m kline file, automatically finds matching
microstructure & derivatives files by date/keywords in the `data/` folder,
merges them to the 5m frame, computes features (prevents lookahead), and saves the result.

Files we attempt to auto-find (optional):
- aggtrades_5s (CSV)
- depth_asks / depth_bids (CSV) or depth_snapshot.json (JSON lines with `bids`/`asks`)
- open_interest.json / open_interest.csv
- funding.json / funding.csv
""")

# --- Helpers --- #
def find_merged_files(data_folder):
    pattern = os.path.join(data_folder, "*combined*.csv")
    return sorted(glob.glob(pattern))

@st.cache_data(show_spinner=False)
def load_kline(file_path):
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    # ensure timestamp timezone-aware if present as string with +00:00 etc.
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors ='coerce')
    return df

def guess_date_string_from_df(df):
    # Use the earliest timestamp's date as YYYY-MM-DD
    ts_min = pd.to_datetime(df['timestamp']).min()
    return ts_min.strftime("%Y-%m-%d")

def find_aux_files_for_date(date_str):
    """
    Look for files in DATA_FOLDER containing the date and common keywords.
    Returns a dict of found paths (or None).
    """
    results = {
        'aggtrades': None,
        'depth_snapshot': None,
        'depth_asks': None,
        'depth_bids': None,
        'open_interest': None,
        'funding': None
    }
    # search patterns (case-insensitive)
    all_files = glob.glob(os.path.join(DATA_FOLDER, "*"))
    for p in all_files:
        name = os.path.basename(p).lower()
        if date_str in name:
            if 'aggtrade' in name or 'aggtrades' in name:
                results['aggtrades'] = p
            if 'depth_snapshot' in name or 'depthsnapshot' in name or 'depth_snap' in name:
                results['depth_snapshot'] = p
            if 'depth_asks' in name or 'asks' in name and p.endswith('.csv'):
                results['depth_asks'] = p
            if 'depth_bids' in name or 'bids' in name and p.endswith('.csv'):
                results['depth_bids'] = p
            if 'open_interest' in name or 'oi' in name:
                results['open_interest'] = p
            if 'funding' in name or 'fundingrate' in name:
                results['funding'] = p

    # fallback: more generic keyword match without date (useful if filenames don't include date)
    if not any(results.values()):
        for p in all_files:
            name = os.path.basename(p).lower()
            if 'aggtrade' in name or 'aggtrades' in name:
                results['aggtrades'] = p
            if 'depth_asks' in name:
                results['depth_asks'] = p
            if 'depth_bids' in name:
                results['depth_bids'] = p
            if 'depth_snapshot' in name or 'depthsnapshot' in name:
                results['depth_snapshot'] = p
            if 'open_interest' in name or 'openinterest' in name:
                results['open_interest'] = p
            if 'funding' in name:
                results['funding'] = p
    return results

@st.cache_data(show_spinner=False)
def safe_load_csv(path, parse_dates=['timestamp']):
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except Exception:
        # try without parsing then convert
        df = pd.read_csv(path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

@st.cache_data(show_spinner=False)
def safe_load_json_lines(path):
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        # try load normal json
        try:
            with open(path, 'r') as f:
                j = json.load(f)
            return pd.DataFrame(j)
        except Exception:
            return None

# --- Main UI flow --- #
merged_files = find_merged_files(DATA_FOLDER)
if not merged_files:
    st.warning(f"No 'combined' CSVs found in `{DATA_FOLDER}`. Place your merged kline file(s) there.")
    st.stop()

selected_path = st.selectbox("Select merged (kline) CSV", merged_files, format_func=lambda p: os.path.basename(p))

if selected_path:
    st.write(f"Loading `{os.path.basename(selected_path)}` ...")
    kline_df = load_kline(selected_path)
    st.subheader("Kline preview (tail)")
    st.dataframe(kline_df.tail(6))

    date_hint = guess_date_string_from_df(kline_df)
    st.info(f"Inferred kline date (from earliest timestamp): **{date_hint}**")

    st.write("Searching `data/` for related microscopic/derivative files (aggtrades, depth, funding, open_interest)...")
    found = find_aux_files_for_date(date_hint)

    # Show discoveries
    for k, v in found.items():
        st.write(f"- {k}: " + (os.path.basename(v) if v else "not found"))

    # Provide option to override found files manually
    st.markdown("**Override any auto-found file** (optional):")
    col1, col2 = st.columns(2)
    with col1:
        agg_path = st.text_input("aggtrades file path", value=found.get('aggtrades') or "")
        depth_snap_path = st.text_input("depth_snapshot file path (json)", value=found.get('depth_snapshot') or "")
        depth_asks_path = st.text_input("depth_asks CSV path", value=found.get('depth_asks') or "")
    with col2:
        depth_bids_path = st.text_input("depth_bids CSV path", value=found.get('depth_bids') or "")
        oi_path = st.text_input("open_interest file path", value=found.get('open_interest') or "")
        funding_path = st.text_input("funding file path", value=found.get('funding') or "")

    # Convert empty strings to None
    
    agg_path = agg_path.strip() or None
    depth_snap_path = depth_snap_path.strip() or None
    depth_asks_path = depth_asks_path.strip() or None
    depth_bids_path = depth_bids_path.strip() or None
    oi_path = oi_path.strip() or None
    funding_path = funding_path.strip() or None

    if st.button("Merge sources & Build feature set", type="primary"):
        with st.spinner("Loading auxiliary files and merging..."):
            # Load aggtrades if present
            agg_df = None
            if agg_path:
                try:
                    agg_df = safe_load_csv(agg_path)
                    st.write(f"Loaded aggtrades: {os.path.basename(agg_path)} ({agg_df.shape[0]} rows)")
                except Exception as e:
                    st.error(f"Failed to load aggtrades: {e}")
                    agg_df = None

            # Load depth snapshot or depth_asks/bid
            depth_snap_df = None
            if depth_snap_path:
                try:
                    # parse into the standardized snapshot format (timestamp, bids, asks)
                    from src.compute_features import parse_depth_snapshot_json
                    depth_snap_df = parse_depth_snapshot_json(depth_snap_path)
                    if depth_snap_df is None or depth_snap_df.empty:
                        st.warning(f"Depth snapshot loaded but empty: {os.path.basename(depth_snap_path)}")
                        depth_snap_df = None
                    else:
                        st.write(f"Loaded depth_snapshot: {os.path.basename(depth_snap_path)} ({len(depth_snap_df)} rows)")
                except Exception as e:
                    st.error(f"Failed to load depth_snapshot: {e}")
                    depth_snap_df = None

            else:
                # if individual ask/bid CSVs are provided, attempt a basic join into a snapshot format
                if depth_asks_path and depth_bids_path:
                    try:
                        asks = safe_load_csv(depth_asks_path)
                        bids = safe_load_csv(depth_bids_path)
                        # basic alignment: assume both have timestamp and price/qty columns; produce one DF of snapshots if possible
                        # NOTE: this is a best-effort; your snapshot format may differ.
                        st.write("Loaded depth_asks and depth_bids (will attempt to merge into snapshot rows).")
                        # Simple approach: keep separate asks/bids dfs and pass bids/asks as lists per timestamp is not trivial here.
                        # Prefer providing depth_snapshot.json for best results.
                    except Exception as e:
                        st.error(f"Failed to load depth ask/bid: {e}")

            # Load funding
            funding_df = None
            if funding_path:
                try:
                    # Use the robust loader that handles JSON/CSV and mixed timestamp formats
                    funding_df = load_and_normalize_funding(funding_path)
                    if funding_df is None or funding_df.empty:
                        st.warning(f"Funding file loaded but empty or unparseable: {os.path.basename(funding_path)}")
                        funding_df = None
                    else:
                        # load_and_normalize_funding ensures a 'timestamp' (tz-aware or NaT) and numeric fundingRate
                        # but normalize to UTC explicitly for merging safety (this is fast)
                        funding_df['timestamp'] = parse_datetime_series(funding_df['timestamp'])
                        st.write(f"Loaded funding data: {os.path.basename(funding_path)} ({len(funding_df)} rows)")
                except Exception as e:
                    st.error(f"Failed to load funding file (robust loader): {e}")
                    funding_df = None

            # Load open interest
            oi_df = None
            if oi_path:
                try:
                    if oi_path.endswith('.json'):
                        oi_df = safe_load_json_lines(oi_path)
                    else:
                        oi_df = safe_load_csv(oi_path)
                    # normalize column names
                    if 'time' in oi_df.columns and 'openInterest' not in oi_df.columns:
                        # assume fields 'time' and 'openInterest' or 'open_interest'
                        oi_df = oi_df.rename(columns={'time':'timestamp'})
                    if 'open_interest' in oi_df.columns:
                        oi_df = oi_df.rename(columns={'open_interest':'openInterest'})
                    if 'timestamp' in oi_df.columns:
                        oi_df['timestamp'] = pd.to_datetime(oi_df['timestamp'])
                    st.write(f"Loaded open interest: {os.path.basename(oi_path)} ({len(oi_df)} rows)")
                except Exception as e:
                    st.error(f"Failed to load open interest: {e}")
                    oi_df = None

            # Merge all into the 5m kline frame
            try:
                merged_all = merge_all_sources_to_5m(
                    kline_df=kline_df,
                    agg_df=agg_df,
                    depth_snapshots_df=depth_snap_df,
                    funding_df=funding_df,
                    oi_df=oi_df,
                    ts_col='timestamp',
                    resample_rule='5min'
                )
            except Exception as e:
                st.error(f"Failed while merging sources: {e}")
                st.stop()

        # show preview of merged frame
        st.subheader("Merged 5m frame (tail)")
        st.dataframe(merged_all.tail(8))

        # compute features
        with st.spinner("Computing indicators & features..."):
            try:
                features_df = build_features(merged_all, main_tf='5m', context_tfs=['15m'])
            except Exception as e:
                st.error(f"Feature computation failed: {e}")
                st.stop()

        if features_df is None or features_df.empty:
            st.error("Feature engineering returned empty DataFrame.")
            st.stop()

        st.success("Feature set built successfully!")
        st.subheader("Final Feature Set (tail)")
        st.dataframe(features_df.tail(10))

        st.write("Shape:", features_df.shape)
        st.write("Columns:", features_df.columns.tolist())

        # Save result
        out_name = os.path.splitext(os.path.basename(selected_path))[0] + "_features.csv"
        out_path = os.path.join(DATA_FOLDER, out_name)
        features_df.to_csv(out_path, index=False)
        st.success(f"Saved features to `{out_name}` in the `data/` folder.")
