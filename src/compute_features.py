# src/compute_features.py
import numpy as np
import pandas as pd
import talib
from typing import Optional

EPS = 1e-12
import json

def parse_datetime_series(s: pd.Series) -> pd.Series:
    """Robustly parse mixed datetime formats and epoch numbers to tz-aware UTC datetimes."""
    # Try straightforward parse first (no infer_datetime_format arg)
    out = pd.to_datetime(s, utc=True, errors='coerce')

    # If everything parsed, return
    if out.notna().all():
        return out

    # Fallback: attempt cleaning (strip quotes) and numeric epoch detection
    raw = s.astype(str).str.strip().str.replace('"', '')
    is_digits = raw.str.match(r'^\d{10,16}$')  # seconds or ms-ish
    if is_digits.any():
        sample = raw[is_digits].iloc[0]
        try:
            if len(sample) >= 13:
                # epoch ms
                out.loc[is_digits] = pd.to_datetime(raw[is_digits].astype(np.int64), unit='ms', utc=True, errors='coerce')
            else:
                # epoch seconds
                out.loc[is_digits] = pd.to_datetime(raw[is_digits].astype(np.int64), unit='s', utc=True, errors='coerce')
        except Exception:
            pass

    # Final attempt: per-item parse (slower but robust)
    mask = out.isna()
    if mask.any():
        out.loc[mask] = raw[mask].apply(lambda x: pd.to_datetime(x, utc=True, errors='coerce'))

    return out



# -------------------------
# Utilities
# -------------------------
def parse_depth_snapshot_json(path: str, ts_field_candidates=('fetched_at','fetchedAt','timestamp','time')):
    """
    Load depth snapshot JSON/JSONL, rename parsed timestamp column to 'timestamp' (UTC),
    and ensure bids/asks are lists of [float_price, float_qty].
    """
    # load as json lines or single json
    try:
        df = pd.read_json(path, lines=True)
    except ValueError:
        with open(path, 'r') as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            payload = [payload]
        df = pd.DataFrame(payload)

    if df.empty:
        return df

    # find timestamp column
    ts_col = None
    for c in ts_field_candidates:
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        # fallback heuristics
        poss = [c for c in df.columns if 'time' in c.lower() or 'fetch' in c.lower()]
        ts_col = poss[0] if poss else None
    if ts_col is None:
        raise ValueError("No timestamp-like field in depth snapshot")

    # parse timestamp robustly
    df['timestamp'] = parse_datetime_series(df[ts_col])

    # robust normalizer for bids/asks
    def norm_side(cell):
        # handle None / NaN
        if cell is None:
            return []
        # catch pandas NA / numpy NaN scalars
        if isinstance(cell, (float,)) and pd.isna(cell):
            return []
        # if it's a numpy array, convert to list
        if isinstance(cell, (np.ndarray, tuple)):
            parsed = list(cell)
        elif isinstance(cell, list):
            parsed = cell
        elif isinstance(cell, str):
            # try parsing JSON string like '[["115349.39","4.81254"], ...]'
            try:
                parsed = json.loads(cell)
            except Exception:
                # sometimes strings can be like "[(...)]" or other; fail gracefully
                return []
        else:
            # unknown type — try to coerce to list if iterable
            try:
                parsed = list(cell)
            except Exception:
                return []

        out = []
        for item in parsed:
            # item should be a 2-item sequence: [price, qty]
            try:
                # guard against nested arrays represented as numpy types or strings
                p = float(item[0])
                q = float(item[1])
                out.append([p, q])
            except Exception:
                # skip malformed entries silently
                continue
        return out

    if 'bids' in df.columns:
        df['bids'] = df['bids'].apply(norm_side)
    if 'asks' in df.columns:
        df['asks'] = df['asks'].apply(norm_side)

    # keep useful cols
    keep = ['timestamp']
    if 'lastUpdateId' in df.columns:
        keep.append('lastUpdateId')
    if 'bids' in df.columns:
        keep.append('bids')
    if 'asks' in df.columns:
        keep.append('asks')
    return df[keep]


def _ensure_dt(df: pd.DataFrame, ts_col: str = 'timestamp') -> pd.DataFrame:
    """
    Ensure df[ts_col] exists and is timezone-aware UTC datetime64[ns, UTC].
    If column missing, returns df unchanged.
    """
    df = df.copy()
    if ts_col not in df.columns:
        return df
    # Use the robust parser you defined above
    df[ts_col] = parse_datetime_series(df[ts_col])
    return df

def safe_talib(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception:
        n= len(args[0]) if len(args) > 0 else 0
        nan_arr = np.full(n, np.nan)
        return (nan_arr, nan_arr,nan_arr)

def robust_zscore(s: pd.Series, window: int = 50) -> pd.Series:
    med = s.rolling(window=window, min_periods=1).median()
    mad = s.rolling(window=window, min_periods=1).apply(
        lambda x: np.median(np.abs(x - np.median(x))) if len(x) > 0 else np.nan, raw=True
    )
    mad_adj = mad * 1.4826 + EPS
    return (s - med) / mad_adj

# -------------------------
# Aggregation helpers
# -------------------------
def load_and_normalize_funding(path: str):
    try:
        if path.endswith('.json') or path.endswith('.jsonl'):
            df = pd.read_json(path, lines=True)
        else:
            df = pd.read_csv(path)
    except Exception:
        # try json-lines manual
        try:
            with open(path, 'r') as f:
                lines = f.read().strip().splitlines()
            parsed = [json.loads(l) for l in lines if l.strip()]
            df = pd.DataFrame(parsed)
        except Exception:
            return None

    if df is None or df.empty:
        return None

    # Try to find a timestamp column
    ts_candidates = ['timestamp', 'fundingTime', 'funding_time', 'time', 'fetched_at']
    ts_col = next((c for c in ts_candidates if c in df.columns), None)
    if ts_col is None:
        # try heuristic
        possible = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()]
        ts_col = possible[0] if possible else None

    # create timestamp column (may be NaT if unable to parse)
    if ts_col is not None:
        df['timestamp'] = parse_datetime_series(df[ts_col])
    else:
        # create timestamp column of NaT so merges won't break, but caller can see it's empty
        df['timestamp'] = pd.NaT

    # find funding rate column
    fr_col = None
    if 'fundingRate' in df.columns:
        fr_col = 'fundingRate'
    elif 'funding_rate' in df.columns:
        fr_col = 'funding_rate'
        df = df.rename(columns={fr_col: 'fundingRate'})
        fr_col = 'fundingRate'
    else:
        cand = next((c for c in df.columns if 'fund' in c.lower() and 'rate' in c.lower()), None)
        if cand:
            df = df.rename(columns={cand: 'fundingRate'})
            fr_col = 'fundingRate'

    if fr_col is not None:
        df['fundingRate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
    else:
        df['fundingRate'] = np.nan

    return df[['timestamp','fundingRate']].sort_values('timestamp').reset_index(drop=True)

def load_and_normalize_oi(path: str):
    """
    Robust loader for open interest files.
    Handles:
      - JSON lines or single JSON objects with fields like {"time": 1758482274421, "openInterest": "88794.463"}
      - CSV with time/open_interest columns
    Returns DataFrame with ['timestamp' (datetime64[ns, UTC]), 'openInterest' (float)] or None.
    """
    try:
        if path.endswith('.json') or path.endswith('.jsonl') or path.endswith('.ndjson'):
            df = pd.read_json(path, lines=True)
        else:
            df = pd.read_csv(path)
    except Exception:
        # try manual json load
        try:
            with open(path, 'r') as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                payload = [payload]
            df = pd.DataFrame(payload)
        except Exception:
            return None

    if df is None or df.empty:
        return None

    # Normalize time -> timestamp
    if 'time' in df.columns:
        # time is epoch ms (as in your example)
        # coerce to numeric then to datetime
        try:
            df['timestamp'] = pd.to_datetime(pd.to_numeric(df['time'], errors='coerce'), unit='ms', utc=True)
        except Exception:
            # fallback to robust parser
            df['timestamp'] = parse_datetime_series(df['time'].astype(str))
    else:
        # fallback to other possible time cols or robust parsing
        tcol = next((c for c in df.columns if 'time' in c.lower() or 'timestamp' in c.lower()), None)
        if tcol:
            df['timestamp'] = parse_datetime_series(df[tcol])
        else:
            df['timestamp'] = pd.NaT

    # Normalize openInterest -> numeric
    if 'openInterest' in df.columns:
        df['openInterest'] = pd.to_numeric(df['openInterest'], errors='coerce')
    elif 'open_interest' in df.columns:
        df['openInterest'] = pd.to_numeric(df['open_interest'], errors='coerce')
    else:
        # try to guess a column name containing both tokens
        cand = next((c for c in df.columns if 'open' in c.lower() and 'interest' in c.lower()), None)
        if cand:
            df['openInterest'] = pd.to_numeric(df[cand], errors='coerce')
        else:
            df['openInterest'] = np.nan

    # keep only useful cols and sort
    out = df[['timestamp','openInterest']].copy()
    out = out.sort_values('timestamp').reset_index(drop=True)
    return out

def aggregate_aggtrades_to_5m(
    agg_df: pd.DataFrame,
    ts_col: str = 'timestamp',
    price_candidates=('price','p','P','Price'),
    qty_candidates=('qty','q','quantity','volume','Q'),
    taker_buy_vol_col: str = 'taker_buy_vol',
    resample_rule: str = '5min'
) -> pd.DataFrame:

    if agg_df is None or agg_df.empty:
        return pd.DataFrame(columns=[ts_col])

    df = agg_df.copy()

    # normalize timestamp if present
    if ts_col in df.columns:
        df[ts_col] = parse_datetime_series(df[ts_col])
    else:
        # try common alternates
        for alt in ('T','time','timestamp_ms','ts'):
            if alt in df.columns:
                df[ts_col] = parse_datetime_series(df[alt])
                break

    # detect whether file is per-trade (price+qty) or pre-aggregated OHLCV
    price_col = next((c for c in price_candidates if c in df.columns), None)
    qty_col = next((c for c in qty_candidates if c in df.columns), None)
    has_ohlcv = all(c in df.columns for c in ('open', 'high', 'low', 'close', 'volume'))

    # If neither price/qty nor OHLCV present -> fail with helpful message
    if (price_col is None or qty_col is None) and not has_ohlcv:
        raise KeyError(
            "aggregate_aggtrades_to_5m: couldn't find price/qty or ohlcv columns. "
            f"Found columns: {list(df.columns)}. Expected price in {price_candidates} or ohlcv columns."
        )

    # If OHLCV provided, treat each row as a micro-bucket tick and resample
    if has_ohlcv and (price_col is None or qty_col is None):
        # use 'close' as representative price and 'volume' as qty
        df = df.rename(columns={'close': 'tick_price', 'volume': 'tick_qty'}, errors='ignore')
        price_col = 'tick_price'
        qty_col = 'tick_qty'
        df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
        df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce')
        # ensure taker column numeric if present
        if taker_buy_vol_col in df.columns:
            df[taker_buy_vol_col] = pd.to_numeric(df[taker_buy_vol_col], errors='coerce')
    else:
        # per-trade case: coerce numeric types
        df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
        df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce')
        if taker_buy_vol_col in df.columns:
            df[taker_buy_vol_col] = pd.to_numeric(df[taker_buy_vol_col], errors='coerce')

    # set index and resample
    if ts_col not in df.columns:
        # if still no timestamp, try to proceed but warn
        raise KeyError("aggregate_aggtrades_to_5m: no timestamp column found after attempts.")
    df = df.set_index(ts_col).sort_index()

    def agg_func(g: pd.DataFrame):
        out = {}
        total_vol = float(g[qty_col].sum()) if qty_col in g.columns else 0.0
        out['trade_count_5m'] = int(g.shape[0])
        out['volume_5m_from_agg'] = total_vol
        out['vwap_all_5m'] = float((g[price_col] * g[qty_col]).sum() / (total_vol + EPS)) if total_vol > 0 else np.nan

        if taker_buy_vol_col in g.columns:
            tb = float(g[taker_buy_vol_col].sum())
            out['taker_buy_vol_5m'] = tb
            out['taker_buy_ratio_5m'] = tb / (total_vol + EPS) if total_vol > 0 else np.nan
            out['trade_imbalance_5m'] = (2.0*tb - total_vol) / (total_vol + EPS) if total_vol > 0 else np.nan
            if tb > 0:
                out['vwap_taker_5m'] = float((g[price_col] * g[taker_buy_vol_col]).sum() / (tb + EPS))
                out['vwap_skew_5m'] = out['vwap_taker_5m'] - out['vwap_all_5m']
            else:
                out['vwap_taker_5m'] = np.nan
                out['vwap_skew_5m'] = 0.0
        else:
            out['taker_buy_vol_5m'] = np.nan
            out['taker_buy_ratio_5m'] = np.nan
            out['trade_imbalance_5m'] = np.nan
            out['vwap_skew_5m'] = np.nan

        # max tick return inside bucket (use per-row price sequence)
        if len(g) > 1:
            prices = g[price_col].dropna()
            if len(prices) > 1:
                lr = np.log(prices / prices.shift(1) + EPS).dropna()
                out['max_tick_ret_5m'] = float(lr.abs().max()) if not lr.empty else 0.0
            else:
                out['max_tick_ret_5m'] = 0.0
        else:
            out['max_tick_ret_5m'] = 0.0
        return pd.Series(out)

    agg5 = df.resample(resample_rule).apply(lambda g: agg_func(g))
    agg5 = agg5.reset_index()
    # normalize timestamp tz
    agg5[ts_col] = parse_datetime_series(agg5[ts_col])
    return agg5



def aggregate_depth_snapshot_to_5m(
    depth_snap_df: pd.DataFrame,
    ts_col: str = 'timestamp',
    bids_col: str = 'bids',
    asks_col: str = 'asks',
    resample_rule: str = '5min',
    band_pcts = (0.001, 0.005)
) -> pd.DataFrame:
    if depth_snap_df is None or depth_snap_df.empty:
        return pd.DataFrame(columns=['timestamp'])

    df = depth_snap_df.copy()
    df = _ensure_dt(df, ts_col)
    df = df.sort_values(ts_col)

    def compute_row_features(row):
        out = {}
        bids = row.get(bids_col, None)
        asks = row.get(asks_col, None)
        if (not bids) or (not asks):
            # missing data
            for pct in band_pcts:
                out[f'bid_depth_{int(pct*10000)}bps'] = np.nan
                out[f'ask_depth_{int(pct*10000)}bps'] = np.nan
                out[f'pressure_index_{int(pct*10000)}bps'] = np.nan
            out['spread_pct'] = np.nan
            return pd.Series(out)

        try:
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
        except Exception:
            # unexpected format
            for pct in band_pcts:
                out[f'bid_depth_{int(pct*10000)}bps'] = np.nan
                out[f'ask_depth_{int(pct*10000)}bps'] = np.nan
                out[f'pressure_index_{int(pct*10000)}bps'] = np.nan
            out['spread_pct'] = np.nan
            return pd.Series(out)

        mid = (best_bid + best_ask) / 2.0
        out['spread_pct'] = (best_ask - best_bid) / (mid + EPS)

        for pct in band_pcts:
            low_bound = mid * (1.0 - pct)
            high_bound = mid * (1.0 + pct)
            bid_qty = 0.0
            for p, q in bids:
                p = float(p); q = float(q)
                if p >= low_bound:
                    bid_qty += q
            ask_qty = 0.0
            for p, q in asks:
                p = float(p); q = float(q)
                if p <= high_bound:
                    ask_qty += q
            out[f'bid_depth_{int(pct*10000)}bps'] = bid_qty
            out[f'ask_depth_{int(pct*10000)}bps'] = ask_qty
            out[f'pressure_index_{int(pct*10000)}bps'] = bid_qty / (ask_qty + EPS)
        return pd.Series(out)

    snap_feats = df.apply(compute_row_features, axis=1)
    snap_feats[ts_col] = df[ts_col].values
    snap_feats = snap_feats.set_index(ts_col).resample(resample_rule).last().reset_index()
    return snap_feats

def merge_funding_and_oi_to_5m(
    kline_df: pd.DataFrame,
    funding_df: Optional[pd.DataFrame] = None,
    oi_df: Optional[pd.DataFrame] = None,
    ts_col: str = 'timestamp'
) -> pd.DataFrame:
    """
    Merge funding and open interest data into kline frame using merge_asof (latest known <= candle time).
    Adds fundingRate, fundingRate_d_1h (12 bars), openInterest, d_oi_5m
    """
    k = _ensure_dt(kline_df, ts_col).sort_values(ts_col).reset_index(drop=True)
    merged = k[[ts_col]].copy()

    if funding_df is not None and not funding_df.empty:
        f = _ensure_dt(funding_df, ts_col).sort_values(ts_col)
        # normalize column names if necessary
        if 'fundingTime' in f.columns and 'timestamp' not in f.columns:
            f = f.rename(columns={'fundingTime':'timestamp'})
        if 'fundingRate' not in f.columns and 'funding_rate' in f.columns:
            f = f.rename(columns={'funding_rate':'fundingRate'})
        f = f[[ts_col, 'fundingRate']].dropna(subset=[ts_col])
        merged = pd.merge_asof(merged, f.sort_values(ts_col), left_on=ts_col, right_on=ts_col, direction='backward')
        merged['fundingRate'] = merged['fundingRate'].astype(float)
        merged['fundingRate_d_1h'] = merged['fundingRate'].diff(periods=12).fillna(0.0)

    if oi_df is not None and not oi_df.empty:
        o = _ensure_dt(oi_df, ts_col).sort_values(ts_col)
        if 'time' in o.columns and 'timestamp' not in o.columns:
            o = o.rename(columns={'time':'timestamp'})
        if 'openInterest' not in o.columns and 'open_interest' in o.columns:
            o = o.rename(columns={'open_interest':'openInterest'})
        o = o[[ts_col, 'openInterest']].dropna(subset=[ts_col])
        merged = pd.merge_asof(merged, o.sort_values(ts_col), left_on=ts_col, right_on=ts_col, direction='backward')
        merged['openInterest'] = merged['openInterest'].astype(float)
        merged['d_oi_5m'] = merged['openInterest'].diff(periods=1).fillna(0.0)

    result = pd.merge(k, merged, on=ts_col, how='left')
    return result

# -------------------------
# Top-level merge
# -------------------------
def merge_all_sources_to_5m(
    kline_df: pd.DataFrame,
    agg_df: Optional[pd.DataFrame] = None,
    depth_snapshots_df: Optional[pd.DataFrame] = None,
    funding_df: Optional[pd.DataFrame] = None,
    oi_df: Optional[pd.DataFrame] = None,
    ts_col: str = 'timestamp',
    resample_rule: str = '5min'
) -> pd.DataFrame:

    # normalize kline
    k = _ensure_dt(kline_df, ts_col)
    if ts_col not in k.columns:
        raise ValueError(f"kline missing timestamp column '{ts_col}'")
    k = k.sort_values(ts_col).reset_index(drop=True)

    # helper to normalize optional dfs
    def _norm(df):
        if df is None:
            return None
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None
        if ts_col in df.columns:
            df = df.copy()
            df[ts_col] = parse_datetime_series(df[ts_col])
            df = df.sort_values(ts_col).reset_index(drop=True)
            return df
        return None

    agg_in = _norm(agg_df)
    depth_in = _norm(depth_snapshots_df)
    fund_in = _norm(funding_df)
    oi_in = _norm(oi_df)

    # aggtrades merge (only if agg_in has timestamp)
    if agg_in is not None:
        agg5 = aggregate_aggtrades_to_5m(agg_in, ts_col=ts_col, resample_rule=resample_rule)
        if ts_col in agg5.columns:
            agg5[ts_col] = parse_datetime_series(agg5[ts_col])
            agg5 = agg5.sort_values(ts_col).reset_index(drop=True)
            k = pd.merge_asof(k, agg5, left_on=ts_col, right_on=ts_col, direction='backward', tolerance=pd.Timedelta(resample_rule))
        else:
            # skip agg merge if agg5 lacks timestamp
            pass

    # depth snapshot merge
    if depth_in is not None and 'timestamp' in depth_in.columns:
        snap5 = aggregate_depth_snapshot_to_5m(depth_in, ts_col=ts_col, resample_rule=resample_rule)
        if 'timestamp' in snap5.columns:
            snap5['timestamp'] = parse_datetime_series(snap5['timestamp'])
            snap5 = snap5.sort_values('timestamp').reset_index(drop=True)
            k = pd.merge_asof(k, snap5, left_on=ts_col, right_on='timestamp', direction='backward', tolerance=pd.Timedelta(resample_rule))

    # funding & oi merges
    if fund_in is not None and 'timestamp' in fund_in.columns:
        fund_in = fund_in.sort_values('timestamp').reset_index(drop=True)
        k = pd.merge_asof(k, fund_in, left_on=ts_col, right_on='timestamp', direction='backward')
        # safe fill
        k['fundingRate'] = k.get('fundingRate', pd.Series(dtype=float)).fillna(0.0)
        k['fundingRate_d_1h'] = k['fundingRate'].diff(periods=12).fillna(0.0)

    if oi_in is not None and 'timestamp' in oi_in.columns:
        oi_in = oi_in.sort_values('timestamp').reset_index(drop=True)
        k = pd.merge_asof(k, oi_in, left_on=ts_col, right_on='timestamp', direction='backward')
        k['openInterest'] = k.get('openInterest', pd.Series(dtype=float)).fillna(method='ffill').fillna(0.0)
        k['d_oi_5m'] = k['openInterest'].diff(periods=1).fillna(0.0)

    # final fill for aggregated numeric columns
    agg_cols = [c for c in k.columns if c not in ['open_5m','high_5m','low_5m','close_5m','volume_5m','timestamp'] and pd.api.types.is_numeric_dtype(k[c])]
    k[agg_cols] = k[agg_cols].fillna(0.0)

    return k


# -------------------------
# Feature computation (OHLCV indicators)
# -------------------------
def build_features(
    df: pd.DataFrame,
    main_tf: str = '5m',
    context_tfs: Optional[list] = None,
    use_robust_volume_z: bool = False,
    dropna: bool = True
) -> pd.DataFrame:
    if context_tfs is None:
        context_tfs = ['15m']

    df_features = df.copy()
    all_tfs = [main_tf] + [tf for tf in context_tfs if tf != main_tf]

    # basic sanity check for main tf presence
    required_main = [f'close_{main_tf}', f'open_{main_tf}', f'high_{main_tf}', f'low_{main_tf}', f'volume_{main_tf}']
    missing = [c for c in required_main if c not in df_features.columns]
    if missing:
        raise ValueError(f"build_features: missing required main timeframe columns: {missing}")

    for tf in all_tfs:
        close_col = f'close_{tf}'
        high_col = f'high_{tf}'
        low_col = f'low_{tf}'
        open_col = f'open_{tf}'
        vol_col = f'volume_{tf}'

        if not all(c in df_features.columns for c in (close_col, high_col, low_col, open_col, vol_col)):
            # skip TF if columns not present
            continue

        close = df_features[close_col].astype(float)
        high = df_features[high_col].astype(float)
        low = df_features[low_col].astype(float)
        open_ = df_features[open_col].astype(float)
        volume = df_features[vol_col].astype(float)

        # Returns
        df_features[f'log_ret_1_{tf}'] = np.log(close / close.shift(1) + EPS)

        # Trend: EMA ratio 9/21
        ema9 = pd.Series(safe_talib(talib.EMA, close.values, timeperiod=9), index=df_features.index)
        ema21 = pd.Series(safe_talib(talib.EMA, close.values, timeperiod=21), index=df_features.index)
        df_features[f'ema_ratio_9_21_{tf}'] = ema9 / (ema21 + EPS)

        # MACD hist
        macd, macd_signal, macd_hist = safe_talib(talib.MACD, close.values, 12, 26, 9)
        df_features[f'macd_hist_{tf}'] = pd.Series(macd_hist, index=df_features.index)

        # ADX
        adx = pd.Series(safe_talib(talib.ADX, high.values, low.values, close.values, timeperiod=14), index=df_features.index)
        df_features[f'adx_{tf}'] = adx

        # Volatility: ATR normalized and BB width
        atr14 = pd.Series(safe_talib(talib.ATR, high.values, low.values, close.values, timeperiod=14), index=df_features.index)
        df_features[f'atr_norm_{tf}'] = atr14 / (close + EPS)
        upper, middle, lower = safe_talib(talib.BBANDS, close.values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        upper_s = pd.Series(upper, index=df_features.index)
        middle_s = pd.Series(middle, index=df_features.index)
        lower_s = pd.Series(lower, index=df_features.index)
        df_features[f'bb_width_{tf}'] = (upper_s - lower_s) / (middle_s + EPS)

        # Momentum
        df_features[f'rsi_14_{tf}'] = pd.Series(safe_talib(talib.RSI, close.values, timeperiod=14), index=df_features.index)

        # Volume z-score (50-window) - robust or mean/std
        if use_robust_volume_z:
            df_features[f'volume_zscore_50_{tf}'] = robust_zscore(volume, window=50)
        else:
            vol_mean_50 = volume.rolling(window=50, min_periods=1).mean()
            vol_std_50 = volume.rolling(window=50, min_periods=1).std().replace(0, EPS)
            df_features[f'volume_zscore_50_{tf}'] = (volume - vol_mean_50) / vol_std_50

    # Shift context features by 1 main bar to prevent lookahead bias
    # Select context-derived columns explicitly by suffix
    context_feature_cols = []
    for tf in context_tfs:
        suffix = f'_{tf}'
        cols = [
            c for c in df_features.columns
            if c.endswith(suffix) and not any(c.startswith(prefix) for prefix in ('open_', 'high_', 'low_', 'close_', 'volume_'))
        ]
        context_feature_cols.extend(cols)

    if context_feature_cols:
        # shift by 1 row: assumption is df rows are main_tf cadence
        df_features.loc[:, context_feature_cols] = df_features.loc[:, context_feature_cols].shift(1)

    if dropna:
        df_features = df_features.dropna().reset_index(drop=True)
    else:
        df_features = df_features.reset_index(drop=True)

    return df_features
