# src/compute_features.py
import numpy as np
import pandas as pd
import talib
from typing import Optional

EPS = 1e-12

# -------------------------
# Utilities
# -------------------------
def _ensure_dt(df: pd.DataFrame, ts_col: str = 'timestamp') -> pd.DataFrame:
    df = df.copy()
    if ts_col in df.columns and not np.issubdtype(df[ts_col].dtype, np.datetime64):
        df[ts_col] = pd.to_datetime(df[ts_col])
    return df

def safe_talib(func, *args, **kwargs):
    """
    Call a talib function defensively and return a numpy array
    of NaNs on failure (length inferred from first arg).
    """
    try:
        return func(*args, **kwargs)
    except Exception:
        if len(args) > 0:
            n = len(args[0])
            return np.full(n, np.nan)
        return np.array([])

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
def aggregate_aggtrades_to_5m(
    agg_df: pd.DataFrame,
    ts_col: str = 'timestamp',
    price_col: str = 'price',
    qty_candidates = ('qty', 'q', 'quantity', 'volume'),
    taker_buy_vol_col: str = 'taker_buy_vol',
    resample_rule: str = '5T'
) -> pd.DataFrame:
    """
    Aggregate raw aggtrades (tick or short-interval) to 5m microstructure features.
    Returns DataFrame with timestamp (period start) and columns:
      trade_count_5m, volume_5m_from_agg, vwap_all_5m,
      taker_buy_vol_5m, taker_buy_ratio_5m, trade_imbalance_5m, vwap_skew_5m, max_tick_ret_5m
    """
    if agg_df is None or agg_df.empty:
        return pd.DataFrame(columns=['timestamp'])

    df = agg_df.copy()
    df = _ensure_dt(df, ts_col)
    df = df.sort_values(ts_col)

    # pick qty col
    qty_col = None
    for c in qty_candidates:
        if c in df.columns:
            qty_col = c
            break
    if qty_col is None:
        raise ValueError(f"aggregate_aggtrades_to_5m: no qty column found. Tried {qty_candidates}")

    # ensure numeric
    df[price_col] = df[price_col].astype(float)
    df[qty_col] = df[qty_col].astype(float)
    if taker_buy_vol_col in df.columns:
        df[taker_buy_vol_col] = df[taker_buy_vol_col].astype(float)

    df = df.set_index(ts_col)

    def agg_func(g: pd.DataFrame):
        out = {}
        out['trade_count_5m'] = int(g.shape[0])
        out['volume_5m_from_agg'] = g[qty_col].sum()
        total_vol = out['volume_5m_from_agg'] + EPS
        out['vwap_all_5m'] = (g[price_col] * g[qty_col]).sum() / total_vol

        # taker fields if present
        if taker_buy_vol_col in g.columns:
            out['taker_buy_vol_5m'] = g[taker_buy_vol_col].sum()
            out['taker_buy_ratio_5m'] = out['taker_buy_vol_5m'] / total_vol
            out['trade_imbalance_5m'] = (2.0 * out['taker_buy_vol_5m'] - out['volume_5m_from_agg']) / total_vol
            taker_sum = g[taker_buy_vol_col].sum()
            if taker_sum > 0:
                out['vwap_taker_5m'] = (g[price_col] * g[taker_buy_vol_col]).sum() / (taker_sum + EPS)
                out['vwap_skew_5m'] = out['vwap_taker_5m'] - out['vwap_all_5m']
            else:
                out['vwap_taker_5m'] = np.nan
                out['vwap_skew_5m'] = 0.0
        else:
            out['taker_buy_vol_5m'] = np.nan
            out['taker_buy_ratio_5m'] = np.nan
            out['trade_imbalance_5m'] = np.nan
            out['vwap_skew_5m'] = np.nan

        # max tick return inside bucket
        if len(g) > 1:
            prices = g[price_col]
            logrets = np.log(prices / prices.shift(1) + EPS).dropna()
            out['max_tick_ret_5m'] = float(logrets.abs().max()) if not logrets.empty else 0.0
        else:
            out['max_tick_ret_5m'] = 0.0

        return pd.Series(out)

    agg5 = df.resample(resample_rule).apply(lambda g: agg_func(g))
    agg5 = agg5.reset_index().rename(columns={agg5.index.name: 'timestamp'}) if 'timestamp' not in agg5.columns else agg5.reset_index()
    # ensure timestamp column exists
    if agg5.index.name is not None:
        agg5 = agg5.reset_index().rename(columns={agg5.index.name: 'timestamp'})
    agg5 = agg5.reset_index(drop=True)
    # keep timestamp as column
    if 'timestamp' not in agg5.columns and agg5.index.name == 'timestamp':
        agg5['timestamp'] = agg5.index
    return agg5

def aggregate_depth_snapshot_to_5m(
    depth_snap_df: pd.DataFrame,
    ts_col: str = 'timestamp',
    bids_col: str = 'bids',
    asks_col: str = 'asks',
    resample_rule: str = '5T',
    band_pcts = (0.001, 0.005)
) -> pd.DataFrame:
    """
    Convert depth snapshot rows (each row contains bids/asks lists of [price,qty])
    to per-5m snapshot features: spread_pct, bid_depth_10bps, ask_depth_10bps, pressure_index_10bps, etc.
    """
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
    resample_rule: str = '5T'
) -> pd.DataFrame:
    """
    Merge kline (5m) with aggregated aggtrades, depth snapshot features, funding, and OI.
    Returns a kline-aligned DataFrame containing original kline columns plus:
      trade_count_5m, volume_5m_from_agg, taker_buy_ratio_5m, trade_imbalance_5m, vwap_skew_5m, max_tick_ret_5m,
      spread_pct, bid_depth_10bps, ask_depth_10bps, pressure_index_10bps, ... (for bands),
      fundingRate, fundingRate_d_1h, openInterest, d_oi_5m
    """
    k = _ensure_dt(kline_df, ts_col).sort_values(ts_col).reset_index(drop=True)

    # aggtrades
    if agg_df is not None and not agg_df.empty:
        agg5 = aggregate_aggtrades_to_5m(agg_df, ts_col=ts_col, resample_rule=resample_rule)
        # merge_asof: align latest agg bucket <= kline timestamp
        k = pd.merge_asof(k.sort_values(ts_col), agg5.sort_values('timestamp'),
                          left_on=ts_col, right_on='timestamp', direction='backward', tolerance=pd.Timedelta(resample_rule))
        # drop the extra timestamp column from agg5 if present
        if 'timestamp_y' in k.columns:
            k = k.drop(columns=['timestamp_y'])
        # rename timestamp_x -> timestamp if happened
        if 'timestamp_x' in k.columns:
            k = k.rename(columns={'timestamp_x':'timestamp'})

    # depth snapshots
    if depth_snapshots_df is not None and not depth_snapshots_df.empty:
        snap5 = aggregate_depth_snapshot_to_5m(depth_snapshots_df, ts_col=ts_col, resample_rule=resample_rule)
        k = pd.merge_asof(k.sort_values(ts_col), snap5.sort_values(ts_col),
                          left_on=ts_col, right_on=ts_col, direction='backward', tolerance=pd.Timedelta(resample_rule))

    # funding & oi
    k = merge_funding_and_oi_to_5m(k, funding_df=funding_df, oi_df=oi_df, ts_col=ts_col)

    # housekeeping: fill aggregated numeric NaNs with 0 for safety (you can change this policy)
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
    """
    Compute OHLCV-derived features for main_tf and context_tfs in df.
    Expects df to already contain columns like close_5m, high_5m, low_5m, open_5m, volume_5m, and similar for context tfs.
    Shifts context features by 1 bar to prevent lookahead bias.

    Returns feature DataFrame (index reset).
    """
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
