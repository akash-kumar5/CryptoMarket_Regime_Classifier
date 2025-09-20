# src/binance_fetcher.py
import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

SESSION = requests.Session()
OUT_DIR = "data"
KL_LIMIT = 1000
AGG_LIMIT = 1000
DEFAULT_CHUNK_MINUTES = 5
MAX_WORKERS = 6
SPOT_BASE = "https://api.binance.com"
FUTURES_BASE = "https://fapi.binance.com"
RETRY_SLEEP = 0.5

def ensure_out_dir(path: str = OUT_DIR):
    os.makedirs(path, exist_ok=True)

def ms_now() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)

def interval_to_millis(interval: str) -> int:
    unit = interval[-1]; val = int(interval[:-1])
    if unit == 'm': return val * 60 * 1000
    if unit == 'h': return val * 60 * 60 * 1000
    if unit == 'd': return val * 24 * 60 * 60 * 1000
    if unit == 'w': return val * 7 * 24 * 60 * 60 * 1000
    if unit == 'M': return val * 30 * 24 * 60 * 60 * 1000
    raise ValueError("Unsupported interval")

def request_with_retry(url: str, params: dict = None, max_retries: int = 5, timeout: int = 20) -> Any:
    for attempt in range(max_retries):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            wait = RETRY_SLEEP * (2 ** attempt)
            time.sleep(wait)
    raise RuntimeError(f"Failed to GET {url} after {max_retries} attempts")

# --- KLINES
def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int, out_path: Optional[str] = None) -> pd.DataFrame:
    ensure_out_dir()
    url = SPOT_BASE + "/api/v3/klines"
    interval_ms = interval_to_millis(interval)
    all_rows = []
    cur_start = start_ms
    while cur_start < end_ms:
        params = {"symbol": symbol, "interval": interval, "startTime": cur_start, "endTime": end_ms, "limit": KL_LIMIT}
        data = request_with_retry(url, params=params)
        if not data:
            break
        all_rows.extend(data)
        last_open = data[-1][0]
        cur_start = last_open + interval_ms
        if len(data) < KL_LIMIT:
            break
        time.sleep(0.12)
    cols = ["open_time_ms","open","high","low","close","volume","close_time_ms",
            "quote_asset_volume","num_trades","taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume","ignore"]
    df = pd.DataFrame(all_rows, columns=cols) if all_rows else pd.DataFrame(columns=cols)
    numeric = ["open","high","low","close","volume","quote_asset_volume",
               "taker_buy_base_asset_volume","taker_buy_quote_asset_volume"]
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "open_time_ms" in df.columns:
        df["timestamp"] = pd.to_datetime(df["open_time_ms"], unit='ms', utc=True)
        df = df[["timestamp","open","high","low","close","volume","num_trades","open_time_ms","close_time_ms"]]
    if out_path:
        df.to_csv(out_path, index=False)
    return df

# --- AGGTRADES (parallel chunked with internal paging)
AGG_ENDPOINT = SPOT_BASE + "/api/v3/aggTrades"

def _fetch_agg_chunk_paged(symbol: str, chunk_start_ms: int, chunk_end_ms: int) -> List[Dict[str,Any]]:
    collected = []
    cur_start = chunk_start_ms
    while cur_start <= chunk_end_ms:
        params = {"symbol": symbol, "startTime": int(cur_start), "endTime": int(chunk_end_ms), "limit": AGG_LIMIT}
        data = request_with_retry(AGG_ENDPOINT, params=params)
        if not data:
            break
        collected.extend(data)
        if len(data) < AGG_LIMIT:
            break
        last_trade_time = data[-1]['T']
        cur_start = last_trade_time + 1
        time.sleep(0.05)
    return collected

def fetch_aggtrades_parallel(symbol: str, start_ms: int, end_ms: int, out_path: Optional[str] = None,
                             chunk_minutes: int = DEFAULT_CHUNK_MINUTES, max_workers: int = MAX_WORKERS) -> pd.DataFrame:
    ensure_out_dir()
    ms_per_chunk = chunk_minutes * 60 * 1000
    ranges = []
    cur = start_ms
    while cur < end_ms:
        ranges.append((cur, min(cur + ms_per_chunk - 1, end_ms)))
        cur += ms_per_chunk
    all_trades = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_agg_chunk_paged, symbol, s, e): (s,e) for s,e in ranges}
        for fut in as_completed(futures):
            try:
                data = fut.result()
                if data: all_trades.extend(data)
            except Exception:
                pass
    if not all_trades:
        df = pd.DataFrame(columns=['aggTradeId','price','qty','first_trade_id','last_trade_id','timestamp'])
        if out_path: df.to_csv(out_path, index=False)
        return df
    df = pd.DataFrame(all_trades)
    for col in ['a','p','q','f','l','T','m']:
        if col not in df.columns: df[col] = pd.NA
    if 'a' in df.columns:
        df = df.drop_duplicates(subset=['a'])
    df['price'] = pd.to_numeric(df['p'], errors='coerce')
    df['qty'] = pd.to_numeric(df['q'], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['T'], unit='ms', utc=True)
    df.rename(columns={'a':'aggTradeId','f':'first_trade_id','l':'last_trade_id','m':'is_market_maker'}, inplace=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    if out_path: df.to_csv(out_path, index=False)
    return df

# --- ORDERBOOK snapshot
def fetch_orderbook_snapshot(symbol: str, limit: int = 100) -> Dict[str, Any]:
    url = SPOT_BASE + "/api/v3/depth"
    params = {"symbol": symbol, "limit": limit}
    data = request_with_retry(url, params=params)
    bids = pd.DataFrame(data.get('bids', []), columns=['price','qty']) if data.get('bids') else pd.DataFrame(columns=['price','qty'])
    asks = pd.DataFrame(data.get('asks', []), columns=['price','qty']) if data.get('asks') else pd.DataFrame(columns=['price','qty'])
    for df in (bids, asks):
        if not df.empty:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df['qty'] = pd.to_numeric(df['qty'], errors='coerce')
    return {"lastUpdateId": data.get('lastUpdateId'), "bids": bids, "asks": asks, "fetched_at": datetime.now(timezone.utc)}

# --- FUTURES (open interest current + funding history)
def fetch_futures(symbol: str, start_time_ms: int = None, end_time_ms: int = None) -> Dict[str,Any]:
    out = {}
    # open interest
    try:
        oi = request_with_retry(FUTURES_BASE + "/fapi/v1/openInterest", params={"symbol": symbol})
        out['open_interest_now'] = oi
    except Exception:
        out['open_interest_now'] = None
    # funding history
    try:
        params = {"symbol": symbol, "limit": 100}
        if start_time_ms: params['startTime'] = int(start_time_ms)
        if end_time_ms: params['endTime'] = int(end_time_ms)
        fr = request_with_retry(FUTURES_BASE + "/fapi/v1/fundingRate", params=params)
        df_fr = pd.DataFrame(fr) if fr else pd.DataFrame()
        if not df_fr.empty and 'fundingTime' in df_fr.columns:
            df_fr['fundingTime'] = pd.to_datetime(df_fr['fundingTime'], unit='ms', utc=True)
        out['funding'] = df_fr
    except Exception:
        out['funding'] = pd.DataFrame()
    return out

# --- small helper to build 5s bars from aggtrades (keeps timestamp column)
def aggtrades_to_5s_bars(agg_df: pd.DataFrame, bucket_ms: int = 5000) -> pd.DataFrame:
    if agg_df is None or agg_df.empty:
        return pd.DataFrame()
    df = agg_df.copy()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    elif 'trade_time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['trade_time'], utc=True, errors='coerce')
    else:
        df.iloc[:,0] = pd.to_datetime(df.iloc[:,0], utc=True, errors='coerce')
        df['timestamp'] = df.iloc[:,0]
    df = df.dropna(subset=['timestamp'])
    df['ts_ms'] = (df['timestamp'].view('int64') // 1_000_000).astype('int64')
    bucket_start = (df['ts_ms'] // bucket_ms) * bucket_ms
    gb = df.groupby(bucket_start)
    rows = []
    for bstart, g in gb:
        prices = g['price'].values
        o, h, l, c = prices[0], prices.max(), prices.min(), prices[-1]
        vol = g['qty'].sum()
        taker_buy_vol = g.loc[~g.get('is_market_maker', pd.Series(False)), 'qty'].sum() if 'is_market_maker' in g.columns else 0.0
        trade_count = len(g)
        taker_ratio = float(taker_buy_vol / vol) if vol > 0 else 0.0
        rows.append({
            "timestamp": pd.to_datetime(int(bstart), unit="ms", utc=True),
            "open": float(o),
            "high": float(h),
            "low": float(l),
            "close": float(c),
            "volume": float(vol),
            "taker_buy_vol": float(taker_buy_vol),
            "taker_buy_ratio": taker_ratio,
            "trade_count": trade_count
        })
    out = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return out
