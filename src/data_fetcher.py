# src/data_fetcher.py
import time
import math
import requests
import pandas as pd
from typing import Optional, List, Dict, Any

# Base endpoints
SPOT_BASE = "https://api.binance.com"
FUTURES_BASE = "https://fapi.binance.com"
# ---------------------------
# Helper
# ---------------------------
def _safe_get(url: str, params: Dict[str, Any] = None, headers: Dict[str, str] = None, timeout: int = 10):
    """Simple wrapper for requests.get with basic error handling."""
    params = params or {}
    headers = headers or {}
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    try:
        data = resp.json()
    except Exception:
        resp.raise_for_status()
    if isinstance(data, dict) and data.get("code") and data.get("msg"):
        # Binance error format
        raise Exception(f"Binance API Error: {data.get('code')} {data.get('msg')}")
    return data

# ---------------------------
# Klines (existing, improved)
# ---------------------------
def fetch_historical_klines(symbol: str = "BTCUSDT", interval: str = "5m",
                           start_time: Optional[int] = None, end_time: Optional[int] = None,
                           limit: int = 1000, sleep: float = 0.25) -> pd.DataFrame:
    url = f"{SPOT_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_time is not None:
        params["startTime"] = int(start_time)
    if end_time is not None:
        params["endTime"] = int(end_time)

    all_data = []
    while True:
        data = _safe_get(url, params=params)
        if not data:
            break
        all_data.extend(data)

        # Binance returns (open_time, open, high, low, close, volume, close_time, ...)
        last_open_time = int(data[-1][0])
        next_start = last_open_time + 1
        # stop if we've reached end_time
        if end_time and next_start >= end_time:
            break
        # prepare next page
        params["startTime"] = next_start
        # avoid rate-limit
        time.sleep(sleep)
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

# ---------------------------
# Trade-level data
# ---------------------------
def fetch_agg_trades(symbol: str = "BTCUSDT", start_time: Optional[int] = None,
                     end_time: Optional[int] = None, limit: int = 1000,
                     sleep: float = 0.2) -> pd.DataFrame:
    url = f"{SPOT_BASE}/api/v3/aggTrades"
    params = {"symbol": symbol, "limit": limit}
    if start_time is not None:
        params["startTime"] = int(start_time)
    if end_time is not None:
        params["endTime"] = int(end_time)

    all_trades = []
    while True:
        data = _safe_get(url, params=params)
        if not data:
            break
        all_trades.extend(data)

        last_ts = int(data[-1]["T"])
        next_start = last_ts + 1
        if end_time and next_start >= end_time:
            break
        params["startTime"] = next_start
        time.sleep(sleep)
        if len(data) < limit:
            break

    if not all_trades:
        return pd.DataFrame()

    df = pd.DataFrame(all_trades)
    # fields include: a(aggId), p, q, f, l, T, m, M
    df = df.rename(columns={
        "a": "aggId", "p": "price", "q": "qty", "f": "first_trade_id",
        "l": "last_trade_id", "T": "timestamp", "m": "is_buyer_maker"
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["price"] = df["price"].astype(float)
    df["qty"] = df["qty"].astype(float)
    return df[["aggId", "price", "qty", "first_trade_id", "last_trade_id", "timestamp", "is_buyer_maker"]]

def fetch_recent_trades(symbol: str = "BTCUSDT", limit: int = 500) -> pd.DataFrame:
    url = f"{SPOT_BASE}/api/v3/trades"
    params = {"symbol": symbol, "limit": limit}
    data = _safe_get(url, params=params)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    # fields: id, price, qty, quoteQty, time, isBuyerMaker, isMaker, ignore
    df = df.rename(columns={"time": "timestamp", "isBuyerMaker": "is_buyer_maker"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["price"] = df["price"].astype(float)
    df["qty"] = df["qty"].astype(float)
    return df[["id", "price", "qty", "quoteQty", "timestamp", "is_buyer_maker"]]

# ---------------------------
# Order book snapshots / depth
# ---------------------------
def fetch_order_book_snapshot(symbol: str = "BTCUSDT", limit: int = 100) -> Dict[str, Any]:
    url = f"{SPOT_BASE}/api/v3/depth"
    params = {"symbol": symbol, "limit": limit}
    data = _safe_get(url, params=params)
    return data

def order_book_as_dataframe(snapshot: Dict[str, Any]) -> pd.DataFrame:
    bids = snapshot.get("bids", [])
    asks = snapshot.get("asks", [])
    rows = []
    for p, q in bids:
        rows.append({"side": "bid", "price": float(p), "qty": float(q)})
    for p, q in asks:
        rows.append({"side": "ask", "price": float(p), "qty": float(q)})
    return pd.DataFrame(rows)

# ---------------------------
# Futures data (USDT-m, fapi)
# ---------------------------
def fetch_futures_funding_rate_history(symbol: str = "BTCUSDT", start_time: Optional[int] = None,
                                       end_time: Optional[int] = None, limit: int = 1000, sleep: float = 0.25) -> pd.DataFrame:
    url = f"{FUTURES_BASE}/fapi/v1/fundingRate"
    params = {"symbol": symbol, "limit": limit}
    if start_time is not None:
        params["startTime"] = int(start_time)
    if end_time is not None:
        params["endTime"] = int(end_time)

    all_rows = []
    while True:
        data = _safe_get(url, params=params)
        if not data:
            break
        all_rows.extend(data)
        last_ts = int(data[-1].get("fundingTime", data[-1].get("time", 0)))
        next_start = last_ts + 1
        if end_time and next_start >= end_time:
            break
        params["startTime"] = next_start
        time.sleep(sleep)
        if len(data) < limit:
            break

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    if "fundingTime" in df.columns:
        df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
    return df

def fetch_futures_open_interest(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """
    Get current open interest for a symbol (USDT-M futures).
    Endpoint: GET /fapi/v1/openInterest
    Returns dict with: symbol, openInterest (string), time
    """
    url = f"{FUTURES_BASE}/fapi/v1/openInterest"
    params = {"symbol": symbol}
    data = _safe_get(url, params=params)
    return data

def fetch_futures_open_interest_hist(pair: str = "BTCUSDT", contract_type: str = "PERPETUAL",
                                     period: str = "5m", start_time: Optional[int] = None,
                                     end_time: Optional[int] = None, limit: int = 500, sleep: float = 0.25) -> pd.DataFrame:
    """
    Fetch historical open interest statistics (aggregated) via:
      GET /futures/data/openInterestHist
    Params: pair (underlying), contractType (ALL/CURRENT_QUARTER/NEXT_QUARTER/PERPETUAL),
            period ("5m","15m","30m","1h",...), startTime, endTime, limit
    Note: This endpoint may return only recent days (doc varies); use responsibly.
    """
    url = f"{FUTURES_BASE}/futures/data/openInterestHist"
    params = {"pair": pair, "contractType": contract_type, "period": period, "limit": limit}
    if start_time is not None:
        params["startTime"] = int(start_time)
    if end_time is not None:
        params["endTime"] = int(end_time)

    all_rows = []
    while True:
        data = _safe_get(url, params=params)
        if not data:
            break
        all_rows.extend(data)
        # `openInterestHist` returns chronological pages; use last item's timestamp if available
        last_ts = None
        if isinstance(data[-1], dict):
            # some responses may include a 'timestamp' or 'time' key
            last_ts = data[-1].get("timestamp") or data[-1].get("time") or data[-1].get("openTime")
        if last_ts:
            next_start = int(last_ts) + 1
            if end_time and next_start >= end_time:
                break
            params["startTime"] = next_start
        else:
            break
        time.sleep(sleep)
        if len(data) < limit:
            break

    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    # attempt to parse common time fields to datetime
    for ts_col in ("timestamp", "time", "openTime"):
        if ts_col in df.columns:
            try:
                df[ts_col] = pd.to_datetime(df[ts_col], unit="ms")
            except Exception:
                pass
    return df

# ---------------------------
# Example: build simple microstructure features (helper)
# ---------------------------
def aggtrades_to_5s_bars(agg_df: pd.DataFrame, bucket_ms: int = 5000) -> pd.DataFrame:
    """
    Aggregate aggTrades DataFrame into fixed-time buckets (e.g., 5s).
    Expects agg_df with columns ['price','qty','timestamp','is_buyer_maker'] where timestamp is datetime.
    Returns DataFrame with columns: ts_start, open, high, low, close, volume, taker_buy_vol, taker_buy_ratio, trade_count
    """
    if agg_df.empty:
        return pd.DataFrame()

    df = agg_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["ts_ms"] = (df["timestamp"].view("int64") // 1_000_000).astype("int64")
    bucket_start = (df["ts_ms"] // bucket_ms) * bucket_ms
    gb = df.groupby(bucket_start)

    rows = []
    for bstart, g in gb:
        prices = g["price"].values
        o, h, l, c = prices[0], prices.max(), prices.min(), prices[-1]
        vol = g["qty"].sum()
        taker_buy_vol = g.loc[~g["is_buyer_maker"], "qty"].sum()  # note: is_buyer_maker True means buyer is maker; invert accordingly vs docs
        trade_count = len(g)
        taker_ratio = float(taker_buy_vol / vol) if vol > 0 else 0.0
        rows.append({
            "ts_start": pd.to_datetime(int(bstart), unit="ms"),
            "open": float(o),
            "high": float(h),
            "low": float(l),
            "close": float(c),
            "volume": float(vol),
            "taker_buy_vol": float(taker_buy_vol),
            "taker_buy_ratio": taker_ratio,
            "trade_count": trade_count
        })

    out = pd.DataFrame(rows).sort_values("ts_start").reset_index(drop=True)
    return out

# ---------------------------
# If module run directly, simple demo (will not execute on import)
# ---------------------------
if __name__ == "__main__":
    # quick smoke test (fetch a few klines and a depth snapshot)
    print("Fetching 10 1m klines for BTCUSDT...")
    df_kl = fetch_historical_klines("BTCUSDT", "1m", limit=10)
    print(df_kl.tail())

    print("Fetching depth snapshot...")
    snap = fetch_order_book_snapshot("BTCUSDT", limit=100)
    print("Top bids:", snap.get("bids", [])[:3])
