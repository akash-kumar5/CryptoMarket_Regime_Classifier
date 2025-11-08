# main.py

import os
import sys
import json
import time
import joblib
import schedule
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from tensorflow.keras.models import load_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.fetch_binance_klines import fetch_binance_klines  # returns short cols; we standardize here


MODEL_FOLDER = "models/"
SYMBOL = "BTCUSDT"
MAIN_TIMEFRAME = "5m"
CONTEXT_TIMEFRAMES = ["15m", "1m"]
TIME_STEPS = 64


def standardize_ohlcv(raw_dataframe: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Convert raw kline frame with short columns (t,o,h,l,c,v) to:
    timestamp/open/high/low/close/volume.
    The timestamp is tz-aware UTC and floored to the timeframe start.
    """
    # Allow this function to be used regardless of incoming column names
    dataframe = raw_dataframe.rename(
        columns={
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }
    ).copy()

    # normalize timestamp to tz-aware UTC
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], utc=True)

    # floor to timeframe start
    # pandas accepts "5min" or a string like "5m" with dt.floor if mapped to offset alias
    # simple mapping for minutes timeframes:
    try:
        minutes = int(timeframe[:-1])
    except Exception as exc:
        raise ValueError(f"Unsupported timeframe format: {timeframe}") from exc

    pandas_offset = f"{minutes}T"  # e.g., "5T"
    dataframe["timestamp"] = dataframe["timestamp"].dt.floor(pandas_offset)

    return dataframe[["timestamp", "open", "high", "low", "close", "volume"]]


def drop_last_if_unclosed(ohlcv_dataframe: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Drop last bar if its close time hasn't arrived yet."""
    if ohlcv_dataframe.empty:
        return ohlcv_dataframe

    if not isinstance(ohlcv_dataframe["timestamp"].dtype, pd.DatetimeTZDtype):
        ohlcv_dataframe = ohlcv_dataframe.copy()
        ohlcv_dataframe["timestamp"] = pd.to_datetime(ohlcv_dataframe["timestamp"], utc=True)

    try:
        minutes = int(timeframe[:-1])
    except Exception:
        raise ValueError(f"Unsupported timeframe format: {timeframe}")

    last_bar_start = ohlcv_dataframe["timestamp"].iloc[-1]
    now_utc = pd.Timestamp.now(tz="UTC")
    if now_utc < last_bar_start + pd.Timedelta(minutes=minutes):
        return ohlcv_dataframe.iloc[:-1]
    return ohlcv_dataframe


class LiveInferencePipeline:
    """
    Keeps a rolling store of recent OHLCV for the main timeframe and context timeframes,
    and provides model/scaler/metadata needed by the UI to run predictions.
    """

    def __init__(self, model_path: str, scaler_path: str, metadata_path: str):
        # Load model artifacts
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.feature_columns = self.metadata["features"]
        # metadata["regime_map"] example: {"Uptrend": 0, "Downtrend": 1, ...}
        self.regime_map = {int(class_id): name for name, class_id in self.metadata["regime_map"].items()}

        # In-memory OHLCV store: {timeframe: DataFrame[timestamp, open, high, low, close, volume]}
        self.data_store: dict[str, pd.DataFrame] = {}

        # Pre-fill with historical data
        self._prefill_data(days_back=200)

    def _prefill_data(self, days_back: int = 200) -> None:
        start_datetime = datetime.utcnow() - timedelta(days=days_back)
        start_ms = int(start_datetime.timestamp() * 1000)

        for timeframe in [MAIN_TIMEFRAME] + CONTEXT_TIMEFRAMES:
            raw = fetch_binance_klines(SYMBOL, timeframe, start_time_ms=start_ms)
            if raw is None or raw.empty:
                self.data_store[timeframe] = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
            else:
                standardized = standardize_ohlcv(raw, timeframe)
                self.data_store[timeframe] = standardized

    def _append_and_dedupe(self, existing: pd.DataFrame, new_raw: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        if new_raw is None or new_raw.empty:
            return existing

        new_standardized = standardize_ohlcv(new_raw, timeframe)
        combined = pd.concat([existing, new_standardized], ignore_index=True)

        # De-duplicate on timestamp and keep the latest row
        combined = combined.drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
        return combined

    def refresh_data(self, fetch_open_candles: bool = False) -> None:
        """
        Refresh the in-memory OHLCV store by fetching the latest data.
        - If fetch_open_candles=True, also include the most-recent open candle (limit=1).
        - Else, fetch from just after the last known bar's source timestamp.
        """
        try:
            for timeframe in [MAIN_TIMEFRAME] + CONTEXT_TIMEFRAMES:
                if timeframe not in self.data_store or self.data_store[timeframe].empty:
                    # fall back to a small backfill
                    raw = fetch_binance_klines(SYMBOL, timeframe, limit=500)
                    self.data_store[timeframe] = standardize_ohlcv(raw, timeframe) if raw is not None else pd.DataFrame(
                        columns=["timestamp", "open", "high", "low", "close", "volume"]
                    )
                    continue

                if fetch_open_candles:
                    latest_raw = fetch_binance_klines(SYMBOL, timeframe, limit=1)
                else:
                    # Fetch strictly after last known source timestamp (we rely on API's own time column)
                    last_known_source_time = self.data_store[timeframe]["timestamp"].iloc[-1]
                    # The upstream fetch function expects milliseconds since epoch (UTC)
                    start_ms = int(pd.Timestamp(last_known_source_time, tz="UTC").timestamp() * 1000) + 1
                    latest_raw = fetch_binance_klines(SYMBOL, timeframe, start_time_ms=start_ms)

                self.data_store[timeframe] = self._append_and_dedupe(self.data_store[timeframe], latest_raw, timeframe)

            # Trim memory to recent window (200 days)
            cutoff = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(days=200)
            for timeframe in [MAIN_TIMEFRAME] + CONTEXT_TIMEFRAMES:
                frame = self.data_store[timeframe]
                if not frame.empty:
                    self.data_store[timeframe] = frame[frame["timestamp"] >= cutoff].reset_index(drop=True)

        except Exception as exc:
            print("refresh_data error:", exc)


if __name__ == "__main__":
    # Optional CLI loop to keep the store hot (no prediction here)
    pipeline = LiveInferencePipeline(
        model_path=os.path.join(MODEL_FOLDER, "lstm_regime_model.keras"),
        scaler_path=os.path.join(MODEL_FOLDER, "scaler.joblib"),
        metadata_path=os.path.join(MODEL_FOLDER, "lstm_model_metadata.json"),
    )

    # refresh every 5 minutes
    pipeline.refresh_data(fetch_open_candles=False)
    schedule.every(5).minutes.do(lambda: pipeline.refresh_data(fetch_open_candles=False))

    while True:
        schedule.run_pending()
        time.sleep(1)
