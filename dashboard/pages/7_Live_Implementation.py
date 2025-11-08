# live_implementation.py

import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------------------
# imports and setup
# -------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.live_inference import (
    MODEL_FOLDER,
    SYMBOL,
    MAIN_TIMEFRAME,
    CONTEXT_TIMEFRAMES,
    TIME_STEPS,
    LiveInferencePipeline,
)
from src.data_cleaner import merge_timeframes, to_pandas_freq
from src.compute_features import build_features

st.set_page_config(page_title="Live Regime Classification", layout="wide")

# -------------------------------------------------------------------
# sidebar / controls
# -------------------------------------------------------------------
st.sidebar.title("Controls")
model_folder = st.sidebar.text_input("Model folder", MODEL_FOLDER)
symbol = st.sidebar.text_input("Symbol", SYMBOL)
sequence_length = st.sidebar.number_input(
    "Sequence length", min_value=16, max_value=512, value=TIME_STEPS, step=1
)

auto_refresh_seconds = st.sidebar.slider("Auto refresh (seconds)", min_value=5, max_value=60, value=10)
live_toggle = st.sidebar.checkbox("Live (auto refresh)", value=False)
run_once_button = st.sidebar.button("Run once (closed candle only)")
clear_audit_button = st.sidebar.button("Clear audit log")

# -------------------------------------------------------------------
# cached model loader
# -------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline(folder_path: str) -> LiveInferencePipeline:
    model_file = folder_path.rstrip("/") + "/lstm_regime_model.keras"
    scaler_file = folder_path.rstrip("/") + "/scaler.joblib"
    metadata_file = folder_path.rstrip("/") + "/lstm_model_metadata.json"
    return LiveInferencePipeline(model_file, scaler_file, metadata_file)

try:
    pipeline = load_pipeline(model_folder)
except Exception as exc:
    st.error(f"Failed to load model: {exc}")
    st.stop()

# -------------------------------------------------------------------
# audit / session state
# -------------------------------------------------------------------
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

if clear_audit_button:
    st.session_state.audit_log = []

# -------------------------------------------------------------------
# utilities (full, explicit naming)
# -------------------------------------------------------------------
def floor_and_standardize_ohlcv(raw_dataframe: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Convert raw kline frame (possibly using short column names) into:
    timestamp/open/high/low/close/volume with tz-aware UTC and floored timestamp.
    """
    pandas_freq = to_pandas_freq(timeframe)  # e.g. "5m" -> "5T"
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

    # ensure tz-aware UTC and floor to timeframe period start
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], utc=True).dt.floor(pandas_freq)
    dataframe = dataframe[["timestamp", "open", "high", "low", "close", "volume"]]
    return dataframe


def drop_last_if_candle_unclosed(ohlcv_dataframe: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Drop last bar if its close time hasn't arrived yet, based on timeframe (e.g., '5m', '1m', '15m').
    All timestamps are expected tz-aware UTC.
    """
    if ohlcv_dataframe.empty:
        return ohlcv_dataframe

    if not isinstance(ohlcv_dataframe["timestamp"].dtype, pd.DatetimeTZDtype):
        # normalize to tz-aware UTC if somehow naive crept in
        ohlcv_dataframe = ohlcv_dataframe.copy()
        ohlcv_dataframe["timestamp"] = pd.to_datetime(ohlcv_dataframe["timestamp"], utc=True)

    try:
        minutes = int(timeframe[:-1])
    except Exception:
        raise ValueError(f"Unsupported timeframe format: {timeframe}")

    last_bar_start = ohlcv_dataframe["timestamp"].iloc[-1]
    now_utc = pd.Timestamp.now(tz="UTC")
    close_due = last_bar_start + pd.Timedelta(minutes=minutes)
    return ohlcv_dataframe.iloc[:-1] if now_utc < close_due else ohlcv_dataframe


def latest_timestamp_from_features(features: pd.DataFrame) -> pd.Timestamp:
    """
    Robustly fetch the latest timestamp from features:
    - Prefer 'timestamp' column if present
    - Else fallback to index (if DatetimeIndex)
    """
    if "timestamp" in features.columns:
        return pd.to_datetime(features["timestamp"].iloc[-1], utc=True)
    if isinstance(features.index, pd.DatetimeIndex):
        return pd.to_datetime(features.index[-1], utc=True)
    # final fallback to row name if it is datetime-like
    return pd.to_datetime(features.iloc[-1].name, utc=True)


# -------------------------------------------------------------------
# prediction runner
# -------------------------------------------------------------------
def run_prediction(fetch_open_candles: bool = False):
    """
    Refresh pipeline data, merge across timeframes, build features,
    and run a single prediction with the cached model & scaler.
    """
    pipeline.refresh_data(fetch_open_candles=fetch_open_candles)

    main_frame = drop_last_if_candle_unclosed(
        floor_and_standardize_ohlcv(pipeline.data_store[MAIN_TIMEFRAME], MAIN_TIMEFRAME),
        MAIN_TIMEFRAME,
    )

    context_frames_map = {
        timeframe: drop_last_if_candle_unclosed(
            floor_and_standardize_ohlcv(pipeline.data_store[timeframe], timeframe),
            timeframe,
        )
        for timeframe in CONTEXT_TIMEFRAMES
    }

    merged = merge_timeframes(
        symbol=symbol,
        main_tf=MAIN_TIMEFRAME,
        context_tfs=CONTEXT_TIMEFRAMES,
        klines_map={MAIN_TIMEFRAME: main_frame, **context_frames_map},
    )

    features = build_features(
        merged,
        main_tf=MAIN_TIMEFRAME,
        context_tfs=CONTEXT_TIMEFRAMES,
    )

    # ensure we have a 'timestamp' column for logging
    if "timestamp" not in features.columns:
        if isinstance(features.index, pd.DatetimeIndex) or "index" in features.columns:
            features = features.reset_index().rename(columns={"index": "timestamp"})
        else:
            if "timestamp" in merged.columns:
                features["timestamp"] = merged["timestamp"]
            else:
                raise RuntimeError("No timestamp available on features or merged data.")

    # validate sequence window
    if len(features) < sequence_length or features.tail(sequence_length).isna().any().any():
        return None, "Not enough valid feature rows yet"

    model_input_window = features[pipeline.feature_columns].tail(sequence_length)
    model_input_scaled = pipeline.scaler.transform(model_input_window)
    model_input_batched = np.expand_dims(model_input_scaled, axis=0)

    probabilities = pipeline.model.predict(model_input_batched, verbose=0)[0]
    class_index = int(np.argmax(probabilities))
    predicted_regime = pipeline.regime_map.get(class_index, "Unknown")

    last_ts = latest_timestamp_from_features(features)

    return {
        "timestamp": last_ts,
        "regime": predicted_regime,
        "confidence": float(probabilities[class_index]),
        "probs": {
            regime_name: float(probabilities[int(class_id)])
            for regime_name, class_id in pipeline.metadata["regime_map"].items()
        },
    }, None


# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------
st.title("Live Regime Classification")

if run_once_button:
    result, message = run_prediction(fetch_open_candles=False)
    if result:
        st.session_state.audit_log.append(result)
    else:
        st.warning(message)

from streamlit_autorefresh import st_autorefresh

if live_toggle:
    st_autorefresh(interval=auto_refresh_seconds * 1000, key="refresh_live")
    result, message = run_prediction(fetch_open_candles=True)
    if result:
        st.session_state.audit_log.append(result)
    elif message:
        st.warning(message)

# display last prediction
if st.session_state.audit_log:
    latest = st.session_state.audit_log[-1]
    st.subheader(f"Current Regime: {latest['regime']} — {latest['confidence']:.2%}")

st.dataframe(pd.DataFrame(st.session_state.audit_log))
