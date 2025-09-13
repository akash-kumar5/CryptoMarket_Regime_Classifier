# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import time
import io
from datetime import datetime, timezone
import plotly.graph_objects as go
import os, sys

# Import the pipeline and constants from your main.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.live_inference import MODEL_FOLDER, SYMBOL, MAIN_TF, CONTEXT_TFS, TIME_STEPS, LiveInferencePipeline

st.set_page_config(page_title="Live Regime Classification", layout="wide")

# ---------------------------
# Sidebar / Controls
# ---------------------------
st.sidebar.title("Controls")
model_folder = st.sidebar.text_input("Model folder", MODEL_FOLDER)
symbol = st.sidebar.text_input("Symbol", SYMBOL)
main_tf = st.sidebar.selectbox("Main timeframe", [MAIN_TF], index=0, disabled=True)
context_tfs = st.sidebar.multiselect("Context timeframes", CONTEXT_TFS, default=CONTEXT_TFS)
seq_len = st.sidebar.number_input("Sequence length (TIME_STEPS)", min_value=16, max_value=512, value=TIME_STEPS, step=1)

st.sidebar.markdown("---")
refresh_seconds = st.sidebar.slider("Live refresh (seconds)", min_value=5, max_value=60, value=10)
live_toggle = st.sidebar.checkbox("Live (auto refresh)", value=False)
simulate_mode = st.sidebar.checkbox("Simulate (replay historical)", value=False)
run_once_btn = st.sidebar.button("Run once (fetch closed candles)")

st.sidebar.markdown("---")
st.sidebar.write("Quick actions")
clear_log_btn = st.sidebar.button("Clear audit log")

# ---------------------------
# Cached pipeline loader
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline(model_folder_path: str):
    model_file = model_folder_path.rstrip("/") + "/lstm_regime_model.keras"
    scaler_file = model_folder_path.rstrip("/") + "/scaler.joblib"
    meta_file = model_folder_path.rstrip("/") + "/lstm_model_metadata.json"
    pipeline = LiveInferencePipeline(model_file, scaler_file, meta_file)
    return pipeline

# instantiate pipeline
try:
    pipeline = load_pipeline(model_folder)
except Exception as e:
    st.error(f"Failed to load pipeline: {e}")
    st.stop()

# ---------------------------
# Session state for logs and simulate pointer
# ---------------------------
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []  # list of dicts
if "simulate_index" not in st.session_state:
    st.session_state.simulate_index = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if clear_log_btn:
    st.session_state.audit_log = []

# ---------------------------
# Helper: perform prediction and return structured info
# ---------------------------
from src.data_cleaner import merge_timeframes
from src.compute_features import build_features

def do_prediction(fetch_open_candle: bool = False):
    """
    Calls pipeline.run_prediction_cycle to update internal data_store,
    then builds features and makes a prediction (so we can capture results/DFs).
    Returns dict: {predicted_regime, confidence, probs, features_df, merged_df, timestamp}
    """
    # run to update internal data_store (prints too)
    pipeline.run_prediction_cycle(fetch_open_candle=fetch_open_candle)

    # build merged and features (same logic as main)
    main_df = pipeline.data_store[MAIN_TF]
    context_dfs = {tf: pipeline.data_store[tf] for tf in CONTEXT_TFS}
    merged_df = merge_timeframes(MAIN_TF, main_df, context_dfs)
    features_df = build_features(merged_df, main_tf=MAIN_TF, context_tfs=CONTEXT_TFS)

    result = {"ok": False}
    if len(features_df) < seq_len:
        result["message"] = f"Not enough data: have {len(features_df)}, need {seq_len}"
        return result

    final_features = features_df[pipeline.feature_cols]
    sequence_data = final_features.tail(seq_len)
    scaled_sequence = pipeline.scaler.transform(sequence_data)
    input_sequence = np.expand_dims(scaled_sequence, axis=0)

    prediction_probs = pipeline.model.predict(input_sequence, verbose=0)[0]
    predicted_class_index = int(np.argmax(prediction_probs))
    predicted_regime = pipeline.regime_map.get(predicted_class_index, "Unknown")
    confidence = float(prediction_probs[predicted_class_index])

    # friendly probabilities mapping
    probs_map = {}
    for regime_name, index in pipeline.metadata["regime_map"].items():
        probs_map[regime_name] = float(prediction_probs[int(index)])

    timestamp = features_df['t'].iloc[-1] if 't' in features_df.columns else datetime.now(timezone.utc)

    result.update({
        "ok": True,
        "predicted_regime": predicted_regime,
        "confidence": confidence,
        "probs": probs_map,
        "features_df": features_df,
        "merged_df": merged_df,
        "timestamp": timestamp
    })
    return result

# ---------------------------
# UI layout
# ---------------------------
st.title("Live Regime Classification — Demo")
top_col, right_col = st.columns([3,1])

with top_col:
    st.subheader(f"{symbol} — {MAIN_TF} | Sequence: {seq_len}")

with right_col:
    if st.button("Download audit log (CSV)"):
        if len(st.session_state.audit_log) == 0:
            st.warning("Audit log empty.")
        else:
            df_log = pd.DataFrame(st.session_state.audit_log)
            csv = df_log.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, file_name="audit_log.csv", mime="text/csv")

# ---------------------------
# Simulation mode: prefill pointer if needed
# ---------------------------
if simulate_mode:
    # If simulation mode, we use historical main_df and advance pointer on each refresh
    hist_main = pipeline.data_store[MAIN_TF].copy()
    if st.session_state.simulate_index is None:
        st.session_state.simulate_index = max(0, len(hist_main) - 500)  # start a bit in the past

# ---------------------------
# Action: run once button
# ---------------------------
if run_once_btn:
    res = do_prediction(fetch_open_candle=False)
    if not res.get("ok", False):
        st.warning(res.get("message", "Prediction failed"))
    else:
        # append to audit log
        log_entry = {
            "timestamp": str(res["timestamp"]),
            "regime": res["predicted_regime"],
            "confidence": res["confidence"],
            **{f"p_{k}": v for k, v in res["probs"].items()}
        }
        st.session_state.audit_log.append(log_entry)
        st.session_state.last_prediction = res

# ---------------------------
# Live auto-refresh logic
# ---------------------------
from streamlit_autorefresh import st_autorefresh

if live_toggle:
    # autorefresh every refresh_seconds
    count = st_autorefresh(interval=refresh_seconds * 1000, key="live_refresh")
    # run prediction on each refresh
    res = do_prediction(fetch_open_candle=True)
    if res.get("ok", False):
        log_entry = {
            "timestamp": str(res["timestamp"]),
            "regime": res["predicted_regime"],
            "confidence": res["confidence"],
            **{f"p_{k}": v for k, v in res["probs"].items()}
        }
        st.session_state.audit_log.append(log_entry)
        st.session_state.last_prediction = res
    else:
        st.warning(res.get("message", "Live prediction skipped"))

# ---------------------------
# Simulate mode step (advance pointer and run prediction from simulated data)
# ---------------------------
if simulate_mode and not live_toggle and not run_once_btn:
    # advance pointer
    st.session_state.simulate_index += 1
    hist_main = pipeline.data_store[MAIN_TF].copy()
    idx = min(st.session_state.simulate_index, len(hist_main)-1)
    # trim each timeframe to that index to mimic "current time"
    for tf in [MAIN_TF] + CONTEXT_TFS:
        df = pipeline.data_store[tf]
        # cut by index
