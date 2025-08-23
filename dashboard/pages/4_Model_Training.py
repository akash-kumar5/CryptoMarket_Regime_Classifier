# dashboard/pages/5_LSTM_Training.py
import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.lstm_model import create_sequences, build_lstm_model

DATA_FOLDER = "data/"
MODEL_FOLDER = "models/"

st.set_page_config(page_title="LSTM Model Training", layout="wide")
st.title("Train Regime Prediction Model (LSTM)")
st.markdown("""
This page trains an LSTM network to predict the HMM-generated market regime for the next bar.
""")

# ----- 1. Load Data -----
try:
    labeled_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith("_hmm_labels.csv")]
except FileNotFoundError:
    st.error(f"Data folder '{DATA_FOLDER}' not found.")
    labeled_files = []

if not labeled_files:
    st.warning("No HMM-labeled files found. Please run 'HMM Labeling' first.")
    st.stop()

selected_file = st.selectbox("Select HMM-labeled dataset", labeled_files)
df = pd.read_csv(os.path.join(DATA_FOLDER, selected_file))

# ----- 2. Target and Feature Preparation -----
unique_regimes = sorted(df['regime'].dropna().unique())
regime_map = {regime: idx for idx, regime in enumerate(unique_regimes)}
inv_regime_map = {v: k for k, v in regime_map.items()}
num_classes = len(regime_map)

df['target'] = df['regime'].map(regime_map)
df['target'] = df['target'].shift(-1) # Predict next bar's regime

feature_cols = [col for col in df.columns if not (
    col.startswith(('open_', 'high_', 'low_', 'close_', 'volume_')) or
    col in ['timestamp', 'state', 'regime', 'target']
)]

df.dropna(inplace=True)
X_raw = df[feature_cols]
y_raw = df['target'].astype(int)

# ----- 3. Train/Test Split -----
st.sidebar.header("Data & Model Configuration")
test_size = st.sidebar.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
n = len(df)
train_n = int(n * (1 - test_size))

X_train_raw, X_test_raw = X_raw.iloc[:train_n], X_raw.iloc[train_n:]
y_train_raw, y_test_raw = y_raw.iloc[:train_n], y_raw.iloc[train_n:]

# ----- 4. Scaling -----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# ----- 5. Create Sequences -----
time_steps = st.sidebar.slider("LSTM Lookback Window (Time Steps)", 16, 128, 64, 4)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_raw.values, time_steps)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_raw.values, time_steps)

# One-hot encode the labels
y_train_cat = to_categorical(y_train_seq, num_classes=num_classes)
y_test_cat = to_categorical(y_test_seq, num_classes=num_classes)

st.write(f"**Training sequences shape:** {X_train_seq.shape}")
st.write(f"**Test sequences shape:** {X_test_seq.shape}")

# ----- 6. Model Training -----
st.sidebar.header("LSTM Architecture")
lstm_units = st.sidebar.slider("LSTM Units", 32, 256, 96, 8)
dense_units = st.sidebar.slider("Dense Units", 16, 128, 64, 8)
dropout_rate = st.sidebar.slider("Dropout Rate", 0.1, 0.5, 0.3, 0.05)

st.sidebar.header("Training Parameters")
epochs = st.sidebar.number_input("Epochs", 10, 200, 50, 5)
batch_size = st.sidebar.select_slider("Batch Size", options=[32, 64, 128, 256], value=128)

if st.button("Train LSTM Model", type="primary"):
    model = build_lstm_model(
        input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
        num_classes=num_classes,
        lstm_units=lstm_units,
        dense_units=dense_units,
        dropout_rate=dropout_rate
    )
    st.write(model.summary())

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    ]

    with st.spinner("Training LSTM model..."):
        history = model.fit(
            X_train_seq, y_train_cat,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.15, # Use last 15% of training data for validation
            callbacks=callbacks,
            verbose=1
        )
        st.session_state['lstm_model'] = model
        st.session_state['lstm_history'] = history.history
        st.session_state['scaler'] = scaler

# ----- 7. Evaluation & Saving -----
if 'lstm_model' in st.session_state:
    st.success("Model training complete!")
    model = st.session_state['lstm_model']
    history = st.session_state['lstm_history']
    scaler = st.session_state['scaler']

    # Plot training history
    st.subheader("Training & Validation Loss")
    fig, ax = plt.subplots()
    ax.plot(history['loss'], label='Training Loss')
    ax.plot(history['val_loss'], label='Validation Loss')
    ax.legend()
    st.pyplot(fig)

    # Evaluate on test set
    y_pred_probs = model.predict(X_test_seq)
    y_pred = np.argmax(y_pred_probs, axis=1)

    unique_labels = sorted(list(np.unique(np.concatenate([y_test_seq, y_pred]))))
    target_names = [inv_regime_map.get(i, f"Unknown_{i}") for i in unique_labels]

    st.subheader("Classification Report (Test Set)")
    report = classification_report(y_test_seq, y_pred, labels=unique_labels, target_names=target_names, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Confusion Matrix (Test Set)")
    cm = confusion_matrix(y_test_seq, y_pred, labels=unique_labels)
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names, ax=ax2)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    st.pyplot(fig2)

    # --- Save Model, Scaler, and Metadata ---
    if st.button("Save LSTM Model and Scaler"):
        os.makedirs(MODEL_FOLDER, exist_ok=True)
        
        model_path = os.path.join(MODEL_FOLDER, "lstm_regime_model.keras")
        model.save(model_path)
        
        scaler_path = os.path.join(MODEL_FOLDER, "scaler.joblib")
        joblib.dump(scaler, scaler_path)
        
        meta = {
            "model_type": "LSTM",
            "target": "regime",
            "regime_map": regime_map,
            "features": feature_cols,
            "time_steps": time_steps,
            "training_file": selected_file,
        }
        meta_path = os.path.join(MODEL_FOLDER, "lstm_model_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)
            
        st.success(f"Model saved to: `{model_path}`")
        st.success(f"Scaler saved to: `{scaler_path}`")
        st.success(f"Metadata saved to: `{meta_path}`")
