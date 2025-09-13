# dashboard/pages/7_LSTM_Tuning.py
import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt

# Add src path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.lstm_model import create_sequences
from src.lstm_tuner import build_tuned_model

DATA_FOLDER = 'data/'
TUNER_FOLDER = 'tuner_logs/'

st.set_page_config(page_title="LSTM Hyperparameter Tuning", layout="wide")
st.title("Automated LSTM Hyperparameter Tuning")
st.markdown("""
This page uses **KerasTuner** to automatically find the best hyperparameters for the LSTM model.
It will test many different combinations of model architecture and training settings to maximize validation accuracy.
""")

# --- File Selection ---
try:
    labeled_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith("_labeled.csv")]
except FileNotFoundError:
    st.error(f"Data folder '{DATA_FOLDER}' not found.")
    labeled_files = []

if not labeled_files:
    st.warning("No HMM-labeled files found. Please run 'HMM Labeling' first.")
else:
    selected_file = st.selectbox("Select HMM-labeled dataset", labeled_files)
    df = pd.read_csv(os.path.join(DATA_FOLDER, selected_file))
    
    # --- Data Preparation ---
    regime_map = {label: i for i, label in enumerate(df['regime'].unique())}
    num_classes = len(regime_map)
    df['target'] = df['regime'].map(regime_map)
    df['target'] = df['target'].shift(-1)
    
    feature_cols = [col for col in df.columns if not (
        col.startswith(('open_', 'high_', 'low_', 'close_', 'volume_')) or
        col in ['timestamp', 'state', 'regime', 'target']
    )]
    
    df.dropna(inplace=True)
    X_raw = df[feature_cols]
    y_raw = df['target'].astype(int)
    
    # Train/Test Split
    test_size = 0.2
    train_n = int(len(df) * (1 - test_size))
    X_train_raw, X_test_raw = X_raw.iloc[:train_n], X_raw.iloc[train_n:]
    y_train_raw, y_test_raw = y_raw.iloc[:train_n], y_raw.iloc[train_n:]
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    
    # --- Tuner Configuration ---
    st.sidebar.header("Tuning Configuration")
    time_steps = st.sidebar.slider("LSTM Lookback Window (Time Steps)", 16, 128, 64, 4)
    
    # Create sequences based on selected lookback
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_raw.values, time_steps)
    y_train_cat = to_categorical(y_train_seq, num_classes=num_classes)

    st.sidebar.subheader("Hyperband Tuner Settings")
    max_epochs = st.sidebar.number_input("Max Epochs per Trial", 10, 100, 20)
    factor = st.sidebar.number_input("Reduction Factor (factor)", 2, 5, 3)
    
    if st.sidebar.button("Run Hyperparameter Search", type="primary"):
        # The tuner needs access to the input shape and num_classes
        # We use a lambda to pass them to our model builder function
        model_builder_with_args = lambda hp: build_tuned_model(hp, 
                                                               input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
                                                               num_classes=num_classes)
        
        tuner = kt.Hyperband(
            model_builder_with_args,
            objective='val_accuracy',
            max_epochs=max_epochs,
            factor=factor,
            directory=TUNER_FOLDER,
            project_name=f'lstm_tuning_{selected_file.replace(".csv", "")}'
        )
        
        stop_early = EarlyStopping(monitor='val_loss', patience=5)
        
        with st.spinner(f"Running KerasTuner search... This will take a while."):
            tuner.search(
                X_train_seq, y_train_cat,
                epochs=max_epochs,
                validation_split=0.2,
                callbacks=[stop_early]
            )
            st.session_state['lstm_tuner'] = tuner

    # --- Display Results ---
    if 'lstm_tuner' in st.session_state:
        st.success("Hyperparameter search complete!")
        tuner = st.session_state['lstm_tuner']
        
        st.subheader("Optimal Hyperparameters Found")
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        
        st.metric("Optimal LSTM Units", best_hps.get('lstm_units'))
        st.metric("Optimal Dense Units", best_hps.get('dense_units'))
        st.metric("Optimal Dropout Rate", f"{best_hps.get('dropout'):.2f}")
        st.metric("Optimal Learning Rate", f"{best_hps.get('learning_rate'):.4f}")
        
        st.subheader("Tuning Results Summary")
        # Get a dataframe of the results
        results_df = pd.DataFrame(tuner.oracle.get_best_trials(num_trials=10))
        st.dataframe(results_df)

        st.info("You can now use these optimal parameters to train your final model on the `5_LSTM_Training.py` page.")
