# dashboard/pages/3_HMM_Labeling.py
import streamlit as st
import sys
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.regime_label import get_hmm_features, train_hmm, map_states_to_regimes

DATA_FOLDER = 'data/'
MODELS_FOLDER = 'models/'

st.set_page_config(page_title="HMM Regime Labeling", layout="wide")
st.title("Unsupervised Regime Labeling with HMM")
st.markdown("""
This page uses a Hidden Markov Model (HMM) to identify underlying market regimes (e.g., Trend, Range, Squeeze) from the feature data.

**Process:**
1.  **Select Features:** Choose a small, interpretable set of features for the HMM.
2.  **Preprocess:** The features are standardized (z-scored) and dimensionality is reduced using PCA.
3.  **Train HMM:** An HMM is trained to find a specified number of hidden states.
4.  **Interpret & Label:** The states are automatically interpreted based on their volatility and trend characteristics, then mapped to human-readable labels.
""")

# --- File Selection ---
try:
    feature_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('_features.csv')]
except FileNotFoundError:
    st.error(f"Data folder '{DATA_FOLDER}' not found.")
    feature_files = []

if not feature_files:
    st.warning("No feature files found. Please run the 'Feature Engineering' step first.")
else:
    selected_file = st.selectbox("Select Feature Data File", feature_files)
    file_path = os.path.join(DATA_FOLDER, selected_file)
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    
    # --- HMM Configuration ---
    st.sidebar.header("HMM Configuration")
    
    # Feature selection for HMM
    default_hmm_features = [
        'log_ret_1_5m', 'bb_width_5m', 'atr_norm_5m', 'adx_5m',
        'volume_zscore_50_5m', 'macd_hist_5m', 'rsi_5m'
    ]
    all_feature_cols = [col for col in df.columns if col not in ['timestamp'] and not col.startswith(('open_', 'high_', 'low_', 'close_', 'volume_'))]
    
    filtered_default_features = [f for f in default_hmm_features if f in all_feature_cols]

    hmm_features = st.sidebar.multiselect(
        "Select features for HMM", 
        options=all_feature_cols, 
        default=filtered_default_features
    )
    
    n_states = st.sidebar.slider("Number of HMM States (Regimes)", 2, 6, 6)

    # --- FIX: Dynamically set the max value for the PCA components slider ---
    if hmm_features:
        # The number of components cannot exceed the number of selected features.
        max_pca_components = len(hmm_features)
        # Set a reasonable default value, capped by the max possible.
        default_pca_value = min(8, max_pca_components)
        
        n_pca_components = st.sidebar.slider(
            "Number of PCA Components", 
            min_value=1, 
            max_value=max_pca_components, 
            value=default_pca_value
        )
    else:
        # If no features are selected, disable the slider.
        st.sidebar.markdown("_Select features to configure PCA._")
        n_pca_components = 0

    
    if st.sidebar.button("Run HMM Training and Labeling", type="primary"):
        if not hmm_features:
            st.sidebar.error("Please select at least one feature for the HMM.")
        else:
            with st.spinner("Preparing features, training HMM, and labeling data..."):
                
                # 1. Prepare Features
                hmm_input_features, scaler, pca_model = get_hmm_features(
                    df, 
                    feature_list=hmm_features, 
                    n_components=n_pca_components
                )
                
                # 2. Train HMM
                hmm_model = train_hmm(hmm_input_features, n_states=n_states)
                
                # 3. Predict States and Align with Original DataFrame
                valid_indices = df[hmm_features].dropna().index
                df_valid = df.loc[valid_indices].copy()
                
                hidden_states = hmm_model.predict(hmm_input_features)
                df_valid['state'] = hidden_states
                
                # 4. Interpret States
                state_mapping = map_states_to_regimes(df_valid, hidden_states, main_tf='5m')
                df_valid['regime'] = df_valid['state'].map(state_mapping)
                
                st.session_state['labeled_df'] = df_valid
                st.session_state['state_mapping'] = state_mapping

    # --- Display Results ---
    if 'labeled_df' in st.session_state:
        st.success("HMM training and labeling complete!")
        labeled_df = st.session_state['labeled_df']
        state_mapping = st.session_state['state_mapping']
        
        st.subheader("Regime Interpretation")
        st.write("States were mapped to regimes based on their average volatility and trend strength:")
        st.json(state_mapping)

        st.subheader("Preview of Labeled Data")
        st.dataframe(labeled_df[['timestamp', 'close_5m', 'state', 'regime']].tail(15))
        

        # --- Save Data ---
        st.subheader("Save Labeled Data")
        save_name = selected_file.replace('_features.csv', '_labeled.csv')
        save_path = os.path.join(DATA_FOLDER, save_name)
        
        if st.button(f"Save as {save_name}"):
            labeled_df.to_csv(save_path, index=False)
            st.success(f"Labeled data saved to `{save_path}`")
