# dashboard/pages/6_HMM_Tuning.py
import streamlit as st
import sys
import os
import pandas as pd

# Add src path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.hmm_tuner import find_best_hmm

DATA_FOLDER = 'data/'

st.set_page_config(page_title="HMM Hyperparameter Tuning", layout="wide")
st.title("Automated HMM Hyperparameter Tuning")
st.markdown("""
This page performs a grid search to find the optimal hyperparameters for the HMM labeling process. 
It tests different combinations of **HMM states** and **PCA components**, scoring each with the **Bayesian Information Criterion (BIC)**.

**The best model is the one with the lowest BIC score**, as it represents the best balance between model fit and complexity.
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
    
    # --- Tuning Configuration ---
    st.sidebar.header("Tuning Configuration")
    
    # Select features for the HMM
    default_hmm_features = [
        'log_ret_1_15m', 'bb_width_15m', 'atr_norm_15m', 'adx_15m',
        'volume_zscore_50_15m', 'macd_hist_15m', 'rsi_15m'
    ]
    all_feature_cols = [col for col in df.columns if col not in ['timestamp'] and not col.startswith(('open_', 'high_', 'low_', 'close_', 'volume_'))]
    filtered_default_features = [f for f in default_hmm_features if f in all_feature_cols]
    hmm_features = st.sidebar.multiselect(
        "Select features for HMM", 
        options=all_feature_cols, 
        default=filtered_default_features
    )
    
    # Define the search grid
    st.sidebar.subheader("Hyperparameter Search Grid")
    n_states_range = st.sidebar.select_slider(
        "Range of HMM States to test",
        options=list(range(2, 8)),
        value=(3, 5)
    )
    
    max_pca_val = len(hmm_features) if hmm_features else 1
    n_pca_range = st.sidebar.select_slider(
        "Range of PCA Components to test",
        options=list(range(2, max_pca_val + 1)),
        value=(min(4, max_pca_val), min(8, max_pca_val))
    )

    if st.sidebar.button("Run Hyperparameter Search", type="primary"):
        if not hmm_features:
            st.sidebar.error("Please select at least one feature for the HMM.")
        else:
            param_grid = {
                'n_states': list(range(n_states_range[0], n_states_range[1] + 1)),
                'n_pca_components': list(range(n_pca_range[0], n_pca_range[1] + 1))
            }
            
            with st.spinner(f"Running grid search for {len(param_grid['n_states']) * len(param_grid['n_pca_components'])} combinations..."):
                results_df = find_best_hmm(df, hmm_features, param_grid)
                st.session_state['hmm_tune_results'] = results_df

    # --- Display Results ---
    if 'hmm_tune_results' in st.session_state:
        st.success("Hyperparameter search complete!")
        results = st.session_state['hmm_tune_results']
        
        best_params = results.iloc[0]
        
        st.subheader("Optimal Parameters Found")
        st.metric(label="Best Number of HMM States", value=int(best_params['n_states']))
        st.metric(label="Best Number of PCA Components", value=int(best_params['n_pca_components']))
        st.metric(label="Lowest BIC Score", value=f"{best_params['bic_score']:.2f}")

        st.subheader("Full Grid Search Results")
        st.dataframe(results.style.highlight_min(subset=['bic_score'], color='lightgreen'))

