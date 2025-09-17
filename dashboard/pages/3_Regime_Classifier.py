# dashboard/pages/3_HMM_Labeling_simple_nographs.py
import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import joblib

# add src path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.regime_label import get_hmm_features, train_hmm, map_states_to_regimes

DATA_FOLDER = "data/"
MODELS_FOLDER = "models/"
os.makedirs(MODELS_FOLDER, exist_ok=True)

st.set_page_config(page_title="HMM Labeling (no charts)", layout="wide")
st.title("HMM Regime Labelings")

# --- File selection ---
if not os.path.exists(DATA_FOLDER):
    st.error(f"Data folder '{DATA_FOLDER}' not found.")
    st.stop()

feature_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('_features.csv')]
if not feature_files:
    st.warning("No *_features.csv files found in data/. Run feature engineering first.")
    st.stop()

selected_file = st.selectbox("Feature file", feature_files)
df = pd.read_csv(os.path.join(DATA_FOLDER, selected_file), parse_dates=['timestamp'])
st.markdown(f"Rows loaded: **{len(df):,}**")

# --- Feature & param selection ---
default_features = ['log_ret_1_5m','atr_norm_5m','adx_5m','bb_width_5m','volume_zscore_50_5m']
available_features = [c for c in df.columns if c not in ['timestamp'] and not c.startswith(('open_','high_','low_','close_','volume_'))]
hmm_features = st.multiselect("Select features for HMM", available_features, default=[f for f in default_features if f in available_features])

n_states = st.slider("Number of HMM states", min_value=2, max_value=8, value=4)
max_pca = max(1, len(hmm_features))
n_pca = st.slider("PCA components (if used)", min_value=1, max_value=max_pca, value=min(4, max_pca))

# --- Train action ---
if st.button("Train HMM and label"):
    if not hmm_features:
        st.error("Select at least one feature.")
        st.stop()

    with st.spinner("Preparing features and training HMM..."):
        X, scaler, pca_model = get_hmm_features(df, feature_list=hmm_features, n_components=n_pca)
        hmm_model = train_hmm(X, n_states=n_states)

        # align predictions to source rows
        valid_idx = df[hmm_features].dropna().index
        labeled_df = df.loc[valid_idx].copy()
        states = hmm_model.predict(X)
        labeled_df['state'] = states
        state_mapping = map_states_to_regimes(labeled_df, states, main_tf='5m')
        labeled_df['regime'] = labeled_df['state'].map(state_mapping)

        # store in session for use
        st.session_state['labeled_df'] = labeled_df
        st.session_state['hmm_model'] = hmm_model
        st.session_state['scaler'] = scaler
        st.session_state['pca'] = pca_model
        st.session_state['hmm_features'] = hmm_features
        st.session_state['state_mapping'] = state_mapping

    st.success("HMM trained and data labeled. See metrics below.")

# --- If labeled, show metrics and tables (no charts) ---
if 'labeled_df' in st.session_state:
    labeled_df = st.session_state['labeled_df']
    hmm_model = st.session_state['hmm_model']
    state_mapping = st.session_state['state_mapping']
    X_used = st.session_state.get('hmm_input_features', None)

    st.header("Model & Label Summary")

    # compute log-likelihood, AIC, BIC
    try:
        # we compute likelihood on the features used for training
        X_for_score = st.session_state.get('hmm_input_features')
        if X_for_score is None:
            # recompute features if not stored
            X_for_score, _, _ = get_hmm_features(df, feature_list=st.session_state['hmm_features'], n_components=n_pca)
    except Exception:
        X_for_score = None

    try:
        ll = float(hmm_model.score(X_for_score)) if X_for_score is not None else float('nan')
    except Exception:
        ll = float('nan')

    n_obs = int(X_for_score.shape[0]) if X_for_score is not None else len(labeled_df)
    n_components = int(hmm_model.n_components)
    n_features = int(X_for_score.shape[1]) if X_for_score is not None else len(hmm_features)
    k_params = (n_components * (n_components - 1)) + (n_components - 1) + 2 * n_components * n_features

    aic = (-2.0 * ll + 2 * k_params) if not np.isnan(ll) else None
    bic = (-2.0 * ll + k_params * np.log(n_obs)) if not np.isnan(ll) else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Log-Likelihood", f"{ll:.2f}" if not np.isnan(ll) else "N/A")
    c2.metric("AIC", f"{aic:.2f}" if aic is not None else "N/A")
    c3.metric("BIC", f"{bic:.2f}" if bic is not None else "N/A")
    c4.metric("Model params (k)", f"{k_params}")

    st.markdown(f"- Observations used: **{n_obs}**  \n- HMM states: **{n_components}**  \n- Feature dims (PCA): **{n_features}**")

    # state counts table
    st.subheader("State counts and percentages")
    state_counts = labeled_df['state'].value_counts().sort_index()
    state_pct = (state_counts / state_counts.sum() * 100).round(2)
    freq_df = pd.DataFrame({
        'state': state_counts.index,
        'count': state_counts.values,
        'percent': state_pct.values,
        'regime_label': [state_mapping.get(s, f"State_{s}") for s in state_counts.index]
    }).reset_index(drop=True)
    st.dataframe(freq_df)

    # transition matrix as table
    st.subheader("Transition matrix (rows=from, cols=to)")
    trans = np.round(hmm_model.transmat_, 4)
    trans_df = pd.DataFrame(trans, columns=[f"to_{i}" for i in range(trans.shape[1])], index=[f"from_{i}" for i in range(trans.shape[0])])
    st.dataframe(trans_df)

    # per-state feature summary table (means & std)
    st.subheader("Per-state feature summary (means & std)")
    candidate_features = [f for f in ['log_ret_1_5m', 'atr_norm_5m', 'adx_5m', 'bb_width_5m', 'rsi_5m', 'volume_zscore_50_5m', 'macd_hist_5m'] if f in labeled_df.columns]
    if candidate_features:
        stats = labeled_df.groupby('state')[candidate_features].agg(['mean', 'std']).round(6)
        stats.columns = ["_".join(col).strip() for col in stats.columns.values]
        stats = stats.reset_index()
        stats['regime_label'] = stats['state'].map(lambda s: state_mapping.get(s, f"State_{s}"))
        st.dataframe(stats)
    else:
        st.write("No common interpretability features present for summary.")

    # Save labeled data
    st.subheader("Save outputs")
    save_name = selected_file.replace('_features.csv', '_labeled.csv')
    save_path = os.path.join(DATA_FOLDER, save_name)
    if st.button(f"Save labeled CSV as `{save_name}`"):
        labeled_df.to_csv(save_path, index=False)
        st.success(f"Labeled data saved to `{save_path}`")

    # Save model artifacts
    if st.button("Save model artifacts (hmm + scaler + pca)"):
        artifact = {
            'hmm_model': hmm_model,
            'scaler': st.session_state.get('scaler', None),
            'pca': st.session_state.get('pca', None),
            'features': st.session_state.get('hmm_features', []),
            'state_mapping': st.session_state.get('state_mapping', {})
        }
        model_name = f"hmm_{os.path.splitext(selected_file)[0]}_states{n_components}.joblib"
        model_path = os.path.join(MODELS_FOLDER, model_name)
        joblib.dump(artifact, model_path)
        st.success(f"Saved model artifacts to {model_path}")
