# dashboard/pages/3_HMM_Labeling_simple_nographs.py
import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import joblib

# add src path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.regime_label import get_hmm_features, train_hmm, map_states_to_regimes  # <- unify import

DATA_FOLDER = "data/"
MODELS_FOLDER = "models/"
os.makedirs(MODELS_FOLDER, exist_ok=True)

st.set_page_config(page_title="HMM Labeling", layout="wide")
st.title("HMM Regime Labeling")

# --- File selection ---
if not os.path.exists(DATA_FOLDER):
    st.error(f"Data folder '{DATA_FOLDER}' not found.")
    st.stop()

feature_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('_features.csv')]
if not feature_files:
    st.warning("No *_features.csv files found in data/. Run feature engineering first.")
    st.stop()

selected_file = st.selectbox("Feature file", sorted(feature_files))
df = pd.read_csv(os.path.join(DATA_FOLDER, selected_file), parse_dates=['timestamp'])
st.markdown(f"Rows loaded: **{len(df):,}**")

# --- Feature & param selection ---
# (rsi may be rsi_14_5m in your pipeline; we guard for missing)
default_features = ['atr_norm_5m', 'bb_width_5m', 'adx_5m', 'volume_zscore_50_5m', 'log_ret_1_5m']

# [atr_norm_5m, bb_width_5m, adx_5m, volume_zscore_50_5m, taker_buy_ratio_5m, vwap_skew_5m, trade_imbalance_5m]


available_features = [
    c for c in df.columns
    if c != 'timestamp' and not c.startswith(('open_', 'high_', 'low_', 'close_'))
]
hmm_features = st.multiselect(
    "Select features for HMM",
    options=sorted(available_features),
    default=[f for f in default_features if f in available_features]
)

n_states = st.slider("Number of HMM states", min_value=2, max_value=8, value=4, step=1)

c1, c2, c3 = st.columns(3)
with c1:
    use_scaler = st.checkbox("Scale (StandardScaler)", value=True)
with c2:
    use_pca = st.checkbox("Use PCA", value=True)
with c3:
    # n_pca cannot exceed #features
    max_pca = max(1, len(hmm_features))
    n_pca = st.slider("PCA components", min_value=1, max_value=max_pca, value=min(4, max_pca), disabled=not use_pca)

# --- Train action ---
if st.button("Train HMM and label"):
    if not hmm_features:
        st.error("Select at least one feature.")
        st.stop()

    # drop rows that have NA in selected features to keep alignment clean
    valid_mask = df[hmm_features].notna().all(axis=1)
    if not valid_mask.any():
        st.error("No rows with all selected features available (after NA filter).")
        st.stop()

    df_used = df.loc[valid_mask].copy()

    with st.spinner("Preparing features and training HMM..."):
        X, scaler, pca_model = get_hmm_features(
            df_used, feature_list=hmm_features,
            n_components=(n_pca if use_pca else None),
            scale=use_scaler, use_pca=use_pca
        )
        hmm_model = train_hmm(X, n_states=n_states, n_iter=200, random_state=42)

        # Predictions aligned with df_used
        states = hmm_model.predict(X)
        labeled_df = df_used.copy()
        labeled_df['state'] = states

        # Map states → regimes using consistent columns (5m by default)
        state_mapping = map_states_to_regimes(labeled_df, states, main_tf='5m')
        labeled_df['regime'] = labeled_df['state'].map(state_mapping)

        # Store in session
        st.session_state['labeled_df'] = labeled_df
        st.session_state['hmm_model'] = hmm_model
        st.session_state['scaler'] = scaler
        st.session_state['pca'] = pca_model
        st.session_state['hmm_features'] = hmm_features
        st.session_state['state_mapping'] = state_mapping
        st.session_state['hmm_input_features'] = X  # <- keep the training matrix

    st.success("HMM trained and data labeled. See metrics below.")

# --- If labeled, show metrics and tables (no charts) ---
if 'labeled_df' in st.session_state:
    labeled_df = st.session_state['labeled_df']
    hmm_model = st.session_state['hmm_model']
    state_mapping = st.session_state['state_mapping']
    hmm_features = st.session_state['hmm_features']
    X_for_score = st.session_state.get('hmm_input_features', None)

    st.header("Model & Label Summary")

    # log-likelihood, AIC, BIC (diag-cov GaussianHMM)
    try:
        ll = float(hmm_model.score(X_for_score)) if X_for_score is not None else float('nan')
    except Exception:
        ll = float('nan')

    n_obs = int(X_for_score.shape[0]) if X_for_score is not None else len(labeled_df)
    n_components = int(hmm_model.n_components)
    n_features = int(X_for_score.shape[1]) if X_for_score is not None else len(hmm_features)

    # parameter count (approx): startprob (n-1) + trans (n*(n-1)) + means (n*d) + diag covars (n*d)
    k_params = (n_components - 1) + (n_components * (n_components - 1)) + (n_components * n_features) + (n_components * n_features)

    aic = (None if np.isnan(ll) else (-2.0 * ll + 2 * k_params))
    bic = (None if np.isnan(ll) else (-2.0 * ll + k_params * np.log(max(n_obs, 1))))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Log-Likelihood", f"{ll:.2f}" if not np.isnan(ll) else "N/A")
    c2.metric("AIC", f"{aic:.2f}" if aic is not None else "N/A")
    c3.metric("BIC", f"{bic:.2f}" if bic is not None else "N/A")
    c4.metric("Model params (k)", f"{k_params}")

    st.markdown(f"- Observations used: **{n_obs}**  \n- HMM states: **{n_components}**  \n- Feature dims (post-PCA): **{n_features}**")

    # state counts
    st.subheader("State counts and percentages")
    state_counts = labeled_df['state'].value_counts().sort_index()
    state_pct = (state_counts / state_counts.sum() * 100).round(2)
    freq_df = pd.DataFrame({
        'state': state_counts.index,
        'count': state_counts.values,
        'percent': state_pct.values,
        'regime_label': [state_mapping.get(s, f"State_{s}") for s in state_counts.index]
    }).reset_index(drop=True)
    st.dataframe(freq_df, use_container_width=True)

    # transition matrix
    st.subheader("Transition matrix (rows=from, cols=to)")
    trans = np.round(hmm_model.transmat_, 4)
    trans_df = pd.DataFrame(trans, columns=[f"to_{i}" for i in range(trans.shape[1])], index=[f"from_{i}" for i in range(trans.shape[0])])
    st.dataframe(trans_df, use_container_width=True)

    # per-state feature summary (means & std) – use whatever exists in DF
    st.subheader("Per-state feature summary (means & std)")
    interpret_feats = [f for f in [
        'log_ret_1_5m','atr_norm_5m','adx_5m','bb_width_5m','rsi_14_5m','volume_zscore_50_5m','macd_hist_5m'
    ] if f in labeled_df.columns]
    if interpret_feats:
        stats = labeled_df.groupby('state')[interpret_feats].agg(['mean', 'std']).round(6)
        stats.columns = ["_".join(col).strip() for col in stats.columns.values]
        stats = stats.reset_index()
        stats['regime_label'] = stats['state'].map(lambda s: state_mapping.get(s, f"State_{s}"))
        st.dataframe(stats, use_container_width=True)
    else:
        st.write("No common interpretability features present for summary.")

    # Save labeled data
    st.subheader("Save outputs")
    save_name = selected_file.replace('_features.csv', f'_states{n_components}_labeled.csv')
    save_path = os.path.join(DATA_FOLDER, save_name)
    if st.button(f"Save labeled CSV as `{save_name}`"):
        labeled_df.to_csv(save_path, index=False)
        st.success(f"Labeled data saved to `{save_path}`")

    # Save model artifacts
    if st.button("Save model artifacts (HMM + scaler + pca)"):
        artifact = {
            'hmm_model': hmm_model,
            'scaler': st.session_state.get('scaler', None),
            'pca': st.session_state.get('pca', None),
            'features': st.session_state.get('hmm_features', []),
            'state_mapping': st.session_state.get('state_mapping', {}),
            'n_states': n_components,
            'file_source': selected_file,
        }
        model_name = f"hmm_{os.path.splitext(selected_file)[0]}_states{n_components}.joblib"
        model_path = os.path.join(MODELS_FOLDER, model_name)
        joblib.dump(artifact, model_path)
        st.success(f"Saved model artifacts to {model_path}")
