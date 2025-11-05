# src/regime_label.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from hmmlearn import hmm

def get_hmm_features(df, feature_list, n_components=None, scale=True, use_pca=True):
    """
    Selects features, drops NA rows, scales, and (optionally) applies PCA.
    Returns: X, scaler, pca_model
    - n_components: PCA components (int) if use_pca else ignored
    """
    features = df[feature_list].copy()
    features = features.dropna(axis=0, how='any')

    scaler = None
    X = features.values

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(features.values)

    pca_model = None
    if use_pca:
        if n_components is None:
            # keep full rank if not specified
            n_components = min(features.shape[0], features.shape[1])
        pca_model = PCA(n_components=n_components, random_state=42)
        X = pca_model.fit_transform(X)
        # Optionally log variance explained (caller can print if needed)
        # print(f"PCA explained variance ratio: {np.sum(pca_model.explained_variance_ratio_):.4f}")

    return X, scaler, pca_model

def train_hmm(features, n_states=3, n_iter=150, random_state=42):
    """
    Trains a diagonal-covariance Gaussian HMM.
    """
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=random_state,
        verbose=False
    )
    model.fit(features)
    return model

def map_states_to_regimes(df, labels, main_tf='5m'):
    """
    Map HMM states -> interpretable regimes using volatility (ATR-normalized) and trend (ADX).
    If required columns are missing, fall back to ordinal mapping by volatility proxy when possible.
    """
    df_labeled = df.copy()
    df_labeled['state'] = labels
    n_states = len(np.unique(labels))

    vol_col = f'atr_norm_{main_tf}'
    trend_col = f'adx_{main_tf}'

    # if missing columns, fall back to simple mapping
    if vol_col not in df_labeled.columns or trend_col not in df_labeled.columns:
        order = (
            df_labeled.groupby('state')['state'].count().sort_values(ascending=True).index.tolist()
            if vol_col not in df_labeled.columns else
            df_labeled.groupby('state')[vol_col].mean().sort_values().index.tolist()
        )
        return {s: f"State_{i}" for i, s in enumerate(order)}

    state_stats = df_labeled.groupby('state')[[vol_col, trend_col]].mean()
    state_stats = state_stats.sort_values(by=vol_col)  # low → high vol

    mapping = {}

    if n_states == 3:
        mapping[state_stats.index[0]] = 'Squeeze'
        # decide Range vs Trend using ADX
        mid, high = state_stats.index[1], state_stats.index[2]
        if state_stats.loc[mid, trend_col] >= state_stats.loc[high, trend_col]:
            mapping[mid] = 'Range'; mapping[high] = 'Trend'
        else:
            mapping[mid] = 'Trend'; mapping[high] = 'Range'

    elif n_states == 4:
        # lowest vol → Squeeze; highest vol → High-Vol Trend/Spike
        mapping[state_stats.index[0]] = 'Squeeze'
        mapping[state_stats.index[-1]] = 'High-Vol Trend'
        mid1, mid2 = state_stats.index[1], state_stats.index[2]
        if state_stats.loc[mid1, trend_col] >= state_stats.loc[mid2, trend_col]:
            mapping[mid1] = 'Range'; mapping[mid2] = 'Mid-Vol Trend'
        else:
            mapping[mid1] = 'Mid-Vol Trend'; mapping[mid2] = 'Range'

    elif n_states == 6:
        # 0: Squeeze, 5: Vol Spike, mids split by ADX
        mapping[state_stats.index[0]] = 'Squeeze'
        mapping[state_stats.index[-1]] = 'Volatility Spike'
        mids = state_stats.index[1:-1].tolist()
        # sort mids by ADX to separate trends vs range
        mids_sorted = state_stats.loc[mids].sort_values(by=trend_col).index.tolist()
        # low-mid (range-ish), mid (weak trend), high-mid (strong trend), high (choppy high vol)
        if len(mids_sorted) == 4:
            mapping[mids_sorted[0]] = 'Range'
            mapping[mids_sorted[1]] = 'Weak Trend'
            mapping[mids_sorted[2]] = 'Strong Trend'
            mapping[mids_sorted[3]] = 'Choppy High-Vol'
        else:
            # fallback
            for i, s in enumerate(mids_sorted):
                mapping[s] = f"Mid_{i}"

    else:
        # generic fallback
        for i, s in enumerate(state_stats.index):
            mapping[s] = f"State_{i}"

    return mapping
