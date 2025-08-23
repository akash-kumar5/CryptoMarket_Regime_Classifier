# src/hmm_labeler.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from hmmlearn import hmm
import joblib
import os

def get_hmm_features(df, feature_list, n_components=10, scale=True, use_pca=True):
    """
    Prepares the feature set for the HMM by selecting, scaling, and applying PCA.
    """
    features = df[feature_list].copy()
    features.dropna(inplace=True)
    
    scaler = None
    if scale:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
    else:
        scaled_features = features.values

    pca_model = None
    if use_pca:
        pca_model = PCA(n_components=n_components)
        prepared_features = pca_model.fit_transform(scaled_features)
        print(f"PCA explained variance ratio: {np.sum(pca_model.explained_variance_ratio_):.4f}")
    else:
        prepared_features = scaled_features
        
    return prepared_features, scaler, pca_model

def train_hmm(features, n_states=3, n_iter=100, random_state=42):
    """
    Trains a Gaussian Hidden Markov Model.
    """
    model = hmm.GaussianHMM(n_components=n_states, 
                            covariance_type="diag", 
                            n_iter=n_iter, 
                            random_state=random_state)
    model.fit(features)
    return model

def map_states_to_regimes(df, labels, main_tf='15m'):
    """
    Interprets the HMM states by analyzing their characteristics.
    Handles 3, 4, or 6 states explicitly, falls back to generic otherwise.
    """
    df_labeled = df.copy()
    df_labeled['state'] = labels
    n_states = len(np.unique(labels))
    
    vol_col = f'atr_norm_{main_tf}'
    trend_col = f'adx_{main_tf}'
    
    # Compute average volatility & trend per state
    state_stats = df_labeled.groupby('state')[[vol_col, trend_col]].mean()
    state_stats = state_stats.sort_values(by=vol_col)  # sort by volatility
    
    mapping = {}

    if n_states == 3:
        # --- Logic for 3 states (Squeeze, Range, Trend) ---
        mapping[state_stats.index[0]] = 'Squeeze'
        if state_stats.loc[state_stats.index[1], trend_col] > state_stats.loc[state_stats.index[2], trend_col]:
            mapping[state_stats.index[1]] = 'Range'
            mapping[state_stats.index[2]] = 'Trend'
        else:
            mapping[state_stats.index[1]] = 'Trend'
            mapping[state_stats.index[2]] = 'Range'

    elif n_states == 4:
        # --- Logic for 4 states (Squeeze, Range, Mid-Vol Trend, High-Vol Trend) ---
        mapping[state_stats.index[0]] = 'Squeeze'
        mapping[state_stats.index[3]] = 'High-Vol Trend'
        if state_stats.loc[state_stats.index[1], trend_col] > state_stats.loc[state_stats.index[2], trend_col]:
            mapping[state_stats.index[1]] = 'Range'
            mapping[state_stats.index[2]] = 'Mid-Vol Trend'
        else:
            mapping[state_stats.index[1]] = 'Mid-Vol Trend'
            mapping[state_stats.index[2]] = 'Range'

    elif n_states == 6:
        # --- Logic for 6 states ---
        # Sorted by volatility: [lowest → highest]
        mapping[state_stats.index[0]] = 'Squeeze'
        mapping[state_stats.index[5]] = 'Volatility Spike'
        
        # Two low/mid-vol states (likely ranges vs weak trends)
        low_mid_states = [state_stats.index[1], state_stats.index[2]]
        if state_stats.loc[low_mid_states[0], trend_col] > state_stats.loc[low_mid_states[1], trend_col]:
            mapping[low_mid_states[0]] = 'Weak Trend'
            mapping[low_mid_states[1]] = 'Range'
        else:
            mapping[low_mid_states[0]] = 'Range'
            mapping[low_mid_states[1]] = 'Weak Trend'
        
        # Two higher-vol states (likely strong trends)
        high_mid_states = [state_stats.index[3], state_stats.index[4]]
        if state_stats.loc[high_mid_states[0], trend_col] > state_stats.loc[high_mid_states[1], trend_col]:
            mapping[high_mid_states[0]] = 'Strong Trend'
            mapping[high_mid_states[1]] = 'Choppy High-Vol'
        else:
            mapping[high_mid_states[0]] = 'Choppy High-Vol'
            mapping[high_mid_states[1]] = 'Strong Trend'

    else:
        # Fallback for other numbers of states
        for i, state_index in enumerate(state_stats.index):
            mapping[state_index] = f"State_{i}"

    return mapping


