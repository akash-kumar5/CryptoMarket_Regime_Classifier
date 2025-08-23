# src/hmm_tuner.py
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from hmmlearn import hmm

def calculate_bic(model, X):
    """
    Calculates the Bayesian Information Criterion (BIC) for a fitted HMM model.
    Lower BIC is better.
    
    Args:
        model (hmm.GaussianHMM): The fitted HMM model.
        X (np.array): The data the model was fitted on.

    Returns:
        float: The BIC score.
    """
    # Log-likelihood of the data given the model
    log_likelihood = model.score(X)
    
    # Number of data points
    n_samples = X.shape[0]
    
    # Number of free parameters in the model
    # n_features is the number of PCA components
    n_features = model.n_features
    n_states = model.n_components
    
    # Parameters for transition matrix (n_states * (n_states - 1))
    n_trans_params = n_states * (n_states - 1)
    # Parameters for initial probabilities (n_states - 1)
    n_start_prob_params = n_states - 1
    # Parameters for emission probabilities (means and variances)
    # n_states * n_features for means, n_states * n_features for diagonal covariances
    n_emission_params = 2 * n_states * n_features
    
    n_params = n_trans_params + n_start_prob_params + n_emission_params
    
    # BIC formula
    bic_score = -2 * log_likelihood + n_params * np.log(n_samples)
    
    return bic_score


def find_best_hmm(df, feature_list, param_grid):
    """
    Performs a grid search to find the best HMM hyperparameters based on BIC.
    
    Args:
        df (pd.DataFrame): The input dataframe with all features.
        feature_list (list): The specific features to use for the HMM.
        param_grid (dict): A dictionary with lists of values for 'n_states' and 'n_pca_components'.

    Returns:
        pd.DataFrame: A dataframe with the results of the grid search, sorted by BIC.
    """
    results = []
    
    features_subset = df[feature_list].copy().dropna()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_subset)
    
    # Generate all combinations of parameters
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for params in param_combinations:
        n_states = params['n_states']
        n_pca = params['n_pca_components']
        
        print(f"Testing: {n_states} states, {n_pca} PCA components...")
        
        # Apply PCA
        pca = PCA(n_components=n_pca)
        pca_features = pca.fit_transform(scaled_features)
        
        # Train HMM
        model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100, random_state=42)
        model.fit(pca_features)
        
        # Calculate BIC
        bic = calculate_bic(model, pca_features)
        
        results.append({
            'n_states': n_states,
            'n_pca_components': n_pca,
            'bic_score': bic,
            'log_likelihood': model.score(pca_features)
        })

    results_df = pd.DataFrame(results)
    return results_df.sort_values(by='bic_score', ascending=True)

