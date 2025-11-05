# dashboard/pages/5_HMM_Tuning.py
import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from hmmlearn.hmm import GaussianHMM
from typing import List, Tuple

# Add src path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

st.set_page_config(page_title="HMM Hyperparameter Tuning", layout="wide")
st.title("HMM Hyperparameter Tuning — Grid Search")

DATA_FOLDER = 'data'
MIN_TRAIN_FRAC = 0.6  # minimum fraction to allocate to training

# ------- Helpers -------
def list_feature_files(data_folder: str) -> List[str]:
    try:
        files = [f for f in os.listdir(data_folder) if f.endswith('_features.csv')]
    except FileNotFoundError:
        files = []
    return sorted(files)

@st.cache_data(show_spinner=False)
def load_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['timestamp'], infer_datetime_format=True)
    # normalize timestamp to tz-aware UTC if possible
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    except Exception:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    return df

def numeric_feature_list(df: pd.DataFrame) -> List[str]:
    return [
        c for c in df.columns
        if c != 'timestamp' and np.issubdtype(df[c].dtype, np.number)
        and not c.startswith(("open_","high_","low_","close_","volume_"))
    ]

def prep_matrix(df: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X = df[cols].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    # drop rows with any NaNs in chosen features
    mask = ~X.isna().any(axis=1)
    X_clean = X.loc[mask]
    return X_clean.values, X_clean.index.values

def fit_scale_pca(Xtr: np.ndarray, Xva: np.ndarray, p_dim: int = None):
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xva_s = scaler.transform(Xva)
    pca = None
    if p_dim:
        p_dim = min(p_dim, Xtr_s.shape[1])
        pca = PCA(n_components=p_dim, random_state=42).fit(Xtr_s)
        Xtr_s = pca.transform(Xtr_s)
        Xva_s = pca.transform(Xva_s)
    return scaler, pca, Xtr_s, Xva_s

def bic_score(model: GaussianHMM, X: np.ndarray) -> Tuple[float, float]:
    """
    Returns (BIC, log_likelihood)
    BIC computed on provided X (commonly training set).
    """
    n, d = X.shape
    try:
        ll = model.score(X)
    except Exception:
        ll = -1e99
    # params per state: mean(d) + cov params
    if model.covariance_type == "full":
        params_state = d + d*(d+1)/2
    else:  # diag
        params_state = 2*d
    k = model.n_components
    k_params = int(k * params_state + k*(k-1) + (k-1))
    bic = -2 * ll + k_params * np.log(n) if n > 0 else 1e99
    return bic, ll

# ------- UI: File selection -------
feature_files = list_feature_files(DATA_FOLDER)
if not feature_files:
    st.warning("No '*_features.csv' files found in data/. Run feature engineering first.")
    st.stop()

selected_file = st.selectbox("Select feature CSV", feature_files, index=0)
df = load_features(os.path.join(DATA_FOLDER, selected_file))
st.subheader("Preview (tail)")
st.dataframe(df.tail(6))

# ------- UI: Tuning config -------
st.sidebar.header("Tuning Configuration")

ALL_FEATS = numeric_feature_list(df)
st.sidebar.markdown(f"Detected numeric features: {len(ALL_FEATS)}")

# sensible tiny default if user chooses "custom"
TINY = ['log_ret_1_5m','atr_norm_5m','ema_ratio_9_21_5m','volume_zscore_50_5m']
TINY = [c for c in TINY if c in ALL_FEATS]

# Pre-defined bundles (attempt to pick existing columns only)
def select_existing(cols: List[str]) -> List[str]:
    return [c for c in cols if c in ALL_FEATS]

BUNDLES = {
    "compact": select_existing(['log_ret_1_5m','rsi_14_5m','ema_ratio_9_21_5m','macd_hist_5m','atr_norm_5m','bb_width_5m']),
    "trend_vol": select_existing([
        'taker_buy_ratio_5m','trade_imbalance_5m','spread_pct','pressure_index_10bps','bid_depth_50bps','ask_depth_50bps'    ]),
    "depth_micro": select_existing([
        'log_ret_1_15m','ema_ratio_9_21_15m','macd_hist_15m','rsi_14_15m','fundingRate','fundingRate_d_1h'    ])
}

bundle_names = list(BUNDLES.keys()) + ["custom"]
bundle_choice = st.sidebar.selectbox("Feature bundle", bundle_names, index=0)

if bundle_choice == "custom":
    chosen_features = st.sidebar.multiselect("Select features (≤ 25)", ALL_FEATS, default=TINY)
else:
    chosen_features = BUNDLES[bundle_choice]

if not chosen_features:
    st.warning("No features selected. Pick a bundle or choose custom features.")
    st.stop()

# hyperparameter controls
k_min, k_max = st.sidebar.slider("States K range", 2, 8, (4, 6))
cov_types = st.sidebar.multiselect("Covariance types", ["diag","full"], default=["diag"])
use_pca = st.sidebar.checkbox("Use PCA (recommended)", value=True)
if use_pca:
    max_dim = min(len(chosen_features), 20)
    pca_min, pca_max = st.sidebar.slider("PCA components range", min_value=2, max_value=max_dim, value=(2, 2))
    pca_grid = list(range(pca_min, pca_max + 1))
else:
    pca_grid = [None]

n_iter = st.sidebar.number_input("HMM n_iter", min_value=50, max_value=2000, value=200, step=50)
train_frac = st.sidebar.slider("Train fraction (chronological)", 0.5, 0.95, 0.8, 0.05)

st.sidebar.caption("Tip: start small (K=2..4, diag, PCA 4–8). Validate regimes manually after top configs.")

# ------- Run Grid Search -------
def run_grid_search(df: pd.DataFrame, cols: List[str], k_min: int, k_max: int, cov_types: List[str], pca_grid: List[int], n_iter: int, train_frac: float):
    X_all, idx_all = prep_matrix(df, cols)
    if X_all.size == 0:
        raise ValueError("After dropping NaNs/inf, no rows left.")

    # chronological split
    split_i = max(2, int(train_frac * len(X_all)))
    Xtr_raw, Xva_raw = X_all[:split_i], X_all[split_i:]
    if len(Xtr_raw) < 5 or len(Xva_raw) < 5:
        raise ValueError("Train/validation split too small. Reduce features or lower train_frac.")

    combos = [(K, cov, p_dim) for K in range(k_min, k_max+1) for cov in cov_types for p_dim in pca_grid]
    rows = []
    total = len(combos)
    pb = st.progress(0)
    st.info(f"Running {total} configs — this may take a while depending on grid size.")

    for i, (K, cov, p_dim) in enumerate(combos, start=1):
        try:
            scaler, pca, Xtr, Xva = fit_scale_pca(Xtr_raw, Xva_raw, p_dim)
            # guard numeric stability
            if Xtr.size == 0 or Xva.size == 0:
                raise ValueError("Empty arrays after scaling/PCA")

            hmm = GaussianHMM(n_components=K, covariance_type=cov, n_iter=n_iter, random_state=42)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                try:
                    hmm.fit(Xtr)
                except Exception as exc:
                    # record a failed row with very poor scores
                    rows.append({
                        "K": K, "cov": cov, "pca_dim": (p_dim or Xtr.shape[1]),
                        "train_rows": len(Xtr), "val_rows": len(Xva),
                        "train_ll_per_row": -1e99, "val_ll_per_row": -1e99, "train_BIC": 1e99
                    })
                    pb.progress(i/total)
                    continue

            train_bic, train_ll = bic_score(hmm, Xtr)
            try:
                val_ll = hmm.score(Xva)
            except Exception:
                val_ll = -1e99

            rows.append({
                "K": K, "cov": cov, "pca_dim": (p_dim or Xtr.shape[1]),
                "train_rows": len(Xtr), "val_rows": len(Xva),
                "train_ll_per_row": (train_ll / len(Xtr)) if len(Xtr) else -1e99,
                "val_ll_per_row": (val_ll / len(Xva)) if len(Xva) else -1e99,
                "train_BIC": train_bic
            })

        except Exception as e:
            # robust: record failure and continue
            rows.append({
                "K": K, "cov": cov, "pca_dim": (p_dim or None),
                "train_rows": len(Xtr_raw[:split_i]) if 'Xtr_raw' in locals() else 0,
                "val_rows": len(Xva_raw) if 'Xva_raw' in locals() else 0,
                "train_ll_per_row": -1e99, "val_ll_per_row": -1e99, "train_BIC": 1e99,
                "err": str(e)
            })
        pb.progress(i/total)

    res = pd.DataFrame(rows)
    # sort by val_ll_per_row desc then BIC asc
    res_sorted = res.sort_values(['val_ll_per_row','train_BIC'], ascending=[False, True]).reset_index(drop=True)
    return res_sorted

if st.button("Run Grid Search", type="primary"):
    try:
        res = run_grid_search(df, chosen_features, k_min, k_max, cov_types, pca_grid, n_iter, train_frac)
    except Exception as e:
        st.error(f"Grid search failed: {e}")
        st.stop()

    st.subheader("Grid search results (top 20)")
    st.dataframe(res.head(20), use_container_width=True)

    if not res.empty:
        top = res.iloc[0]
        st.success(
            f"Top config → K={int(top.K)}, cov={top.cov}, PCA={int(top.pca_dim)}, "
            f"val_ll/row={top.val_ll_per_row:.4g}, BIC={int(top.train_BIC)}"
        )

    # export
    out_name = f"hmm_grid_{os.path.splitext(os.path.basename(selected_file))[0]}.csv"
    out_path = os.path.join(DATA_FOLDER, out_name)
    res.to_csv(out_path, index=False)
    st.info(f"Saved results to `{out_path}`")

    st.caption("Next steps: pick 1–3 top configs, decode states on the labeling page and inspect state means / returns / vol.")
