import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Project paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
DATA_FOLDER = "data/"
MODEL_FOLDER = "models/"

st.title("Market Regime Model Training — LightGBM (5m target, 15m/1h as context)")

# ----- Load Data -----
csv_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith("_labeled.csv")]
if not csv_files:
    st.warning("No _labeled.csv files found. Please label regimes first.")
    st.stop()

selected_file = st.selectbox("Select labeled dataset", csv_files)
df = pd.read_csv(os.path.join(DATA_FOLDER, selected_file))
st.subheader(f"Preview of {selected_file}")
st.dataframe(df.tail(10))

# ----- Config -----
st.markdown("### Settings")
test_size = st.slider("Test size (fraction at end of series)", 0.05, 0.4, 0.2, 0.01)
early_stop_rounds = st.number_input("Early stopping rounds", min_value=10, max_value=1000, value=100, step=10)
use_class_balance = st.checkbox("Balance classes (LightGBM class_weight)", value=True)

# ----- Label Encoding -----
regime_map = {"Uptrend": 0, "Downtrend": 1, "Range": 2, "Squeeze": 3, "Undefined": -1}
inv_regime_map = {v: k for k, v in regime_map.items() if v >= 0}

if "Regime_5m" not in df.columns:
    st.error("Missing 'Regime_5m' column. Your labeled dataset must include it.")
    st.stop()

df["Regime_5m_label"] = df["Regime_5m"].map(regime_map)

if "Regime_15m" in df.columns:
    df["Regime_15m_ctx"] = df["Regime_15m"].map(regime_map)
if "Regime_1h" in df.columns:
    df["Regime_1h_ctx"] = df["Regime_1h"].map(regime_map)

# ----- Features -----
exclude_cols = {"timestamp", "Regime_5m", "Regime_15m", "Regime_1h"}
candidate_cols = [c for c in df.columns if c not in exclude_cols]
numeric_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]

target_col = "Regime_5m_label"
df = df[df[target_col] >= 0].copy()
df = df.dropna(subset=numeric_cols + [target_col])

X = df[numeric_cols]
y = df[target_col].astype(int)

st.write(f"Total usable samples: {len(df)}")
st.write(f"Features used: {len(numeric_cols)}")

# ----- Split Data (time-based) -----
n = len(df)
test_n = max(1, int(n * test_size))
train_n = n - test_n

X_train, X_test = X.iloc[:train_n], X.iloc[train_n:]
y_train, y_test = y.iloc[:train_n], y.iloc[train_n:]

# validation split from tail of train
val_n = max(1, int(len(X_train) * 0.1))
X_tr, X_val = X_train.iloc[:-val_n], X_train.iloc[-val_n:]
y_tr, y_val = y_train.iloc[:-val_n], y_train.iloc[-val_n:]

st.write(f"Train: {len(X_tr)} | Val: {len(X_val)} | Test: {len(X_test)}")
st.write("Class distribution in training:", y_tr.value_counts().to_dict())

# ----- Train -----
if st.button("Train Model"):
    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=4,
        boosting_type="gbdt",
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced" if use_class_balance else None,
    )

    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(early_stop_rounds, verbose=False)]
    )

    # ----- Eval -----
    y_pred = clf.predict(X_test)

    unique_labels = sorted(set(y_test) | set(y_pred))
    target_names = [inv_regime_map[i] for i in unique_labels]

    st.subheader("Classification Report (Test)")
    report = classification_report(
        y_test, y_pred,
        labels=unique_labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Confusion Matrix (Test)")
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    fig, ax = plt.subplots()
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=target_names, yticklabels=target_names, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)

    # ----- Feature Importance -----
    st.subheader("Feature Importance (Gain)")
    try:
        importances = clf.booster_.feature_importance(importance_type="gain")
        feature_names = clf.booster_.feature_name()
    except Exception:
        importances = clf.feature_importances_
        feature_names = list(X.columns)

    fi = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    fi = fi.sort_values("Importance", ascending=False)
    st.dataframe(fi)

    fig2, ax2 = plt.subplots(figsize=(8, max(4, len(fi.head(25)) * 0.3)))
    sns.barplot(data=fi.head(25), x="Importance", y="Feature", ax=ax2)
    ax2.set_title("Top 25 Features")
    st.pyplot(fig2)

    # ----- Save -----
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    base_name = os.path.splitext(selected_file)[0]
    model_path = os.path.join(MODEL_FOLDER, f"lgbm_regime5m__{base_name}.pkl")
    joblib.dump(clf, model_path)

    meta = {
        "target": "Regime_5m",
        "regime_map": {k: int(v) for k, v in regime_map.items()},
        "features": feature_names,
        "train_size": int(len(X_tr)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "selected_file": selected_file,
        "params": clf.get_params()
    }
    meta_path = os.path.join(MODEL_FOLDER, f"lgbm_regime5m__{base_name}.meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    st.success(f"Model saved: {model_path}")
    st.caption(f"Metadata saved: {meta_path}")
