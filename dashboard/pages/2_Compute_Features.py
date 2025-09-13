# dashboard/pages/2_Feature_Engineering.py
import streamlit as st
import sys
import os
import pandas as pd

# Add src path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from compute_features import build_features

DATA_FOLDER = 'data/'

st.set_page_config(page_title="Feature Engineering", layout="wide")
st.title("Build Feature Set")
st.markdown("""
This step takes the merged multi-timeframe data and computes a rich set of technical indicators and features.

**Key Operations:**
- **Calculates** features like returns, trend, volatility, and momentum for each timeframe (5m, 15m, 1h).
- **Prevents Lookahead Bias:** Critically, all features from context timeframes (5m, 1h) are shifted by one bar. This ensures that the model only uses information that was available at the close of the previous 15m candle, simulating a live environment correctly.
- **Cleans Data:** Removes rows with `NaN` values that result from indicator calculation windows.
""")

# List all merged CSV files in /data/
try:
    merged_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv') and 'merged' in f and 'features' not in f]
except FileNotFoundError:
    st.error(f"Data folder '{DATA_FOLDER}' not found. Please run the data fetching step first.")
    merged_files = []


if merged_files:
    selected_file = st.selectbox("Select Merged Data File", merged_files)

    if selected_file:
        file_path = os.path.join(DATA_FOLDER, selected_file)
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        
        st.subheader(f"Preview of raw merged data from `{selected_file}`")
        st.dataframe(df.tail(10))

        if st.button("Build Feature Set", type="primary"):
            with st.spinner("Computing features and applying leakage control..."):
                # Call the new, robust feature engineering function
                features_df = build_features(df, main_tf='5m', context_tfs=['1m', '15m'])

            if features_df is not None and not features_df.empty:
                st.success("Feature set built successfully!")
                
                st.subheader("Preview of Final Feature Set (tail)")
                st.dataframe(features_df.tail(10))
                
                st.subheader("Feature Set Info")
                st.write(f"**Shape:** {features_df.shape}")
                st.write(f"**Columns:** {features_df.columns.tolist()}")


                # Save the feature-enriched file
                save_name = selected_file.replace('.csv', '_features.csv')
                save_path = os.path.join(DATA_FOLDER, save_name)
                features_df.to_csv(save_path, index=False)
                st.success(f"Feature set saved as `{save_name}`")
            else:
                st.error("Feature engineering failed. The resulting DataFrame is empty.")
else:
    st.warning("No merged data files found in the `data/` folder. Please run the 'Fetch Data' step first.")

