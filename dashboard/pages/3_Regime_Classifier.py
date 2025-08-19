import streamlit as st
import sys
import os
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.regime_label import label_market_regime  # updated function with prefix support

DATA_FOLDER = 'data/'

st.title("Market Regime Labeling")

# Find CSV files
csv_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('_features.csv')]

if csv_files:
    selected_file = st.selectbox("Select Feature File", csv_files)

    if selected_file:
        df = pd.read_csv(os.path.join(DATA_FOLDER, selected_file))
        st.subheader(f"Preview of {selected_file}")
        st.dataframe(df.tail(10))

        if st.button("Label Market Regimes"):
            # Label for multiple timeframes
            for tf in ["5m", "15m", "1h"]:
                df = label_market_regime(df, prefix=tf)

            st.success("Market Regimes Labeled Successfully for 5m, 15m, and 1h!")
            st.dataframe(df.tail(10))

            # Save labeled data
            save_name = selected_file.replace('_features.csv', '_labeled.csv')
            df.to_csv(os.path.join(DATA_FOLDER, save_name), index=False)
            st.success(f"Labeled data saved as {save_name}")
else:
    st.warning("No _features.csv files found. Compute indicators first.")
