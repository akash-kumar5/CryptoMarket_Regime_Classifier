import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_cleaner import load_data
from src.indicators import compute_indicators, compute_multi_tf_indicators
import pandas as pd

DATA_FOLDER = 'data/'

st.title("Compute Technical Indicators")

# List all CSV files in /data/
csv_files = [f for f in os.listdir(DATA_FOLDER) if not f.endswith('_features.csv')]

if csv_files:
    selected_file = st.selectbox("Select Data File", csv_files)

    if selected_file:
        df = pd.read_csv(os.path.join(DATA_FOLDER, selected_file), parse_dates=['timestamp'])
        st.subheader(f"Preview of {selected_file}")
        st.dataframe(df.tail(10))

        if st.button("Compute Indicators"):
            computed_df = compute_multi_tf_indicators(df)

            if computed_df is not None and not computed_df.empty:
                st.success("Indicators Computed Successfully!")
                st.dataframe(computed_df.tail(10))

                # Save enriched file
                save_name = selected_file.replace('.csv', '_features.csv')
                computed_df.to_csv(os.path.join(DATA_FOLDER, save_name), index=False)
                st.success(f"Saved as {save_name}")
            else:
                st.error("Indicator computation failed. DataFrame is empty or invalid.")
else:
    st.warning("No CSV files found in the data folder.")
