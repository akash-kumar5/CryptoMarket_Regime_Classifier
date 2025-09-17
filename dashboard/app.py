import streamlit as st

st.set_page_config(page_title="Market Regime Classifier", layout="wide")

st.title("Market Regime Classifier Dashboard")

st.markdown("""
## Select a Page from Sidebar
- **Fetch Data:** Download raw OHLCV data from Binance.
- **Compute Features:** Compute Technical Indicators on existing CSV files.
- **Regime Classifier:** Classify each tick into 4 class of regime using previous tick's data.
- **Model Training:** Train model using LSTM Algo
- **Tuning:** Tuner for HMM and LSTM
- **Live Implementation:** Live inference of the model on real-time data
""")
