import time
import pandas as pd
import joblib
import plotly.graph_objects as go
from src.data_fetcher import fetch_historical_data
from src.indicators import compute_indicators

# Load trained model
clf = joblib.load('models/regime_classifier_model.pkl')

# Reverse Mapping for Predictions
regime_mapping_reverse = {0: 'Uptrend', 1: 'Downtrend', 2: 'Range', 3: 'Squeeze'}

# Features used in training
features = ['EMA20', 'EMA50', 'EMA200', 'RSI14', 'MACD_Hist', 'ATR14', 'BB_Width', 'ADX14']

# Binance parameters
symbol = 'BTCUSDT'
interval = '5m'  # You can change this to '1m', '15m', etc.
poll_interval_sec = 30  # Refresh every 30 seconds

def get_latest_data():
    end_time = int(time.time() * 1000)
    start_time = end_time - (100 * 5 * 60 * 1000)  # Last 100 candles of 5min
    df = fetch_historical_data(symbol=symbol, interval=interval, start_time=start_time, limit=100)
    df = compute_indicators(df)
    df = df.dropna()
    return df

def predict_and_plot(df):
    X_live = df[features]
    predicted_labels = clf.predict(X_live)
    df['Predicted_Regime'] = [regime_mapping_reverse[label] for label in predicted_labels]

    # Plot Candlestick Chart
    fig = go.Figure(data=[go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="Price"
    )])

    # Add regime markers
    color_map = {'Uptrend': 'green', 'Downtrend': 'red', 'Squeeze': 'blue', 'Range': 'gray'}
    for idx, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['timestamp']],
            y=[row['high'] * 1.001],  # Slightly above candle
            mode="text",
            text=[row['Predicted_Regime']],
            textposition="top center",
            textfont=dict(color=color_map.get(row['Predicted_Regime'], 'black'), size=10),
            showlegend=False
        ))

    fig.update_layout(title=f"{symbol} {interval} — Regime Predictions", xaxis_rangeslider_visible=False, height=700)
    fig.show()

# Main Loop
while True:
    print(f"Fetching latest data for {symbol}...")
    df_live = get_latest_data()
    predict_and_plot(df_live)
    print(f"Waiting {poll_interval_sec} seconds for next update...\n")
    time.sleep(poll_interval_sec)
