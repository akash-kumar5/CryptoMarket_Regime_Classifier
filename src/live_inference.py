# main.py
import os, sys
import time
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime, timezone, timedelta
import schedule
import json
import traceback

# --- Add src path to import our feature engineering logic ---

#
# !! CRITICAL: Import the NEW merge_timeframes function and the corrected generate_features !!
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.compute_features import build_features 
from src.data_cleaner import merge_timeframes
from src.fetch_binance_klines import fetch_binance_klines

# --- Configuration ---
MODEL_FOLDER = "models/"
SYMBOL = "BTCUSDT"
MAIN_TF = '5m'
CONTEXT_TFS = ['1m', '15m']
TIME_STEPS = 64 # This MUST match the lookback window used for training

class LiveInferencePipeline:
    def __init__(self, model_path, scaler_path, metadata_path):
        print("Initializing live inference pipeline...")
        # 1. Load all necessary artifacts
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.feature_cols = self.metadata['features']
        # Invert the map for easy lookup: {0: 'Range', 1: 'Squeeze', ...}
        self.regime_map = {int(v): k for k, v in self.metadata['regime_map'].items()}

        print(f"Loaded LSTM model, scaler, and metadata. Expecting {len(self.feature_cols)} features.")
        
        # 2. Initialize data storage
        self.data_store = {}
        self.prefill_data()

    def prefill_data(self):
        """
        Fetches a consistent date range of historical data across all timeframes.
        """
        print("Prefilling historical data for all timeframes...")
        
        # Calculate start time as a millisecond timestamp
        start_dt = datetime.now() - timedelta(days=200)
        start_ms = int(start_dt.timestamp() * 1000)
        
        print(f"Fetching data since {start_dt.strftime('%Y-%m-%d')} for all timeframes...")

        for tf in [MAIN_TF] + CONTEXT_TFS:
            df = fetch_binance_klines(SYMBOL, tf, start_time_ms=start_ms)
            self.data_store[tf] = df
            print(f" -> Fetched {len(df)} candles for {tf}")

        print("Historical data prefilled.")

    # In main.py (or live_pipeline.py)

    def run_prediction_cycle(self, fetch_open_candle=False):
        """
        The main prediction function.
        :param fetch_open_candle: If True, fetches the latest in-progress candle for real-time
                                  updates. If False, only fetches new closed candles.
        """
        print(f"\n--- Running prediction cycle at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} ---")
        try:
            print("Fetching latest candle data...")
            if fetch_open_candle:
                # --- REAL-TIME MODE ---
                # Fetches the latest candle (even if it's not closed) to get live price updates.
                for tf in [MAIN_TF] + CONTEXT_TFS:
                    latest_candle = fetch_binance_klines(SYMBOL, tf,limit=1) # Just get the very last one
                    if not latest_candle.empty:
                        updated_df = pd.concat([self.data_store[tf], latest_candle], ignore_index=True)
                        self.data_store[tf] = updated_df.drop_duplicates(subset='t', keep='last')
            else:
                # --- SCHEDULED MODE (existing logic) ---
                # Fetches only new candles that have closed since the last update.
                for tf in [MAIN_TF] + CONTEXT_TFS:
                    last_timestamp_ms = int(self.data_store[tf]['t'].iloc[-1].timestamp() * 1000)
                    new_data = fetch_binance_klines(SYMBOL, tf, start_time_ms=last_timestamp_ms)
                    if not new_data.empty:
                        updated_df = pd.concat([self.data_store[tf], new_data], ignore_index=True)
                        self.data_store[tf] = updated_df.drop_duplicates(subset='t', keep='last')

            # The rest of the function remains the same...
            cutoff_date = datetime.now() - timedelta(days=200)
            for tf in [MAIN_TF] + CONTEXT_TFS:
                self.data_store[tf] = self.data_store[tf][self.data_store[tf]['t'] >= cutoff_date]

            # ... (merge, build features, predict, and print logic is unchanged) ...
            # 3. Merge and build features
            print("Merging timeframes...")
            main_df = self.data_store[MAIN_TF]
            context_dfs = {tf: self.data_store[tf] for tf in CONTEXT_TFS}
            merged_df = merge_timeframes(MAIN_TF, main_df, context_dfs)

            print("Building features...")
            features_df = build_features(merged_df, main_tf=MAIN_TF, context_tfs=CONTEXT_TFS)
            
            if len(features_df) < TIME_STEPS:
                print(f"Not enough data to form a sequence. Have {len(features_df)}, need {TIME_STEPS}. Waiting for more data.")
                return

            # 4. Scale and Predict
            print("Scaling data and creating input sequence...")
            final_features = features_df[self.feature_cols]
            sequence_data = final_features.tail(TIME_STEPS)
            scaled_sequence = self.scaler.transform(sequence_data)
            input_sequence = np.expand_dims(scaled_sequence, axis=0)
            
            print("Making prediction...")
            prediction_probs = self.model.predict(input_sequence, verbose=0)[0]
            predicted_class_index = np.argmax(prediction_probs)
            predicted_regime = self.regime_map.get(predicted_class_index, "Unknown")
            confidence = prediction_probs[predicted_class_index]
            
            # --- IMPROVED DYNAMIC PRINTING LOGIC ---
            # This will create a clean string of all probabilities without hardcoding.
            prob_strings = []
            # Sort by class index (0, 1, 2...) for a consistent order
            for regime_name, index in sorted(self.metadata['regime_map'].items(), key=lambda item: item[1]):
                prob_strings.append(f"{regime_name}={prediction_probs[index]:.2%}")
            
            probabilities_text = ", ".join(prob_strings)
            
            print("\n" + "="*50)
            print(f"PREDICTION FOR NEXT {MAIN_TF} CANDLE ({SYMBOL})")
            print(f"Timestamp: {features_df['t'].iloc[-1]}")
            print(f"Predicted Regime: {predicted_regime.upper()} (Confidence: {confidence:.2%})")
            print(f"Probabilities: {probabilities_text}")
            print("="*50 + "\n")
            
        except Exception as e:
            print(f"An error occurred during the prediction cycle: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    model_file = os.path.join(MODEL_FOLDER, "lstm_regime_model.keras")
    scaler_file = os.path.join(MODEL_FOLDER, "scaler.joblib")
    meta_file = os.path.join(MODEL_FOLDER, "lstm_model_metadata.json")

    if not all(os.path.exists(f) for f in [model_file, scaler_file, meta_file]):
        print("Error: Model, scaler, or metadata file not found. Please train the model first.")
    else:
        pipeline = LiveInferencePipeline(model_file, scaler_file, meta_file)
        
        # Run once immediately to test
        pipeline.run_prediction_cycle()
        
        print("Scheduling prediction job for every 5 minutes...")
        schedule.every().hour.at(":00").do(pipeline.run_prediction_cycle)
        schedule.every().hour.at(":05").do(pipeline.run_prediction_cycle)
        schedule.every().hour.at(":10").do(pipeline.run_prediction_cycle)
        schedule.every().hour.at(":15").do(pipeline.run_prediction_cycle)
        schedule.every().hour.at(":20").do(pipeline.run_prediction_cycle)
        schedule.every().hour.at(":25").do(pipeline.run_prediction_cycle)
        schedule.every().hour.at(":30").do(pipeline.run_prediction_cycle)
        schedule.every().hour.at(":35").do(pipeline.run_prediction_cycle)
        schedule.every().hour.at(":40").do(pipeline.run_prediction_cycle)
        schedule.every().hour.at(":45").do(pipeline.run_prediction_cycle)
        schedule.every().hour.at(":50").do(pipeline.run_prediction_cycle)
        schedule.every().hour.at(":55").do(pipeline.run_prediction_cycle)
        while True:
            schedule.run_pending()
            time.sleep(1)