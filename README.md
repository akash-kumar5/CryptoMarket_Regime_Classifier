CryptoMarket Regime Classifier
==============================
**Adaptive Market Intelligence for Crypto Strategies**

Trading strategies don’t exist in a vacuum — they succeed or fail depending on the market regime they operate in. A breakout strategy that crushes in a trending market will bleed in a choppy one.

CryptoMarket Regime Classifier is a complete machine learning pipeline that identifies and predicts market regimes in crypto markets using multi-timeframe data, technical features, Hidden Markov Models (HMM), and LSTMs.

It’s designed as a foundational intelligence layer for strategy selection, position sizing, and risk management — and will power the regime-awareness module in Dazai[].

Pipeline Overview
--------

From raw data to deployable model:

1. Data Fetching – Periodically pulls OHLCV data (5m, 15m, 1h) from Binance.

2. Feature Engineering – Computes momentum, volatility, and trend indicators across timeframes.

3. Unsupervised Labeling (HMM) – Uses PCA-reduced feature space to discover market regimes.

4. Supervised Prediction (LSTM) – Trains a sequence model on HMM labels to predict regimes.

5. Model Export – Saves trained model & scalers for integration with live systems.

6. Live Classification – Periodically classifies the current regime with plans for probabilistic outputs.


## Key Features
- Multi-timeframe data (5m, 15m, 1h) from Binance
- Feature engineering with technical indicators (momentum, volatility, trend)
- Hidden Markov Models (HMM) for unsupervised regime discovery
- LSTM classifier trained on HMM-labeled sequences
- 6 distinct regimes identified:
    -- Choppy High-Volatility
    -- Strong Trend
    -- Volatility Spike
    -- Weak Trend
    -- Range
    -- Squeeze
- Plug-and-play model + scaler for downstream usage
- Evaluation metrics: Precision, Recall, F1 Score, Confusion Matrix
    

📂 Project Structure
--------------------
├── dashboard/        # Visualizations, regime plots  
├── models/           # Trained models & scalers  
├── src/              # Feature engineering + training scripts  
├── main.py           # End-to-end pipeline execution  
├── requirements.txt  # Dependencies  
└── README.md

⚙️ Workflow
-----------

1. **Data Fetching**

    * Periodically fetches OHLCV data from Binance.

    * Currently optimized for 5m data (can be adapted for other timeframes).

2. **Feature Engineering**

    Calculates multi-timeframe indicators: momentum, trend, volatility.

    Scales and normalizes data for ML pipelines.

3. **Regime Discovery (HMM)**

    PCA-reduced features used for Hidden Markov Model labeling.

    Optimal configuration: 6 states, 4 PCA components (lowest BIC).

4. **Regime Prediction (LSTM)**

    Sequence model trained on HMM labels.

    Hyperparameter tuning via Keras Tuner.

    Planned: probabilistic regime outputs (distribution across states).

5. **Model Export & Live Integration**

    Saves trained LSTM + scaler for downstream usage.

    Periodically classifies current market regime.

    Future: direct integration with Dazai’s trading logic.
        

📊 Results
----------

*   LSTM successfully distinguishes between complex market states.
    
*   Confusion matrix shows strong performance in trend vs. non-trend regimes.
    
*   Transitional regimes (weak trend ↔ range, volatility spike ↔ choppy) remain challenging but meaningful.
    

🔮 Future Work
--------------

*    Probabilistic regime predictions for richer decision-making.

*    Real-time integration with live trading pipelines.

*    Reinforcement Learning for adaptive position sizing.

*    Explainability layer (feature importance, transition probabilities).

*    Exploration of alternative regime discovery techniques (Bayesian HMM, clustering).
    

🛠️ Installation
----------------
git clone https://github.com/akash-kumar5/CryptoMarket_Regime_Classifier.git
cd CryptoMarket_Regime_Classifier
pip install -r requirements.txt


▶️ Usage
--------

Run the full pipeline:

python main.py

    

Models & scalers will be saved in /models for reuse.

📌 Notes
--------

*   Data range: ~2 years (to prioritize recent regime behavior and avoid stale market patterns).
    
*   Designed as a **research + foundational tool** for live trading systems.
    
*   Future versions will connect directly into **Dazai** as a core regime intelligence component.
