CryptoMarket Regime Classifier
==============================

Market regimes define the context in which trading strategies succeed or fail. A strategy that thrives in a strong trend will collapse in a choppy high-volatility market.This project builds a machine learning pipeline to **classify crypto market regimes** using **multi-timeframe data, Hidden Markov Models (HMM), and LSTMs**, making it easier to design adaptive strategies, position sizing, and risk management.

Features
--------

*   **Multi-timeframe data (5m, 15m, 1h)** from Binance API.
    
*   **Feature engineering** with technical indicators across timeframes.
    
*   **Hidden Markov Models (HMM)** for unsupervised regime labeling.
    
*   **6 distinct regimes** discovered:
    
    1.  Choppy High-Volatility
        
    2.  Strong Trend
        
    3.  Volatility Spike
        
    4.  Weak Trend
        
    5.  Range
        
    6.  Squeeze
        
*   **LSTM classifier** trained on HMM labels for regime prediction.
    
*   **Evaluation metrics**: Precision, Recall, F1 Score, Confusion Matrix.
    
*   **Plug-and-play** saved models + scalers for use in downstream trading systems.
    

📂 Project Structure
--------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   .  ├── dashboard/        # Visualizations, regime plots  ├── models/           # Trained models & scalers  ├── src/              # Feature engineering + training scripts  ├── main.py           # Entry point for pipeline  ├── requirements.txt  # Dependencies  └── README.md   `

⚙️ Workflow
-----------

1.  **Data Fetching**
    
    *   Download OHLCV data from Binance API.
        
    *   Build dataset with multiple timeframes (5m, 15m, 1h).
        
2.  **Feature Engineering**
    
    *   Compute technical indicators (momentum, volatility, trend).
        
    *   Normalize & scale features for ML.
        
3.  **Regime Labeling with HMM**
    
    *   Hidden Markov Model with PCA reduction.
        
    *   Hyperparameter tuning for states/components.
        
    *   Optimal config: **6 states, 4 PCA components (lowest BIC score)**.
        
4.  **LSTM Classification**
    
    *   Train LSTM on HMM-labeled data.
        
    *   Perform hyperparameter tuning (via Keras Tuner).
        
    *   Evaluate via Recall, Precision, F1, and Confusion Matrix.
        
5.  **Model Export**
    
    *   Save trained model + scaler for reuse.
        
    *   Integrate into live trading pipelines.
        

📊 Results
----------

*   LSTM successfully distinguishes between complex market states.
    
*   Confusion matrix shows strong performance in trend vs. non-trend regimes.
    
*   Transitional regimes (weak trend ↔ range, volatility spike ↔ choppy) remain challenging but meaningful.
    

🔮 Future Work
--------------

*   Real-time integration with live trading system.
    
*   Reinforcement Learning for **dynamic position sizing**.
    
*   Expanded dashboard with explainability tools (feature importance, transition probabilities).
    
*   Experiment with alternative regime discovery methods (e.g., clustering, Bayesian HMM).
    

🛠️ Installation
----------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   git clone https://github.com/akash-kumar5/CryptoMarket_Regime_Classifier.git  cd CryptoMarket_Regime_Classifier  pip install -r requirements.txt   `

▶️ Usage
--------

Run the full pipeline:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python main.py   `

Or train components separately:

*   src/hmm\_tuning.py – Tune Hidden Markov Model.
    
*   src/lstm\_train.py – Train LSTM on labeled data.
    

Models & scalers will be saved in /models for reuse.

📌 Notes
--------

*   Data range: ~2 years (to prioritize recent regime behavior and avoid stale market patterns).
    
*   Designed as a **research + foundational tool** for live trading systems.
    
*   Separate repo will handle **live market testing**.
