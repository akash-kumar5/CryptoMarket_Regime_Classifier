# src/lstm_model.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

def create_sequences(X, y, time_steps=64):
    """
    Creates 3D sequences for LSTM model training.
    
    Args:
        X (pd.DataFrame or np.array): Input features.
        y (pd.Series or np.array): Target variable.
        time_steps (int): The lookback window for each sequence.

    Returns:
        tuple: A tuple containing the reshaped features (X_seq) and corresponding labels (y_seq).
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def build_lstm_model(input_shape, num_classes, lstm_units=64, dense_units=32, dropout_rate=0.3):
    """
    Builds a sequential LSTM model using Keras.
    
    Args:
        input_shape (tuple): The shape of the input data (time_steps, n_features).
        num_classes (int): The number of output classes for the classification task.
        lstm_units (int): The number of units in the LSTM layer.
        dense_units (int): The number of units in the Dense hidden layer.
        dropout_rate (float): The dropout rate for regularization.

    Returns:
        tensorflow.keras.models.Model: The compiled Keras model.
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units=lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(units=dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(units=num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

