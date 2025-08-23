# src/lstm_tuner.py
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

def build_tuned_model(hp, input_shape, num_classes):
    """
    Builds a Keras model with hyperparameters defined by the KerasTuner.
    This function is the core of the automated tuning process.
    
    Args:
        hp (HyperParameters): The KerasTuner HyperParameters object.
        input_shape (tuple): The shape of the input data (time_steps, n_features).
        num_classes (int): The number of output classes.

    Returns:
        A compiled Keras model.
    """
    # 1. Define the search space for each hyperparameter
    hp_lstm_units = hp.Int('lstm_units', min_value=32, max_value=128, step=16)
    hp_dense_units = hp.Int('dense_units', min_value=32, max_value=96, step=16)
    hp_dropout = hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    # 2. Build the model using the hyperparameters
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units=hp_lstm_units, return_sequences=False),
        Dropout(rate=hp_dropout),
        Dense(units=hp_dense_units, activation='relu'),
        Dropout(rate=hp_dropout),
        Dense(units=num_classes, activation='softmax')
    ])
    
    # 3. Compile the model with the hyperparameter learning rate
    model.compile(
        optimizer=Adam(learning_rate=hp_learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

