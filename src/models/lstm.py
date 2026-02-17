import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

"""Will update when data that is actually being inputed is available. Just need to swap out load_data for
generate_dummy_data."""

def load_data(path=data_path) -> (np.ndarray, np.ndarray):
    # To be implemented depending
    raise NotImplementedError

def generate_dummy_data(n_samples=1000,window_size=20, n_features=5,) -> (np.ndarray, np.ndarray):
    X = np.random.randn(n_samples, window_size, n_features)
    y = np.random.randint(0, 2, size=(n_samples,))
    return X, y

def build_model(window_size: int,n_features: int,lstm_units: int = 32,
                dropout_rate: float = 0.2, learning_rate: float = 1e-3,
) -> tf.keras.Model:

    inputs = Input(shape=(window_size, n_features), name="Features")

    x = LSTM(lstm_units, name="LSTM")(inputs)

    x = Dropout(dropout_rate, name="Dropout")(x)

    outputs = Dense(1, activation="sigmoid", name="Output")(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model

def train_model(model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray, 
                batch_size: int = 32, epochs: int = 10, validation_split: float = 0.2,
                lstm_path: str = "src/trained/lstm_classification.keras",
) -> tf.keras.callbacks.History:
    
    os.makedirs(os.path.dirname(lstm_path), exist_ok=True)


    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=5,
                                   restore_best_weights=True,
                                   mode='min')

    checkpointer = ModelCheckpoint(filepath=lstm_path,
                                   verbose=1,
                                   monitor='val_loss',
                                   mode='min',
                                   save_best_only=True)

    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
    )

    return history

def print_simple_results(history: tf.keras.callbacks.History) -> None:
    h = history.history

    train_loss = h["loss"][-1]
    train_acc = h["accuracy"][-1]
    val_loss = h.get("val_loss", [None])[-1]
    val_acc = h.get("val_accuracy", [None])[-1]

    best_val_loss = min(h["val_loss"]) if "val_loss" in h else None

    print("\n=== Simple Results ===")
    print(f"Final train loss: {train_loss:.4f} | train acc: {train_acc:.4f}")
    if val_loss is not None:
        print(f"Final   val loss: {val_loss:.4f} | val acc: {val_acc:.4f}")
        print(f"Best    val loss: {best_val_loss:.4f}")


def main():
    X, y = generate_dummy_data(n_samples=2000, window_size=20, n_features=5)

    window_size = X.shape[1]
    n_features = X.shape[2]

    model = build_model(window_size=window_size, n_features=n_features)

    history = train_model(model, X, y, epochs=10)

    print_simple_results(history)



if __name__ == "__main__":
    main()