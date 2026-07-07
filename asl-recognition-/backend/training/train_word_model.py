from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "backend" / "sign_data"
MODEL_DIR = ROOT_DIR / "backend" / "models"
MODEL_PATH = MODEL_DIR / "asl_word_lstm_model.h5"
LABELS_PATH = ROOT_DIR / "backend" / "labels_words.txt"

ACTIONS = [
    "hello",
    "thank_you",
    "please",
    "sorry",
    "yes",
    "no",
    "help",
    "water",
    "food",
    "eat",
    "drink",
    "doctor",
    "school",
    "home",
    "i",
    "you",
    "want",
    "need",
    "stop",
    "go",
]


def load_dataset():
    sequences = []
    labels = []
    available_dirs = {path.name for path in DATA_DIR.iterdir() if path.is_dir()}
    extra_actions = sorted(available_dirs - set(ACTIONS))
    action_names = [action for action in ACTIONS if action in available_dirs] + extra_actions

    for action in action_names:
        action_dir = DATA_DIR / action
        for file_path in sorted(action_dir.glob("*.npy")):
            sequence = np.load(file_path)
            if sequence.shape != (30, 126):
                continue
            sequences.append(sequence)
            labels.append(action)

    return np.array(sequences, dtype=np.float32), np.array(labels), action_names


def build_model(num_classes: int):
    model = Sequential([
        LSTM(64, return_sequences=True, activation="relu", input_shape=(30, 126)),
        Dropout(0.3),
        LSTM(128, return_sequences=True, activation="relu"),
        Dropout(0.3),
        LSTM(64, return_sequences=False, activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    return model


def main():
    X, y, action_names = load_dataset()
    print(f"Samples loaded: {len(X)}")
    if len(X) == 0:
        raise RuntimeError("No training data found in backend/sign_data.")

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    np.savetxt(LABELS_PATH, action_names, fmt="%s")

    label_to_index = {label: index for index, label in enumerate(action_names)}
    y_indices = np.array([label_to_index[label] for label in y], dtype=np.int32)
    y_encoded = to_categorical(y_indices, num_classes=len(action_names))

    model = build_model(len(action_names))
    class_counts = np.bincount(y_indices, minlength=len(action_names))
    nonzero_counts = class_counts[class_counts > 0]
    can_stratify = len(X) >= 10 and len(nonzero_counts) > 1 and np.min(nonzero_counts) >= 2
    max_epochs = 50 if can_stratify else 1
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    if can_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=0.2,
            random_state=42,
            stratify=y_indices,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=42,
        )
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=max_epochs,
            callbacks=[early_stopping],
            verbose=1,
        )
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    else:
        print("Dataset is too small for stratified splitting. Training with validation_split=0.2 and saving the model anyway.")
        history = model.fit(
            X,
            y_encoded,
            validation_split=0.2,
            epochs=max_epochs,
            callbacks=[early_stopping],
            verbose=1,
        )
        train_loss, train_acc = model.evaluate(X, y_encoded, verbose=0)
        val_loss, val_acc = float("nan"), float("nan")
        test_loss, test_acc = float("nan"), float("nan")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    print(f"Training accuracy: {history.history['categorical_accuracy'][-1]:.4f}")
    if "val_categorical_accuracy" in history.history:
        print(f"Validation accuracy: {history.history['val_categorical_accuracy'][-1]:.4f}")
    else:
        print("Validation accuracy: n/a")
    if not np.isnan(test_acc):
        print(f"Test accuracy: {test_acc:.4f}")
    else:
        print("Test accuracy: n/a")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved labels to: {LABELS_PATH}")


if __name__ == "__main__":
    main()
