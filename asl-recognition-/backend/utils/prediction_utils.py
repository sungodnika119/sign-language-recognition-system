from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, List

import numpy as np
from tensorflow.keras.models import load_model


WORD_MODEL_MISSING_MESSAGE = "Word model not found. Please train the model first."


def load_word_labels(labels_path: str | Path) -> List[str]:
    path = Path(labels_path)
    if not path.exists():
        return []
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return labels


@lru_cache(maxsize=4)
def load_word_model(model_path: str | Path):
    path = Path(model_path)
    if not path.exists():
        return None
    return load_model(path)


def _normalize_sequence(sequence: Iterable[Iterable[float]]) -> np.ndarray | None:
    array = np.asarray(sequence, dtype=np.float32)
    if array.shape != (30, 126):
        return None
    return array


def predict_word_sign(sequence, model, labels, threshold: float = 0.85):
    normalized = _normalize_sequence(sequence)
    if normalized is None:
        return {
            "word": "",
            "confidence": 0.0,
            "accepted": False,
            "error": "Invalid sequence shape. Expected 30 x 126.",
        }

    if model is None:
        return {
            "word": "",
            "confidence": 0.0,
            "accepted": False,
            "error": WORD_MODEL_MISSING_MESSAGE,
        }

    if not labels:
        return {
            "word": "",
            "confidence": 0.0,
            "accepted": False,
            "error": "Word labels not found. Please train the model first.",
        }

    prediction = model.predict(normalized.reshape(1, 30, 126), verbose=0)[0]
    index = int(np.argmax(prediction))
    confidence = float(prediction[index])
    word = labels[index] if index < len(labels) else str(index)
    accepted = confidence >= threshold
    return {
        "word": word,
        "confidence": round(confidence, 4),
        "accepted": accepted,
    }
