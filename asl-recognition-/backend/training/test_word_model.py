from __future__ import annotations

import time
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.utils.mediapipe_utils import create_hand_detector, extract_hand_keypoints, mediapipe_detection
from backend.utils.prediction_utils import load_word_labels, load_word_model, predict_word_sign


MODEL_PATH = ROOT_DIR / "backend" / "models" / "asl_word_lstm_model.h5"
LABELS_PATH = ROOT_DIR / "backend" / "labels_words.txt"
HAND_TASK_PATH = ROOT_DIR / "backend" / "models" / "hand_landmarker.task"


def main():
    model = load_word_model(MODEL_PATH)
    labels = load_word_labels(LABELS_PATH)
    if model is None:
        raise FileNotFoundError("Word model not found. Please train the model first.")
    if not labels:
        raise FileNotFoundError("Word labels not found. Please train the model first.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam.")

    sequence = deque(maxlen=30)
    detected_sentence = []
    last_word = ""
    last_word_time = 0.0

    with create_hand_detector(HAND_TASK_PATH, max_num_hands=2) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            image, results = mediapipe_detection(frame, hands)
            keypoints = extract_hand_keypoints(results)
            sequence.append(keypoints)

            if len(sequence) == 30:
                output = predict_word_sign(list(sequence), model, labels, threshold=0.85)
                word = output["word"]
                confidence = output["confidence"]
                accepted = output["accepted"]

                if accepted and word and word != last_word and time.time() - last_word_time > 1.0:
                    detected_sentence.append(word)
                    last_word = word
                    last_word_time = time.time()

                cv2.putText(image, f"Word: {word} ({confidence:.2f})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, "Collecting sequence...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

            sentence_text = " ".join(detected_sentence) if detected_sentence else ""
            cv2.rectangle(image, (0, image.shape[0] - 60), (image.shape[1], image.shape[0]), (0, 0, 0), -1)
            cv2.putText(image, sentence_text or "Sentence will appear here", (20, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Word Model Test", image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                detected_sentence.clear()
                last_word = ""
            if key == 8 and detected_sentence:
                detected_sentence.pop()
                last_word = detected_sentence[-1] if detected_sentence else ""

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
