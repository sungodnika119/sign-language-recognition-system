from __future__ import annotations

import time
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.utils.mediapipe_utils import create_hand_detector, draw_landmarks, extract_hand_keypoints, mediapipe_detection


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

NO_SEQUENCES = 50
SEQUENCE_LENGTH = 30

DATA_DIR = ROOT_DIR / "backend" / "sign_data"
HAND_TASK_PATH = ROOT_DIR / "backend" / "models" / "hand_landmarker.task"


def draw_status(image, action, sequence, frame, countdown=None):
    cv2.rectangle(image, (0, 0), (640, 110), (0, 0, 0), -1)
    cv2.putText(image, f"Sign: {action}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"Sequence: {sequence + 1}/{NO_SEQUENCES}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 240, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"Frame: {frame + 1}/{SEQUENCE_LENGTH}", (320, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 240, 255), 2, cv2.LINE_AA)
    if countdown is not None:
        cv2.putText(image, f"Start in: {countdown:.1f}s", (460, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, "q = quit | s = skip sequence", (320, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2, cv2.LINE_AA)
    return image


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for action in ACTIONS:
        (DATA_DIR / action).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam.")

    skipped = False
    with create_hand_detector(HAND_TASK_PATH, max_num_hands=2) as hands:
        for action in ACTIONS:
            for sequence in range(NO_SEQUENCES):
                skipped = False
                start_time = time.time()
                while time.time() - start_time < 2:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    image, results = mediapipe_detection(frame, hands)
                    countdown = 2 - (time.time() - start_time)
                    draw_status(image, action, sequence, 0, max(countdown, 0))
                    draw_landmarks(image, results)
                    cv2.imshow("Collect Word Data", image)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    if key == ord("s"):
                        skipped = True
                        break
                if skipped:
                    continue

                sequence_data = []
                for frame_num in range(SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    image, results = mediapipe_detection(frame, hands)
                    draw_landmarks(image, results)
                    draw_status(image, action, sequence, frame_num)
                    keypoints = extract_hand_keypoints(results)
                    sequence_data.append(keypoints)
                    cv2.imshow("Collect Word Data", image)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    if key == ord("s"):
                        skipped = True
                        break

                if skipped:
                    continue

                output_path = DATA_DIR / action / f"{sequence}.npy"
                np.save(output_path, np.array(sequence_data, dtype=np.float32))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
