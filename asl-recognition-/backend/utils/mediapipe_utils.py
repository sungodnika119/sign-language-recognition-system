from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np

try:
    from mediapipe.python.solutions import hands as mp_hands
except Exception:  # pragma: no cover - used by newer Tasks-only mediapipe builds
    mp_hands = None

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision
except Exception:  # pragma: no cover
    mp = None
    mp_tasks = None
    vision = None


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


class TasksHandDetector:
    def __init__(self, model_path: str | Path, max_num_hands: int = 2):
        if mp is None or mp_tasks is None or vision is None:
            raise RuntimeError("MediaPipe Tasks API is not available.")
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"MediaPipe hand landmarker task not found: {path}")
        options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_path=str(path),
                delegate=mp_tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process(self, image_rgb):
        image_rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        return self.detector.detect(mp_image)

    def detect(self, mp_image):
        return self.detector.detect(mp_image)

    def close(self):
        self.detector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class TasksFaceDetector:
    def __init__(self, model_path: str | Path, max_num_faces: int = 1):
        if mp is None or mp_tasks is None or vision is None:
            raise RuntimeError("MediaPipe Tasks API is not available.")
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"MediaPipe face landmarker task not found: {path}")
        options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_path=str(path),
                delegate=mp_tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=max_num_faces,
            output_face_blendshapes=True,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, mp_image):
        return self.detector.detect(mp_image)

    def close(self):
        self.detector.close()


def create_hand_detector(model_path: str | Path | None = None, max_num_hands: int = 2):
    if model_path is not None and Path(model_path).exists():
        return TasksHandDetector(model_path=model_path, max_num_hands=max_num_hands)
    if mp_hands is not None:
        return mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    if model_path is None:
        raise RuntimeError("MediaPipe solutions API is unavailable and no task model path was provided.")
    return TasksHandDetector(model_path=model_path, max_num_hands=max_num_hands)


def create_face_detector(model_path: str | Path, max_num_faces: int = 1):
    return TasksFaceDetector(model_path=model_path, max_num_faces=max_num_faces)


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    if results is None:
        return image

    hands_list, _ = _extract_hands(results)
    height, width = image.shape[:2]

    for hand_landmarks in hands_list:
        points = hand_landmarks.landmark if hasattr(hand_landmarks, "landmark") else hand_landmarks
        canvas_points = []
        for point in points:
            canvas_points.append((int(point.x * width), int(point.y * height)))

        for start, end in HAND_CONNECTIONS:
            if start < len(canvas_points) and end < len(canvas_points):
                cv2.line(image, canvas_points[start], canvas_points[end], (0, 255, 120), 2)
        for point in canvas_points:
            cv2.circle(image, point, 4, (255, 255, 255), -1)
            cv2.circle(image, point, 5, (0, 255, 120), 1)

    return image


def _points_to_array(points: Iterable) -> np.ndarray:
    coords: List[float] = []
    for point in points:
        coords.extend([float(point.x), float(point.y), float(getattr(point, "z", 0.0))])
    return np.array(coords, dtype=np.float32)


def _zero_hand() -> np.ndarray:
    return np.zeros(63, dtype=np.float32)


def _hand_label(handedness_item) -> str:
    if not handedness_item:
        return "Unknown"
    label = getattr(handedness_item[0], "category_name", None) or getattr(handedness_item[0], "display_name", None)
    return str(label or "Unknown")


def _extract_hands(results):
    hands_list = getattr(results, "multi_hand_landmarks", None)
    handedness = getattr(results, "multi_handedness", None)
    if hands_list is None:
        hands_list = getattr(results, "hand_landmarks", None)
    if handedness is None:
        handedness = getattr(results, "handedness", None)
    return hands_list or [], handedness or []


def extract_hand_keypoints(results):
    hands_list, handedness = _extract_hands(results)
    left_hand = _zero_hand()
    right_hand = _zero_hand()

    for index, hand_landmarks in enumerate(hands_list[:2]):
        label = _hand_label(handedness[index]) if index < len(handedness) else "Unknown"
        points = _points_to_array(hand_landmarks.landmark if hasattr(hand_landmarks, "landmark") else hand_landmarks)
        if points.shape[0] != 63:
            points = _zero_hand()
        if label.lower() == "left":
            left_hand = points
        elif label.lower() == "right":
            right_hand = points
        elif index == 0:
            left_hand = points
        else:
            right_hand = points

    return np.concatenate([left_hand, right_hand]).astype(np.float32)
