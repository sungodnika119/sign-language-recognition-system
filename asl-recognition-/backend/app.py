import base64
import os
import pickle
import subprocess
import threading
import traceback
import sys
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(BASE_DIR, ".matplotlib"))

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify
from flask_cors import CORS
from flask import request
from tensorflow.keras.models import load_model


PROJECT_ROOT = os.path.dirname(BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.utils.mediapipe_utils import create_face_detector, create_hand_detector, extract_hand_keypoints
from backend.utils.prediction_utils import (
    WORD_MODEL_MISSING_MESSAGE,
    load_word_labels,
    load_word_model,
    predict_word_sign,
)
from backend.utils.sentence_generator import generate_paragraph, generate_sentence

MODEL_PATH = os.path.join(BASE_DIR, "models", "asl_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "models", "labels.pkl")
HAND_LANDMARKER_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")
FACE_LANDMARKER_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")
WORD_MODEL_PATH = os.path.join(BASE_DIR, "models", "asl_word_lstm_model.h5")
WORD_LABELS_PATH = os.path.join(BASE_DIR, "labels_words.txt")
SIGN_DATA_DIR = os.path.join(BASE_DIR, "sign_data")
TRAIN_SCRIPT_PATH = os.path.join(BASE_DIR, "training", "train_word_model.py")


app = Flask(__name__)
CORS(app)
app.logger.setLevel("INFO")

hand_landmarker = None
face_landmarker = None
hand_lock = threading.Lock()

model = None
label_encoder = None
word_model = None
word_labels = []
training_process = None
training_words_cache = []


def load_assets():
    global model, label_encoder, hand_landmarker, face_landmarker, word_model, word_labels
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "rb") as f:
            label_encoder = pickle.load(f)
    word_model = load_word_model(WORD_MODEL_PATH)
    word_labels = load_word_labels(WORD_LABELS_PATH)
    sync_word_labels_from_data()
    refresh_training_words_cache()
    hand_landmarker = create_hand_detector(HAND_LANDMARKER_PATH, max_num_hands=2)
    try:
        face_landmarker = create_face_detector(FACE_LANDMARKER_PATH, max_num_faces=1)
    except Exception as exc:
        app.logger.warning("Face landmarker unavailable; continuing without face overlay: %s", exc)
        face_landmarker = None


def decode_image(image_data: str):
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    binary = base64.b64decode(image_data)
    arr = np.frombuffer(binary, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def extract_features(results):
    primary_index = get_primary_hand_index(results)
    landmarks = get_hand_landmarks(results)[primary_index]
    points_source = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks
    points = np.array([[point.x, point.y, point.z] for point in points_source], dtype=np.float32)

    handedness = "Right"
    handedness_list = get_handedness(results)
    if handedness_list and primary_index < len(handedness_list) and handedness_list[primary_index]:
        handedness = get_handedness_label(handedness_list[primary_index])

    if handedness.lower() == "left":
        points[:, 0] = 1.0 - points[:, 0]

    wrist = points[0].copy()
    points = points - wrist

    scale = np.max(np.linalg.norm(points[:, :2], axis=1))
    if scale > 1e-6:
        points = points / scale

    points = np.clip(points, -2.0, 2.0)
    return points.flatten().astype(np.float32)


def get_primary_hand_index(results):
    handedness_list = get_handedness(results)
    if not handedness_list:
        return 0

    best_index = 0
    best_score = -1.0
    for index, handedness in enumerate(handedness_list):
        if not handedness:
            continue
        score = get_handedness_score(handedness)
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


def get_hand_landmarks(results):
    return getattr(results, "hand_landmarks", None) or getattr(results, "multi_hand_landmarks", None) or []


def get_handedness(results):
    return getattr(results, "handedness", None) or getattr(results, "multi_handedness", None) or []


def get_handedness_category(handedness_item):
    if not handedness_item:
        return None
    if isinstance(handedness_item, list):
        return handedness_item[0] if handedness_item else None
    classification = getattr(handedness_item, "classification", None)
    if classification:
        return classification[0]
    return handedness_item[0] if hasattr(handedness_item, "__getitem__") else handedness_item


def get_handedness_label(handedness_item):
    category = get_handedness_category(handedness_item)
    if category is None:
        return "Unknown"
    return (
        getattr(category, "category_name", None)
        or getattr(category, "display_name", None)
        or getattr(category, "label", None)
        or "Unknown"
    )


def get_handedness_score(handedness_item):
    category = get_handedness_category(handedness_item)
    return float(getattr(category, "score", 0.0)) if category is not None else 0.0


def format_landmarks(results):
    formatted = []
    hand_landmarks = get_hand_landmarks(results)
    handedness = get_handedness(results)

    for index, landmarks in enumerate(hand_landmarks):
        hand_label = "Unknown"
        if index < len(handedness) and handedness[index]:
            hand_label = get_handedness_label(handedness[index])

        points = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks

        formatted.append(
            {
                "handedness": hand_label,
                "landmarks": [
                    {"x": point.x, "y": point.y, "z": point.z}
                    for point in points
                ],
            }
        )

    return formatted


def format_face_landmarks(results):
    face_landmarks = getattr(results, "face_landmarks", None)
    if not face_landmarks:
        mesh_landmarks = getattr(results, "multi_face_landmarks", None) or []
        face_landmarks = [item.landmark for item in mesh_landmarks]
    if not face_landmarks:
        return []
    return [
        {"x": point.x, "y": point.y, "z": point.z}
        for point in face_landmarks[0]
    ]


def extract_word_features(results):
    features = extract_hand_keypoints(results)
    return features.tolist() if hasattr(features, "tolist") else list(features)


def sanitize_word_label(value):
    cleaned = str(value or "").strip().lower().replace(" ", "_")
    cleaned = "".join(char for char in cleaned if char.isalnum() or char == "_")
    return cleaned


def reload_word_assets():
    global word_model, word_labels
    load_word_model.cache_clear()
    word_model = load_word_model(WORD_MODEL_PATH)
    word_labels = load_word_labels(WORD_LABELS_PATH)
    return word_model is not None


def discover_training_words():
    base = Path(SIGN_DATA_DIR)
    if not base.exists():
        return []
    return sorted(
        [path.name for path in base.iterdir() if path.is_dir() and path.name != "__pycache__"]
    )


def sync_word_labels_from_data():
    global word_labels
    discovered_words = discover_training_words()
    labels_path = Path(WORD_LABELS_PATH)
    model_exists = os.path.exists(WORD_MODEL_PATH)

    if not discovered_words:
        return

    existing_labels = []
    if labels_path.exists():
        existing_labels = [
            line.strip()
            for line in labels_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    if not model_exists and (not existing_labels or set(existing_labels) != set(discovered_words)):
        labels_path.write_text("\n".join(discovered_words) + "\n", encoding="utf-8")
        word_labels = discovered_words
        return

    if not word_labels:
        word_labels = existing_labels or discovered_words


def refresh_training_words_cache():
    global training_words_cache
    training_words_cache = [
        {"word": word, "sequences": len(list((Path(SIGN_DATA_DIR) / word).glob("*.npy")))}
        for word in discover_training_words()
    ]
    return training_words_cache


def frame_to_hand_landmarks(frame):
    flat = np.asarray(frame, dtype=np.float32).reshape(-1)
    if flat.shape[0] != 126:
        return []

    hands = []
    for index, offset in enumerate((0, 63)):
        chunk = flat[offset:offset + 63]
        if not np.any(chunk):
            continue
        points = chunk.reshape(21, 3)
        hands.append(
            {
                "handedness": "Left" if index == 0 else "Right",
                "landmarks": [
                    {"x": float(point[0]), "y": float(point[1]), "z": float(point[2])}
                    for point in points
                ],
            }
        )

    return hands


def get_loaded_assets_state():
    return {
        "letter_model_loaded": model is not None,
        "letter_labels_loaded": label_encoder is not None,
        "word_model_loaded": word_model is not None,
        "word_labels_loaded": bool(word_labels),
        "hand_landmarker_loaded": hand_landmarker is not None,
        "face_landmarker_loaded": face_landmarker is not None,
        "training_words": refresh_training_words_cache(),
        "word_labels": word_labels,
    }


def detect_face_expression(results):
    if not results.face_blendshapes:
        return "Neutral"

    scores = {}
    for item in results.face_blendshapes[0]:
        scores[getattr(item, "category_name", "")] = float(getattr(item, "score", 0.0))

    smile = max(scores.get("mouthSmileLeft", 0), scores.get("mouthSmileRight", 0))
    jaw_open = scores.get("jawOpen", 0)
    brow_up = max(scores.get("browInnerUp", 0), scores.get("browOuterUpLeft", 0), scores.get("browOuterUpRight", 0))
    brow_down = max(scores.get("browDownLeft", 0), scores.get("browDownRight", 0))

    if smile >= 0.5:
        return "Smile"
    if jaw_open >= 0.5 and brow_up >= 0.2:
        return "Surprised"
    if brow_down >= 0.4:
        return "Focused"
    return "Neutral"


def prepare_input(features):
    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if input_shape and len(input_shape) == 3:
        return features.reshape(1, 63, 1)

    return features.reshape(1, 63)


def decode_label(index):
    if label_encoder is None:
        return str(index)
    if hasattr(label_encoder, "inverse_transform"):
        try:
            return label_encoder.inverse_transform([index])[0]
        except Exception:
            pass
    if hasattr(label_encoder, "classes_"):
        classes = list(label_encoder.classes_)
        if 0 <= index < len(classes):
            return str(classes[index])
    if isinstance(label_encoder, dict):
        for key, value in label_encoder.items():
            if value == index:
                return str(key)
    if isinstance(label_encoder, (list, tuple)) and 0 <= index < len(label_encoder):
        return str(label_encoder[index])
    return str(index)


def format_top_letter_predictions(prediction, limit=3):
    scores = np.asarray(prediction, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        return []

    top_indices = np.argsort(scores)[::-1][:limit]
    return [
        {
            "label": str(decode_label(int(index))),
            "confidence": round(float(scores[index] * 100), 2),
        }
        for index in top_indices
    ]


@app.route("/")
def home():
    return "ASL Detection Backend Running"


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "message": "Backend is running"})


@app.route("/api/generate-sentence", methods=["POST"])
def api_generate_sentence():
    payload = request.get_json(silent=True) or {}
    words = payload.get("words", [])
    sentence = generate_sentence(words)
    return jsonify({"words": words, "sentence": sentence})


@app.route("/api/generate-paragraph", methods=["POST"])
def api_generate_paragraph():
    payload = request.get_json(silent=True) or {}
    sentence = payload.get("sentence", "")
    paragraph = generate_paragraph(sentence)
    return jsonify({"sentence": sentence, "paragraph": paragraph})


@app.route("/api/word-predict", methods=["POST"])
def api_word_predict():
    if word_model is None:
        return jsonify({"word": "", "confidence": 0, "accepted": False, "error": WORD_MODEL_MISSING_MESSAGE}), 200
    if not word_labels:
        return jsonify({"word": "", "confidence": 0, "accepted": False, "error": "Word labels not found. Please train the model first."}), 200

    payload = request.get_json(silent=True) or {}
    sequence = payload.get("sequence")
    if not isinstance(sequence, list):
        return jsonify({"word": "", "confidence": 0, "accepted": False, "error": "Invalid sequence payload."}), 400

    result = predict_word_sign(sequence, word_model, word_labels, threshold=0.85)
    return jsonify(result), 200


@app.route("/api/training/words")
def api_training_words():
    return jsonify({"words": refresh_training_words_cache()})


@app.route("/api/practice/reference")
def api_practice_reference():
    word = sanitize_word_label(request.args.get("word"))
    if not word:
        return jsonify({"success": False, "error": "Word is required."}), 400

    sample_dir = Path(SIGN_DATA_DIR) / word
    if not sample_dir.exists():
        return jsonify({"success": False, "error": "Practice word not found."}), 404

    sample_files = sorted(sample_dir.glob("*.npy"))
    if not sample_files:
        return jsonify({"success": False, "error": "No practice samples found for this word."}), 404

    sample = np.load(sample_files[0])
    if sample.ndim != 2 or sample.shape[1] != 126:
        return jsonify({"success": False, "error": "Invalid practice sample format."}), 500

    frame_index = min(len(sample) // 2, len(sample) - 1)
    guide_landmarks = frame_to_hand_landmarks(sample[frame_index])
    return jsonify(
        {
            "success": True,
            "word": word,
            "frame_index": frame_index,
            "sequence_length": int(len(sample)),
            "guide_landmarks": guide_landmarks,
        }
    )


@app.route("/api/startup-state")
def api_startup_state():
    return jsonify(get_loaded_assets_state())


@app.route("/api/training/save-sequence", methods=["POST"])
def api_training_save_sequence():
    payload = request.get_json(silent=True) or {}
    word = sanitize_word_label(payload.get("word"))
    sequence = np.asarray(payload.get("sequence", []), dtype=np.float32)

    if not word:
        return jsonify({"success": False, "error": "Word label is required."}), 400
    if sequence.shape != (30, 126):
        return jsonify({"success": False, "error": "Invalid sequence shape. Expected 30 x 126."}), 400

    word_dir = Path(SIGN_DATA_DIR) / word
    word_dir.mkdir(parents=True, exist_ok=True)
    existing_indices = []
    for file_path in word_dir.glob("*.npy"):
        try:
            existing_indices.append(int(file_path.stem))
        except ValueError:
            continue
    next_index = max(existing_indices, default=-1) + 1
    output_path = word_dir / f"{next_index}.npy"
    np.save(output_path, sequence)

    return jsonify(
        {
            "success": True,
            "word": word,
            "sequence_index": next_index,
            "path": str(output_path),
            "sequences": len(list(word_dir.glob("*.npy"))),
        }
    )


@app.route("/api/training/train-word-model", methods=["POST"])
def api_training_train_word_model():
    global training_process
    if training_process is not None and training_process.poll() is None:
        return jsonify({"success": False, "message": "Training is already running."}), 409

    if not os.path.exists(TRAIN_SCRIPT_PATH):
        return jsonify({"success": False, "message": "Training script not found."}), 500

    training_process = subprocess.Popen(
        [sys.executable, TRAIN_SCRIPT_PATH],
        cwd=PROJECT_ROOT,
    )
    return jsonify({"success": True, "message": "Training started. Watch the backend terminal for progress if run manually."})


@app.route("/api/training/status")
def api_training_status():
    if training_process is None:
        return jsonify({"running": False, "returncode": None})
    return jsonify({"running": training_process.poll() is None, "returncode": training_process.poll()})


@app.route("/api/training/reload-word-model", methods=["POST"])
def api_training_reload_word_model():
    loaded = reload_word_assets()
    if not loaded:
        return jsonify({"success": False, "message": WORD_MODEL_MISSING_MESSAGE}), 404
    return jsonify({"success": True, "message": "Word model reloaded.", "labels": word_labels})


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or label_encoder is None or hand_landmarker is None:
        return jsonify(
            {
                "success": False,
                "prediction": "Model, labels, or landmarker not found",
                "confidence": 0,
            }
        ), 500

    payload = request.get_json(silent=True) or {}

    image_data = payload.get("image")
    if not image_data:
        return jsonify({"success": False, "prediction": "No image provided", "confidence": 0}), 400

    try:
        frame = decode_image(image_data)
        if frame is None:
            return jsonify({"success": False, "prediction": "Invalid image", "confidence": 0}), 400

        rgb_frame = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        with hand_lock:
            results = hand_landmarker.detect(mp_image)
            face_results = face_landmarker.detect(mp_image) if face_landmarker is not None else None

        if not results.hand_landmarks:
            hand_data = []
            hands_detected = 0
            left_hand_detected = False
            right_hand_detected = False
        else:
            hand_data = format_landmarks(results)
            hands_detected = len(results.hand_landmarks or [])
            left_hand_detected = any(
                (getattr(item[0], "category_name", "") == "Left")
                for item in (results.handedness or [])
                if item
            )
            right_hand_detected = any(
                (getattr(item[0], "category_name", "") == "Right")
                for item in (results.handedness or [])
                if item
            )

        if not results.hand_landmarks:
            hand_prediction = "No hand detected"
            confidence = 0
            label = hand_prediction
        else:
            features = extract_features(results)
            if features.shape[0] != 63:
                return jsonify({"success": False, "prediction": "Incomplete landmarks", "confidence": 0})

            prediction = model.predict(prepare_input(features), verbose=0)[0]
            predicted_index = int(np.argmax(prediction))
            confidence = round(float(np.max(prediction) * 100), 2)
            label = decode_label(predicted_index)
            top_predictions = format_top_letter_predictions(prediction)

        face_detected = bool(face_results and face_results.face_landmarks)
        face_landmarks = format_face_landmarks(face_results) if face_detected else []
        face_expression = detect_face_expression(face_results) if face_detected else ""
        stable_status = "Stable" if confidence >= 90 else "Unstable"
        word_features = extract_word_features(results)

        return jsonify(
            {
                "success": bool(results.hand_landmarks),
                "prediction": str(label),
                "confidence": confidence,
                "top_predictions": top_predictions if results.hand_landmarks else [],
                "hands_detected": hands_detected,
                "left_hand_detected": left_hand_detected,
                "right_hand_detected": right_hand_detected,
                "hand_landmarks": hand_data,
                "face_detected": face_detected,
                "face_expression": face_expression,
                "face_landmarks": face_landmarks,
                "stable_status": stable_status,
                "word_features": word_features,
            }
        )
    except Exception as exc:
        app.logger.exception("Prediction failed: %s", exc)
        return jsonify({"success": False, "prediction": "No hand detected", "confidence": 0}), 200


load_assets()
app.logger.info(
    "Loaded assets: %s",
    get_loaded_assets_state(),
)
if word_model is None and discover_training_words():
    app.logger.warning(
        "Training folders were found in backend/sign_data, but backend/models/asl_word_lstm_model.h5 is missing."
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=False)
