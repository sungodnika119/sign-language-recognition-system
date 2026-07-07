from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "backend" / "models"
MODEL_PATH = MODEL_DIR / "asl_model.h5"
LABELS_PATH = MODEL_DIR / "labels.pkl"
HAND_TASK_PATH = MODEL_DIR / "hand_landmarker.task"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
NUMERIC_LABELS = {
    **{str(index): letter for index, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")},
    "26": "del",
    "27": "space",
}
DEFAULT_IGNORED_DIRS = {
    "__MACOSX",
    ".ipynb_checkpoints",
    "asl_alphabet_test",
    "test",
    "valid",
    "validation",
}


def normalize_label(value: str) -> str:
    label = value.strip()
    if label in NUMERIC_LABELS:
        return NUMERIC_LABELS[label]
    aliases = {
        "delete": "del",
        "nothing": "nothing",
        "space": "space",
    }
    return aliases.get(label.lower(), label.upper() if len(label) == 1 else label.lower())


def build_model(num_classes: int):
    model = Sequential(
        [
            Conv1D(64, kernel_size=3, activation="relu", input_shape=(63, 1)),
            MaxPooling1D(pool_size=2),
            Conv1D(128, kernel_size=3, activation="relu"),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.35),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def iter_class_dirs(dataset_dir: Path):
    class_dirs = []
    for path in dataset_dir.rglob("*"):
        if not path.is_dir() or path.name in DEFAULT_IGNORED_DIRS:
            continue
        if any(child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS for child in path.iterdir()):
            class_dirs.append(path)
    return sorted(class_dirs, key=lambda item: str(item).lower())


def iter_images(class_dir: Path, max_per_class: int | None):
    images = [
        path
        for path in sorted(class_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return images[:max_per_class] if max_per_class else images


def create_landmarker():
    if not HAND_TASK_PATH.exists():
        raise FileNotFoundError(f"Missing MediaPipe hand model: {HAND_TASK_PATH}")
    options = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(
            model_asset_path=str(HAND_TASK_PATH),
            delegate=mp_tasks.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.45,
        min_hand_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_letter_features(image_path: Path, landmarker) -> np.ndarray | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), dtype=np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = landmarker.detect(mp_image)
    if not results.hand_landmarks:
        return None

    landmarks = results.hand_landmarks[0]
    points = np.array([[point.x, point.y, point.z] for point in landmarks], dtype=np.float32)

    handedness = "Right"
    if results.handedness and results.handedness[0]:
        handedness = getattr(results.handedness[0][0], "category_name", "Right")

    if handedness.lower() == "left":
        points[:, 0] = 1.0 - points[:, 0]

    wrist = points[0].copy()
    points = points - wrist

    scale = np.max(np.linalg.norm(points[:, :2], axis=1))
    if scale > 1e-6:
        points = points / scale

    points = np.clip(points, -2.0, 2.0)
    return points.flatten().astype(np.float32)


def load_dataset(dataset_dir: Path, max_per_class: int | None, include_extra_classes: bool):
    features = []
    labels = []
    skipped = 0
    class_dirs = iter_class_dirs(dataset_dir)

    if not class_dirs:
        raise RuntimeError(f"No image class folders found under {dataset_dir}")

    landmarker = create_landmarker()
    try:
        for class_dir in class_dirs:
            label = normalize_label(class_dir.name)
            if not include_extra_classes and len(label) != 1:
                print(f"{label}: skipped non-letter class")
                continue
            image_paths = iter_images(class_dir, max_per_class)
            loaded_for_label = 0
            for image_path in image_paths:
                sample = extract_letter_features(image_path, landmarker)
                if sample is None:
                    skipped += 1
                    continue
                features.append(sample)
                labels.append(label)
                loaded_for_label += 1
            print(f"{label}: {loaded_for_label}/{len(image_paths)} usable images")
    finally:
        landmarker.close()

    print(f"Skipped images without a detected hand: {skipped}")
    return np.array(features, dtype=np.float32), np.array(labels)


def train(
    dataset_dir: Path,
    max_per_class: int | None,
    epochs: int,
    batch_size: int,
    include_extra_classes: bool,
):
    X, y = load_dataset(dataset_dir, max_per_class, include_extra_classes)
    if len(X) == 0:
        raise RuntimeError("No usable training samples found after MediaPipe landmark extraction.")

    label_encoder = LabelEncoder()
    y_indices = label_encoder.fit_transform(y)
    y_encoded = to_categorical(y_indices, num_classes=len(label_encoder.classes_))
    X = X.reshape(-1, 63, 1)

    class_counts = np.bincount(y_indices, minlength=len(label_encoder.classes_))
    if np.min(class_counts) < 2:
        raise RuntimeError("Every class needs at least 2 usable images for training and validation.")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_indices,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model = build_model(len(label_encoder.classes_))
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ModelCheckpoint(str(MODEL_PATH), monitor="val_accuracy", save_best_only=True),
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    with LABELS_PATH.open("wb") as file:
        pickle.dump(label_encoder, file)

    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Classes: {', '.join(label_encoder.classes_)}")
    print(f"Samples: {len(X)}")
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Last epoch accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved labels to: {LABELS_PATH}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the ASL letter model from a Kaggle-style image folder dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to the downloaded Kaggle dataset folder containing class subfolders.",
    )
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--include-extra-classes",
        action="store_true",
        help="Include non-letter Kaggle classes such as del/space. By default only A-Z are trained.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train(
        args.dataset_dir.expanduser().resolve(),
        args.max_per_class,
        args.epochs,
        args.batch_size,
        args.include_extra_classes,
    )


if __name__ == "__main__":
    main()
