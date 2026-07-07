# API

Backend endpoints for SIGN ASL.

## Base URL

`http://localhost:8000`

The frontend defaults to this backend URL through `frontend/src/utils/api.ts`. If needed, override it with `VITE_API_BASE_URL`.

## `GET /`

Returns a simple running message.

### Response

```text
ASL Detection Backend Running
```

## `GET /api/health`

Health check for the backend.

### Response

```json
{
  "status": "ok",
  "message": "Backend is running"
}
```

## `POST /predict`

Predicts an ASL letter from a webcam image.

### Request

```json
{
  "image": "data:image/jpeg;base64,..."
}
```

### Response

```json
{
  "success": true,
  "prediction": "A",
  "confidence": 92.14,
  "stable_status": "Stable",
  "hands_detected": 1,
  "left_hand_detected": true,
  "right_hand_detected": false,
  "face_detected": true,
  "face_expression": "Neutral",
  "hand_landmarks": [],
  "face_landmarks": [],
  "word_features": []
}
```

## `POST /api/word-predict`

Predicts a word from a 30-frame sequence.

### Request

```json
{
  "sequence": [[0, 0, 0]]
}
```

The real payload should be a `30 x 126` sequence.

### Response

```json
{
  "word": "hello",
  "confidence": 0.91,
  "accepted": true
}
```

If the model is missing or the payload is invalid, the backend returns an `error` field.

## `POST /api/generate-sentence`

Builds a sentence from detected words.

### Request

```json
{
  "words": ["i", "want", "water"]
}
```

### Response

```json
{
  "words": ["i", "want", "water"],
  "sentence": "I want water."
}
```

## `POST /api/generate-paragraph`

Builds a short paragraph from a sentence.

### Request

```json
{
  "sentence": "I want water."
}
```

### Response

```json
{
  "sentence": "I want water.",
  "paragraph": "..."
}
```

## `GET /api/training/words`

Lists saved training words and sequence counts.

### Response

```json
{
  "words": [
    { "word": "hello", "sequences": 12 }
  ]
}
```

## `POST /api/training/save-sequence`

Saves one `30 x 126` training sequence for a word.

### Request

```json
{
  "word": "hello",
  "sequence": [[0, 0, 0]]
}
```

### Response

```json
{
  "success": true,
  "word": "hello",
  "sequence_index": 0,
  "path": "backend/sign_data/hello/0.npy",
  "sequences": 1
}
```

## `POST /api/training/train-word-model`

Starts word model training in the backend.

### Response

```json
{
  "success": true,
  "message": "Training started. Watch the backend terminal for progress if run manually."
}
```

## `GET /api/training/status`

Returns the training process state.

### Response

```json
{
  "running": false,
  "returncode": 0
}
```

## `POST /api/training/reload-word-model`

Reloads the trained word model from disk.

### Response

```json
{
  "success": true,
  "message": "Word model reloaded.",
  "labels": ["hello", "thank_you"]
}
```

## `GET /api/startup-state`

Returns loaded model and training state.

### Response

```json
{
  "letter_model_loaded": true,
  "letter_labels_loaded": true,
  "word_model_loaded": true,
  "word_labels_loaded": true,
  "hand_landmarker_loaded": true,
  "face_landmarker_loaded": true,
  "training_words": [],
  "word_labels": []
}
```

## `GET /api/practice/reference`

Returns a practice guide sample for a word.

### Query

`?word=hello`

### Response

```json
{
  "success": true,
  "word": "hello",
  "frame_index": 15,
  "sequence_length": 30,
  "guide_landmarks": []
}
```
