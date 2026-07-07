# SIGN ASL Project Report

## Title

SIGN ASL: Real-Time American Sign Language Recognition and Practice Platform

## Abstract

SIGN ASL is a web-based application that recognizes American Sign Language letters and trained word signs using webcam input. The system combines a React frontend, a Flask backend, MediaPipe landmark detection, TensorFlow/Keras models, and text-generation helpers. It provides live prediction, visual landmark overlays, practice feedback, sentence generation, paragraph generation, text-to-speech output, and custom word training support.

## Problem Statement

ASL learners often need immediate visual feedback while practicing signs. Static learning resources do not show whether the learner's hand pose is detected correctly, and many tools do not connect sign recognition to useful sentence output. This project addresses that gap by creating an interactive system that detects signs from a webcam and turns recognized letters or words into readable output.

## Objectives

- Detect ASL letters from live webcam images.
- Detect trained ASL words from hand landmark sequences.
- Display hand and face landmarks for visual feedback.
- Provide practice scoring for letter learning.
- Generate sentences and paragraphs from detected text.
- Support text-to-speech playback.
- Allow users to save custom word sequences and retrain the word model.

## Scope

The project focuses on real-time browser-based recognition for educational and demonstration use. It supports single-frame letter recognition and sequence-based word recognition. The system includes training utilities for custom words, but high-accuracy production recognition would require a larger and more balanced dataset.

## System Architecture

The system is divided into two main parts:

- Frontend: React + TypeScript application that manages the camera, overlays, prediction display, practice mode, word history, generated text, and speech output.
- Backend: Flask API that loads ML models, extracts landmarks, predicts letters and words, saves training data, and triggers retraining.

High-level flow:

1. The browser opens the webcam.
2. The frontend captures frames from the video stream.
3. Frames are sent to the Flask backend.
4. MediaPipe extracts hand and face landmarks.
5. TensorFlow/Keras models predict letters or words.
6. The backend returns prediction results, confidence, landmarks, and status.
7. The frontend updates the UI, overlays, sentence, paragraph, and speech output.

## Frontend Implementation

Main frontend files:

- `frontend/src/App.tsx` - central application state and screen layout
- `frontend/src/components/WebcamDetector.tsx` - webcam capture and prediction calls
- `frontend/src/components/DetectionOverlay.tsx` - landmark canvas rendering
- `frontend/src/components/CaptionBox.tsx` - live caption display
- `frontend/src/hooks/useStablePrediction.ts` - stabilizes repeated predictions
- `frontend/src/utils/api.ts` - backend API helper functions
- `frontend/src/utils/landmarkDrawing.ts` - drawing utilities

The frontend supports two recognition modes:

- Letters mode detects individual ASL letters and can build text letter by letter.
- Words mode collects a sequence of 30 frames and sends the landmark sequence for word prediction.

The frontend also includes help and ASL information pages, theme switching, practice controls, training controls, and generated text panels.

## Backend Implementation

Main backend files:

- `backend/app.py` - Flask app, routes, asset loading, and prediction flow
- `backend/utils/mediapipe_utils.py` - MediaPipe detector wrappers and landmark extraction
- `backend/utils/prediction_utils.py` - word model loading and prediction helpers
- `backend/utils/sentence_generator.py` - sentence and paragraph generation helpers
- `backend/training/train_word_model.py` - word model training script
- `backend/training/collect_word_data.py` - webcam data collection script

The backend loads:

- `backend/models/asl_model.h5` for ASL letter prediction
- `backend/models/labels.pkl` for letter labels
- `backend/models/asl_word_lstm_model.h5` for word prediction
- `backend/labels_words.txt` for word labels
- `backend/models/hand_landmarker.task` for hand landmarks
- `backend/models/face_landmarker.task` for face landmarks

The Flask development reloader is disabled so TensorFlow and MediaPipe do not initialize twice.

## API Summary

Important endpoints:

- `GET /api/health` - confirms backend status
- `GET /api/startup-state` - returns loaded model and training state
- `POST /predict` - predicts ASL letter and returns landmarks/status
- `POST /api/word-predict` - predicts word from a 30-frame sequence
- `POST /api/generate-sentence` - converts detected words into a sentence
- `POST /api/generate-paragraph` - expands a sentence into a paragraph
- `GET /api/training/words` - lists saved training words
- `POST /api/training/save-sequence` - saves one training sequence
- `POST /api/training/train-word-model` - starts word model training
- `GET /api/training/status` - checks training process state
- `POST /api/training/reload-word-model` - reloads trained word assets
- `GET /api/practice/reference` - returns reference landmarks for practice

## Data Format

Letter prediction uses one detected hand with 21 landmarks. Each landmark includes:

- `x`
- `y`
- `z`

Word prediction uses 30 frames. Each frame has 126 values:

- 63 values for the left hand
- 63 values for the right hand

Saved word samples are stored as `.npy` files in:

```text
backend/sign_data/<word>/
```

## Model and Training Flow

The word model is trained from saved landmark sequences. The training script reads all word folders, prepares labels, trains an LSTM classifier, and saves the updated model and label list. After training, the backend can reload the model without restarting the entire app.

Training steps:

1. Record or save a 30-frame sequence.
2. Store the sequence in `backend/sign_data/<word>/`.
3. Run `backend/training/train_word_model.py` or start training from the frontend.
4. Save the trained model to `backend/models/asl_word_lstm_model.h5`.
5. Reload the word model through the backend API.

## Testing and Verification

The following checks were used during development:

- Frontend production build:

```bash
npm run build
```

- Python syntax check:

```bash
backend/venv311/bin/python -m py_compile backend/app.py backend/utils/mediapipe_utils.py
```

- Backend runtime verification:

```text
GET /api/health
POST /predict
```

Expected result:

- Backend starts on `http://localhost:8000`.
- Frontend starts on `http://localhost:5173` or the next available Vite port.
- Prediction requests return HTTP `200`.
- Startup state shows loaded letter model, word model, labels, and landmarkers.

## Limitations

- Recognition quality depends on camera quality, lighting, hand position, and training data.
- Word recognition is limited to the words present in the trained dataset.
- Small custom datasets can reduce model accuracy.
- Browser camera permission is required.
- Real-time ML performance can vary by machine.

## Future Scope

- Add more ASL letters, words, and sentence examples.
- Expand the dataset with more signers and lighting conditions.
- Add user login and personal learning progress.
- Store practice history and export progress reports.
- Add backend unit tests and API integration tests.
- Improve grammar correction for generated sentences.

## Conclusion

SIGN ASL demonstrates how webcam input, MediaPipe landmarks, TensorFlow models, and a React interface can work together to create an interactive ASL learning platform. The project supports recognition, practice, training, and generated text output, making it suitable for classroom demonstration, learning support, and future extension.
