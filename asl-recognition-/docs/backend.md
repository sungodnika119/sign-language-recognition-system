# Backend

The backend is a Flask API that powers ASL detection, sentence generation, practice assets, and word-model training.

## Main features

- Loads the ASL letter model from `backend/models/asl_model.h5`
- Loads the word LSTM model from `backend/models/asl_word_lstm_model.h5`
- Uses MediaPipe hand and face landmark models
- Serves webcam prediction endpoints for letters and words
- Builds sentences and paragraphs from detected words
- Saves new word sequences into `backend/sign_data`
- Triggers training for the word model
- Reloads trained word assets without restarting the app
- Exposes practice reference data for letter tracing

## Main files

- `backend/app.py` - Flask app and API routes
- `backend/utils/prediction_utils.py` - word prediction helpers
- `backend/utils/mediapipe_utils.py` - landmark extraction helpers
- `backend/utils/sentence_generator.py` - sentence and paragraph generation
- `backend/training/train_word_model.py` - trains the LSTM word model
- `backend/training/collect_word_data.py` - collects custom training sequences

## Important routes

- `GET /` - basic server message
- `GET /api/health` - health check
- `GET /api/startup-state` - model and training status
- `POST /predict` - letter and hand/face prediction
- `POST /api/word-predict` - word prediction from a 30-frame sequence
- `POST /api/generate-sentence` - convert word list to sentence
- `POST /api/generate-paragraph` - expand sentence into paragraph
- `GET /api/training/words` - list saved training words
- `POST /api/training/save-sequence` - save one word sequence
- `POST /api/training/train-word-model` - start model training
- `GET /api/training/status` - check training progress
- `POST /api/training/reload-word-model` - reload the new model
- `GET /api/practice/reference` - fetch practice reference data

## Data flow

1. Frontend sends a webcam frame to `POST /predict`.
2. Backend runs MediaPipe landmark extraction.
3. The letter model predicts a letter.
4. For words, the frontend collects 30 frames and sends them to `POST /api/word-predict`.
5. The backend returns the predicted word and confidence.
6. Sentence and paragraph endpoints turn the detected text into readable output.

## Training flow

1. User records a new word sequence in the frontend.
2. Frontend saves the sequence through `POST /api/training/save-sequence`.
3. Backend stores `.npy` frames under `backend/sign_data/<word>/`.
4. Training can be started with `POST /api/training/train-word-model`.
5. After training, `POST /api/training/reload-word-model` loads the new model.

