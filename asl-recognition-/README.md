# SIGN ASL

SIGN ASL is a real-time American Sign Language recognition platform. It recognizes ASL letters and trained word signs from webcam input, displays hand and face landmarks, supports practice feedback, builds sentences and paragraphs, and can speak generated text with browser text-to-speech.

## Project Purpose

Many ASL learning tools focus only on static images or isolated alphabets. This project combines live camera input, landmark extraction, machine learning prediction, practice scoring, and sentence generation in one interactive application. The system is designed for students, beginners, and demonstrators who want immediate feedback while practicing ASL.

## Main Features

- Live ASL letter detection from webcam frames
- Word sign detection from 30-frame hand landmark sequences
- Hand and face landmark overlays
- Practice mode with target letters, confidence feedback, points, and stars
- Sentence and paragraph generation from detected text
- Browser speech output for generated sentences
- Training support for custom word signs
- Backend startup-state reporting for model and training readiness

## Tech Stack

- Frontend: React, TypeScript, Vite
- Backend: Flask, Flask-CORS
- Machine learning: TensorFlow, Keras
- Landmark detection: MediaPipe hand and face landmark models
- Data processing: NumPy, OpenCV
- Camera: Browser `getUserMedia`
- Voice output: Browser `SpeechSynthesis`

## Project Structure

```text
asl-recognition-/
  backend/
    app.py                    Flask API and prediction routes
    models/                   Letter model, word model, labels, MediaPipe tasks
    sign_data/                Saved custom word training sequences
    training/                 Data collection and model training scripts
    utils/                    Prediction, landmark, and sentence helpers
  frontend/
    src/                      React components, hooks, and API helpers
    public/                   Static frontend assets
    package.json              Frontend scripts and dependencies
  docs/                       API, backend, frontend, training, and report docs
```

## Quick Start

Run the backend first, then the frontend.

### Backend

On macOS or Linux:

```bash
cd asl-recogonition-
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

If the repository already contains the prepared Python environment, this also works from the project root:

```bash
backend/venv311/bin/python backend/app.py
```

Backend URL:

```text
http://localhost:8000
```

Health check:

```text
http://localhost:8000/api/health
```

### Frontend

```bash
cd asl-recognition-
cd frontend
npm install
npm run dev
```

Frontend URL:

```text
http://localhost:5173
```

If port `5173` is already in use, Vite automatically chooses the next available port, such as:

```text
http://localhost:5174
```

## Configuration

The frontend API base URL defaults to `http://localhost:8000`. To override it, create a frontend environment file:

```env
VITE_API_BASE_URL=http://localhost:8000
```

## Important Notes

- Keep the backend running before using webcam prediction in the frontend.
- On macOS, MediaPipe may need permission to access native graphics or ML resources when started from restricted environments.
- The Flask reloader is disabled in `backend/app.py` so TensorFlow and MediaPipe do not initialize twice during development.
- If `backend/models/asl_word_lstm_model.h5` is missing, train the word model before using word recognition.
- Saved word samples are stored under `backend/sign_data/<word>/` as NumPy `.npy` files.

## Verification

Useful checks before presenting or submitting the project:

```bash
cd frontend
npm run build
```

```bash
backend/venv311/bin/python -m py_compile backend/app.py backend/utils/mediapipe_utils.py
```

Expected runtime behavior:

- Backend loads the letter model, word model, labels, and MediaPipe task models.
- `GET /api/health` returns backend status.
- Frontend opens the webcam and sends frames to `POST /predict`.
- Backend prediction responses return HTTP `200`.

## Documentation

- [docs/project-report.md](docs/project-report.md) - project report overview
- [docs/README.md](docs/README.md) - documentation index
- [docs/api.md](docs/api.md) - backend endpoints with request and response examples
- [docs/backend.md](docs/backend.md) - backend modules, routes, and data flow
- [docs/frontend.md](docs/frontend.md) - frontend structure and UI behavior
- [docs/practice.md](docs/practice.md) - practice mode flow and scoring
- [docs/structure.md](docs/structure.md) - program flow and diagram
- [docs/training.md](docs/training.md) - data collection, saving, and retraining flow

## Future Enhancements

- Add more trained ASL words and larger datasets
- Improve sentence generation with grammar-aware language processing
- Add user accounts and progress history
- Export practice reports for learners
- Add automated backend endpoint tests
