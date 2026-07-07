# Training

This project supports collecting and training custom ASL word data.

## Data collection

Use `backend/training/collect_word_data.py` to record sign sequences from the webcam.

### What it does

- Opens the webcam
- Tracks hand landmarks with MediaPipe
- Records 30 frames per sequence
- Saves each sequence as `.npy`
- Stores data in `backend/sign_data/<word>/`

### Controls

- `q` - quit
- `s` - skip the current sequence

## Data format

Each saved sample is a NumPy array shaped like:

- `30` frames
- `126` values per frame

That means a single training sample contains two hands worth of flattened landmark data.

## Training the model

Use `backend/training/train_word_model.py` to train the LSTM word model.

### What it does

- Loads `.npy` sequences from `backend/sign_data`
- Builds an LSTM classifier
- Saves the model to `backend/models/asl_word_lstm_model.h5`
- Saves labels to `backend/labels_words.txt`

### Training notes

- The script includes fallback logic for small datasets
- It uses early stopping when the dataset is large enough
- Extra words discovered in `backend/sign_data` are included automatically

## Retraining flow

1. Record a new word sequence in the frontend or collector script.
2. Save it into the word folder.
3. Run the training script or trigger training from the app.
4. Reload the updated model in the backend.

## Live testing

Use `backend/training/test_word_model.py` to test the trained model live from the webcam.

### Controls

- `q` - quit
- `c` - clear sentence
- `Backspace` - remove last word

