# Frontend

The frontend is a React + TypeScript app that handles the camera UI, prediction display, practice mode, training controls, and text building.

## Main features

- Live webcam preview with landmark overlay
- Letter and word recognition modes
- Practice mode for letters only
- Points and star feedback in practice
- Auto caption and sentence building
- Text-to-speech output
- Word training and model reload controls
- Light/dark theme switching
- Info and help pages inside the app

## Main files

- `frontend/src/App.tsx` - full app state and page layout
- `frontend/src/App.css` - all app styling
- `frontend/src/main.tsx` - React entry point
- `frontend/src/components/WebcamDetector.tsx` - camera capture and backend calls
- `frontend/src/components/DetectionOverlay.tsx` - hand and face landmark canvas
- `frontend/src/components/CaptionBox.tsx` - live caption panel
- `frontend/src/hooks/useStablePrediction.ts` - stabilizes repeated predictions
- `frontend/src/utils/api.ts` - API helpers
- `frontend/src/utils/aslPrediction.ts` - prediction validation helpers
- `frontend/src/utils/landmarkDrawing.ts` - overlay drawing logic

## App layout

- Left side:
  - camera panel
  - live caption box
- Right side:
  - prediction card
  - hands / face status card
  - detected words card
  - generated text card
  - paragraph card

## Recognition modes

- `letters` mode:
  - detects single letters
  - adds letters into the sentence
  - enables practice mode

- `words` mode:
  - detects common signs / words
  - builds a word history
  - generates a sentence from detected words

## Practice mode

- Only available for letters
- Shows a ghost hand pose on the camera
- User traces the pose
- The app shows:
  - result text
  - points
  - stars
- User can move to the next letter or skip manually

## Component notes

### `WebcamDetector`

- Opens the browser camera
- Captures frames at an interval
- Sends images to the backend
- Renders the pose reference overlay in practice mode

### `DetectionOverlay`

- Draws landmarks on a canvas over the video
- Supports hand landmarks and face landmarks

### `CaptionBox`

- Shows the current generated caption
- Changes state when speech playback is active

## State management

`App.tsx` keeps most of the UI state in one place:

- camera status
- prediction text
- confidence
- hand and face detection flags
- sentence text
- detected words
- practice state
- training state

This makes the screen responsive, but the file is large because it owns the whole app flow.
