# Practice

Practice mode helps users trace and learn ASL letters.

## Availability

- Practice mode is available only for letters
- It is hidden when recognition mode is set to words

## Flow

1. User opens the practice tab.
2. The app shows a single target letter.
3. A ghost hand guide appears on the camera screen.
4. The user traces the shape with their hand.
5. The backend predicts the hand sign.
6. The app compares the prediction with the target letter.
7. The result is shown as:
   - Correct
   - Wrong
   - Try again
8. The app shows points and stars based on confidence.
9. The user moves to the next letter or skips it.

## Scoring

- Points come from the prediction confidence
- Stars are mapped from confidence ranges

### Star mapping

- `90+` -> 5 stars
- `80-89` -> 4 stars
- `70-79` -> 3 stars
- `60-69` -> 2 stars
- `1-59` -> 1 star
- `0` -> 0 stars

## UI elements

- Target letter label
- Ghost pose overlay
- Result message
- Points card
- Stars card
- `Next Letter` button
- `Skip` button

## Notes

- The practice guide is shown on the camera feed
- The feedback is based on the stable prediction and confidence
- The next letter only appears after the current one is resolved or skipped

