import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import DetectionOverlay from "./DetectionOverlay";
import aslImage from "../../images/image.png";

const PRACTICE_GRID = {
  A: { row: 0, col: 0 },
  B: { row: 0, col: 1 },
  C: { row: 0, col: 2 },
  D: { row: 0, col: 3 },
  E: { row: 0, col: 4 },
  F: { row: 1, col: 0 },
  G: { row: 1, col: 1 },
  H: { row: 1, col: 2 },
  I: { row: 1, col: 3 },
  J: { row: 1, col: 4 },
  K: { row: 2, col: 0 },
  L: { row: 2, col: 1 },
  M: { row: 2, col: 2 },
  N: { row: 2, col: 3 },
  O: { row: 2, col: 4 },
  P: { row: 3, col: 0 },
  R: { row: 3, col: 1 },
  S: { row: 3, col: 2 },
  T: { row: 3, col: 3 },
  U: { row: 3, col: 4 },
  V: { row: 4, col: 0 },
  W: { row: 4, col: 1 },
  X: { row: 4, col: 2 },
  Y: { row: 4, col: 3 },
  Z: { row: 4, col: 4 },
};

function getPracticeBackgroundPosition(letter) {
  const entry = PRACTICE_GRID[String(letter || "").toUpperCase()] || PRACTICE_GRID.A;
  return `${entry.col * 25}% ${entry.row * 25}%`;
}

const WebcamDetector = forwardRef(function WebcamDetector({
  backendUrl,
  onPredictionUpdate,
  onCameraStatusChange,
  handLandmarks,
  faceLandmarks,
  faceRenderMode = "mesh",
  practiceLetter,
  captureInterval = 700,
}, ref) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const timerRef = useRef(null);
  const predictingRef = useRef(false);
  const hasRequestedRef = useRef(false);
  const onPredictionUpdateRef = useRef(onPredictionUpdate);
  const onCameraStatusChangeRef = useRef(onCameraStatusChange);
  const backendUrlRef = useRef(backendUrl);
  const captureIntervalRef = useRef(captureInterval);

  useEffect(() => {
    onPredictionUpdateRef.current = onPredictionUpdate;
  }, [onPredictionUpdate]);

  useEffect(() => {
    onCameraStatusChangeRef.current = onCameraStatusChange;
  }, [onCameraStatusChange]);

  useEffect(() => {
    backendUrlRef.current = backendUrl;
  }, [backendUrl]);

  useEffect(() => {
    captureIntervalRef.current = captureInterval;
  }, [captureInterval]);

  useEffect(() => {
    if (!hasRequestedRef.current) {
      hasRequestedRef.current = true;
      startCamera();
    }
    return () => {
      stopCamera();
    };
  }, []);

  useEffect(() => {
    if (!streamRef.current) return;
    restartLoop();
  }, [captureInterval]);

  const startCamera = async () => {
    if (streamRef.current) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      stream.getVideoTracks().forEach((track) => {
        track.onended = () => onCameraStatusChangeRef.current?.(false);
      });
      videoRef.current.srcObject = stream;
      onCameraStatusChangeRef.current?.(true);
      restartLoop();
    } catch (error) {
      onCameraStatusChangeRef.current?.(false);
      console.error(error);
    }
  };

  const stopCamera = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    onCameraStatusChangeRef.current?.(false);
  };

  const restartLoop = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    startLoop();
  };

  const startLoop = () => {
    if (timerRef.current) return;
    timerRef.current = setInterval(captureFrame, captureIntervalRef.current);
  };

  const captureFrame = async () => {
    if (predictingRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !video.videoWidth || !video.videoHeight) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const image = canvas.toDataURL("image/jpeg", 0.85);

    predictingRef.current = true;
    try {
      const response = await fetch(`${backendUrlRef.current}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image }),
      });
      const data = await response.json();
      onPredictionUpdateRef.current?.(data);
    } catch (error) {
      onPredictionUpdateRef.current?.({ success: false, prediction: "Backend error", confidence: 0 });
      console.error(error);
    } finally {
      predictingRef.current = false;
    }
  };

  useImperativeHandle(ref, () => ({
    startCamera,
    stopCamera,
  }));

  return (
      <div className="video-shell">
        <video ref={videoRef} autoPlay playsInline muted className="video-feed" />
        <DetectionOverlay
          videoRef={videoRef}
          data={{ hand_landmarks: handLandmarks, face_landmarks: faceLandmarks }}
          faceRenderMode={faceRenderMode}
        />
        {practiceLetter ? (
          <div className="practice-overlay">
            <div className="practice-ghost-card">
              <div
                className="practice-ghost-sheet"
                style={{
                  backgroundImage: `url(${aslImage})`,
                  backgroundPosition: getPracticeBackgroundPosition(practiceLetter),
                }}
              />
              <div className="practice-ghost-label">Trace {practiceLetter.toUpperCase()}</div>
            </div>
          </div>
        ) : null}
        <canvas ref={canvasRef} className="hidden-canvas" />
      </div>
  );
});

export default WebcamDetector;
