import { useEffect, useRef } from "react";
import { drawLandmarksOverlay } from "../utils/landmarkDrawing";

export default function DetectionOverlay({ videoRef, data, faceRenderMode }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef?.current;
    if (!canvas || !video) return;

    const draw = () => {
      const width = video.videoWidth;
      const height = video.videoHeight;
      if (!width || !height) return;

      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      drawLandmarksOverlay(ctx, data, width, height, { faceRenderMode });
    };

    draw();
  }, [data, faceRenderMode, videoRef]);

  return <canvas ref={canvasRef} className="overlay-canvas" />;
}
