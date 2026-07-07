const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

const FACE_CONNECTIONS = [
  [10, 338], [338, 297], [297, 332], [332, 284], [284, 251], [251, 389], [389, 356], [356, 454],
  [454, 323], [323, 361], [361, 288], [288, 397], [397, 365], [365, 379], [379, 378], [378, 400],
  [400, 377], [377, 152], [152, 148], [148, 176], [176, 149], [149, 150], [150, 136], [136, 172],
  [172, 58], [58, 132], [132, 93], [93, 234], [234, 127], [127, 162], [162, 21], [21, 54],
  [54, 103], [103, 67], [67, 109], [109, 10],
  [33, 7], [7, 163], [163, 144], [144, 145], [145, 153], [153, 154], [154, 155], [155, 133],
  [33, 246], [246, 161], [161, 160], [160, 159], [159, 158], [158, 157], [157, 173], [173, 133],
  [362, 382], [382, 381], [381, 380], [380, 374], [374, 373], [373, 390], [390, 249], [249, 263],
  [263, 466], [466, 388], [388, 387], [387, 386], [386, 385], [385, 384], [384, 398], [398, 362],
  [61, 146], [146, 91], [91, 181], [181, 84], [84, 17], [17, 314], [314, 405], [405, 321], [321, 375],
  [375, 291], [291, 308], [308, 324], [324, 318], [318, 402], [402, 317], [317, 14], [14, 87],
  [87, 178], [178, 88], [88, 95], [95, 185], [185, 40], [40, 39], [39, 37], [37, 0], [0, 267],
  [267, 269], [269, 270], [270, 409], [409, 291],
];

function drawPoint(ctx, x, y, radius, color) {
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
}

function drawLine(ctx, x1, y1, x2, y2, color, width = 2) {
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}

function toCanvasPoint(point, width, height) {
  return {
    x: point.x * width,
    y: point.y * height,
  };
}

function drawPracticeFrame(ctx, width, height) {
  const boxWidth = width * 0.42;
  const boxHeight = height * 0.58;
  const left = (width - boxWidth) / 2;
  const top = (height - boxHeight) / 2;

  ctx.save();
  ctx.setLineDash([10, 8]);
  ctx.strokeStyle = "rgba(255, 208, 122, 0.75)";
  ctx.lineWidth = 3;
  ctx.strokeRect(left, top, boxWidth, boxHeight);
  ctx.fillStyle = "rgba(255, 208, 122, 0.08)";
  ctx.fillRect(left, top, boxWidth, boxHeight);
  ctx.setLineDash([]);
  ctx.font = "700 16px Inter, system-ui, sans-serif";
  ctx.fillStyle = "rgba(255, 239, 204, 0.95)";
  ctx.fillText("Trace here", left + 16, top + 26);
  ctx.restore();
}

function transformPracticeHands(hands, width, height) {
  const allPoints = [];
  hands.forEach((hand) => {
    (hand.landmarks || []).forEach((point) => {
      if (point) allPoints.push(point);
    });
  });

  if (!allPoints.length) {
    return hands;
  }

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;

  allPoints.forEach((point) => {
    minX = Math.min(minX, point.x);
    maxX = Math.max(maxX, point.x);
    minY = Math.min(minY, point.y);
    maxY = Math.max(maxY, point.y);
  });

  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const sourceWidth = Math.max(maxX - minX, 1e-3);
  const sourceHeight = Math.max(maxY - minY, 1e-3);

  const targetWidth = width * 0.22;
  const targetHeight = height * 0.32;
  const scale = Math.min(targetWidth / (sourceWidth * width), targetHeight / (sourceHeight * height));
  const targetCenterX = width * 0.5;
  const targetCenterY = height * 0.5;

  return hands.map((hand) => ({
    ...hand,
    landmarks: (hand.landmarks || []).map((point) => ({
      x: ((point.x - centerX) * scale) + (targetCenterX / width),
      y: ((point.y - centerY) * scale) + (targetCenterY / height),
      z: point.z,
    })),
  }));
}

function drawHandSet(ctx, hands, width, height, options = {}) {
  const alpha = options.alpha ?? 1;
  const handColors = options.handColors || ["#34ddff", "#9b7bff"];
  const labelColor = options.labelColor || "#ffffff";
  const pointRadius = options.pointRadius ?? 4;
  const lineWidth = options.lineWidth ?? 3;

  ctx.save();
  ctx.globalAlpha = alpha;

  hands.forEach((hand, handIndex) => {
    const landmarks = hand.landmarks || [];
    const handColor = handColors[handIndex % handColors.length];

    HAND_CONNECTIONS.forEach(([start, end]) => {
      const p1 = landmarks[start];
      const p2 = landmarks[end];
      if (!p1 || !p2) return;
      const c1 = toCanvasPoint(p1, width, height);
      const c2 = toCanvasPoint(p2, width, height);
      drawLine(ctx, c1.x, c1.y, c2.x, c2.y, handColor, lineWidth);
    });

    landmarks.forEach((point) => {
      const { x, y } = toCanvasPoint(point, width, height);
      drawPoint(ctx, x, y, pointRadius, handColor);
    });

    const wrist = landmarks[0];
    if (wrist && options.showLabels !== false) {
      const { x, y } = toCanvasPoint(wrist, width, height);
      ctx.font = "600 12px Inter, system-ui, sans-serif";
      ctx.fillStyle = labelColor;
      ctx.fillText(hand.handedness || `Hand ${handIndex + 1}`, x + 8, y - 8);
    }
  });

  ctx.restore();
}

export function drawLandmarksOverlay(ctx, overlayData, width, height, options = {}) {
  ctx.clearRect(0, 0, width, height);
  if (!overlayData) return;
  const faceRenderMode = options.faceRenderMode || "mesh";

  if (options.practiceGuide?.length) {
    drawPracticeFrame(ctx, width, height);
    drawHandSet(ctx, transformPracticeHands(options.practiceGuide, width, height), width, height, {
      alpha: 0.28,
      handColors: ["#ffb86b", "#ffd27a"],
      labelColor: "#ffefcc",
      pointRadius: 5,
      lineWidth: 4,
    });
  }

  const hands = overlayData.hand_landmarks || [];
  drawHandSet(ctx, hands, width, height, {
    alpha: 1,
    handColors: ["#34ddff", "#9b7bff"],
    labelColor: "#ffffff",
    pointRadius: 4,
    lineWidth: 3,
  });

  const face = overlayData.face_landmarks || [];
  if (faceRenderMode === "points") {
    face.forEach((point, index) => {
      const { x, y } = toCanvasPoint(point, width, height);
      drawPoint(ctx, x, y, index % 10 === 0 ? 2.4 : 1.1, "#ffe95f");
    });
  } else {
    FACE_CONNECTIONS.forEach(([start, end]) => {
      const p1 = face[start];
      const p2 = face[end];
      if (!p1 || !p2) return;
      const c1 = toCanvasPoint(p1, width, height);
      const c2 = toCanvasPoint(p2, width, height);
      drawLine(ctx, c1.x, c1.y, c2.x, c2.y, "#ffe95f", 1.1);
    });

    face.forEach((point, index) => {
      const { x, y } = toCanvasPoint(point, width, height);
      drawPoint(ctx, x, y, index % 12 === 0 ? 2.3 : 1.15, "#ffe95f");
    });
  }
}
