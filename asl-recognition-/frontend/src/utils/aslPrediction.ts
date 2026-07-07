export function normalizePrediction(value) {
  return String(value ?? "").trim();
}

export function isValidASLPrediction(value) {
  const prediction = normalizePrediction(value);
  if (!prediction) return false;
  if (["No hand detected", "Error", "Backend error"].includes(prediction)) return false;
  if (/^error/i.test(prediction)) return false;
  return /^[A-Za-z]$/.test(prediction);
}

export function getStableLabel(response) {
  return normalizePrediction(response?.stable_status || response?.stableStatus || "");
}

