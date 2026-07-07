const DEFAULT_BASE_URL = "http://localhost:8000";

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || DEFAULT_BASE_URL).replace(/\/$/, "");

async function postJson(path, body) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data?.error || data?.message || "Request failed");
  }
  return data;
}

export function generateSentence(words) {
  return postJson("/api/generate-sentence", { words });
}

export function generateParagraph(sentence) {
  return postJson("/api/generate-paragraph", { sentence });
}

export function predictWord(sequence) {
  return postJson("/api/word-predict", { sequence });
}

export async function listTrainingWords() {
  const response = await fetch(`${API_BASE_URL}/api/training/words`);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data?.error || data?.message || "Request failed");
  }
  return data;
}

export function saveTrainingSequence(word, sequence) {
  return postJson("/api/training/save-sequence", { word, sequence });
}

export function startWordModelTraining() {
  return postJson("/api/training/train-word-model", {});
}

export function reloadWordModel() {
  return postJson("/api/training/reload-word-model", {});
}

export async function getPracticeReference(word) {
  const response = await fetch(`${API_BASE_URL}/api/practice/reference?word=${encodeURIComponent(word)}`);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data?.error || data?.message || "Request failed");
  }
  return data;
}

export async function getTrainingStatus() {
  const response = await fetch(`${API_BASE_URL}/api/training/status`);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data?.error || data?.message || "Request failed");
  }
  return data;
}

export async function getStartupState() {
  const response = await fetch(`${API_BASE_URL}/api/startup-state`);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data?.error || data?.message || "Request failed");
  }
  return data;
}
