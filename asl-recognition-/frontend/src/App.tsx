import { useEffect, useMemo, useRef, useState } from "react";
import WebcamDetector from "./components/WebcamDetector";
import CaptionBox from "./components/CaptionBox";
import useStablePrediction from "./hooks/useStablePrediction";
import AslInfoPage from "./pages/AslInfoPage";
import HowToUsePage from "./pages/HowToUsePage";
import { isValidASLPrediction, normalizePrediction } from "./utils/aslPrediction";
import {
  FiCopy,
  FiHelpCircle,
  FiMic,
  FiMoon,
  FiImage,
  FiPlay,
  FiRefreshCw,
  FiSave,
  FiSmile,
  FiSun,
  FiTrash2,
  FiZap,
} from "react-icons/fi";
import { FaStar } from "react-icons/fa";
import {
  API_BASE_URL,
  generateParagraph,
  generateSentence,
  listTrainingWords,
  getTrainingStatus,
  getStartupState,
  predictWord,
  reloadWordModel,
  saveTrainingSequence,
  startWordModelTraining,
} from "./utils/api";

const BACKEND_URL = API_BASE_URL;
const ZERO_HAND = Array.from({ length: 63 }, () => 0);
const PRACTICE_LETTERS = "ABCDEFGHIJKLMNOPRSTUVWXYZ".split("");

function getPracticeStars(score) {
  if (score >= 90) return 5;
  if (score >= 80) return 4;
  if (score >= 70) return 3;
  if (score >= 60) return 2;
  if (score > 0) return 1;
  return 0;
}

function getNextPracticeLetter(current) {
  const pool = PRACTICE_LETTERS.filter((letter) => letter !== current);
  return pool[Math.floor(Math.random() * pool.length)] || current;
}

function getInitialTheme() {
  try {
    const saved = window.localStorage.getItem("asl-theme");
    return saved === "light" ? "light" : "dark";
  } catch {
    return "dark";
  }
}

function isValidWordPrediction(value) {
  const prediction = normalizePrediction(value);
  if (!prediction) return false;
  if (["No hand detected", "Error", "Backend error", "Idle", "Waiting for hand"].includes(prediction)) return false;
  if (/^error/i.test(prediction)) return false;
  return true;
}

function buildWordFeaturesFromPrediction(result) {
  const hands = Array.isArray(result?.hand_landmarks) ? result.hand_landmarks : [];
  const left = [...ZERO_HAND];
  const right = [...ZERO_HAND];

  hands.forEach((hand, index) => {
    const handedness = normalizePrediction(hand?.handedness).toLowerCase();
    const landmarks = Array.isArray(hand?.landmarks) ? hand.landmarks : [];
    const flat = [];
    landmarks.forEach((point) => {
      flat.push(Number(point?.x ?? 0), Number(point?.y ?? 0), Number(point?.z ?? 0));
    });
    while (flat.length < 63) {
      flat.push(0);
    }
    const values = flat.slice(0, 63);

    if (handedness === "left") {
      for (let i = 0; i < 63; i += 1) left[i] = values[i];
    } else if (handedness === "right") {
      for (let i = 0; i < 63; i += 1) right[i] = values[i];
    } else if (index === 0) {
      for (let i = 0; i < 63; i += 1) left[i] = values[i];
    } else {
      for (let i = 0; i < 63; i += 1) right[i] = values[i];
    }
  });

  return [...left, ...right];
}

export default function App() {
  const webcamRef = useRef(null);
  const lastAutoCaptionRef = useRef("");
  const lastAutoCaptionAtRef = useRef(0);
  const lastWordRef = useRef("");
  const lastWordAtRef = useRef(0);
  const wordPredictingRef = useRef(false);
  const [theme, setTheme] = useState(getInitialTheme);
  const [showInfoPage, setShowInfoPage] = useState(false);
  const [showHowToUsePage, setShowHowToUsePage] = useState(false);
  const [activeTab, setActiveTab] = useState("detect");
  const [recognitionMode, setRecognitionMode] = useState("letters");
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [currentPrediction, setCurrentPrediction] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [detectionStatus, setDetectionStatus] = useState("Idle");
  const [handsDetected, setHandsDetected] = useState(0);
  const [leftHandDetected, setLeftHandDetected] = useState(false);
  const [rightHandDetected, setRightHandDetected] = useState(false);
  const [faceDetected, setFaceDetected] = useState(false);
  const [faceExpression, setFaceExpression] = useState("");
  const [faceRenderMode, setFaceRenderMode] = useState("mesh");
  const [handLandmarks, setHandLandmarks] = useState([]);
  const [faceLandmarks, setFaceLandmarks] = useState([]);
  const [topPredictions, setTopPredictions] = useState([]);
  const [stableStatus, setStableStatus] = useState("Unstable");
  const [sentence, setSentence] = useState("");
  const [detectedWords, setDetectedWords] = useState([]);
  const [generatedParagraph, setGeneratedParagraph] = useState("");
  const [wordSequence, setWordSequence] = useState([]);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [confidenceHistory, setConfidenceHistory] = useState([]);
  const [isGeneratingParagraph, setIsGeneratingParagraph] = useState(false);
  const [trainingWord, setTrainingWord] = useState("");
  const [trainingFrames, setTrainingFrames] = useState([]);
  const [isRecordingTraining, setIsRecordingTraining] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState("Ready to record a new sign.");
  const [trainingWords, setTrainingWords] = useState([]);
  const [isTrainingModel, setIsTrainingModel] = useState(false);
  const [trainingToast, setTrainingToast] = useState("");
  const [practiceLetter, setPracticeLetter] = useState("A");
  const [practiceResult, setPracticeResult] = useState("Ready to practice");
  const [practicePoints, setPracticePoints] = useState(0);
  const [practiceStars, setPracticeStars] = useState(0);
  const [practiceCompleted, setPracticeCompleted] = useState(false);
  const [startupState, setStartupState] = useState({
    letter_model_loaded: false,
    word_model_loaded: false,
    word_labels_loaded: false,
  });
  const trainingToastTimerRef = useRef(null);
  const trainingPollRef = useRef(null);
  const trainingSeenRunningRef = useRef(false);
  const practiceResolvedRef = useRef(false);
  const stablePrediction = useStablePrediction({
    prediction: currentPrediction,
    confidence,
    stable_status: stableStatus,
  });
  const estimatedFps = 30;
  const hasDetectedLetter = recognitionMode === "letters" && isValidASLPrediction(currentPrediction);
  const hasDetectedWord = recognitionMode === "words" && isValidWordPrediction(currentPrediction);
  const detectedLetter = recognitionMode === "words"
    ? (hasDetectedWord ? "SIGN" : "-")
    : (hasDetectedLetter ? currentPrediction.toUpperCase() : "-");
  const detectedLabel = recognitionMode === "words"
    ? (currentPrediction || "Waiting for word")
    : (currentPrediction || "Waiting for hand");
  const averageConfidence =
    confidenceHistory.length > 0
      ? confidenceHistory.reduce((sum, value) => sum + value, 0) / confidenceHistory.length
      : confidence;

  const caption = useMemo(() => sentence, [sentence]);
  const statusLabel = isSpeaking ? "Speaking..." : detectionStatus;
  const isPracticeMode = recognitionMode === "letters" && activeTab === "practice";
  const cameraFooterLabel = isPracticeMode ? practiceResult : statusLabel;
  const practiceStarIcons = Array.from({ length: 5 }, (_, index) => index < practiceStars);
  const normalizedTrainingWord = normalizePrediction(trainingWord).toLowerCase().replace(/\s+/g, "_");
  const detectedLabelStateClass = /no hand detected|waiting for hand|idle|error/i.test(detectedLabel) ? "status-bad" : "status-good";
  const handsStateClass = handsDetected > 0 ? "status-good" : "status-bad";
  const leftStateClass = leftHandDetected ? "status-good" : "status-bad";
  const rightStateClass = rightHandDetected ? "status-good" : "status-bad";
  const faceStateClass = faceDetected ? "status-good" : "status-bad";

  const showTrainingToast = (message) => {
    setTrainingToast(message);
    if (trainingToastTimerRef.current) {
      clearTimeout(trainingToastTimerRef.current);
    }
    trainingToastTimerRef.current = setTimeout(() => {
      setTrainingToast("");
    }, 3000);
  };

  useEffect(() => {
    if (!isSpeaking) {
      return undefined;
    }
    return () => window.speechSynthesis.cancel();
  }, [isSpeaking]);

  useEffect(() => {
    try {
      window.localStorage.setItem("asl-theme", theme);
    } catch {
      // ignore storage failures
    }
    document.documentElement.style.colorScheme = theme;
    document.documentElement.style.background = theme === "dark" ? "#030405" : "#ece3d6";
    document.body.style.background = theme === "dark" ? "#030405" : "#ece3d6";
    document.body.style.color = theme === "dark" ? "#f5f7fb" : "#282219";
  }, [theme]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const response = await listTrainingWords();
        if (!cancelled) {
          setTrainingWords(Array.isArray(response?.words) ? response.words : []);
        }
      } catch (error) {
        console.error(error);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const response = await getStartupState();
        if (cancelled) return;
        setStartupState({
          letter_model_loaded: Boolean(response?.letter_model_loaded),
          word_model_loaded: Boolean(response?.word_model_loaded),
          word_labels_loaded: Boolean(response?.word_labels_loaded),
        });
        if (Array.isArray(response?.training_words)) {
          setTrainingWords(response.training_words);
        }
      } catch (error) {
        console.error(error);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (recognitionMode !== "letters" && activeTab === "practice") {
      setActiveTab("detect");
    }
  }, [recognitionMode, activeTab]);

  useEffect(() => {
    if (activeTab === "practice" && recognitionMode === "letters" && !practiceLetter) {
      setPracticeLetter("A");
    }
  }, [activeTab, recognitionMode, practiceLetter]);

  useEffect(() => {
    practiceResolvedRef.current = false;
    setPracticePoints(0);
    setPracticeStars(0);
    setPracticeCompleted(false);
    setPracticeResult(`Trace letter ${practiceLetter}`);
  }, [practiceLetter, isPracticeMode]);

  useEffect(() => {
    setWordSequence([]);
    lastWordRef.current = "";
    lastWordAtRef.current = 0;
    setDetectionStatus(recognitionMode === "words" ? "Waiting for word sequence" : "Idle");
    if (recognitionMode === "letters") {
      setGeneratedParagraph("");
    }
  }, [recognitionMode]);

  useEffect(() => {
    if (isPracticeMode) {
      setDetectionStatus(`Practicing letter ${practiceLetter}`);
      setCurrentPrediction("");
      setConfidence(0);
      setConfidenceHistory([]);
    }
  }, [isPracticeMode, practiceLetter]);

  useEffect(() => {
    if (activeTab !== "train") {
      setIsRecordingTraining(false);
      setTrainingFrames([]);
      setTrainingStatus("Ready to record a new sign.");
    }
  }, [activeTab]);

  useEffect(() => {
    return () => {
      if (trainingToastTimerRef.current) {
        clearTimeout(trainingToastTimerRef.current);
      }
      if (trainingPollRef.current) {
        clearInterval(trainingPollRef.current);
      }
    };
  }, []);

  useEffect(() => {
    const letter = stablePrediction || currentPrediction;
    const now = Date.now();

    if (recognitionMode !== "letters" || isPracticeMode) {
      lastAutoCaptionRef.current = "";
      return;
    }

    if (!isCameraOn || !isValidASLPrediction(letter) || confidence < 80) {
      lastAutoCaptionRef.current = "";
      return;
    }

    if (letter === lastAutoCaptionRef.current) return;
    if (now - lastAutoCaptionAtRef.current < 1500) return;

    appendLetter(letter);
    lastAutoCaptionRef.current = letter;
    lastAutoCaptionAtRef.current = now;
  }, [stablePrediction, currentPrediction, confidence, isCameraOn, recognitionMode, isPracticeMode]);

  useEffect(() => {
    if (!isPracticeMode) return;

    const target = normalizePrediction(practiceLetter).toUpperCase();
    const candidate = normalizePrediction(stablePrediction).toUpperCase();
    const score = Number(confidence ?? 0);

    if (!isCameraOn || practiceResolvedRef.current || !candidate || !isValidASLPrediction(candidate)) {
      setPracticeResult(`Try again - trace ${target}`);
      return;
    }

    practiceResolvedRef.current = true;
    setPracticeCompleted(true);
    const stars = getPracticeStars(score);
    setPracticePoints(Math.round(score));
    setPracticeStars(stars);

    if (score < 60) {
      setPracticeResult(`Try again - ${candidate} (${score.toFixed(0)}%)`);
    } else if (candidate === target) {
      setPracticeResult(`Correct - ${candidate} (${score.toFixed(0)}%)`);
    } else {
      setPracticeResult(`Wrong - ${candidate} (${score.toFixed(0)}%)`);
    }
  }, [isPracticeMode, practiceLetter, stablePrediction, confidence, isCameraOn]);

  const advancePracticeLetter = (skipCurrent = false) => {
    const target = normalizePrediction(practiceLetter).toUpperCase();
    const next = getNextPracticeLetter(target);
    practiceResolvedRef.current = false;
    setPracticeCompleted(false);
    setPracticeLetter(next);
    setPracticePoints(0);
    setPracticeStars(0);
    setCurrentPrediction("");
    setConfidence(0);
    setPracticeResult(skipCurrent ? `Skipped - trace letter ${next}` : `Trace letter ${next}`);
  };

  useEffect(() => {
    if (recognitionMode !== "words" || wordSequence.length < 30) {
      return undefined;
    }

    if (wordPredictingRef.current) {
      return undefined;
    }

    let cancelled = false;
    wordPredictingRef.current = true;

    (async () => {
      try {
        const result = await predictWord(wordSequence);
        if (cancelled) return;
        const word = normalizePrediction(result?.word);
        const confidenceValue = Number(result?.confidence ?? 0);
        const displayConfidence = confidenceValue <= 1 ? confidenceValue * 100 : confidenceValue;
        const accepted = Boolean(result?.accepted);

        if (word && accepted) {
          setCurrentPrediction(word);
        } else if (!word) {
          setCurrentPrediction("No hand detected");
        }
        setConfidence(displayConfidence);
        setDetectionStatus(accepted ? "Word detected" : "Building sequence");
        setStableStatus(accepted ? "Stable" : "Unstable");
        setConfidenceHistory((prev) => [...prev, displayConfidence].slice(-20));

        if (
          accepted &&
          isValidWordPrediction(word) &&
          word !== lastWordRef.current &&
          Date.now() - lastWordAtRef.current > 1200
        ) {
          appendWord(word);
          lastWordRef.current = word;
          lastWordAtRef.current = Date.now();
        }
      } catch (error) {
        console.error(error);
        if (!cancelled) {
          setDetectionStatus(error?.message || "Word model not available");
          setStableStatus("Unstable");
        }
      } finally {
        wordPredictingRef.current = false;
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [wordSequence, recognitionMode]);

  const appendLetter = (letter) => {
    if (!isValidASLPrediction(letter)) return;
    setSentence((prev) => prev + letter.toUpperCase());
  };

  const appendWord = (word) => {
    const token = normalizePrediction(word).toLowerCase();
    if (!isValidWordPrediction(token)) return;
    setDetectedWords((prev) => {
      const last = prev[prev.length - 1];
      if (last === token) {
        return prev;
      }
      return [...prev, token];
    });
  };

  const addSpace = () => setSentence((prev) => (prev.endsWith(" ") ? prev : `${prev} `));
  const deleteLetter = () => setSentence((prev) => prev.slice(0, -1));
  const clearSentence = () => setSentence("");
  const clearWordSentence = () => {
    setDetectedWords([]);
    setSentence("");
    setGeneratedParagraph("");
    setWordSequence([]);
    lastWordRef.current = "";
    lastWordAtRef.current = 0;
  };

  const startTrainingSession = () => {
    const label = normalizedTrainingWord;
    if (!label) {
      setTrainingStatus("Enter a sign name first.");
      return;
    }
    setTrainingFrames([]);
    setIsRecordingTraining(true);
    setTrainingStatus(`Recording ${label}... show the sign steadily.`);
  };

  const stopTrainingSession = () => {
    setIsRecordingTraining(false);
    setTrainingStatus("Recording stopped.");
  };

  const clearTrainingSession = () => {
    setTrainingFrames([]);
    setIsRecordingTraining(false);
    setTrainingStatus("Ready to record a new sign.");
  };

  const saveTrainingSession = async () => {
    const label = normalizedTrainingWord;
    if (!label) {
      setTrainingStatus("Enter a sign name first.");
      return;
    }
    if (trainingFrames.length !== 30) {
      setTrainingStatus("Need exactly 30 frames before saving.");
      return;
    }
    try {
      setTrainingStatus("Saving sequence...");
      await saveTrainingSequence(label, trainingFrames);
      const response = await listTrainingWords();
      setTrainingWords(Array.isArray(response?.words) ? response.words : []);
      setTrainingFrames([]);
      setTrainingStatus(`Saved sequence for ${label}. Starting training...`);
      await trainWordModel();
    } catch (error) {
      console.error(error);
      setTrainingStatus(error?.message || "Failed to save sequence.");
    }
  };

  const trainWordModel = async () => {
    try {
      setIsTrainingModel(true);
      trainingSeenRunningRef.current = false;
      setTrainingStatus("Training the word model in the backend...");
      await startWordModelTraining();
      setTrainingStatus("Training started in the backend.");
      if (trainingPollRef.current) {
        clearInterval(trainingPollRef.current);
      }
      trainingPollRef.current = setInterval(async () => {
        try {
          const status = await getTrainingStatus();
          if (status?.running) {
            trainingSeenRunningRef.current = true;
            setTrainingStatus("Training in progress...");
            return;
          }
          if (trainingSeenRunningRef.current || status?.returncode !== null) {
            clearInterval(trainingPollRef.current);
            trainingPollRef.current = null;
            const ok = Number(status?.returncode ?? 1) === 0;
            if (ok) {
              try {
                await reloadWordModel();
                const message = "Word model trained and reloaded.";
                setTrainingStatus(message);
                showTrainingToast(message);
              } catch (reloadError) {
                console.error(reloadError);
                const message = "Training completed, but reload failed.";
                setTrainingStatus(message);
                showTrainingToast(message);
              }
            } else {
              const message = "Word model training finished with an error.";
              setTrainingStatus(message);
              showTrainingToast(message);
            }
            setIsTrainingModel(false);
          }
        } catch (error) {
          console.error(error);
        }
      }, 3000);
    } catch (error) {
      console.error(error);
      setTrainingStatus(error?.message || "Unable to start training.");
      setIsTrainingModel(false);
    } finally {
      // keep button loading while poll is active
    }
  };

  const reloadTrainedModel = async () => {
    try {
      setTrainingStatus("Reloading trained model...");
      await reloadWordModel();
      setTrainingStatus("Word model reloaded.");
      showTrainingToast("Word model reloaded.");
    } catch (error) {
      console.error(error);
      setTrainingStatus(error?.message || "Unable to reload model.");
    }
  };

  const copyText = async () => {
    const text = generatedParagraph || sentence;
    if (!text) return;
    await navigator.clipboard.writeText(text);
  };

  const handleGenerateParagraph = async () => {
    if (!sentence.trim()) return;
    setIsGeneratingParagraph(true);
    try {
      const response = await generateParagraph(sentence);
      setGeneratedParagraph(response?.paragraph || "");
    } catch (error) {
      console.error(error);
      setGeneratedParagraph(
        `The detected message is: ${sentence}. This indicates that the user may be requesting assistance or communicating a need through sign language.`
      );
    } finally {
      setIsGeneratingParagraph(false);
    }
  };

  const speakSentence = () => {
    const text = sentence.trim();
    if (!text) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.95;
    utterance.pitch = 1;
    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => {
      setIsSpeaking(false);
    };
    utterance.onerror = () => {
      setIsSpeaking(false);
    };
    window.speechSynthesis.speak(utterance);
  };

  const handlePrediction = (result) => {
    const prediction = normalizePrediction(result?.prediction);
    const score = Number(result?.confidence ?? 0);
    const success = Boolean(result?.success);
    const wordFeatures = Array.isArray(result?.word_features) && result.word_features.length === 126
      ? result.word_features
      : buildWordFeaturesFromPrediction(result);
    setHandsDetected(Number(result?.hands_detected ?? 0));
    setLeftHandDetected(Boolean(result?.left_hand_detected));
    setRightHandDetected(Boolean(result?.right_hand_detected));
    setFaceDetected(Boolean(result?.face_detected));
    setFaceExpression(normalizePrediction(result?.face_expression));
    setHandLandmarks(Array.isArray(result?.hand_landmarks) ? result.hand_landmarks : []);
    setFaceLandmarks(Array.isArray(result?.face_landmarks) ? result.face_landmarks : []);
    setTopPredictions(Array.isArray(result?.top_predictions) ? result.top_predictions : []);
    setStableStatus(normalizePrediction(result?.stable_status) || "Unstable");

    if (isPracticeMode) {
      setCurrentPrediction(prediction);
      setConfidence(score);
      setDetectionStatus(success ? `Checking ${practiceLetter}` : "Show your hand");
      setConfidenceHistory((prev) => [...prev, score].slice(-20));
      return;
    }

    if (activeTab === "train" && isRecordingTraining) {
      if (wordFeatures.length === 126) {
        setTrainingFrames((prev) => {
          const next = [...prev, wordFeatures].slice(-30);
          if (next.length === 30) {
            setTrainingStatus("30 frames captured. Save the sequence.");
          } else {
            setTrainingStatus(`Recording ${normalizedTrainingWord || "sign"}... ${next.length}/30 frames`);
          }
          return next;
        });
      }
      return;
    }

    if (recognitionMode === "words") {
      if (wordFeatures.length === 126) {
        setWordSequence((prev) => [...prev, wordFeatures].slice(-30));
      }
      setDetectionStatus(success ? "Collecting word sequence" : "No hand detected");
      if (!success) {
        setCurrentPrediction("No hand detected");
        setConfidence(0);
      }
      return;
    }

    setCurrentPrediction(prediction);
    setConfidence(score);
    setDetectionStatus(result?.stable_status || (success ? "Hand detected" : prediction || "Idle"));
    setConfidenceHistory((prev) => [...prev, score].slice(-20));
  };

  useEffect(() => {
    if (recognitionMode !== "words") return undefined;
    if (detectedWords.length === 0) {
      setSentence("");
      return undefined;
    }

    let cancelled = false;
    (async () => {
      try {
        const response = await generateSentence(detectedWords);
        if (!cancelled) {
          setSentence(response?.sentence || detectedWords.join(" "));
        }
      } catch (error) {
        console.error(error);
        if (!cancelled) {
          setSentence(detectedWords.join(" ").replace(/_/g, " "));
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [detectedWords, recognitionMode]);

  return (
    <div className={showInfoPage ? "app info-mode" : "app"} data-theme={theme}>
      <header className="topbar">
        <div className="brand-block">
          <div>
            <h1>American Sign Language Detection System</h1>
            <p className="subtitle">Real-time ASL alphabet detection with sentence building and voice output</p>
            <div className="header-controls">
              <div className="mode-switch">
                <span className="mode-label">Recognition Mode</span>
                <div className="segmented-control mode-control">
                  <button
                    className={recognitionMode === "letters" ? "segment active" : "segment"}
                    type="button"
                    onClick={() => setRecognitionMode("letters")}
                  >
                    Letters
                  </button>
                  <button
                    className={recognitionMode === "words" ? "segment active" : "segment"}
                    type="button"
                    onClick={() => setRecognitionMode("words")}
                  >
                    Common Signs / Words
                  </button>
                </div>
              </div>
            <div className="segmented-control view-tabs">
              <button className={activeTab === "detect" ? "segment active" : "segment"} type="button" onClick={() => setActiveTab("detect")}>
                Detect
              </button>
              <button className={activeTab === "train" ? "segment active" : "segment"} type="button" onClick={() => setActiveTab("train")}>
                Train
              </button>
              {recognitionMode === "letters" ? (
                <button className={activeTab === "practice" ? "segment active" : "segment"} type="button" onClick={() => setActiveTab("practice")}>
                  Practice
                </button>
              ) : null}
            </div>
            </div>
          </div>
        </div>
        <div className="topbar-actions">
          <button
            className="page-link info-link"
            type="button"
            onClick={() => setShowInfoPage((current) => !current)}
          >
            <FiImage />
            {showInfoPage ? "Back to App" : "ASL Info Page"}
          </button>
          <button
            className="page-link info-link"
            type="button"
            onClick={() => setShowHowToUsePage((current) => !current)}
          >
            <FiHelpCircle />
            {showHowToUsePage ? "Back to App" : "How to Use"}
          </button>
          <button
            className="theme-toggle"
            type="button"
            onClick={() => setTheme((current) => (current === "dark" ? "light" : "dark"))}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          >
            {theme === "dark" ? <FiSun /> : <FiMoon />}
            {theme === "dark" ? "Light Mode" : "Dark Mode"}
          </button>
        </div>
      </header>

      {showInfoPage ? (
        <div
          className="modal-backdrop"
          role="presentation"
          onClick={() => setShowInfoPage(false)}
        >
          <div
            className="modal-shell"
            role="dialog"
            aria-modal="true"
            aria-label="ASL info page"
            onClick={(event) => event.stopPropagation()}
          >
            <AslInfoPage onClose={() => setShowInfoPage(false)} />
          </div>
        </div>
      ) : (
        <div className="page-shell">
          <main className="dashboard-grid">
        <aside className="left-rail">
          <section className="glass-card camera-panel">
            <div className="camera-header">
              <span className={isCameraOn ? "live-badge online" : "live-badge offline"}>
                <i /> {isCameraOn ? "LIVE" : "OFFLINE"}
              </span>
              <button
                className={isCameraOn ? "camera-toggle on" : "camera-toggle off"}
                type="button"
                onClick={() => {
                  if (isCameraOn) {
                    webcamRef.current?.stopCamera();
                  } else {
                    webcamRef.current?.startCamera();
                  }
                }}
              >
                <i />
                {isCameraOn ? "Camera ON" : "Camera OFF"}
              </button>
            </div>
            <WebcamDetector
              ref={webcamRef}
              backendUrl={BACKEND_URL}
              onPredictionUpdate={handlePrediction}
              onCameraStatusChange={setIsCameraOn}
              handLandmarks={handLandmarks}
              faceLandmarks={faceLandmarks}
              faceRenderMode={faceRenderMode}
              practiceLetter={isPracticeMode ? practiceLetter : ""}
              captureInterval={activeTab === "train" || isPracticeMode ? 120 : 700}
            />
            <div className="camera-footer-chip">{cameraFooterLabel}</div>
          </section>

          <CaptionBox caption={caption} isSpeaking={isSpeaking} />
        </aside>

        <section className="right-stack">
          {isPracticeMode ? (
            <section className="glass-card practice-card">
              <div className="card-head">
                <div>
                  <h2>Practice Letter</h2>
                  <p>Pick a letter and trace the ghost hand pose on the camera feed.</p>
                </div>
                <div className="tiny-pill">Letter guide</div>
              </div>
              <div className="practice-result">{practiceResult}</div>
              <div className="practice-scoreboard">
                <div className="practice-score-chip">
                  <span>Points</span>
                  <strong>{practicePoints}</strong>
                </div>
                <div className="practice-score-chip">
                  <span>Stars</span>
                  <strong className="practice-stars" aria-label={`${practiceStars} out of 5 stars`}>
                    {practiceStarIcons.map((filled, index) => (
                      <FaStar
                        key={index}
                        className={filled ? "practice-star filled" : "practice-star empty"}
                        aria-hidden="true"
                      />
                    ))}
                  </strong>
                </div>
              </div>
              <div className="practice-controls">
                <div className="action-row">
                  <button
                    className="btn success"
                    type="button"
                    onClick={() => advancePracticeLetter(false)}
                    disabled={!practiceCompleted}
                  >
                    Next Letter
                  </button>
                  <button
                    className="btn secondary"
                    type="button"
                    onClick={() => advancePracticeLetter(true)}
                  >
                    Skip
                  </button>
                </div>
                <div className="practice-tip">
                  Keep your hand inside the ghost pose on the camera screen and match the shape as closely as you can.
                </div>
              </div>
            </section>
          ) : activeTab === "train" ? (
            <section className="glass-card train-card">
              <div className="card-head">
                <div>
                  <h2>Train New Word</h2>
                  <p>Capture 30-frame sequences and save them to the backend sign data folder.</p>
                </div>
                <div className="tiny-pill">{trainingFrames.length}/30</div>
              </div>
              <div className="train-controls">
                <input
                  className="train-input"
                  type="text"
                  value={trainingWord}
                  onChange={(event) => setTrainingWord(event.target.value)}
                  placeholder="Enter new word or sign name"
                />
                <div className="action-row train-actions">
                  <button className="btn primary" type="button" onClick={startTrainingSession}>
                    <FiPlay />
                    Start Record
                  </button>
                  <button className="btn secondary" type="button" onClick={stopTrainingSession}>
                    Stop Record
                  </button>
                  <button className="btn success" type="button" onClick={saveTrainingSession} disabled={trainingFrames.length !== 30}>
                    <FiSave />
                    Save Sequence
                  </button>
                  <button className="btn secondary" type="button" onClick={clearTrainingSession}>
                    Clear
                  </button>
                  <button className="btn secondary" type="button" onClick={trainWordModel} disabled={isTrainingModel}>
                    {isTrainingModel ? "Training..." : "Train Model"}
                  </button>
                  <button className="btn secondary" type="button" onClick={reloadTrainedModel}>
                    Reload Model
                  </button>
                </div>
              </div>
              <div className="training-status">{trainingStatus}</div>
              <div className="training-summary">
                {trainingWords.length > 0 ? trainingWords.map((item) => (
                  <div className="training-summary-row" key={item.word}>
                    <span>{item.word}</span>
                    <strong>{item.sequences}</strong>
                  </div>
                )) : <span className="empty-copy">No saved training words yet</span>}
              </div>
            </section>
          ) : (
            <>
              <section className="glass-card prediction-card-shell">
                <div className="card-head">
                  <div>
                    <h2>Prediction</h2>
                    <p>{recognitionMode === "letters" ? "Live letter classification from MediaPipe landmarks" : "Live common sign classification from an LSTM sequence"}</p>
                  </div>
                </div>
                <div className="prediction-hero">
                  <div className={recognitionMode === "words" ? "prediction-letter word-symbol" : "prediction-letter"}>
                    {detectedLetter}
                  </div>
                  <div className="prediction-detail">
                    <div>
                      <span>{recognitionMode === "words" ? "Detected Sign" : hasDetectedLetter ? "Detected Letter" : "Detection Status"}</span>
                      <strong className={`detected-label ${detectedLabelStateClass}`}>{detectedLabel}</strong>
                    </div>
                    <div>
                      <span>Confidence Score</span>
                      <strong className="confidence-readout">{Number(confidence).toFixed(2)}%</strong>
                    </div>
                    <div className="confidence-bar">
                      <i style={{ width: `${Math.max(0, Math.min(100, confidence))}%` }} />
                    </div>
                    <div className="prediction-tags">
                      <span className="confidence-chip">Stable: {stablePrediction || "-"}</span>
                      <span className="confidence-chip muted-chip">{stableStatus || "Unstable"}</span>
                    </div>
                    {recognitionMode === "letters" && topPredictions.length > 0 ? (
                      <div className="top-guesses" aria-label="Top letter guesses">
                        {topPredictions.map((item, index) => (
                          <span className={index === 0 ? "guess-chip best" : "guess-chip"} key={`${item.label}-${index}`}>
                            {item.label}: {Number(item.confidence ?? 0).toFixed(1)}%
                          </span>
                        ))}
                      </div>
                    ) : null}
                  </div>
                </div>
              </section>

              <section className="glass-card compact-card">
                <div className="compact-row">
                  <div>
                    <span className="compact-label">Hands</span>
                    <strong className={handsStateClass}>{handsDetected}</strong>
                  </div>
                  <div>
                    <span className="compact-label">Left</span>
                    <strong className={leftStateClass}>{leftHandDetected ? "Yes" : "No"}</strong>
                  </div>
                  <div>
                    <span className="compact-label">Right</span>
                    <strong className={rightStateClass}>{rightHandDetected ? "Yes" : "No"}</strong>
                  </div>
                  <div>
                    <span className="compact-label">Face</span>
                    <strong className={faceStateClass}>{faceDetected ? faceExpression || "Yes" : "No"}</strong>
                  </div>
                </div>
                <div className="segmented-control">
                  <button
                    className={faceRenderMode === "points" ? "segment active" : "segment"}
                    onClick={() => setFaceRenderMode("points")}
                  >
                    <FiSmile />
                    Face Points
                  </button>
                  <button
                    className={faceRenderMode === "mesh" ? "segment active" : "segment"}
                    onClick={() => setFaceRenderMode("mesh")}
                  >
                    <FiZap />
                    Face Mesh
                  </button>
                </div>
              </section>

              <section className="glass-card words-card">
                <div className="card-head">
                  <div>
                    <h2>Detected Words</h2>
                    <p>Live word history used for sentence formation</p>
                  </div>
                  <div className="tiny-pill">{detectedWords.length} words</div>
                </div>
                <div className="word-chips">
                  {detectedWords.length > 0 ? (
                    detectedWords.map((word, index) => (
                      <span className="word-chip" key={`${word}-${index}`}>
                        {word.replace(/_/g, " ")}
                      </span>
                    ))
                  ) : (
                    <span className="empty-copy">No words detected yet</span>
                  )}
                </div>
              </section>

              <section className="glass-card text-card">
                <div className="card-head">
                  <div>
                    <h2>Generated Text</h2>
                    <p>{recognitionMode === "letters" ? "Build words and sentences from detected letters" : "Build words and sentences from detected signs"}</p>
                  </div>
                  <div className="tiny-pill">{sentence.length} chars</div>
                </div>
                <div className={`text-output ${isSpeaking ? "speaking-text" : ""}`}>{sentence || "Your generated text will appear here..."}</div>
                <div className="action-row">
                  <button
                    className="btn primary"
                    type="button"
                    onClick={() => {
                      if (recognitionMode === "letters") {
                        const chosen = stablePrediction || currentPrediction;
                        if (isValidASLPrediction(chosen) && confidence >= 70) appendLetter(chosen);
                      } else if (isValidWordPrediction(currentPrediction) && confidence >= 85) {
                        appendWord(currentPrediction);
                      }
                    }}
                  >
                    <FiPlay />
                    {recognitionMode === "letters" ? "Add Letter" : "Add Word"}
                  </button>
                  <button className="btn secondary" type="button" onClick={addSpace}>Space</button>
                  <button className="btn danger" type="button" onClick={recognitionMode === "letters" ? deleteLetter : () => setDetectedWords((prev) => prev.slice(0, -1))}>
                    <FiTrash2 />
                    Delete
                  </button>
                  <button className="btn secondary" type="button" onClick={recognitionMode === "letters" ? clearSentence : clearWordSentence}>
                    <FiRefreshCw />
                    Clear
                  </button>
                  <button className="btn secondary" type="button" onClick={copyText}>
                    <FiCopy />
                    Copy
                  </button>
                  <button className="btn success" type="button" onClick={speakSentence}>
                    <FiMic />
                    Speak
                  </button>
                  <button className="btn secondary" type="button" onClick={handleGenerateParagraph} disabled={isGeneratingParagraph}>
                    <FiZap />
                    {isGeneratingParagraph ? "Generating..." : "Generate Paragraph"}
                  </button>
                </div>
              </section>

              <section className="glass-card paragraph-card">
                <div className="card-head">
                  <div>
                    <h2>Generated Paragraph</h2>
                    <p>Short paragraph generated from the sentence</p>
                  </div>
                </div>
                <div className="paragraph-output">
                  {generatedParagraph || "Click Generate Paragraph to expand the sentence into a short paragraph."}
                </div>
              </section>
            </>
          )}
        </section>
          </main>
        </div>
      )}

      {showHowToUsePage ? (
        <div
          className="modal-backdrop"
          role="presentation"
          onClick={() => setShowHowToUsePage(false)}
        >
          <div
            className="modal-shell"
            role="dialog"
            aria-modal="true"
            aria-label="How to use page"
            onClick={(event) => event.stopPropagation()}
          >
            <HowToUsePage onClose={() => setShowHowToUsePage(false)} />
          </div>
        </div>
      ) : null}

      {trainingToast ? <div className="toast-message">{trainingToast}</div> : null}

      {!showInfoPage && !showHowToUsePage ? (
        <section className="status-bar">
          <div><span>System Status</span><strong className="good">Active</strong></div>
          <div>
            <span>Model Status</span>
            <strong className={startupState.word_model_loaded || startupState.letter_model_loaded ? "good" : "danger"}>
              {startupState.word_model_loaded || startupState.letter_model_loaded ? "Loaded" : "Not Ready"}
            </strong>
          </div>
          <div><span>Avg Confidence</span><strong className="good">{averageConfidence.toFixed(1)}%</strong></div>
          <div><span>Connection</span><strong className="good">Good</strong></div>
          <div><span>FPS</span><strong className="accent">{estimatedFps}</strong></div>
        </section>
      ) : null}
    </div>
  );
}
