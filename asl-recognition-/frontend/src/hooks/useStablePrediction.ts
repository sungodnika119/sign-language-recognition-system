import { useEffect, useRef, useState } from "react";
import { getStableLabel, isValidASLPrediction, normalizePrediction } from "../utils/aslPrediction";

export default function useStablePrediction(response) {
  const [stablePrediction, setStablePrediction] = useState("");
  const historyRef = useRef([]);

  useEffect(() => {
    const rawPrediction = normalizePrediction(response?.prediction);
    if (!isValidASLPrediction(rawPrediction)) {
      historyRef.current = [];
      setStablePrediction("");
      return;
    }

    historyRef.current = [...historyRef.current, rawPrediction].slice(-5);
    const stableLabel = getStableLabel(response);
    const isStableByBackend = stableLabel === "Stable";
    const lastThree = historyRef.current.slice(-3);
    const isStableByHistory =
      lastThree.length === 3 && lastThree.every((item) => item === rawPrediction);

    if (isStableByBackend || isStableByHistory || Number(response?.confidence ?? 0) >= 90) {
      setStablePrediction(rawPrediction);
    }
  }, [response]);

  return stablePrediction;
}
