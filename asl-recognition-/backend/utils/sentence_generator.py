from __future__ import annotations

from typing import Iterable, List


RULES = {
    ("i", "want", "water", "please"): "I want water, please.",
    ("i", "need", "help"): "I need help.",
    ("i", "want", "food"): "I want food.",
    ("thank_you",): "Thank you.",
    ("you", "go", "school"): "You are going to school.",
    ("i", "go", "home"): "I am going home.",
    ("doctor", "help"): "The doctor is helping.",
    ("yes",): "Yes.",
    ("no",): "No.",
    ("sorry",): "Sorry.",
}


def clean_detected_words(words: Iterable[str]) -> List[str]:
    cleaned: List[str] = []
    for word in words or []:
        token = str(word or "").strip().lower().replace(" ", "_")
        if token:
            cleaned.append(token)
    return cleaned


def _capitalize_sentence(sentence: str) -> str:
    sentence = " ".join(sentence.split()).strip()
    if not sentence:
        return ""
    if sentence[-1] not in ".!?":
        sentence += "."
    return sentence[:1].upper() + sentence[1:]


def generate_sentence(words: Iterable[str]) -> str:
    cleaned = clean_detected_words(words)
    if not cleaned:
        return ""

    key = tuple(cleaned)
    if key in RULES:
        return RULES[key]

    joined = " ".join(word.replace("_", " ") for word in cleaned)
    return _capitalize_sentence(joined)


def generate_paragraph(sentence: str) -> str:
    text = " ".join(str(sentence or "").split()).strip()
    if not text:
        return ""
    if text[-1] not in ".!?":
        text += "."
    return (
        f"The detected message is: {text} "
        "This indicates that the user may be requesting assistance or communicating a need through sign language."
    )
