from __future__ import annotations
from typing import Dict, List, Tuple
from transformers import pipeline


def build_emotion_classifier(model_name: str = "j-hartmann/emotion-english-distilroberta-base"):

    """
    Builds a text emotion classifier.
    NOTE: This model is English. If your interviews are in Spanish,
    you can still test, but for best results switch to a multilingual emotion model.

    Returns a transformers pipeline.
    """
    clf = pipeline(
        task="text-classification",
        model=model_name,
        top_k=None,       # return all labels with scores
        truncation=True,
    )
    return clf


def predict_text_emotions(classifier, text: str) -> Dict[str, float]:
    """
    Returns a dict {label: score}.
    """
    if not text or not text.strip():
        return {}

    out = classifier(text)
    # pipeline returns: List[List[{'label':..., 'score':...}, ...]] when top_k=None
    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
        items = out[0]
    elif isinstance(out, list):
        items = out
    else:
        items = []

    emotions: Dict[str, float] = {}
    for item in items:
        label = str(item.get("label"))
        score = float(item.get("score", 0.0))
        emotions[label] = score

    # Sort not needed for dict, but useful to keep deterministic printing if you later serialize.
    return emotions
