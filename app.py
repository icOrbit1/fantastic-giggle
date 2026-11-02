from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import pandas as pd

from src.train_model import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, load_or_train_model


def predict_single_text(text: str, model: Any | None = None) -> Dict[str, Any]:
    """Return sentiment prediction, confidence, and class probabilities for one text example."""
    if text is None or not isinstance(text, str):
        text = str(text) if text is not None else ""
    if not text or not text.strip():
        raise ValueError("Text input cannot be empty.")

    if model is None:
        model_bundle = load_or_train_model(DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH)
        model = model_bundle["model"]

    probabilities = model.predict_proba([text])[0]
    class_labels = model.classes_
    label_index = int(np.argmax(probabilities))
    predicted_label = class_labels[label_index]
    confidence = float(probabilities[label_index])
    return {
        "sentiment": predicted_label,
        "confidence": confidence,
        "probabilities": {label: float(prob) for label, prob in zip(class_labels, probabilities)},
    }


def predict_batch_texts(texts: Iterable[str], model: Any | None = None) -> pd.DataFrame:
    """Predict sentiment for multiple texts and return enriched results."""
    texts = list(texts)
    if not texts:
        raise ValueError("No text rows were provided for batch analysis.")

    if model is None:
        model_bundle = load_or_train_model(DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH)
        model = model_bundle["model"]

    probabilities = model.predict_proba(texts)
    class_labels = list(model.classes_)
    output_rows = []
    for index, text in enumerate(texts):
        prob_map = {label: float(value) for label, value in zip(class_labels, probabilities[index])}
        predicted_label = max(prob_map, key=prob_map.get)
        output_rows.append(
            {
                "text": text,
                "predicted_sentiment": predicted_label,
                "confidence": float(prob_map[predicted_label]),
                **{f"prob_{label}": prob_map[label] for label in class_labels},
            }
        )
    return pd.DataFrame(output_rows)


def validate_uploaded_csv(upload_path: str | Path) -> pd.DataFrame:
    """Validate uploaded CSVs for batch analysis and return a DataFrame with text values."""
    path = Path(upload_path)
    if not path.exists():
        raise FileNotFoundError(f"Uploaded file not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError("Only CSV files are allowed for batch analysis.")

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive path
        raise ValueError(f"Could not parse the uploaded CSV file: {exc}") from exc

    if df.empty:
        raise ValueError("Uploaded CSV is empty.")
    if "text" not in df.columns:
        raise ValueError("Uploaded CSV must contain a 'text' column.")

    cleaned = df.copy()
    cleaned["text"] = cleaned["text"].fillna("").astype(str).str.strip()
    if (cleaned["text"] == "").any():
        raise ValueError("Uploaded CSV contains one or more empty text values.")
    return cleaned[["text"]].reset_index(drop=True)
