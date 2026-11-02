from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.preprocessing import preprocess_text
from src.train_model import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, load_or_train_model


def predict_single_text(
    text: str,
    model: Any | None = None,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    """Predict the sentiment and class probabilities for a single text input."""
    if text is None:
        raise ValueError("Text input cannot be empty.")
    if not isinstance(text, str):
        text = str(text)
    cleaned = preprocess_text(text)
    if not cleaned:
        raise ValueError("Text input cannot be empty or contain only punctuation.")

    if model is None:
        bundle = load_or_train_model(dataset_path=dataset_path, model_path=model_path)
        model = bundle["model"]

    probabilities = model.predict_proba([cleaned])[0]
    labels = list(model.classes_)
    index = int(np.argmax(probabilities))
    predicted = labels[index]
    confidence = float(probabilities[index])

    return {
        "sentiment": predicted,
        "confidence": confidence,
        "probabilities": {label: float(prob) for label, prob in zip(labels, probabilities)},
        "preprocessed_text": cleaned,
    }


def predict_batch_texts(
    texts: Iterable[str] | pd.DataFrame,
    model: Any | None = None,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> pd.DataFrame:
    """Predict sentiment labels for a list of texts and return a DataFrame of probabilities."""
    if isinstance(texts, pd.DataFrame):
        if "text" not in texts.columns:
            raise ValueError("The DataFrame must include a 'text' column for batch prediction.")
        texts = texts["text"].tolist()

    items = list(texts)
    if not items:
        raise ValueError("No text values were provided for batch analysis.")

    prepared = []
    for value in items:
        cleaned = preprocess_text(value)
        if not cleaned:
            prepared.append("")
        else:
            prepared.append(cleaned)

    if model is None:
        bundle = load_or_train_model(dataset_path=dataset_path, model_path=model_path)
        model = bundle["model"]

    probabilities = model.predict_proba(prepared)
    labels = list(model.classes_)
    rows = []

    for index, original_text in enumerate(items):
        class_probs = {label: float(value) for label, value in zip(labels, probabilities[index])}
        predicted_label = max(class_probs, key=class_probs.get)
        rows.append(
            {
                "text": str(original_text),
                "predicted_sentiment": predicted_label,
                "confidence": float(class_probs[predicted_label]),
                **{f"prob_{label}": class_probs[label] for label in labels},
            }
        )

    return pd.DataFrame(rows)


def validate_uploaded_csv(uploaded_file: Any) -> pd.DataFrame:
    """Validate uploaded CSV files and return a DataFrame with a clean 'text' column."""
    if uploaded_file is None:
        raise ValueError("No file was uploaded.")

    file_path = Path(getattr(uploaded_file, "name", "uploaded.csv"))
    if file_path.suffix.lower() != ".csv":
        raise ValueError("Only CSV files are supported for batch uploads.")

    try:
        if hasattr(uploaded_file, "read"):
            data = uploaded_file.read()
            if not data:
                raise ValueError("The uploaded CSV file is empty.")
            df = pd.read_csv(pd.io.common.BytesIO(data))
        else:
            df = pd.read_csv(file_path)
    except Exception as exc:  # pragma: no cover - defensive path
        raise ValueError(f"The uploaded CSV file could not be parsed: {exc}") from exc

    if df.empty:
        raise ValueError("The uploaded CSV file is empty.")
    if "text" not in df.columns:
        raise ValueError("The uploaded CSV file must contain a 'text' column.")

    cleaned = df[["text"]].copy()
    cleaned["text"] = cleaned["text"].fillna("").astype(str).str.strip()
    if (cleaned["text"] == "").any():
        raise ValueError("The uploaded CSV file contains one or more empty text rows.")
    return cleaned.reset_index(drop=True)
