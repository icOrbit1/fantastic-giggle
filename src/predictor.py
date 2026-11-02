from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.preprocessing import preprocess_text
from src.train_model import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, load_or_train_model


def predict_single_text(
    text: str,
    model: Any | None = None,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    """Predict sentiment for a single text input."""
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
    max_index = int(probabilities.argmax())
    predicted = labels[max_index]
    confidence = float(probabilities[max_index])

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
    """Predict sentiment for a list or DataFrame of text values."""
    if isinstance(texts, pd.DataFrame):
        if "text" not in texts.columns:
            raise ValueError("Batch prediction requires a 'text' column.")
        texts = texts["text"].tolist()

    values = list(texts)
    if not values:
        raise ValueError("No text values were provided for batch analysis.")

    processed = [preprocess_text(value) for value in values]
    if model is None:
        bundle = load_or_train_model(dataset_path=dataset_path, model_path=model_path)
        model = bundle["model"]

    probabilities = model.predict_proba(processed)
    labels = list(model.classes_)
    rows = []
    for original_text, probs in zip(values, probabilities):
        prob_map = {label: float(value) for label, value in zip(labels, probs)}
        predicted = max(prob_map, key=prob_map.get)
        rows.append(
            {
                "text": str(original_text),
                "predicted_sentiment": predicted,
                "confidence": float(prob_map[predicted]),
                **{f"prob_{label}": prob_map[label] for label in labels},
            }
        )

    return pd.DataFrame(rows)


def validate_uploaded_csv(uploaded_file: Any) -> pd.DataFrame:
    """Validate uploaded CSV files and return a DataFrame with a clean 'text' column."""
    if uploaded_file is None:
        raise ValueError("No file was uploaded.")

    if getattr(uploaded_file, "size", 0) and uploaded_file.size > 5 * 1024 * 1024:
        raise ValueError("Uploaded files must be 5 MB or smaller.")

    try:
        if hasattr(uploaded_file, "read"):
            file_bytes = uploaded_file.read()
            if not file_bytes:
                raise ValueError("The uploaded CSV file is empty.")
            df = pd.read_csv(pd.io.common.BytesIO(file_bytes))
        else:
            file_path = Path(uploaded_file)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
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
