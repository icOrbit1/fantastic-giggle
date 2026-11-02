from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.evaluation import compute_metrics
from src.preprocessing import preprocess_text

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
DEFAULT_DATASET_PATH = DATA_DIR / "sample_sentiment.csv"
DEFAULT_MODEL_PATH = MODELS_DIR / "sentiment_model.joblib"
DEFAULT_METADATA_PATH = MODELS_DIR / "sentiment_model_metadata.json"

VALID_SENTIMENTS = {"positive", "negative", "neutral"}


def normalize_sentiment_label(label: Any) -> str:
    """Normalize common sentiment labels into a consistent form."""
    if label is None:
        raise ValueError("Sentiment label is missing.")
    normalized = str(label).strip().lower()
    mapping = {
        "positive": "positive",
        "pos": "positive",
        "good": "positive",
        "happy": "positive",
        "negative": "negative",
        "neg": "negative",
        "bad": "negative",
        "sad": "negative",
        "neutral": "neutral",
        "neut": "neutral",
        "mixed": "neutral",
        "ok": "neutral",
    }
    if normalized in mapping:
        return mapping[normalized]
    raise ValueError(f"Unsupported sentiment label '{label}'. Expected one of: {sorted(VALID_SENTIMENTS)}")


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Validate dataset schema, missing values, and class labels before training."""
    missing_columns = {"text", "sentiment"} - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    if df.empty:
        raise ValueError("Dataset is empty. Add labelled examples before training.")

    cleaned_df = df.copy()
    cleaned_df["text"] = cleaned_df["text"].fillna("")
    cleaned_df["sentiment"] = cleaned_df["sentiment"].fillna("")

    cleaned_df["text"] = cleaned_df["text"].astype(str).str.strip()
    cleaned_df["sentiment"] = cleaned_df["sentiment"].astype(str).str.strip()

    if (cleaned_df["text"] == "").any():
        raise ValueError("Found one or more empty text values in the dataset.")

    cleaned_df["sentiment"] = cleaned_df["sentiment"].apply(normalize_sentiment_label)
    if cleaned_df["sentiment"].isnull().any():
        raise ValueError("Dataset contains invalid sentiment labels.")

    if cleaned_df["sentiment"].eq("").any():
        raise ValueError("Dataset contains empty sentiment labels.")

    invalid_rows = cleaned_df[~cleaned_df["sentiment"].isin(VALID_SENTIMENTS)]
    if not invalid_rows.empty:
        invalid_values = sorted(set(invalid_rows["sentiment"].tolist()))
        raise ValueError(f"Unsupported sentiment labels found: {invalid_values}")

    return cleaned_df[["text", "sentiment"]].reset_index(drop=True)


def load_dataset(dataset_path: str | Path) -> pd.DataFrame:
    """Read and validate a CSV dataset file with text/sentiment columns."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)
    return validate_dataset(df)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _create_pipeline(random_state: int = 42) -> Pipeline:
    def preprocess_series(values: Iterable[str]) -> List[str]:
        if isinstance(values, pd.Series):
            return [preprocess_text(str(value)) for value in values.tolist()]
        return [preprocess_text(str(value)) for value in values]

    vectorizer = TfidfVectorizer(
        lowercase=False,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    classifier = LogisticRegression(
        max_iter=3000,
        solver="liblinear",
        random_state=random_state,
    )
    pipeline = Pipeline([
        ("preprocess", __import__("sklearn.preprocessing").preprocessing.FunctionTransformer(preprocess_series, validate=False)),
        ("tfidf", vectorizer),
        ("clf", classifier),
    ])
    return pipeline


def compute_holdout_metrics(y_true: Iterable[Any], y_pred: Iterable[Any], labels: List[Any] | None = None) -> Dict[str, Any]:
    return compute_metrics(y_true, y_pred, labels=labels)


def train_model(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train a sentiment model from a labelled dataset and persist it to disk."""
    dataset = load_dataset(dataset_path)
    X = dataset["text"]
    y = dataset["sentiment"]

    has_stratify = len(dataset) >= 6 and min(y.value_counts().min(), len(pd.unique(y))) >= 2
    if has_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

    pipeline = _create_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    metrics = compute_holdout_metrics(y_test, y_pred, labels=list(sorted(VALID_SENTIMENTS)))

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    metadata = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "dataset_hash": _file_hash(Path(dataset_path)),
        "labels": sorted(VALID_SENTIMENTS),
        "random_state": random_state,
        "test_size": test_size,
        "metrics": {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        },
    }
    metadata_path = Path(DEFAULT_METADATA_PATH)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "model": pipeline,
        "metrics": metrics,
        "metadata": metadata,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> Any:
    """Safely load a persisted model artifact from disk."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}")
    return joblib.load(path)


def load_or_train_model(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    force_retrain: bool = False,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Load an existing saved model when it matches the dataset, otherwise train a fresh one."""
    dataset_path = Path(dataset_path)
    model_path = Path(model_path)
    metadata_path = Path(DEFAULT_METADATA_PATH)

    if force_retrain:
        return train_model(dataset_path=dataset_path, model_path=model_path, random_state=random_state)

    if model_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        try:
            dataset_hash = _file_hash(dataset_path)
            if metadata.get("dataset_hash") == dataset_hash:
                model = load_model(model_path)
                metrics = metadata.get("metrics", {})
                return {
                    "model": model,
                    "metrics": metrics,
                    "metadata": metadata,
                }
        except Exception:
            pass

    return train_model(dataset_path=dataset_path, model_path=model_path, random_state=random_state)
