from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import preprocess_text

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
DEFAULT_DATASET_PATH = DATA_DIR / "sample_sentiment.csv"
DEFAULT_MODEL_PATH = MODELS_DIR / "sentiment_model.joblib"
DEFAULT_METADATA_PATH = MODELS_DIR / "sentiment_model_metadata.json"

VALID_SENTIMENTS = ("positive", "negative", "neutral")
LABEL_ALIASES = {
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
    "okay": "neutral",
}


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_sentiment_label(label: Any) -> str:
    """Convert common sentiment labels to canonical values."""
    if label is None:
        raise ValueError("Sentiment label is missing.")
    normalized = str(label).strip().lower()
    if not normalized:
        raise ValueError("Sentiment label is empty.")
    if normalized in LABEL_ALIASES:
        return LABEL_ALIASES[normalized]
    raise ValueError(f"Unsupported sentiment label '{label}'. Expected one of: {', '.join(VALID_SENTIMENTS)}")


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Validate dataset schema, missing values, and labels before training."""
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Dataset must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Dataset is empty. Add labelled examples before training.")

    required = {"text", "sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    cleaned = df.copy()
    cleaned["text"] = cleaned["text"].fillna("").astype(str).str.strip()
    cleaned["sentiment"] = cleaned["sentiment"].fillna("").astype(str).str.strip()

    if (cleaned["text"] == "").any():
        raise ValueError("Dataset contains one or more empty text values.")
    if (cleaned["sentiment"] == "").any():
        raise ValueError("Dataset contains one or more empty sentiment labels.")

    cleaned["sentiment"] = cleaned["sentiment"].apply(normalize_sentiment_label)
    cleaned = cleaned[["text", "sentiment"]].reset_index(drop=True)

    if len(cleaned["sentiment"].unique()) < 2:
        raise ValueError("Dataset must include at least two sentiment classes for training.")

    return cleaned


def load_dataset(dataset_path: str | Path) -> pd.DataFrame:
    """Read and validate the input dataset from CSV."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive path
        raise ValueError(f"Could not read dataset '{path}': {exc}") from exc
    return validate_dataset(df)


def compute_metrics(y_true: Iterable[Any], y_pred: Iterable[Any], labels: Sequence[Any] | None = None) -> dict[str, Any]:
    """Return accuracy, macro precision, macro recall, macro F1, and confusion matrix."""
    y_true = list(y_true)
    y_pred = list(y_pred)
    label_list = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", labels=label_list, zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", labels=label_list, zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", labels=label_list, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=label_list)
    report = classification_report(y_true, y_pred, labels=label_list, output_dict=True, zero_division=0)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": matrix,
        "classification_report": report,
        "labels": list(label_list),
    }


def confusion_matrix_figure(y_true: Iterable[Any], y_pred: Iterable[Any], labels: Sequence[Any] | None = None):
    """Create a confusion-matrix figure for Streamlit."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    metrics = compute_metrics(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        metrics["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=metrics["labels"],
        yticklabels=metrics["labels"],
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    return fig


def classification_report_dataframe(y_true: Iterable[Any], y_pred: Iterable[Any], labels: Sequence[Any] | None = None) -> pd.DataFrame:
    """Build a compact dataframe from the sklearn classification report."""
    metrics = compute_metrics(y_true, y_pred, labels=labels)
    report = metrics["classification_report"]
    rows = []
    for label in metrics["labels"]:
        label_data = report.get(str(label), report.get(label, {}))
        rows.append(
            {
                "label": label,
                "precision": float(label_data.get("precision", 0.0)),
                "recall": float(label_data.get("recall", 0.0)),
                "f1-score": float(label_data.get("f1-score", 0.0)),
                "support": int(label_data.get("support", 0)),
            }
        )
    rows.append(
        {
            "label": "macro avg",
            "precision": float(report.get("macro avg", {}).get("precision", 0.0)),
            "recall": float(report.get("macro avg", {}).get("recall", 0.0)),
            "f1-score": float(report.get("macro avg", {}).get("f1-score", 0.0)),
            "support": int(report.get("macro avg", {}).get("support", 0)),
        }
    )
    return pd.DataFrame(rows)


def _create_pipeline(random_state: int = 42) -> Pipeline:
    """Create the TF-IDF + Logistic Regression model pipeline."""
    vectorizer = TfidfVectorizer(
        lowercase=False,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    classifier = LogisticRegression(
        solver="liblinear",
        random_state=random_state,
        max_iter=3000,
    )
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def train_model(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train a model from a labelled dataset and save it to disk."""
    dataset = load_dataset(dataset_path)
    texts = dataset["text"].map(preprocess_text).tolist()
    labels = dataset["sentiment"].tolist()

    if len(dataset) < 3:
        raise ValueError("At least three labelled examples are required to train the model.")
    if len(set(labels)) < 2:
        raise ValueError("At least two sentiment classes are required for training.")

    if len(dataset) >= 6 and len(set(labels)) == 3:
        X_train, X_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=random_state,
        )

    model = _create_pipeline(random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, labels=list(VALID_SENTIMENTS))

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    metadata = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "dataset_hash": _file_hash(Path(dataset_path)),
        "labels": list(VALID_SENTIMENTS),
        "random_state": random_state,
        "test_size": test_size,
        "metrics": {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "labels": metrics["labels"],
        },
    }

    metadata_path = Path(DEFAULT_METADATA_PATH)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "model": model,
        "metrics": metrics,
        "metadata": metadata,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "y_true": y_test,
        "y_pred": y_pred,
    }


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH):
    """Load a saved scikit-learn model artifact safely."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at '{path}'")
    model = joblib.load(path)
    if not hasattr(model, "predict"):
        raise ValueError(f"Loaded model at '{path}' is not a valid scikit-learn pipeline.")
    return model


def load_or_train_model(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    force_retrain: bool = False,
    random_state: int = 42,
) -> dict[str, Any]:
    """Load an existing valid model; otherwise train a fresh one."""
    dataset_path = Path(dataset_path)
    model_path = Path(model_path)
    metadata_path = Path(DEFAULT_METADATA_PATH)

    if force_retrain:
        return train_model(dataset_path=dataset_path, model_path=model_path, random_state=random_state)

    if model_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("dataset_hash") == _file_hash(dataset_path):
                return {"model": load_model(model_path), "metrics": metadata.get("metrics", {}), "metadata": metadata}
        except Exception:
            pass

    return train_model(dataset_path=dataset_path, model_path=model_path, random_state=random_state)
