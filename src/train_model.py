from __future__ import annotations

from typing import Any, Dict, Iterable, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

matplotlib.use("Agg")


def compute_metrics(y_true: Iterable[Any], y_pred: Iterable[Any], labels: List[Any] | None = None) -> Dict[str, Any]:
    """Compute multiclass evaluation metrics with macro averaging and safe handling for edge cases."""
    y_true = list(y_true)
    y_pred = list(y_pred)
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
        "classification_report": report,
        "labels": labels,
    }


def confusion_matrix_figure(y_true: Iterable[Any], y_pred: Iterable[Any], labels: List[Any] | None = None) -> plt.Figure:
    """Build a confusion matrix figure for the held-out test set."""
    metrics = compute_metrics(y_true, y_pred, labels=labels)
    labels_display = metrics["labels"]
    cm = metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels_display, yticklabels=labels_display, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def classification_report_dataframe(y_true: Iterable[Any], y_pred: Iterable[Any], labels: List[Any] | None = None) -> pd.DataFrame:
    metrics = compute_metrics(y_true, y_pred, labels=labels)
    report = metrics["classification_report"]
    rows = []
    for label in metrics["labels"]:
        label_data = report.get(str(label), report.get(label, {}))
        rows.append(
            {
                "label": label,
                "precision": label_data.get("precision", 0.0),
                "recall": label_data.get("recall", 0.0),
                "f1-score": label_data.get("f1-score", 0.0),
                "support": label_data.get("support", 0),
            }
        )
    rows.append(
        {
            "label": "macro avg",
            "precision": report.get("macro avg", {}).get("precision", 0.0),
            "recall": report.get("macro avg", {}).get("recall", 0.0),
            "f1-score": report.get("macro avg", {}).get("f1-score", 0.0),
            "support": report.get("macro avg", {}).get("support", 0),
        }
    )
    return pd.DataFrame(rows)
