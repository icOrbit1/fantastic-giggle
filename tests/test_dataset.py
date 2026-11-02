from __future__ import annotations

import pandas as pd
import pytest

from src.train_model import load_dataset, normalize_sentiment_label, validate_dataset


def test_normalize_sentiment_label_accepts_common_variants():
    assert normalize_sentiment_label("Positive") == "positive"
    assert normalize_sentiment_label("NEGATIVE") == "negative"
    assert normalize_sentiment_label("neutral") == "neutral"
    assert normalize_sentiment_label("okay") == "neutral"


def test_validate_dataset_accepts_valid_data():
    df = pd.DataFrame(
        {
            "text": ["I love this product", "This is bad", "It is okay"],
            "sentiment": ["positive", "negative", "neutral"],
        }
    )
    cleaned = validate_dataset(df)
    assert list(cleaned.columns) == ["text", "sentiment"]
    assert cleaned["sentiment"].tolist() == ["positive", "negative", "neutral"]


def test_validate_dataset_raises_for_missing_values():
    df = pd.DataFrame(
        {
            "text": ["Good", "", "This is okay"],
            "sentiment": ["positive", "negative", "neutral"],
        }
    )
    with pytest.raises(ValueError):
        validate_dataset(df)


def test_validate_dataset_raises_for_invalid_label():
    df = pd.DataFrame(
        {"text": ["Good", "Bad"], "sentiment": ["positive", "unknown"]}
    )
    with pytest.raises(ValueError):
        validate_dataset(df)


def test_load_dataset_reads_csv_and_validates_schema(tmp_path):
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame(
        {"text": ["I love this", "I dislike this", "This is okay"], "sentiment": ["Positive", "negative", "neutral"]}
    ).to_csv(csv_path, index=False)

    loaded = load_dataset(csv_path)
    assert len(loaded) == 3
    assert set(loaded["sentiment"]) == {"positive", "negative", "neutral"}
