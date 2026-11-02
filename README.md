from __future__ import annotations

import pandas as pd

from src.predictor import predict_batch_texts, predict_single_text
from src.train_model import train_model


def test_train_model_and_predict_single_text(tmp_path):
    csv_path = tmp_path / "sentiment.csv"
    pd.DataFrame(
        {
            "text": [
                "I love this product",
                "The service is terrible",
                "This is okay",
                "I am happy with the update",
                "The experience was awful",
                "The app is fine",
            ],
            "sentiment": [
                "positive",
                "negative",
                "neutral",
                "positive",
                "negative",
                "neutral",
            ],
        }
    ).to_csv(csv_path, index=False)

    model_bundle = train_model(csv_path, tmp_path / "sentiment_model.joblib")
    model = model_bundle["model"]
    result = predict_single_text("I really enjoyed the new update", model=model)
    assert result["sentiment"] in {"positive", "negative", "neutral"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["probabilities"]) == len(model.classes_)


def test_predict_batch_texts_returns_valid_probability_shape():
    texts = [
        "I love the product",
        "The service was bad",
        "The app is average",
    ]
    model_bundle = train_model("data/sample_sentiment.csv", "models/sentiment_model.joblib")
    result = predict_batch_texts(texts, model=model_bundle["model"])
    assert list(result.columns[:3]) == ["text", "predicted_sentiment", "confidence"]
    assert all(col.startswith("prob_") for col in result.columns[3:])
    assert len(result.columns) == 3 + len(model_bundle["model"].classes_)
    assert result["confidence"].between(0.0, 1.0).all()
