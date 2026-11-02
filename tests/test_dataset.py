from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing import preprocess_text


def test_preprocess_text_removes_urls_and_handles_empty_input():
    assert preprocess_text("") == ""
    assert preprocess_text("!!!") == ""
    assert "example" not in preprocess_text("Visit https://example.com today")


def test_preprocess_preserves_negation_words():
    result = preprocess_text("I am not happy at all")
    assert "not" in result
    assert "happy" in result


def test_preprocess_handles_mixed_whitespace_and_punctuation():
    result = preprocess_text("  Very!!!    good!!!  ")
    assert "very" in result
    assert "good" in result


def test_preprocess_handles_unicode_and_repeated_whitespace():
    result = preprocess_text("Mañana   mañana   😊")
    assert "mañana" in result or "manana" in result


def test_preprocess_removes_stopwords_but_keeps_meaningful_sentiment_terms():
    result = preprocess_text("The service was not terrible and genuinely helpful")
    assert "not" in result
    assert "terrible" in result
    assert "helpful" in result
