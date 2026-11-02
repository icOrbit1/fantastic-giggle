from __future__ import annotations

import html
import re
from typing import Iterable, List, Sequence

from nltk import download as nltk_download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

NEGATION_WORDS = {
    "not", "no", "never", "none", "cannot", "can't", "won't", "isn't", "aren't",
    "wasn't", "weren't", "don't", "doesn't", "didn't", "haven't", "hasn't", "hadn't",
    "nor", "hardly", "barely"
}

_LEMMATIZER = WordNetLemmatizer()
_STOPWORDS = None


def _ensure_nltk_resources() -> None:
    """Download the minimal NLTK resources used by the preprocessing pipeline."""
    resources = ["punkt", "stopwords", "wordnet", "omw-1.4"]
    for resource in resources:
        try:
            if resource == "punkt":
                word_tokenize("test")
            elif resource == "stopwords":
                stopwords.words("english")
            elif resource == "wordnet":
                _LEMMATIZER.lemmatize("test")
            elif resource == "omw-1.4":
                _LEMMATIZER.lemmatize("test")
        except Exception:
            try:
                nltk_download(resource, quiet=True)
            except Exception:
                pass


_ensure_nltk_resources()


def _get_stopwords() -> set[str]:
    global _STOPWORDS
    if _STOPWORDS is None:
        try:
            _STOPWORDS = set(stopwords.words("english"))
        except Exception:
            _STOPWORDS = set()
        _STOPWORDS = {word for word in _STOPWORDS if word not in NEGATION_WORDS}
    return _STOPWORDS


def normalize_text(text: str | None) -> str:
    """Convert raw text into a clean, comparable format for sentiment modeling."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = html.unescape(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ")
    text = text.replace("\u2019", "'")
    text = text.replace("\u2018", "'")
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip().lower()
    if not text:
        return ""

    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    cleaned_tokens: List[str] = []
    for token in tokens:
        token = token.strip("'")
        if not token:
            continue
        if token in NEGATION_WORDS:
            cleaned_tokens.append(token)
            continue
        if token in _get_stopwords():
            continue
        cleaned_tokens.append(_LEMMATIZER.lemmatize(token, pos="v"))

    return " ".join(cleaned_tokens).strip()


def preprocess_text(text: str | None) -> str:
    """A safe, reusable preprocessing function for both training and inference."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if not text:
        return ""
    normalized = normalize_text(text)
    if not normalized:
        return ""
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def preprocess_batch(texts: Iterable[str]) -> List[str]:
    """Preprocess a sequence of text entries while preserving consistent handling."""
    return [preprocess_text(text) for text in texts]
