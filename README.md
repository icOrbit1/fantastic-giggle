# Sentiment Analysis System

A lightweight sentiment analysis application built with Python, scikit-learn, NLTK, and Streamlit. The project includes a public-facing web interface, a reusable preprocessing pipeline, dataset validation, and a simple training/evaluation workflow for Positive, Negative, and Neutral sentiment labels.

## Features

- Single-text sentiment analysis
- Batch CSV analysis with downloadable results
- Validation for empty inputs, malformed CSV uploads, and unsupported labels
- Reusable preprocessing for URLs, HTML, punctuation, whitespace, and negation-aware filtering
- TF-IDF + Logistic Regression training pipeline
- Model evaluation with accuracy, precision, recall, F1-score, and confusion matrix
- Mobile-friendly Streamlit UI for Android browsers and desktop browsers
- Small sample dataset for demo and testing

## Technology stack

- Python 3.10+
- Streamlit
- Pandas
- NumPy
- scikit-learn
- NLTK
- Matplotlib
- Seaborn
- Pytest

## Repository structure

```text
fantastic-giggle/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── data/
│   └── sample_sentiment.csv
├── models/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── predictor.py
│   └── evaluation.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_predictor.py
│   └── test_dataset.py
└── .venv/
```

## Local setup

### Windows

```bash
cd fantastic-giggle
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
cd fantastic-giggle
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run locally

```bash
streamlit run app.py
```

The app opens in the browser at a local URL such as http://localhost:8501.

## Dataset format

The sample CSV uses the following format:

```csv
text,sentiment
"I absolutely love this product.",positive
"The service was terrible.",negative
"The update is okay.",neutral
```

Expected columns:

- `text`: raw review or message
- `sentiment`: one of `positive`, `negative`, or `neutral`

The app normalizes common label variants such as `Positive`, `NEGATIVE`, and `okay`.

## Retraining the model

```bash
python -c "from src.train_model import train_model; train_model('data/sample_sentiment.csv', 'models/sentiment_model.joblib')"
```

The app automatically loads the saved model when the dataset hash matches. If the dataset changes, the model is retrained automatically.

## Run tests

```bash
pytest
```

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open https://share.streamlit.io/
3. Choose the repository: `icOrbit1/fantastic-giggle`
4. Use the branch: `main`
5. Set the app entry file to: `app.py`
6. Keep the Python version at 3.10 or later.
7. Deploy.

The app is ready to be selected directly in Streamlit Community Cloud using the repository root and `app.py` as the entry point.

## Common deployment issues

- `ModuleNotFoundError` or missing dependency: install all requirements with `pip install -r requirements.txt`.
- `FileNotFoundError` for `data/sample_sentiment.csv` or the model file: confirm the repository contains those files and the paths are relative.
- NLTK missing resources: run:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

- Model not loading: delete `models/sentiment_model.joblib` and retrain it.

## Mobile usage

This web app is designed to work from an Android browser and desktop browser. It uses a single-column layout where appropriate, large touch-friendly controls, and readable charts to keep it usable on mobile screens.

## Responsible use

- The sample dataset is intentionally small and is not a benchmark dataset.
- Model predictions are estimates based on the available training data.
- Model confidence should not be treated as certainty.
- For real-world usage, retrain the model on a dataset that matches your target domain.
