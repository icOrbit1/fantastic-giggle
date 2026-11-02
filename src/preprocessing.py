# fantastic-giggle

This repository contains a small social-intent sentiment analysis project built with Python, NLTK, Scikit-learn, and Streamlit.

## Project overview

The app loads a small sample CSV dataset, validates it, preprocesses text, trains a TF-IDF + logistic regression sentiment model, and exposes the workflow through a Streamlit frontend. The project is intentionally beginner-friendly and modular so a real labelled dataset can replace the demo data easily.

## Features

- Text sentiment analyzer for single inputs
- Batch CSV sentiment analysis
- Dataset validation and class distribution reporting
- Model evaluation with held-out test metrics
- Preprocessing utility with URL, HTML, whitespace, and punctuation cleanup
- Streamlit dashboard and dataset explorer pages

## Technology stack

- Python 3.10+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn
- Pytest

## Project structure

```text
fantastic-giggle/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
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
└── .streamlit/
    └── config.toml
```

## Installation

### Windows

```bash
cd fantastic-giggle
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux/macOS

```bash
cd fantastic-giggle
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the app locally

```bash
streamlit run app.py
```

## Retrain the model

```bash
python -c "from src.train_model import train_model; train_model('data/sample_sentiment.csv', 'models/sentiment_model.joblib')"
```

## Dataset format

The CSV should have two columns:

```csv
text,sentiment
"I really enjoyed the product.",positive
"The service was terrible.",negative
"The update is okay.",neutral
```

## How to run tests

```bash
pytest
```

## Streamlit Community Cloud deployment

1. Push the repository to GitHub.
2. Open https://share.streamlit.io/
3. Choose the repository `icOrbit1/fantastic-giggle`
4. Use branch `main`
5. Set the entry point to `app.py`
6. Deploy

## Troubleshooting

- If NLTK resources are missing, run:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

- If the model is stale or incompatible, delete `models/sentiment_model.joblib` and retrain it.

## Limitations

This repository ships with a small demonstration dataset for learning and testing. It should not be treated as a real-world benchmark, and the model should be retrained on domain-specific data before production use.

