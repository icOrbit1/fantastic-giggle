from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.predictor import predict_batch_texts, predict_single_text, validate_uploaded_csv
from src.train_model import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_PATH,
    classification_report_dataframe,
    confusion_matrix_figure,
    load_dataset,
    load_or_train_model,
)

ROOT_DIR = Path(__file__).resolve().parent
DATASET_PATH = ROOT_DIR / "data" / "sample_sentiment.csv"
MODEL_PATH = ROOT_DIR / "models" / "sentiment_model.joblib"
METADATA_PATH = ROOT_DIR / "models" / "sentiment_model_metadata.json"


@st.cache_resource
def get_model_bundle():
    """Reuse a trained model across Streamlit reruns without retraining every time."""
    return load_or_train_model(DATASET_PATH, MODEL_PATH)


@st.cache_data
def get_dataset() -> pd.DataFrame:
    """Load the small demonstration dataset used by the app."""
    return load_dataset(DATASET_PATH)


def render_dashboard() -> None:
    st.title("Sentiment Analysis Dashboard")
    st.markdown(
        """
        This app classifies short text into **Positive**, **Negative**, or **Neutral** sentiment.
        It is designed for learning, demos, and small internal experiments.
        """
    )

    dataset = get_dataset()
    bundle = get_model_bundle()
    metrics = bundle.get("metrics", {})

    st.subheader("Project overview")
    st.info(
        "The bundled dataset is intentionally small and should not be treated as a real-world benchmark. "
        "Use it for demonstration and learning, then retrain on a domain-specific labelled dataset."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", len(dataset))
    col2.metric("Classes", len(dataset["sentiment"].unique()))
    col3.metric("Accuracy", f"{metrics.get('accuracy', 0.0):.3f}" if metrics.get("accuracy") is not None else "n/a")

    st.subheader("Dataset statistics")
    label_order = ["positive", "negative", "neutral"]
    counts = dataset["sentiment"].value_counts().reindex(label_order, fill_value=0)
    st.bar_chart(counts)

    st.subheader("Model evaluation summary")
    summary = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
            "Value": [
                metrics.get("accuracy", 0.0),
                metrics.get("precision", 0.0),
                metrics.get("recall", 0.0),
                metrics.get("f1", 0.0),
            ],
        }
    )
    st.dataframe(summary, use_container_width=True)


def render_analyzer() -> None:
    st.title("Sentiment Analyzer")
    st.caption("Type a review, message, or comment and classify its sentiment.")

    text = st.text_area(
        "Text to analyze",
        height=180,
        placeholder="Example: I really liked the product and the support was great.",
    )

    if st.button("Analyze sentiment", use_container_width=True):
        if text is None or not text.strip():
            st.error("Please enter some text before analyzing.")
            return

        try:
            with st.spinner("Analyzing text..."):
                bundle = get_model_bundle()
                result = predict_single_text(text, model=bundle["model"])

            sentiment = result["sentiment"].title()
            confidence = float(result["confidence"])

            st.subheader("Prediction")
            st.markdown(f"### Predicted sentiment: **{sentiment}**")
            st.progress(min(1.0, max(0.0, confidence)))
            st.caption(f"Model confidence: {confidence:.2%}. This is an estimate, not a guarantee.")

            probability_df = pd.DataFrame(
                {
                    "Sentiment": [label.title() for label in result["probabilities"].keys()],
                    "Probability": [float(value) for value in result["probabilities"].values()],
                }
            )
            st.bar_chart(probability_df.set_index("Sentiment"))

            with st.expander("View preprocessed text"):
                st.code(result.get("preprocessed_text", text), language="text")
        except Exception as exc:  # pragma: no cover - safety net
            st.error(f"Unable to analyze the text: {exc}")


def render_batch_analysis() -> None:
    st.title("Batch Analysis")
    st.caption("Upload a CSV file containing a text column to classify many rows at once.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is None:
        st.info("No CSV uploaded yet.")
        return

    if uploaded_file.size and uploaded_file.size > 5 * 1024 * 1024:
        st.error("Uploaded files must be 5 MB or smaller.")
        return

    try:
        data = validate_uploaded_csv(uploaded_file)
        with st.spinner("Predicting batch sentiment..."):
            bundle = get_model_bundle()
            results = predict_batch_texts(data["text"].tolist(), model=bundle["model"])

        st.subheader("Analysis results")
        st.dataframe(results, use_container_width=True)

        csv_bytes = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download predictions as CSV",
            data=csv_bytes,
            file_name="sentiment_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:  # pragma: no cover - safety net
        st.error(f"The uploaded CSV could not be processed: {exc}")


def render_model_evaluation() -> None:
    st.title("Model Evaluation")
    bundle = get_model_bundle()
    metrics = bundle.get("metrics", {})

    if not metrics:
        st.warning("No evaluation metrics are available yet. The model may need to be retrained.")
        return

    st.caption("These metrics come from a held-out test split and are reported only when valid evaluation data is available.")
    summary = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
            "Value": [
                metrics.get("accuracy", 0.0),
                metrics.get("precision", 0.0),
                metrics.get("recall", 0.0),
                metrics.get("f1", 0.0),
            ],
        }
    )
    st.dataframe(summary, use_container_width=True)

    if "y_true" in bundle and "y_pred" in bundle and bundle.get("y_true") is not None and bundle.get("y_pred") is not None:
        labels = list(metrics.get("labels", ["positive", "negative", "neutral"]))
        st.subheader("Confusion matrix")
        st.pyplot(confusion_matrix_figure(bundle["y_true"], bundle["y_pred"], labels=labels))

        st.subheader("Classification report")
        st.dataframe(classification_report_dataframe(bundle["y_true"], bundle["y_pred"], labels=labels), use_container_width=True)
    else:
        st.info("Confusion matrix and classification report are unavailable because held-out evaluation data is missing.")


def render_dataset_explorer() -> None:
    st.title("Dataset Explorer")
    dataset = get_dataset()

    st.dataframe(dataset, use_container_width=True)

    st.subheader("Class distribution")
    counts = dataset["sentiment"].value_counts().reindex(["positive", "negative", "neutral"], fill_value=0)
    st.bar_chart(counts)

    csv_bytes = dataset.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download sample dataset",
        data=csv_bytes,
        file_name="sample_sentiment.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.warning(
        "This sample dataset is intentionally small and demonstration-only. It should not be treated as a benchmark or production dataset."
    )


def main() -> None:
    st.set_page_config(page_title="Sentiment Analysis System", page_icon="📊", layout="wide")

    nav = {
        "Dashboard": render_dashboard,
        "Sentiment Analyzer": render_analyzer,
        "Batch Analysis": render_batch_analysis,
        "Model Evaluation": render_model_evaluation,
        "Dataset Explorer": render_dataset_explorer,
    }

    selection = st.sidebar.radio("Navigation", list(nav.keys()))
    nav[selection]()


if __name__ == "__main__":
    main()
