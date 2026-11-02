from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.predictor import predict_batch_texts, predict_single_text, validate_uploaded_csv
from src.train_model import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_PATH,
    compute_metrics,
    confusion_matrix_figure,
    classification_report_dataframe,
    load_dataset,
    load_or_train_model,
)


DATASET_PATH = Path(DEFAULT_DATASET_PATH)
MODEL_PATH = Path(DEFAULT_MODEL_PATH)


@st.cache_resource
def get_model_bundle():
    """Load a validated model, retraining only if the saved artifact is stale or missing."""
    return load_or_train_model(DATASET_PATH, MODEL_PATH, force_retrain=False)


@st.cache_data
def get_dataset() -> pd.DataFrame:
    """Return the sample dataset used by the app."""
    return load_dataset(DATASET_PATH)


def render_dashboard() -> None:
    st.title("Sentiment Analysis Dashboard")
    st.markdown(
        """
        This project uses a lightweight NLP pipeline to classify short text into
        **Positive**, **Negative**, or **Neutral** sentiment.
        """
    )

    dataset = get_dataset()
    bundle = get_model_bundle()

    st.subheader("Project overview")
    st.info(
        "The model is trained on a small demonstration dataset. It is designed for learning and testing, "
        "not for production decision-making without a domain-specific labelled dataset."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset rows", len(dataset))
    col2.metric("Classes", len(bundle["metrics"]["labels"]))
    col3.metric("Accuracy", f"{bundle['metrics']['accuracy']:.3f}")

    st.subheader("Dataset statistics")
    class_counts = dataset["sentiment"].value_counts().reindex(["positive", "negative", "neutral"], fill_value=0)
    st.bar_chart(class_counts)

    st.subheader("Model evaluation summary")
    metrics = bundle["metrics"]
    summary = {
        "Accuracy": metrics["accuracy"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1-score": metrics["f1"],
    }
    st.dataframe(pd.DataFrame({"Metric": list(summary.keys()), "Value": list(summary.values())}), use_container_width=True)


def render_analyzer() -> None:
    st.title("Sentiment Analyzer")
    text = st.text_area("Enter text to classify", height=180, placeholder="Type a customer review, comment, or message...")

    if st.button("Analyze sentiment"):
        if text is None or not text.strip():
            st.error("Please enter some text before analyzing.")
            return

        try:
            bundle = get_model_bundle()
            result = predict_single_text(text, model=bundle["model"])
            label = result["sentiment"].title()
            confidence = result["confidence"]

            st.subheader("Prediction")
            st.markdown(f"### Predicted sentiment: **{label}**")
            st.progress(min(0.0, max(1.0, confidence)))
            st.caption(f"Model confidence: {confidence:.2%} (estimate, not a guarantee)")

            probs = pd.DataFrame(
                {
                    "Sentiment": [item.title() for item in result["probabilities"].keys()],
                    "Probability": [float(value) for value in result["probabilities"].values()],
                }
            )
            st.bar_chart(probs.set_index("Sentiment"))

            with st.expander("View preprocessed text"):
                st.code(result["preprocessed_text"], language="text")
        except Exception as exc:  # pragma: no cover - UI safety net
            st.error(f"An unexpected error occurred while analyzing the text: {exc}")


def render_batch_analysis() -> None:
    st.title("Batch Analysis")
    st.caption("Upload a CSV file containing a text column to classify multiple rows at once.")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is None:
        st.info("No file uploaded yet.")
        return

    try:
        dataset = validate_uploaded_csv(uploaded_file)
        bundle = get_model_bundle()
        results = predict_batch_texts(dataset["text"].tolist(), model=bundle["model"])

        st.subheader("Prediction results")
        st.dataframe(results, use_container_width=True)

        csv_data = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download predictions as CSV",
            data=csv_data,
            file_name="sentiment_predictions.csv",
            mime="text/csv",
        )
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:  # pragma: no cover - UI safety net
        st.error(f"The uploaded file could not be processed: {exc}")


def render_evaluation() -> None:
    st.title("Model Evaluation")
    bundle = get_model_bundle()
    metrics = bundle["metrics"]

    st.caption("These are held-out test metrics from the model. They are not synthetic demo results.")
    metric_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
            "Value": [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]],
        }
    )
    st.dataframe(metric_df, use_container_width=True)

    labels = list(metrics["labels"])
    cm_figure = confusion_matrix_figure(bundle["y_true"], bundle["y_pred"], labels=labels)
    st.subheader("Confusion matrix")
    st.pyplot(cm_figure)

    report_df = classification_report_dataframe(bundle["y_true"], bundle["y_pred"], labels=labels)
    st.subheader("Classification report")
    st.dataframe(report_df, use_container_width=True)


def render_dataset_explorer() -> None:
    st.title("Dataset Explorer")
    dataset = get_dataset()
    st.dataframe(dataset, use_container_width=True)

    class_counts = dataset["sentiment"].value_counts().reindex(["positive", "negative", "neutral"], fill_value=0)
    st.subheader("Class distribution")
    st.bar_chart(class_counts)

    csv_bytes = dataset.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download sample dataset",
        data=csv_bytes,
        file_name="sample_sentiment.csv",
        mime="text/csv",
    )

    st.warning(
        "This sample dataset is intentionally small and demonstrates the app workflow. It should not be treated as a benchmark or as a production-quality training set."
    )


def main() -> None:
    st.set_page_config(page_title="Sentiment Analysis System", page_icon="📊", layout="wide")

    pages = {
        "Dashboard": render_dashboard,
        "Sentiment Analyzer": render_analyzer,
        "Batch Analysis": render_batch_analysis,
        "Model Evaluation": render_evaluation,
        "Dataset Explorer": render_dataset_explorer,
    }

    selected_page = st.sidebar.radio("Navigation", list(pages.keys()))
    pages[selected_page]()


if __name__ == "__main__":
    main()
