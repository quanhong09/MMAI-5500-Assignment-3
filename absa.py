"""Aspect-based sentiment analysis for MMAI 5400 Assignment 3.

Usage:
    from absa import absa

    results = absa(["The fried rice was great but the waiter was rude."])

Loads the models saved by finetuning.py, extracts aspect spans, classifies
their sentiment and aspect category, and returns a DataFrame with the
columns review_id, aspect, category and sentiment.
"""

from pathlib import Path

import pandas as pd
from setfit import AbsaModel, SetFitModel

MODEL_DIR = Path(__file__).resolve().parent / "models"
ASPECT_PATH = MODEL_DIR / "absa-aspect"
POLARITY_PATH = MODEL_DIR / "absa-polarity"
CATEGORY_PATH = MODEL_DIR / "category"
SPACY_MODEL = "en_core_web_sm"
COLUMNS = ["review_id", "aspect", "category", "sentiment"]

_MODELS = {}


def load_models():
    """Load the saved models once and reuse them on later calls."""
    if not _MODELS:
        _MODELS["absa"] = AbsaModel.from_pretrained(
            str(ASPECT_PATH),
            str(POLARITY_PATH),
            spacy_model=SPACY_MODEL,
        )
        _MODELS["category"] = SetFitModel.from_pretrained(str(CATEGORY_PATH))
    return _MODELS["absa"], _MODELS["category"]


def absa(reviews):
    """Run aspect extraction, sentiment and category classification.

    Args:
        reviews: A list of review strings. A single string is also accepted.

    Returns:
        A pandas DataFrame with columns review_id (int), aspect (str),
        category (Food, Price, Service or Ambience) and sentiment
        (Positive or Negative), one row per extracted aspect.
    """
    if isinstance(reviews, str):
        reviews = [reviews]
    reviews = ["" if review is None else str(review) for review in reviews]

    rows = []
    if any(review.strip() for review in reviews):
        absa_model, category_model = load_models()
        predictions = absa_model.predict(reviews)

        for review_id, (review, aspects) in enumerate(
                zip(reviews, predictions)):
            for aspect in aspects:
                rows.append({
                    "review_id": review_id,
                    "aspect": aspect["span"],
                    "sentiment": str(aspect["polarity"]).title(),
                    "text": f"{aspect['span']}: {review}",
                })

    if not rows:
        return pd.DataFrame(columns=COLUMNS).astype({"review_id": int})

    categories = category_model.predict([row["text"] for row in rows])
    for row, category in zip(rows, categories):
        row["category"] = str(category).title()

    results = pd.DataFrame(rows, columns=COLUMNS)
    return results.astype({"review_id": int})


if __name__ == "__main__":
    example_reviews = [
        "The fried rice was amazing, but the waiter was rude.",
        "Great green curry and a lovely patio. A bit expensive though.",
    ]
    print(absa(example_reviews))
