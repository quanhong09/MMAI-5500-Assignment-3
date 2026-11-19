"""Fine-tune the MMAI 5400 Assignment 3 ABSA pipeline.

Trains and saves two SetFit models from the Label Studio JSON-MIN export:

1. A SetFit ``AbsaModel`` that extracts aspect spans and classifies their
   sentiment polarity (saved as ``models/absa-aspect`` and
   ``models/absa-polarity``).
2. A standard ``SetFitModel`` that classifies each span into its aspect
   category, using ``f"{span}: {text}"`` inputs (saved as
   ``models/category``).

Usage:
    python finetuning.py
    python finetuning.py --data labeled_reviews.json --epochs 1
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import Dataset
from setfit import AbsaModel, AbsaTrainer, SetFitModel, Trainer
from setfit import TrainingArguments

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPACY_MODEL = "en_core_web_sm"
MODEL_DIR = Path(__file__).resolve().parent / "models"
ABSA_PATH = MODEL_DIR / "absa"  # saved as absa-aspect and absa-polarity
CATEGORY_PATH = MODEL_DIR / "category"
SEED = 42


def load_labelstudio_data(json_path):
    """Convert Label Studio JSON-MIN export into ABSA and Category datasets."""
    polarity_map = {
        "Pos": "Positive",
        "Neg": "Negative",
    }

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    absa_rows = []
    category_rows = []

    for item in data:
        text = item.get("Review") or item.get("text", "")
        for ann in item.get("label", []):
            span = ann["text"]
            start = ann["start"]
            category, pol_short = ann["labels"][0].split("-")
            polarity = polarity_map.get(pol_short, pol_short)

            # Calculate 0-indexed occurrence (ordinal) of span in text
            ordinal = text[:start].count(span)

            absa_rows.append({
                "text": text,
                "span": span,
                "label": polarity,
                "ordinal": ordinal,
            })
            category_rows.append({
                "text": f"{span}: {text}",
                "label": category,
            })

    return Dataset.from_list(absa_rows), Dataset.from_list(category_rows)


def make_args(epochs):
    """Return SetFit training arguments shared by both models."""
    return TrainingArguments(
        batch_size=16,
        num_epochs=epochs,
        seed=SEED,
        report_to="none",
    )


def train_absa_model(absa_dataset, base_model, epochs):
    """Fine-tune and save the aspect extraction + polarity AbsaModel."""
    model = AbsaModel.from_pretrained(base_model, spacy_model=SPACY_MODEL)
    trainer = AbsaTrainer(
        model,
        args=make_args(epochs),
        train_dataset=absa_dataset,
    )
    trainer.train()
    model.save_pretrained(ABSA_PATH)
    print(f"Saved AbsaModel to {ABSA_PATH}-aspect and {ABSA_PATH}-polarity")


def train_category_model(category_dataset, base_model, epochs):
    """Fine-tune and save the aspect category SetFitModel."""
    labels = sorted(set(category_dataset["label"]))
    model = SetFitModel.from_pretrained(base_model, labels=labels)
    trainer = Trainer(
        model=model,
        args=make_args(epochs),
        train_dataset=category_dataset,
    )
    trainer.train()
    model.save_pretrained(CATEGORY_PATH)
    print(f"Saved category SetFitModel to {CATEGORY_PATH}")


def main():
    """Load the labelled data, then train and save both models."""
    parser = argparse.ArgumentParser(description="Fine-tune the A3 models.")
    parser.add_argument("--data", default="labeled_reviews.json",
                        help="Label Studio JSON-MIN export")
    parser.add_argument("--base-model", default=BASE_MODEL,
                        help="sentence transformer to fine-tune")
    parser.add_argument("--epochs", type=int, default=1,
                        help="training epochs for each model (default: 1)")
    args = parser.parse_args()

    absa_dataset, category_dataset = load_labelstudio_data(args.data)
    print(f"{len(absa_dataset)} labelled spans")
    print("Sentiment counts:", dict(Counter(absa_dataset["label"])))
    print("Category counts:", dict(Counter(category_dataset["label"])))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_absa_model(absa_dataset, args.base_model, args.epochs)
    train_category_model(category_dataset, args.base_model, args.epochs)


if __name__ == "__main__":
    main()
