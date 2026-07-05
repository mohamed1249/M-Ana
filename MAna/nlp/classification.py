"""Leakage-safe classical and transformer text classification helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_text_classifier(
    text_column: str,
    *,
    numeric_columns: Optional[Sequence[str]] = None,
    categorical_columns: Optional[Sequence[str]] = None,
    estimator: Optional[Any] = None,
    max_features: Optional[int] = 50_000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 1,
    stop_words: Optional[str] = None,
    random_state: int = 42,
) -> Pipeline:
    """Build a sparse mixed-feature classifier without preprocessing leakage.

    Fit the returned pipeline only on the training split. TF-IDF vocabulary,
    imputation, scaling, and category encoding are all learned inside ``fit``.
    """
    if not text_column:
        raise ValueError("text_column is required")

    transformers = [
        (
            "text",
            TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                stop_words=stop_words,
                sublinear_tf=True,
            ),
            text_column,
        )
    ]
    numeric_columns = list(numeric_columns or [])
    categorical_columns = list(categorical_columns or [])

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", _one_hot_encoder()),
                    ]
                ),
                categorical_columns,
            )
        )

    classifier = estimator or LogisticRegression(
        max_iter=1_000,
        class_weight="balanced",
        random_state=random_state,
    )
    return Pipeline(
        [
            (
                "features",
                ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.2),
            ),
            ("classifier", classifier),
        ]
    )


def evaluate_classifier(model: Any, features: Any, targets: Sequence[Any]) -> Dict[str, Any]:
    """Return balanced classification metrics and a per-class report."""
    predictions = model.predict(features)
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "precision_macro": float(
            precision_score(targets, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(targets, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(targets, predictions, average="weighted", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(targets, predictions),
        "report": classification_report(
            targets, predictions, output_dict=True, zero_division=0
        ),
        "predictions": np.asarray(predictions),
    }


class TransformerClassifier:
    """Thin Hugging Face/PyTorch classifier with dynamic padding."""

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        *,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        max_length: int = 512,
        device: str = "auto",
    ) -> None:
        os.environ.setdefault("USE_TF", "0")
        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                DataCollatorWithPadding,
            )
        except ImportError as exc:
            raise ImportError(
                "Transformer classification requires the NLP extra: "
                "pip install 'M_Ana_package[nlp]'"
            ) from exc

        self.torch = torch
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    def make_loader(
        self,
        texts: Sequence[object],
        labels: Optional[Sequence[int]] = None,
        *,
        batch_size: int = 16,
        shuffle: bool = False,
    ):
        """Create a dynamically padded PyTorch DataLoader."""
        if labels is not None and len(labels) != len(texts):
            raise ValueError("labels must have the same length as texts")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        tokenizer = self.tokenizer
        max_length = self.max_length
        torch = self.torch

        class TextDataset(torch.utils.data.Dataset):
            def __len__(self):
                return len(texts)

            def __getitem__(self, index):
                item = tokenizer(
                    str(texts[index]),
                    truncation=True,
                    max_length=max_length,
                )
                if labels is not None:
                    item["labels"] = int(labels[index])
                return item

        return torch.utils.data.DataLoader(
            TextDataset(),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collator,
        )

    def train_epoch(self, loader: Any, *, max_grad_norm: float = 1.0) -> float:
        """Train for one epoch and return mean loss."""
        self.model.train()
        total_loss = 0.0
        for batch in loader:
            batch = {key: value.to(self.device) for key, value in batch.items()}
            self.optimizer.zero_grad()
            outputs = self.model(**batch)
            outputs.loss.backward()
            self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()
            total_loss += float(outputs.loss.detach().cpu())
        return total_loss / max(len(loader), 1)

    def evaluate(self, loader: Any) -> Dict[str, Any]:
        """Evaluate a labeled loader."""
        self.model.eval()
        predictions, labels = [], []
        total_loss = 0.0
        with self.torch.no_grad():
            for batch in loader:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                outputs = self.model(**batch)
                total_loss += float(outputs.loss.detach().cpu())
                predictions.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
                labels.extend(batch["labels"].cpu().tolist())
        metrics = evaluate_classifier(_PredictionModel(predictions), None, labels)
        metrics["loss"] = total_loss / max(len(loader), 1)
        return metrics

    def predict(self, texts: Sequence[object], *, batch_size: int = 32) -> Dict[str, np.ndarray]:
        """Run batched inference and return labels and probabilities."""
        loader = self.make_loader(texts, batch_size=batch_size)
        self.model.eval()
        logits = []
        with self.torch.no_grad():
            for batch in loader:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                logits.append(self.model(**batch).logits.cpu())
        if not logits:
            return {
                "labels": np.asarray([], dtype=int),
                "probabilities": np.empty((0, self.num_labels)),
            }
        values = self.torch.cat(logits)
        return {
            "labels": values.argmax(dim=-1).numpy(),
            "probabilities": self.torch.softmax(values, dim=-1).numpy(),
        }

    def save(self, path: str) -> None:
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(destination)
        self.tokenizer.save_pretrained(destination)


class _PredictionModel:
    """Adapter used to reuse classical metric calculation."""

    def __init__(self, predictions: Sequence[int]) -> None:
        self.predictions = np.asarray(predictions)

    def predict(self, features: Any) -> np.ndarray:
        return self.predictions
