"""Sentiment analysis with lightweight and optional model-backed engines."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from .preprocessing import tokenize


POSITIVE_WORDS = {
    "amazing",
    "awesome",
    "beautiful",
    "best",
    "brilliant",
    "delightful",
    "easy",
    "excellent",
    "fantastic",
    "fast",
    "good",
    "great",
    "happy",
    "helpful",
    "impressive",
    "love",
    "loved",
    "perfect",
    "recommend",
    "reliable",
    "satisfied",
    "smooth",
    "strong",
    "wonderful",
}

NEGATIVE_WORDS = {
    "awful",
    "bad",
    "broken",
    "confusing",
    "disappointing",
    "difficult",
    "fail",
    "failed",
    "hate",
    "hated",
    "horrible",
    "poor",
    "problem",
    "slow",
    "terrible",
    "ugly",
    "unhappy",
    "unreliable",
    "useless",
    "weak",
    "worst",
}

NEGATIONS = {"no", "not", "never", "none", "hardly", "barely", "without"}
INTENSIFIERS = {
    "absolutely": 1.5,
    "extremely": 1.5,
    "highly": 1.3,
    "really": 1.25,
    "so": 1.15,
    "too": 1.15,
    "very": 1.3,
}


@dataclass(frozen=True)
class SentimentResult:
    """Normalized sentiment output."""

    text: str
    label: str
    score: float
    positive: float
    neutral: float
    negative: float
    backend: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SentimentAnalyzer:
    """Analyze sentiment using ``lexicon``, ``vader``, or ``transformers``."""

    def __init__(
        self,
        backend: str = "lexicon",
        *,
        model_name: Optional[str] = None,
        positive_threshold: float = 0.05,
        negative_threshold: float = -0.05,
        **kwargs: Any,
    ) -> None:
        backend = backend.lower()
        if backend not in {"lexicon", "vader", "transformers"}:
            raise ValueError("backend must be 'lexicon', 'vader', or 'transformers'")
        self.backend = backend
        self.model_name = model_name
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.kwargs = kwargs
        self._engine: Any = None

    def _label(self, score: float) -> str:
        if score >= self.positive_threshold:
            return "positive"
        if score <= self.negative_threshold:
            return "negative"
        return "neutral"

    def _analyze_lexicon(self, text: str) -> SentimentResult:
        tokens = [token.lower() for token in tokenize(text)]
        raw_score = 0.0
        positive_hits = 0.0
        negative_hits = 0.0

        for index, token in enumerate(tokens):
            polarity = 1.0 if token in POSITIVE_WORDS else -1.0 if token in NEGATIVE_WORDS else 0.0
            if polarity == 0:
                continue

            window = tokens[max(0, index - 3) : index]
            if any(previous in NEGATIONS for previous in window):
                polarity *= -1
            if index > 0:
                polarity *= INTENSIFIERS.get(tokens[index - 1], 1.0)

            raw_score += polarity
            if polarity > 0:
                positive_hits += polarity
            else:
                negative_hits += abs(polarity)

        score = (
            raw_score / math.sqrt(raw_score * raw_score + 4)
            if raw_score
            else 0.0
        )
        positive = max(score, 0.0)
        negative = max(-score, 0.0)
        neutral = max(0.0, 1.0 - abs(score))

        return SentimentResult(
            text=text,
            label=self._label(score),
            score=float(score),
            positive=float(positive),
            neutral=float(neutral),
            negative=float(negative),
            backend=self.backend,
        )

    def _load_vader(self):
        if self._engine is None:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            except ImportError as exc:
                raise ImportError(
                    "VADER sentiment requires the NLP extra: "
                    "pip install 'M_Ana_package[nlp]'"
                ) from exc
            self._engine = SentimentIntensityAnalyzer()
        return self._engine

    def _analyze_vader(self, text: str) -> SentimentResult:
        scores = self._load_vader().polarity_scores(text)
        return SentimentResult(
            text=text,
            label=self._label(scores["compound"]),
            score=float(scores["compound"]),
            positive=float(scores["pos"]),
            neutral=float(scores["neu"]),
            negative=float(scores["neg"]),
            backend=self.backend,
        )

    def _load_transformer(self):
        if self._engine is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise ImportError(
                    "Transformer sentiment requires the NLP extra: "
                    "pip install 'M_Ana_package[nlp]'"
                ) from exc
            self._engine = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                **self.kwargs,
            )
        return self._engine

    def _analyze_transformer(self, text: str) -> SentimentResult:
        prediction = self._load_transformer()(text)[0]
        label_value = str(prediction["label"]).lower()
        confidence = float(prediction["score"])
        if "neg" in label_value or label_value in {"0", "label_0"}:
            score, label = -confidence, "negative"
        elif "pos" in label_value or label_value in {"1", "label_1"}:
            score, label = confidence, "positive"
        else:
            score, label = 0.0, "neutral"
        return SentimentResult(
            text=text,
            label=label,
            score=score,
            positive=confidence if label == "positive" else 0.0,
            neutral=confidence if label == "neutral" else 1.0 - confidence,
            negative=confidence if label == "negative" else 0.0,
            backend=self.backend,
        )

    def analyze(self, text: object) -> SentimentResult:
        """Analyze one text value."""
        value = "" if text is None else str(text)
        if self.backend == "lexicon":
            return self._analyze_lexicon(value)
        if self.backend == "vader":
            return self._analyze_vader(value)
        return self._analyze_transformer(value)

    def transform(self, texts: Sequence[object]) -> List[SentimentResult]:
        """Analyze several text values."""
        return [self.analyze(text) for text in texts]

    def to_dataframe(self, texts: Sequence[object]) -> pd.DataFrame:
        """Analyze texts and return one result row per input."""
        return pd.DataFrame([result.to_dict() for result in self.transform(texts)])


def analyze_sentiment(
    text: Union[object, Sequence[object]],
    backend: str = "lexicon",
    **kwargs: Any,
):
    """Convenience sentiment function for one value or a sequence."""
    analyzer = SentimentAnalyzer(backend=backend, **kwargs)
    if isinstance(text, (str, bytes)) or not isinstance(text, Iterable):
        return analyzer.analyze(text)
    return analyzer.transform(list(text))
