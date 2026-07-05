"""Text cleaning, normalization, and tokenization utilities."""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Set, Union

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_HTML_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)


def normalize_unicode(text: str, form: str = "NFKC") -> str:
    """Normalize Unicode text using one of Python's Unicode forms."""
    if form not in {"NFC", "NFD", "NFKC", "NFKD"}:
        raise ValueError("form must be one of: NFC, NFD, NFKC, NFKD")
    return unicodedata.normalize(form, str(text))


def strip_accents(text: str) -> str:
    """Remove combining accent marks while retaining base characters."""
    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(char for char in normalized if not unicodedata.combining(char))


def resolve_stop_words(
    stop_words: Optional[Union[str, Iterable[str]]],
    preserve_negations: bool = True,
) -> Set[str]:
    """Resolve a stop-word configuration into a normalized set."""
    if stop_words is None:
        return set()
    if isinstance(stop_words, str):
        if stop_words.lower() != "english":
            raise ValueError("string stop_words must be 'english' or None")
        resolved = set(ENGLISH_STOP_WORDS)
    else:
        resolved = {str(word).lower() for word in stop_words}

    if preserve_negations:
        resolved.difference_update({"no", "nor", "not", "never", "none"})
    return resolved


def tokenize(
    text: str,
    lowercase: bool = False,
    stop_words: Optional[Union[str, Iterable[str]]] = None,
    min_token_length: int = 1,
    preserve_negations: bool = True,
) -> List[str]:
    """Tokenize text without requiring an external NLP model."""
    if min_token_length < 1:
        raise ValueError("min_token_length must be at least 1")

    value = normalize_unicode(text)
    if lowercase:
        value = value.lower()
    tokens = _TOKEN_RE.findall(value)
    excluded = resolve_stop_words(stop_words, preserve_negations)
    return [
        token
        for token in tokens
        if len(token) >= min_token_length and token.lower() not in excluded
    ]


def clean_text(
    text: object,
    *,
    lowercase: bool = True,
    remove_html: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_punctuation: bool = True,
    remove_numbers: bool = False,
    strip_diacritics: bool = False,
    stop_words: Optional[Union[str, Iterable[str]]] = None,
    min_token_length: int = 1,
    preserve_negations: bool = True,
) -> str:
    """Clean one text value using a predictable, dependency-light pipeline."""
    if text is None:
        return ""

    value = html.unescape(normalize_unicode(str(text)))
    if remove_html:
        value = _HTML_RE.sub(" ", value)
    if remove_urls:
        value = _URL_RE.sub(" ", value)
    if remove_emails:
        value = _EMAIL_RE.sub(" ", value)
    if strip_diacritics:
        value = strip_accents(value)
    if lowercase:
        value = value.lower()
    if remove_numbers:
        value = re.sub(r"\d+", " ", value)
    if remove_punctuation:
        value = re.sub(r"[^\w\s'’-]", " ", value, flags=re.UNICODE)
        value = value.replace("_", " ")

    value = _WHITESPACE_RE.sub(" ", value).strip()
    if stop_words is not None or min_token_length > 1:
        value = " ".join(
            tokenize(
                value,
                stop_words=stop_words,
                min_token_length=min_token_length,
                preserve_negations=preserve_negations,
            )
        )
    return value


def clean_corpus(texts: Sequence[object], **kwargs) -> List[str]:
    """Clean a sequence of text values."""
    return [clean_text(text, **kwargs) for text in texts]


@dataclass
class TextCleaner:
    """Reusable text-cleaning configuration."""

    lowercase: bool = True
    remove_html: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_punctuation: bool = True
    remove_numbers: bool = False
    strip_diacritics: bool = False
    stop_words: Optional[Union[str, Iterable[str]]] = None
    min_token_length: int = 1
    preserve_negations: bool = True
    _stop_words: Set[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._stop_words = resolve_stop_words(
            self.stop_words, preserve_negations=self.preserve_negations
        )

    def clean(self, text: object) -> str:
        """Clean one value."""
        return clean_text(
            text,
            lowercase=self.lowercase,
            remove_html=self.remove_html,
            remove_urls=self.remove_urls,
            remove_emails=self.remove_emails,
            remove_punctuation=self.remove_punctuation,
            remove_numbers=self.remove_numbers,
            strip_diacritics=self.strip_diacritics,
            stop_words=self._stop_words,
            min_token_length=self.min_token_length,
            preserve_negations=self.preserve_negations,
        )

    def transform(self, texts: Sequence[object]) -> List[str]:
        """Clean a sequence of values."""
        return [self.clean(text) for text in texts]

    def tokenize(self, text: object) -> List[str]:
        """Clean and tokenize one value."""
        return tokenize(
            self.clean(text),
            stop_words=None,
            min_token_length=self.min_token_length,
            preserve_negations=self.preserve_negations,
        )

    def __call__(self, text: object) -> str:
        return self.clean(text)
