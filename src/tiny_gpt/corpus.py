"""Pretraining corpus prep: exact dedup, heuristic quality, eval contamination.

Near-duplicate MinHash and a learned quality classifier are not in this module.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from tiny_gpt.data import jsonl_rows, jsonl_texts

DEFAULT_MIN_CHARS = 200
DEFAULT_MAX_SYMBOL_RATIO = 0.3
DEFAULT_MAX_URL_RATIO = 0.3
DEFAULT_MAX_REPEAT_RATIO = 0.3
DEFAULT_CONTAM_N = 8
_REPEAT_NGRAM_N = 5
_MIN_REPEAT_LINES = 4
_WS = re.compile(r"\s+")


def normalize_document(text: str) -> str:
    """Collapse whitespace and lowercase so exact-dup and n-grams share one form."""
    return _WS.sub(" ", text.strip().lower())


def document_words(text: str) -> list[str]:
    """Word tokens of ``normalize_document``."""
    normalized = normalize_document(text)
    if not normalized:
        return []
    return normalized.split(" ")


def word_ngrams(words: list[str], n: int) -> Iterator[tuple[str, ...]]:
    """Yield consecutive word n-grams. Empty when the document is shorter than ``n``."""
    if n <= 0 or len(words) < n:
        return
    for i in range(len(words) - n + 1):
        yield tuple(words[i : i + n])


def exact_fingerprint(text: str) -> str:
    """Stable hash of the normalized document."""
    return hashlib.sha256(normalize_document(text).encode("utf-8")).hexdigest()


def _duplicate_ratio(items: list) -> float:
    """Share of items that are copies of an earlier item. 0 when every item is unique."""
    if not items:
        return 0.0
    return 1 - (len(set(items)) / len(items))


def quality_reject_reason(
    text: str,
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_symbol_ratio: float = DEFAULT_MAX_SYMBOL_RATIO,
    max_url_ratio: float = DEFAULT_MAX_URL_RATIO,
    max_repeat_ratio: float = DEFAULT_MAX_REPEAT_RATIO,
) -> str | None:
    """Return a drop reason, or None when the document passes heuristic checks."""
    stripped = text.strip()
    if len(stripped) < min_chars:
        return "too_short"

    symbol_chars = sum(1 for ch in stripped if not ch.isalnum() and not ch.isspace())
    if symbol_chars / len(stripped) > max_symbol_ratio:
        return "symbol_ratio"

    words = document_words(text)
    if words:
        url_words = sum(1 for word in words if word.startswith(("http://", "https://", "www.")))
        if url_words / len(words) > max_url_ratio:
            return "url_ratio"

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(lines) >= _MIN_REPEAT_LINES and _duplicate_ratio(lines) > max_repeat_ratio:
        return "repetitive"

    grams = list(word_ngrams(words, _REPEAT_NGRAM_N))
    if _duplicate_ratio(grams) > max_repeat_ratio:
        return "repetitive"
    return None


def build_eval_ngrams(texts: Iterable[str], n: int) -> set[tuple[str, ...]]:
    """Word n-grams from the fixed eval set. Training docs that contain one are dropped."""
    grams: set[tuple[str, ...]] = set()
    for text in texts:
        grams.update(word_ngrams(document_words(text), n))
    return grams


def is_contaminated(text: str, eval_ngrams: set[tuple[str, ...]], n: int) -> bool:
    """True when any ``n``-gram of ``text`` is in the eval index."""
    if not eval_ngrams:
        return False
    return any(gram in eval_ngrams for gram in word_ngrams(document_words(text), n))


def classify_document(
    text: str,
    seen_fingerprints: set[str],
    eval_ngrams: set[tuple[str, ...]],
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_symbol_ratio: float = DEFAULT_MAX_SYMBOL_RATIO,
    max_url_ratio: float = DEFAULT_MAX_URL_RATIO,
    max_repeat_ratio: float = DEFAULT_MAX_REPEAT_RATIO,
    contam_n: int = DEFAULT_CONTAM_N,
) -> str | None:
    """Drop reason for one document, in pipeline order: exact dup, quality, contamination.

    The fingerprint is recorded on first sight, including documents later dropped for
    quality or contamination, so a later copy is an exact duplicate.
    """
    fingerprint = exact_fingerprint(text)
    if fingerprint in seen_fingerprints:
        return "exact_dup"
    seen_fingerprints.add(fingerprint)

    reason = quality_reject_reason(
        text,
        min_chars=min_chars,
        max_symbol_ratio=max_symbol_ratio,
        max_url_ratio=max_url_ratio,
        max_repeat_ratio=max_repeat_ratio,
    )
    if reason is not None:
        return reason
    if is_contaminated(text, eval_ngrams, contam_n):
        return "contaminated"
    return None


@dataclass
class PrepareStats:
    """Counts from one ``prepare_jsonl`` pass."""

    seen: int = 0
    kept: int = 0
    dropped: dict[str, int] = field(default_factory=dict)

    def record(self, reason: str | None) -> None:
        self.seen += 1
        if reason is None:
            self.kept += 1
            return
        self.dropped[reason] = self.dropped.get(reason, 0) + 1


def prepare_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    eval_path: str | Path | None = None,
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_symbol_ratio: float = DEFAULT_MAX_SYMBOL_RATIO,
    max_url_ratio: float = DEFAULT_MAX_URL_RATIO,
    max_repeat_ratio: float = DEFAULT_MAX_REPEAT_RATIO,
    contam_n: int = DEFAULT_CONTAM_N,
) -> PrepareStats:
    """Write kept ``{"text": ...}`` rows and return drop counts.

    ``eval_path`` uses the same ``text`` field as training JSONL. Omit it to skip
    contamination checking.
    """
    eval_ngrams = (
        build_eval_ngrams(jsonl_texts(eval_path), contam_n) if eval_path is not None else set()
    )
    stats = PrepareStats()
    seen: set[str] = set()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8") as out:
        for row in jsonl_rows(input_path):
            if not isinstance(row, dict):
                continue
            text = row.get("text")
            if not isinstance(text, str) or not text:
                continue
            reason = classify_document(
                text,
                seen,
                eval_ngrams,
                min_chars=min_chars,
                max_symbol_ratio=max_symbol_ratio,
                max_url_ratio=max_url_ratio,
                max_repeat_ratio=max_repeat_ratio,
                contam_n=contam_n,
            )
            stats.record(reason)
            if reason is None:
                out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    return stats
