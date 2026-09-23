#!/usr/bin/env python3
"""Filter a pretraining JSONL corpus before tokenizer training and encoding.

Runs exact dedup, heuristic quality checks, then optional eval-set contamination
removal. Does not run MinHash near-dedup or a learned quality classifier.
"""

from __future__ import annotations

import argparse

from tiny_gpt.corpus import (
    DEFAULT_CONTAM_N,
    DEFAULT_MAX_REPEAT_RATIO,
    DEFAULT_MAX_SYMBOL_RATIO,
    DEFAULT_MAX_URL_RATIO,
    DEFAULT_MIN_CHARS,
    prepare_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL file with a 'text' field per line")
    parser.add_argument("--output", required=True, help="Cleaned JSONL of kept documents")
    parser.add_argument(
        "--eval",
        default=None,
        help="Optional eval JSONL ('text' field). Overlapping train docs are dropped",
    )
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    parser.add_argument("--max-symbol-ratio", type=float, default=DEFAULT_MAX_SYMBOL_RATIO)
    parser.add_argument("--max-url-ratio", type=float, default=DEFAULT_MAX_URL_RATIO)
    parser.add_argument("--max-repeat-ratio", type=float, default=DEFAULT_MAX_REPEAT_RATIO)
    parser.add_argument(
        "--contam-n",
        type=int,
        default=DEFAULT_CONTAM_N,
        help="Word n-gram length used to match train docs against the eval set",
    )
    args = parser.parse_args()

    stats = prepare_jsonl(
        args.input,
        args.output,
        args.eval,
        min_chars=args.min_chars,
        max_symbol_ratio=args.max_symbol_ratio,
        max_url_ratio=args.max_url_ratio,
        max_repeat_ratio=args.max_repeat_ratio,
        contam_n=args.contam_n,
    )
    drops = ", ".join(f"{reason}={count}" for reason, count in sorted(stats.dropped.items()))
    print(f"seen={stats.seen} kept={stats.kept}" + (f" dropped: {drops}" if drops else ""))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
