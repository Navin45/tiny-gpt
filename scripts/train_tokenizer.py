#!/usr/bin/env python3
"""Train the byte-level BPE tokenizer from a JSONL corpus."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from tiny_gpt.data import jsonl_texts
from tiny_gpt.tokenizer import train_bpe


def _update_config_vocab_size(config_path: Path, vocab_size: int) -> None:
    """Overwrite the top-level ``vocab_size:`` line in a flat training YAML."""
    text = config_path.read_text(encoding="utf-8")
    updated, n = re.subn(
        r"(?m)^(vocab_size:\s*)\d+\s*$",
        rf"\g<1>{vocab_size}",
        text,
        count=1,
    )
    if n != 1:
        raise SystemExit(f"Could not find a top-level vocab_size: integer in {config_path}")
    config_path.write_text(updated, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL file with a 'text' field per line")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--config",
        required=True,
        help="Training YAML whose vocab_size is the BPE target, then rewritten to the actual size",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "vocab_size" not in raw:
        raise SystemExit(f"Config {config_path} must contain a top-level vocab_size")
    target_vocab_size = int(raw["vocab_size"])

    actual_vocab_size = train_bpe(
        jsonl_texts(args.input), target_vocab_size, args.output_dir
    )
    _update_config_vocab_size(config_path, actual_vocab_size)

    print(f"Tokenizer saved under {args.output_dir}")
    print(
        f"Updated {config_path}: vocab_size {target_vocab_size} -> {actual_vocab_size}"
    )


if __name__ == "__main__":
    main()
