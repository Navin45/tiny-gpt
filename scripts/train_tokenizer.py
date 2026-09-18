#!/usr/bin/env python3
"""Train the byte-level BPE tokenizer from a JSONL corpus."""

import argparse

from tiny_gpt.data import jsonl_texts
from tiny_gpt.tokenizer import train_bpe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL file with a 'text' field per line")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vocab-size", type=int, default=32_000)
    args = parser.parse_args()
    train_bpe(jsonl_texts(args.input), args.vocab_size, args.output_dir)
    print(f"Tokenizer saved under {args.output_dir}")


if __name__ == "__main__":
    main()
