#!/usr/bin/env python3
"""Encode a JSONL corpus into a flat token binary."""

import argparse

from tiny_gpt.data import write_token_bin
from tiny_gpt.tokenizer import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)
    write_token_bin(tokenizer, args.input, args.output)
    print(f"Encoded corpus written to {args.output}")


if __name__ == "__main__":
    main()
