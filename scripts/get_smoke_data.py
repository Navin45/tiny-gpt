#!/usr/bin/env python3
"""Download a tiny public-domain corpus for smoke-testing the training pipeline.

This is plumbing validation data only — not a real pretraining mix.
Writes artifacts/data/train.jsonl and artifacts/data/val.jsonl as {"text": "..."} lines.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="artifacts/data")
    p.add_argument("--val-fraction", type=float, default=0.05)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"downloading {URL}")
    text = urllib.request.urlopen(URL, timeout=60).read().decode("utf-8")
    lines = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    if len(lines) < 2:
        raise SystemExit("Downloaded text produced too few chunks")

    split = max(1, int(len(lines) * (1.0 - args.val_fraction)))
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for line in lines[:split]:
            f.write(json.dumps({"text": line}, ensure_ascii=False) + "\n")
    with val_path.open("w", encoding="utf-8") as f:
        for line in lines[split:]:
            f.write(json.dumps({"text": line}, ensure_ascii=False) + "\n")

    print(f"wrote {train_path} ({split} docs)")
    print(f"wrote {val_path} ({len(lines) - split} docs)")


if __name__ == "__main__":
    main()
