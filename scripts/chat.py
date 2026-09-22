#!/usr/bin/env python3
"""Tiny local chat CLI for a trained checkpoint."""

from __future__ import annotations

import argparse

import torch
import yaml

from tiny_gpt.config import Config
from tiny_gpt.model import TinyGPT
from tiny_gpt.tokenizer import load_tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--max-new-tokens", type=int, default=256)
    args = p.parse_args()

    cfg = Config.from_dict(yaml.safe_load(open(args.config)))
    tok = load_tokenizer(args.tokenizer)
    assert cfg.model.vocab_size == tok.get_vocab_size(), (
        f"config vocab_size={cfg.model.vocab_size} != tokenizer "
        f"vocab_size={tok.get_vocab_size()}; re-run train_tokenizer.py "
        f"with this --config or fix the mismatch"
    )
    model = TinyGPT(cfg.model)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    print("Type Ctrl+C to exit.")
    while True:
        user = input("you> ")
        prompt = f"<|user|>{user}<|eot|><|assistant|>"
        ids = torch.tensor([tok.encode(prompt).ids], device=device)
        eot_id = tok.token_to_id("<|eot|>")
        out = model.generate(
            ids,
            max_new_tokens=args.max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=eot_id,
        )
        text = tok.decode(out[0, ids.size(1) :].tolist())
        print(f"assistant> {text}")


if __name__ == "__main__":
    main()
