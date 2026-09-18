#!/usr/bin/env python3
"""Supervised fine-tuning entry point.

Input JSONL lines must contain a `messages` array. Demonstrates assistant-only loss masking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from tiny_gpt.checkpoint import load_checkpoint, save_checkpoint
from tiny_gpt.config import Config
from tiny_gpt.model import TinyGPT
from tiny_gpt.posttrain import next_token_labels, render_messages
from tiny_gpt.tokenizer import load_tokenizer


class SFTDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_len: int):
        self.rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        ids, mask = render_messages(self.tokenizer, self.rows[idx]["messages"], self.max_len)
        labels = next_token_labels(ids, mask, len(ids))
        return torch.tensor(ids), torch.tensor(labels)


def make_collate(pad_id: int):
    def collate(batch):
        max_len = max(x[0].numel() for x in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i, (ids, y) in enumerate(batch):
            input_ids[i, : ids.numel()] = ids
            labels[i, : y.numel()] = y
        return input_ids, labels

    return collate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--resume", default=None, help="Resume SFT optimizer/model from this checkpoint")
    p.add_argument(
        "--output",
        default="artifacts/checkpoints/sft/final.pt",
        help="Where to write the SFT checkpoint",
    )
    args = p.parse_args()

    cfg = Config.from_dict(yaml.safe_load(open(args.config)))
    tokenizer = load_tokenizer(args.tokenizer)
    cfg.model.vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("<|pad|>")
    if pad_id is None:
        raise SystemExit("Tokenizer is missing <|pad|>")

    model = TinyGPT(cfg.model)
    payload = torch.load(args.base_checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, opt, map_location=device) + 1
        print(f"resumed SFT from {args.resume} at step={start_step}")

    ds = SFTDataset(args.data, tokenizer, cfg.model.max_seq_len)
    loader = DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        collate_fn=make_collate(pad_id),
        shuffle=True,
    )

    step = start_step
    while step < args.steps:
        for x, y in loader:
            if step >= args.steps:
                break
            x, y = x.to(device), y.to(device)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                loss, aux = model(x, targets=y)
                total = loss + cfg.model.moe_aux_loss_coef * aux
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
            opt.step()
            if step % 10 == 0:
                print(f"sft_step={step:05d} loss={total.item():.4f}")
            step += 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(str(out), model, opt, step)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
