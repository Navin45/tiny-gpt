"""Data loading and sequence packing for language-model training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def jsonl_texts(path: str | Path):
    """Yield text fields from a JSONL corpus."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text")
            if isinstance(text, str) and text:
                yield text


def document_ids(input_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """Assign a document id to each position; a new doc starts after each EOS."""
    is_eos = input_ids == eos_token_id
    doc = torch.zeros_like(input_ids, dtype=torch.long)
    if input_ids.size(1) > 1:
        doc[:, 1:] = torch.cumsum(is_eos[:, :-1].to(torch.long), dim=1)
    return doc


def document_causal_mask(input_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """Bool attention mask (B, 1, T, T) for SDPA: True = allowed to attend.

    Tokens may only attend to earlier-or-equal positions inside the same document.
    Documents are delimited by ``eos_token_id`` in the packed pretraining stream.

    Matches ``scaled_dot_product_attention``'s bool convention (True keeps the score;
    False positions are filled with -inf).
    """
    bsz, seq_len = input_ids.shape
    doc = document_ids(input_ids, eos_token_id)
    same_doc = doc[:, None, :] == doc[:, :, None]
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device))
    allowed = same_doc & causal.unsqueeze(0)
    return allowed.unsqueeze(1)


class TokenDataset(Dataset):
    """Random fixed-length windows sampled from a flat int32 token file.

    Call ``set_epoch(n)`` between passes so epoch 2 does not replay epoch 1's windows.
    A packed token file makes the training loop cheap: each sample is a contiguous slice.
    """

    def __init__(self, token_bin: str, seq_len: int, seed: int = 1337, epoch: int = 0) -> None:
        self.tokens = np.memmap(token_bin, dtype=np.int32, mode="r")
        self.seq_len = seq_len
        self.seed = seed
        self.epoch = epoch

    def set_epoch(self, epoch: int) -> None:
        """Advance the sampling epoch so window offsets reshuffle reproducibly."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return max(1, len(self.tokens) // self.seq_len)

    def __getitem__(self, idx: int):
        # Mix seed, epoch, and index so multi-epoch runs see fresh windows.
        rng = np.random.default_rng(self.seed + idx + self.epoch * 1_000_000_007)
        start = int(rng.integers(0, len(self.tokens) - self.seq_len - 1))
        chunk = np.asarray(self.tokens[start : start + self.seq_len + 1], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return x, y


def write_token_bin(tokenizer, input_jsonl: str, output_bin: str) -> None:
    """Encode JSONL text into one flat int32 token array separated by EOS.

    Documents are packed back-to-back with ``<|eos|>``. Enable
    ``mask_document_boundaries`` in the model config so attention cannot cross those
    boundaries during pretraining.
    """
    output_path = Path(output_bin)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    eos_id = tokenizer.token_to_id("<|eos|>")
    if eos_id is None:
        raise ValueError("Tokenizer does not contain <|eos|>")

    with open(output_path, "wb") as out:
        for text in jsonl_texts(input_jsonl):
            ids = tokenizer.encode(text).ids
            np.asarray([*ids, eos_id], dtype=np.int32).tofile(out)
