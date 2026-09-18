"""Tokenizer training/loading utilities.

Uses the Hugging Face `tokenizers` library (Rust BPE) for byte-level BPE. The language model
itself is plain PyTorch in this repo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKENS = [
    "<|pad|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|eot|>",
]


def train_bpe(texts: Iterable[str], vocab_size: int, output_dir: str) -> None:
    """Train a byte-level BPE tokenizer from plain-text strings and save tokenizer.json.

    Pass an iterator of document strings (e.g. via ``jsonl_texts``), never raw JSONL file
    paths. The Hugging Face ``Tokenizer.train(files, ...)`` API reads each line of a file as
    literal text and would otherwise burn vocabulary on JSON syntax.
    """
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<|unk|>", *SPECIAL_TOKENS],
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out / "tokenizer.json"))
    (out / "special_tokens.txt").write_text("\n".join(SPECIAL_TOKENS), encoding="utf-8")


def load_tokenizer(path: str | Path) -> Tokenizer:
    """Load a previously trained tokenizer."""
    return Tokenizer.from_file(str(path))
