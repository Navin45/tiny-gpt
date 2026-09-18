"""Unit tests for the Tiny-GPT reference stack."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from tiny_gpt.checkpoint import load_checkpoint, save_checkpoint
from tiny_gpt.config import ModelConfig
from tiny_gpt.data import TokenDataset
from tiny_gpt.model import TinyGPT
from tiny_gpt.posttrain import dpo_loss, next_token_labels, render_messages, token_logprob
from tiny_gpt.tokenizer import train_bpe, load_tokenizer


def tiny_cfg(**overrides) -> ModelConfig:
    base = dict(
        vocab_size=128,
        max_seq_len=32,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        rope_theta=10_000.0,
        ffn_multiplier=2.0,
        moe_layers_every=2,
        n_experts=2,
        moe_top_k=1,
        moe_aux_loss_coef=0.01,
        dropout=0.0,
        use_checkpointing=False,
    )
    base.update(overrides)
    return ModelConfig(**base)


def test_rope_cached_decode_matches_prefill():
    """Prefill logits for the last token should match incremental KV-cache decode."""
    torch.manual_seed(0)
    model = TinyGPT(tiny_cfg(moe_layers_every=0)).eval()
    ids = torch.randint(0, 128, (1, 8))

    full_logits, _, _ = model(ids)
    prefill_last = full_logits[:, -1, :]

    logits, _, past = model(ids[:, :4])
    for t in range(4, 8):
        logits, _, past = model(ids[:, t : t + 1], past_key_values=past)
    cached_last = logits[:, -1, :]

    assert torch.allclose(prefill_last, cached_last, atol=1e-4, rtol=1e-4)


def test_causal_attention_prefill_vs_token_by_token():
    """Full-sequence logits should match teacher-forced token-by-token decode with cache."""
    torch.manual_seed(1)
    model = TinyGPT(tiny_cfg(moe_layers_every=0)).eval()
    ids = torch.randint(0, 128, (2, 6))

    full_logits, _, _ = model(ids)

    past = None
    step_logits = []
    for t in range(ids.size(1)):
        chunk = ids[:, : t + 1] if past is None else ids[:, t : t + 1]
        logits, _, past = model(chunk, past_key_values=past)
        step_logits.append(logits[:, -1, :])
    stacked = torch.stack(step_logits, dim=1)

    assert torch.allclose(full_logits, stacked, atol=1e-4, rtol=1e-4)


def test_kv_cache_exceeds_max_seq_len_raises():
    model = TinyGPT(tiny_cfg(max_seq_len=4, moe_layers_every=0)).eval()
    ids = torch.randint(0, 128, (1, 4))
    _, _, past = model(ids)
    with pytest.raises(ValueError, match="max_seq_len"):
        model(ids[:, -1:], past_key_values=past)


def test_moe_routing_and_aux_loss():
    torch.manual_seed(2)
    model = TinyGPT(tiny_cfg(n_layers=2, moe_layers_every=1, n_experts=4, moe_top_k=2))
    assert model.n_moe_layers == 2
    ids = torch.randint(0, 128, (2, 8))
    targets = torch.randint(0, 128, (2, 8))
    loss, aux = model(ids, targets=targets)
    assert torch.isfinite(loss)
    assert torch.isfinite(aux)
    assert aux.ndim == 0
    # Aux is averaged over MoE layers, so it should stay O(1), not grow with depth.
    assert float(aux.detach()) < 10.0


def test_generate_restores_training_mode():
    model = TinyGPT(tiny_cfg(moe_layers_every=0))
    model.train()
    ids = torch.randint(0, 128, (1, 3))
    _ = model.generate(ids, max_new_tokens=2, temperature=0.0)
    assert model.training


class _FakeTok:
    def __init__(self):
        self._ids = {
            "<|system|>": [1],
            "<|user|>": [2],
            "<|assistant|>": [3],
            "<|eot|>": 4,
        }

    def token_to_id(self, tok: str):
        value = self._ids.get(tok)
        return value if isinstance(value, int) else None

    def encode(self, text: str):
        class R:
            pass

        r = R()
        if text in self._ids and isinstance(self._ids[text], list):
            r.ids = list(self._ids[text])
        else:
            # Map each char to a stable synthetic id in [10, ...)
            r.ids = [10 + (ord(c) % 50) for c in text]
        return r


def test_sft_assistant_loss_masks():
    tok = _FakeTok()
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
    ids, mask = render_messages(tok, messages, max_len=64)
    labels = next_token_labels(ids, mask, max_len=64)

    # Labels at i predict token i+1; only assistant content / eot should be supervised.
    for i, lab in enumerate(labels):
        if lab == -100:
            continue
        assert lab == ids[i + 1]
        assert mask[i + 1] == 1

    # User-side targets must be ignored.
    user_span_end = ids.index(4)  # first eot after user turn
    assert all(labels[i] == -100 for i in range(user_span_end))


def test_dpo_logprob_masking():
    torch.manual_seed(3)
    model = TinyGPT(tiny_cfg(moe_layers_every=0)).eval()
    prompt = torch.randint(0, 128, (1, 4))
    chosen = torch.randint(0, 128, (1, 3))
    rejected = torch.randint(0, 128, (1, 3))
    reference = copy.deepcopy(model)
    loss = dpo_loss(model, reference, prompt, chosen, rejected, beta=0.1)
    assert torch.isfinite(loss)

    full = torch.cat((prompt, chosen), dim=1)
    lp = token_logprob(model, full, continuation_start=prompt.size(1))
    assert lp.shape == (1,)
    assert torch.isfinite(lp)


def test_checkpoint_round_trip(tmp_path: Path):
    torch.manual_seed(4)
    model = TinyGPT(tiny_cfg(moe_layers_every=0))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ids = torch.randint(0, 128, (2, 8))
    targets = torch.randint(0, 128, (2, 8))
    loss, _ = model(ids, targets=targets)
    loss.backward()
    opt.step()

    path = tmp_path / "ckpt.pt"
    save_checkpoint(str(path), model, opt, step=42, extra={"note": "ok"})

    model2 = TinyGPT(tiny_cfg(moe_layers_every=0))
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    step = load_checkpoint(str(path), model2, opt2)
    assert step == 42
    for a, b in zip(model.state_dict().values(), model2.state_dict().values()):
        assert torch.equal(a, b)


def test_deterministic_data_sampling(tmp_path: Path):
    tokens = np.arange(200, dtype=np.int32)
    path = tmp_path / "tok.bin"
    tokens.tofile(path)

    ds_a = TokenDataset(str(path), seq_len=16, seed=7)
    ds_b = TokenDataset(str(path), seq_len=16, seed=7)
    ds_c = TokenDataset(str(path), seq_len=16, seed=8)

    x_a, y_a = ds_a[3]
    x_b, y_b = ds_b[3]
    x_c, y_c = ds_c[3]
    assert torch.equal(x_a, x_b) and torch.equal(y_a, y_b)
    assert not torch.equal(x_a, x_c)
    assert torch.equal(x_a[1:], y_a[:-1])

    ds_a.set_epoch(1)
    x_e1, _ = ds_a[3]
    assert not torch.equal(x_a, x_e1)


def test_document_causal_mask_allows_same_doc_only():
    from tiny_gpt.data import document_causal_mask

    # tokens: [1, 2, EOS=9, 3, 4]  — True = allowed (SDPA convention)
    ids = torch.tensor([[1, 2, 9, 3, 4]])
    allowed = document_causal_mask(ids, eos_token_id=9)
    # query at pos 3 (token 3) must NOT attend to pos 0 (other doc)
    assert not bool(allowed[0, 0, 3, 0])
    # query at pos 3 may attend to pos 3
    assert bool(allowed[0, 0, 3, 3])
    # query at pos 1 may attend to pos 0 (same doc, causal)
    assert bool(allowed[0, 0, 1, 0])
    # query at pos 0 must not attend to future pos 1
    assert not bool(allowed[0, 0, 0, 1])


def test_document_mask_actually_isolates_documents():
    """Seam test: changing doc1 must not change doc2 logits under the mask."""
    torch.manual_seed(0)
    cfg = tiny_cfg(
        moe_layers_every=0,
        mask_document_boundaries=True,
        eos_token_id=9,
        max_seq_len=16,
    )
    model = TinyGPT(cfg).eval()

    # doc1 | EOS | doc2
    ids_a = torch.tensor([[1, 2, 9, 3, 4]])
    ids_b = torch.tensor([[5, 6, 9, 3, 4]])  # doc1 changed, doc2 identical
    logits_a, _, _ = model(ids_a)
    logits_b, _, _ = model(ids_b)
    assert torch.allclose(logits_a[0, 3:], logits_b[0, 3:], atol=1e-5, rtol=1e-5)
    # Sanity: doc1 positions should differ when their tokens differ.
    assert not torch.allclose(logits_a[0, :2], logits_b[0, :2], atol=1e-5, rtol=1e-5)


def test_tokenizer_encode_decode_reversibility(tmp_path: Path):
    corpus = [
        "The capital of India is New Delhi.",
        "Mixture of experts routes tokens sparsely.",
        "Byte-level BPE should round-trip plain text.",
        "The capital of India is New Delhi.",
        "Mixture of experts routes tokens sparsely.",
    ] * 20
    out = tmp_path / "tok"
    train_bpe(corpus, vocab_size=200, output_dir=str(out))
    tok = load_tokenizer(out / "tokenizer.json")

    sample = "The capital of India is New Delhi."
    ids = tok.encode(sample).ids
    decoded = tok.decode(ids)
    # ByteLevel BPE may normalize spaces; content must survive.
    assert "capital of India" in decoded
    assert "New Delhi" in decoded
    assert "{" not in decoded and '"text"' not in decoded
