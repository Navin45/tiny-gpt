# Tiny-GPT

From-scratch PyTorch decoder-only LLM for learning: tokenizer → pretrain → SFT → DPO / RLVR → chat.

## Docs

| Doc | Contents |
|-----|----------|
| [docs/README.md](docs/README.md) | Documentation index |
| [docs/architecture.md](docs/architecture.md) | System + model diagrams (Mermaid) |
| [docs/artifacts.md](docs/artifacts.md) | Artifact layout |
| [docs/data.md](docs/data.md) | Where JSONL corpora come from |
| [docs/pipeline.md](docs/pipeline.md) | End-to-end commands |
| [docs/architecture-proposal.md](docs/architecture-proposal.md) | Optional notes if you scale later |

## Quick start

```bash
uv sync
uv run pytest -q
```

Optional tiny corpus for a plumbing run:

```bash
uv run python scripts/get_smoke_data.py
```

Train tokenizer (`{"text": "..."}` JSONL):

```bash
uv run python scripts/train_tokenizer.py \
  --input artifacts/data/train.jsonl \
  --output-dir artifacts/tokenizer \
  --vocab-size 32000
```

Encode corpus:

```bash
uv run python scripts/encode_corpus.py \
  --input artifacts/data/train.jsonl \
  --tokenizer artifacts/tokenizer/tokenizer.json \
  --output artifacts/data/train.bin
```

Smoke pretrain:

```bash
uv run python scripts/pretrain.py --config configs/smoke.yaml
```

Single-GPU run / resume:

```bash
uv run python scripts/pretrain.py --config configs/small.yaml
uv run python scripts/pretrain.py --config configs/small.yaml \
  --resume artifacts/checkpoints/base/step_500.pt
```

If you set `mask_document_boundaries: true` in the config, also pass
`--tokenizer artifacts/tokenizer/tokenizer.json`.

Chat (after you have a checkpoint):

```bash
uv run python scripts/chat.py \
  --config configs/small.yaml \
  --tokenizer artifacts/tokenizer/tokenizer.json \
  --checkpoint artifacts/checkpoints/base/step_500.pt
```

## Layout

```text
tiny-gpt/
├── LICENSE
├── src/tiny_gpt/
├── scripts/
├── configs/
├── artifacts/    # you add data; scripts write tokenizer / bins / checkpoints
├── docs/
└── tests/
```

## Model pieces

RMSNorm · RoPE · GQA · SwiGLU · sparse MoE · KV-cache decode · DDP · SFT / DPO / RLVR

## License

[MIT](LICENSE) — free to use, copy, modify, and distribute. See `LICENSE` for the full text.
