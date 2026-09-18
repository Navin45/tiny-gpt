# Tiny-GPT documentation

Learning-oriented docs for a from-scratch PyTorch decoder-only training stack.

## Contents

| Page | Description |
|------|-------------|
| [Architecture](architecture.md) | System and model diagrams (Mermaid) |
| [Artifacts](artifacts.md) | What lives under `artifacts/` |
| [Data](data.md) | You provide JSONL; nothing scrapes a corpus for you |
| [Pipeline](pipeline.md) | Tokenizer → encode → pretrain → post-train → chat |
| [Scaling notes](architecture-proposal.md) | Optional ideas if you grow past smoke/small |

## Install & verify

```bash
uv sync --extra quality
uv run pytest -q
```

## License

MIT — see [LICENSE](../LICENSE) in the repo root.
