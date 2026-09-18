# Where training data comes from

**Nowhere in this repo.** Scripts only consume JSONL you place under `artifacts/data/`. Empty `.gitkeep` dirs are intentional.

| Stage | File | Schema | Typical source (you fetch) |
|-------|------|--------|----------------------------|
| Pretrain | `train.jsonl` / `val.jsonl` | `{"text":"..."}` | Tiny Shakespeare smoke script, TinyStories, FineWeb-Edu, WikiText |
| SFT | `sft.jsonl` | `{"messages":[...]}` | e.g. UltraChat |
| DPO | `dpo.jsonl` | `{"prompt","chosen","rejected"}` | e.g. UltraFeedback |
| RLVR | `rlvr.jsonl` | `{"prompt","answer"}` | e.g. GSM8K |

## Fastest plumbing check

```bash
uv run python scripts/get_smoke_data.py
```

Writes a small public-domain Tiny Shakespeare split into `artifacts/data/train.jsonl` and `val.jsonl` so you can run tokenizer → encode → `configs/smoke.yaml` before touching real corpora.

## Larger corpora

```bash
uv sync --extra data
# then use Hugging Face `datasets` yourself, or extend get_smoke_data.py
```

Do not confuse corpus prep with model training: training never downloads text.
