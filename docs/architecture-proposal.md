# Optional scaling notes

Tiny-GPT is a **learning stack**: small configs, readable code, full train → chat path.

When experiments outgrow `configs/smoke.yaml` / `configs/small.yaml`, scale gradually and keep evals fixed so bigger runs are comparable.

## Suggested size steps


| Step | Goal                                                 | Rough size          |
| ---- | ---------------------------------------------------- | ------------------- |
| 1    | Prove tokenizer → encode → train → checkpoint → chat | smoke (~tiny)       |
| 2    | Longer training on a real small corpus               | dense, ~100M–1B     |
| 3    | Try MoE routing stability                            | small MoE           |
| 4    | Only then increase depth/width/context               | larger dense or MoE |


Accept a larger model only if it beats the previous step on the **same** evals at similar data/compute.

## Model ingredients (already in this repo)

- decoder-only Transformer, pre-norm, RMSNorm
- RoPE, GQA, SwiGLU
- optional sparse MoE + load-balance aux loss
- KV-cache generation, DDP pretrain, SFT / DPO / simple RLVR

## Data (you supply it)

Quality of `artifacts/data/*.jsonl` matters more than diagram tweaks. See [data.md](data.md).

`scripts/prepare_corpus.py` does exact dedup, heuristic quality filters, and n-gram contamination checks against an eval JSONL. Still later: MinHash near-dedup, a learned quality classifier, versioned mixtures.

## Systems (later)

This repo uses simple DDP. Larger runs may need FSDP, better checkpointing, fused kernels, and longer-context techniques (curriculum / interpolation).

## Evaluation

Track loss/PPL first. Add a small fixed task set before you trust that a bigger run is “better.”