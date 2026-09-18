# Architecture

Tiny-GPT is a decoder-only Transformer reference implementation with modern components (GQA, RoPE, SwiGLU, optional sparse MoE) and a full training loop path.

## System overview

```mermaid
flowchart TB
  subgraph data_plane["Data plane"]
    raw["Raw JSONL<br/>artifacts/data/*.jsonl"]
    tok["Byte-level BPE<br/>artifacts/tokenizer/"]
    bins["Packed token store<br/>artifacts/data/*.bin"]
  end

  subgraph train_plane["Training plane"]
    pre["Base pretrain<br/>scripts/pretrain.py"]
    ckpt_base["Checkpoints<br/>artifacts/checkpoints/base/"]
    sft["SFT<br/>scripts/sft.py"]
    dpo["DPO<br/>scripts/dpo.py"]
    rlvr["RLVR / GRPO<br/>scripts/grpo.py"]
  end

  subgraph infer_plane["Inference"]
    chat["Chat CLI<br/>scripts/chat.py"]
  end

  raw -->|train_tokenizer.py| tok
  raw -->|encode_corpus.py| bins
  tok --> bins
  bins --> pre
  pre --> ckpt_base
  ckpt_base --> sft
  sft --> dpo
  dpo --> rlvr
  ckpt_base --> chat
  sft --> chat
```

## Model stack (`TinyGPT`)

```mermaid
flowchart TB
  ids["input_ids"] --> emb["Token embedding"]
  emb --> blocks

  subgraph blocks["Transformer × N (pre-norm)"]
    direction TB
    n1["RMSNorm"] --> attn["GQA + RoPE<br/>optional KV cache"]
    attn --> res1["+ residual"]
    res1 --> n2["RMSNorm"]
    n2 --> ffn["SwiGLU  or  SparseMoE"]
    ffn --> res2["+ residual"]
  end

  blocks --> fn["Final RMSNorm"]
  fn --> head["lm_head → logits"]
  ffn -.->|aux load-balance loss| aux["moe_aux_loss<br/>(mean over MoE layers)"]
```

## Attention & KV cache

```mermaid
sequenceDiagram
  participant Gen as generate()
  participant Attn as GQAAttention
  participant Rope as RotaryEmbedding
  participant Cache as past_key_values

  Gen->>Attn: prefill full prompt
  Attn->>Rope: positions 0..T-1
  Attn->>Cache: store K,V (RoPE already applied)
  loop each new token
    Gen->>Attn: current token + past
    Attn->>Rope: absolute offset past_len
    Note over Attn: no causal mask needed<br/>for single new query
    Attn->>Cache: append K,V
    Attn-->>Gen: logits / next sample
  end
```

## MoE routing (reference)

```mermaid
flowchart LR
  x["hidden tokens"] --> router["Linear router"]
  router --> topk["top-k softmax"]
  topk --> e0["Expert 0 SwiGLU"]
  topk --> e1["Expert 1 SwiGLU"]
  topk --> ek["Expert …"]
  e0 --> mix["weighted sum"]
  e1 --> mix
  ek --> mix
  topk --> aux["load-balance aux loss"]
```

> MoE loops over experts for clarity. Faster stacks use token-dispatch kernels.

## Package modules

| Module | Role |
|--------|------|
| `tiny_gpt.model` | `TinyGPT`, attention, RoPE, MoE, generate |
| `tiny_gpt.config` | `ModelConfig` / `TrainConfig` / YAML load |
| `tiny_gpt.tokenizer` | BPE train / load |
| `tiny_gpt.data` | JSONL text extract, packed `TokenDataset` |
| `tiny_gpt.training` | DDP setup, LR schedule |
| `tiny_gpt.checkpoint` | save / load |
| `tiny_gpt.posttrain` | SFT masks, DPO, RLVR trainer |
| `tiny_gpt.evals` | loss / perplexity helpers |

## Configs

| File | Intent |
|------|--------|
| `configs/smoke.yaml` | Tiny model (d=64, 2 layers) — pipeline smoke test |
| `configs/small.yaml` | ~350–400M-param single-GPU reference |

See also [artifacts.md](artifacts.md) and [pipeline.md](pipeline.md).
