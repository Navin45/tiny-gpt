"""Decoder-only Transformer in PyTorch.

Clarity over peak kernel efficiency. Includes GQA, RoPE, RMSNorm, SwiGLU, optional
sparse MoE, and residual connections. Larger training runs can later swap in fused
or distributed kernels without changing the learning API.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .config import ModelConfig
from .data import document_causal_mask


class RMSNorm(nn.Module):
    """Root-mean-square normalization without a bias term."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms).to(x.dtype) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the final dimension by 90 degrees in pairs."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    """Pre-computed rotary position embeddings (RoPE)."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(positions, inv_freq)
        # Repeat each angular frequency for the two coordinates of a rotary pair.
        emb = torch.repeat_interleave(freqs, 2, dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-2)
        if seq_len > self.cos.size(0):
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.cos.size(0)}"
            )
        cos = self.cos[:seq_len].to(q.device, q.dtype)[None, None, :, :]
        sin = self.sin[:seq_len].to(q.device, q.dtype)[None, None, :, :]
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class GQAAttention(nn.Module):
    """Grouped-query self-attention.

    Query heads can outnumber key/value heads. K/V are repeated logically to match the
    query-head count. Using fewer KV heads reduces KV-cache memory at inference time.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.kv_repeat = cfg.n_heads // cfg.n_kv_heads
        inner = cfg.n_heads * cfg.head_dim
        kv_inner = cfg.n_kv_heads * cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, inner, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, kv_inner, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, kv_inner, bias=False)
        self.o_proj = nn.Linear(inner, cfg.d_model, bias=False)
        self.dropout = cfg.dropout
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        past_k: Optional[torch.Tensor] = None,
        past_v: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if past_k is not None or past_v is not None:
            # During incremental decoding the new positions must use absolute RoPE positions.
            past_len = 0 if past_k is None else past_k.size(-2)
            end = past_len + seq_len
            max_seq_len = self.rope.cos.size(0)
            if end > max_seq_len:
                raise ValueError(
                    f"KV cache context length {end} exceeds max_seq_len={max_seq_len}"
                )
            cos = self.rope.cos[past_len:end].to(q.device, q.dtype)
            sin = self.rope.sin[past_len:end].to(q.device, q.dtype)
            cos = cos[None, None]
            sin = sin[None, None]
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
        else:
            q, k = self.rope(q, k)

        if past_k is not None:
            k = torch.cat((past_k, k), dim=-2)
        if past_v is not None:
            v = torch.cat((past_v, v), dim=-2)

        k_full = k.repeat_interleave(self.kv_repeat, dim=1)
        v_full = v.repeat_interleave(self.kv_repeat, dim=1)

        is_incremental = past_k is not None
        if is_incremental:
            # Single-document generation path; custom packed-doc masks are training-only.
            sdpa_mask = None
            is_causal = False
        elif attn_mask is not None:
            sdpa_mask = attn_mask
            is_causal = False
        else:
            sdpa_mask = None
            is_causal = True

        out = F.scaled_dot_product_attention(
            q,
            k_full,
            v_full,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)
        return self.o_proj(out), k, v


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, d_model: int, multiplier: float) -> None:
        super().__init__()
        # Align to 256 when large enough; keep the exact size for tiny smoke configs
        # where rounding down to a multiple of 256 would collapse the hidden width to 0.
        hidden = int(d_model * multiplier)
        if hidden >= 256:
            hidden = (hidden // 256) * 256
        if hidden <= 0:
            raise ValueError(f"Invalid SwiGLU hidden size {hidden} for d_model={d_model}")
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class SparseMoE(nn.Module):
    """Top-k sparse mixture-of-experts layer.

    Loops over experts so routing is easy to follow. Faster stacks use token-dispatch kernels.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_experts = cfg.n_experts
        self.top_k = cfg.moe_top_k
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(cfg.d_model, cfg.ffn_multiplier) for _ in range(cfg.n_experts)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        router_logits = self.router(flat)
        router_probs = F.softmax(router_logits.float(), dim=-1)
        top_probs, top_idx = torch.topk(router_probs, self.top_k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(flat)
        assignment_fraction = torch.zeros(
            self.n_experts, device=flat.device, dtype=torch.float32
        )

        for expert_id, expert in enumerate(self.experts):
            for route_idx in range(self.top_k):
                mask = top_idx[:, route_idx] == expert_id
                if not mask.any():
                    continue
                selected = flat[mask]
                contribution = expert(selected) * top_probs[mask, route_idx].to(flat.dtype).unsqueeze(-1)
                out[mask] += contribution
                assignment_fraction[expert_id] += mask.float().mean()

        # Load-balancing auxiliary loss encourages probability mass and token assignment to
        # be spread across experts rather than collapsing onto a small subset.
        mean_router_prob = router_probs.mean(dim=0)
        assignment_fraction = assignment_fraction / self.top_k
        aux_loss = self.n_experts * torch.sum(mean_router_prob * assignment_fraction)
        return out.reshape(original_shape), aux_loss


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with optional sparse MoE feed-forward."""

    def __init__(self, cfg: ModelConfig, use_moe: bool) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = GQAAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SparseMoE(cfg) if use_moe else SwiGLU(cfg.d_model, cfg.ffn_multiplier)
        self.use_moe = use_moe

    def forward(
        self,
        x: torch.Tensor,
        past_k: Optional[torch.Tensor] = None,
        past_v: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_out, k, v = self.attn(self.norm1(x), past_k, past_v, attn_mask=attn_mask)
        x = x + attn_out

        ffn_out = self.ffn(self.norm2(x))
        if self.use_moe:
            ffn_out, aux_loss = ffn_out
        else:
            aux_loss = torch.zeros((), device=x.device, dtype=torch.float32)
        x = x + ffn_out
        return x, k, v, aux_loss


class TinyGPT(nn.Module):
    """Decoder-only language model.

    During training the model returns next-token logits and an MoE auxiliary loss. During
    generation the same block stack can carry a list of KV caches for incremental decoding.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    cfg, use_moe=(cfg.moe_layers_every > 0 and (i + 1) % cfg.moe_layers_every == 0)
                )
                for i in range(cfg.n_layers)
            ]
        )
        self.n_moe_layers = sum(1 for block in self.blocks if block.use_moe)
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

        # Residual projection scaling stabilizes very deep networks during initialization.
        scale = 1.0 / math.sqrt(2 * cfg.n_layers)
        for block in self.blocks:
            block.attn.o_proj.weight.data.mul_(scale)
            if isinstance(block.ffn, SparseMoE):
                for expert in block.ffn.experts:
                    expert.w2.weight.data.mul_(scale)
            else:
                block.ffn.w2.weight.data.mul_(scale)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        x = self.tok_emb(input_ids)
        new_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        total_aux_loss = x.new_zeros((), dtype=torch.float32)

        attn_mask = None
        if (
            self.cfg.mask_document_boundaries
            and self.cfg.eos_token_id is not None
            and past_key_values is None
        ):
            attn_mask = document_causal_mask(input_ids, self.cfg.eos_token_id)

        for idx, block in enumerate(self.blocks):
            past_k, past_v = (None, None) if past_key_values is None else past_key_values[idx]

            if self.training and self.cfg.use_checkpointing and past_key_values is None:
                # Checkpoint only the training path; cached decoding needs actual K/V tensors.
                def block_fn(hidden: torch.Tensor, mask: Optional[torch.Tensor] = attn_mask):
                    y, _, _, aux = block(hidden, None, None, attn_mask=mask)
                    return y, aux

                x, aux = activation_checkpoint(block_fn, x, use_reentrant=False)
                total_aux_loss = total_aux_loss + aux
            else:
                x, k, v, aux = block(x, past_k, past_v, attn_mask=attn_mask)
                new_cache.append((k.detach(), v.detach()))
                total_aux_loss = total_aux_loss + aux

        x = self.final_norm(x)
        logits = self.lm_head(x)

        # Mean over MoE layers so moe_aux_loss_coef stays stable across architecture sweeps.
        if self.n_moe_layers > 0:
            total_aux_loss = total_aux_loss / self.n_moe_layers

        if targets is None:
            return logits, total_aux_loss, new_cache

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        )
        return loss, total_aux_loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate tokens using KV caching and nucleus sampling."""
        was_training = self.training
        self.eval()
        try:
            ids = input_ids
            past = None
            for _ in range(max_new_tokens):
                current = ids[:, -1:] if past is not None else ids
                logits, _, past = self(current, past_key_values=past)
                next_logits = logits[:, -1, :]
                if temperature <= 0:
                    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                else:
                    next_logits = next_logits / temperature
                    probs = F.softmax(next_logits, dim=-1)
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cumulative = torch.cumsum(sorted_probs, dim=-1)
                    remove = cumulative > top_p
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = False
                    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    sampled = torch.multinomial(sorted_probs, num_samples=1)
                    next_token = sorted_idx.gather(-1, sampled)

                ids = torch.cat((ids, next_token), dim=1)
                if eos_token_id is not None and torch.all(next_token == eos_token_id):
                    break
            return ids
        finally:
            if was_training:
                self.train()
