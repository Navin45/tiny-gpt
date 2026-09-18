"""Post-training utilities: chat rendering, SFT, DPO, and a compact RLVR/GRPO-style loop."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


ROLE_TOKENS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}


def render_messages(tokenizer, messages: list[dict[str, str]], max_len: int) -> tuple[list[int], list[int]]:
    """Render a conversation and mark which token targets belong to the assistant.

    The mask is the key SFT idea: the model sees the whole conversation, but loss is normally
    computed only on assistant tokens. This prevents wasting optimization pressure on reproducing
    the user's prompt.
    """
    input_ids: list[int] = []
    assistant_mask: list[int] = []
    eos_id = tokenizer.token_to_id("<|eot|>")
    if eos_id is None:
        raise ValueError("Tokenizer is missing <|eot|>")

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        role_token = ROLE_TOKENS[role]
        role_ids = tokenizer.encode(role_token).ids
        content_ids = tokenizer.encode(content).ids
        input_ids.extend(role_ids)
        assistant_mask.extend([0] * len(role_ids))
        input_ids.extend(content_ids)
        assistant_mask.extend([1 if role == "assistant" else 0] * len(content_ids))
        input_ids.append(eos_id)
        # The end-of-turn token is part of the assistant target for assistant turns.
        assistant_mask.append(1 if role == "assistant" else 0)

        if len(input_ids) >= max_len:
            break

    return input_ids[:max_len], assistant_mask[:max_len]


def next_token_labels(token_ids: list[int], assistant_mask: list[int], max_len: int):
    """Convert token-level assistant masks into next-token labels with ignore_index."""
    labels = [-100] * max_len
    for i in range(min(len(token_ids) - 1, max_len)):
        next_pos = i + 1
        if next_pos < len(assistant_mask) and assistant_mask[next_pos]:
            labels[i] = token_ids[next_pos]
    return labels


@dataclass
class PreferenceExample:
    """One DPO preference comparison."""

    prompt: str
    chosen: str
    rejected: str


def token_logprob(model, input_ids: torch.Tensor, continuation_start: int) -> torch.Tensor:
    """Return summed log-probability of the continuation.

    The continuation starts at `continuation_start` in the complete sequence. We calculate
    logits for all positions and sum only the log-probabilities corresponding to continuation
    tokens.
    """
    logits, _, _ = model(input_ids)
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    target = input_ids[:, 1:]
    token_lp = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    mask = torch.arange(token_lp.size(1), device=token_lp.device).unsqueeze(0) >= continuation_start - 1
    return (token_lp * mask).sum(dim=-1)


def dpo_loss(
    policy,
    reference,
    prompt_ids: torch.Tensor,
    chosen_ids: torch.Tensor,
    rejected_ids: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Compute the Direct Preference Optimization objective for one batch."""
    chosen = torch.cat((prompt_ids, chosen_ids), dim=1)
    rejected = torch.cat((prompt_ids, rejected_ids), dim=1)
    prompt_len = prompt_ids.size(1)

    pi_chosen = token_logprob(policy, chosen, prompt_len)
    pi_rejected = token_logprob(policy, rejected, prompt_len)
    with torch.no_grad():
        ref_chosen = token_logprob(reference, chosen, prompt_len)
        ref_rejected = token_logprob(reference, rejected, prompt_len)

    advantage = (pi_chosen - pi_rejected) - (ref_chosen - ref_rejected)
    return -F.logsigmoid(beta * advantage).mean()


class RLVRTrainer:
    """Small RLVR/GRPO-style trainer around a task-specific verifier.

    For each prompt, sample a group of answers, score them with a verifier, normalize rewards
    within the group, then apply a policy-gradient update. This is a teaching loop; fuller GRPO
    would add old-policy handling, clipping, KL control, and faster generation.
    """

    def __init__(
        self,
        policy,
        tokenizer,
        optimizer,
        verifier: Callable[[str, str], float],
        device: torch.device,
        group_size: int = 4,
    ) -> None:
        self.policy = policy
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.verifier = verifier
        self.device = device
        self.group_size = group_size

    def train_prompt(self, prompt: str, max_new_tokens: int = 128) -> float:
        """Sample a reward group and perform one policy-gradient update."""
        prompt_ids = torch.tensor([self.tokenizer.encode(prompt).ids], device=self.device)
        completions: list[str] = []
        logps: list[torch.Tensor] = []
        rewards: list[float] = []

        for _ in range(self.group_size):
            generated = self.policy.generate(prompt_ids, max_new_tokens=max_new_tokens)
            completion_ids = generated[:, prompt_ids.size(1) :]
            completion = self.tokenizer.decode(completion_ids[0].tolist())
            completions.append(completion)
            rewards.append(float(self.verifier(prompt, completion)))

            # Re-score under the current policy for a differentiable log-prob.
            # Simpler (and slower) than returning log-probs from generate().
            logits, _, _ = self.policy(generated)
            lp = F.log_softmax(logits[:, :-1], dim=-1)
            target = generated[:, 1:]
            token_lp = lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            logps.append(token_lp[:, prompt_ids.size(1) - 1 :].sum())

        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        advantages = (reward_tensor - reward_tensor.mean()) / (reward_tensor.std() + 1e-5)
        loss = torch.stack([-adv * lp for adv, lp in zip(advantages, logps)]).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(reward_tensor.mean().item())
