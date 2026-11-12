# Copyright 2026 individual contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Pure tensor EA helpers. No rollout, Ray, or accelerator-specific dependencies."""

import math

import torch

from verl.utils.metric import Metric

EA_RAW_PREFIX = "actor/ea/raw/"


@torch.no_grad()
def entropy_advantage_from_logits(logits: torch.Tensor, *, chunk_size: int) -> torch.Tensor:
    """Full-vocabulary Shannon entropy in bounded FP32 token chunks, on the logits device.

    Slice before casting, including for non-contiguous dense logits. Never reshape/copy the
    entire [B,S,V] tensor or retain a differentiable entropy graph. Logits are read-only.
    """
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("entropy_advantage chunk_size must be a positive integer")
    if logits.ndim not in (2, 3) or logits.shape[-1] == 0 or not logits.is_floating_point():
        raise ValueError("EA logits must be floating point [N,V] or [B,S,V] with nonempty vocabulary")
    result = torch.empty(logits.shape[:-1], device=logits.device, dtype=torch.float32)
    rows = ((logits, result),) if logits.ndim == 2 else zip(logits.unbind(0), result.unbind(0), strict=True)
    for row, destination in rows:
        for start in range(0, row.shape[0], chunk_size):
            z = row[start : start + chunk_size].float()
            z = z - z.amax(dim=-1, keepdim=True)
            probabilities = torch.softmax(z, dim=-1)
            destination[start : start + chunk_size] = torch.logsumexp(z, dim=-1) - (probabilities * z).sum(-1)
    return result


@torch.no_grad()
def shape_entropy_advantage(advantages, entropy, response_mask, *, coef: float, kappa: float):
    """Return fresh shaped advantages, bonus and additive token statistics; never modify inputs."""
    if isinstance(coef, bool) or not math.isfinite(coef) or coef < 0:
        raise ValueError("EA coef must be finite and >= 0")
    if isinstance(kappa, bool) or not math.isfinite(kappa) or kappa <= 1:
        raise ValueError("EA kappa must be finite and > 1")
    if advantages.ndim != 2 or advantages.shape != entropy.shape or advantages.shape != response_mask.shape:
        raise ValueError("EA advantages, entropy and response_mask must have identical [B,R] shapes")
    if advantages.device != entropy.device or advantages.device != response_mask.device:
        raise ValueError("EA advantages, entropy and response_mask must be on the same device")
    mask = response_mask.bool()
    base = torch.where(mask, advantages.float(), 0.0)
    h = torch.where(mask, entropy.float(), 0.0)
    if not (torch.isfinite(base).all() & torch.isfinite(h).all() & (h >= -1e-5).all()).item():
        raise ValueError("EA requires finite advantages and finite nonnegative entropy on valid response tokens")
    h = h.clamp_min(0)
    cap = base.abs() / kappa
    scaled_entropy = coef * h
    bonus = torch.minimum(scaled_entropy, cap)
    shaped = base + bonus
    nonzero = mask & (base != 0)
    positive, negative = mask & (base > 0), mask & (base < 0)
    # Avoid division by zero even in the unselected branch of torch.where.
    ratios = bonus / torch.where(nonzero, base.abs(), 1.0)
    stats = {
        "valid_tokens_sum": mask.float().sum(),
        "nonzero_adv_tokens_sum": nonzero.float().sum(),
        "positive_tokens_sum": positive.float().sum(),
        "negative_tokens_sum": negative.float().sum(),
        "entropy_sum": h.sum(),
        "bonus_sum": bonus.sum(),
        "bonus_ratio_sum": ratios.sum(),
        "cap_hits_sum": (nonzero & (scaled_entropy >= cap)).float().sum(),
        "sign_flips_sum": (nonzero & (base.sign() != shaped.sign())).float().sum(),
        "base_positive_sum": torch.where(positive, base, 0.0).sum(),
        "shaped_positive_sum": torch.where(positive, shaped, 0.0).sum(),
        "base_negative_sum": torch.where(negative, base, 0.0).sum(),
        "shaped_negative_sum": torch.where(negative, shaped, 0.0).sum(),
    }
    return shaped, bonus, stats


def sum_entropy_advantage_metrics(metrics: dict) -> dict:
    """Pop additive EA stats before generic reduction, including the no-DP Metric case."""
    result = {}
    for key in list(metrics):
        if key.startswith(EA_RAW_PREFIX):
            value = metrics.pop(key)
            if isinstance(value, Metric):
                value = value.aggregate()
            elif isinstance(value, list | tuple):
                value = sum(value)
            result[key] = float(value)
    return result


def finalize_entropy_advantage_metrics(metrics: dict) -> None:
    """Derive token-weighted metrics only after summing DP/micro/mini/local updates."""
    if EA_RAW_PREFIX + "valid_tokens_sum" not in metrics:
        return

    def ratio(numerator, denominator):
        count = metrics[EA_RAW_PREFIX + denominator]
        return metrics[EA_RAW_PREFIX + numerator] / count if count else 0.0

    for name, numerator, denominator in (
        ("entropy_token_mean", "entropy_sum", "valid_tokens_sum"),
        ("bonus_mean", "bonus_sum", "valid_tokens_sum"),
        ("nonzero_adv_fraction", "nonzero_adv_tokens_sum", "valid_tokens_sum"),
        ("bonus_ratio_mean", "bonus_ratio_sum", "nonzero_adv_tokens_sum"),
        ("cap_fraction", "cap_hits_sum", "nonzero_adv_tokens_sum"),
        ("base_positive_mean", "base_positive_sum", "positive_tokens_sum"),
        ("shaped_positive_mean", "shaped_positive_sum", "positive_tokens_sum"),
        ("base_negative_mean", "base_negative_sum", "negative_tokens_sum"),
        ("shaped_negative_mean", "shaped_negative_sum", "negative_tokens_sum"),
    ):
        metrics["actor/ea/" + name] = ratio(numerator, denominator)
    metrics["actor/ea/sign_flips"] = metrics[EA_RAW_PREFIX + "sign_flips_sum"]
