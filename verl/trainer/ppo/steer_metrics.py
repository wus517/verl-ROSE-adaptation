# Copyright 2026 The verl team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Quadrant statistics for STEER and high-entropy weight adjustments."""

from functools import cache

import torch
import torch.distributed as dist


@cache
def _clamp_endpoint_gradients(device, dtype):
    """Return this PyTorch build's clamp gradients at the lower and upper endpoints."""
    with torch.enable_grad():
        endpoints = torch.tensor([-1.0, 1.0], device=device, dtype=dtype, requires_grad=True)
        gradient = torch.autograd.grad(endpoints.clamp(-1.0, 1.0).sum(), endpoints)[0]
    return gradient.detach()


def _clamp_gradient(values, lower, upper):
    endpoint_gradients = _clamp_endpoint_gradients(values.device, values.dtype)
    return torch.where(
        values == lower,
        endpoint_gradients[0],
        torch.where(
            values == upper,
            endpoint_gradients[1],
            ((values > lower) & (values < upper)).to(values.dtype),
        ),
    )


@torch.no_grad()
def compute_steer_v4_statistics(
    *,
    advantages,
    ratio,
    log_prob_difference,
    losses1,
    losses2,
    clipped,
    response_mask,
    weights,
    quadrants,
    other_mask,
    clip_low,
    clip_high,
    clip_ratio_c,
    q2_release: tuple[torch.Tensor, torch.Tensor] | None = None,
    recovery: tuple[torch.Tensor, torch.Tensor] | None = None,
    low_entropy_mask: torch.Tensor | None = None,
):
    """Compute mask and weight sufficient statistics for one policy-loss call."""
    difference_grad = _clamp_gradient(log_prob_difference, -20.0, 20.0)
    ratio_clamp_grad = _clamp_gradient(ratio, 1.0 - clip_low, 1.0 + clip_high)
    ratio_grad = ratio * difference_grad
    losses1_grad = -advantages * ratio_grad
    losses2_grad = -advantages * ratio_clamp_grad * ratio_grad

    # torch.maximum splits gradients equally when its inputs are equal.
    clipped_grad = torch.where(
        losses1 > losses2,
        losses1_grad,
        torch.where(losses1 < losses2, losses2_grad, (losses1_grad + losses2_grad) * 0.5),
    )
    dual_clip_value = -advantages * clip_ratio_c
    dual_grad = torch.where(
        clipped < dual_clip_value,
        clipped_grad,
        torch.where(clipped > dual_clip_value, 0.0, clipped_grad * 0.5),
    )
    loss_grad = torch.where(advantages < 0, dual_grad, clipped_grad)
    active = response_mask.bool() & torch.isfinite(loss_grad) & (loss_grad != 0)
    valid = response_mask.bool()
    other_active = other_mask & active

    # [valid, active, q counts, q active counts, q weight sums, other count,
    #  other active count, total weight sum]. Keep the window on the model device.
    values = [
        valid.sum(),
        active.sum(),
        *(quadrant.sum() for quadrant in quadrants),
        *((quadrant & active).sum() for quadrant in quadrants),
        *(weights[quadrant].sum(dtype=torch.float32) for quadrant in quadrants),
        other_mask.sum(),
        other_active.sum(),
        weights[valid].sum(dtype=torch.float32),
    ]
    if recovery is not None:
        recovery_mask, base_weights = recovery
        # Keep the legacy Q2 release fields at the same offsets.
        q2_release = (recovery_mask & quadrants[1], base_weights)
    if q2_release is not None:
        release_mask, q2_base_weights = q2_release
        active_release = release_mask & active
        values.extend(
            [
                release_mask.sum(),
                active_release.sum(),
                q2_base_weights[active_release].sum(dtype=torch.float32),
                (weights[active_release] - 1.0).sum(dtype=torch.float32),
            ]
        )
    if recovery is not None:
        weight_delta = weights - base_weights
        for quadrant in quadrants:
            recovered = recovery_mask & quadrant
            active_recovered = recovered & active
            values.extend(
                [recovered.sum(), active_recovered.sum(), weight_delta[active_recovered].sum(dtype=torch.float32)]
            )
        if low_entropy_mask is not None:
            for quadrant in quadrants:
                low = low_entropy_mask & quadrant
                high = recovery_mask & quadrant
                values.extend([low.sum(), weights[low].sum(dtype=torch.float32)])
                values.extend([high.sum(), weights[high].sum(dtype=torch.float32)])
    return torch.stack([value.to(dtype=torch.float32) for value in values])


class SteerV4Metrics:
    """Accumulate microbatch counts and weight sums, then reduce a logging step."""

    def __init__(self):
        self.totals = None

    def add(self, statistics):
        if self.totals is None:
            self.totals = statistics.detach().clone()
        else:
            self.totals.add_(statistics.detach())

    def flush(self, dp_group=None):
        if self.totals is None:
            return {}

        totals = self.totals.clone()
        if dist.is_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM, group=dp_group)

        values = totals.cpu().tolist()
        valid_count, active_count = values[:2]
        quadrant_counts = values[2:6]
        quadrant_active_counts = values[6:10]
        quadrant_weight_sums = values[10:14]
        other_count, other_active_count, weight_sum = values[14:17]

        metrics = {
            "steer/valid_count": valid_count,
            "steer/active_count": active_count,
        }
        if valid_count > 0:
            metrics["steer/weight_mean"] = weight_sum / valid_count
            for index, (count, active, weight_sum) in enumerate(
                zip(quadrant_counts, quadrant_active_counts, quadrant_weight_sums, strict=True), start=1
            ):
                metrics[f"steer/q{index}_frac"] = count / valid_count
                if count > 0:
                    metrics[f"steer/q{index}_weight_mean"] = weight_sum / count
                    metrics[f"steer/q{index}_inactive_frac"] = (count - active) / count
            metrics["steer/other_frac"] = other_count / valid_count
        if active_count > 0:
            for index, count in enumerate(quadrant_active_counts, start=1):
                metrics[f"steer/q{index}_active_frac"] = count / active_count
            metrics["steer/other_active_frac"] = other_active_count / active_count

        if len(values) > 17:
            release_count, active_release_count, active_base_weight_sum, active_boost_sum = values[17:21]
            q2_count = quadrant_counts[1]
            q2_active_count = quadrant_active_counts[1]
            if q2_count > 0:
                metrics["steer/q2_release_frac"] = release_count / q2_count
            if q2_active_count > 0:
                metrics["steer/q2_release_active_frac"] = active_release_count / q2_active_count
            if active_release_count > 0:
                metrics["steer/q2_release_base_weight_mean"] = active_base_weight_sum / active_release_count
                metrics["steer/q2_release_boost_mean"] = active_boost_sum / active_release_count

        if len(values) > 21:
            for index, (count, active) in enumerate(zip(quadrant_counts, quadrant_active_counts, strict=True)):
                offset = 21 + 3 * index
                recovered, active_recovered, delta_sum = values[offset : offset + 3]
                prefix = f"steer/q{index + 1}_recovery"
                if count > 0:
                    metrics[f"{prefix}_frac"] = recovered / count
                if active > 0:
                    metrics[f"{prefix}_active_frac"] = active_recovered / active
                if active_recovered > 0:
                    metrics[f"{prefix}_delta_mean"] = delta_sum / active_recovered

        if len(values) > 33:
            for index in range(4):
                offset = 33 + 4 * index
                low_count, low_weight_sum, high_count, high_weight_sum = values[offset : offset + 4]
                if low_count > 0:
                    metrics[f"steer/q{index + 1}_low_entropy_weight_mean"] = low_weight_sum / low_count
                if high_count > 0:
                    metrics[f"steer/q{index + 1}_high_entropy_weight_mean"] = high_weight_sum / high_count

        self.totals.zero_()
        return metrics
