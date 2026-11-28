# Copyright 2026 The verl team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""V8 entropy groups, policy loss, and logging."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import (
    apply_steer_recovery,
    compute_policy_loss_with_entropy,
    compute_steer_v4_weights,
    compute_steer_v8_weights,
)
from verl.trainer.ppo.steer_metrics import SteerV4Metrics
from verl.utils import tensordict_utils as tu
from verl.utils.metric import reduce_metrics
from verl.workers.config import ActorConfig, PolicyLossConfig
from verl.workers.utils.losses import ppo_loss


def actor_config(**overrides):
    defaults = {
        "loss_mode": "entropy_control",
        "linear": False,
        "token_weight_min": 0.7,
        "token_weight_max": 1.0,
        "steer_quadrant_logging": True,
        "steer_recovery_quadrants": [1, 2, 3, 4],
        "steer_entropy_weighting": True,
    }
    return ActorConfig(
        strategy="fsdp2",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        policy_loss=PolicyLossConfig(**(defaults | overrides)),
    )


def batch():
    log_prob = torch.tensor([[-1, -2, -2, -1, -2, -2, -2, -2, -2, -1, -1]], dtype=torch.float32)
    old = log_prob.clone()
    old[0, 4] -= 0.5  # High entropy Q2 token whose PPO gradient is clipped to zero.
    advantages = torch.tensor([[1, 20, -1, -1, 1, 1, 0, 1, -1, -1, 1]], dtype=torch.float32)
    entropy = torch.tensor([[1.5, 1.5, 1.5, 1.5, 1.5, 0.65, 2.0, 0.5, 0.1, 1.2, 99.0]])
    mask = torch.tensor([[True] * 10 + [False]])
    return advantages, entropy, old, log_prob, mask


def loss_and_gradient(data, config, collector=None):
    advantages, entropy, old, log_prob, mask = data
    leaf = log_prob.detach().clone().requires_grad_()
    loss, _ = compute_policy_loss_with_entropy(
        old, leaf, advantages, mask, config=config, entropys=entropy, steer_metrics=collector
    )
    return loss, torch.autograd.grad(loss, leaf)[0]


def baseline_weights(data, config):
    policy = config.policy_loss
    return compute_steer_v4_weights(
        *data, token_weight_min=policy.token_weight_min, token_weight_max=policy.token_weight_max
    )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("steer_entropy_weighting", 1),
        ("steer_low_entropy_weight_min", 0.0),
        ("steer_low_entropy_weight_min", float("inf")),
        ("steer_high_entropy_weight_min_by_quadrant", [0.7, 0.8]),
        ("steer_high_entropy_weight_min_by_quadrant", [0.7, 0.7, float("nan"), 0.8]),
        ("steer_recovery_rate_by_quadrant", [0.35] * 5),
        ("steer_recovery_rate_by_quadrant", [0.0, 0.25, 1.1, 1.0]),
        ("steer_recovery_rate_by_quadrant", [0.0, 0.25, False, 1.0]),
    ],
)
def test_reject_invalid_v8_configuration(field, invalid):
    with pytest.raises(ValueError, match=field):
        actor_config(**{field: invalid})


def test_v8_conflicts_with_legacy_q2_release():
    with pytest.raises(ValueError, match="cannot be enabled together"):
        actor_config(q2_high_entropy_ratio=0.2)


@pytest.mark.parametrize(
    ("entropy", "valid", "high", "low"),
    [
        ([0.50, 0.49, 0.1], [True] * 3, [], [0, 1, 2]),
        ([0.49, 0.30, 0.1], [True] * 3, [], [0, 1, 2]),
        ([0.9, 0.8, 0.8, 0.7, float("nan"), float("inf"), 50.0], [True] * 6 + [False], [0, 1, 2], [3]),
        ([float("nan"), float("inf")], [True, True], [], []),
        ([0.9, 0.8], [False, False], [], []),
    ],
)
def test_entropy_groups_at_rank_and_absolute_boundaries(entropy, valid, high, low):
    entropy = torch.tensor([entropy])
    advantages = torch.ones_like(entropy)
    log_prob = torch.full_like(entropy, -2.0, requires_grad=True)
    mask = torch.tensor([valid])
    data = (advantages, entropy, log_prob.detach(), log_prob, mask)
    config = actor_config(steer_recovery_quadrants=[2], steer_recovery_top_ratio=0.5)
    weights, _, _, (high_mask, _), low_mask = compute_steer_v8_weights(*data, config.policy_loss)
    assert high_mask[0].nonzero(as_tuple=True)[0].tolist() == high
    assert low_mask[0].nonzero(as_tuple=True)[0].tolist() == low
    base, _, _ = baseline_weights(data, config)
    rest = ~(high_mask | low_mask)
    torch.testing.assert_close(weights[rest], base[rest], rtol=0, atol=0)
    assert not weights.requires_grad


def test_default_v8_weights_match_the_specification_examples():
    entropy = torch.tensor([[0.4, 0.4, 0.8, 0.8]])
    log_prob = torch.full_like(entropy, -2.0)
    magnitude = torch.tensor([[0.04, 0.08, 0.04, 0.08]])
    advantages = magnitude / ((1 - log_prob.exp()) * (log_prob + entropy).abs())
    mask = torch.ones_like(entropy, dtype=torch.bool)
    config = actor_config(steer_recovery_quadrants=[2])
    weights, _, _, (_, pre), _ = compute_steer_v8_weights(
        advantages, entropy, log_prob, log_prob, mask, config.policy_loss
    )
    torch.testing.assert_close(pre, torch.tensor([[0.5**0.5, 0.5, 0.7**0.5, 0.7]]))
    torch.testing.assert_close(weights, torch.tensor([[0.5**0.5, 0.5, 0.35 + 0.65 * 0.7**0.5, 0.805]]))


@pytest.mark.parametrize("selection", [[1, 2], [2, 4], [4], [1, 2, 3, 4]])
def test_per_quadrant_floors_share_full_steer_denominator(selection):
    data = batch()
    advantages, entropy, old, log_prob, mask = data
    floors = [0.7, 0.6, 0.8, 0.9]
    rates = [0.35, 0.0, 0.35, 1.0]
    config = actor_config(
        steer_recovery_quadrants=selection,
        steer_high_entropy_weight_min_by_quadrant=floors,
        steer_recovery_rate_by_quadrant=rates,
    )
    final, quadrants, _, (high, pre), low = compute_steer_v8_weights(*data, config.policy_loss)
    selected = torch.stack([quadrants[q - 1] for q in selection]).any(dim=0)
    # A zero-advantage token takes the top rank, leaving the tied entropy of 1.5 as the cutoff.
    assert torch.equal(high, selected & (entropy == 1.5))
    assert torch.equal(low, selected & (entropy < 1.5))
    probability = log_prob.exp()
    magnitude = ((advantages / old.exp()) * (probability * (1 - probability) * (probability.log() + entropy))).abs()
    denominator = max(0.02, magnitude[mask].max().item())
    assert magnitude[0, 1].item() == denominator
    expected_pre, _, _ = baseline_weights(data, config)
    expected_pre[low] = 0.5 ** (magnitude[low] / denominator)
    expected_final = expected_pre.clone()
    for q in selection:
        group = high & quadrants[q - 1]
        mapped = floors[q - 1] ** (magnitude[group] / denominator)
        expected_pre[group] = mapped
        expected_final[group] = mapped + rates[q - 1] * (1 - mapped)
    torch.testing.assert_close(pre, expected_pre)
    torch.testing.assert_close(final, expected_final)
    base, _, _ = baseline_weights(data, config)
    torch.testing.assert_close(final[~selected], base[~selected], rtol=0, atol=0)

    loss, gradient = loss_and_gradient(data, config)
    ratio = (log_prob - old).exp()
    clipped = torch.maximum(-advantages * ratio, -advantages * ratio.clamp(0.8, 1.2))
    ppo = torch.where(advantages < 0, torch.minimum(clipped, -advantages * 3.0), clipped)
    _, ppo_gradient = loss_and_gradient(data, actor_config(steer_recovery_quadrants=[], token_weight_min=1.0))
    torch.testing.assert_close(loss, (ppo * expected_final * mask).sum() / mask.sum())
    torch.testing.assert_close(gradient, ppo_gradient * expected_final)
    assert gradient[0, 4] == 0


def test_disabled_v8_and_matching_maps_agree_with_legacy_weights_loss_and_gradient():
    data = batch()
    v7 = actor_config(steer_entropy_weighting=False, steer_recovery_quadrants=[1, 2])
    equivalent_v8 = actor_config(
        steer_recovery_quadrants=[1, 2],
        steer_low_entropy_weight_min=0.7,
        steer_high_entropy_weight_min_by_quadrant=[0.7] * 4,
        steer_recovery_rate_by_quadrant=[0.25] * 4,
    )
    base, quadrants, _ = baseline_weights(data, v7)
    expected, _ = apply_steer_recovery(base, data[1], data[-1], quadrants, v7.policy_loss)
    actual, *_ = compute_steer_v8_weights(*data, equivalent_v8.policy_loss)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for old_config, v8_config in [
        (v7, equivalent_v8),
        (
            actor_config(steer_entropy_weighting=False, steer_recovery_quadrants=[]),
            actor_config(steer_recovery_quadrants=[]),
        ),
    ]:
        for actual_value, expected_value in zip(
            loss_and_gradient(data, v8_config), loss_and_gradient(data, old_config), strict=True
        ):
            torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)


def test_tensorboard_retains_legacy_metrics_and_logs_final_group_weights(tmp_path, monkeypatch):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from verl.utils.tracking import Tracking

    config = actor_config(
        steer_recovery_quadrants=[2],
        steer_high_entropy_weight_min_by_quadrant=[0.7, 0.6, 0.7, 0.7],
    )
    advantages, entropy, old, log_prob, mask = data = batch()
    length = int(mask.sum())
    rollout = TensorDict(
        {
            "prompts": torch.ones(1, 1, dtype=torch.long),
            "responses": torch.ones_like(mask, dtype=torch.long),
            "attention_mask": torch.cat((torch.ones(1, 1, dtype=torch.long), mask.long()), dim=-1),
            "response_mask": mask,
            "old_log_probs": old,
            "advantages": advantages,
            "entropy": torch.nested.as_nested_tensor([entropy[0, :length]], layout=torch.jagged),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor(
        rollout, dp_size=1, batch_num_tokens=length, global_batch_size=1, steer_local_mini_batch_size=1
    )
    model_log_prob = torch.cat((log_prob[0, :length], torch.zeros(1))).requires_grad_()
    collector = SteerV4Metrics()
    loss, actor_metrics = ppo_loss(config, {"log_probs": model_log_prob}, rollout, steer_metrics=collector)
    loss.backward()
    assert collector.totals.shape == (49,)
    assert collector.totals.dtype == torch.float32
    metrics = collector.flush()
    final, _, _, (high, _), low = compute_steer_v8_weights(*data, config.policy_loss)
    assert set(tag for tag in metrics if "_entropy_weight_mean" in tag) == {
        "steer/q2_low_entropy_weight_mean",
        "steer/q2_high_entropy_weight_mean",
    }
    assert metrics["steer/q2_high_entropy_weight_mean"] == pytest.approx(final[high].mean().item())
    assert metrics["steer/q2_low_entropy_weight_mean"] == pytest.approx(final[low].mean().item())
    assert metrics["steer/q2_release_base_weight_mean"] == pytest.approx(0.6)
    assert metrics["steer/q2_recovery_delta_mean"] == pytest.approx(0.35 * (1 - 0.6))

    monkeypatch.setenv("TENSORBOARD_DIR", str(tmp_path))
    tracker = Tracking("steer", "v8", default_backend=["tensorboard"])
    logged = {**reduce_metrics(actor_metrics), **metrics, "val/accuracy": 0.5}
    tracker.log(logged, step=1)
    tracker.finish()
    event = EventAccumulator(str(tmp_path)).Reload()
    assert {"actor/pg_loss", "val/accuracy", "steer/q2_weight_mean", "steer/q2_release_frac"} <= logged.keys()
    assert set(event.Tags()["scalars"]) == logged.keys()
    for tag, value in logged.items():
        assert event.Scalars(tag)[0].value == pytest.approx(value)


def _distributed_entropy_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        config = actor_config(steer_recovery_quadrants=[2])
        data = batch()
        collector = SteerV4Metrics()
        if rank == 0:
            short = tuple(tensor[:, 7:8].repeat(1, 3) for tensor in data)
            short[0].fill_(0.01)  # Below the denominator floor, so the low weights differ from rank 1.
            loss_and_gradient(short, config, collector)
        else:
            loss_and_gradient(data, config, collector)
        assert collector.totals.shape == (49,)
        assert collector.totals.dtype == torch.float32
        metrics = collector.flush(dp_group=dist.group.WORLD)

        final, _, _, (high, _), low = compute_steer_v8_weights(*data, config.policy_loss)
        short_magnitude = 0.01 * (1 - torch.exp(torch.tensor(-2.0))) * 1.5
        short_weight = (0.5 ** (short_magnitude / 0.02)).item()
        expected_low = (final[low].sum().item() + 3 * short_weight) / (int(low.sum()) + 3)
        assert metrics["steer/q2_low_entropy_weight_mean"] == pytest.approx(expected_low)
        assert metrics["steer/q2_high_entropy_weight_mean"] == pytest.approx(final[high].mean().item())
    finally:
        dist.destroy_process_group()


def test_two_process_v8_entropy_group_statistics(tmp_path):
    mp.spawn(_distributed_entropy_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
