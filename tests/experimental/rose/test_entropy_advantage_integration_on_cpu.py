# Copyright 2026 individual contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.trainer.ppo.v1.utils import MetricsAggregator
from verl.utils import tensordict_utils as tu
from verl.utils import transferqueue_utils as tqu
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, EntropyAdvantageConfig, FSDPEngineConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.engine.utils import prepare_micro_batches
from verl.workers.utils.entropy_advantage import (
    EA_RAW_PREFIX,
    finalize_entropy_advantage_metrics,
    shape_entropy_advantage,
    sum_entropy_advantage_metrics,
)
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding


def nested(rows):
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def batch_data():
    prompts = [torch.tensor([1, 2]), torch.tensor([3, 4, 5])]
    responses = [torch.tensor([3, 4, 0]), torch.tensor([2, 0])]
    ids = nested([torch.cat(pair) for pair in zip(prompts, responses, strict=True)])
    data = TensorDict(
        {
            "input_ids": ids,
            "prompts": nested(prompts),
            "responses": nested(responses),
            "response_mask": nested([torch.ones(3), torch.ones(2)]),
            "advantages": nested([torch.tensor([0.2, -0.2, 0.0]), torch.tensor([-0.4, 0.4])]),
            "old_log_probs": nested([torch.full((3,), -2.0), torch.full((2,), -2.0)]),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=5, global_batch_size=2)
    return data


def actor_config(enable=True, coef=0.1, **kwargs):
    return ActorConfig(
        strategy="fsdp2",
        rollout_n=8,
        ppo_micro_batch_size_per_gpu=1,
        entropy_advantage=EntropyAdvantageConfig(enable=enable, coef=coef),
        **kwargs,
    )


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("ordinary_entropy", [False, True])
def test_real_engine_outputs_shift_and_backward(packed, ordinary_entropy):
    torch.manual_seed(3)
    data = batch_data()
    tu.assign_non_tensor(
        data,
        use_remove_padding=packed,
        calculate_entropy_advantage=True,
        entropy_advantage_chunk_size=2,
        calculate_entropy=ordinary_entropy,
    )
    dense = torch.randn(2, 5, 7)
    source = (dense.reshape(1, 10, 7) if packed else dense).requires_grad_()
    labels = torch.roll(data["input_ids"].values(), -1)
    engine = SimpleNamespace(
        engine_config=FSDPEngineConfig(),
        use_ulysses_sp=False,
        compute_entropy_from_logits=lambda z: -(z.softmax(-1) * z.log_softmax(-1)).sum(-1),
    )
    engine._gather_and_unpad_packed = lambda t, p: FSDPEngineWithLMHead._gather_and_unpad_packed(engine, t, p)
    args = {
        "input_ids_rmpad_rolled": labels,
        "temperature_rmpad": torch.ones(10),
        "temperature": torch.ones(2),
        "temperature_is_one": True,
        "pad_size": 0,
    }
    result = FSDPEngineWithLMHead.prepare_model_outputs(engine, SimpleNamespace(logits=source), args, data, None)
    reference = -(dense.log_softmax(-1).exp() * dense.log_softmax(-1)).sum(-1)
    expected = torch.stack([reference[0, 1:4], torch.cat([reference[1, 2:4], torch.zeros(1)])])
    h = no_padding_2_padding(result["ea_entropy"], data)
    torch.testing.assert_close(h, expected)
    assert not result["ea_entropy"].requires_grad
    assert ("entropy" in result) is ordinary_entropy
    objective = result["log_probs"].values().sum()
    if ordinary_entropy:
        assert result["entropy"].requires_grad
        objective = objective + result["entropy"].values().sum() * 0.03
    objective.backward()
    assert source.grad is not None and torch.isfinite(source.grad).all()
    reference_source = source.detach().clone().requires_grad_()
    flat = reference_source.reshape(-1, 7)
    expected_loss = flat.log_softmax(-1).gather(-1, labels[:, None]).sum()
    if ordinary_entropy:
        lp = flat.log_softmax(-1)
        expected_loss = expected_loss - (lp.exp() * lp).sum() * 0.03
    expected_loss.backward()
    torch.testing.assert_close(source.grad, reference_source.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("enable,coef", [(True, 0.1), (False, 0.1), (True, 0)])
def test_real_ppo_loss_receives_shaped_advantages_no_mutation(enable, coef, monkeypatch):
    data = batch_data()
    old_base = data["advantages"].values().clone()
    lp = torch.full((10,), -2.0, requires_grad=True)
    h = torch.arange(10).float().div(10).requires_grad_()
    output = {"log_probs": lp}
    if enable and coef:
        output["ea_entropy"] = h
    captured = {}

    def policy(**kwargs):
        captured.update(kwargs)
        return -(kwargs["log_prob"] * kwargs["advantages"] * kwargs["response_mask"]).sum(), {}

    monkeypatch.setattr("verl.workers.utils.losses.get_policy_loss_fn", lambda mode: policy)
    loss, metrics = ppo_loss(actor_config(enable, coef), output, data)
    base = data["advantages"].to_padded_tensor(0)
    expected = base
    if enable and coef:
        expected, _, _ = shape_entropy_advantage(
            base, no_padding_2_padding(h, data), data["response_mask"].to_padded_tensor(0), coef=coef, kappa=2
        )
    torch.testing.assert_close(captured["advantages"], expected)
    torch.testing.assert_close(data["advantages"].values(), old_base)
    assert any(k.startswith("ea/raw/") for k in metrics) is bool(enable and coef)
    loss.backward()
    assert lp.grad is not None and h.grad is None


def test_missing_ea_entropy_and_unsupported_engine_fail():
    data = batch_data()
    with pytest.raises(ValueError, match="ea_entropy.*missing"):
        ppo_loss(actor_config(), {"log_probs": torch.zeros(10)}, data)
    for fused, sp in [(True, 1), (False, 2)]:
        tu.assign_non_tensor(data, calculate_entropy_advantage=True, use_fused_kernels=fused)
        engine = SimpleNamespace(engine_config=SimpleNamespace(ulysses_sequence_parallel_size=sp))
        with pytest.raises(NotImplementedError, match="non-fused"):
            FSDPEngineWithLMHead.prepare_model_outputs(engine, None, {}, data, None)


@pytest.mark.parametrize("packed", [False, True])
def test_inactive_engine_does_not_compute_ea(packed, monkeypatch):
    data = batch_data()
    tu.assign_non_tensor(data, use_remove_padding=packed)

    def unexpected(*args, **kwargs):
        raise AssertionError("inactive EA must not compute entropy")

    monkeypatch.setattr("verl.workers.engine.fsdp.transformer_impl.entropy_advantage_from_logits", unexpected)
    engine = SimpleNamespace(engine_config=FSDPEngineConfig(), use_ulysses_sp=False)
    engine._gather_and_unpad_packed = lambda t, p: FSDPEngineWithLMHead._gather_and_unpad_packed(engine, t, p)
    logits = torch.randn(1, 10, 7) if packed else torch.randn(2, 5, 7)
    args = {
        "input_ids_rmpad_rolled": torch.roll(data["input_ids"].values(), -1),
        "temperature_rmpad": torch.ones(10),
        "temperature": torch.ones(2),
        "temperature_is_one": True,
        "pad_size": 0,
    }
    output = FSDPEngineWithLMHead.prepare_model_outputs(engine, SimpleNamespace(logits=logits), args, data, None)
    assert set(output) == {"log_probs"}


def test_original_entropy_regularizer_keeps_gradient_without_ea():
    data = batch_data()
    lp = torch.full((10,), -2.0, requires_grad=True)
    entropy = torch.ones(10, requires_grad=True)
    loss, metrics = ppo_loss(
        actor_config(enable=False, entropy_coeff=0.03), {"log_probs": lp, "entropy": entropy}, data
    )
    loss.backward()
    assert not any(key.startswith("ea/") for key in metrics)
    expected = torch.zeros(10)
    expected[[1, 2, 3, 7, 8]] = -0.03 / 5
    torch.testing.assert_close(entropy.grad, expected)


@pytest.mark.parametrize("packed", [False, True])
def test_engine_unequal_lengths_and_packed_tail_padding(packed):
    data = batch_data()
    data["prompts"] = nested([torch.tensor([1, 2]), torch.tensor([3, 4])])
    data["input_ids"] = nested([torch.tensor([1, 2, 3, 4, 0]), torch.tensor([3, 4, 2, 0])])
    tu.assign_non_tensor(
        data, use_remove_padding=packed, calculate_entropy_advantage=True, entropy_advantage_chunk_size=2
    )
    dense = torch.randn(2, 5, 7)
    dense[1, 4] = float("nan")
    valid = torch.cat([dense[0], dense[1, :4]])
    source = torch.cat([valid, torch.full((3, 7), float("nan"))]).unsqueeze(0) if packed else dense
    labels = torch.roll(data["input_ids"].values(), -1)
    if packed:
        labels = torch.cat([labels, torch.zeros(3, dtype=torch.long)])
    engine = SimpleNamespace(engine_config=FSDPEngineConfig(), use_ulysses_sp=False)
    engine._gather_and_unpad_packed = lambda t, p: FSDPEngineWithLMHead._gather_and_unpad_packed(engine, t, p)
    args = {
        "input_ids_rmpad_rolled": labels,
        "temperature_rmpad": torch.ones(12),
        "temperature": torch.ones(2),
        "temperature_is_one": True,
        "pad_size": 3 if packed else 0,
    }
    output = FSDPEngineWithLMHead.prepare_model_outputs(engine, SimpleNamespace(logits=source), args, data, None)
    h = -(valid.softmax(-1) * valid.log_softmax(-1)).sum(-1)
    torch.testing.assert_close(output["ea_entropy"].values(), h)
    response_h = no_padding_2_padding(output["ea_entropy"], data)
    torch.testing.assert_close(response_h, torch.stack([h[1:4], torch.cat([h[6:8], torch.zeros(1)])]))


def test_true_vanilla_gradient_and_stats_invariant_to_microbatch_split():
    full_data = batch_data()
    gradients, all_stats = [], []
    for split in [False, True]:
        lp = torch.full((10,), -2.0, requires_grad=True)
        h = torch.linspace(0.1, 1, 10)
        stats = {}
        parts = [full_data] if not split else [tu.index_select_tensor_dict(full_data, [i]) for i in range(2)]
        for i, data in enumerate(parts):
            selected = slice(5 * i, 5 * i + 5) if split else slice(None)
            loss, metrics = ppo_loss(actor_config(), {"log_probs": lp[selected], "ea_entropy": h[selected]}, data)
            loss.backward()
            for key, metric in metrics.items():
                if key.startswith("ea/raw/"):
                    stats[key] = stats.get(key, 0) + metric.aggregate()
        gradients.append(lp.grad)
        all_stats.append(stats)
    torch.testing.assert_close(gradients[0], gradients[1])
    assert all_stats[0] == pytest.approx(all_stats[1])


def test_dp_mini_and_local_step_weighted_statistics():
    aggregator = MetricsAggregator()
    # Each local update contains two DP ranks, two microbatches and two minibatches.
    reference = 0
    total = 0
    for update in range(2):
        raw_lists = {EA_RAW_PREFIX + "entropy_sum": [], EA_RAW_PREFIX + "valid_tokens_sum": []}
        for mini in range(2):
            counts = [[1, 2], [4, 7]]
            entropies = [[1 + update, 3 + mini], [2 + update, 4 + mini]]
            rank_counts, rank_sums = [], []
            for rank in range(2):
                count_metric = Metric(AggregationType.SUM)
                count_metric.extend([2 * c for c in counts[rank]])
                sum_metric = Metric(AggregationType.SUM)
                sum_metric.extend([2 * c * h for c, h in zip(counts[rank], entropies[rank], strict=False)])
                rank_counts.append(count_metric)
                rank_sums.append(sum_metric)
                total += sum(counts[rank])
                reference += sum(c * h for c, h in zip(counts[rank], entropies[rank], strict=False))
            raw_lists[EA_RAW_PREFIX + "entropy_sum"].append(Metric.aggregate_dp(rank_sums))
            raw_lists[EA_RAW_PREFIX + "valid_tokens_sum"].append(Metric.aggregate_dp(rank_counts))
        raw = sum_entropy_advantage_metrics(raw_lists)
        for key in (
            "nonzero_adv_tokens_sum",
            "bonus_sum",
            "bonus_ratio_sum",
            "cap_hits_sum",
            "sign_flips_sum",
            "positive_tokens_sum",
            "negative_tokens_sum",
            "base_positive_sum",
            "shaped_positive_sum",
            "base_negative_sum",
            "shaped_negative_sum",
        ):
            raw[EA_RAW_PREFIX + key] = 0
        assert raw_lists == {}
        aggregator.add_step_metrics(raw, sample_count=10 + update)
    final = aggregator.get_aggregated_metrics()
    finalize_entropy_advantage_metrics(final)
    assert final["actor/ea/entropy_token_mean"] == pytest.approx(reference / total)
    assert final[EA_RAW_PREFIX + "valid_tokens_sum"] == total
    assert final["actor/ea/cap_fraction"] == final["actor/ea/sign_flips"] == 0


def test_no_dp_raw_metric_and_disabled_finalize():
    metric = Metric(AggregationType.SUM)
    metric.extend([2, 3])
    stats = {EA_RAW_PREFIX + "valid_tokens_sum": metric, "actor/loss": [1, 2]}
    assert sum_entropy_advantage_metrics(stats) == {EA_RAW_PREFIX + "valid_tokens_sum": 5}
    assert stats == {"actor/loss": [1, 2]}
    finalize_entropy_advantage_metrics(stats)
    assert stats == {"actor/loss": [1, 2]}


@pytest.mark.parametrize("enable,coef", [(True, 0.1), (False, 0.1), (True, 0)])
def test_trainer_tq_mini_micro_metadata(enable, coef, monkeypatch):
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "ppo_mini_batch_size": 1,
                    "ppo_epochs": 1,
                    "calculate_entropy": False,
                    "entropy_coeff": 0,
                    "data_loader_seed": 42,
                    "shuffle": False,
                    "entropy_advantage": {"enable": enable, "coef": coef},
                },
                "rollout": {"n": 2, "temperature": 1.0},
            }
        }
    )
    config.actor_rollout_ref.actor.entropy_advantage.enable = enable
    config.actor_rollout_ref.actor.entropy_advantage.coef = coef
    trainer = SimpleNamespace(config=config)
    trainer.actor_rollout_wg = SimpleNamespace(update_actor=Mock(return_value={"metrics": {"mfu": [0.1]}}))
    meta = SimpleNamespace(extra_info={}, size=2)
    PPOTrainer._update_actor(trainer, meta, {})
    active = bool(enable and coef)
    assert meta.extra_info["calculate_entropy"] is False
    assert ("calculate_entropy_advantage" in meta.extra_info) is active
    monkeypatch.setattr(tqu, "KVBatchMeta", type("NotThisMeta", (), {}))

    class Client:
        async def async_get_data(self, metadata):
            return batch_data()

    monkeypatch.setattr(tqu.tq, "get_client", lambda: Client())
    data = asyncio.run(tqu._async_meta_to_realdata(meta))
    tu.assign_non_tensor(data, use_dynamic_bsz=False, micro_batch_size_per_gpu=1)
    for mini in tu.make_iterator(data, mini_batch_size=2, epochs=2):
        microbatches, _ = prepare_micro_batches(mini)
        for micro in microbatches:
            assert tu.get_non_tensor_data(micro, "calculate_entropy_advantage", False) is active
            assert tu.get_non_tensor_data(micro, "calculate_entropy", None) is False
            if active:
                assert tu.get_non_tensor_data(micro, "entropy_advantage_chunk_size", None) == 64
