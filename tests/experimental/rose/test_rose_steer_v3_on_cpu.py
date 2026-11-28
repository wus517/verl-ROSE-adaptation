"""Check the ROSE advantage path with DAPO rewards and STEER v8."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.dapo import DAPORewardManager
from verl.experimental.rose.tree import BranchRecord, finalize_tree
from verl.experimental.rose.tree_advantage import compute_rose_tree_advantage
from verl.trainer.ppo.steer_metrics import SteerV4Metrics
from verl.trainer.ppo.v1 import trainer_base
from verl.utils import tensordict_utils as tu
from verl.workers.config.actor import ActorConfig, PolicyLossConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import response_from_nested


async def _dapo_rewards(lengths):
    async def grader(ground_truth, **_):
        # Exercise both scoring forms that DAPO passes through.
        return {"score": 1.0} if ground_truth == "correct-dict" else float(ground_truth == "correct-scalar")

    config = OmegaConf.create(
        {
            "reward": {
                "reward_kwargs": {
                    "max_resp_len": 8,
                    "overlong_buffer_cfg": {"enable": True, "len": 4, "penalty_factor": 1.0, "log": False},
                }
            }
        }
    )
    manager = DAPORewardManager(config, SimpleNamespace(decode=lambda *_, **__: "response"), grader)
    results = []
    for length, label in zip(lengths, ("correct-dict", "correct-scalar", "wrong", "wrong"), strict=True):
        sample = DataProto(
            batch=TensorDict(
                {
                    "responses": torch.ones((1, length), dtype=torch.long),
                    "attention_mask": torch.ones((1, length + 1), dtype=torch.long),
                },
                batch_size=[1],
            ),
            non_tensor_batch={"data_source": np.array(["math"]), "reward_model": np.array([{"ground_truth": label}])},
        )
        results.append(await manager.run_single(sample))
    return results


def _nested(rows):
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def test_dapo_rewards_drive_tree_values_but_unshaped_scores_drive_calibration(monkeypatch):
    lengths = (4, 6, 6, 3)
    scores = asyncio.run(_dapo_rewards(lengths))
    assert [float(result["reward_score"]) for result in scores] == [1.0, 0.5, -0.5, 0.0]
    metadata = finalize_tree(
        "prompt",
        [
            BranchRecord(0, None, 0, True, 4),
            BranchRecord(1, 0, 2, False, 6),
            BranchRecord(2, 1, 4, False, 6),
            BranchRecord(3, None, 0, True, 3),
        ],
    )
    reward_rows = [torch.zeros(length) for length in lengths]
    for row, result in zip(reward_rows, scores, strict=True):
        row[-1] = result["reward_score"]
    batch = MagicMock(keys=[f"prompt_{row}_0" for row in range(len(lengths))], partition_id="train")
    batch.__len__.return_value = len(lengths)
    queue_data = tu.get_tensordict(
        {
            "uid": ["prompt"] * len(lengths),
            "response_mask": _nested([torch.ones(length, dtype=torch.long) for length in lengths]),
            "rm_scores": _nested(reward_rows),
            "extra_fields": [
                {"rose_tree_metadata": metadata[row], "reward_extra_info": result["reward_extra_info"]}
                for row, result in enumerate(scores)
            ],
        }
    )
    queue = SimpleNamespace(kv_batch_get=Mock(return_value=queue_data), kv_batch_put=Mock(return_value=batch))
    monkeypatch.setattr(trainer_base, "tq", queue)
    trainer = SimpleNamespace(
        config=OmegaConf.create(
            {
                "algorithm": {
                    "adv_estimator": "rose_tree",
                    "gamma": 1.0,
                    "lam": 1.0,
                    "use_kl_in_reward": False,
                    "advantage_extra_fields": ["rose_tree_metadata", "reward_extra_info"],
                    "rose": {"length_calibration_alpha": 0.5, "use_unshaped_reward_for_correctness": True},
                },
                "actor_rollout_ref": {"rollout": {"n": len(lengths)}},
            }
        )
    )
    assert trainer_base.PPOTrainer._compute_advantage(trainer, batch, {}) is batch
    assert "extra_fields" in queue.kv_batch_get.call_args.kwargs["select_fields"]
    written = queue.kv_batch_put.call_args.kwargs["fields"]

    # Root value 1/4, first shared node 1/3, second shared node 0.
    # The long correct path is still calibrated even though its shaped reward is 0.5.
    prefix = 1 / 12
    calibration = 0.5**0.5
    middle = -(2 - calibration) / 3
    end = 0.5 * calibration
    expected = torch.tensor(
        [
            [prefix, prefix, 2 / 3, 2 / 3, 0, 0],
            [prefix, prefix, middle, middle, end, end],
            [prefix, prefix, -1 / 3, -1 / 3, -0.5, -0.5],
            [-0.25, -0.25, -0.25, 0, 0, 0],
        ]
    )
    torch.testing.assert_close(written.to_padded_tensor()["advantages"], expected)
    torch.testing.assert_close(written.to_padded_tensor()["returns"], expected)

    # Consume the second path directly through the inherited STEER v8 loss bridge.
    entropy = torch.tensor([1.6, 0.7, 0.7, 1.4, 1.8, 0.8])
    current = torch.tensor([-1.0] * 6 + [0.0], requires_grad=True)
    response_mask = _nested([torch.ones(6, dtype=torch.long)])
    actor = ActorConfig(
        strategy="fsdp2",
        rollout_n=4,
        ppo_micro_batch_size_per_gpu=1,
        policy_loss=PolicyLossConfig(
            loss_mode="entropy_control",
            token_weight_min=0.7,
            token_weight_max=1.0,
            linear=False,
            steer_quadrant_logging=True,
            steer_entropy_weighting=True,
            steer_recovery_quadrants=[1, 2, 3, 4],
        ),
    )
    data = TensorDict(
        {
            "prompts": torch.ones((1, 1), dtype=torch.long),
            "responses": torch.ones((1, 6), dtype=torch.long),
            "attention_mask": torch.ones((1, 7), dtype=torch.long),
            "response_mask": response_mask,
            "old_log_probs": response_from_nested(_nested([current.detach()]), response_mask),
            "entropy": _nested([entropy]),
            "advantages": _nested([written["advantages"][1]]),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor(
        data,
        dp_size=1,
        batch_num_tokens=19,
        global_batch_size=4,
        steer_local_mini_batch_size=4,
        steer_response_length=6,
    )
    metrics = SteerV4Metrics()
    loss, _ = ppo_loss(actor, {"log_probs": current}, data, steer_metrics=metrics)
    loss.backward()
    scale = (1.0 - torch.exp(torch.tensor(-1.0))) * (entropy - 1.0).abs()
    magnitude = expected[1].abs() * scale
    exponent = magnitude / magnitude.max().clamp_min(0.02)
    high_entropy = torch.tensor([True, False, False, False, True, False])
    weights = torch.where(high_entropy, 0.35 + 0.65 * (0.7**exponent), 0.5**exponent)
    gradient = -expected[1] * weights / (6 * 4)
    torch.testing.assert_close(current.grad, torch.cat([gradient, torch.zeros(1)]))
    torch.testing.assert_close(loss.detach(), gradient.sum())
    logged = metrics.flush()
    for quadrant, count in enumerate((2, 2, 1, 1), start=1):
        assert logged[f"steer/q{quadrant}_frac"] == pytest.approx(count / 6)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (None, "reward_extra_info for each response"),
        ([{}], "unshaped score or acc"),
        ([{"acc": 0.5}], "binary rewards"),
    ],
)
def test_unshaped_correctness_requires_a_binary_score(extra, message):
    metadata = finalize_tree("prompt", [BranchRecord(0, None, 0, True, 2)])[0]
    with pytest.raises(ValueError, match=message):
        compute_rose_tree_advantage(
            torch.tensor([[0.0, 0.5]]),
            torch.tensor([[1, 1]]),
            np.array(["prompt"], dtype=object),
            [metadata],
            config={"rose": {"use_unshaped_reward_for_correctness": True}},
            reward_extra_info=extra,
        )
