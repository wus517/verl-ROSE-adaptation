# Copyright 2026 The verl team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.v1.utils import compute_advantage_for_multi_trajectories
from verl.utils import tensordict_utils as tu
from verl.workers.config.actor import ActorConfig, PolicyLossConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import response_from_nested


def test_rose_rollouts_drive_steer_with_session_advantage_and_cached_entropy():
    # The two complete ROSE responses share the first token and have distinct terminal rewards.
    batch = DataProto(
        batch=TensorDict(
            {
                "response_mask": torch.tensor([[1, 1], [1, 0]]),
                "token_level_rewards": torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
            },
            batch_size=[2],
        ),
        non_tensor_batch={"uid": np.array(["prompt", "prompt"], dtype=object)},
    )
    batch = compute_advantage_for_multi_trajectories(
        batch,
        batch_keys=["prompt_0_0", "prompt_1_0"],
        adv_estimator=AdvantageEstimator.GRPO,
        num_repeat=2,
        norm_adv_by_std_in_grpo=True,
    )
    group_advantage = 0.5 / (2**-0.5 + 1e-6)
    torch.testing.assert_close(
        batch.batch["advantages"],
        torch.tensor([[group_advantage, group_advantage], [-group_advantage, 0.0]]),
    )

    # Emulate the first actor microbatch with V1's response-only cached entropy and old log-prob.
    config = ActorConfig(
        strategy="fsdp2",
        rollout_n=2,
        ppo_micro_batch_size_per_gpu=1,
        policy_loss=PolicyLossConfig(
            loss_mode="entropy_control",
            token_weight_min=0.7,
            token_weight_max=1.0,
            linear=False,
        ),
    )
    response_mask = torch.nested.nested_tensor([torch.tensor([1, 1])], layout=torch.jagged)
    old_logp = torch.nested.nested_tensor([torch.tensor([-1.0, -1.0, 0.0])], layout=torch.jagged)
    old_entropy = torch.nested.nested_tensor([torch.tensor([2.0, 1.0, 0.0])], layout=torch.jagged)
    data = TensorDict(
        {
            "prompts": torch.ones((1, 1), dtype=torch.long),
            "responses": torch.ones((1, 2), dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "response_mask": response_mask,
            "old_log_probs": response_from_nested(old_logp, response_mask),
            "entropy": response_from_nested(old_entropy, response_mask),
            "advantages": torch.nested.nested_tensor([batch.batch["advantages"][0]], layout=torch.jagged),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor(
        data,
        dp_size=1,
        batch_num_tokens=3,
        global_batch_size=2,
        steer_response_length=3,
        steer_local_mini_batch_size=2,
    )
    current_logp = torch.tensor([-1.0, -1.0, 0.0], requires_grad=True)

    loss, _ = ppo_loss(config, {"log_probs": current_logp, "entropy": torch.zeros_like(current_logp)}, data)
    loss.backward()

    # Cached entropy makes the first token hit the weight floor and the second score zero.
    # Local token normalization and two microbatches each contribute a factor of 1/2.
    expected_gradient = -group_advantage * torch.tensor([0.7, 1.0, 0.0]) / 4
    torch.testing.assert_close(current_logp.grad, expected_gradient)
    torch.testing.assert_close(loss.detach(), expected_gradient.sum())
