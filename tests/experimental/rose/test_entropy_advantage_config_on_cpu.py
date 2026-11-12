# Copyright 2026 individual contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

from pathlib import Path

import hydra
import pytest
from omegaconf import OmegaConf

from verl.utils.config import omega_conf_to_dataclass, validate_entropy_advantage_config
from verl.workers.config.actor import EntropyAdvantageConfig, get_entropy_advantage_config

ROOT = Path(__file__).resolve().parents[3]
MANAGER = "verl.experimental.rose.rose_agent_loop_tq.RoseAgentLoopManagerTQ"


def compose_config(overrides=()):
    with hydra.initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        return hydra.compose(config_name="ppo_trainer", overrides=list(overrides))


def active_config():
    return compose_config(
        [
            "algorithm.adv_estimator=rose_tree",
            "actor_rollout_ref.rollout.rose.enable=true",
            f"actor_rollout_ref.rollout.agent.agent_loop_manager_class={MANAGER}",
            "actor_rollout_ref.actor.entropy_advantage.enable=true",
            "actor_rollout_ref.rollout.name=vllm",
        ]
    )


def test_default_schema_hydra_and_direct_config():
    config = compose_config()
    ea = hydra.utils.instantiate(config.actor_rollout_ref.actor.entropy_advantage)
    assert isinstance(ea, EntropyAdvantageConfig) and not ea.active
    assert (ea.coef, ea.kappa, ea.entropy_chunk_size) == (0.1, 2, 64)
    config.actor_rollout_ref.rollout.n = 8
    config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = 1
    actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    assert isinstance(actor.entropy_advantage, EntropyAdvantageConfig)
    assert get_entropy_advantage_config({}) == ea
    assert get_entropy_advantage_config(actor) is actor.entropy_advantage


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enable": 1},
        {"coef": True},
        {"coef": -1},
        {"coef": float("nan")},
        {"coef": float("inf")},
        {"kappa": 1},
        {"kappa": -1},
        {"kappa": float("inf")},
        {"kappa": True},
        {"entropy_chunk_size": 0},
        {"entropy_chunk_size": True},
        {"entropy_chunk_size": 1.5},
    ],
)
def test_invalid_fields(kwargs):
    with pytest.raises(ValueError, match="entropy_advantage"):
        EntropyAdvantageConfig(**kwargs)


def test_active_valid_and_zero_coef_noop():
    config = active_config()
    validate_entropy_advantage_config(config)
    config.actor_rollout_ref.rollout.val_kwargs.temperature = 0
    validate_entropy_advantage_config(config)
    config.actor_rollout_ref.actor.entropy_advantage.coef = 0
    config.trainer.v1.trainer_mode = "fully_async"
    validate_entropy_advantage_config(config)


@pytest.mark.parametrize(
    "path,value",
    [
        ("trainer.use_v1", False),
        ("trainer.v1.trainer_mode", "fully_async"),
        ("algorithm.adv_estimator", "grpo"),
        ("actor_rollout_ref.rollout.rose.enable", False),
        ("actor_rollout_ref.rollout.agent.agent_loop_manager_class", "wrong.Manager"),
        ("actor_rollout_ref.actor.strategy", "megatron"),
        ("actor_rollout_ref.rollout.name", "sglang"),
        ("actor_rollout_ref.actor.policy_loss.loss_mode", "gspo"),
        ("actor_rollout_ref.rollout.temperature", 0.6),
        ("actor_rollout_ref.actor.entropy_coeff", 0.01),
        ("distillation.enabled", True),
        ("actor_rollout_ref.model.use_fused_kernels", True),
        ("actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size", 2),
        ("actor_rollout_ref.actor.ulysses_sequence_parallel_size", 2),
        ("actor_rollout_ref.rollout.multi_turn.enable", True),
        ("actor_rollout_ref.rollout.agent.default_agent_loop", "tool_agent"),
        ("actor_rollout_ref.actor.use_prefix_grouper", True),
    ],
)
def test_active_guards_only_when_active(path, value):
    config = active_config()
    OmegaConf.update(config, path, value)
    with pytest.raises(ValueError, match="Entropy advantage currently requires"):
        validate_entropy_advantage_config(config)
    config.actor_rollout_ref.actor.entropy_advantage.enable = False
    validate_entropy_advantage_config(config)
