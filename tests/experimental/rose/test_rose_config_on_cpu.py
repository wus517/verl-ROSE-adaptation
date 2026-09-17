"""CPU-only tests for ROSE configuration validation."""

import pytest

from verl.trainer.config.algorithm import RoseAlgorithmConfig
from verl.workers.config.rollout import RoseRolloutConfig


def test_rose_rollout_config_accepts_enabled_cpu_configuration():
    config = RoseRolloutConfig(enable=True, embedding_path="/tmp/embedding.f16", num_trajectories=8)

    assert config.top_k == 20
    assert config.semantic_device == "cpu"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_trajectories": 0}, "num_trajectories"),
        ({"epsilon": -0.1}, "epsilon"),
        ({"epsilon": 1.1}, "epsilon"),
        ({"top_k": 1}, "top_k"),
        ({"semantic_device": "npu"}, "semantic_device"),
        ({"semantic_chunk_size": 0}, "semantic_chunk_size"),
        ({"validation_mode": "tree"}, "validation_mode"),
        ({"enable": True}, "embedding_path"),
    ],
)
def test_rose_rollout_config_rejects_unsupported_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RoseRolloutConfig(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"length_calibration_alpha": -1.0}, "length_calibration_alpha"),
        ({"binary_reward_tolerance": -1.0}, "binary_reward_tolerance"),
    ],
)
def test_rose_algorithm_config_rejects_negative_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RoseAlgorithmConfig(**kwargs)
