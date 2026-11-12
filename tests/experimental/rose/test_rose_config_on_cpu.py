"""CPU-only tests for ROSE configuration validation."""

import pytest
from omegaconf import OmegaConf

from verl.trainer.config.algorithm import RoseAlgorithmConfig
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config.rollout import RoseLookaheadConfig, RoseRolloutConfig


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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"windows": []},
        {"windows": [20, 20]},
        {"windows": [30, 20]},
        {"windows": [True]},
        {"edit_distance_threshold": float("nan")},
        {"edit_distance_threshold": 1.1},
        {"max_trials_per_pivot": 0},
        {"max_pivots_per_slot": -1},
        {"max_probe_tokens_per_group": -1},
        {"request_timeout_seconds": float("inf")},
        {"request_timeout_seconds": 0},
        {"mode": "online"},
        {"fallback": "padding"},
        {"short_terminal_policy": "guess"},
    ],
)
def test_lookahead_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        RoseLookaheadConfig(**kwargs)


def test_lookahead_default_lists_and_parent_dependency():
    first, second = RoseLookaheadConfig(), RoseLookaheadConfig()
    first.windows.append(60)
    assert second.windows == [20, 30, 50]
    with pytest.raises(ValueError, match="requires"):
        RoseRolloutConfig(lookahead=RoseLookaheadConfig(enable=True))


def test_actual_rollout_yaml_instantiates_nested_lookahead():
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "verl/trainer/config/rollout/rollout.yaml"
    config = OmegaConf.load(path).rose
    config.enable = True
    config.embedding_path = "/tmp/embedding.f16"
    config.lookahead.enable = True
    rose = omega_conf_to_dataclass(config)
    assert isinstance(rose.lookahead, RoseLookaheadConfig)
    assert rose.lookahead.enable and rose.lookahead.windows == [20, 30, 50]
    typed = omega_conf_to_dataclass(config, RoseRolloutConfig)
    assert isinstance(typed.lookahead, RoseLookaheadConfig)
