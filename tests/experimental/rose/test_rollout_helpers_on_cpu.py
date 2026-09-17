"""CPU-only tests for ROSE rollout control-flow helpers."""

import asyncio
import random
from types import SimpleNamespace

import numpy as np
import pytest

from verl.experimental.rose.rose_agent_loop import RoseAgentLoop
from verl.experimental.rose.rose_agent_loop_tq import (
    RoseAgentLoopWorkerTQBase,
    _choose_branch,
    _select_branch,
    _Trajectory,
    _validate_policy_versions,
)
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ, AgentLoopWorkerTQBase
from verl.trainer.ppo.v1.trainer_base import _compute_rose_rollout_metrics, _extract_advantage_extra_fields
from verl.workers.rollout.logprobs import (
    ROSE_TOPK_LOGPROBS_FIELD,
    ROSE_TOPK_TOKEN_IDS_FIELD,
    ROSE_TOPK_VALID_MASK_FIELD,
)


def _trajectory(scores, *, global_steps=3, min_global_steps=3, max_global_steps=3):
    output = SimpleNamespace(
        extra_fields={
            "global_steps": global_steps,
            "min_global_steps": min_global_steps,
            "max_global_steps": max_global_steps,
        }
    )
    return _Trajectory(output, np.asarray(scores, dtype=np.float32), None, 0, True)


def test_select_branch_uses_global_maximum_and_stable_tie_break():
    trajectories = [_trajectory([0.1, 0.8]), _trajectory([0.8, 0.2])]

    assert _select_branch(trajectories) == (0, 1)


def test_select_branch_returns_none_without_finite_score():
    assert _select_branch([_trajectory([np.nan, -np.inf])]) is None


def test_choose_branch_obeys_epsilon_extremes():
    trajectories = [_trajectory([0.1, 0.8])]

    assert _choose_branch(trajectories, random.Random(0), epsilon=1.0) is None
    assert _choose_branch(trajectories, random.Random(0), epsilon=0.0) == (0, 1)
    assert _choose_branch([], random.Random(0), epsilon=0.0) is None


def test_validate_policy_versions_accepts_one_atomic_version():
    _validate_policy_versions([_trajectory([0.1]), _trajectory([0.2])], expected_version=3)


@pytest.mark.parametrize(
    ("trajectories", "expected_version", "message"),
    [
        ([_trajectory([0.1], min_global_steps=2, max_global_steps=3)], 3, "spans policy versions"),
        ([_trajectory([0.1], global_steps=2, min_global_steps=3, max_global_steps=3)], 3, "inconsistent"),
        ([_trajectory([0.1], global_steps=2, min_global_steps=2, max_global_steps=2)], 3, "expected policy version"),
    ],
)
def test_validate_policy_versions_fails_closed(trajectories, expected_version, message):
    with pytest.raises(RuntimeError, match=message):
        _validate_policy_versions(trajectories, expected_version)


def test_tq_worker_base_remains_public_and_extensible():
    assert hasattr(AgentLoopWorkerTQ, "remote")
    assert issubclass(RoseAgentLoopWorkerTQBase, AgentLoopWorkerTQBase)


def test_extract_advantage_extra_fields_preserves_object_metadata():
    metadata = [{"tree_id": "a"}, {"tree_id": "b"}]
    extracted = _extract_advantage_extra_fields(
        [{"rose_tree_metadata": metadata[0]}, {"rose_tree_metadata": metadata[1]}],
        ["rose_tree_metadata"],
    )

    assert extracted["rose_tree_metadata"].shape == (2,)
    assert extracted["rose_tree_metadata"].tolist() == metadata


def test_extract_advantage_extra_fields_rejects_missing_rows():
    with pytest.raises(KeyError, match="missing from rows"):
        _extract_advantage_extra_fields(
            [{"rose_tree_metadata": {"tree_id": "a"}}, {}],
            ["rose_tree_metadata"],
        )


def test_generate_from_prefix_consumes_topk_and_preserves_prefix():
    class FakeServer:
        def __init__(self):
            self.request = None

        async def generate(self, **kwargs):
            self.request = kwargs
            return SimpleNamespace(
                token_ids=[13, 14],
                log_probs=[-0.3, -0.4],
                num_preempted=0,
                extra_fields={
                    "global_steps": 3,
                    "min_global_steps": 3,
                    "max_global_steps": 3,
                    ROSE_TOPK_TOKEN_IDS_FIELD: np.array([[13, 15], [14, 16]], dtype=np.int32),
                    ROSE_TOPK_LOGPROBS_FIELD: np.array([[-0.3, -1.0], [-0.4, -1.1]], dtype=np.float32),
                    ROSE_TOPK_VALID_MASK_FIELD: np.ones((2, 2), dtype=np.bool_),
                },
            )

    async def run():
        server = FakeServer()
        agent_loop = RoseAgentLoop.__new__(RoseAgentLoop)
        agent_loop.rollout_config = SimpleNamespace(response_length=4)
        agent_loop.server_manager = server

        output, topk_ids, topk_logprobs, topk_valid_mask = await agent_loop.generate_from_prefix(
            request_id="rose-test",
            prompt_ids=[1, 2],
            prefix_ids=[11, 12],
            prefix_logprobs=[-0.1, -0.2],
            sampling_params={"temperature": 1.0, "logprobs": 2},
        )

        assert server.request["prompt_ids"] == [1, 2, 11, 12]
        assert server.request["sampling_params"]["max_tokens"] == 2
        assert output.response_ids == [11, 12, 13, 14]
        assert output.response_logprobs == [-0.1, -0.2, -0.3, -0.4]
        assert output.extra_fields["global_steps"] == 3
        assert ROSE_TOPK_TOKEN_IDS_FIELD not in output.extra_fields
        np.testing.assert_array_equal(topk_ids, [[13, 15], [14, 16]])
        np.testing.assert_allclose(topk_logprobs, [[-0.3, -1.0], [-0.4, -1.1]])
        np.testing.assert_array_equal(topk_valid_mask, np.ones((2, 2), dtype=np.bool_))

    asyncio.run(run())


def test_compute_rose_rollout_metrics_ignores_padding_and_reports_timing():
    rows = [
        {
            "rose_tree_metadata": {"path_node_ids": [0, 1]},
            "rose_semantic_score_seconds": 0.25,
            "rose_generation_seconds": 1.0,
            "rose_prefix_reuse_tokens": 0,
            "rose_new_generated_tokens": 4,
            "rose_root_restart": True,
            "rose_branch_pos": 0,
            "rose_semantic_entropy_max": 0.4,
            "rose_topk_payload_bytes": 100,
        },
        {
            "rose_tree_metadata": {"path_node_ids": [0, 2, 3]},
            "rose_semantic_score_seconds": 0.25,
            "rose_generation_seconds": 1.0,
            "rose_prefix_reuse_tokens": 2,
            "rose_new_generated_tokens": 2,
            "rose_root_restart": False,
            "rose_branch_pos": 2,
            "rose_semantic_entropy_max": 0.8,
            "rose_topk_payload_bytes": 50,
        },
        {},
    ]

    metrics = _compute_rose_rollout_metrics(rows, np.array([True, True, False]))

    assert metrics["rose/root_restart_ratio"] == pytest.approx(0.5)
    assert metrics["rose/prefix_reuse_tokens"] == pytest.approx(1.0)
    assert metrics["rose/tree_depth_max"] == pytest.approx(2.0)
    assert metrics["rose/semantic_entropy_max"] == pytest.approx(0.8)
    assert metrics["rose/semantic_tokens_per_second"] == pytest.approx(12.0)
    assert metrics["rose/semantic_score_fraction_of_generation"] == pytest.approx(0.25)
