"""CPU-only tests for ROSE tree construction and advantages."""

import numpy as np
import pytest
import torch

from verl.experimental.rose.tree import BranchRecord, finalize_tree
from verl.experimental.rose.tree_advantage import compute_rose_tree_advantage
from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import compute_advantage


def _config(alpha: float = 1.0):
    return {"rose": {"length_calibration_alpha": alpha, "require_binary_reward": True}}


def test_finalize_tree_splits_existing_edges_and_preserves_root_restarts():
    metadata = finalize_tree(
        "prompt-1",
        [
            BranchRecord(0, None, 0, True, 8),
            BranchRecord(1, 0, 4, False, 7),
            BranchRecord(2, 1, 2, False, 6),
            BranchRecord(3, None, 0, True, 5),
        ],
    )

    assert metadata[0]["path_ends"] == [0, 2, 4, 8]
    assert metadata[1]["path_ends"] == [0, 2, 4, 7]
    assert metadata[2]["path_ends"] == [0, 2, 6]
    assert metadata[3]["path_ends"] == [0, 5]
    assert metadata[0]["path_node_ids"][:3] == metadata[1]["path_node_ids"][:3]
    assert metadata[2]["path_node_ids"][:2] == metadata[1]["path_node_ids"][:2]
    assert metadata[3]["path_node_ids"][0] == metadata[0]["path_node_ids"][0]
    assert metadata[3]["path_node_ids"][1] not in metadata[0]["path_node_ids"]


def test_tree_advantage_assigns_node_value_differences_and_ignores_padding():
    metadata_by_id = finalize_tree(
        "prompt-1",
        [
            BranchRecord(0, None, 0, True, 4),
            BranchRecord(1, 0, 2, False, 4),
        ],
    )
    rewards = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]])
    padding_metadata = {"not": "parsed"}

    advantages, returns = compute_rose_tree_advantage(
        rewards,
        mask,
        np.array(["prompt-1", "prompt-1", "padding"], dtype=object),
        [metadata_by_id[0], metadata_by_id[1], padding_metadata],
        config=_config(),
    )

    expected = torch.tensor(
        [
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, -0.5, -0.5],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(returns, expected)


def test_tree_advantage_is_invariant_to_batch_order():
    metadata_by_id = finalize_tree(
        "prompt-1",
        [BranchRecord(0, None, 0, True, 3), BranchRecord(1, 0, 1, False, 3)],
    )
    rewards = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    mask = torch.ones_like(rewards, dtype=torch.int64)
    first, _ = compute_rose_tree_advantage(
        rewards,
        mask,
        np.array(["prompt-1", "prompt-1"], dtype=object),
        [metadata_by_id[0], metadata_by_id[1]],
        config=_config(),
    )
    second, _ = compute_rose_tree_advantage(
        rewards.flip(0),
        mask.flip(0),
        np.array(["prompt-1", "prompt-1"], dtype=object),
        [metadata_by_id[1], metadata_by_id[0]],
        config=_config(),
    )
    torch.testing.assert_close(first, second.flip(0))


def test_length_calibration_uses_real_last_common_node():
    metadata_by_id = finalize_tree(
        "prompt-1",
        [
            BranchRecord(0, None, 0, True, 4),
            BranchRecord(1, 0, 2, False, 6),
            BranchRecord(2, 1, 4, False, 6),
        ],
    )
    rewards = torch.zeros((3, 6), dtype=torch.float32)
    rewards[0, 3] = 1.0
    rewards[1, 5] = 1.0
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ]
    )

    uncalibrated, _ = compute_rose_tree_advantage(
        rewards,
        mask,
        np.array(["prompt-1", "prompt-1", "prompt-1"], dtype=object),
        [metadata_by_id[0], metadata_by_id[1], metadata_by_id[2]],
        config=_config(alpha=0.0),
    )
    calibrated, _ = compute_rose_tree_advantage(
        rewards,
        mask,
        np.array(["prompt-1", "prompt-1", "prompt-1"], dtype=object),
        [metadata_by_id[0], metadata_by_id[1], metadata_by_id[2]],
        config=_config(alpha=1.0),
    )

    torch.testing.assert_close(calibrated[1, :2], uncalibrated[1, :2])
    expected_suffix = uncalibrated[1, 2:6] - uncalibrated[1, 2:6].abs() * 0.5
    torch.testing.assert_close(calibrated[1, 2:6], expected_suffix)


def test_tree_advantage_rejects_non_binary_rewards():
    metadata = finalize_tree("prompt-1", [BranchRecord(0, None, 0, True, 2)])[0]
    with pytest.raises(ValueError, match="binary rewards"):
        compute_rose_tree_advantage(
            torch.tensor([[0.0, 0.5]]),
            torch.tensor([[1, 1]]),
            np.array(["prompt-1"], dtype=object),
            [metadata],
            config=_config(),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("parent_trajectory_id", 99, "unknown parent"),
        ("root_restart", True, "cannot set root_restart"),
        ("branch_pos", 1, "not a finalized node"),
    ],
)
def test_tree_advantage_rejects_invalid_parent_relationships(field, value, message):
    metadata_by_id = finalize_tree(
        "prompt-1",
        [BranchRecord(0, None, 0, True, 4), BranchRecord(1, 0, 2, False, 4)],
    )
    metadata_by_id[1][field] = value

    with pytest.raises(ValueError, match=message):
        compute_rose_tree_advantage(
            torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]]),
            torch.ones((2, 4), dtype=torch.int64),
            np.array(["prompt-1", "prompt-1"], dtype=object),
            [metadata_by_id[0], metadata_by_id[1]],
            config=_config(),
        )


def test_tree_advantage_rejects_parent_cycles():
    metadata_by_id = finalize_tree(
        "prompt-1",
        [BranchRecord(0, None, 0, True, 4), BranchRecord(1, 0, 2, False, 4)],
    )
    metadata_by_id[0]["parent_trajectory_id"] = 1
    metadata_by_id[0]["root_restart"] = False

    with pytest.raises(ValueError, match="cycle"):
        compute_rose_tree_advantage(
            torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]]),
            torch.ones((2, 4), dtype=torch.int64),
            np.array(["prompt-1", "prompt-1"], dtype=object),
            [metadata_by_id[0], metadata_by_id[1]],
            config=_config(),
        )


def test_generic_advantage_dispatch_forwards_tree_metadata():
    metadata_by_id = finalize_tree(
        "prompt-1",
        [BranchRecord(0, None, 0, True, 4), BranchRecord(1, 0, 2, False, 4)],
    )
    rewards = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    mask = torch.ones_like(rewards, dtype=torch.int64)
    data = DataProto.from_dict(
        tensors={"token_level_rewards": rewards, "response_mask": mask},
        non_tensors={
            "uid": np.array(["prompt-1", "prompt-1"], dtype=object),
            "rose_tree_metadata": np.array([metadata_by_id[0], metadata_by_id[1]], dtype=object),
        },
    )

    result = compute_advantage(
        data,
        adv_estimator="rose_tree",
        config={"advantage_extra_fields": ["rose_tree_metadata"], **_config()},
    )

    torch.testing.assert_close(
        result.batch["advantages"],
        torch.tensor([[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, -0.5, -0.5]]),
    )
