from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from verl.trainer.ppo.core_algos import register_adv_est


def _config_value(config: Any, name: str, default: Any) -> Any:
    rose_config = config.get("rose", None) if config is not None else None
    if rose_config is None:
        return default
    if isinstance(rose_config, Mapping):
        return rose_config.get(name, default)
    return getattr(rose_config, name, default)


def _validate_metadata(metadata: Mapping[str, Any], response_length: int) -> tuple[list[int], list[int]]:
    if int(metadata.get("schema_version", -1)) != 1:
        raise ValueError(f"unsupported ROSE tree schema: {metadata.get('schema_version')}")
    node_ids = [int(value) for value in metadata["path_node_ids"]]
    path_ends = [int(value) for value in metadata["path_ends"]]
    if len(node_ids) != len(path_ends) or len(node_ids) < 2:
        raise ValueError("path_node_ids and path_ends must be equal-length root-to-terminal paths")
    if path_ends[0] != 0 or path_ends[-1] != response_length:
        raise ValueError(f"path endpoints must span [0, {response_length}], got [{path_ends[0]}, {path_ends[-1]}]")
    if any(left >= right for left, right in zip(path_ends, path_ends[1:], strict=False)):
        raise ValueError(f"path_ends must be strictly increasing, got {path_ends}")
    if len(set(node_ids)) != len(node_ids):
        raise ValueError(f"tree path contains repeated node ids: {node_ids}")
    return node_ids, path_ends


def _last_common_node_position(
    first_node_ids: Sequence[int],
    first_path_ends: Sequence[int],
    second_node_ids: Sequence[int],
) -> int:
    common_count = 0
    for first, second in zip(first_node_ids, second_node_ids, strict=False):
        if first != second:
            break
        common_count += 1
    if common_count == 0:
        raise ValueError("ROSE trajectories in one tree do not share a root node")
    return int(first_path_ends[common_count - 1])


def _validate_tree_relationships(
    tree_id: str,
    rows: list[int],
    rose_tree_metadata: Sequence[Mapping[str, Any]],
    parsed_paths: Mapping[int, tuple[list[int], list[int]]],
) -> None:
    row_by_trajectory: dict[int, int] = {}
    parent_by_trajectory: dict[int, int | None] = {}
    terminal_owners: dict[int, int] = {}
    node_signatures: dict[int, tuple[tuple[int, int], ...]] = {}
    root_node_id: int | None = None

    for row in rows:
        metadata = rose_tree_metadata[row]
        trajectory_id = int(metadata["trajectory_id"])
        if trajectory_id in row_by_trajectory:
            raise ValueError(f"duplicate trajectory_id {trajectory_id} in tree {tree_id}")
        row_by_trajectory[trajectory_id] = row

        parent_value = metadata.get("parent_trajectory_id")
        parent_id = None if parent_value is None else int(parent_value)
        parent_by_trajectory[trajectory_id] = parent_id
        root_restart = bool(metadata.get("root_restart", False))
        branch_pos = int(metadata.get("branch_pos", -1))
        if parent_id is None:
            if not root_restart or branch_pos != 0:
                raise ValueError(
                    f"root trajectory {trajectory_id} in tree {tree_id} must set root_restart=true and branch_pos=0"
                )
        elif root_restart:
            raise ValueError(
                f"trajectory {trajectory_id} in tree {tree_id} cannot set root_restart=true with parent {parent_id}"
            )

        node_ids, path_ends = parsed_paths[row]
        if root_node_id is None:
            root_node_id = node_ids[0]
        elif node_ids[0] != root_node_id:
            raise ValueError(f"trajectories in tree {tree_id} do not share one root node")

        terminal_node_id = node_ids[-1]
        if terminal_node_id in terminal_owners:
            raise ValueError(
                f"terminal node {terminal_node_id} is shared by trajectories "
                f"{terminal_owners[terminal_node_id]} and {trajectory_id} in tree {tree_id}"
            )
        terminal_owners[terminal_node_id] = trajectory_id

        for index, node_id in enumerate(node_ids):
            signature = tuple(zip(node_ids[: index + 1], path_ends[: index + 1], strict=True))
            known_signature = node_signatures.setdefault(node_id, signature)
            if known_signature != signature:
                raise ValueError(f"node {node_id} has inconsistent ancestry in tree {tree_id}")

    for trajectory_id, parent_id in parent_by_trajectory.items():
        if parent_id is not None and parent_id not in row_by_trajectory:
            raise ValueError(f"trajectory {trajectory_id} in tree {tree_id} references unknown parent {parent_id}")

    visit_state: dict[int, int] = {}

    def visit(trajectory_id: int) -> None:
        state = visit_state.get(trajectory_id, 0)
        if state == 1:
            raise ValueError(f"parent graph contains a cycle at trajectory {trajectory_id} in tree {tree_id}")
        if state == 2:
            return
        visit_state[trajectory_id] = 1
        parent_id = parent_by_trajectory[trajectory_id]
        if parent_id is not None:
            visit(parent_id)
        visit_state[trajectory_id] = 2

    for trajectory_id in row_by_trajectory:
        visit(trajectory_id)

    for trajectory_id, parent_id in parent_by_trajectory.items():
        if parent_id is None:
            continue
        row = row_by_trajectory[trajectory_id]
        parent_row = row_by_trajectory[parent_id]
        branch_pos = int(rose_tree_metadata[row]["branch_pos"])
        node_ids, path_ends = parsed_paths[row]
        parent_node_ids, parent_path_ends = parsed_paths[parent_row]
        if branch_pos not in path_ends or branch_pos not in parent_path_ends:
            raise ValueError(
                f"trajectory {trajectory_id} branch_pos={branch_pos} is not a finalized node in tree {tree_id}"
            )
        child_index = path_ends.index(branch_pos)
        parent_index = parent_path_ends.index(branch_pos)
        if node_ids[: child_index + 1] != parent_node_ids[: parent_index + 1]:
            raise ValueError(
                f"trajectory {trajectory_id} does not share its parent path through branch_pos={branch_pos} "
                f"in tree {tree_id}"
            )


@register_adv_est("rose_tree")
def compute_rose_tree_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    rose_tree_metadata: Sequence[Mapping[str, Any]],
    config: Any = None,
    **_: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ROSE segment advantages from finalized rollout trees."""
    if token_level_rewards.shape != response_mask.shape:
        raise ValueError(
            f"token_level_rewards shape {token_level_rewards.shape} does not match response_mask {response_mask.shape}"
        )
    if len(rose_tree_metadata) != token_level_rewards.shape[0] or len(index) != token_level_rewards.shape[0]:
        raise ValueError("ROSE metadata, uid index, and reward batch must have equal lengths")

    advantages = torch.zeros_like(token_level_rewards, dtype=torch.float32)
    response_lengths = response_mask.sum(dim=-1).to(torch.int64)
    leaf_rewards = token_level_rewards.sum(dim=-1).to(torch.float32)
    groups: dict[str, list[int]] = defaultdict(list)
    parsed_paths: dict[int, tuple[list[int], list[int]]] = {}

    require_binary_reward = bool(_config_value(config, "require_binary_reward", True))
    binary_tolerance = float(_config_value(config, "binary_reward_tolerance", 1e-6))

    for row in range(token_level_rewards.shape[0]):
        response_length = int(response_lengths[row].item())
        if response_length == 0:
            continue
        metadata = rose_tree_metadata[row]
        if not isinstance(metadata, Mapping):
            raise TypeError(f"ROSE tree metadata at row {row} must be a mapping, got {type(metadata)!r}")
        tree_id = str(metadata["tree_id"])
        if tree_id != str(index[row]):
            raise ValueError(f"ROSE tree_id {tree_id!r} does not match uid {index[row]!r} at row {row}")
        reward = float(leaf_rewards[row].item())
        if require_binary_reward and min(abs(reward), abs(reward - 1.0)) > binary_tolerance:
            raise ValueError(f"ROSE requires binary rewards, got {reward} at row {row}")
        parsed_paths[row] = _validate_metadata(metadata, response_length)
        groups[tree_id].append(row)

    alpha = float(_config_value(config, "length_calibration_alpha", 1.0))
    if alpha < 0:
        raise ValueError(f"length_calibration_alpha must be non-negative, got {alpha}")

    with torch.no_grad():
        for tree_id, rows in groups.items():
            node_members: dict[int, list[int]] = defaultdict(list)
            node_positions: dict[int, int] = {}
            for row in rows:
                node_ids, path_ends = parsed_paths[row]
                for node_id, position in zip(node_ids, path_ends, strict=True):
                    known_position = node_positions.setdefault(node_id, position)
                    if known_position != position:
                        raise ValueError(
                            f"node {node_id} has inconsistent positions {known_position} and {position} "
                            f"in tree {tree_id}"
                        )
                    node_members[node_id].append(row)

            _validate_tree_relationships(tree_id, rows, rose_tree_metadata, parsed_paths)

            node_values = {
                node_id: torch.stack([leaf_rewards[row] for row in member_rows]).mean()
                for node_id, member_rows in node_members.items()
            }
            for row in rows:
                node_ids, path_ends = parsed_paths[row]
                for node_index in range(1, len(node_ids)):
                    start = path_ends[node_index - 1]
                    stop = path_ends[node_index]
                    segment_advantage = node_values[node_ids[node_index]] - node_values[node_ids[node_index - 1]]
                    advantages[row, start:stop] = segment_advantage

            correct_rows = [row for row in rows if abs(float(leaf_rewards[row].item()) - 1.0) <= binary_tolerance]
            if len(correct_rows) < 2 or alpha == 0:
                continue
            shortest_row = min(
                correct_rows,
                key=lambda row: (int(response_lengths[row].item()), int(rose_tree_metadata[row]["trajectory_id"])),
            )
            shortest_length = int(response_lengths[shortest_row].item())
            shortest_node_ids, shortest_path_ends = parsed_paths[shortest_row]

            for row in correct_rows:
                if row == shortest_row:
                    continue
                node_ids, _ = parsed_paths[row]
                pivot_position = _last_common_node_position(shortest_node_ids, shortest_path_ends, node_ids)
                response_length = int(response_lengths[row].item())
                denominator = response_length - pivot_position
                numerator = shortest_length - pivot_position
                if denominator <= 0 or numerator < 0:
                    raise ValueError(
                        f"invalid length calibration lengths in tree {tree_id}: "
                        f"shortest={shortest_length}, current={response_length}, pivot={pivot_position}"
                    )
                ratio = min(max(numerator / denominator, 0.0), 1.0)
                penalty_scale = 1.0 - ratio**alpha
                suffix = advantages[row, pivot_position:response_length]
                advantages[row, pivot_position:response_length] = suffix - suffix.abs() * penalty_scale

    advantages *= response_mask.to(dtype=advantages.dtype)
    return advantages, advantages.clone()
