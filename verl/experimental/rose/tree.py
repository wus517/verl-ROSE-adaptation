from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BranchRecord:
    trajectory_id: int
    parent_trajectory_id: int | None
    branch_pos: int
    root_restart: bool
    response_length: int


def _find_adjacent_edge(path: list[tuple[int, int]], position: int) -> tuple[int, int] | None:
    for index in range(1, len(path)):
        if path[index - 1][1] < position < path[index][1]:
            return path[index - 1][0], path[index][0]
    return None


def _insert_pivot_on_edge(
    paths: dict[int, list[tuple[int, int]]],
    edge: tuple[int, int],
    pivot: tuple[int, int],
) -> None:
    parent_node_id, child_node_id = edge
    for trajectory_id, path in paths.items():
        for index in range(1, len(path)):
            if path[index - 1][0] == parent_node_id and path[index][0] == child_node_id:
                paths[trajectory_id] = [*path[:index], pivot, *path[index:]]
                break


def finalize_tree(tree_id: str, records: list[BranchRecord]) -> dict[int, dict[str, Any]]:
    """Build stable root/pivot/terminal paths from ordered branch records."""
    if not records:
        raise ValueError("ROSE tree requires at least one trajectory")

    paths: dict[int, list[tuple[int, int]]] = {}
    metadata: dict[int, dict[str, Any]] = {}
    next_node_id = 1

    for record in records:
        trajectory_id = int(record.trajectory_id)
        response_length = int(record.response_length)
        branch_pos = int(record.branch_pos)
        if trajectory_id in paths:
            raise ValueError(f"duplicate trajectory id {trajectory_id}")
        if response_length < 1:
            raise ValueError(f"trajectory {trajectory_id} has empty response")

        is_root_rollout = record.parent_trajectory_id is None
        if record.root_restart != is_root_rollout:
            raise ValueError(
                f"trajectory {trajectory_id} root_restart={record.root_restart} is inconsistent with "
                f"parent {record.parent_trajectory_id}"
            )

        if is_root_rollout:
            if branch_pos != 0:
                raise ValueError(f"root trajectory {trajectory_id} must have branch_pos=0, got {branch_pos}")
            shared_path = [(0, 0)]
        else:
            parent_id = int(record.parent_trajectory_id)
            if parent_id not in paths:
                raise ValueError(f"trajectory {trajectory_id} references unknown parent {parent_id}")
            parent_path = paths[parent_id]
            parent_length = parent_path[-1][1]
            if not 0 <= branch_pos < parent_length:
                raise ValueError(f"trajectory {trajectory_id} branch_pos={branch_pos} must be in [0, {parent_length})")
            if response_length <= branch_pos:
                raise ValueError(
                    f"trajectory {trajectory_id} response_length={response_length} must exceed branch_pos={branch_pos}"
                )

            pivot_index = next(
                (index for index, (_, position) in enumerate(parent_path) if position == branch_pos),
                None,
            )
            if pivot_index is None:
                edge = _find_adjacent_edge(parent_path, branch_pos)
                if edge is None:
                    raise ValueError(
                        f"trajectory {trajectory_id} branch_pos={branch_pos} does not lie on parent path {parent_id}"
                    )
                pivot_node = (next_node_id, branch_pos)
                next_node_id += 1
                _insert_pivot_on_edge(paths, edge, pivot_node)
                parent_path = paths[parent_id]
                pivot_index = next(index for index, (node_id, _) in enumerate(parent_path) if node_id == pivot_node[0])
            shared_path = parent_path[: pivot_index + 1]

        terminal_node = (next_node_id, response_length)
        next_node_id += 1
        paths[trajectory_id] = [*shared_path, terminal_node]
        metadata[trajectory_id] = {
            "schema_version": 1,
            "tree_id": str(tree_id),
            "trajectory_id": trajectory_id,
            "parent_trajectory_id": record.parent_trajectory_id,
            "branch_pos": branch_pos,
            "root_restart": bool(record.root_restart),
        }

    for trajectory_id, path in paths.items():
        metadata[trajectory_id]["path_node_ids"] = [node_id for node_id, _ in path]
        metadata[trajectory_id]["path_ends"] = [position for _, position in path]

    return metadata
