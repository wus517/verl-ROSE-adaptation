import asyncio
import hashlib
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import ray
import transfer_queue as tq

from verl.experimental.agent_loop.agent_loop import DictConfigWrap
from verl.experimental.rose import tree_advantage as _tree_advantage  # noqa: F401
from verl.experimental.rose.rose_agent_loop import RoseAgentLoop
from verl.experimental.rose.semantic_entropy import SemanticEntropyScorer
from verl.experimental.rose.tree import BranchRecord, finalize_tree
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ, AgentLoopWorkerTQBase, apply_greedy_sampling_params
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@dataclass
class _Trajectory:
    output: Any
    semantic_entropy: np.ndarray
    parent_trajectory_id: int | None
    branch_pos: int
    root_restart: bool


def _seed_for_prompt(base_seed: int, uid: str, global_steps: int) -> int:
    payload = f"{base_seed}:{global_steps}:{uid}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big", signed=False)


def _select_branch(trajectories: list[_Trajectory]) -> tuple[int, int] | None:
    best_score = -math.inf
    best_trajectory = -1
    best_position = -1
    for trajectory_id, trajectory in enumerate(trajectories):
        for position, score in enumerate(trajectory.semantic_entropy):
            score = float(score)
            if math.isfinite(score) and score > best_score:
                best_score = score
                best_trajectory = trajectory_id
                best_position = position
    if best_trajectory < 0:
        return None
    return best_trajectory, best_position


def _choose_branch(
    trajectories: list[_Trajectory],
    rng: random.Random,
    epsilon: float,
) -> tuple[int, int] | None:
    if not trajectories or rng.random() < epsilon:
        return None
    return _select_branch(trajectories)


def _validate_policy_versions(trajectories: list[_Trajectory], expected_version: int) -> None:
    tree_versions: set[int] = set()
    for trajectory_id, trajectory in enumerate(trajectories):
        extra_fields = trajectory.output.extra_fields
        global_version = extra_fields.get("global_steps")
        min_version = extra_fields.get("min_global_steps", global_version)
        max_version = extra_fields.get("max_global_steps", global_version)
        if min_version is None or max_version is None:
            raise RuntimeError(f"ROSE trajectory {trajectory_id} is missing rollout policy version metadata")
        min_version = int(min_version)
        max_version = int(max_version)
        if min_version != max_version:
            raise RuntimeError(f"ROSE trajectory {trajectory_id} spans policy versions {min_version}..{max_version}")
        if global_version is not None and int(global_version) != max_version:
            raise RuntimeError(
                f"ROSE trajectory {trajectory_id} has inconsistent global_steps={global_version} "
                f"and max_global_steps={max_version}"
            )
        tree_versions.add(max_version)

    if tree_versions != {int(expected_version)}:
        raise RuntimeError(f"ROSE tree expected policy version {expected_version}, got {sorted(tree_versions)}")


class RoseAgentLoopWorkerTQBase(AgentLoopWorkerTQBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rose_config = self.rollout_config.rose
        if not rose_config.enable:
            raise ValueError("RoseAgentLoopManagerTQ requires actor_rollout_ref.rollout.rose.enable=true")
        if self.rollout_config.name != "vllm":
            raise ValueError("ROSE currently supports only the vLLM rollout backend")
        if self.rollout_config.multi_turn.enable:
            raise ValueError("ROSE currently supports only single-turn rollout")
        if isinstance(self.llm_client, FullyAsyncLLMServerClient):
            raise ValueError("ROSE requires the synchronous V1 trainer; fully async rollout is not supported")
        if rose_config.semantic_device != "cpu":
            raise ValueError("ROSE currently supports semantic_device=cpu; NPU scorer is not implemented")
        if not 0.0 <= rose_config.epsilon <= 1.0:
            raise ValueError(f"ROSE epsilon must be in [0, 1], got {rose_config.epsilon}")
        if rose_config.top_k < 2:
            raise ValueError(f"ROSE top_k must be at least 2, got {rose_config.top_k}")
        if not rose_config.embedding_path:
            raise ValueError("ROSE embedding_path must point to a prepared normalized embedding table")

        self.rose_config = rose_config
        self.semantic_scorer = SemanticEntropyScorer(
            rose_config.embedding_path,
            metadata_path=rose_config.embedding_meta_path,
            chunk_size=rose_config.semantic_chunk_size,
            exclude_diagonal=rose_config.exclude_semantic_diagonal,
        )

    def _make_agent_loop(self) -> RoseAgentLoop:
        return RoseAgentLoop(
            trainer_config=DictConfigWrap(self.config),
            server_manager=self.llm_client,
            tokenizer=self.tokenizer,
            processor=self.processor,
            hf_model_type=self.hf_model_type,
            dataset_cls=self.dataset_cls,
            data_config=DictConfigWrap(self.config.data),
        )

    async def _run_prompt(self, prompt: dict, sampling_params: dict, trajectory: dict, trace: bool = False) -> None:
        if trajectory["validate"]:
            if self.rose_config.validation_mode != "independent":
                raise ValueError(f"unsupported ROSE validation_mode={self.rose_config.validation_mode!r}")
            await super()._run_prompt(prompt, sampling_params, trajectory, trace)
            return

        uid = str(prompt["uid"])
        partition_id = "train"
        await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "running"})
        postprocess_tasks: list[asyncio.Task] = []
        try:
            agent_name = prompt.pop("agent_name", "single_turn_agent")
            if agent_name != "single_turn_agent":
                raise ValueError(f"ROSE only supports single_turn_agent, got {agent_name!r}")
            num_trajectories = prompt.pop(
                "__rollout_n__",
                self.rose_config.num_trajectories or self.rollout_config.n,
            )
            num_trajectories = int(num_trajectories)
            if num_trajectories < 1:
                raise ValueError(f"ROSE num_trajectories must be positive, got {num_trajectories}")
            if self.rose_config.num_trajectories is not None and num_trajectories != self.rollout_config.n:
                raise ValueError("ROSE num_trajectories must equal actor_rollout_ref.rollout.n")

            do_sample = bool(prompt.pop("__do_sample__", True))
            run_sampling_params = dict(sampling_params)
            if not do_sample:
                apply_greedy_sampling_params(run_sampling_params)
            run_sampling_params["logprobs"] = int(self.rose_config.top_k)

            agent_loop = self._make_agent_loop()
            prompt_ids = await agent_loop.build_prompt(prompt["raw_prompt"])
            global_steps = int(prompt.get("global_steps", trajectory["step"]))
            expected_policy_version = global_steps - 1
            rng = random.Random(_seed_for_prompt(self.rollout_config.seed, uid, global_steps))
            trajectories: list[_Trajectory] = []

            for trajectory_id in range(num_trajectories):
                selected_branch = _choose_branch(trajectories, rng, self.rose_config.epsilon)
                root_restart = selected_branch is None
                if root_restart:
                    parent_trajectory_id = None
                    branch_pos = 0
                    prefix_ids: list[int] = []
                    prefix_logprobs: list[float] = []
                    prefix_entropy = np.empty(0, dtype=np.float32)
                else:
                    parent_trajectory_id, branch_pos = selected_branch
                    parent = trajectories[parent_trajectory_id]
                    prefix_ids = parent.output.response_ids[:branch_pos]
                    prefix_logprobs = parent.output.response_logprobs[:branch_pos]
                    prefix_entropy = parent.semantic_entropy[:branch_pos]

                output, topk_ids, topk_logprobs, topk_valid_mask = await agent_loop.generate_from_prefix(
                    request_id=f"rose-{uid}-{trajectory_id}",
                    prompt_ids=prompt_ids,
                    prefix_ids=prefix_ids,
                    prefix_logprobs=prefix_logprobs,
                    sampling_params=run_sampling_params,
                    priority=int(prompt.get("priority", 0)),
                )
                semantic_score_started = time.perf_counter()
                continuation_entropy = self.semantic_scorer.score(topk_ids, topk_logprobs, topk_valid_mask)
                semantic_score_seconds = time.perf_counter() - semantic_score_started
                continuation_entropy[np.asarray(topk_valid_mask).sum(axis=1) < 2] = -np.inf
                semantic_entropy = np.concatenate([prefix_entropy, continuation_entropy])
                finite_entropy = semantic_entropy[np.isfinite(semantic_entropy)]
                output.extra_fields.update(
                    {
                        "rose_semantic_score_seconds": semantic_score_seconds,
                        "rose_generation_seconds": float(output.metrics.generate_sequences),
                        "rose_prefix_reuse_tokens": len(prefix_ids),
                        "rose_new_generated_tokens": len(output.response_ids) - len(prefix_ids),
                        "rose_root_restart": root_restart,
                        "rose_branch_pos": branch_pos,
                        "rose_semantic_entropy_max": (float(finite_entropy.max()) if finite_entropy.size else None),
                        "rose_topk_payload_bytes": topk_ids.nbytes + topk_logprobs.nbytes + topk_valid_mask.nbytes,
                    }
                )
                trajectories.append(
                    _Trajectory(
                        output=output,
                        semantic_entropy=semantic_entropy,
                        parent_trajectory_id=parent_trajectory_id,
                        branch_pos=branch_pos,
                        root_restart=root_restart,
                    )
                )

            _validate_policy_versions(trajectories, expected_policy_version)

            tree_metadata = finalize_tree(
                uid,
                [
                    BranchRecord(
                        trajectory_id=trajectory_id,
                        parent_trajectory_id=item.parent_trajectory_id,
                        branch_pos=item.branch_pos,
                        root_restart=item.root_restart,
                        response_length=len(item.output.response_ids),
                    )
                    for trajectory_id, item in enumerate(trajectories)
                ],
            )
            for trajectory_id, item in enumerate(trajectories):
                item.output.extra_fields["rose_tree_metadata"] = tree_metadata[trajectory_id]
                postprocess_kwargs = {**prompt, "uid": uid, "session_id": trajectory_id}
                postprocess_tasks.append(
                    asyncio.create_task(self._agent_loop_postprocess(item.output, False, **postprocess_kwargs))
                )

            postprocess_results = await asyncio.gather(*postprocess_tasks, return_exceptions=True)
            errors = [result for result in postprocess_results if isinstance(result, BaseException)]
            if errors:
                for error in errors:
                    logger.error(
                        "Error postprocessing ROSE tree uid=%s",
                        uid,
                        exc_info=(type(error), error, error.__traceback__),
                    )
                status = "failure"
            else:
                status = "finished"
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": status})
        except Exception:
            logger.exception("Error generating ROSE tree uid=%s", uid)
            if postprocess_tasks:
                await asyncio.gather(*postprocess_tasks, return_exceptions=True)
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "failure"})


RoseAgentLoopWorkerTQ = ray.remote(RoseAgentLoopWorkerTQBase)


class RoseAgentLoopManagerTQ(AgentLoopManagerTQ):
    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = RoseAgentLoopWorkerTQ
        super().__init__(*args, **kwargs)
