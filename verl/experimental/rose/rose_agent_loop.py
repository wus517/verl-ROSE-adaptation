import time
from typing import Any

import numpy as np

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput
from verl.workers.rollout.logprobs import (
    ROSE_TOPK_LOGPROBS_FIELD,
    ROSE_TOPK_TOKEN_IDS_FIELD,
    ROSE_TOPK_VALID_MASK_FIELD,
)


class RoseAgentLoop(AgentLoopBase):
    """Single-turn token-in/token-out helper used by the ROSE tree worker."""

    async def build_prompt(self, raw_prompt: list[dict[str, Any]]) -> list[int]:
        messages = list(raw_prompt)
        multi_modal_data = await self.process_multi_modal_info(messages)
        if multi_modal_data:
            raise ValueError("ROSE currently supports text-only prompts")
        return await self.ct_build_initial_tokens(messages)

    async def generate_from_prefix(
        self,
        *,
        request_id: str,
        prompt_ids: list[int],
        prefix_ids: list[int],
        prefix_logprobs: list[float],
        sampling_params: dict[str, Any],
        priority: int = 0,
    ) -> tuple[AgentLoopOutput, np.ndarray, np.ndarray, np.ndarray]:
        remaining_tokens = self.rollout_config.response_length - len(prefix_ids)
        if remaining_tokens < 1:
            raise ValueError("ROSE branch prefix leaves no response token budget")

        request_sampling_params = dict(sampling_params)
        request_sampling_params["max_tokens"] = remaining_tokens
        started = time.perf_counter()
        token_output = await self.server_manager.generate(
            request_id=request_id,
            prompt_ids=[*prompt_ids, *prefix_ids],
            sampling_params=request_sampling_params,
            priority=int(priority),
        )
        generate_seconds = time.perf_counter() - started

        continuation_ids = list(token_output.token_ids[:remaining_tokens])
        continuation_logprobs = list((token_output.log_probs or [])[: len(continuation_ids)])
        if not continuation_ids:
            raise RuntimeError("ROSE rollout returned an empty continuation")
        if len(continuation_logprobs) != len(continuation_ids):
            raise RuntimeError(
                f"ROSE rollout returned {len(continuation_logprobs)} sampled logprobs for "
                f"{len(continuation_ids)} tokens"
            )

        try:
            topk_token_ids = np.asarray(token_output.extra_fields.pop(ROSE_TOPK_TOKEN_IDS_FIELD))
            topk_logprobs = np.asarray(token_output.extra_fields.pop(ROSE_TOPK_LOGPROBS_FIELD))
            topk_valid_mask = np.asarray(token_output.extra_fields.pop(ROSE_TOPK_VALID_MASK_FIELD))
        except KeyError as exc:
            raise RuntimeError("ROSE rollout requires packed top-k logprobs from the vLLM server") from exc
        expected_length = len(continuation_ids)
        if topk_token_ids.shape[0] != expected_length:
            raise RuntimeError(
                f"ROSE top-k length {topk_token_ids.shape[0]} does not match continuation {expected_length}"
            )

        response_ids = [*prefix_ids, *continuation_ids]
        response_logprobs = [*prefix_logprobs, *continuation_logprobs]
        metrics = AgentLoopMetrics(
            generate_sequences=generate_seconds,
            num_preempted=token_output.num_preempted if token_output.num_preempted is not None else -1,
        )
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=[1] * len(response_ids),
            response_logprobs=response_logprobs,
            num_turns=2,
            metrics=metrics,
            extra_fields=token_output.extra_fields,
        )
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        return output, topk_token_ids, topk_logprobs, topk_valid_mask

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        prompt_ids = await self.build_prompt(kwargs["raw_prompt"])
        output, _, _, _ = await self.generate_from_prefix(
            request_id=str(kwargs["uid"]),
            prompt_ids=prompt_ids,
            prefix_ids=[],
            prefix_logprobs=[],
            sampling_params=sampling_params,
            priority=int(kwargs.get("priority", 0)),
        )
        return output
