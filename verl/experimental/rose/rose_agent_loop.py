import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput
from verl.workers.rollout.logprobs import (
    ROSE_TOPK_LOGPROBS_FIELD,
    ROSE_TOPK_TOKEN_IDS_FIELD,
    ROSE_TOPK_VALID_MASK_FIELD,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@dataclass
class GenerationSegment:
    """New tokens and backend metadata for one explicitly bounded generation call."""

    token_ids: list[int]
    log_probs: list[float]
    topk_ids: np.ndarray
    topk_logprobs: np.ndarray
    topk_valid_mask: np.ndarray
    extra_fields: dict[str, Any]
    seconds: float
    num_preempted: int | None

    def terminal(self, response_tokens: int, response_limit: int) -> bool:
        """Distinguish natural stop or exhausted total budget from an intermediate cap."""
        reason = self.extra_fields.get("rollout_finish_reason")
        if reason == "stop":
            return True
        if reason != "length":
            raise RuntimeError(f"ROSE segment has invalid finish reason: {reason!r}")
        effective = self.extra_fields["rollout_effective_max_tokens"]
        if len(self.token_ids) != effective:
            raise RuntimeError("ROSE length finish did not fill effective max_tokens")
        return (
            response_tokens >= response_limit
            or self.extra_fields["rollout_input_tokens"] + len(self.token_ids)
            >= self.extra_fields["rollout_context_limit"]
        )


class RoseAgentLoop(AgentLoopBase):
    """Single-turn token-in/token-out helper used by the ROSE tree worker."""

    # async def build_prompt(self, raw_prompt: list[dict[str, Any]]) -> list[int]:
    #     messages = list(raw_prompt)
    #     multi_modal_data = await self.process_multi_modal_info(messages)
    #     if multi_modal_data:
    #         raise ValueError("ROSE currently supports text-only prompts")
    #     return await self.ct_build_initial_tokens(messages)

    async def build_prompt(self, raw_prompt: list[dict[str, Any]]) -> list[int]:
        messages = list(raw_prompt)
        multi_modal_data = await self.process_multi_modal_info(messages)

        # print(
        #     "[ROSE_PROMPT_PATH]",
        #     f"file={__file__}",
        #     f"pid={os.getpid()}",
        #     f"tokenizer={type(self.tokenizer).__name__}",
        #     f"processor={type(self.processor).__name__ if self.processor is not None else None}",
        #     f"builder={type(self.continuous_token_builder).__name__}",
        #     f"multimodal_keys={list(multi_modal_data.keys())}",
        #     flush=True,
        # )

        if multi_modal_data:
            raise ValueError(
                f"ROSE currently supports text-only prompts, but got multimodal keys: {list(multi_modal_data.keys())}"
            )

        prompt_ids = await self.ct_build_initial_tokens(messages)
        # print(
        #     "[ROSE_PROMPT]",
        #     f"file={__file__}",
        #     f"pid={os.getpid()}",
        #     f"prompt_len={len(prompt_ids)}",
        #     f"prompt_ids_head={prompt_ids[:100]}",
        #     f"prompt_ids_tail={prompt_ids[-100:]}",
        #     f"decoded={self.tokenizer.decode(prompt_ids[:300], skip_special_tokens=False)!r}",
        #     flush=True,
        # )
        return prompt_ids

    async def generate_segment(
        self,
        *,
        request_id: str,
        prompt_ids: list[int],
        prefix_ids: list[int],
        sampling_params: dict[str, Any],
        max_new_tokens: int,
        priority: int = 0,
        strict: bool = True,
    ) -> GenerationSegment:
        """Generate new tokens without reclassifying response prefixes as training prompts."""
        remaining_tokens = self.rollout_config.response_length - len(prefix_ids)
        if remaining_tokens < 1 or type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("ROSE branch prefix leaves no response token budget")
        remaining_tokens = min(remaining_tokens, max_new_tokens)

        request_sampling_params = dict(sampling_params)
        request_sampling_params["max_tokens"] = remaining_tokens
        started = time.perf_counter()
        token_output = await self.server_manager.generate(
            request_id=request_id,
            prompt_ids=[*prompt_ids, *prefix_ids],
            sampling_params=request_sampling_params,
            priority=int(priority),
        )
        # raw_token_ids = list(token_output.token_ids)
        # print(
        #     "[ROSE_RAW_OUTPUT]",
        #     f"file={__file__}",
        #     f"pid={os.getpid()}",
        #     f"request_id={request_id}",
        #     f"prompt_len={len(prompt_ids)}",
        #     f"prefix_len={len(prefix_ids)}",
        #     f"max_tokens={request_sampling_params.get('max_tokens')}",
        #     f"logprobs={request_sampling_params.get('logprobs')}",
        #     f"output_len={len(raw_token_ids)}",
        #     f"stop_reason={token_output.stop_reason}",
        #     f"ids_head={raw_token_ids[:100]}",
        #     f"text_head={self.tokenizer.decode(raw_token_ids[:300], skip_special_tokens=False)!r}",
        #     f"text_tail={self.tokenizer.decode(raw_token_ids[-300:], skip_special_tokens=False)!r}",
        #     flush=True,
        # )
        generate_seconds = time.perf_counter() - started

        continuation_ids = list(token_output.token_ids)
        continuation_logprobs = list(token_output.log_probs or [])
        if len(continuation_ids) > remaining_tokens or any(type(t) is not int or t < 0 for t in continuation_ids):
            raise RuntimeError("ROSE rollout returned invalid token IDs or exceeded token budget")
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
        if (
            topk_token_ids.ndim != 2
            or topk_token_ids.shape[0] != expected_length
            or topk_logprobs.shape != topk_token_ids.shape
            or topk_valid_mask.shape != topk_token_ids.shape
        ):
            raise RuntimeError(f"ROSE top-k shape {topk_token_ids.shape} does not match continuation {expected_length}")

        if strict:
            required = (
                "rollout_finish_reason",
                "rollout_effective_max_tokens",
                "rollout_input_tokens",
                "rollout_context_limit",
            )
            if any(key not in token_output.extra_fields for key in required):
                raise RuntimeError("ROSE lookahead requires raw stopping metadata from the rollout server")
            if token_output.extra_fields["rollout_finish_reason"] not in ("stop", "length"):
                raise RuntimeError(
                    f"ROSE rollout aborted/unknown: {token_output.extra_fields['rollout_finish_reason']}"
                )
            effective = token_output.extra_fields["rollout_effective_max_tokens"]
            input_tokens = token_output.extra_fields["rollout_input_tokens"]
            context_limit = token_output.extra_fields["rollout_context_limit"]
            if (
                type(effective) is not int
                or not len(continuation_ids) <= effective <= remaining_tokens
                or input_tokens != len(prompt_ids) + len(prefix_ids)
                or type(context_limit) is not int
                or input_tokens + effective > context_limit
            ):
                raise RuntimeError("ROSE segment has inconsistent backend budget metadata")
            if not np.all(np.isfinite(continuation_logprobs)):
                raise RuntimeError("ROSE sampled logprobs must be finite")
        return GenerationSegment(
            continuation_ids,
            continuation_logprobs,
            topk_token_ids,
            topk_logprobs,
            topk_valid_mask,
            dict(token_output.extra_fields),
            generate_seconds,
            token_output.num_preempted,
        )

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
        """Compatibility wrapper for the original complete-suffix ROSE request."""
        segment = await self.generate_segment(
            request_id=request_id,
            prompt_ids=prompt_ids,
            prefix_ids=prefix_ids,
            sampling_params=sampling_params,
            max_new_tokens=self.rollout_config.response_length - len(prefix_ids),
            priority=priority,
            strict=False,
        )
        response_ids = [*prefix_ids, *segment.token_ids]
        response_logprobs = [*prefix_logprobs, *segment.log_probs]
        metrics = AgentLoopMetrics(
            generate_sequences=segment.seconds,
            num_preempted=segment.num_preempted if segment.num_preempted is not None else -1,
        )
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=[1] * len(response_ids),
            response_logprobs=response_logprobs,
            num_turns=2,
            metrics=metrics,
            extra_fields=segment.extra_fields,
        )
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        return output, segment.topk_ids, segment.topk_logprobs, segment.topk_valid_mask

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
