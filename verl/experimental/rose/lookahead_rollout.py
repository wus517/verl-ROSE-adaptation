# Copyright 2025 Individual Contributor: ROSE-LATR adaptation contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sampled lookahead orchestration. Only complete retained trajectories leave this module."""

import asyncio
import json
import logging
import os
import socket
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput
from verl.experimental.rose.lookahead import ProbeBudget, evaluate_probe, rank_legal_pivots, segment_seed

logger = logging.getLogger(__name__)


@dataclass
class LookaheadTrajectory:
    """A complete retained leaf, compatible with ROSE finalization and version checks."""

    output: AgentLoopOutput
    semantic_entropy: np.ndarray
    parent_trajectory_id: int | None
    branch_pos: int
    root_restart: bool


class LookaheadRollout:
    """One prompt group; all counters, reservations and trace buffers are group-local."""

    def __init__(self, *, agent_loop, scorer, config, response_length, uid, step, seed, sampling_params, priority=0):
        self.agent_loop = agent_loop
        self.scorer = scorer
        self.config = config
        self.response_length = response_length
        self.uid, self.step = uid, step
        self.expected_version = step - 1
        self.seed = int(seed if sampling_params.get("seed") is None else sampling_params["seed"])
        self.sampling_params = dict(sampling_params)
        self.priority = priority
        self.budget = ProbeBudget(config.max_probe_tokens_per_group)
        self.stats = Counter(
            dict.fromkeys(
                (
                    "probe_attempts",
                    "probe_full_window_accepted",
                    "probe_rejected",
                    "short_terminal_kept",
                    "short_terminal_duplicate_rejected",
                    "epsilon_root_slots",
                    "accepted_branch_slots",
                    "fallback_root_slots",
                    "fallback_no_legal_pivot",
                    "fallback_probe_budget_exhausted",
                    "fallback_pivot_trials_exhausted",
                    "probe_generated_tokens",
                    "rejected_probe_tokens",
                    "accepted_probe_tokens",
                    "root_generated_tokens",
                    "completion_generated_tokens",
                    "probe_rpc_seconds",
                    "completion_rpc_seconds",
                    "root_rpc_seconds",
                    "semantic_score_seconds",
                    "cached_tokens_known_sum",
                    "cached_tokens_observations",
                    "failed_requests_unknown_tokens",
                    "trace_events_dropped",
                ),
                0,
            )
        )
        self.events = []
        self.started = time.perf_counter()

    def event(self, phase, **fields):
        """Publish bounded progress metadata without full prompt/response text."""
        row = {"uid": self.uid, "step": self.step, "phase": phase, **fields}
        # Progress remains observable before a long RPC completes; no prompt/response text.
        logger.warning("[ROSE_LOOKAHEAD_PROGRESS] %s", json.dumps(row, ensure_ascii=False))
        if self.config.trace_dir:
            if len(self.events) < self.config.trace_max_events_per_group:
                self.events.append(row)
            else:
                self.stats["trace_events_dropped"] += 1

    def summary(self):
        """Snapshot scalar costs once before duplicating them onto retained leaves."""
        return {"tree_id": self.uid, **dict(self.stats), "group_wall_seconds": time.perf_counter() - self.started}

    def finish(self, status, failure_reason=None):
        """Emit group-wide workload evidence, even for groups discarded by the trainer."""
        summary = {**self.summary(), "step": self.step, "status": status, "failure_reason": failure_reason}
        logger.warning("[ROSE_LOOKAHEAD_GROUP] %s", json.dumps(summary, ensure_ascii=False))
        if self.config.trace_dir:
            try:
                directory = Path(self.config.trace_dir)
                directory.mkdir(parents=True, exist_ok=True)
                # A Ray worker is process-local; hostname+pid separates nodes and workers.
                path = directory / f"{socket.gethostname()}-{os.getpid()}.jsonl"
                with path.open("a", encoding="utf-8") as stream:
                    for row in [*self.events, {"phase": "group_summary", **summary}]:
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            except OSError:
                logger.exception("ROSE lookahead optional trace write failed")

    async def request(self, prompt_ids, prefix, slot, pivot, trial, phase):
        """Execute one bounded RPC and validate its policy version before accepting it."""
        seed = segment_seed(self.seed, slot, pivot, trial, phase)
        params = {**self.sampling_params, "seed": seed}
        limit = max(self.config.windows) if phase == "probe" else self.response_length - len(prefix)
        self.event(f"{phase}_start", slot=slot, pivot=pivot, trial=trial, prefix_len=len(prefix), seed=seed)
        started = time.perf_counter()
        try:
            segment = await asyncio.wait_for(
                self.agent_loop.generate_segment(
                    request_id=f"rose-la-{self.uid}-{slot}-{pivot}-{trial}-{phase}",
                    prompt_ids=prompt_ids,
                    prefix_ids=prefix,
                    sampling_params=params,
                    max_new_tokens=limit,
                    priority=self.priority,
                ),
                timeout=self.config.request_timeout_seconds,
            )
        except BaseException:
            self.stats[f"{phase}_rpc_seconds"] += time.perf_counter() - started
            self.stats["failed_requests_unknown_tokens"] += 1
            logger.error(
                "ROSE lookahead request failed; generated token cost unknown; cancellation does not "
                "guarantee backend cancellation uid=%s slot=%s phase=%s",
                self.uid,
                slot,
                phase,
            )
            raise
        self.stats[f"{phase}_rpc_seconds"] += segment.seconds
        self.stats[f"{phase}_generated_tokens"] += len(segment.token_ids)
        if phase == "probe":
            self.budget.settle(len(segment.token_ids))
        cached = segment.extra_fields.get("num_cached_tokens")
        if cached is not None:
            self.stats["cached_tokens_known_sum"] += cached
            self.stats["cached_tokens_observations"] += 1
        extra = segment.extra_fields
        version = extra.get("global_steps")
        if version is None or any(
            int(v) != self.expected_version
            for v in (version, extra.get("min_global_steps", version), extra.get("max_global_steps", version))
        ):
            raise RuntimeError("ROSE lookahead segment policy version mismatch")
        terminal = segment.terminal(len(prefix) + len(segment.token_ids), self.response_length)
        self.event(
            f"{phase}_finished",
            slot=slot,
            pivot=pivot,
            trial=trial,
            tokens=len(segment.token_ids),
            finish_reason=extra["rollout_finish_reason"],
            terminal=terminal,
            seconds=segment.seconds,
        )
        return segment, terminal

    def score(self, segment):
        """Score retained new tokens once and release their packed candidate arrays."""
        started = time.perf_counter()
        entropy = self.scorer.score(segment.topk_ids, segment.topk_logprobs, segment.topk_valid_mask)
        entropy[np.asarray(segment.topk_valid_mask).sum(axis=1) < 2] = -np.inf
        elapsed = time.perf_counter() - started
        self.stats["semantic_score_seconds"] += elapsed
        payload = segment.topk_ids.nbytes + segment.topk_logprobs.nbytes + segment.topk_valid_mask.nbytes
        # Only the retained entropy vector survives; no full-vocabulary or long-lived top-k cache.
        segment.topk_ids = segment.topk_logprobs = segment.topk_valid_mask = np.empty((0, 0))
        return entropy, elapsed, payload

    def assemble(self, prompt_ids, parent, parent_id, b, segments, scores, leaf):
        """Join inherited prefix, accepted probe and completion without a new tree pivot."""
        prefix_ids = parent.output.response_ids[:b] if parent is not None else []
        prefix_probs = parent.output.response_logprobs[:b] if parent is not None else []
        prefix_entropy = parent.semantic_entropy[:b] if parent is not None else np.empty(0, np.float32)
        ids = prefix_ids + [t for segment in segments for t in segment.token_ids]
        probs = prefix_probs + [p for segment in segments for p in segment.log_probs]
        entropy = np.concatenate([prefix_entropy, *[s[0] for s in scores]])
        if not len(ids) == len(probs) == len(entropy) or len(ids) > self.response_length:
            raise RuntimeError("ROSE lookahead assembled response violates token alignment/budget")
        finite = entropy[np.isfinite(entropy)]
        extra = dict(segments[-1].extra_fields)
        cached = [segment.extra_fields.get("num_cached_tokens") for segment in segments]
        extra["num_cached_tokens"] = sum(cached) if all(value is not None for value in cached) else None
        extra.update(
            {
                "turn_scores": [],
                "tool_rewards": [],
                "rose_lookahead_leaf": {"parent": parent_id, "branch_pos": b, **leaf},
                "rose_semantic_score_seconds": sum(s[1] for s in scores),
                "rose_generation_seconds": sum(s.seconds for s in segments),
                "rose_topk_payload_bytes": sum(s[2] for s in scores),
                "rose_prefix_reuse_tokens": b,
                "rose_new_generated_tokens": len(ids) - b,
                "rose_root_restart": parent is None,
                "rose_branch_pos": b,
                "rose_semantic_entropy_max": float(finite.max()) if finite.size else None,
                "global_steps": self.expected_version,
                "min_global_steps": self.expected_version,
                "max_global_steps": self.expected_version,
            }
        )
        preempted = [s.num_preempted for s in segments if s.num_preempted is not None and s.num_preempted >= 0]
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=ids,
            response_mask=[1] * len(ids),
            response_logprobs=probs,
            num_turns=2,
            extra_fields=extra,
            metrics=AgentLoopMetrics(
                generate_sequences=sum(s.seconds for s in segments),
                num_preempted=sum(preempted) if len(preempted) == len(segments) else -1,
            ),
        )
        return LookaheadTrajectory(output, entropy, parent_id, b, parent is None)

    async def root(self, prompt_ids, slot, source):
        """Generate one independent final response, including budget/trial fallbacks."""
        segment, terminal = await self.request(prompt_ids, [], slot, -1, 0, "root")
        if not terminal:
            raise RuntimeError("ROSE complete root request unexpectedly requires continuation")
        return self.assemble(
            prompt_ids,
            None,
            None,
            0,
            [segment],
            [self.score(segment)],
            {"source": source, "probe_tokens": 0, "distances": {}},
        )

    async def generate_slot(self, prompt_ids, trajectories, slot, rng, epsilon):
        """Fill exactly one leaf slot through bounded pruning or one root fallback."""
        if not trajectories:
            return await self.root(prompt_ids, slot, "initial_root")
        if rng.random() < epsilon:
            self.stats["epsilon_root_slots"] += 1
            return await self.root(prompt_ids, slot, "epsilon_root")
        candidates = rank_legal_pivots(trajectories, self.response_length, max(self.config.windows))
        fallback = "no_legal_pivot" if not candidates else "pivot_trials_exhausted"
        for pivot_index, candidate in enumerate(candidates[: self.config.max_pivots_per_slot]):
            parent_id, b = candidate.parent_id, candidate.position
            parent = trajectories[parent_id]
            prefix = parent.output.response_ids[:b]
            for trial in range(self.config.max_trials_per_pivot):
                if not self.budget.reserve(max(self.config.windows)):
                    fallback = "probe_budget_exhausted"
                    break
                self.stats["probe_attempts"] += 1
                probe, terminal = await self.request(prompt_ids, prefix, slot, pivot_index, trial, "probe")
                short = len(probe.token_ids) < max(self.config.windows)
                distances = {}
                if short:
                    if probe.extra_fields["rollout_finish_reason"] != "stop":
                        raise RuntimeError("ROSE probe context exhausted before full lookahead window")
                    duplicate = any(t.output.response_ids == prefix + probe.token_ids for t in trajectories)
                    accepted = not duplicate
                    if duplicate:
                        self.stats["short_terminal_duplicate_rejected"] += 1
                else:
                    accepted, distances = evaluate_probe(
                        parent.output.response_ids[b:],
                        probe.token_ids,
                        self.config.windows,
                        self.config.edit_distance_threshold,
                    )
                event_fields = dict(
                    slot=slot,
                    trial=trial,
                    parent=parent_id,
                    branch_pos=b,
                    prefix_digest=candidate.prefix_key,
                    distances=distances,
                )
                if self.config.trace_include_token_ids:
                    event_fields.update(
                        probe_ids=probe.token_ids,
                        parent_ids=parent.output.response_ids[b : b + max(self.config.windows)],
                    )
                self.event("accept" if accepted else "reject", **event_fields)
                if not accepted:
                    self.stats["probe_rejected"] += 1
                    self.stats["rejected_probe_tokens"] += len(probe.token_ids)
                    del probe
                    continue
                self.stats["short_terminal_kept" if short else "probe_full_window_accepted"] += 1
                self.stats["accepted_probe_tokens"] += len(probe.token_ids)
                segments, scores = [probe], [self.score(probe)]
                if not terminal:
                    completion, terminal = await self.request(
                        prompt_ids,
                        prefix + probe.token_ids,
                        slot,
                        pivot_index,
                        trial,
                        "completion",
                    )
                    if not terminal:
                        raise RuntimeError("ROSE complete suffix request unexpectedly requires continuation")
                    segments.append(completion)
                    scores.append(self.score(completion))
                self.stats["accepted_branch_slots"] += 1
                return self.assemble(
                    prompt_ids,
                    parent,
                    parent_id,
                    b,
                    segments,
                    scores,
                    {
                        "source": "short_terminal" if short else "lookahead",
                        "probe_tokens": len(probe.token_ids),
                        "distances": distances,
                    },
                )
            if fallback == "probe_budget_exhausted":
                break
        self.stats["fallback_root_slots"] += 1
        self.stats[f"fallback_{fallback}"] += 1
        self.event("root_fallback", slot=slot, reason=fallback)
        return await self.root(prompt_ids, slot, fallback)
