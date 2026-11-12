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
"""Real segment helper plus scripted token backend, without model inference or Ray actors."""

import asyncio
import random
from types import SimpleNamespace

import numpy as np
import pytest

from verl.experimental.rose.lookahead_rollout import LookaheadRollout
from verl.experimental.rose.rose_agent_loop import RoseAgentLoop
from verl.experimental.rose.tree import BranchRecord, finalize_tree
from verl.workers.config.rollout import RoseLookaheadConfig
from verl.workers.rollout.logprobs import (
    ROSE_TOPK_LOGPROBS_FIELD,
    ROSE_TOPK_TOKEN_IDS_FIELD,
    ROSE_TOPK_VALID_MASK_FIELD,
)


class ScriptedServer:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    async def generate(self, **kwargs):
        self.requests.append(kwargs)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        ids, reason, *version = reply
        v = version[0] if version else 3
        n = len(ids)
        return SimpleNamespace(
            token_ids=ids,
            log_probs=[-0.25] * n,
            num_preempted=0,
            stop_reason="completed",
            extra_fields={
                "global_steps": v,
                "min_global_steps": v,
                "max_global_steps": v,
                "rollout_finish_reason": reason,
                "rollout_effective_max_tokens": kwargs["sampling_params"]["max_tokens"],
                "rollout_input_tokens": len(kwargs["prompt_ids"]),
                "rollout_context_limit": 100,
                ROSE_TOPK_TOKEN_IDS_FIELD: np.ones((n, 2), np.int32),
                ROSE_TOPK_LOGPROBS_FIELD: np.full((n, 2), -0.25, np.float32),
                ROSE_TOPK_VALID_MASK_FIELD: np.ones((n, 2), bool),
            },
        )


class FakeScorer:
    def __init__(self):
        self.tokens = 0

    def score(self, ids, probs, mask):
        self.tokens += len(ids)
        # Highest pivot is b=1 when a parent has sufficient remaining tokens.
        scores = np.zeros(len(ids), np.float32)
        if len(scores) > 1:
            scores[1] = 1
        return scores


def make_search(replies, **overrides):
    server = ScriptedServer(replies)
    helper = RoseAgentLoop.__new__(RoseAgentLoop)
    helper.rollout_config = SimpleNamespace(response_length=8)
    helper.server_manager = server
    scorer = FakeScorer()
    config = RoseLookaheadConfig(windows=[2, 3], **overrides)
    search = LookaheadRollout(
        agent_loop=helper,
        scorer=scorer,
        config=config,
        response_length=8,
        uid="fixed-uid",
        step=4,
        seed=42,
        sampling_params={"logprobs": 2},
    )
    return search, server, scorer


async def slots(search, count=2, epsilon=0):
    rows = []
    rng = random.Random(42)
    for slot in range(count):
        rows.append(await search.generate_slot([99], rows, slot, rng, epsilon))
    return rows


def test_accepted_probe_reused_and_tree_keeps_original_pivot():
    search, server, scorer = make_search([([1] * 8, "length"), ([2] * 3, "length"), ([3] * 4, "length")])
    rows = asyncio.run(slots(search))
    assert rows[1].output.prompt_ids == [99]
    assert rows[1].output.response_ids == [1, 2, 2, 2, 3, 3, 3, 3]
    assert rows[1].output.response_logprobs == [-0.25] * 8
    assert len(rows[1].semantic_entropy) == 8
    assert server.requests[1]["sampling_params"]["max_tokens"] == 3
    assert server.requests[2]["prompt_ids"] == [99, 1, 2, 2, 2]
    assert server.requests[2]["sampling_params"]["max_tokens"] == 4
    assert scorer.tokens == 15 and search.budget.used == 3
    metadata = finalize_tree(
        "tree",
        [
            BranchRecord(i, r.parent_trajectory_id, r.branch_pos, r.root_restart, len(r.output.response_ids))
            for i, r in enumerate(rows)
        ],
    )
    assert metadata[1]["branch_pos"] == 1 and metadata[1]["parent_trajectory_id"] == 0
    assert 4 not in metadata[1]["path_ends"]  # The RPC split is not a tree split.


def test_rejected_probe_not_scored_then_retry_and_complete():
    search, server, scorer = make_search([([1] * 8, "length"), ([1] * 3, "length"), ([2] * 3, "length"), ([3], "stop")])
    rows = asyncio.run(slots(search))
    assert len(rows) == 2 and search.stats["probe_rejected"] == 1
    assert scorer.tokens == 12  # root + accepted probe + completion, never rejected probe
    assert search.budget.used == 6
    assert server.requests[1]["sampling_params"]["seed"] != server.requests[2]["sampling_params"]["seed"]


@pytest.mark.parametrize("budget,attempts", [(0, 0), (2, 0), (3, 1), (6, 2)])
def test_hard_budget_and_root_fallback(budget, attempts):
    replies = [([1] * 8, "length"), *[([1] * 3, "length")] * attempts, ([9], "stop")]
    search, server, scorer = make_search(replies, max_probe_tokens_per_group=budget)
    rows = asyncio.run(slots(search))
    assert search.stats["probe_attempts"] == attempts
    assert rows[1].root_restart and rows[1].parent_trajectory_id is None
    assert len(server.requests) == attempts + 2
    assert scorer.tokens == 9


@pytest.mark.parametrize("probe", [[2], [2, 2, 2]])
def test_stop_before_or_exactly_at_window_never_continues(probe):
    search, server, _ = make_search([([1] * 8, "length"), (probe, "stop")])
    rows = asyncio.run(slots(search))
    assert rows[1].output.response_ids == [1, *probe]
    assert len(server.requests) == 2
    assert search.stats["short_terminal_kept" if len(probe) < 3 else "probe_full_window_accepted"] == 1


@pytest.mark.parametrize(
    "reply,message",
    [
        (([2] * 3, "length", 2), "version mismatch"),
        (([2], "length"), "did not fill"),
        (([2], "abort"), "aborted/unknown"),
        (([2], "unknown"), "aborted/unknown"),
    ],
)
def test_protocol_failure_is_not_a_rejection_or_fallback(reply, message):
    search, server, _ = make_search([([1] * 8, "length"), reply])
    with pytest.raises(RuntimeError, match=message):
        asyncio.run(slots(search))
    assert search.stats["probe_rejected"] == search.stats["fallback_root_slots"] == 0


def test_backend_error_and_timeout_do_not_retry():
    search, server, _ = make_search([([1] * 8, "length"), RuntimeError("backend broken")])
    with pytest.raises(RuntimeError, match="backend broken"):
        asyncio.run(slots(search))
    assert len(server.requests) == 2 and search.stats["failed_requests_unknown_tokens"] == 1

    search, server, _ = make_search([], request_timeout_seconds=0.001)

    async def never(**kwargs):
        await asyncio.sleep(10)

    server.generate = never
    with pytest.raises(TimeoutError):
        asyncio.run(slots(search, count=1))


def test_epsilon_one_and_single_leaf_have_no_probe():
    search, server, _ = make_search([([1], "stop"), ([2], "stop")])
    rows = asyncio.run(slots(search, epsilon=1))
    assert all(row.root_restart for row in rows) and search.stats["probe_attempts"] == 0
    search, _, _ = make_search([([1], "stop")])
    assert len(asyncio.run(slots(search, count=1))) == 1


def test_trace_bounded_and_write_error_is_nonfatal(tmp_path):
    search, _, _ = make_search([], trace_dir=str(tmp_path), trace_max_events_per_group=1)
    search.event("first")
    search.event("second")
    search.finish("finished")
    assert len(list(tmp_path.glob("*.jsonl"))) == 1
    assert search.stats["trace_events_dropped"] == 1
    file_path = tmp_path / "not-a-directory"
    file_path.touch()
    search.config = RoseLookaheadConfig(trace_dir=str(file_path))
    search.finish("failure", "test")  # Optional I/O must not raise.


def test_trials_and_pivots_bound_even_with_large_budget():
    search, server, _ = make_search(
        [([1] * 8, "length"), *[([1] * 3, "length")] * 4, ([9], "stop")], max_pivots_per_slot=2, max_trials_per_pivot=2
    )
    rows = asyncio.run(slots(search))
    assert search.stats["probe_attempts"] == 4
    assert rows[1].output.extra_fields["rose_lookahead_leaf"]["source"] == "pivot_trials_exhausted"


def test_short_duplicate_rejected_and_no_legal_pivot_falls_back():
    # The highest legal pivot has b=5, leaving a suffix exactly one window long.
    search, server, _ = make_search(
        [([1] * 8, "length"), ([1] * 3, "stop"), ([9], "stop")], max_trials_per_pivot=1, max_pivots_per_slot=1
    )

    async def run():
        rows = await slots(search, count=1)
        rows[0].semantic_entropy[:] = -np.inf
        rows[0].semantic_entropy[5] = 1
        # With windows [2,3] this is a full-window duplicate: distance rejection.
        rows.append(await search.generate_slot([99], rows, 1, random.Random(0), 0))
        return rows

    assert asyncio.run(run())[1].root_restart
    search, server, _ = make_search([([1], "stop"), ([2], "stop")])
    assert asyncio.run(slots(search))[1].root_restart
    assert search.stats["fallback_no_legal_pivot"] == 1


@pytest.mark.parametrize("fault", ["no_reason", "bad_shape", "no_logprob", "bad_input", "version_range"])
def test_segment_contract_fails_closed(fault):
    search, server, _ = make_search([([1], "stop")])
    generate = server.generate

    async def malformed(**kwargs):
        output = await generate(**kwargs)
        if fault == "no_reason":
            del output.extra_fields["rollout_finish_reason"]
        elif fault == "bad_shape":
            output.extra_fields[ROSE_TOPK_VALID_MASK_FIELD] = np.ones((2, 2), bool)
        elif fault == "no_logprob":
            output.log_probs = []
        elif fault == "bad_input":
            output.extra_fields["rollout_input_tokens"] = 999
        else:
            output.extra_fields["min_global_steps"] = 2
        return output

    server.generate = malformed
    with pytest.raises(RuntimeError):
        asyncio.run(slots(search, count=1))


@pytest.mark.parametrize("enabled", [False, True])
def test_worker_submits_only_final_leaves_and_keeps_disabled_requests(monkeypatch, enabled):
    from verl.experimental.rose import rose_agent_loop_tq as worker_module
    from verl.workers.config.rollout import RoseRolloutConfig

    replies = (
        [([1] * 8, "length"), ([1] * 3, "length"), ([2] * 3, "length"), ([3], "stop")]
        if enabled
        else [([1] * 8, "length"), ([2] * 7, "length")]
    )
    search, server, scorer = make_search(replies)
    worker = worker_module.RoseAgentLoopWorkerTQBase.__new__(worker_module.RoseAgentLoopWorkerTQBase)
    worker.rose_config = RoseRolloutConfig(
        enable=True, embedding_path="test", epsilon=0, lookahead=RoseLookaheadConfig(enable=enabled, windows=[2, 3])
    )
    worker.rollout_config = SimpleNamespace(seed=42, n=2, response_length=8)
    worker.semantic_scorer = scorer

    async def build_prompt(raw_prompt):
        return [99]

    search.agent_loop.build_prompt = build_prompt
    worker._make_agent_loop = lambda: search.agent_loop
    submissions, statuses = [], []

    async def submit(output, validate, **kwargs):
        submissions.append((output, validate, kwargs))

    async def put(**kwargs):
        statuses.append(kwargs["tag"]["status"])

    worker._agent_loop_postprocess = submit
    monkeypatch.setattr(worker_module.tq, "async_kv_put", put)
    asyncio.run(
        worker._run_prompt(
            {"uid": "uid", "raw_prompt": [], "global_steps": 4}, {"temperature": 1}, {"validate": False, "step": 4}
        )
    )
    assert statuses == ["running", "finished"] and len(submissions) == 2
    assert [row[2]["session_id"] for row in submissions] == [0, 1]
    assert all("rose_tree_metadata" in row[0].extra_fields for row in submissions)
    if enabled:
        groups = [row[0].extra_fields["rose_lookahead_group"] for row in submissions]
        assert groups[0] == groups[1] and groups[0]["probe_rejected"] == 1
        assert len(server.requests) == 4  # Not 4 reward submissions.
    else:
        assert [r["sampling_params"]["max_tokens"] for r in server.requests] == [8, 7]
        assert all("seed" not in r["sampling_params"] for r in server.requests)
        assert all("rose_lookahead_group" not in row[0].extra_fields for row in submissions)


def test_worker_generation_failure_submits_nothing_and_validation_bypasses(monkeypatch):
    from verl.experimental.rose import rose_agent_loop_tq as worker_module
    from verl.workers.config.rollout import RoseRolloutConfig

    search, _, scorer = make_search([([1] * 8, "length"), RuntimeError("failed")])
    worker = worker_module.RoseAgentLoopWorkerTQBase.__new__(worker_module.RoseAgentLoopWorkerTQBase)
    worker.rose_config = RoseRolloutConfig(
        enable=True, embedding_path="test", epsilon=0, lookahead=RoseLookaheadConfig(enable=True, windows=[2, 3])
    )
    worker.rollout_config = SimpleNamespace(seed=42, n=2, response_length=8)
    worker.semantic_scorer = scorer

    async def build_prompt(raw_prompt):
        return [99]

    search.agent_loop.build_prompt = build_prompt
    worker._make_agent_loop = lambda: search.agent_loop
    statuses, submissions = [], []

    async def put(**kwargs):
        statuses.append(kwargs["tag"]["status"])

    async def submit(*args, **kwargs):
        submissions.append(args)

    worker._agent_loop_postprocess = submit
    monkeypatch.setattr(worker_module.tq, "async_kv_put", put)
    asyncio.run(
        worker._run_prompt({"uid": "uid", "raw_prompt": [], "global_steps": 4}, {}, {"validate": False, "step": 4})
    )
    assert statuses == ["running", "failure"] and not submissions

    async def independent(*args, **kwargs):
        submissions.append("independent")

    monkeypatch.setattr(worker_module.AgentLoopWorkerTQBase, "_run_prompt", independent)
    asyncio.run(worker._run_prompt({}, {}, {"validate": True}))
    assert submissions == ["independent"]


def test_short_terminal_duplicate_against_another_retained_leaf():
    search, server, _ = make_search(
        [([1] * 8, "length"), ([2], "stop"), ([9], "stop")], max_trials_per_pivot=1, max_pivots_per_slot=1
    )

    async def run():
        rows = await slots(search, count=1)
        # An already retained short root has the same complete response as prefix+probe.
        duplicate = SimpleNamespace(
            output=SimpleNamespace(response_ids=[1, 2]), semantic_entropy=np.array([-np.inf, -np.inf])
        )
        rows.append(duplicate)
        return await search.generate_slot([99], rows, 2, random.Random(0), 0)

    row = asyncio.run(run())
    assert row.root_restart
    assert search.stats["short_terminal_duplicate_rejected"] == 1


def test_group_budget_persists_across_slots_and_trials_reset_per_slot():
    search, server, _ = make_search(
        [([1] * 8, "length"), ([2] * 3, "length"), ([3], "stop"), ([9], "stop")], max_probe_tokens_per_group=3
    )
    rows = asyncio.run(slots(search, count=3))
    assert not rows[1].root_restart and rows[2].root_restart
    assert search.budget.used == 3 and search.stats["probe_attempts"] == 1
    assert search.stats["fallback_probe_budget_exhausted"] == 1


def test_context_limit_is_terminal_not_a_request_to_generate_past_context():
    search, server, _ = make_search([([1] * 3, "length")])
    original = server.generate

    async def clamped(**kwargs):
        output = await original(**kwargs)
        output.extra_fields["rollout_effective_max_tokens"] = 3
        output.extra_fields["rollout_context_limit"] = 4
        return output

    server.generate = clamped
    assert len(asyncio.run(slots(search, count=1))[0].output.response_ids) == 3
    assert len(server.requests) == 1


def test_default_fifty_token_window_stop_and_eight_final_leaves():
    # Default paper-sized windows, b=1, EOS exactly at probe token 50.
    search, server, scorer = make_search(
        [([1] * 80, "length"), ([2] * 50, "stop"), *[([token] * 50, "stop") for token in range(3, 9)]]
    )
    search.response_length = search.agent_loop.rollout_config.response_length = 80
    search.config = RoseLookaheadConfig(enable=True)
    rows = asyncio.run(slots(search, count=8))
    assert len(rows) == 8 and len(server.requests) == 8
    assert all(row.output.prompt_ids == [99] for row in rows)
    assert search.stats["probe_full_window_accepted"] == 7
    assert search.budget.used == 350
    assert all(request["sampling_params"]["max_tokens"] == 50 for request in server.requests[1:])
    assert len({request["request_id"] for request in server.requests}) == 8


def test_future_branch_can_select_previously_accepted_child():
    search, _, _ = make_search(
        [([1] * 8, "length"), ([2] * 3, "length"), ([3] * 4, "length"), ([4] * 3, "length"), ([5] * 3, "length")]
    )

    async def run():
        rows = await slots(search)
        rows[0].semantic_entropy[:] = -np.inf
        rows[1].semantic_entropy[:] = -np.inf
        rows[1].semantic_entropy[2] = 2
        rows.append(await search.generate_slot([99], rows, 2, random.Random(42), 0))
        return rows

    rows = asyncio.run(run())
    assert rows[2].parent_trajectory_id == 1 and rows[2].branch_pos == 2
    assert rows[2].output.response_ids[:2] == rows[1].output.response_ids[:2]


def test_generated_leaf_metadata_runs_existing_rose_advantage_estimator():
    import torch

    from verl.experimental.rose.tree_advantage import compute_rose_tree_advantage

    search, _, _ = make_search([([1] * 8, "length"), ([2] * 3, "length"), ([3] * 4, "length")])
    rows = asyncio.run(slots(search))
    metadata = finalize_tree(
        "fixed-uid",
        [
            BranchRecord(i, row.parent_trajectory_id, row.branch_pos, row.root_restart, len(row.output.response_ids))
            for i, row in enumerate(rows)
        ],
    )
    rewards = torch.zeros(2, 8)
    rewards[0, -1] = 1
    advantages, returns = compute_rose_tree_advantage(
        rewards,
        torch.ones(2, 8, dtype=torch.int64),
        np.array(["fixed-uid"] * 2, dtype=object),
        [metadata[0], metadata[1]],
        config={"rose": {"length_calibration_alpha": 0, "require_binary_reward": True}},
    )
    assert torch.isfinite(advantages).all()
    assert advantages[0, 1:].gt(0).all() and advantages[1, 1:].lt(0).all()
    torch.testing.assert_close(advantages, returns)
