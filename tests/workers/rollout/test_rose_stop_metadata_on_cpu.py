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
"""Execute the production vLLM generate method with a fake engine (no vLLM import).

Only the method is compiled from its AST: backend imports / Ray decorators are not
available in CPU CI. Budget, result extraction and stop mapping remain production code.
"""

import ast
import asyncio
import contextlib
import copy
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from verl.utils import normalize_token_ids
from verl.workers.rollout.logprobs import (
    ROSE_TOPK_LOGPROBS_FIELD,
    ROSE_TOPK_TOKEN_IDS_FIELD,
    ROSE_TOPK_VALID_MASK_FIELD,
    normalize_requested_logprobs,
    pack_topk_logprobs,
)
from verl.workers.rollout.replica import TokenOutput


def production_generate():
    path = Path(__file__).resolve().parents[3] / "verl/workers/rollout/vllm_rollout/vllm_async_server.py"
    source = ast.parse(path.read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "vLLMHttpServer")
    method = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "generate"))
    method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))

    class Params(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(prompt_logprobs=None, **kwargs)

    namespace = dict(
        Any=Any,
        Optional=Optional,
        TokenOutput=TokenOutput,
        SamplingParams=Params,
        normalize_token_ids=normalize_token_ids,
        normalize_requested_logprobs=normalize_requested_logprobs,
        qwen2_5_vl_dedup_image_tokens=lambda ids, processor: ids,
        TokensPrompt=lambda **kwargs: kwargs,
        RLInsightLogger=SimpleNamespace(trace_state=lambda *args, **kwargs: contextlib.nullcontext()),
        extract_prompt_logprobs=lambda **kwargs: None,
        pack_topk_logprobs=pack_topk_logprobs,
        ROSE_TOPK_TOKEN_IDS_FIELD=ROSE_TOPK_TOKEN_IDS_FIELD,
        ROSE_TOPK_LOGPROBS_FIELD=ROSE_TOPK_LOGPROBS_FIELD,
        ROSE_TOPK_VALID_MASK_FIELD=ROSE_TOPK_VALID_MASK_FIELD,
        logger=logging.getLogger(__name__),
    )
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["generate"]


@pytest.mark.parametrize("reason,empty", [("stop", False), ("length", False), ("abort", False), ("abort", True)])
def test_real_server_preserves_raw_reason_and_legacy_top_level_mapping(reason, empty):
    class Config(SimpleNamespace):
        def get(self, name, default=None):
            return getattr(self, name, default)

    config = Config(
        max_model_len=5,
        response_length=8,
        prompt_length=4,
        full_determinism=False,
        enable_rollout_routing_replay=False,
        mtp=None,
    )

    async def engine_generate(**kwargs):
        assert kwargs["sampling_params"].max_tokens == 3  # Context clamps requested 50.
        output = SimpleNamespace(
            token_ids=[1, 2, 3],
            finish_reason=reason,
            stop_reason=3,
            logprobs=[{t: SimpleNamespace(logprob=-0.2)} for t in [1, 2, 3]],
        )
        yield SimpleNamespace(outputs=[] if empty else [output], num_cached_tokens=1)

    server = SimpleNamespace(
        config=config,
        _disaggregation_role=None,
        model_config=SimpleNamespace(processor=None),
        lora_as_adapter=False,
        _submission_paused=False,
        _admitting=0,
        replica_rank=0,
        global_steps=3,
        engine=SimpleNamespace(generate=engine_generate),
    )
    output = asyncio.run(production_generate()(server, [99, 98], {"max_tokens": 50, "logprobs": 2}, "test"))
    assert output.stop_reason == ("aborted" if reason == "abort" else "completed")
    assert output.extra_fields["rollout_finish_reason"] == reason
    assert output.extra_fields["rollout_effective_max_tokens"] == 3
    assert output.extra_fields["rollout_context_limit"] == 5
    assert output.extra_fields["rollout_input_tokens"] == 2
    assert server._admitting == 0
    if not empty:
        assert output.extra_fields["rollout_stop_detail"] == 3
        assert output.extra_fields[ROSE_TOPK_TOKEN_IDS_FIELD].shape == (3, 2)


def test_sync_client_transmits_extra_fields_unchanged():
    from verl.workers.rollout.llm_server import LLMServerClient

    class Remote:
        async def remote(self, **kwargs):
            return TokenOutput(token_ids=[1], extra_fields={"global_steps": 3, "rollout_finish_reason": "stop"})

    client = LLMServerClient.__new__(LLMServerClient)
    client.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(name="vllm")))

    async def acquire(*args, **kwargs):
        return "server", SimpleNamespace(generate=Remote())

    client._acquire_server = acquire
    client._release_server = lambda *args, **kwargs: None
    output = asyncio.run(client.generate("test", prompt_ids=[99], sampling_params={}))
    assert output.extra_fields["rollout_finish_reason"] == "stop"
    assert output.extra_fields["min_global_steps"] == output.extra_fields["max_global_steps"] == 3
