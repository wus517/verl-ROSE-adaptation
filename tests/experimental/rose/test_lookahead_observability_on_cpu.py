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
"""Trainer integration for deduplicated metrics and aligned rollout history."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.trainer.ppo.v1 import trainer_base


def metric_row():
    return {
        "rose_tree_metadata": {"path_node_ids": [0, 1]},
        "rose_root_restart": True,
        "rose_branch_pos": 0,
        "rose_prefix_reuse_tokens": 0,
        "rose_new_generated_tokens": 4,
        "rose_topk_payload_bytes": 8,
        "rose_lookahead_group": {
            "tree_id": "one",
            "probe_attempts": 2,
            "probe_rejected": 1,
            "probe_full_window_accepted": 1,
            "probe_generated_tokens": 100,
        },
    }


def test_metrics_deduplicate_tree_and_ignore_padding():
    first = metric_row()
    second = copy.deepcopy(first)
    metrics = trainer_base._compute_rose_rollout_metrics([first, second, {}], np.array([True, True, False]))
    assert metrics["rose_lookahead/retained_groups"] == 1
    assert metrics["rose_lookahead/retained_probe_generated_tokens_sum"] == 100
    assert metrics["rose_lookahead/probe_full_window_accepted_ratio"] == 0.5
    second["rose_lookahead_group"]["probe_attempts"] = 3
    with pytest.raises(ValueError, match="Inconsistent"):
        trainer_base._compute_rose_rollout_metrics([first, second], np.array([True, True]))


@pytest.mark.parametrize("include_summary", [False, True])
def test_rollout_dump_sorts_metadata_with_response(monkeypatch, include_summary):
    class Padded:
        def __init__(self, values):
            self.values = values

        def to_padded_tensor(self, **kwargs):
            return self.values

    extras = [{"rose_lookahead_leaf": {"source": "second"}}, {"rose_lookahead_leaf": {"source": "first"}}]

    def fetch(**kwargs):
        return {
            "uid": np.array(["z", "a"]),
            "prompts": Padded([[2], [1]]),
            "responses": Padded([[22], [11]]),
            "rm_scores": torch.zeros(2, 1),
            "reward_model": np.array([{}, {}]),
            "extra_fields": np.array(extras if include_summary else [{}, {}], dtype=object),
        }

    monkeypatch.setattr(trainer_base.tq, "kv_batch_get", fetch)
    dumped = []
    trainer = SimpleNamespace(
        tokenizer=SimpleNamespace(pad_token_id=0, decode=lambda ids, **kw: str(ids)),
        _dump_generations=lambda **kwargs: dumped.append(kwargs),
    )
    batch = SimpleNamespace(keys=["z_0_0", "a_0_0"], partition_id="train")
    trainer_base.PPOTrainer._log_rollout_data(trainer, batch, {}, "unused")
    assert dumped[0]["outputs"] == ["[11]", "[22]"]
    info = dumped[0]["reward_extra_infos_dict"]
    assert info["uid"] == ["a_0_0", "z_0_0"]
    if include_summary:
        assert info["rose_lookahead_leaf"] == [{"source": "first"}, {"source": "second"}]
    else:
        assert "rose_lookahead_leaf" not in info
