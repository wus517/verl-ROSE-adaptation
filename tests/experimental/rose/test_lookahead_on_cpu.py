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
"""Pure token-distance, pivot ranking and reservation tests."""

from types import SimpleNamespace

import pytest

from verl.experimental.rose.lookahead import (
    ProbeBudget,
    evaluate_probe,
    levenshtein_distance,
    rank_legal_pivots,
    segment_seed,
)


@pytest.mark.parametrize(
    "a,b,expected",
    [([], [], 0), ([1], [], 1), ([], [2, 3], 2), ([1, 2], [1, 2], 0), ([1, 2], [2, 1], 2), ([1, 1, 2], [1, 2], 1)],
)
def test_edit_distance(a, b, expected):
    assert levenshtein_distance(a, b) == expected


def test_all_windows_and_threshold_equality():
    accepted, distances = evaluate_probe([1] * 50, [2] * 8 + [1] * 42, [20, 30, 50], 0.4)
    assert distances["20"] == 0.4
    assert not accepted
    assert evaluate_probe([1] * 50, [2] * 50, [20, 30, 50], 0.4)[0]
    with pytest.raises(ValueError, match="cover"):
        evaluate_probe([1] * 50, [2] * 10, [20, 50], 0.4)


def test_shared_prefix_dedup_legal_positions_and_stable_ties():
    def trajectory(ids, scores):
        return SimpleNamespace(output=SimpleNamespace(response_ids=ids), semantic_entropy=scores)

    rows = [trajectory([1, 2, 3, 4], [0.1, 0.8, float("nan"), 99]), trajectory([1, 2, 5, 6], [0.1, 0.9, 0.8, 99])]
    candidates = rank_legal_pivots(rows, 4, 2)
    assert [(c.parent_id, c.position) for c in candidates] == [(1, 1), (1, 2), (0, 0)]
    assert len({c.prefix_key for c in candidates}) == 3
    assert rank_legal_pivots(rows, 1, 2) == []


@pytest.mark.parametrize("limit,can_reserve", [(0, False), (49, False), (50, True), (700, True)])
def test_budget_reserves_full_window_and_settles_actual(limit, can_reserve):
    budget = ProbeBudget(limit)
    assert budget.reserve(50) == can_reserve
    if can_reserve:
        budget.settle(12)
        assert budget.used == 12 and budget.reserved == 0
        assert budget.reserve(50) == (limit >= 62)


def test_budget_rejects_invalid_accounting():
    budget = ProbeBudget(100)
    with pytest.raises(ValueError):
        budget.settle(1)
    assert budget.reserve(50)
    with pytest.raises(ValueError):
        budget.reserve(50)
    with pytest.raises(ValueError):
        budget.settle(51)


def test_seed_stable_and_separates_trials_and_phases():
    assert segment_seed(42, 1, 2, 0, "probe") == segment_seed(42, 1, 2, 0, "probe")
    assert len({segment_seed(42, 1, 2, trial, phase) for trial in range(2) for phase in ("probe", "completion")}) == 4
