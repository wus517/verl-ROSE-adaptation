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
"""Backend-independent primitives for ROSE's bounded lookahead search."""

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass


def levenshtein_distance(left: Sequence[int], right: Sequence[int]) -> int:
    """Unit-cost token edit distance, using one rolling row."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def evaluate_probe(parent_suffix, probe_ids, windows, threshold) -> tuple[bool, dict[str, float]]:
    """Require the normalized token distance threshold at every complete window."""
    if len(parent_suffix) < max(windows) or len(probe_ids) < max(windows):
        raise ValueError("Full-window evaluation requires both suffixes to cover all windows")
    distances = {str(r): levenshtein_distance(parent_suffix[:r], probe_ids[:r]) / r for r in windows}
    return all(d >= threshold for d in distances.values()), distances


@dataclass(frozen=True)
class PivotCandidate:
    """A retained-parent pivot identified by its actual shared prefix."""

    parent_id: int
    position: int
    score: float
    prefix_key: str


def rank_legal_pivots(trajectories, response_length: int, window: int) -> list[PivotCandidate]:
    """Deduplicate shared prefixes, keeping the highest-scoring representative.

    Incremental SHA-256 avoids copying/hashing the full prefix at every position.
    Hash input uses fixed-width unsigned token IDs; ties follow parent/position order.
    """
    candidates = {}
    for parent_id, trajectory in enumerate(trajectories):
        ids = trajectory.output.response_ids
        digest = hashlib.sha256()
        for b, score in enumerate(trajectory.semantic_entropy):
            if b > 0:
                digest.update(int(ids[b - 1]).to_bytes(8, "big", signed=False))
            score = float(score)
            if not math.isfinite(score) or min(len(ids), response_length) - b < window:
                continue
            key = f"{b}:{digest.hexdigest()}"
            candidate = PivotCandidate(parent_id, b, score, key)
            if key not in candidates or score > candidates[key].score:
                candidates[key] = candidate
    return sorted(candidates.values(), key=lambda c: (-c.score, c.parent_id, c.position))


@dataclass
class ProbeBudget:
    """Per-group serial reservation budget; accepted probes also consume it."""

    limit: int
    used: int = 0
    reserved: int = 0

    def reserve(self, amount: int) -> bool:
        """Reserve a whole window before dispatch, without shortening it to fit."""
        if amount < 1 or self.reserved:
            raise ValueError("Only one positive probe reservation may be active")
        if self.used + amount > self.limit:
            return False
        self.reserved = amount
        return True

    def settle(self, actual: int) -> None:
        """Charge actual returned tokens and release unused reserved capacity."""
        if not self.reserved or not 0 <= actual <= self.reserved:
            raise ValueError("Invalid probe budget settlement")
        self.used += actual
        self.reserved = 0


def segment_seed(base_seed: int, slot: int, pivot: int, trial: int, phase: str) -> int:
    """Derive backend-compatible seeds that separate repeated probes and completions."""
    key = f"{base_seed}:{slot}:{pivot}:{trial}:{phase}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")
