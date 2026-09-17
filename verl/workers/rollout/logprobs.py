from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

ROSE_TOPK_TOKEN_IDS_FIELD = "rose_topk_token_ids"
ROSE_TOPK_LOGPROBS_FIELD = "rose_topk_logprobs"
ROSE_TOPK_VALID_MASK_FIELD = "rose_topk_valid_mask"


def normalize_requested_logprobs(value: bool | int | None) -> int | None:
    """Normalize VeRL's bool-or-int logprobs request for an inference backend."""
    if value is None:
        return None
    if isinstance(value, bool):
        return 0 if value else None
    value = int(value)
    if value < 0:
        raise ValueError(f"logprobs must be non-negative, got {value}")
    return value


def _entry_logprob(entry: Any) -> float:
    try:
        return float(entry.logprob)
    except AttributeError as exc:
        raise TypeError(f"logprob entry must expose a .logprob attribute, got {type(entry)!r}") from exc


def pack_topk_logprobs(
    position_logprobs: Sequence[Mapping[int | str, Any]],
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack backend per-position logprob mappings into fixed-shape arrays.

    vLLM may return the requested top-k candidates plus the sampled token when
    the sampled token falls outside top-k. Valid ranks are used when available;
    missing rank slots fall back to sorting by logprob.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be positive, got {top_k}")

    length = len(position_logprobs)
    token_ids = np.full((length, top_k), -1, dtype=np.int32)
    logprobs = np.full((length, top_k), -np.inf, dtype=np.float32)
    valid_mask = np.zeros((length, top_k), dtype=np.bool_)

    for position, candidates in enumerate(position_logprobs):
        if candidates is None:
            continue

        candidate_items = [(int(token_id), entry) for token_id, entry in candidates.items()]
        used_token_ids: set[int] = set()

        for token_id, entry in candidate_items:
            rank = getattr(entry, "rank", None)
            if rank is None:
                continue
            rank = int(rank)
            if not 1 <= rank <= top_k:
                continue
            slot = rank - 1
            if valid_mask[position, slot]:
                raise ValueError(f"duplicate logprob rank {rank} at response position {position}")
            token_ids[position, slot] = token_id
            logprobs[position, slot] = _entry_logprob(entry)
            valid_mask[position, slot] = True
            used_token_ids.add(token_id)

        missing_slots = np.flatnonzero(~valid_mask[position]).tolist()
        if missing_slots:
            unranked = sorted(
                (
                    (_entry_logprob(entry), token_id)
                    for token_id, entry in candidate_items
                    if token_id not in used_token_ids
                ),
                reverse=True,
            )
            for slot, (logprob, token_id) in zip(missing_slots, unranked, strict=False):
                token_ids[position, slot] = token_id
                logprobs[position, slot] = logprob
                valid_mask[position, slot] = True

    return token_ids, logprobs, valid_mask
