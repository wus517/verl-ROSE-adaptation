import json
from pathlib import Path
from typing import Any

import numpy as np


class SemanticEntropyScorer:
    """Compute ROSE semantic entropy from compact top-k generation data."""

    def __init__(
        self,
        embedding_path: str | Path,
        *,
        metadata_path: str | Path | None = None,
        chunk_size: int = 256,
        exclude_diagonal: bool = True,
    ) -> None:
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        embedding_path = Path(embedding_path)
        if metadata_path is None:
            metadata_path = embedding_path.with_suffix(".json")
        metadata_path = Path(metadata_path)

        with metadata_path.open(encoding="utf-8") as metadata_file:
            metadata: dict[str, Any] = json.load(metadata_file)

        if not metadata.get("normalized", False):
            raise ValueError("ROSE embedding table must be row-normalized")
        shape = (int(metadata["vocab_size"]), int(metadata["hidden_size"]))
        dtype = np.dtype(metadata.get("dtype", "float16"))
        expected_size = int(np.prod(shape)) * dtype.itemsize
        actual_size = embedding_path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"ROSE embedding file size {actual_size} does not match metadata shape {shape} "
                f"and dtype {dtype} ({expected_size} bytes)"
            )
        self.embeddings = np.memmap(embedding_path, dtype=dtype, mode="r", shape=shape)
        self.vocab_size = shape[0]
        self.hidden_size = shape[1]
        self.chunk_size = chunk_size
        self.exclude_diagonal = exclude_diagonal

    @classmethod
    def from_array(
        cls,
        embeddings: np.ndarray,
        *,
        chunk_size: int = 256,
        exclude_diagonal: bool = True,
        normalize: bool = True,
    ) -> "SemanticEntropyScorer":
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        scorer = cls.__new__(cls)
        array = np.asarray(embeddings)
        if array.ndim != 2:
            raise ValueError(f"embeddings must be rank 2, got shape {array.shape}")
        if normalize:
            norms = np.linalg.norm(array.astype(np.float32), axis=1, keepdims=True)
            if np.any(norms == 0):
                raise ValueError("embedding table contains zero-norm rows")
            array = array.astype(np.float32) / norms
        scorer.embeddings = array
        scorer.vocab_size, scorer.hidden_size = array.shape
        scorer.chunk_size = chunk_size
        scorer.exclude_diagonal = exclude_diagonal
        return scorer

    def score(
        self,
        topk_token_ids: np.ndarray,
        topk_logprobs: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        token_ids = np.asarray(topk_token_ids)
        logprobs = np.asarray(topk_logprobs, dtype=np.float32)
        if token_ids.ndim != 2 or logprobs.shape != token_ids.shape:
            raise ValueError(
                f"top-k ids and logprobs must have the same rank-2 shape, got {token_ids.shape} and {logprobs.shape}"
            )
        if valid_mask is None:
            mask = np.ones(token_ids.shape, dtype=np.bool_)
        else:
            mask = np.asarray(valid_mask, dtype=np.bool_)
            if mask.shape != token_ids.shape:
                raise ValueError(f"valid_mask shape {mask.shape} does not match token ids {token_ids.shape}")

        valid_ids = token_ids[mask]
        if valid_ids.size and (np.any(valid_ids < 0) or np.any(valid_ids >= self.vocab_size)):
            minimum = int(valid_ids.min())
            maximum = int(valid_ids.max())
            raise ValueError(f"top-k token ids [{minimum}, {maximum}] exceed vocabulary size {self.vocab_size}")

        result = np.zeros(token_ids.shape[0], dtype=np.float32)
        for start in range(0, token_ids.shape[0], self.chunk_size):
            stop = min(start + self.chunk_size, token_ids.shape[0])
            chunk_mask = mask[start:stop]
            chunk_ids = np.where(chunk_mask, token_ids[start:stop], 0)
            chunk_logprobs = logprobs[start:stop]
            probabilities = np.where(chunk_mask, np.exp(chunk_logprobs), 0.0).astype(np.float32)
            safe_logprobs = np.where(chunk_mask, chunk_logprobs, 0.0)
            generation_entropy = -np.sum(probabilities * safe_logprobs, axis=-1, dtype=np.float32)

            candidate_embeddings = np.asarray(self.embeddings[chunk_ids], dtype=np.float32)
            weighted_embedding = np.einsum("lk,lkh->lh", probabilities, candidate_embeddings, optimize=True)
            pairwise_similarity = np.sum(weighted_embedding * weighted_embedding, axis=-1, dtype=np.float32)
            if self.exclude_diagonal:
                pairwise_similarity -= np.sum(probabilities * probabilities, axis=-1, dtype=np.float32)
            semantic_divergence = -pairwise_similarity
            result[start:stop] = generation_entropy * semantic_divergence

        if not np.all(np.isfinite(result)):
            raise FloatingPointError("semantic entropy contains NaN or Inf")
        return result
