"""CPU-only tests for ROSE semantic entropy helpers."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from verl.experimental.rose.semantic_entropy import SemanticEntropyScorer
from verl.workers.rollout.logprobs import normalize_requested_logprobs, pack_topk_logprobs


def test_normalize_requested_logprobs_preserves_integer_topk():
    assert normalize_requested_logprobs(None) is None
    assert normalize_requested_logprobs(False) is None
    assert normalize_requested_logprobs(True) == 0
    assert normalize_requested_logprobs(0) == 0
    assert normalize_requested_logprobs(20) == 20
    with pytest.raises(ValueError, match="non-negative"):
        normalize_requested_logprobs(-1)


def test_pack_topk_logprobs_uses_rank_and_excludes_extra_sampled_token():
    position_logprobs = [
        {
            7: SimpleNamespace(logprob=-0.4, rank=2),
            8: SimpleNamespace(logprob=-0.1, rank=1),
            9: SimpleNamespace(logprob=-3.0, rank=3),
        }
    ]

    token_ids, logprobs, valid_mask = pack_topk_logprobs(position_logprobs, top_k=2)

    np.testing.assert_array_equal(token_ids, [[8, 7]])
    np.testing.assert_allclose(logprobs, [[-0.1, -0.4]])
    np.testing.assert_array_equal(valid_mask, [[True, True]])


def test_pack_topk_logprobs_falls_back_to_probability_order_without_ranks():
    position_logprobs = [
        {
            "4": SimpleNamespace(logprob=-1.2, rank=None),
            "5": SimpleNamespace(logprob=-0.2, rank=None),
        }
    ]

    token_ids, logprobs, valid_mask = pack_topk_logprobs(position_logprobs, top_k=3)

    np.testing.assert_array_equal(token_ids, [[5, 4, -1]])
    np.testing.assert_allclose(logprobs[:, :2], [[-0.2, -1.2]])
    np.testing.assert_array_equal(valid_mask, [[True, True, False]])


def test_semantic_entropy_matches_pairwise_reference_and_chunking():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    token_ids = np.array([[0, 1], [0, 2]], dtype=np.int32)
    logprobs = np.log(np.array([[0.6, 0.3], [0.5, 0.25]], dtype=np.float32))

    scorer = SemanticEntropyScorer.from_array(embeddings, chunk_size=1)
    actual = scorer.score(token_ids, logprobs)

    normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    expected = []
    for ids, row_logprobs in zip(token_ids, logprobs, strict=True):
        probabilities = np.exp(row_logprobs)
        entropy = -np.sum(probabilities * row_logprobs)
        divergence = 0.0
        for i in range(len(ids)):
            for j in range(len(ids)):
                if i != j:
                    divergence -= probabilities[i] * probabilities[j] * np.dot(normalized[ids[i]], normalized[ids[j]])
        expected.append(entropy * divergence)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_semantic_entropy_masks_missing_candidates_and_rejects_oov():
    scorer = SemanticEntropyScorer.from_array(np.eye(2, dtype=np.float32))
    token_ids = np.array([[0, -1]], dtype=np.int32)
    logprobs = np.array([[-0.2, -np.inf]], dtype=np.float32)
    mask = np.array([[True, False]])

    np.testing.assert_allclose(scorer.score(token_ids, logprobs, mask), [0.0])

    with pytest.raises(ValueError, match="vocabulary size"):
        scorer.score(np.array([[2]], dtype=np.int32), np.array([[-0.1]], dtype=np.float32))


def test_semantic_entropy_rejects_embedding_file_size_mismatch(tmp_path):
    embedding_path = tmp_path / "embedding.f16"
    embedding_path.write_bytes(b"\x00\x00")
    metadata_path = tmp_path / "embedding.json"
    metadata_path.write_text(
        json.dumps(
            {
                "vocab_size": 2,
                "hidden_size": 2,
                "dtype": "float16",
                "normalized": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="file size"):
        SemanticEntropyScorer(embedding_path, metadata_path=metadata_path)


def test_semantic_entropy_rejects_invalid_chunk_size():
    with pytest.raises(ValueError, match="chunk_size"):
        SemanticEntropyScorer.from_array(np.eye(2, dtype=np.float32), chunk_size=0)
