# Copyright 2026 individual contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""EA numerical contracts, independent of rollout and accelerator libraries."""

import math

import pytest
import torch

from verl.workers.utils.entropy_advantage import entropy_advantage_from_logits, shape_entropy_advantage


def test_formula_mask_detach_and_no_accumulation():
    base = torch.tensor([[0.2, -0.2, 0, 0.2, -0.2, float("nan")]])
    entropy = torch.tensor([[0.5, 0.5, 10, 10, 10, float("inf")]], requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
    shaped, bonus, stats = shape_entropy_advantage(base, entropy, mask, coef=0.1, kappa=2)
    torch.testing.assert_close(shaped, torch.tensor([[0.25, -0.15, 0, 0.3, -0.1, 0]]))
    torch.testing.assert_close(bonus, torch.tensor([[0.05, 0.05, 0, 0.1, 0.1, 0]]))
    assert not shaped.requires_grad and not bonus.requires_grad
    assert stats["valid_tokens_sum"] == 5 and stats["cap_hits_sum"] == 2
    assert stats["sign_flips_sum"] == 0
    again, _, _ = shape_entropy_advantage(base, entropy, mask, coef=0.1, kappa=2)
    torch.testing.assert_close(shaped, again)
    torch.testing.assert_close(base[0, :5], torch.tensor([0.2, -0.2, 0, 0.2, -0.2]))
    assert torch.isnan(base[0, -1])
    logprob = torch.zeros_like(shaped, requires_grad=True)
    loss = -(logprob * shaped).sum()
    loss.backward()
    torch.testing.assert_close(logprob.grad, -shaped)
    assert entropy.grad is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.001])
def test_invalid_valid_entropy_fails(bad):
    with pytest.raises(ValueError, match="finite"):
        shape_entropy_advantage(torch.ones(1, 1), torch.tensor([[bad]]), torch.ones(1, 1), coef=0.1, kappa=2)


@pytest.mark.parametrize("coef,kappa", [(-0.1, 2), (float("inf"), 2), (True, 2), (0.1, 1), (0.1, float("nan"))])
def test_shaping_rejects_invalid_parameters(coef, kappa):
    with pytest.raises(ValueError, match="EA"):
        shape_entropy_advantage(torch.ones(1, 1), torch.ones(1, 1), torch.ones(1, 1), coef=coef, kappa=kappa)


def test_padding_zero_coef_and_negative_tolerance():
    a = torch.tensor([[1.0, -1.0]])
    h = torch.tensor([[-1e-6, 0.5]])
    shaped, bonus, _ = shape_entropy_advantage(a, h, torch.ones_like(a), coef=0, kappa=2)
    torch.testing.assert_close(shaped, a)
    assert bonus.count_nonzero() == 0
    shaped, bonus, stats = shape_entropy_advantage(a * float("nan"), h, torch.zeros_like(a), coef=0.1, kappa=2)
    assert shaped.count_nonzero() == bonus.count_nonzero() == 0
    assert all(value == 0 for value in stats.values())
    with pytest.raises(ValueError, match="shapes"):
        shape_entropy_advantage(a, h[:, :1], torch.ones_like(a), coef=0.1, kappa=2)
    with pytest.raises(ValueError, match="same device"):
        shape_entropy_advantage(a, h.to("meta"), torch.ones_like(a), coef=0.1, kappa=2)
    with pytest.raises(ValueError, match="finite"):
        shape_entropy_advantage(a * float("inf"), h, torch.ones_like(a), coef=0.1, kappa=2)


@pytest.mark.parametrize("shape", [(7, 11), (2, 7, 11)])
@pytest.mark.parametrize("chunk", [1, 4, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_chunk_entropy_matches_full_reference_and_bounds(shape, chunk, dtype, monkeypatch):
    torch.manual_seed(42)
    # Slicing vocabulary creates non-contiguous inputs, without a full contiguous copy in helper.
    logits = torch.randn(*shape[:-1], shape[-1] * 2).to(dtype)[..., ::2].requires_grad_()
    before = logits.detach().clone()
    logp = logits.detach().float().log_softmax(-1)
    expected = -(logp.exp() * logp).sum(-1)
    softmax = torch.softmax
    seen = []

    def bounded_softmax(values, *args, **kwargs):
        seen.append(values.shape)
        assert values.ndim == 2 and values.shape[0] <= chunk
        assert values.dtype == torch.float32 and not torch.is_grad_enabled()
        return softmax(values, *args, **kwargs)

    monkeypatch.setattr(torch, "softmax", bounded_softmax)
    actual = entropy_advantage_from_logits(logits, chunk_size=chunk)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(logits, before)
    assert actual.dtype == torch.float32 and actual.device == logits.device
    assert not actual.requires_grad and actual.grad_fn is None and seen


def test_entropy_stable_uniform_shift_and_extreme():
    uniform = entropy_advantage_from_logits(torch.full((3, 17), 10000.0), chunk_size=2)
    torch.testing.assert_close(uniform, torch.full((3,), math.log(17)))
    logits = torch.tensor([[10000.0, -10000.0], [1.0, 2.0]])
    h = entropy_advantage_from_logits(logits, chunk_size=1)
    torch.testing.assert_close(h, entropy_advantage_from_logits(logits + 100, chunk_size=64))
    assert h[0] == 0


@pytest.mark.parametrize("chunk", [0, -1, True, 1.5])
def test_invalid_chunks(chunk):
    with pytest.raises(ValueError, match="positive integer"):
        entropy_advantage_from_logits(torch.ones(2, 3), chunk_size=chunk)


@pytest.mark.parametrize("logits", [torch.ones(2), torch.ones(1, 2, 3, 4), torch.ones(2, 0), torch.ones(2, 3).long()])
def test_invalid_logits(logits):
    with pytest.raises(ValueError, match="logits"):
        entropy_advantage_from_logits(logits, chunk_size=2)
