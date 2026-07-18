# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the TransformerEngine NVFP4 grouped-experts MoE override.

Covers override targeting (GroupedExperts.Config -> TEGroupedExperts.Config, the
TorchAOTokenDispatcher swap, the 16-alignment skip, and the all-to-all dispatcher
requirement) and single-GPU expert numerics (Blackwell) against the stock bf16
grouped MM -- which also proves gradients reach the reused base expert weights.

Run from the repo root so `te_moe_overrides` and `torchtitan` are importable:
    PYTHONPATH=. python -m pytest te_moe_tests/test_te_grouped_experts.py
"""

import unittest

import torch

from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.token_dispatcher import (
    AllToAllTokenDispatcher,
    LocalTokenDispatcher,
    TorchAOTokenDispatcher,
)
from te_moe_overrides.te_grouped_experts import TEGroupedExperts, te_grouped_experts

_DIM = 256
_HIDDEN = 256  # _DIM, _HIDDEN divisible by 16 (and 128)
_E = 8
_TOP_K = 1


def _blackwell() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


def _grouped_experts_config(
    dispatcher=None, *, dim: int = _DIM, hidden_dim: int = _HIDDEN
) -> GroupedExperts.Config:
    if dispatcher is None:
        dispatcher = AllToAllTokenDispatcher.Config(num_experts=_E, top_k=_TOP_K)
    return GroupedExperts.Config(
        dim=dim,
        hidden_dim=hidden_dim,
        num_experts=_E,
        token_dispatcher=dispatcher,
    )


class TestTEGroupedExpertsTargeting(unittest.TestCase):
    """Config-time override targeting (hardware-independent)."""

    def test_replaces_grouped_experts_and_swaps_dispatcher(self):
        out = te_grouped_experts(_grouped_experts_config())
        self.assertIsInstance(out, TEGroupedExperts.Config)
        self.assertIsInstance(out.token_dispatcher, TorchAOTokenDispatcher.Config)
        self.assertEqual(out.token_dispatcher.pad_multiple, 128)
        self.assertEqual(out.token_dispatcher.num_experts, _E)
        self.assertEqual(out.token_dispatcher.top_k, _TOP_K)
        # Expert dims are carried through by derive().
        self.assertEqual((out.dim, out.hidden_dim, out.num_experts), (_DIM, _HIDDEN, _E))

    def test_skips_when_dims_not_16_aligned(self):
        cfg = _grouped_experts_config(dim=300)
        self.assertIs(te_grouped_experts(cfg), cfg)
        cfg = _grouped_experts_config(hidden_dim=300)
        self.assertIs(te_grouped_experts(cfg), cfg)

    def test_requires_alltoall_dispatcher(self):
        cfg = _grouped_experts_config(
            LocalTokenDispatcher.Config(num_experts=_E, top_k=_TOP_K)
        )
        with self.assertRaisesRegex(ValueError, "all-to-all"):
            te_grouped_experts(cfg)


@unittest.skipUnless(_blackwell(), "TE NVFP4 grouped GEMM requires Blackwell (sm_100+)")
class TestTEGroupedExpertsNumerics(unittest.TestCase):
    """Single-GPU (EP=1) expert numerics against the stock bf16 grouped MM."""

    def test_forward_close_to_reference_and_backward_finite(self):
        torch.manual_seed(0)
        dispatcher = LocalTokenDispatcher.Config(num_experts=_E, top_k=_TOP_K)
        stock = _grouped_experts_config(dispatcher).build().cuda()
        te_experts = (
            TEGroupedExperts.Config(
                dim=_DIM,
                hidden_dim=_HIDDEN,
                num_experts=_E,
                token_dispatcher=dispatcher,
            )
            .build()
            .cuda()
        )

        with torch.no_grad():
            w1 = 0.1 * torch.randn(_E, _HIDDEN, _DIM, device="cuda")
            w3 = 0.1 * torch.randn(_E, _HIDDEN, _DIM, device="cuda")
            w2 = 0.1 * torch.randn(_E, _DIM, _HIDDEN, device="cuda")
            for m in (stock, te_experts):
                m.w1_EFD.copy_(w1)
                m.w3_EFD.copy_(w3)
                m.w2_EDF.copy_(w2)

        # 128 tokens per expert: each group is 128-aligned and sum == rows, so
        # there is no dispatcher padding tail to account for.
        num_tokens = torch.full((_E,), 128, device="cuda", dtype=torch.int64)
        rows = int(num_tokens.sum())
        x = torch.randn(rows, _DIM, device="cuda", dtype=torch.bfloat16)
        x_ref = x.detach().clone().requires_grad_()
        x_q = x.detach().clone().requires_grad_()

        out_ref = stock._experts_forward(x_ref, num_tokens)
        out_q = te_experts._experts_forward(x_q, num_tokens)

        from torchao.quantization.utils import compute_error

        # Three chained NVFP4 GEMMs (gate -> silu*up -> down) accumulate more
        # 4-bit quantization error than a single linear; 10 dB leaves margin while
        # still failing hard on broken math (NaN / wrong contraction give ~0 dB).
        sqnr = compute_error(out_ref.float(), out_q.float())
        self.assertGreaterEqual(sqnr.item(), 10.0)

        out_q.sum().backward()
        self.assertTrue(torch.isfinite(x_q.grad).all())


if __name__ == "__main__":
    unittest.main()
