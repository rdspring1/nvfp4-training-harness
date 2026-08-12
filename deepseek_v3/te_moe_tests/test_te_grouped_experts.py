# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the TransformerEngine NVFP4 MoE grouped-experts converter.

Covers config-time targeting (the DSV3 recipe converts only the leading-85% MoE
layers and swaps their dispatcher) and single-GPU expert numerics on Blackwell
against the stock bf16 grouped MM -- which also proves gradients reach the reused
base expert weights.

Run from `deepseek_v3/` so `te_moe_overrides` and `torchtitan` are importable:
    PYTHONPATH=. python -m pytest te_moe_tests/test_te_grouped_experts.py
"""

import pytest
import torch

from torchtitan.config import ConfigManager
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.token_dispatcher import TorchAOTokenDispatcher

import te_moe_overrides.te_nvfp4 as te_mod
from te_moe_overrides.te_nvfp4 import _get_te_grouped_experts_cls

_DIM = 256
_HIDDEN = 256  # both divisible by 128, which the NVFP4 grouped GEMM requires
_E = 8


def _blackwell() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


def test_converter_targets_leading_moe_layers(monkeypatch):
    """The DSV3 TE recipe swaps only the leading-85% MoE layers' experts to the TE
    subclass and swaps their dispatcher to TorchAOTokenDispatcher(pad=128); the
    bf16-tail MoE layer keeps stock GroupedExperts and its original dispatcher.

    Mirrors torchtitan's own test for the TorchAO converter
    (tests/unit_tests/test_quantization.py) -- the two arms must target
    identically, or the loss-curve comparison is not apples-to-apples.
    """
    pytest.importorskip("transformer_engine")
    # The config-tree transform is GPU-independent: bypass the sm100 gate the
    # converter __init__ enforces.
    monkeypatch.setattr(te_mod, "has_cuda_capability", lambda *_: True)

    config = ConfigManager().parse_args(
        ["--module", "te_moe_overrides", "--config", "deepseek_v3_debugmodel_te_nvfp4"]
    )
    model_config = config.model_spec.model

    TEExperts = _get_te_grouped_experts_cls(GroupedExperts)
    converted, stock = [], []
    for fqn, cfg, parent, _attr in model_config.traverse(GroupedExperts.Config):
        if isinstance(cfg, TEExperts.Config):
            converted.append(fqn)
            assert isinstance(parent.token_dispatcher, TorchAOTokenDispatcher.Config)
            assert parent.token_dispatcher.pad_multiple == 128
        else:
            stock.append(fqn)
            assert not isinstance(
                parent.token_dispatcher, TorchAOTokenDispatcher.Config
            )

    # debugmodel: 6 layers, 1 dense -> MoE in layers 1..5; the 15% tail keeps
    # layer 5 in bf16. Same split the TorchAO recipe produces.
    assert {int(fqn.split(".")[1]) for fqn in converted} == {1, 2, 3, 4}
    assert {int(fqn.split(".")[1]) for fqn in stock} == {5}


def test_subclass_owner_and_seam():
    """_owner wiring and the _grouped_mm seam override, per the converter protocol."""
    pytest.importorskip("transformer_engine")
    cls = _get_te_grouped_experts_cls(GroupedExperts)
    assert cls.Config._owner is cls
    assert issubclass(cls, GroupedExperts)
    assert cls._grouped_mm is not GroupedExperts._grouped_mm
    # Cached: repeated calls return the same class.
    assert _get_te_grouped_experts_cls(GroupedExperts) is cls


@pytest.mark.skipif(
    not _blackwell(), reason="TE NVFP4 grouped GEMM requires Blackwell (sm_100+)"
)
def test_forward_close_to_reference_and_backward_finite():
    """Single-GPU (EP=1) expert numerics against the stock bf16 grouped MM."""
    torch.manual_seed(0)
    kw = dict(dim=_DIM, hidden_dim=_HIDDEN, num_experts=_E)
    stock = GroupedExperts.Config(**kw).build().cuda()
    te_experts = _get_te_grouped_experts_cls(GroupedExperts).Config(**kw).build().cuda()
    te_experts._init_self_buffers(buffer_device=torch.device("cuda"))

    with torch.no_grad():
        w1 = 0.1 * torch.randn(_E, _HIDDEN, _DIM, device="cuda")
        w3 = 0.1 * torch.randn(_E, _HIDDEN, _DIM, device="cuda")
        w2 = 0.1 * torch.randn(_E, _DIM, _HIDDEN, device="cuda")
        for m in (stock, te_experts):
            m.w1_EFD.copy_(w1)
            m.w3_EFD.copy_(w3)
            m.w2_EDF.copy_(w2)

    # 128 tokens per expert: each group is 128-aligned and sum == rows, so there
    # is no dispatcher padding tail to account for.
    num_tokens = torch.full((_E,), 128, device="cuda", dtype=torch.int64)
    rows = int(num_tokens.sum())
    x = torch.randn(rows, _DIM, device="cuda", dtype=torch.bfloat16)
    x_ref = x.detach().clone().requires_grad_()
    x_q = x.detach().clone().requires_grad_()

    out_ref = stock(x_ref, num_tokens)
    out_q = te_experts(x_q, num_tokens)

    from torchao.quantization.utils import compute_error

    # Three chained NVFP4 GEMMs (gate -> silu*up -> down) accumulate more 4-bit
    # quantization error than a single linear; 10 dB leaves margin while still
    # failing hard on broken math (NaN / wrong contraction give ~0 dB).
    sqnr = compute_error(out_ref.float(), out_q.float())
    assert sqnr.item() >= 10.0

    out_q.sum().backward()
    assert torch.isfinite(x_q.grad).all()
    for name in ("w1_EFD", "w2_EDF", "w3_EFD"):
        grad = getattr(te_experts, name).grad
        assert grad is not None and torch.isfinite(grad).all()
