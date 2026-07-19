# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: NVFP4 grouped-experts MoE path backed by TransformerEngine.

This is a second backend for the same idea as
``torchtitan.overrides.nvfp4_grouped_experts`` (which uses TorchAO): it keeps
bf16 expert weights and quantizes the three MoE grouped GEMMs (gate, up, down)
to NVFP4 on the fly, but drives them through TransformerEngine's fused NVFP4
grouped GEMM instead of TorchAO's kernels. Running both under an identical
launch harness gives an apples-to-apples TE-vs-TorchAO comparison (same 4-bit
precision, two kernel libraries).

Usage (from this repo, on PYTHONPATH):

    torchtitan_train --module deepseek_v3 --config deepseek_v3_debugmodel \\
        --override.imports te_moe_overrides.te_grouped_experts

Design -- like the TorchAO override, this subclasses ``GroupedExperts`` and swaps
*only* the three grouped GEMMs. The base's stacked expert weights
(``w1_EFD``/``w2_EDF``/``w3_EFD``) are reused as-is, so TorchTitan keeps owning
expert-weight EP sharding, initialization, and checkpointing; gradients flow back
to those parameters. TE's public ``GroupedLinear`` module owns its own weights and
so cannot be used here (it would fight TorchTitan's sharding and sever autograd);
instead we call TE's autograd-capable grouped-GEMM Function directly on the base
weight tensors -- the TE analogue of TorchAO's
``_to_nvfp4_then_scaled_grouped_mm``.

The override also swaps the token dispatcher for ``TorchAOTokenDispatcher`` so
token groups are padded to a multiple of 128 -- more than the 16-element block
NVFP4 quantization requires -- so every per-expert group is quantizable; the
standard EP all-to-all dispatcher is a prerequisite.

TE's NVFP4 grouped GEMM targets NVIDIA Blackwell (sm_100+). ``transformer_engine``
is imported lazily so this module (and its config-time targeting tests) import
without it; the factory raises a clear error if selected when TE is unavailable,
and the hardware requirement is checked in :meth:`TEGroupedExperts.parallelize`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from torchtitan.config import derive, override
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.token_dispatcher import (
    AllToAllTokenDispatcher,
    TorchAOTokenDispatcher,
)

# The fused NVFP4 grouped GEMM path in TE is gated behind this env var, read at
# forward time. Set it once at import (before any TE forward), self-contained in
# this module -- mirrors the recompile-limit bump in the TorchAO override.
os.environ.setdefault("NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM", "1")

if TYPE_CHECKING:
    _TE_IMPORT_ERROR: ImportError | None = None
else:
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import NVFP4BlockScaling
        from transformer_engine.pytorch import NVFP4Quantizer
        from transformer_engine.pytorch.module.grouped_linear import _GroupedLinear

        _TE_IMPORT_ERROR = None
    except ImportError as e:
        _TE_IMPORT_ERROR = e

__all__ = ["TEGroupedExperts", "te_grouped_experts"]

# TorchAO's EP permute_and_pad pads token groups to this multiple; 128 is a
# superset of NVFP4's 16-element quantization block, so every group is quantizable.
_NVFP4_PAD_MULTIPLE = 128


def _require_te() -> None:
    if _TE_IMPORT_ERROR is not None:
        raise ImportError(
            "te grouped-experts override was requested but transformer_engine is "
            "not importable; install a TransformerEngine build that provides "
            "transformer_engine.pytorch with NVFP4 grouped GEMM support."
        ) from _TE_IMPORT_ERROR


def _assert_te_nvfp4_supported() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("TE NVFP4 grouped experts require CUDA")
    if torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("TE NVFP4 grouped experts require an SM100+ (Blackwell) GPU")


class TEGroupedExperts(GroupedExperts):
    """GroupedExperts backed by TransformerEngine NVFP4 grouped GEMMs."""

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        # Built lazily on first forward (needs CUDA for the RHT matrices).
        self._recipe = None
        self._q_in = None
        self._q_weight = None
        self._q_grad = None

    def parallelize(self, parallel_dims) -> None:
        # Hardware requirement is checked here (not in the override factory) so
        # config-time targeting is hardware-independent; parallelize() runs before
        # any real forward under the EP paths this override targets.
        _assert_te_nvfp4_supported()
        super().parallelize(parallel_dims)
        # Build the NVFP4 quantizers (and their RHT matrices) eagerly, before any
        # forward. Under SelectiveAC the checkpointed block's forward and its
        # backward recompute must emit an identical op sequence; a lazy build on
        # the first forward would run its allocation ops only on the initial pass
        # (not the recompute), desyncing SAC's saved-tensor bookkeeping.
        self._ensure_te_state()

    def _ensure_te_state(self) -> None:
        if self._recipe is not None:
            return
        _require_te()
        recipe = NVFP4BlockScaling()

        def _make(qparams) -> "NVFP4Quantizer":
            # Mirror TE's recipe->quantizer factory (quantization.py `_make`), minus
            # the 4over6 branch (disabled by NVFP4BlockScaling defaults). Usage
            # (rowwise/columnwise) is (re)set per call inside the grouped-GEMM Function.
            return NVFP4Quantizer(
                rowwise=True,
                columnwise=True,
                with_rht=qparams.random_hadamard_transform,
                with_post_rht_amax=qparams.random_hadamard_transform,
                with_2d_quantization=qparams.fp4_2d_quantization,
                stochastic_rounding=qparams.stochastic_rounding,
            )

        self._q_in = _make(recipe.fp4_quant_fwd_inp)
        self._q_weight = _make(recipe.fp4_quant_fwd_weight)
        self._q_grad = _make(recipe.fp4_quant_bwd_grad)
        self._recipe = recipe

    def _te_grouped_gemm(
        self,
        x_Rin: torch.Tensor,
        weight_EOI: torch.Tensor,
        counts_E: torch.Tensor,
    ) -> torch.Tensor:
        """One NVFP4 grouped GEMM: (rows, in) x (E, out, in) -> (rows, out).

        ``weight_EOI`` is the stacked local expert weight in TE "TN" layout
        (out, in) per expert -- exactly the base ``w*`` buffers, no transpose.
        ``counts_E`` are per-expert token COUNTS with ``sum == rows`` (TE derives
        offsets internally). Weights are passed as external tensors so gradients
        flow back to the base parameters.
        """
        num_gemms = weight_EOI.size(0)
        # unbind (one op) rather than per-index select: its backward stacks the E
        # weight grads into a single (E, O, I) tensor, instead of E full-size
        # select_backward scatter-adds that dominate the grouped backward at large E.
        weights = list(torch.unbind(weight_EOI, dim=0))
        # use_bias=False, but the Function indexes biases[0]; pass placeholders.
        biases = [x_Rin.new_empty(0) for _ in range(num_gemms)]

        input_q = [self._q_in] * num_gemms
        weight_q = [self._q_weight] * num_gemms
        output_q = [None] * num_gemms  # output quantization unsupported on this path
        grad_q = [self._q_grad] * num_gemms

        is_grad = torch.is_grad_enabled()
        non_tensor_args = (
            False,  # use_bias
            None,  # is_first_microbatch
            True,  # fp8 (selects the NVFP4 quantized path)
            False,  # fp8_calibration
            None,  # wgrad_store
            input_q,
            weight_q,
            output_q,
            grad_q,  # grad_input_quantizers
            grad_q,  # grad_weight_quantizers
            grad_q,  # grad_output_quantizers
            False,  # fuse_wgrad_accumulation
            False,  # cpu_offloading
            False,  # sequence_parallel
            torch.bfloat16,  # activation_dtype
            is_grad,  # is_grad_enabled
            [None] * num_gemms,  # weight_workspaces
            False,  # cache_weight
            None,  # skip_fp8_weight_update
            False,  # save_original_input
            False,  # debug
        )
        linear_fn = _GroupedLinear.apply if is_grad else _GroupedLinear.forward
        autograd_ctx = () if is_grad else (None,)
        out, _ = linear_fn(
            *autograd_ctx, x_Rin, counts_E, non_tensor_args, *weights, *biases
        )
        return out

    def _experts_forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        self._ensure_te_state()

        if isinstance(self.w1_EFD, DTensor):
            w1_EFD = self.w1_EFD.to_local()
            assert isinstance(self.w2_EDF, DTensor)
            w2_EDF = self.w2_EDF.to_local()
            assert isinstance(self.w3_EFD, DTensor)
            w3_EFD = self.w3_EFD.to_local()
        else:
            w1_EFD = self.w1_EFD
            w2_EDF = self.w2_EDF
            w3_EFD = self.w3_EFD

        # TE grouped GEMM wants x.size(0) == sum(counts); the TorchAOTokenDispatcher
        # appends a padding sentinel tail (rows beyond sum of the padded per-expert
        # counts). Run TE on exactly the counted rows, then zero-pad the output back
        # so combine()/_unpermute can strip the sentinel -- keeping every NVFP4 group
        # 128-aligned. When there is no padding tail (EP=1 / numerics test), total
        # equals the row count and the pad is a no-op.
        counts_E = num_tokens_per_expert_E.to(torch.int64)
        total = int(counts_E.sum())
        x = x_RD[:total].bfloat16()
        w1 = w1_EFD.bfloat16()
        w2 = w2_EDF.bfloat16()
        w3 = w3_EFD.bfloat16()

        with te.autocast(enabled=True, recipe=self._recipe):
            gate_RF = self._te_grouped_gemm(x, w1, counts_E)
            up_RF = self._te_grouped_gemm(x, w3, counts_E)
            h_RF = F.silu(gate_RF) * up_RF
            out_RD = self._te_grouped_gemm(h_RF, w2, counts_E)

        if out_RD.size(0) < x_RD.size(0):
            out_RD = F.pad(out_RD, (0, 0, 0, x_RD.size(0) - out_RD.size(0)))
        return out_RD.type_as(x_RD)


def _torchao_token_dispatcher(
    cfg: GroupedExperts.Config,
) -> TorchAOTokenDispatcher.Config:
    if not isinstance(cfg.token_dispatcher, AllToAllTokenDispatcher.Config):
        raise ValueError(
            "TE NVFP4 grouped experts require the standard EP all-to-all token "
            "dispatcher so token groups are padded before grouped MM; got "
            f"{type(cfg.token_dispatcher).__name__}."
        )

    return TorchAOTokenDispatcher.Config(
        num_experts=cfg.token_dispatcher.num_experts,
        top_k=cfg.token_dispatcher.top_k,
        pad_multiple=_NVFP4_PAD_MULTIPLE,
    )


@override(
    "te_grouped_experts",
    target=GroupedExperts.Config,
    exact=True,
    description="Replace GroupedExperts with TransformerEngine NVFP4 grouped GEMMs (Blackwell)",
)
def te_grouped_experts(
    cfg: GroupedExperts.Config,
) -> GroupedExperts.Config:
    _require_te()
    # NVFP4 quantizes in 16-element blocks along both dims of each expert GEMM.
    if cfg.dim % 16 or cfg.hidden_dim % 16:
        return cfg
    return derive(
        cfg,
        TEGroupedExperts.Config,
        token_dispatcher=_torchao_token_dispatcher(cfg),
    )
