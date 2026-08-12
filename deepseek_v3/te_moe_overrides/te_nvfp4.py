# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NVFP4 MoE quantization converter backed by TransformerEngine.

Second backend for the same recipe as torchtitan's
``NVFP4GroupedExpertsConverter`` (``torchtitan.components.quantization.nvfp4``),
which uses TorchAO's Triton kernels. This one drives TE's fused NVFP4 grouped
GEMM instead. Both converters override the same ``GroupedExperts._grouped_mm``
seam, so a TE-vs-TorchAO training run differs in exactly one thing -- the
quantize+GEMM primitive. silu, the elementwise product, dispatcher padding, the
bf16 casts, and EP sharding are shared code.

TE's public ``GroupedLinear`` module owns its own weights and so cannot be used
here (it would fight TorchTitan's expert sharding and sever autograd); instead we
call TE's autograd-capable grouped-GEMM Function directly on the base
``w1_EFD``/``w2_EDF``/``w3_EFD`` parameters, which is the TE analogue of
TorchAO's ``_to_nvfp4_rht_rs_then_scaled_grouped_mm``.

Usage:

    torchtitan_train --module te_moe_overrides --config deepseek_v3_16b_te_nvfp4

TE's NVFP4 grouped GEMM targets NVIDIA Blackwell (sm_100+). ``transformer_engine``
is imported lazily so this module (and its config-time targeting tests) import
without it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, cast

import torch

from torchtitan.components.quantization import QuantizationConverter
from torchtitan.components.quantization.utils import swap_token_dispatcher
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability

# The fused NVFP4 grouped GEMM path in TE is gated behind this env var, read at
# forward time. Set it once at import (before any TE forward), self-contained in
# this module.
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
    except ImportError as e:  # pragma: no cover - exercised only without TE
        _TE_IMPORT_ERROR = e

__all__ = ["TEGroupedExpertsConverter", "_get_te_grouped_experts_cls"]


def _require_te() -> None:
    if _TE_IMPORT_ERROR is not None:
        raise ImportError(
            "the TE NVFP4 MoE converter was requested but transformer_engine is "
            "not importable; install a TransformerEngine build that provides "
            "transformer_engine.pytorch with NVFP4 grouped GEMM support."
        ) from _TE_IMPORT_ERROR


def _make_quantizers() -> tuple:
    """Build the (input, weight, grad) quantizers for ``NVFP4BlockScaling``.

    Mirrors TE's recipe->quantizer factory (``quantization.py`` ``_make``), minus
    the 4over6 and row-scaled-activation branches, both disabled by
    ``NVFP4BlockScaling`` defaults (``nvfp4_4over6="none"``,
    ``row_scaled_activation=False``). Usage (rowwise/columnwise) is (re)set per
    call inside the grouped-GEMM Function.
    """
    recipe = NVFP4BlockScaling()

    def _make(qparams) -> "NVFP4Quantizer":
        return NVFP4Quantizer(
            rowwise=True,
            columnwise=True,
            with_rht=qparams.random_hadamard_transform,
            with_post_rht_amax=qparams.random_hadamard_transform,
            with_2d_quantization=qparams.fp4_2d_quantization,
            stochastic_rounding=qparams.stochastic_rounding,
        )

    return (
        recipe,
        _make(recipe.fp4_quant_fwd_inp),
        _make(recipe.fp4_quant_fwd_weight),
        _make(recipe.fp4_quant_bwd_grad),
    )


_te_experts_cache: dict[type, type] = {}


def _get_te_grouped_experts_cls(parent_cls: type) -> type:
    """Get or create a TE-NVFP4-quantized subclass of *parent_cls*.

    Works for any experts module exposing the ``_grouped_mm`` seam (the common
    ``GroupedExperts`` and ``GptOssGroupedExperts``). The returned class has a
    proper ``_owner`` set by ``__init_subclass__``.
    """
    if parent_cls in _te_experts_cache:
        return _te_experts_cache[parent_cls]

    parent_config_cls = parent_cls.Config  # type: ignore[attr-defined]

    class TEGroupedExperts(parent_cls):  # type: ignore[valid-type, misc]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc]
            pass

        def _init_self_buffers(
            self, *, buffer_device: torch.device | None = None
        ) -> None:
            super()._init_self_buffers(buffer_device=buffer_device)
            # Built here, not lazily on first forward: under SelectiveAC the
            # checkpointed block's forward and its backward recompute must emit an
            # identical op sequence, and the quantizers allocate their RHT matrices
            # on construction. This is also where the TorchAO converter materializes
            # its own runtime NVFP4 state (_sr_seed / _rht_sign_vector).
            _require_te()
            dev = (
                buffer_device
                if buffer_device is not None
                else cast(torch.Tensor, self.w1_EFD).device
            )
            with torch.cuda.device(dev):
                self._recipe, self._q_in, self._q_weight, self._q_grad = (
                    _make_quantizers()
                )

        def _grouped_mm(self, *, A, B_t, offs):
            # TE derives per-expert row counts from split sizes summing to
            # A.shape[0], so the final offset must cover the dispatcher's padding
            # tail (TorchAOTokenDispatcher leaves a sentinel tail that the
            # per-expert counts exclude). Clone before the in-place write so the
            # shared per-forward offsets tensor is untouched. Identical to the
            # TorchAO seam.
            offs = offs.clone()
            offs[-1] = A.shape[0]
            m_splits = torch.diff(offs, prepend=offs.new_zeros(1)).to(torch.int64)

            # B_t is (E, K, N); transposing back gives the per-expert (out, in)
            # "TN" layout TE wants -- and restores the base parameter's own memory
            # layout, so the unbound slices are contiguous.
            # unbind (one op) rather than per-index select: its backward stacks the
            # E weight grads into a single (E, N, K) tensor, instead of E full-size
            # select_backward scatter-adds that dominate the grouped backward at
            # large E.
            weights = list(torch.unbind(B_t.transpose(-2, -1), dim=0))
            num_gemms = len(weights)
            # use_bias=False, but the Function indexes biases[0]; pass placeholders.
            biases = [A.new_empty(0) for _ in range(num_gemms)]

            grad_q = [self._q_grad] * num_gemms
            is_grad = torch.is_grad_enabled()
            non_tensor_args = (
                False,  # use_bias
                None,  # is_first_microbatch
                True,  # fp8 (selects the NVFP4 quantized path)
                False,  # fp8_calibration
                None,  # wgrad_store
                [self._q_in] * num_gemms,  # input_quantizers
                [self._q_weight] * num_gemms,  # weight_quantizers
                [None] * num_gemms,  # output quantization unsupported on this path
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
            # The Function reads the active recipe from FP8GlobalStateManager, so
            # the autocast context is required even though quantizers are explicit.
            with te.autocast(enabled=True, recipe=self._recipe):
                out, _ = linear_fn(
                    *autograd_ctx,
                    A,
                    m_splits,
                    non_tensor_args,
                    None,  # out
                    None,  # dgrad_out
                    *weights,
                    *biases,
                )
            return out

    TEGroupedExperts.__name__ = f"TENVFP4{parent_cls.__name__}"
    TEGroupedExperts.__qualname__ = f"TENVFP4{parent_cls.__name__}"
    _te_experts_cache[parent_cls] = TEGroupedExperts
    return TEGroupedExperts


class TEGroupedExpertsConverter(QuantizationConverter):
    """Apply TransformerEngine NVFP4 quantization to MoE expert grouped GEMMs."""

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        fqns: list[str] = field(default_factory=list)
        """
        List of fully qualified names of experts modules to quantize. Only
        GroupedExperts.Config entries whose FQN contains a match are converted.
        If empty, all experts are converted. Pass explicit fqns (e.g. the leading
        decoder layers) to keep the tail layers' experts in bf16.
        """
        pad_multiple: int = 128
        """
        Pad per-expert token groups to this multiple for NVFP4 grouped GEMM
        alignment. Matches the TorchAO NVFP4 converter so the two arms see
        identical token grouping.
        """

    def __init__(self, config: Config):
        self.config = config
        _require_te()

        if not has_cuda_capability(10, 0):
            raise ValueError("NVFP4 is only supported on SM100 or later architectures")

        if not self.config.model_compile_enabled:
            logger.warning(
                "torch.compile enablement is required for highest performance "
                "of NVFP4 dynamic quantization."
            )

    def convert(self, model_config):
        fqns = self.config.fqns
        for fqn, config, parent, attr in model_config.traverse(GroupedExperts.Config):
            if fqns and not any(target_fqn in fqn for target_fqn in fqns):
                continue
            # ``parent`` is the RoutedExperts.Config owning inner_experts + dispatcher.
            swap_token_dispatcher(parent, self.config.pad_multiple)
            base_module_cls = type(config)._owner
            quantized_cls = _get_te_grouped_experts_cls(base_module_cls)
            config_cls = quantized_cls.Config  # type: ignore[attr-defined]
            new_config = config_cls(
                **{f.name: getattr(config, f.name) for f in fields(config)}
            )
            if parent is None:
                model_config = new_config
            elif isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)

        logger.info(
            "Converted GroupedExperts to use dynamic NVFP4 quantization "
            "(TransformerEngine) for grouped_mm ops"
        )
        return model_config
