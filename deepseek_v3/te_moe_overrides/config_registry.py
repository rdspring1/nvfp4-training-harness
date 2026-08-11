# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek V3 config flavors that quantize the MoE grouped GEMMs with TE NVFP4.

Selected via torchtitan's ``--module`` mechanism, which imports
``{module}.config_registry`` (``torchtitan/config/manager.py``):

    torchtitan_train --module te_moe_overrides --config deepseek_v3_16b_te_nvfp4

Each flavor is a field-for-field copy of torchtitan's own
``deepseek_v3_*_nvfp4`` (same 15% bf16 tail, same ``pad_multiple=128``, same
``attn_backend``) with ``NVFP4GroupedExpertsConverter`` swapped for
``TEGroupedExpertsConverter``. The converter class is the only delta between the
two arms, which is what makes a TE-vs-TorchAO loss-curve comparison meaningful.
"""

from __future__ import annotations

from torchtitan.components.quantization import NVFP4LinearConverter
from torchtitan.components.quantization.nvfp4 import nvfp4_bf16_tail_fqns
from torchtitan.models.deepseek_v3 import model_registry
from torchtitan.models.deepseek_v3.config_registry import (
    _NVFP4_FFN_SUBMODULES,
    _NVFP4_FFN_SUBMODULES_NO_DENSE,
    _nvfp4_ffn_linear_fqns,
    deepseek_v3_16b,
    deepseek_v3_debugmodel,
)
from torchtitan.trainer import Trainer

from .te_nvfp4 import TEGroupedExpertsConverter

# Keep the leading (1 - fraction) of decoder layers in NVFP4 and the tail in
# bf16. Must match torchtitan's deepseek_v3_*_nvfp4 flavors.
_NVFP4_BF16_TAIL_FRACTION = 0.15


def _converters(
    config: Trainer.Config, ffn_submodules: tuple[str, ...]
) -> list[object]:
    """The TorchAO flavor's converter list with TE swapped in for the experts.

    The FFN Linears use TorchAO's NVFP4Linear on both arms -- TE supplies no
    NVFP4 Linear here, and keeping them identical is what leaves the grouped
    GEMM as the only difference between the two arms.
    """
    assert config.model_spec is not None
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    n_layers = len(config.model_spec.model.layers)
    fqns = nvfp4_bf16_tail_fqns(n_layers, _NVFP4_BF16_TAIL_FRACTION)
    return [
        NVFP4LinearConverter.Config(
            model_compile_enabled=model_compile_enabled,
            fqns=_nvfp4_ffn_linear_fqns(fqns, ffn_submodules),
        ),
        TEGroupedExpertsConverter.Config(
            model_compile_enabled=model_compile_enabled,
            fqns=fqns,
            pad_multiple=128,
        ),
    ]


def deepseek_v3_debugmodel_te_nvfp4() -> Trainer.Config:
    config = deepseek_v3_debugmodel()
    config.model_spec = model_registry(
        "debugmodel",
        converters=_converters(config, _NVFP4_FFN_SUBMODULES),
    )
    return config


def deepseek_v3_16b_te_nvfp4() -> Trainer.Config:
    config = deepseek_v3_16b()
    config.model_spec = model_registry(
        "16B",
        attn_backend="flex",
        converters=_converters(config, _NVFP4_FFN_SUBMODULES_NO_DENSE),
    )
    return config
