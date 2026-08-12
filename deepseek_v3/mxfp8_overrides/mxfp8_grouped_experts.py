"""Override: MXFP8 grouped-experts MoE path.

Thin adapter that exposes torchtitan's first-class ``MXFP8GroupedExpertsConverter``
(``torchtitan.components.quantization.mx``) through the ``--override.imports``
mechanism. All quantization semantics live in the converter (recipe
``mxfp8_rceil``: e4m3 data + e8m0 block-32 scales); the grouped GEMMs dispatch to
``torch._scaled_grouped_mm``. This module only bridges the converter into the
override registry.

``run_titan.py`` no longer launches this path: its ``--mxfp8`` flag went away when
NVFP4 moved from ``--override.imports`` to config-flavor selection
(``--precision``). Invoke torchtitan directly, or reinstate an MXFP8 config flavor
alongside ``te_moe_overrides/config_registry.py``:

    torchtitan_train --module deepseek_v3 --config deepseek_v3_16b \\
        --override.imports mxfp8_overrides.mxfp8_grouped_experts

Requires a CUDA-built torchao (the ``_C_mxfp8`` extension) on ``PYTHONPATH``; the
editable ``USE_CPP=0`` build lacks the MXFP8 quant kernel.
"""

from __future__ import annotations

from torchtitan.components.quantization.mx import MXFP8GroupedExpertsConverter
from torchtitan.config import override
from torchtitan.models.common.moe import GroupedExperts


@override(
    "mxfp8_grouped_experts",
    target=GroupedExperts.Config,
    exact=True,
    description="Replace GroupedExperts with MXFP8 grouped GEMMs (torchtitan MXFP8 converter)",
)
def mxfp8_grouped_experts(cfg: GroupedExperts.Config) -> GroupedExperts.Config:
    # The sm100 CuTeDSL grouped-MM kernel requires each per-expert token group's
    # M divisible by 128 (the converter default pad_multiple=32 under-pads and
    # trips "M must be divisible by 128"); pad to 128 like the NVFP4 path. The
    # converter swaps the dispatcher accordingly and derives the quantized subclass.
    return (
        MXFP8GroupedExpertsConverter.Config(model_compile_enabled=False, pad_multiple=128)
        .build()
        .convert(cfg)
    )
