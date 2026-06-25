"""Registers torchao NVFP4 Linear override for TorchTitan's registry."""

import os

import torch

from torchtitan.config import derive, override
from torchtitan.models.common import Linear

from .nvfp4 import TritonNVFP4Linear

_NVFP4_RECOMPILE_LIMIT = 64

# NVFP4 passes per-module Python sign-vector tuples into the compiled block.
# Under fullgraph=True those constants plus async DTensor collectives can
# exceed Dynamo's default limit before step 1 on Llama3 8B TP runs.
if torch._dynamo.config.recompile_limit < _NVFP4_RECOMPILE_LIMIT:
    torch._dynamo.config.recompile_limit = _NVFP4_RECOMPILE_LIMIT


@override(
    "triton_nvfp4_linear",
    target=Linear.Config,
    description="Replace Linear with torchao Triton NVFP4Linear (Blackwell)",
)
def triton_nvfp4_linear_override(
    cfg: Linear.Config,
) -> TritonNVFP4Linear.Config | None:
    # torchao's NVFP4 kernels require both M and K of the local weight to be
    # divisible by 128. Under TP=N a weight dim becomes dim // N locally, so
    # we conservatively require both dims to clear 128 * N to handle either
    # colwise (out sharded) or rowwise (in sharded). TP degree is passed in
    # via env by run_titan.py.
    tp = int(os.environ.get("TORCHTITAN_AO_TP_DEGREE", "1"))
    local_block = 128 * max(tp, 1)
    if cfg.in_features % local_block != 0 or cfg.out_features % local_block != 0:
        return cfg
    return derive(cfg, TritonNVFP4Linear.Config)
