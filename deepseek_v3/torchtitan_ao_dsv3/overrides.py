"""Registers the torchao NVFP4 grouped-experts override."""

import torch
from torch.utils._triton import has_triton

from torchao.utils import is_sm_at_least_100, torch_version_at_least
from torchtitan.config import derive, override
from torchtitan.models.common.moe import GroupedExperts

from .nvfp4_moe import NVFP4GroupedExperts

_NVFP4_RECOMPILE_LIMIT = 64

if torch._dynamo.config.recompile_limit < _NVFP4_RECOMPILE_LIMIT:
    torch._dynamo.config.recompile_limit = _NVFP4_RECOMPILE_LIMIT


def _assert_nvfp4_supported() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("NVFP4 grouped experts require CUDA")
    if not is_sm_at_least_100():
        raise RuntimeError("NVFP4 grouped experts require an SM100+ GPU")
    if not has_triton():
        raise RuntimeError("NVFP4 grouped experts require Triton")
    if not torch_version_at_least("2.10.0"):
        raise RuntimeError("NVFP4 grouped experts require PyTorch 2.10 or newer")


@override(
    "nvfp4_grouped_experts",
    target=GroupedExperts.Config,
    exact=True,
    description="Replace GroupedExperts with torchao NVFP4 grouped GEMMs (Blackwell)",
)
def nvfp4_grouped_experts_override(
    cfg: GroupedExperts.Config,
) -> GroupedExperts.Config:
    _assert_nvfp4_supported()
    if cfg.dim % 128 or cfg.hidden_dim % 128:
        return cfg
    return derive(cfg, NVFP4GroupedExperts.Config)
