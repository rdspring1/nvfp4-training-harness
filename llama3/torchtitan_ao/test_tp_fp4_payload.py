"""Multi-rank test: the colwise TP all-gather moves NVFP4 codes/scales, not bf16.

This guards the whole point of the SP block path -- keeping activations
sequence-sharded so the column-parallel all-gather communicates fp4 codes
(uint8) and swizzled scales (float8 viewed as uint8), instead of the stock
bf16 activation gather. It fails if a bf16 activation tensor is ever
all-gathered in the colwise forward (i.e. the optimization regressed).

Run on >= 2 Blackwell GPUs from the ``llama3`` dir (PYTHONPATH so the plugin
imports, as run_titan.py does):
    PYTHONPATH=$PWD torchrun --standalone --nproc_per_node 2 \
        torchtitan_ao/test_tp_fp4_payload.py
"""

import os

import torch
import torch.distributed as dist

import torchao.prototype.moe_training.nvfp4_training.nvfp4_tensor_parallel as tp_mod
from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
    get_wgrad_sign_vector,
)

from torchtitan_ao.nvfp4 import _nvfp4_colwise_sp

# Local shapes (per rank). Both dims and the local sequence must clear 128.
B, L_LOCAL, D, N = 2, 128, 256, 256


def _run() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    group = dist.group.WORLD

    gathered_dtypes: list[torch.dtype] = []
    real_all_gather = tp_mod.all_gather_tensor

    def _recording_all_gather(tensor, *args, **kwargs):
        gathered_dtypes.append(tensor.dtype)
        return real_all_gather(tensor, *args, **kwargs)

    tp_mod.all_gather_tensor = _recording_all_gather
    try:
        torch.manual_seed(rank)
        x = torch.randn(B, L_LOCAL, D, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, D, device="cuda", dtype=torch.bfloat16)
        sr_seed = torch.zeros(1, dtype=torch.int64, device="cuda")
        sign = tuple(int(v) for v in get_wgrad_sign_vector(16, device="cuda", dtype=torch.int8))
        out = _nvfp4_colwise_sp(x, w, None, sr_seed, sign, group, world_size)
    finally:
        tp_mod.all_gather_tensor = real_all_gather

    # The colwise forward must all-gather at least once (the fp4 activation codes).
    assert gathered_dtypes, "colwise forward did not all-gather anything"
    # Every gathered payload must be the fp4 representation (uint8 codes, or
    # float8 scales which are reinterpreted as uint8 for NCCL). None may be bf16.
    assert torch.bfloat16 not in gathered_dtypes, (
        f"colwise all-gather moved bf16 activations (regressed to bf16 gather): "
        f"{gathered_dtypes}"
    )
    assert all(dt == torch.uint8 for dt in gathered_dtypes), (
        f"expected only uint8 fp4 codes/scales in the all-gather, got {gathered_dtypes}"
    )
    # Output is full-sequence, feature-sharded: [B, L_full, N].
    assert out.shape == (B, L_LOCAL * world_size, N), out.shape

    if rank == 0:
        print(f"OK: colwise all-gather payload dtypes = {gathered_dtypes}")
    dist.destroy_process_group()


def test_colwise_allgather_is_fp4_not_bf16():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        import pytest

        pytest.skip("needs >= 2 GPUs")
    if "RANK" not in os.environ:
        import pytest

        pytest.skip("run under torchrun --nproc_per_node 2")
    _run()


if __name__ == "__main__":
    _run()
