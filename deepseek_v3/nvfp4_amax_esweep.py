"""Discriminate WHY grouped rht_amax degrades: atomic-max contention vs grid/launch.

Two sweeps of the grouped rht_amax kernel (N=7168):
  A) Fixed TOTAL rows = 65536, vary E. M-per-group = 65536/E shrinks as E grows.
     - atomic-contention hypothesis: fewer rows/group -> less contention per
       per-group amax scalar -> grouped_us DROPS as E grows.
     - grid/launch-overhead hypothesis: same total tiles -> grouped_us ~flat/up.
  B) Fixed tok/expert = 8192, vary E. Per-group work fixed; is grouped_us linear
     in E (independent groups) or super-linear (cross-group interference)?

Also prints single-kernel time at the matching M and the ratio.
    cd third_party/torchao && PYTHONPATH=. python ../../deepseek_v3/nvfp4_amax_esweep.py
"""

import torch
import triton
from tabulate import tabulate

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.nvfp4_training.group_hadamard_utils import (
    BLOCK_M as G_BLOCK_M,
    BLOCK_N as G_BLOCK_N,
    VARYING_FIRST_DIM,
)
from torchao.prototype.moe_training.nvfp4_training.group_hadamard_amax_triton import (
    _group_rht_amax_triton_kernel,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_amax_triton import (
    _hadamard_amax_kernel,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import get_rht_matrix
from torchao.utils import is_sm_at_least_100

dev = torch.device("cuda")
SIGN = (1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1)
N = 7168
NUM_SMS = torch.cuda.get_device_properties(dev).multi_processor_count
if hasattr(triton, "set_allocator"):
    triton.set_allocator(
        lambda s, a, st: torch.empty(s, dtype=torch.int8, device=dev)
    )


def grouped_amax(e, m):
    total = e * m
    A = torch.randn(total, N, dtype=torch.bfloat16, device=dev)
    B = get_rht_matrix(SIGN, dev, torch.bfloat16, 16)
    offsets = torch.arange(1, e + 1, dtype=torch.int32, device=dev) * m
    lpl = offsets[-1:]
    row_amax = torch.zeros((e,), dtype=torch.float32, device=dev)
    col_amax = torch.zeros((e,), dtype=torch.float32, device=dev)
    grid = (triton.cdiv(total, G_BLOCK_M) * triton.cdiv(N, G_BLOCK_N),)

    def run():
        _group_rht_amax_triton_kernel[grid](
            A, B, offsets, row_amax, col_amax, total, N,
            num_tensors=e, SHAPE_REP=VARYING_FIRST_DIM,
            BLOCK_M=G_BLOCK_M, BLOCK_N=G_BLOCK_N, RHT_SIZE=16,
            logical_packed_length_ptr=lpl, num_warps=8, num_stages=3,
        )

    return benchmark_cuda_function_in_microseconds(run)


def single_amax(m):
    x = torch.randn(m, N, dtype=torch.bfloat16, device=dev)
    B = get_rht_matrix(SIGN, dev, torch.bfloat16, 16)
    r = torch.zeros(1, dtype=torch.float32, device=dev)
    a = torch.zeros(1, dtype=torch.float32, device=dev)

    def run():
        _hadamard_amax_kernel[(NUM_SMS,)](
            x, B, r, a, m, N, GROUP_SIZE_N=8, NUM_SMS=NUM_SMS,
        )

    return benchmark_cuda_function_in_microseconds(run)


def main():
    if not is_sm_at_least_100():
        raise RuntimeError("requires SM100+")
    torch.manual_seed(0)

    print("\n### A) FIXED TOTAL rows = 65536, vary E  (M/group = 65536/E)")
    rows = []
    for e in [1, 2, 4, 8, 16, 32, 64]:
        m = 65536 // e
        g = grouped_amax(e, m)
        s = single_amax(m)
        rows.append([e, m, round(g, 2), round(s, 2), round(e * s, 2), round(g / (e * s), 2)])
    print(tabulate(rows, headers=["E", "M/group", "grouped_us", "single_us", "E*single_us", "ratio"]))

    print("\n### B) FIXED tok/expert = 8192, vary E  (grouped_us/E = per-group cost)")
    rows = []
    for e in [1, 2, 4, 8, 16]:
        m = 8192
        g = grouped_amax(e, m)
        rows.append([e, m, e * m, round(g, 2), round(g / e, 2)])
    print(tabulate(rows, headers=["E", "M/group", "total_rows", "grouped_us", "grouped_us/E"]))


if __name__ == "__main__":
    main()
