"""Re-measure Table 5's rht_amax row with the dispatch heuristic LIVE.

Table 5 (nvfp4_grouped_kernel_grouping.py) timed the raw TILED grouped kernel.
This times the public op ``triton_group_rht_amax``, which now auto-dispatches to
the per-group-CTA persistent kernel when avg rows/group >= _PERSISTENT_MIN_AVG_ROWS
(=1024). Confirms the gain shows through the public API vs the E*single baseline.

    cd third_party/torchao && \\
        PYTHONPATH=. python ../../deepseek_v3/nvfp4_amax_dispatch_table5.py
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
    triton_group_rht_amax,
    _PERSISTENT_MIN_AVG_ROWS,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_amax_triton import (
    _hadamard_amax_kernel,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import get_rht_matrix
from torchao.utils import is_sm_at_least_100

dev = torch.device("cuda")
SIGN = (1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1)
E = 8
N_ACT = 7168
TOKENS = [512, 1024, 2048, 4096, 8192]
NUM_SMS = torch.cuda.get_device_properties(dev).multi_processor_count


def set_alloc():
    if hasattr(triton, "set_allocator"):
        triton.set_allocator(
            lambda size, align, stream: torch.empty(size, dtype=torch.int8, device=dev)
        )


def build(e, m, n):
    total = e * m
    A = torch.randn(total, n, dtype=torch.bfloat16, device=dev)
    offsets = (torch.arange(1, e + 1, dtype=torch.int32, device=dev) * m).contiguous()
    return A, offsets, total


def dispatched_amax(e, m, n):
    A, offsets, total = build(e, m, n)
    lpl = offsets[-1:]

    def run():
        triton_group_rht_amax(
            A, list(SIGN), offsets, e, total, n, VARYING_FIRST_DIM, logical_packed_length=lpl
        )

    return benchmark_cuda_function_in_microseconds(run)


def tiled_amax(e, m, n):
    A, offsets, total = build(e, m, n)
    B = get_rht_matrix(SIGN, dev, torch.bfloat16, 16)
    lpl = offsets[-1:]
    row_amax = torch.zeros((e,), dtype=torch.float32, device=dev)
    col_amax = torch.zeros((e,), dtype=torch.float32, device=dev)
    grid = (triton.cdiv(total, G_BLOCK_M) * triton.cdiv(n, G_BLOCK_N),)
    set_alloc()

    def run():
        _group_rht_amax_triton_kernel[grid](
            A, B, offsets, row_amax, col_amax, total, n,
            num_tensors=e, SHAPE_REP=VARYING_FIRST_DIM,
            BLOCK_M=G_BLOCK_M, BLOCK_N=G_BLOCK_N, RHT_SIZE=16,
            logical_packed_length_ptr=lpl, num_warps=8, num_stages=3,
        )

    return benchmark_cuda_function_in_microseconds(run)


def single_amax(m, n):
    set_alloc()
    x = torch.randn(m, n, dtype=torch.bfloat16, device=dev)
    B = get_rht_matrix(SIGN, dev, torch.bfloat16, 16)
    rht_out = torch.zeros(1, dtype=torch.float32, device=dev)
    a_out = torch.zeros(1, dtype=torch.float32, device=dev)

    def run():
        _hadamard_amax_kernel[(NUM_SMS,)](
            x, B, rht_out, a_out, m, n, GROUP_SIZE_N=8, NUM_SMS=NUM_SMS,
        )

    return benchmark_cuda_function_in_microseconds(run)


def main():
    if not is_sm_at_least_100():
        raise RuntimeError("requires SM100+ (Blackwell)")
    torch.manual_seed(0)
    print(f"_PERSISTENT_MIN_AVG_ROWS={_PERSISTENT_MIN_AVG_ROWS}  NUM_SMS={NUM_SMS}  E={E}")

    rows = []
    for tpe in TOKENS:
        d = dispatched_amax(E, tpe, N_ACT)
        t = tiled_amax(E, tpe, N_ACT)
        s = single_amax(tpe, N_ACT)
        path = "persistent" if (E <= NUM_SMS and tpe >= _PERSISTENT_MIN_AVG_ROWS) else "tiled"
        rows.append([tpe, E * tpe, path, round(d, 2), round(t, 2), round(E * s, 2),
                     round(d / (E * s), 2), round(t / (E * s), 2), round(t / d, 2)])
    print("\n### Table 5 rht_amax with dispatch live (public op) vs old tiled")
    print(tabulate(rows, headers=[
        "tok/exp", "rows", "path", "dispatch_us", "tiled_us", "E*single_us",
        "ratio_new", "ratio_old", "speedup"]))


if __name__ == "__main__":
    main()
