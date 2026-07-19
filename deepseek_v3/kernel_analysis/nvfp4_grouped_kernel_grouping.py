"""Grouped NVFP4 Triton kernels vs per-expert single-kernel launches.

For each of the three forward-path NVFP4 quantization kernels in
torchao/prototype/moe_training/nvfp4_training/, time:
  - the GROUPED kernel over E experts in one launch (grouped_us)
  - the SINGLE-expert kernel at the same per-expert size (single_us), then
    estimate E individual launches as E * single_us

ratio = grouped_us / (E * single_us).  >1 => grouping is SLOWER than launching
the single kernel per expert; <1 => grouping wins.

Activation kernels (rht_amax, rht_quantize_row_col) are swept over tokens/expert
at the DSV3-671B activation dim N=7168.  The weight kernel (quantize_2d) does not
depend on tokens; it is reported once per weight shape (gate/up, down).

Raw kernels + pre-allocated buffers (matches the existing bench_* methodology so
only kernel body is timed, not custom-op validation/alloc).

Produces the "grouped-kernel grouping efficiency" table in
nvfp4_grouped_gemm_crossover.md. Run from the torchao repo root (needs its
`benchmarks` and `torchao` packages importable):

    cd third_party/torchao && \\
        PYTHONPATH=. python ../../deepseek_v3/kernel_analysis/nvfp4_grouped_kernel_grouping.py
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
from torchao.prototype.moe_training.nvfp4_training.group_rht_quantize_row_col_triton import (
    _group_rht_quantize_row_col_kernel,
)
from torchao.prototype.moe_training.nvfp4_training.group_quantize_2d_triton import (
    BLOCK_M as W_BLOCK_M,
    BLOCK_N as W_BLOCK_N,
    _group_weight_quantize_2d_kernel,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_amax_triton import (
    _hadamard_amax_kernel,
    triton_rht_amax,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_quantize_row_col_triton import (
    _hadamard_quantize_row_col_kernel,
)
from torchao.prototype.moe_training.nvfp4_training.quantize_2d_triton import (
    triton_quantize_2d_weight,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import get_rht_matrix
from torchao.utils import is_sm_at_least_100

dev = torch.device("cuda")
SIGN = (1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1)

E = 8
N_ACT = 7168  # DSV3-671B gate/up activation dim
TOKENS = [512, 1024, 2048, 4096, 8192]
WEIGHT_SHAPES = [("gate/up", 2048, 7168), ("down", 7168, 2048)]

NUM_SMS = torch.cuda.get_device_properties(dev).multi_processor_count


def set_alloc():
    # make_tensor_descriptor kernels need a Triton scratch allocator; custom-op
    # calls (triton_rht_amax) clear it, so (re)set before every timed launch.
    if hasattr(triton, "set_allocator"):
        triton.set_allocator(
            lambda size, align, stream: torch.empty(size, dtype=torch.int8, device=dev)
        )


# ---------------------------------------------------------------- amax (activation)
def grouped_amax(e, m, n):
    total = e * m
    A = torch.randn(total, n, dtype=torch.bfloat16, device=dev)
    B = get_rht_matrix(SIGN, dev, torch.bfloat16, 16)
    offsets = torch.arange(1, e + 1, dtype=torch.int32, device=dev) * m
    lpl = offsets[-1:]
    row_amax = torch.zeros((e,), dtype=torch.float32, device=dev)
    col_amax = torch.zeros((e,), dtype=torch.float32, device=dev)
    grid = (triton.cdiv(total, G_BLOCK_M) * triton.cdiv(n, G_BLOCK_N),)

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
            x, B, rht_out, a_out, m, n,
            GROUP_SIZE_N=8, NUM_SMS=NUM_SMS,
        )

    return benchmark_cuda_function_in_microseconds(run)


# -------------------------------------------------- rht quantize row/col (activation)
def grouped_quant(e, m, n):
    total = e * m
    A = torch.randn(total, n, dtype=torch.bfloat16, device=dev)
    B = get_rht_matrix(SIGN, dev, torch.bfloat16, 16)
    offsets = torch.arange(1, e + 1, dtype=torch.int32, device=dev) * m
    lpl = offsets[-1:]
    Ag = A.view(e, m, n).float().abs()
    amax_row = Ag.amax(dim=(1, 2)).contiguous()
    amax_col = amax_row.clone()
    qa = torch.empty((total, n // 2), dtype=torch.uint8, device=dev)
    qd = torch.empty((n, total // 2), dtype=torch.uint8, device=dev)
    sfa = torch.empty((total // 128, n // 64, 32, 16), dtype=torch.float8_e4m3fn, device=dev)
    sfd = torch.empty((n // 128, total // 64, 32, 16), dtype=torch.float8_e4m3fn, device=dev)
    grid = (triton.cdiv(total, G_BLOCK_M) * triton.cdiv(n, G_BLOCK_N),)

    def run():
        _group_rht_quantize_row_col_kernel[grid](
            A, B, qa, sfa, offsets, amax_row, amax_col, qd, sfd,
            0, 0, 0, 0, total, n,
            num_tensors=e, STOCHASTIC_ROUNDING=False, SHAPE_REP=VARYING_FIRST_DIM,
            BLOCK_M=G_BLOCK_M, BLOCK_N=G_BLOCK_N,
            logical_packed_length_ptr=lpl, num_warps=8, num_stages=3,
        )

    return benchmark_cuda_function_in_microseconds(run)


def single_quant(m, n):
    x = torch.randn(m, n, dtype=torch.bfloat16, device=dev)
    col_amax, row_amax = triton_rht_amax(x, sign_vector=list(SIGN))
    set_alloc()
    B = get_rht_matrix(SIGN, dev, torch.bfloat16, 16)
    col_C = torch.empty((n, m // 2), dtype=torch.uint8, device=dev)
    col_sf = torch.empty((n // 128, m // 64, 32, 16), dtype=torch.float8_e4m3fn, device=dev)
    row_C = torch.empty((m, n // 2), dtype=torch.uint8, device=dev)
    row_sf = torch.empty((m // 128, n // 64, 32, 16), dtype=torch.float8_e4m3fn, device=dev)

    def run():
        _hadamard_quantize_row_col_kernel[(NUM_SMS,)](
            x, B, col_amax, row_amax, col_C, col_sf, row_C, row_sf,
            0, 0, 0, 0, m, n,
            GROUP_SIZE_N=8, NUM_SMS=NUM_SMS, STOCHASTIC_ROUNDING=False,
        )

    return benchmark_cuda_function_in_microseconds(run)


# ------------------------------------------------------------- weight quantize 2d
def grouped_weight(e, m, n):
    W = torch.randn((e, m, n), dtype=torch.bfloat16, device=dev)
    gamax = W.float().abs().amax(dim=(1, 2))
    qa = torch.empty((e, m, n // 2), dtype=torch.uint8, device=dev)
    sfa = torch.empty((e, m // 128, n // 64, 32, 16), dtype=torch.float8_e4m3fn, device=dev)
    qat = torch.empty((e, n, m // 2), dtype=torch.uint8, device=dev)
    sfat = torch.empty((e, n // 128, m // 64, 32, 16), dtype=torch.float8_e4m3fn, device=dev)
    grid = (triton.cdiv(m, W_BLOCK_M), triton.cdiv(n, W_BLOCK_N), e)

    def run():
        _group_weight_quantize_2d_kernel[grid](
            W, gamax, qa, sfa, qat, sfat, m, n,
            BLOCK_M=W_BLOCK_M, BLOCK_N=W_BLOCK_N, num_warps=8, num_stages=3,
        )

    return benchmark_cuda_function_in_microseconds(run)


def single_weight(m, n):
    set_alloc()
    x = torch.randn(m, n, dtype=torch.bfloat16, device=dev)
    gamax = x.float().abs().max()
    a_fp4 = torch.empty((m, n // 2), dtype=torch.uint8, device=dev)
    a_sf = torch.empty((m // 128, n // 64, 32, 16), dtype=torch.float8_e4m3fn, device=dev)
    at_fp4 = torch.empty((n, m // 2), dtype=torch.uint8, device=dev)
    at_sf = torch.empty((n // 128, m // 64, 32, 16), dtype=torch.float8_e4m3fn, device=dev)

    def run():
        triton_quantize_2d_weight[(NUM_SMS,)](
            x, a_fp4, a_sf, at_fp4, at_sf, gamax, m, n,
            GROUP_SIZE_N=8, NUM_SMS=NUM_SMS,
        )

    return benchmark_cuda_function_in_microseconds(run)


def main():
    if not is_sm_at_least_100():
        raise RuntimeError("requires SM100+ (Blackwell)")
    torch.manual_seed(0)

    for name, gfn, sfn in [
        ("rht_amax (act)", grouped_amax, single_amax),
        ("rht_quantize_row_col (act)", grouped_quant, single_quant),
    ]:
        rows = []
        for tpe in TOKENS:
            g = gfn(E, tpe, N_ACT)
            s = sfn(tpe, N_ACT)
            rows.append([tpe, E * tpe, round(g, 2), round(s, 2),
                         round(E * s, 2), round(g / (E * s), 2)])
        print(f"\n### {name}  (E={E}, N={N_ACT})")
        print(tabulate(rows, headers=["tok/exp", "rows", "grouped_us",
                                      "single_us", "E*single_us", "ratio"]))

    rows = []
    for proj, m, n in WEIGHT_SHAPES:
        g = grouped_weight(E, m, n)
        s = single_weight(m, n)
        rows.append([proj, f"{m}x{n}", round(g, 2), round(s, 2),
                     round(E * s, 2), round(g / (E * s), 2)])
    print(f"\n### weight quantize_2d  (E={E}, per-expert weight)")
    print(tabulate(rows, headers=["proj", "shape", "grouped_us",
                                  "single_us", "E*single_us", "ratio"]))


if __name__ == "__main__":
    main()
